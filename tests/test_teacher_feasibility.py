import fcntl
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from flygo.go import Position
from flygo.research import MODELS


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def long_game(index: int, moves: list[int]) -> bytes:
    nodes = []
    for turn, action in enumerate(moves):
        row, column = divmod(action, 19)
        colour = "B" if turn % 2 == 0 else "W"
        nodes.append(f";{colour}[{chr(97 + column)}{chr(97 + row)}]")
    return f"(;SZ[19]KM[7.5]RE[B+R]C[{index}]{''.join(nodes)})".encode()


def fixture_protocol(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    archive_path = source_root / "archives" / "games.tgz"
    archive_path.parent.mkdir(parents=True)
    position = Position.empty(19)
    moves = []
    for _ in range(200):
        action = next(
            action for action in position.legal_actions() if action != position.pass_action
        )
        moves.append(action)
        position = position.play(action)

    payloads = [long_game(index, moves) for index in range(50)]
    with tarfile.open(archive_path, "w:gz") as archive:
        for index, payload in enumerate(payloads):
            member = tarfile.TarInfo(f"{index:04d}.sgf")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    configuration = source_root / "analysis.cfg"
    network = source_root / "network.bin.gz"
    permission = source_root / "permission.html"
    graph = source_root / "graph.parquet"
    timing = source_root / "timing.json"
    pilot_queries = source_root / "pilot-queries.jsonl"
    for path, content in {
        configuration: b"config",
        network: b"network",
        permission: b"permission",
        graph: b"graph",
        timing: b"{}\n",
        pilot_queries: b'{"id":"pilot","analyzeTurns":[0],"maxVisits":256}\n',
    }.items():
        path.write_bytes(content)

    count = tmp_path / "engine-count"
    engine = source_root / "fake-engine"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"count = pathlib.Path({str(count)!r})\n"
        "count.write_text(str(int(count.read_text()) + 1) if count.exists() else '1')\n"
        "for line in sys.stdin:\n"
        "    query = json.loads(line)\n"
        "    for turn in query['analyzeTurns']:\n"
        "        print(json.dumps({'id': query['id'], 'turnNumber': turn, "
        "'isDuringSearch': False, 'rootInfo': {'visits': query['maxVisits']}, "
        "'moveInfos': [{'move': 'pass', 'visits': query['maxVisits']}]}))\n"
    )
    engine.chmod(0o755)

    source_manifest = tmp_path / "sources.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": "test",
                "pages": [],
                "archives": [
                    {
                        "url": "https://example.com/games.tgz",
                        "file": "archives/games.tgz",
                        "sha256": artifact(archive_path)["sha256"],
                    }
                ],
            }
        )
    )
    work = tmp_path / "corpus"
    protocol = {
        "schema_version": 1,
        "name": "feasibility-19-v1",
        "source_revision": "34264a9",
        "claim_scope": "feasibility-only",
        "corpus": {
            "board_size": 19,
            "komi": 7.5,
            "target_labelled_positions": 10_000,
            "visits": 256,
            "stride": 1,
            "source_manifest": artifact(source_manifest),
            "data_permission": artifact(permission),
            "teacher_network": artifact(network),
            "teacher_executable": artifact(engine),
            "teacher_configuration": artifact(configuration),
            "pilot_queries": artifact(pilot_queries),
            "output": {
                "path": str(tmp_path / "dataset" / "manifest.json"),
                "sha256": None,
            },
            "work_directory": str(work),
            "games_per_shard": 1,
            "timed_batch_positions": 100,
            "timed_batch_report": artifact(timing),
        },
        "circuit": {
            "nodes": 500,
            "source_graph": artifact(graph),
            "output": {"path": str(tmp_path / "circuit.parquet"), "sha256": None},
        },
        "training": {
            "models": list(MODELS),
            "seeds": [7, 17, 27],
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.001,
            "steps": 8,
            "retention": 0.35,
            "recurrent_gain": 0.9,
            "normalization": "incoming",
            "weights": "weighted",
            "value_weight": 1.0,
            "final_test": False,
            "output": str(tmp_path / "runs"),
            "summary": str(tmp_path / "runs" / "summary.json"),
        },
        "approvals": {"timed_batch_approved": True, "full_labelling_approved": True},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return protocol_path, work, count


def command(protocol: Path, action: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/teacher_feasibility.py",
            action,
            "--protocol",
            str(protocol),
            *arguments,
        ],
        capture_output=True,
        text=True,
    )


def test_teacher_corpus_prepares_and_resumes_one_verified_shard(tmp_path: Path) -> None:
    protocol, work, count = fixture_protocol(tmp_path)

    prepared = command(protocol, "prepare")
    reused = command(protocol, "prepare")
    blocked_finalize = command(protocol, "finalize")
    first = command(protocol, "run", "--max-shards", "1", "--timeout", "10")
    second = command(protocol, "run", "--max-shards", "1", "--timeout", "10")

    assert prepared.returncode == 0, prepared.stderr
    assert reused.returncode == 0, reused.stderr
    assert blocked_finalize.returncode != 0
    assert "teacher shards remain incomplete" in blocked_finalize.stderr
    assert json.loads(prepared.stdout)["positions"] == 10_000
    assert json.loads(reused.stdout)["status"] == "reused"
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert count.read_text() == "2"
    first_status = json.loads(first.stdout)
    second_status = json.loads(second.stdout)
    assert first_status["completed_shards"] == 1
    assert second_status["completed_shards"] == 2
    assert (work / "shards" / "0000" / "complete.json").is_file()
    assert (work / "shards" / "0001" / "complete.json").is_file()

    analysis = work / "shards" / "0000" / "analysis.jsonl"
    original = analysis.read_bytes()
    analysis.write_bytes(original + b"\n")
    damaged = command(protocol, "status")
    assert damaged.returncode != 0
    assert "damaged" in damaged.stderr
    analysis.write_bytes(original)

    lock_path = work.parent / f".{work.name}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        concurrent = command(protocol, "status")
    assert concurrent.returncode != 0
    assert "Resource temporarily unavailable" in concurrent.stderr


def test_failed_teacher_shard_never_publishes_completion(tmp_path: Path) -> None:
    protocol, work, _ = fixture_protocol(tmp_path)
    prepared = command(protocol, "prepare")
    assert prepared.returncode == 0, prepared.stderr
    payload = json.loads(protocol.read_text())
    failed_engine = tmp_path / "source" / "failed-engine"
    failed_engine.write_text("#!/bin/sh\nexit 2\n")
    failed_engine.chmod(0o755)
    payload["corpus"]["teacher_executable"] = artifact(failed_engine)
    protocol.write_text(json.dumps(payload))

    failed = command(protocol, "run", "--max-shards", "1", "--timeout", "10")

    assert failed.returncode != 0
    assert "status 2" in failed.stderr
    assert not (work / "shards" / "0000" / "complete.json").exists()

    interrupted_engine = tmp_path / "source" / "interrupted-engine"
    interrupted_engine.write_text("#!/bin/sh\nsleep 10\n")
    interrupted_engine.chmod(0o755)
    payload["corpus"]["teacher_executable"] = artifact(interrupted_engine)
    protocol.write_text(json.dumps(payload))
    interrupted = command(protocol, "run", "--max-shards", "1", "--timeout", "0.1")
    assert interrupted.returncode != 0
    assert "timed out" in interrupted.stderr
    assert not (work / "shards" / "0000" / "complete.json").exists()
