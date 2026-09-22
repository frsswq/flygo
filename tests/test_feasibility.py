import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from flygo.feasibility import (
    FeasibilityProtocol,
    FeasibilitySources,
    dry_run,
    timing_queries,
    validate_protocol,
    verify_timing_analysis,
)
from flygo.research import MODELS


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def protocol_payload(tmp_path: Path) -> dict[str, object]:
    immutable = tmp_path / "input"
    immutable.write_bytes(b"fixed")
    pinned = artifact(immutable)
    return {
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
            "source_manifest": pinned,
            "data_permission": pinned,
            "teacher_network": pinned,
            "teacher_executable": pinned,
            "teacher_configuration": pinned,
            "pilot_queries": pinned,
            "output": {"path": str(tmp_path / "dataset.json"), "sha256": None},
            "work_directory": str(tmp_path / "corpus-work"),
            "games_per_shard": 1,
            "timed_batch_positions": 100,
            "timed_batch_report": {"path": str(tmp_path / "timing.json"), "sha256": None},
        },
        "circuit": {
            "nodes": 500,
            "source_graph": pinned,
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
        "approvals": {
            "timed_batch_approved": True,
            "full_labelling_approved": True,
        },
    }


def test_feasibility_dry_run_lists_every_cell_without_opening_test_data(tmp_path: Path) -> None:
    protocol = FeasibilityProtocol.model_validate(protocol_payload(tmp_path))

    result = dry_run(protocol)

    assert result["status"] == "blocked"
    assert result["training"]["cell_count"] == 18
    assert result["training"]["final_test"] is False
    assert result["teacher"]["target_labelled_positions"] == 10_000
    assert any("dataset has not been completed" in item for item in result["blockers"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stride", 2, "stride must be odd"),
        ("seeds", [7, 7, 27], "seeds must be exactly"),
        ("models", list(reversed(MODELS)), "models must be exactly"),
    ],
)
def test_feasibility_rejects_changed_fixed_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = protocol_payload(tmp_path)
    section = payload["corpus"] if field == "stride" else payload["training"]
    section[field] = value  # type: ignore[index]
    protocol = FeasibilityProtocol.model_validate(payload)

    with pytest.raises(ValueError, match=message):
        validate_protocol(protocol)


def test_committed_feasibility_protocol_has_a_non_executing_dry_run() -> None:
    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_feasibility.py",
            "--protocol",
            "docs/protocols/feasibility-19-v1.json",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "blocked"
    assert result["training"]["cell_count"] == 18
    assert result["training"]["final_test"] is False


def test_timing_queries_and_analysis_require_exact_final_coverage(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "game", "analyzeTurns": [0, 1, 2], "maxVisits": 256}) + "\n"
    )
    queries = timing_queries(source, 2)
    assert queries[0]["analyzeTurns"] == [0, 1]

    analysis = tmp_path / "analysis.jsonl"
    responses = [
        {
            "id": "game",
            "turnNumber": turn,
            "isDuringSearch": False,
            "rootInfo": {"visits": 256},
            "moveInfos": [{"move": "pass", "visits": 256}],
        }
        for turn in (0, 1)
    ]
    analysis.write_text("".join(json.dumps(response) + "\n" for response in responses))
    assert verify_timing_analysis(analysis, queries, visits=256) == 2

    responses[1]["isDuringSearch"] = True
    analysis.write_text("".join(json.dumps(response) + "\n" for response in responses))
    with pytest.raises(ValueError, match="unfinished"):
        verify_timing_analysis(analysis, queries, visits=256)


def test_teacher_timing_command_publishes_and_reuses_verified_report(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "game",
                "moves": [],
                "rules": "tromp-taylor",
                "komi": 7.5,
                "boardXSize": 19,
                "boardYSize": 19,
                "analyzeTurns": list(range(100)),
                "maxVisits": 256,
                "includePolicy": True,
            }
        )
        + "\n"
    )
    count = tmp_path / "engine-count"
    engine = tmp_path / "fake-engine"
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
    payload = protocol_payload(tmp_path)
    corpus = payload["corpus"]
    corpus["pilot_queries"] = artifact(source)  # type: ignore[index]
    corpus["teacher_executable"] = artifact(engine)  # type: ignore[index]
    report_path = tmp_path / "timing" / "timing.json"
    corpus["timed_batch_report"] = {  # type: ignore[index]
        "path": str(report_path),
        "sha256": None,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload))
    command = [
        sys.executable,
        "scripts/measure_teacher_feasibility.py",
        "--protocol",
        str(protocol_path),
        "--timeout",
        "10",
    ]

    first = subprocess.run(command, capture_output=True, text=True)
    second = subprocess.run(command, capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert count.read_text() == "1"
    report = json.loads(report_path.read_text())
    assert report["positions"] == 100
    assert report["visits_per_position"] == 256
    assert report["projection"]["positions"] == 10_000


def test_failed_teacher_timing_does_not_publish_report(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "game", "analyzeTurns": list(range(100)), "maxVisits": 256}) + "\n"
    )
    engine = tmp_path / "failed-engine"
    engine.write_text("#!/bin/sh\nexit 2\n")
    engine.chmod(0o755)
    payload = protocol_payload(tmp_path)
    corpus = payload["corpus"]
    corpus["pilot_queries"] = artifact(source)  # type: ignore[index]
    corpus["teacher_executable"] = artifact(engine)  # type: ignore[index]
    report_path = tmp_path / "timing" / "timing.json"
    corpus["timed_batch_report"] = {  # type: ignore[index]
        "path": str(report_path),
        "sha256": None,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload))

    run = subprocess.run(
        [
            sys.executable,
            "scripts/measure_teacher_feasibility.py",
            "--protocol",
            str(protocol_path),
            "--timeout",
            "10",
        ],
        capture_output=True,
        text=True,
    )

    assert run.returncode != 0
    assert "status 2" in run.stderr
    assert not report_path.exists()


def test_feasibility_sources_reject_path_escape() -> None:
    manifest = FeasibilitySources.model_validate(
        {
            "schema_version": 1,
            "selection": "fixed source order",
            "pages": [
                {
                    "url": "https://example.com/games/",
                    "sha256": "0" * 64,
                    "sampled_game_ids": ["1"],
                }
            ],
            "archives": [
                {
                    "url": "https://example.com/game.tgz",
                    "file": "archives/game.tgz",
                    "sha256": "1" * 64,
                }
            ],
        }
    )
    assert manifest.selection == "fixed source order"
    assert manifest.pages[0].sampled_game_ids == ("1",)
    assert manifest.archives[0].file == "archives/game.tgz"

    payload = manifest.model_dump()
    payload["archives"][0]["file"] = "../game.tgz"
    with pytest.raises(ValueError, match="archive path"):
        FeasibilitySources.model_validate(payload)
