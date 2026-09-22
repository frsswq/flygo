"""Prepare, run, and inspect resumable feasibility teacher shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

from measure_teacher_feasibility import engine_command
from pydantic import ValidationError

from flygo.atomic import write_text
from flygo.dataset import parse_sgf_collection, split_of
from flygo.feasibility import (
    FeasibilityProtocol,
    FeasibilitySources,
    file_sha256,
    validate_protocol,
    verify_timing_analysis,
)
from flygo.teacher import katago_queries


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _verify(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_atomic(path: Path, value: object) -> None:
    text = _json_bytes(value).decode()
    write_text(path, lambda stream: stream.write(text))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_protocol(path: Path) -> FeasibilityProtocol:
    protocol = FeasibilityProtocol.model_validate_json(path.read_bytes())
    blockers = validate_protocol(protocol)
    allowed = ("feasibility dataset", "selected circuit")
    disallowed = [blocker for blocker in blockers if not blocker.startswith(allowed)]
    if disallowed:
        raise ValueError("Cannot prepare feasibility corpus: " + "; ".join(disallowed))
    if not protocol.approvals.full_labelling_approved:
        raise ValueError("Full feasibility labelling is not approved")
    return protocol


def _query(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid shard query on line {line_number}: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Invalid shard query on line {line_number}: {path}")
        rows.append(value)
    if len(rows) != 1:
        raise ValueError(f"Each feasibility shard must contain exactly one game: {path}")
    return rows


def _verify_prepared(protocol: FeasibilityProtocol) -> tuple[dict[str, Any], list[Path]]:
    root = protocol.corpus.work_directory
    selection_path = root / "selection.json"
    selection = _load_json(selection_path)
    if selection.get("schema_version") != 1 or selection.get("protocol") != protocol.name:
        raise ValueError(f"Prepared corpus identity mismatch: {selection_path}")
    if selection.get("source_manifest_sha256") != protocol.corpus.source_manifest.sha256:
        raise ValueError(f"Prepared corpus source mismatch: {selection_path}")
    requested_positions = selection.get("requested_positions")
    if (
        not isinstance(requested_positions, int)
        or requested_positions < protocol.corpus.target_labelled_positions
    ):
        raise ValueError("Prepared corpus does not meet the labelled-position target")
    games = selection.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("Prepared corpus has no games")
    if set(selection.get("game_splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Prepared corpus must contain train, validation, and test games")

    shards: list[Path] = []
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise ValueError("Prepared corpus has an invalid game entry")
        sgf_path = root / str(game["sgf_file"])
        _verify(sgf_path, str(game["sha256"]))
        shard = root / "shards" / f"{index:04d}"
        input_path = shard / "input.json"
        query_path = shard / "queries.jsonl"
        input_manifest = _load_json(input_path)
        if (
            input_manifest.get("schema_version") != 1
            or input_manifest.get("protocol") != protocol.name
            or input_manifest.get("index") != index
            or input_manifest.get("game_id") != game["game_id"]
            or input_manifest.get("sgf_sha256") != game["sha256"]
            or input_manifest.get("query_sha256") != file_sha256(query_path)
        ):
            raise ValueError(f"Prepared shard identity mismatch: {shard}")
        rows = _query(query_path)
        positions = sum(len(row["analyzeTurns"]) for row in rows)
        if positions != input_manifest.get("positions") or positions != game["positions"]:
            raise ValueError(f"Prepared shard position mismatch: {shard}")
        shards.append(shard)
    return selection, shards


def prepare(protocol: FeasibilityProtocol) -> dict[str, Any]:
    root = protocol.corpus.work_directory
    if root.exists():
        selection, shards = _verify_prepared(protocol)
        return {
            "status": "reused",
            "games": len(shards),
            "positions": selection["requested_positions"],
            "work_directory": str(root),
        }

    sources = FeasibilitySources.model_validate_json(
        protocol.corpus.source_manifest.path.read_bytes()
    )
    source_root = protocol.corpus.teacher_network.path.parent
    archive_assets = list(sources.archives)
    if not archive_assets:
        raise ValueError("Source manifest has no game archives")

    selected: list[tuple[dict[str, Any], bytes, Any]] = []
    rejections: Counter[str] = Counter()
    positions = 0
    splits: Counter[str] = Counter()
    seen_games: set[str] = set()
    done = False
    for asset in archive_assets:
        archive_path = source_root / asset.file
        _verify(archive_path, asset.sha256)
        with tarfile.open(archive_path) as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile() or not member.name.lower().endswith(".sgf"):
                    continue
                if member.size > 1_000_000:
                    rejections["SGF larger than 1 MB"] += 1
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    rejections["unreadable SGF member"] += 1
                    continue
                with stream:
                    payload = stream.read()
                try:
                    games = parse_sgf_collection(payload)
                    if len(games) != 1:
                        raise ValueError("SGF member contains multiple games")
                    (game,) = games
                    if game.size != protocol.corpus.board_size:
                        raise ValueError("wrong board size")
                    if game.game_id in seen_games:
                        raise ValueError("duplicate game")
                except ValueError as error:
                    rejections[str(error).split(":", 1)[0]] += 1
                    continue
                query = next(
                    iter(
                        katago_queries(
                            [game],
                            visits=protocol.corpus.visits,
                            stride=protocol.corpus.stride,
                        )
                    )
                )
                game_positions = len(query["analyzeTurns"])
                split = split_of(game.game_id)
                digest = hashlib.sha256(payload).hexdigest()
                record = {
                    "archive": Path(asset.file).name,
                    "archive_sha256": asset.sha256,
                    "member": member.name,
                    "sha256": digest,
                    "game_id": game.game_id,
                    "split": split,
                    "moves": len(game.moves),
                    "positions": game_positions,
                    "sgf_file": f"sgf/{digest}.sgf",
                }
                selected.append((record, payload, query))
                seen_games.add(game.game_id)
                positions += game_positions
                splits[split] += 1
                if positions >= protocol.corpus.target_labelled_positions and set(splits) == {
                    "train",
                    "validation",
                    "test",
                }:
                    done = True
                    break
        if done:
            break
    if not done:
        raise ValueError(
            f"Pinned archives supplied only {positions} eligible positions or missed a split"
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=root.parent, prefix=f".{root.name}."))
    try:
        game_records = []
        for index, (record, payload, query) in enumerate(selected):
            sgf_path = stage / record["sgf_file"]
            sgf_path.parent.mkdir(parents=True, exist_ok=True)
            sgf_path.write_bytes(payload)
            shard = stage / "shards" / f"{index:04d}"
            shard.mkdir(parents=True)
            query_path = shard / "queries.jsonl"
            query_path.write_text(json.dumps(query) + "\n")
            _write(
                shard / "input.json",
                {
                    "schema_version": 1,
                    "protocol": protocol.name,
                    "index": index,
                    "game_id": record["game_id"],
                    "positions": record["positions"],
                    "sgf_sha256": record["sha256"],
                    "query_sha256": file_sha256(query_path),
                },
            )
            game_records.append(record)
        _write(
            stage / "selection.json",
            {
                "schema_version": 1,
                "protocol": protocol.name,
                "source_revision": protocol.source_revision,
                "source_manifest_sha256": protocol.corpus.source_manifest.sha256,
                "selection": (
                    "Pinned archives in manifest order, SGF members in filename order, "
                    "first valid complete games until the target and every split are present"
                ),
                "requested_positions": positions,
                "target_positions": protocol.corpus.target_labelled_positions,
                "games_per_shard": protocol.corpus.games_per_shard,
                "game_splits": dict(splits),
                "rejections": dict(rejections),
                "games": game_records,
            },
        )
        stage.replace(root)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "status": "prepared",
        "games": len(selected),
        "positions": positions,
        "work_directory": str(root),
    }


def _verify_complete(
    protocol: FeasibilityProtocol,
    shard: Path,
    input_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    complete_path = shard / "complete.json"
    if not complete_path.exists():
        return None
    complete = _load_json(complete_path)
    analysis_path = shard / "analysis.jsonl"
    stderr_path = shard / "analysis.log"
    query_path = shard / "queries.jsonl"
    if (
        complete.get("schema_version") != 1
        or complete.get("protocol") != protocol.name
        or complete.get("input_sha256") != file_sha256(shard / "input.json")
        or complete.get("analysis_sha256") != file_sha256(analysis_path)
        or complete.get("stderr_sha256") != file_sha256(stderr_path)
    ):
        raise ValueError(f"Completed shard is damaged or has the wrong identity: {shard}")
    completed = verify_timing_analysis(
        analysis_path,
        _query(query_path),
        visits=protocol.corpus.visits,
    )
    if completed != input_manifest["positions"] or completed != complete.get("positions"):
        raise ValueError(f"Completed shard has the wrong response count: {shard}")
    return complete


def status(protocol: FeasibilityProtocol) -> dict[str, Any]:
    selection, shards = _verify_prepared(protocol)
    completed_shards = 0
    completed_positions = 0
    for shard in shards:
        input_manifest = _load_json(shard / "input.json")
        complete = _verify_complete(protocol, shard, input_manifest)
        if complete is not None:
            completed_shards += 1
            positions = complete.get("positions")
            if not isinstance(positions, int):
                raise ValueError(f"Completed shard has an invalid position count: {shard}")
            completed_positions += positions
    total_positions = selection.get("requested_positions")
    if not isinstance(total_positions, int):
        raise ValueError("Prepared corpus has an invalid position count")
    return {
        "protocol": protocol.name,
        "work_directory": str(protocol.corpus.work_directory),
        "completed_shards": completed_shards,
        "total_shards": len(shards),
        "remaining_shards": len(shards) - completed_shards,
        "completed_positions": completed_positions,
        "total_positions": total_positions,
        "remaining_positions": total_positions - completed_positions,
    }


def _run_process(
    command: list[str],
    query_path: Path,
    analysis_stream: BinaryIO,
    stderr_stream: BinaryIO,
    timeout: float,
) -> int:
    with query_path.open("rb") as query_stream:
        process = subprocess.Popen(
            command,
            stdin=query_stream,
            stdout=analysis_stream,
            stderr=stderr_stream,
        )
        try:
            return process.wait(timeout=timeout)
        except (KeyboardInterrupt, OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
            raise


def run(protocol: FeasibilityProtocol, *, max_shards: int, timeout: float) -> dict[str, Any]:
    if max_shards < 1:
        raise ValueError("--max-shards must be positive")
    _, shards = _verify_prepared(protocol)
    completed_now = 0
    command = engine_command(protocol)
    for shard in shards:
        input_path = shard / "input.json"
        input_manifest = _load_json(input_path)
        if _verify_complete(protocol, shard, input_manifest) is not None:
            continue
        temporary_paths: list[Path] = []
        try:
            handles = [
                tempfile.mkstemp(dir=shard, prefix=".analysis."),
                tempfile.mkstemp(dir=shard, prefix=".analysis-log."),
            ]
            temporary_paths = [Path(name) for _, name in handles]
            started = time.monotonic()
            with (
                os.fdopen(handles[0][0], "wb") as analysis_stream,
                os.fdopen(handles[1][0], "wb") as stderr_stream,
            ):
                return_code = _run_process(
                    command,
                    shard / "queries.jsonl",
                    analysis_stream,
                    stderr_stream,
                    timeout,
                )
            elapsed = time.monotonic() - started
            if return_code != 0:
                raise ValueError(f"Teacher exited with status {return_code} for {shard.name}")
            positions = verify_timing_analysis(
                temporary_paths[0],
                _query(shard / "queries.jsonl"),
                visits=protocol.corpus.visits,
            )
            if positions != input_manifest["positions"]:
                raise ValueError(f"Teacher response count differs for {shard.name}")
            analysis_path = shard / "analysis.jsonl"
            stderr_path = shard / "analysis.log"
            temporary_paths[0].replace(analysis_path)
            temporary_paths[1].replace(stderr_path)
            complete = {
                "schema_version": 1,
                "protocol": protocol.name,
                "index": input_manifest["index"],
                "positions": positions,
                "elapsed_seconds": elapsed,
                "input_sha256": file_sha256(input_path),
                "analysis_sha256": file_sha256(analysis_path),
                "stderr_sha256": file_sha256(stderr_path),
                "teacher_executable_sha256": protocol.corpus.teacher_executable.sha256,
                "teacher_network_sha256": protocol.corpus.teacher_network.sha256,
                "teacher_configuration_sha256": protocol.corpus.teacher_configuration.sha256,
            }
            _write_atomic(shard / "complete.json", complete)
            completed_now += 1
            if completed_now >= max_shards:
                break
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
    result = status(protocol)
    result["completed_now"] = completed_now
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run", "status"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/protocols/feasibility-19-v1.json"),
    )
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=7200)
    arguments = parser.parse_args()
    try:
        protocol = _load_protocol(arguments.protocol)
        root = protocol.corpus.work_directory
        root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = root.parent / f".{root.name}.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if arguments.command == "prepare":
                result = prepare(protocol)
            elif arguments.command == "run":
                result = run(
                    protocol,
                    max_shards=arguments.max_shards,
                    timeout=arguments.timeout,
                )
            else:
                result = status(protocol)
    except (
        KeyError,
        OSError,
        ValueError,
        ValidationError,
        tarfile.TarError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
