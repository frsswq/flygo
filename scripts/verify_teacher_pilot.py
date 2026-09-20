"""Rebuild the pilot from raw answers and check its actual dataset files."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_teacher_pilot import Sources, verify

from flygo.dataset import (
    SPLITS,
    build_dataset,
    load_sgf_games,
    load_teacher_targets,
    sample_id,
    split_of,
)
from flygo.teacher import import_katago_analysis, katago_queries
from flygo.training import load_training_data


def verify_dataset(sources_path: Path, root: Path, dataset: Path) -> dict[str, object]:
    sources = Sources.model_validate_json(sources_path.read_bytes())
    verify(root / "analysis.cfg", sources.configuration_sha256)
    for asset in sources.downloads:
        verify(root / asset.file, asset.sha256)
    paths = []
    for source in sources.games:
        path = root / "sgf" / Path(source.member).name
        verify(path, source.sha256)
        paths.append(path)
    if set((root / "sgf").glob("*.sgf")) != set(paths):
        raise ValueError("Unexpected SGF files")
    games = load_sgf_games(paths)
    queries = list(katago_queries(games, visits=sources.visits, stride=sources.stride))
    if (root / "queries.jsonl").read_text() != "".join(json.dumps(row) + "\n" for row in queries):
        raise ValueError("Query file differs from the pinned sample")
    expected = {sample_id(query["id"], turn) for query in queries for turn in query["analyzeTurns"]}
    analysis = root / "analysis.jsonl"
    responses = [json.loads(line) for line in analysis.read_text().splitlines() if line.strip()]
    if any(
        row.get("isDuringSearch") is not False or row["rootInfo"]["visits"] != sources.visits
        for row in responses
    ):
        raise ValueError("Analysis contains unfinished answers or a different visit budget")
    players = Counter("black" if row["turnNumber"] % 2 == 0 else "white" for row in responses)
    if set(players) != {"black", "white"}:
        raise ValueError("Pilot answers must include both players")
    targets = root / "targets.jsonl"
    if set(load_teacher_targets(targets)) != expected:
        raise ValueError("Teacher targets do not exactly cover the requested positions")
    with tempfile.TemporaryDirectory(prefix="flygo-pilot-verify-") as temporary:
        rebuilt = Path(temporary)
        count = import_katago_analysis(analysis, rebuilt / "targets.jsonl", games)
        if (rebuilt / "targets.jsonl").read_bytes() != targets.read_bytes():
            raise ValueError("Teacher targets differ from imported raw answers")
        summary = build_dataset(
            games,
            rebuilt / "dataset",
            stride=sources.stride,
            teacher_path=targets,
            require_teacher=True,
        )
        manifest_path = dataset / "manifest.json"
        expected_manifest = (rebuilt / "dataset" / "manifest.json").read_bytes()
        if manifest_path.read_bytes() != expected_manifest:
            raise ValueError("Dataset manifest differs from a strict rebuild")
        manifest = json.loads(expected_manifest)
        positions: set[bytes] = set()
        split_examples = {}
        split_players = {}
        for split in SPLITS:
            path = dataset / f"{split}.npz"
            verify(path, manifest["files"][split]["sha256"])
            data = load_training_data(path, size=19)
            if len(data.features) == 0:
                raise ValueError(f"Empty {split} split")
            split_examples[split] = len(data.features)
            if set(data.features[:, -1].tolist()) != {-1.0, 1.0}:
                raise ValueError(f"Both players must occur in {split}")
            split_players[split] = {
                "black": int(np.count_nonzero(data.features[:, -1] == 1)),
                "white": int(np.count_nonzero(data.features[:, -1] == -1)),
            }
            with np.load(path) as arrays:
                if not np.all(arrays["source"] == 1):
                    raise ValueError(f"Non-teacher targets in {split}")
            for features in data.features:
                key = features.tobytes()
                if key in positions:
                    raise ValueError("Duplicate board features across dataset rows")
                positions.add(key)
    tracked = {
        "sources": sources_path,
        "configuration": root / "analysis.cfg",
        "queries": root / "queries.jsonl",
        "raw_analysis": analysis,
        "teacher_targets": targets,
        "dataset_manifest": dataset / "manifest.json",
    }
    if any(asset.file == "katago.zip" for asset in sources.downloads):
        with zipfile.ZipFile(root / "katago.zip") as archive:
            engine_sha256 = hashlib.sha256(archive.read("katago.exe")).hexdigest()
        verify(root / "engine" / "katago.exe", engine_sha256)
        tracked["engine_executable"] = root / "engine" / "katago.exe"
    return {
        "games": len(games),
        "game_splits": dict(Counter(split_of(game.game_id) for game in games)),
        "requested_positions": len(expected),
        "completed_targets": count,
        "positions_by_player": dict(players),
        "visits_per_position": sources.visits,
        "stride": sources.stride,
        "examples": summary.examples,
        "duplicates_removed": summary.duplicate_examples,
        "split_examples": split_examples,
        "split_players": split_players,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "all_targets_from_teacher": True,
        "matches_strict_rebuild": True,
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in tracked.items()
        },
        "files": manifest["files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("docs/teacher-pilot/sources.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/teacher-pilot"))
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/teacher-pilot-19"))
    arguments = parser.parse_args()
    try:
        report = verify_dataset(arguments.sources, arguments.root, arguments.dataset)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
