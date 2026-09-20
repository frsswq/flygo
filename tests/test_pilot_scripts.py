import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from flygo.dataset import SPLITS, build_dataset, load_sgf_games, parse_sgf_collection, split_of
from flygo.teacher import import_katago_analysis


def game_for_split(split: str, column: str) -> bytes:
    for nonce in range(1000):
        payload = (
            f"(;SZ[19]KM[7.5]RE[B+R]C[{nonce}];B[{column}a];W[{column}b];B[{column}c])"
        ).encode()
        if split_of(parse_sgf_collection(payload)[0].game_id) == split:
            return payload
    raise AssertionError(f"Cannot construct {split} fixture")


@pytest.fixture
def pilot_inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "raw"
    archive_path = root / "archives" / "games.tgz"
    archive_path.parent.mkdir(parents=True)
    games = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for index, split in enumerate(SPLITS):
            payload = game_for_split(split, chr(ord("a") + index))
            member = tarfile.TarInfo(f"{split}.sgf")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
            (game,) = parse_sgf_collection(payload)
            games.append(
                {
                    "archive": "games.tgz",
                    "member": member.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "game_id": game.game_id,
                    "split": split,
                    "moves": len(game.moves),
                }
            )
    sources = tmp_path / "sources.json"
    configuration = b"reportAnalysisWinratesAs = BLACK\n"
    (tmp_path / "analysis.cfg").write_bytes(configuration)
    sources.write_text(
        json.dumps(
            {
                "version": 1,
                "selection": "Local test fixture; not research evidence.",
                "configuration_sha256": hashlib.sha256(configuration).hexdigest(),
                "visits": 256,
                "stride": 1,
                "downloads": [
                    {
                        "url": "https://example.invalid/games.tgz",
                        "file": "archives/games.tgz",
                        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    }
                ],
                "games": games,
            }
        )
    )
    return root, sources


def run_script(
    script: str,
    root: Path,
    sources: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            f"scripts/{script}.py",
            "--root",
            str(root),
            "--sources",
            str(sources),
            *arguments,
        ],
        capture_output=True,
        text=True,
    )


def test_pilot_preparation_reuses_verified_files_and_rejects_corruption(
    pilot_inputs: tuple[Path, Path],
) -> None:
    root, sources = pilot_inputs
    first = run_script("prepare_teacher_pilot", root, sources)
    assert first.returncode == 0, first.stderr
    queries = root / "queries.jsonl"
    before = queries.stat().st_mtime_ns
    second = run_script("prepare_teacher_pilot", root, sources)
    assert second.returncode == 0, second.stderr
    assert queries.stat().st_mtime_ns == before
    archive = root / "archives" / "games.tgz"
    archive.write_bytes(b"damaged")
    damaged = run_script("prepare_teacher_pilot", root, sources)
    assert damaged.returncode != 0
    assert "SHA256 mismatch" in damaged.stderr
    assert archive.read_bytes() == b"damaged"
    assert queries.stat().st_mtime_ns == before


def test_pilot_rejects_download_path_escape(pilot_inputs: tuple[Path, Path]) -> None:
    root, sources = pilot_inputs
    manifest = json.loads(sources.read_text())
    manifest["downloads"][0]["file"] = "../escape.tgz"
    sources.write_text(json.dumps(manifest))
    rejected = run_script("prepare_teacher_pilot", root, sources)
    assert rejected.returncode != 0
    assert "relative file path" in rejected.stderr
    assert not (root.parent / "escape.tgz").exists()


def test_pilot_verification_rejects_unfinished_teacher_answers(
    pilot_inputs: tuple[Path, Path],
) -> None:
    root, sources = pilot_inputs
    prepared = run_script("prepare_teacher_pilot", root, sources)
    assert prepared.returncode == 0, prepared.stderr
    query = json.loads((root / "queries.jsonl").read_text().splitlines()[0])
    (root / "analysis.jsonl").write_text(
        json.dumps(
            {
                "id": query["id"],
                "turnNumber": 0,
                "isDuringSearch": True,
                "rootInfo": {"visits": 256, "winrate": 0.5},
            }
        )
        + "\n"
    )
    rejected = run_script("verify_teacher_pilot", root, sources)
    assert rejected.returncode != 0
    assert "unfinished" in rejected.stderr
    assert not (root / "targets.jsonl").exists()


def test_pilot_rejects_a_sampling_interval_that_excludes_white(
    pilot_inputs: tuple[Path, Path],
) -> None:
    root, sources = pilot_inputs
    manifest = json.loads(sources.read_text())
    manifest["stride"] = 32
    sources.write_text(json.dumps(manifest))
    rejected = run_script("prepare_teacher_pilot", root, sources)
    assert rejected.returncode != 0
    assert "stride must be odd" in rejected.stderr
    assert not (root / "queries.jsonl").exists()


def test_pilot_rebuild_verifies_arrays_not_only_the_manifest(
    pilot_inputs: tuple[Path, Path],
) -> None:
    root, sources = pilot_inputs
    prepared = run_script("prepare_teacher_pilot", root, sources)
    assert prepared.returncode == 0, prepared.stderr
    queries = [json.loads(line) for line in (root / "queries.jsonl").read_text().splitlines()]
    responses = [
        {
            "id": query["id"],
            "turnNumber": turn,
            "isDuringSearch": False,
            "rootInfo": {"visits": 256, "winrate": 0.75},
            "moveInfos": [{"move": query["moves"][turn][1], "visits": 256}],
        }
        for query in queries
        for turn in query["analyzeTurns"]
    ]
    analysis = root / "analysis.jsonl"
    analysis.write_text("".join(json.dumps(row) + "\n" for row in reversed(responses)))
    (root / "analysis.cfg").write_text("reportAnalysisWinratesAs = BLACK\n")
    games = load_sgf_games((root / "sgf").glob("*.sgf"))
    targets = root / "targets.jsonl"
    import_katago_analysis(analysis, targets, games)
    dataset = root / "dataset"
    build_dataset(games, dataset, teacher_path=targets, require_teacher=True)
    verified = run_script("verify_teacher_pilot", root, sources, "--dataset", str(dataset))
    assert verified.returncode == 0, verified.stderr
    report = json.loads(verified.stdout)
    assert report["completed_targets"] == 9
    assert report["examples"] == 7
    assert report["matches_strict_rebuild"] is True
    assert report["all_targets_from_teacher"] is True
    assert set(report["split_examples"]) == set(SPLITS)
    (dataset / "train.npz").write_bytes(b"damaged")
    damaged = run_script("verify_teacher_pilot", root, sources, "--dataset", str(dataset))
    assert damaged.returncode != 0
    assert "SHA256 mismatch" in damaged.stderr
