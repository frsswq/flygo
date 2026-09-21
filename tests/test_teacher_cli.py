import json
import subprocess
from pathlib import Path

import numpy as np


def test_cli_sampled_teacher_pipeline_requires_complete_coverage(tmp_path: Path) -> None:
    sgf = tmp_path / "game.sgf"
    sgf.write_bytes(b"(;SZ[5]KM[0]RE[B+R];B[aa];W[bb];B[cc])")
    queries = tmp_path / "queries.jsonl"
    command = ["uv", "run", "--no-sync", "flygo"]
    subprocess.run(
        [*command, "teacher-queries", "--sgf", str(sgf), "--output", str(queries), "--stride", "2"],
        check=True,
        capture_output=True,
    )
    query = json.loads(queries.read_text())
    assert query["komi"] == 0
    assert query["analyzeTurns"] == [0, 2]
    responses = [
        {
            "id": query["id"],
            "turnNumber": turn,
            "rootInfo": {"winrate": 0.75},
            "moveInfos": [{"move": query["moves"][turn][1], "visits": 256}],
        }
        for turn in reversed(query["analyzeTurns"])
    ]
    analysis = tmp_path / "analysis.jsonl"
    analysis.write_text("".join(json.dumps(row) + "\n" for row in responses))
    targets = tmp_path / "targets.jsonl"
    subprocess.run(
        [
            *command,
            "teacher-import",
            "--sgf",
            str(sgf),
            "--analysis",
            str(analysis),
            "--output",
            str(targets),
            "--winrate-perspective",
            "black",
        ],
        check=True,
        capture_output=True,
    )
    output = tmp_path / "dataset"
    build = [
        *command,
        "build-dataset",
        "--sgf",
        str(sgf),
        "--teacher",
        str(targets),
        "--output",
        str(output),
        "--size",
        "5",
        "--stride",
        "2",
        "--require-teacher",
    ]
    subprocess.run(build, check=True, capture_output=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["teacher_targets"] == 2
    assert manifest["require_teacher"] is True
    assert sum(row["examples"] for row in manifest["files"].values()) == 2
    for split in manifest["files"]:
        with np.load(output / f"{split}.npz") as arrays:
            np.testing.assert_allclose(arrays["value"], 0.5)
    targets.write_text(targets.read_text().splitlines()[0] + "\n")
    incomplete = subprocess.run(build, capture_output=True, text=True)
    assert incomplete.returncode != 0
    assert "Missing teacher targets for 1 sampled positions" in incomplete.stderr


def test_cli_requires_final_teacher_responses(tmp_path: Path) -> None:
    sgf = tmp_path / "game.sgf"
    sgf.write_bytes(b"(;SZ[5]KM[0]RE[B+R];B[aa];W[bb];B[cc])")
    queries = tmp_path / "queries.jsonl"
    command = ["uv", "run", "--no-sync", "flygo"]
    subprocess.run(
        [*command, "teacher-queries", "--sgf", str(sgf), "--output", str(queries), "--stride", "2"],
        check=True,
        capture_output=True,
    )
    query = json.loads(queries.read_text())
    analysis = tmp_path / "analysis.jsonl"

    def response(turn: int, *, during: bool, visits: int) -> dict[str, object]:
        return {
            "id": query["id"],
            "turnNumber": turn,
            "isDuringSearch": during,
            "rootInfo": {"winrate": 0.75},
            "moveInfos": [{"move": query["moves"][turn][1], "visits": visits}],
        }

    provisional_only = [response(turn, during=True, visits=1) for turn in query["analyzeTurns"]]
    analysis.write_text("".join(json.dumps(row) + "\n" for row in provisional_only))
    targets = tmp_path / "targets.jsonl"
    rejected = subprocess.run(
        [
            *command,
            "teacher-import",
            "--sgf",
            str(sgf),
            "--analysis",
            str(analysis),
            "--output",
            str(targets),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "only provisional responses" in rejected.stderr
    assert not targets.exists()

    completed = [
        row
        for turn in query["analyzeTurns"]
        for row in (response(turn, during=True, visits=1), response(turn, during=False, visits=64))
    ]
    analysis.write_text("".join(json.dumps(row) + "\n" for row in completed))
    output = tmp_path / "dataset"
    subprocess.run(
        [
            *command,
            "teacher-import",
            "--sgf",
            str(sgf),
            "--analysis",
            str(analysis),
            "--output",
            str(targets),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            *command,
            "build-dataset",
            "--sgf",
            str(sgf),
            "--teacher",
            str(targets),
            "--output",
            str(output),
            "--size",
            "5",
            "--stride",
            "2",
            "--require-teacher",
        ],
        check=True,
        capture_output=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["teacher_targets"] == 2
    for split in manifest["files"]:
        with np.load(output / f"{split}.npz") as arrays:
            if arrays["value"].size:
                np.testing.assert_allclose(arrays["value"], 0.5)
                np.testing.assert_allclose(arrays["source"], 1)
