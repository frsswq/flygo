import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygo.dataset import build_dataset, parse_sgf_collection, split_of
from flygo.research import ResearchConfig, run_research


def research_dataset(path: Path) -> None:
    games = []
    for index, split in enumerate(("train", "validation", "test")):
        for salt in range(1000):
            sgf = (
                f"(;SZ[19]KM[7.5]RE[B+R]C[{index}-{salt}];B[{chr(97 + index)}a];W[ss];B[jj])"
            ).encode()
            game = parse_sgf_collection(sgf)[0]
            if split_of(game.game_id) == split:
                games.append(game)
                break
        else:
            raise AssertionError("Could not construct a game for each split")
    build_dataset(games, path)


def test_experiment_cli_is_matched_reproducible_and_keeps_test_data_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    research_dataset(dataset)
    (dataset / "test.npz").unlink()
    graph = tmp_path / "graph.parquet"
    pl.DataFrame({"pre": [1, 2, 3, 4], "post": [2, 3, 4, 1], "weight": [1, 2, 3, 4]}).write_parquet(
        graph
    )
    command = [
        "uv",
        "run",
        "--no-sync",
        "flygo",
        "experiment",
        "--dataset",
        str(dataset),
        "--graph",
        str(graph),
        "--seeds",
        "7",
        "11",
        "--epochs",
        "2",
        "--steps",
        "2",
        "--batch-size",
        "2",
    ]
    reports = []
    for folder in ("first", "first", "second"):
        output = tmp_path / folder
        subprocess.run(
            [*command, "--output", str(output)], check=True, capture_output=True, text=True
        )
        report = json.loads((output / "report.json").read_text())
        assert report["test_evaluated"] is False
        assert len(report["runs"]) == 12
        assert {run["model"] for run in report["runs"]} == {
            "male-cns",
            "rewired",
            "weight-shuffled",
            "disconnected",
            "linear",
            "mlp",
        }
        for seed in (7, 11):
            runs = [run for run in report["runs"] if run["seed"] == seed]
            assert (
                len({run["trainable_parameters"] for run in runs if run["model"] != "linear"}) == 1
            )
            assert len({run["optimizer_updates"] for run in runs}) == 1
        for run in report["runs"]:
            assert run["test"] is None
            assert run["selected_epoch"] in (1, 2)
            assert run["validation"]["examples"] > 0
            assert run["training_seconds"] > 0
            assert run["peak_traced_bytes"] > 0
            checkpoint = output / run["checkpoint"]
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == run["checkpoint_sha256"]
        reports.append(report)
    assert reports[0] == reports[1]
    for first, second in zip(reports[0]["runs"], reports[2]["runs"], strict=True):
        assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
        assert first["validation"] == second["validation"]
        assert first["history"] == second["history"]
    changed = subprocess.run(
        [*command, "--output", str(tmp_path / "first"), "--learning-rate", "0.2"],
        capture_output=True,
        text=True,
    )
    assert changed.returncode != 0
    assert "different experiment" in changed.stderr
    assert not np.isnan(reports[0]["runs"][0]["validation"]["policy_loss"])


def test_final_test_requires_opt_in_and_resume_recovers_an_incomplete_run(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    research_dataset(dataset)
    graph = tmp_path / "graph.parquet"
    pl.DataFrame({"pre": [1, 2], "post": [2, 1], "weight": [1, 2]}).write_parquet(graph)
    output = tmp_path / "final"
    config = ResearchConfig(seeds=(7,), epochs=1, steps=2, final_test=True)
    report = run_research(dataset, graph, output, config)
    assert report["test_evaluated"] is True
    assert all(run["test"]["examples"] > 0 for run in report["runs"])
    assert (
        report["validation_comparisons"]["mlp"]["policy_loss"]["paired_seed_bootstrap_95"] is None
    )

    # A crash after saving a checkpoint but before publishing its result is recoverable.
    (output / "male-cns-7" / "result.json").unlink()
    recovered = run_research(dataset, graph, output, config)
    assert recovered["runs"][0]["checkpoint_sha256"] == report["runs"][0]["checkpoint_sha256"]
    assert recovered["runs"][1:] == report["runs"][1:]
    checkpoint = output / "mlp-7" / "policy.npz"
    checkpoint.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="damaged"):
        run_research(dataset, graph, output, config)


def test_experiment_rejects_changed_dataset_before_writing_results(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    research_dataset(dataset)
    with (dataset / "train.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        run_research(dataset, tmp_path / "unused.parquet", tmp_path / "output", ResearchConfig())
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("seeds", [(), (7, 7), (-1,)])
def test_experiment_rejects_invalid_seed_sets(seeds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="seeds"):
        ResearchConfig(seeds=seeds)
