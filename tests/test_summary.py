import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from flygo.research import MODELS, ResearchConfig
from flygo.summary import render_markdown, summarize

GraphHashes = dict[str, str | None]
ChangedEdges = dict[str, int | None]

GRAPH: GraphHashes = {
    "male-cns": "m" * 64,
    "rewired": "r" * 64,
    "weight-shuffled": "w" * 64,
    "disconnected": "d" * 64,
    "linear": None,
    "mlp": None,
}
CHANGED: ChangedEdges = {
    "male-cns": 0,
    "rewired": 100,
    "weight-shuffled": 0,
    "disconnected": None,
    "linear": None,
    "mlp": None,
}


def normalized(config: ResearchConfig) -> dict[str, Any]:
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in config.__dict__.items()
    }


def make_report(
    tmp_path: Path,
    *,
    config: ResearchConfig | None = None,
    policy_loss: dict[str, float] | None = None,
    value_mse: dict[str, float] | None = None,
    graphs: GraphHashes | None = None,
    changed: ChangedEdges | None = None,
) -> tuple[Path, ResearchConfig, dict[str, Any]]:
    config = config if config is not None else ResearchConfig(seeds=(7, 17), epochs=2, batch_size=8)
    graphs = graphs if graphs is not None else dict(GRAPH)
    changed = changed if changed is not None else dict(CHANGED)
    policy_loss = policy_loss if policy_loss is not None else {model: 5.0 for model in MODELS}
    value_mse = value_mse if value_mse is not None else {model: 0.5 for model in MODELS}

    runs = tmp_path / "runs"
    records = []
    for seed in config.seeds:
        for model in MODELS:
            directory = runs / f"{model}-{seed}"
            directory.mkdir(parents=True)
            checkpoint = directory / "policy.npz"
            checkpoint.write_bytes(f"{model}-{seed}".encode())
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "graph_sha256": graphs[model],
                    "changed_target_edges": changed[model],
                    "trainable_parameters": 100,
                    "optimizer_updates": 10,
                    "selected_epoch": 2,
                    "edge_accumulations_per_evaluation": 50,
                    "checkpoint": f"{model}-{seed}/policy.npz",
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "training_seconds": 1.0,
                    "peak_traced_bytes": 1000,
                    "evaluation_seconds_per_example": 0.001,
                    "validation": {
                        "examples": 10,
                        "policy_loss": policy_loss[model],
                        "value_mse": value_mse[model],
                        "target_top1": 0.1,
                        "target_top3": 0.3,
                        "legal_action_rate": 0.9,
                    },
                    "test": None,
                }
            )
    report: dict[str, Any] = {
        "version": 1,
        "experiment_sha256": "0" * 64,
        "request": {
            "config": normalized(config),
            "models": list(MODELS),
            "dataset_manifest_sha256": "a" * 64,
            "graph_file_sha256": "b" * 64,
        },
        "test_evaluated": False,
        "runs": records,
    }
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "report.json").write_text(json.dumps(report))
    return runs, config, report


def rewrite(runs: Path, report: dict[str, Any]) -> None:
    (runs / "report.json").write_text(json.dumps(report))


def test_summary_reports_no_advantage_for_flat_metrics(tmp_path: Path) -> None:
    runs, config, _ = make_report(tmp_path)
    summary = summarize(runs=runs, config=config, models=MODELS)

    assert summary["verdict"]["topology_advantage"] is False
    assert summary["test_evaluated"] is False
    assert summary["optimizer_updates"] == 10
    assert all(item["informative"] for item in summary["controls"].values())
    assert any("threshold" in reason for reason in summary["verdict"]["reasons"])


def test_summary_reports_advantage_when_reference_clears_every_control(tmp_path: Path) -> None:
    policy = {model: 5.0 for model in MODELS} | {"male-cns": 1.0}
    value = {model: 0.5 for model in MODELS} | {"male-cns": 0.1}
    runs, config, _ = make_report(tmp_path, policy_loss=policy, value_mse=value)
    summary = summarize(runs=runs, config=config, models=MODELS)

    assert summary["verdict"]["topology_advantage"] is True
    assert summary["ranking"]["policy_loss"][0] == "male-cns"
    assert summary["ranking"]["value_mse"][0] == "male-cns"


def test_summary_records_value_regression(tmp_path: Path) -> None:
    value = {model: 0.5 for model in MODELS} | {"male-cns": 0.9}
    runs, config, _ = make_report(tmp_path, value_mse=value)
    summary = summarize(runs=runs, config=config, models=MODELS)

    assert summary["verdict"]["topology_advantage"] is False
    assert any("value error is higher" in reason for reason in summary["verdict"]["reasons"])


def test_unchanged_rewired_control_is_non_informative(tmp_path: Path) -> None:
    policy = {model: 5.0 for model in MODELS} | {"male-cns": 1.0}
    value = {model: 0.5 for model in MODELS} | {"male-cns": 0.1}
    graphs = dict(GRAPH) | {"rewired": GRAPH["male-cns"]}
    changed = dict(CHANGED) | {"rewired": 0}
    runs, config, _ = make_report(
        tmp_path, policy_loss=policy, value_mse=value, graphs=graphs, changed=changed
    )
    summary = summarize(runs=runs, config=config, models=MODELS)

    assert summary["controls"]["rewired"]["informative"] is False
    assert summary["verdict"]["topology_advantage"] is False
    assert any("non-informative" in reason for reason in summary["verdict"]["reasons"])


def test_missing_run_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["runs"] = [item for item in report["runs"] if item["model"] != "mlp"]
    rewrite(runs, report)

    with pytest.raises(ValueError, match="Missing runs"):
        summarize(runs=runs, config=config, models=MODELS)


def test_unexpected_run_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["runs"].append(dict(report["runs"][0]))
    rewrite(runs, report)

    with pytest.raises(ValueError, match="Duplicate run"):
        summarize(runs=runs, config=config, models=MODELS)


def test_test_metrics_are_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["runs"][0]["test"] = dict(report["runs"][0]["validation"])
    rewrite(runs, report)

    with pytest.raises(ValueError, match="test split is closed"):
        summarize(runs=runs, config=config, models=MODELS)


def test_damaged_checkpoint_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    (runs / report["runs"][0]["checkpoint"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checkpoint hash"):
        summarize(runs=runs, config=config, models=MODELS)


def test_config_mismatch_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["request"]["config"]["epochs"] = 99
    rewrite(runs, report)

    with pytest.raises(ValueError, match="config does not match"):
        summarize(runs=runs, config=config, models=MODELS)


def test_open_test_split_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["test_evaluated"] = True
    rewrite(runs, report)

    with pytest.raises(ValueError, match="closed test split"):
        summarize(runs=runs, config=config, models=MODELS)


def test_mixed_dataset_hash_is_rejected(tmp_path: Path) -> None:
    runs, config, _ = make_report(tmp_path)

    with pytest.raises(ValueError, match="different dataset manifest"):
        summarize(
            runs=runs,
            config=config,
            models=MODELS,
            dataset_manifest_sha256="f" * 64,
        )


def test_tampered_reported_comparison_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["validation_comparisons"] = {
        "rewired": {
            "policy_loss": {
                "mean_difference": 999.0,
                "paired_seed_bootstrap_95": [0.0, 0.0],
            }
        }
    }
    rewrite(runs, report)

    with pytest.raises(ValueError, match="comparison mean differs"):
        summarize(runs=runs, config=config, models=MODELS)


def test_non_finite_metric_is_rejected(tmp_path: Path) -> None:
    runs, config, report = make_report(tmp_path)
    report["runs"][0]["validation"]["policy_loss"] = float("inf")
    rewrite(runs, report)

    with pytest.raises(ValueError, match="not finite"):
        summarize(runs=runs, config=config, models=MODELS)


def test_markdown_contains_verdict_and_tables(tmp_path: Path) -> None:
    runs, config, _ = make_report(tmp_path)
    markdown = render_markdown(summarize(runs=runs, config=config, models=MODELS))

    assert "No topology advantage was detected." in markdown
    assert "| model | policy loss |" in markdown
    assert "| rewired |" in markdown
    assert "male-cns" in markdown
