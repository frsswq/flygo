"""Validate completed matched research runs and aggregate them into a summary.

The summary is derived, never hand-written.
It refuses an incomplete grid, a metric that is not finite, a checkpoint that fails its
hash, a run that read the closed test split, a request that does not match the protocol,
and an unchanged control that cannot support attribution.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flygo.official_data import file_sha256
from flygo.research import ResearchConfig

EXPERIMENT_MODEL = "male-cns"
POLICY_LOSS = "policy_loss"
VALUE_MSE = "value_mse"
COMPARED_METRICS = (POLICY_LOSS, VALUE_MSE)
GRAPH_CONTROLS = ("rewired", "weight-shuffled", "disconnected")
POLICY_LOSS_THRESHOLD = 0.01
STANDARD_CONFIDENCE = 0.95
STANDARD_RESAMPLES = 2000
FINAL_RESAMPLES = 20_000
FINAL_FAMILY_ALPHA = 0.01
REPORT_SCOPE = "Supervised screening only; not an Elo or topology-benefit claim"


class ValidationMetrics(BaseModel):
    """Held-out metrics for one run, taken at its selected epoch."""

    model_config = ConfigDict(extra="ignore")

    examples: int = Field(ge=1)
    policy_loss: float
    value_mse: float
    target_top1: float
    target_top3: float
    legal_action_rate: float

    def reject_non_finite(self, where: str) -> None:
        for name in (
            POLICY_LOSS,
            VALUE_MSE,
            "target_top1",
            "target_top3",
            "legal_action_rate",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{where}: validation metric {name} is not finite")
        if not 0.0 <= self.target_top1 <= 1.0 or not 0.0 <= self.target_top3 <= 1.0:
            raise ValueError(f"{where}: target agreement is outside [0, 1]")
        if not 0.0 <= self.legal_action_rate <= 1.0:
            raise ValueError(f"{where}: legal-action rate is outside [0, 1]")


class RunRecord(BaseModel):
    """One model and one seed, as written by the experiment runner."""

    model_config = ConfigDict(extra="ignore")

    model: str
    seed: int = Field(ge=0)
    graph_sha256: str | None = None
    changed_target_edges: int | None = None
    trainable_parameters: int = Field(ge=0)
    optimizer_updates: int = Field(ge=0)
    selected_epoch: int = Field(ge=1)
    edge_accumulations_per_evaluation: int = Field(ge=0)
    checkpoint: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_seconds: float
    peak_traced_bytes: int = Field(ge=0)
    evaluation_seconds_per_example: float
    validation: ValidationMetrics
    test: ValidationMetrics | None = None


def _short(value: str | None) -> str:
    return value[:12] if value else "none"


def _graph_text(hashes: Sequence[str]) -> str:
    if not hashes:
        return "none"
    if len(hashes) == 1:
        return f"`{_short(hashes[0])}`"
    return f"{len(hashes)} distinct hashes, one per seed"


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    return value


def load_report(runs: Path) -> dict[str, Any]:
    """Read the run directory's report, which is the experiment's publication record."""
    path = runs / "report.json"
    if not path.is_file():
        raise ValueError(f"Runs directory has no report: {path}")
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable report: {path}") from error
    if not isinstance(report, dict):
        raise ValueError("Report must be a JSON object")
    return report


def _expected_grid(models: Sequence[str], seeds: Sequence[int]) -> set[tuple[str, int]]:
    return {(model, seed) for seed in seeds for model in models}


def _parse_runs(
    report: dict[str, Any],
    *,
    models: Sequence[str],
    seeds: Sequence[int],
) -> list[RunRecord]:
    raw = report.get("runs")
    if not isinstance(raw, list):
        raise ValueError("Report has no runs list")
    records = [RunRecord.model_validate(item) for item in raw]

    seen: dict[tuple[str, int], RunRecord] = {}
    for record in records:
        key = (record.model, record.seed)
        if key in seen:
            raise ValueError(f"Duplicate run for model {record.model} seed {record.seed}")
        seen[key] = record

    expected = _expected_grid(models, seeds)
    missing = sorted(expected - set(seen))
    if missing:
        raise ValueError(f"Missing runs for {missing}")
    extra = sorted(set(seen) - expected)
    if extra:
        raise ValueError(f"Unexpected runs for {extra}")
    return [seen[key] for key in sorted(seen, key=lambda key: (key[1], key[0]))]


def _check_request(
    report: dict[str, Any],
    *,
    config: ResearchConfig,
    models: Sequence[str],
    seeds: Sequence[int],
    dataset_manifest_sha256: str | None,
    graph_file_sha256: str | None,
) -> None:
    request = report.get("request")
    if not isinstance(request, dict):
        raise ValueError("Report has no request block")
    expected_config = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in config.__dict__.items()
    }
    if request.get("config") != expected_config:
        raise ValueError("Report request config does not match the expected configuration")
    if list(request.get("models") or []) != list(models):
        raise ValueError("Report request models do not match the expected models")
    if list(request.get("config", {}).get("seeds") or []) != list(seeds):
        raise ValueError("Report request seeds do not match the expected seeds")
    test_evaluated = report.get("test_evaluated")
    if not isinstance(test_evaluated, bool) or test_evaluated:
        raise ValueError("Report does not record a closed test split")
    if dataset_manifest_sha256 is not None and (
        request.get("dataset_manifest_sha256") != dataset_manifest_sha256
    ):
        raise ValueError("Report was produced from a different dataset manifest")
    if graph_file_sha256 is not None and (request.get("graph_file_sha256") != graph_file_sha256):
        raise ValueError("Report was produced from a different circuit file")


def _check_runs(records: Sequence[RunRecord], runs: Path) -> None:
    budgets = {record.optimizer_updates for record in records}
    if len(budgets) != 1:
        raise ValueError(f"Runs do not share one optimizer budget: {sorted(budgets)}")
    for record in records:
        where = f"{record.model} seed {record.seed}"
        if record.test is not None:
            raise ValueError(f"{where}: test metrics are present but the test split is closed")
        record.validation.reject_non_finite(where)
        if not math.isfinite(record.training_seconds) or record.training_seconds <= 0:
            raise ValueError(f"{where}: training time is not positive and finite")
        if (
            not math.isfinite(record.evaluation_seconds_per_example)
            or record.evaluation_seconds_per_example < 0
        ):
            raise ValueError(f"{where}: evaluation time is not finite and nonnegative")
        checkpoint = runs / record.checkpoint
        if not checkpoint.is_file():
            raise ValueError(f"{where}: checkpoint is missing: {checkpoint}")
        if file_sha256(checkpoint) != record.checkpoint_sha256:
            raise ValueError(f"{where}: checkpoint hash does not match the record")


def _mean(values: Sequence[float]) -> float:
    return np.asarray(values, dtype=np.float64).mean().item()


def _one_value(values: Sequence[Any], where: str) -> Any:
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(f"{where}: runs disagree: {sorted(distinct, key=str)}")
    return next(iter(distinct))


def _model_summaries(records: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model].append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for model, group in by_model.items():
        validation = [record.validation for record in group]
        summaries[model] = {
            "seeds": sorted(record.seed for record in group),
            "validation_examples": _one_value(
                [item.examples for item in validation], f"{model} validation examples"
            ),
            "policy_loss": _mean([item.policy_loss for item in validation]),
            "value_mse": _mean([item.value_mse for item in validation]),
            "target_top1": _mean([item.target_top1 for item in validation]),
            "target_top3": _mean([item.target_top3 for item in validation]),
            "legal_action_rate": _mean([item.legal_action_rate for item in validation]),
            "trainable_parameters": _one_value(
                [record.trainable_parameters for record in group], f"{model} parameters"
            ),
            "edge_accumulations_per_evaluation": _one_value(
                [record.edge_accumulations_per_evaluation for record in group],
                f"{model} edge accumulations",
            ),
            "optimizer_updates": _one_value(
                [record.optimizer_updates for record in group], f"{model} optimizer updates"
            ),
            "selected_epochs": sorted({record.selected_epoch for record in group}),
            "graph_sha256": sorted(
                {record.graph_sha256 for record in group if record.graph_sha256 is not None}
            ),
            "changed_target_edges": sorted(
                {
                    record.changed_target_edges
                    for record in group
                    if record.changed_target_edges is not None
                }
            ),
            "training_seconds_mean": _mean([record.training_seconds for record in group]),
            "training_seconds_total": sum(record.training_seconds for record in group),
            "peak_traced_bytes_max": max(record.peak_traced_bytes for record in group),
            "evaluation_seconds_per_example": _mean(
                [record.evaluation_seconds_per_example for record in group]
            ),
        }
    return summaries


def _bootstrap(differences: np.ndarray, resamples: int, tails: tuple[float, float]) -> list[float]:
    generator = np.random.default_rng(0)
    means = generator.choice(differences, (resamples, len(differences))).mean(axis=1)
    return np.quantile(means, list(tails)).tolist()


def _comparisons(
    records: Sequence[RunRecord],
    *,
    models: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    lookup = {(record.model, record.seed): record for record in records}
    controls = [model for model in models if model != EXPERIMENT_MODEL]
    per_comparison_alpha = FINAL_FAMILY_ALPHA / len(controls)
    final_tails = (per_comparison_alpha / 2, 1 - per_comparison_alpha / 2)
    standard_tails = ((1 - STANDARD_CONFIDENCE) / 2, 1 - (1 - STANDARD_CONFIDENCE) / 2)

    comparisons: dict[str, dict[str, Any]] = {}
    for control in controls:
        metrics: dict[str, Any] = {}
        for metric in COMPARED_METRICS:
            differences = np.asarray(
                [
                    getattr(lookup[(EXPERIMENT_MODEL, seed)].validation, metric)
                    - getattr(lookup[(control, seed)].validation, metric)
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            metrics[metric] = {
                "male_cns_minus_control": differences.tolist(),
                "mean_difference": differences.mean().item(),
                "standard_interval": _bootstrap(differences, STANDARD_RESAMPLES, standard_tails),
                "family_adjusted_interval": _bootstrap(differences, FINAL_RESAMPLES, final_tails),
            }
        comparisons[control] = metrics
    return comparisons


def _check_reported_comparisons(
    report: dict[str, Any], comparisons: dict[str, dict[str, Any]]
) -> None:
    reported = report.get("validation_comparisons")
    if not isinstance(reported, dict):
        return
    for control, metrics in comparisons.items():
        for metric in COMPARED_METRICS:
            entry = reported.get(control, {}).get(metric)
            if not isinstance(entry, dict):
                raise ValueError(f"Report comparison is missing for {control} {metric}")
            reported_mean = _number(
                entry.get("mean_difference"), f"{control} {metric} reported mean"
            )
            if abs(reported_mean - metrics[metric]["mean_difference"]) > 1e-9:
                raise ValueError(f"Report comparison mean differs for {control} {metric}")
            reported_interval = entry.get("paired_seed_bootstrap_95")
            if not isinstance(reported_interval, list) or len(reported_interval) != 2:
                raise ValueError(f"Report comparison interval is malformed for {control} {metric}")
            reported_lower = _number(reported_interval[0], f"{control} {metric} reported interval")
            if abs(reported_lower - metrics[metric]["standard_interval"][0]) > 1e-9:
                raise ValueError(f"Report comparison interval differs for {control} {metric}")


def _control_status(
    summaries: dict[str, dict[str, Any]], models: Sequence[str]
) -> dict[str, dict[str, Any]]:
    reference_hashes = set(summaries[EXPERIMENT_MODEL]["graph_sha256"])
    status: dict[str, dict[str, Any]] = {}
    for control in (model for model in models if model != EXPERIMENT_MODEL):
        summary = summaries[control]
        hashes = set(summary["graph_sha256"])
        graph_differs = bool(hashes) and hashes.isdisjoint(reference_hashes)
        changed = summary["changed_target_edges"]
        if control == "rewired":
            informative = graph_differs and bool(changed) and min(changed) > 0
            note = "endpoints permuted" if informative else "no endpoint change; non-informative"
        elif control == "weight-shuffled":
            informative = graph_differs
            note = "weights reshuffled" if informative else "same effective graph; non-informative"
        elif control == "disconnected":
            informative = graph_differs
            note = "no connectome edges" if informative else "same graph; non-informative"
        else:
            informative = True
            note = "no connectome graph"
        status[control] = {
            "informative": informative,
            "graph_differs_from_reference": (graph_differs if control in GRAPH_CONTROLS else None),
            "changed_target_edges": changed,
            "note": note,
        }
    return status


def _verdict(
    comparisons: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    non_informative = sorted(
        control for control, item in controls.items() if not item["informative"]
    )
    if non_informative:
        reasons.append(f"non-informative controls: {', '.join(non_informative)}")

    policy_clears = {
        control: item[POLICY_LOSS]["family_adjusted_interval"][1] < -POLICY_LOSS_THRESHOLD
        for control, item in comparisons.items()
    }
    fails_policy = sorted(control for control, cleared in policy_clears.items() if not cleared)
    if fails_policy:
        reasons.append(
            "policy loss does not clear the "
            f"{POLICY_LOSS_THRESHOLD} nat threshold against: {', '.join(fails_policy)}"
        )

    value_regressions = sorted(
        control for control, item in comparisons.items() if item[VALUE_MSE]["mean_difference"] > 0
    )
    if value_regressions:
        reasons.append(f"value error is higher against: {', '.join(value_regressions)}")

    advantage = not reasons
    if advantage:
        reasons.append(
            "policy loss clears the threshold against every control with no value regression"
        )
    return {
        "topology_advantage": advantage,
        "policy_loss_threshold": POLICY_LOSS_THRESHOLD,
        "policy_loss_clears_threshold": policy_clears,
        "reasons": reasons,
    }


def _ranking(summaries: dict[str, dict[str, Any]], metric: str) -> list[str]:
    return sorted(summaries, key=lambda model: summaries[model][metric])


def summarize(
    *,
    runs: Path,
    config: ResearchConfig,
    models: Sequence[str],
    dataset_manifest_sha256: str | None = None,
    graph_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a completed run directory and return its machine-readable summary."""
    report = load_report(runs)
    seeds = tuple(config.seeds)
    _check_request(
        report,
        config=config,
        models=models,
        seeds=seeds,
        dataset_manifest_sha256=dataset_manifest_sha256,
        graph_file_sha256=graph_file_sha256,
    )
    records = _parse_runs(report, models=models, seeds=seeds)
    _check_runs(records, runs)
    summaries = _model_summaries(records)
    comparisons = _comparisons(records, models=models, seeds=seeds)
    _check_reported_comparisons(report, comparisons)
    controls = _control_status(summaries, models)
    verdict = _verdict(comparisons, controls)

    return {
        "version": 1,
        "scope": REPORT_SCOPE,
        "test_evaluated": False,
        "protocol": report.get("request", {}).get("protocol"),
        "experiment_sha256": report.get("experiment_sha256"),
        "dataset_manifest_sha256": report.get("request", {}).get("dataset_manifest_sha256"),
        "graph_file_sha256": report.get("request", {}).get("graph_file_sha256"),
        "models": list(models),
        "seeds": list(seeds),
        "optimizer_updates": records[0].optimizer_updates,
        "confidence": {
            "standard": STANDARD_CONFIDENCE,
            "standard_resamples": STANDARD_RESAMPLES,
            "family_alpha": FINAL_FAMILY_ALPHA,
            "family_resamples": FINAL_RESAMPLES,
            "family_adjustment": "Bonferroni across the compared controls",
        },
        "verdict": verdict,
        "controls": controls,
        "ranking": {
            "policy_loss": _ranking(summaries, POLICY_LOSS),
            "value_mse": _ranking(summaries, VALUE_MSE),
        },
        "models_summary": summaries,
        "comparisons": comparisons,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the one-page human readout for a summary."""
    models = summary["models"]
    summaries = summary["models_summary"]
    comparisons = summary["comparisons"]
    controls = summary["controls"]
    verdict = summary["verdict"]
    lines: list[str] = []
    add = lines.append

    add("# Feasibility supervised screen result")
    add("")
    add("## Verdict")
    add("")
    if verdict["topology_advantage"]:
        add("The frozen MaleCNS wiring cleared the predeclared policy-loss threshold.")
    else:
        add("No topology advantage was detected.")
    for reason in verdict["reasons"]:
        add(f"- {reason}")
    add("")
    add(f"- Scope: {summary['scope']}.")
    add("- The closed test split was not read.")
    add(f"- Optimizer updates per run: {summary['optimizer_updates']}.")
    add(
        f"- Intervals: {summary['confidence']['standard']:.0%} for reading, "
        f"{1 - summary['confidence']['family_alpha']:.0%} family-adjusted "
        "(Bonferroni) for the threshold."
    )
    add("")
    add("## Held-out metrics at the selected epoch")
    add("")
    add("Means over seeds. Lower loss and higher agreement are better.")
    add("")
    add("| model | policy loss | value MSE | top-1 | top-3 | legal | parameters | val examples |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in models:
        row = summaries[model]
        add(
            f"| {model} | {row['policy_loss']:.4f} | {row['value_mse']:.4f} | "
            f"{row['target_top1']:.2%} | {row['target_top3']:.2%} | "
            f"{row['legal_action_rate']:.3f} | {row['trainable_parameters']:,} | "
            f"{row['validation_examples']} |"
        )
    add("")
    add(f"- Policy loss ranking, best first: {', '.join(summary['ranking']['policy_loss'])}.")
    add(f"- Value error ranking, best first: {', '.join(summary['ranking']['value_mse'])}.")
    add("")
    add("## Paired comparison against each control")
    add("")
    add(f"`{EXPERIMENT_MODEL} minus control`. A positive value means the reference model is worse.")
    add("")
    add(
        "| control | policy difference | "
        f"{summary['confidence']['standard']:.0%} interval | adjusted interval | "
        "clears threshold | value MSE difference | control informative |"
    )
    add("| --- | ---: | ---: | ---: | :---: | ---: | :---: |")
    for control in models:
        if control == EXPERIMENT_MODEL:
            continue
        policy = comparisons[control][POLICY_LOSS]
        value = comparisons[control][VALUE_MSE]
        standard = policy["standard_interval"]
        adjusted = policy["family_adjusted_interval"]
        add(
            f"| {control} | {policy['mean_difference']:+.4f} | "
            f"{standard[0]:+.4f} to {standard[1]:+.4f} | "
            f"{adjusted[0]:+.4f} to {adjusted[1]:+.4f} | "
            f"{'yes' if verdict['policy_loss_clears_threshold'][control] else 'no'} | "
            f"{value['mean_difference']:+.4f} | "
            f"{'yes' if controls[control]['informative'] else 'no'} |"
        )
    add("")
    add("## Control informativeness")
    add("")
    for control in models:
        if control == EXPERIMENT_MODEL:
            continue
        item = controls[control]
        changed = item["changed_target_edges"]
        changed_text = ", ".join(str(value) for value in changed) if changed else "none"
        differs = item["graph_differs_from_reference"]
        differs_text = "n/a" if differs is None else ("yes" if differs else "no")
        add(
            f"- {control}: {item['note']}. Endpoint changes: {changed_text}. "
            f"Graph differs: {differs_text}."
        )
    add("")
    add("## Limits")
    add("")
    add("- This is supervised screening on held-out positions, not playing strength or Elo.")
    add("- Comparison is at validation only; model selection used validation, not test.")
    add(
        "- Absolute quality is weak, so read the differences as the result, not the "
        "absolute numbers."
    )
    add(
        f"- {len(summary['seeds'])} seeds give wide intervals. An unfavourable direction "
        "is not proof of harm."
    )
    add(
        "- The locked final experiment requires a stronger teacher corpus, 99 percent "
        "intervals, and a fixed external-engine league."
    )
    add("")
    add("## Provenance")
    add("")
    add(f"- Experiment: `{summary['experiment_sha256']}`")
    add(f"- Dataset manifest: `{summary['dataset_manifest_sha256']}`")
    add(f"- Circuit file: `{summary['graph_file_sha256']}`")
    for model in models:
        add(
            f"- {model}: graph {_graph_text(summaries[model]['graph_sha256'])}, "
            f"edge accumulations {summaries[model]['edge_accumulations_per_evaluation']:,}, "
            f"training seconds total {summaries[model]['training_seconds_total']:.1f}, "
            f"peak traced bytes {summaries[model]['peak_traced_bytes_max']:,}"
        )
    add("")
    return "\n".join(lines)
