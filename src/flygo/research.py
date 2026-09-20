"""Matched, resumable supervised experiments. No playing-strength claims are made here."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter

from flygo.connectome import FrozenConnectome, load_connectome
from flygo.go import BOARD_SIZES, RULESET
from flygo.model import (
    DEFAULT_RECURRENT_GAIN,
    DEFAULT_RETENTION,
    ConnectomePolicy,
    DensePolicy,
    Policy,
    policy_parameters,
    validate_dynamics,
)
from flygo.official_data import file_sha256
from flygo.training import (
    EpochMetrics,
    EvaluationMetrics,
    TrainingData,
    evaluate_policy,
    fit_policy,
    graph_sha256,
    load_policy,
    load_training_data,
    save_policy,
    selected_epoch,
)

type ModelName = Literal["male-cns", "rewired", "weight-shuffled", "disconnected", "linear", "mlp"]
type Normalization = Literal["incoming", "none"]
type WeightMode = Literal["weighted", "binary"]
MODELS: tuple[ModelName, ...] = (
    "male-cns",
    "rewired",
    "weight-shuffled",
    "disconnected",
    "linear",
    "mlp",
)


@dataclass(frozen=True)
class ResearchConfig:
    size: int = 19
    seeds: tuple[int, ...] = (7, 17, 27)
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    steps: int = 8
    retention: float = DEFAULT_RETENTION
    recurrent_gain: float = DEFAULT_RECURRENT_GAIN
    normalization: Normalization = "incoming"
    weights: WeightMode = "weighted"
    value_weight: float = 1.0
    final_test: bool = False

    def __post_init__(self) -> None:
        if self.size not in BOARD_SIZES:
            raise ValueError("Unsupported research board size")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Research seeds must be distinct nonnegative integers")
        if min(self.epochs, self.batch_size) < 1:
            raise ValueError("Research counts must be positive")
        validate_dynamics(
            steps=self.steps,
            retention=self.retention,
            recurrent_gain=self.recurrent_gain,
        )
        if self.normalization not in ("incoming", "none"):
            raise ValueError("Research normalization must be incoming or none")
        if self.weights not in ("weighted", "binary"):
            raise ValueError("Research weight mode must be weighted or binary")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Research learning rate must be finite and positive")
        if not math.isfinite(self.value_weight) or self.value_weight < 0:
            raise ValueError("Research value weight must be finite and nonnegative")


class SplitManifest(BaseModel):
    file: str
    examples: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(BaseModel):
    version: Literal[1]
    size: Literal[5, 19]
    ruleset: str
    files: dict[str, SplitManifest]


@dataclass(frozen=True)
class RunResult:
    model: ModelName
    seed: int
    graph_sha256: str | None
    changed_target_edges: int | None
    trainable_parameters: int
    optimizer_updates: int
    selected_epoch: int
    edge_accumulations_per_evaluation: int
    checkpoint: str
    checkpoint_sha256: str
    training_seconds: float
    peak_traced_bytes: int
    evaluation_seconds_per_example: float
    history: list[EpochMetrics]
    validation: EvaluationMetrics
    test: EvaluationMetrics | None


def _write_json(path: Path, value: Any) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _load_splits(dataset: Path, config: ResearchConfig) -> dict[str, TrainingData]:
    manifest = DatasetManifest.model_validate_json((dataset / "manifest.json").read_bytes())
    if manifest.size != config.size or manifest.ruleset != RULESET:
        raise ValueError("Dataset size or rules do not match the experiment")
    names = ("train", "validation", "test") if config.final_test else ("train", "validation")
    splits: dict[str, TrainingData] = {}
    for name in names:
        entry = manifest.files.get(name)
        if entry is None or entry.file != f"{name}.npz":
            raise ValueError(f"Missing or invalid {name} split manifest")
        path = dataset / entry.file
        if file_sha256(path) != entry.sha256:
            raise ValueError(f"Dataset {name} hash does not match the manifest")
        data = load_training_data(path, size=config.size)
        if data.features.shape[0] != entry.examples or entry.examples == 0:
            raise ValueError(f"Dataset {name} must have its declared nonzero example count")
        splits[name] = data
    return splits


def research_model(
    name: ModelName, graph: FrozenConnectome, config: ResearchConfig, *, seed: int
) -> Policy:
    """Construct one comparison with matched boundary initialization and frozen wiring."""
    if name in {"linear", "mlp"}:
        width = graph.node_count if name == "mlp" else None
        return DensePolicy.initialize(size=config.size, seed=seed, hidden_count=width)
    match name:
        case "male-cns":
            control = graph
        case "rewired":
            control = graph.randomized(seed=seed)
        case "weight-shuffled":
            control = graph.shuffled_weights(seed=seed)
        case "disconnected":
            control = graph.without_connections()
        case _:
            raise ValueError(f"Unknown research model: {name}")
    if config.weights == "binary":
        control = control.binarized_weights()
    if config.normalization == "none":
        control = control.without_normalization()
    return ConnectomePolicy.initialize(
        control,
        size=config.size,
        seed=seed,
        steps=config.steps,
        retention=config.retention,
        recurrent_gain=config.recurrent_gain,
    )


def _run_model(
    name: ModelName,
    seed: int,
    graph: FrozenConnectome,
    splits: dict[str, TrainingData],
    config: ResearchConfig,
    output: Path,
    experiment_sha256: str,
) -> RunResult:
    directory = output / f"{name}-{seed}"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "policy.npz"
    result_path = directory / "result.json"
    model = research_model(name, graph, config, seed=seed)
    if result_path.exists():
        result = TypeAdapter(RunResult).validate_json(result_path.read_bytes())
        if (
            result.model != name
            or result.seed != seed
            or file_sha256(checkpoint) != result.checkpoint_sha256
        ):
            raise ValueError(f"Cached result or checkpoint is damaged: {directory}")
        _, metadata = load_policy(
            checkpoint, model.connectome if isinstance(model, ConnectomePolicy) else None
        )
        if metadata.get("experiment_sha256") != experiment_sha256:
            raise ValueError(f"Checkpoint belongs to a different experiment: {directory}")
        return result

    started = time.perf_counter()
    tracemalloc.start()
    try:
        history = fit_policy(
            model,
            splits["train"],
            validation=splits["validation"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            value_weight=config.value_weight,
            seed=seed,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    training_seconds = time.perf_counter() - started
    selected = selected_epoch(history, value_weight=config.value_weight)
    save_policy(
        checkpoint,
        model,
        metadata={
            "experiment_sha256": experiment_sha256,
            "model": name,
            "seed": seed,
            "selected_epoch": selected,
            "history": [asdict(epoch) for epoch in history],
        },
    )
    # Evaluate the persisted artifact, not only the in-memory model.
    restored, _ = load_policy(
        checkpoint, model.connectome if isinstance(model, ConnectomePolicy) else None
    )
    started = time.perf_counter()
    validation = evaluate_policy(restored, splits["validation"], batch_size=config.batch_size)
    evaluation_seconds = time.perf_counter() - started
    test = (
        evaluate_policy(restored, splits["test"], batch_size=config.batch_size)
        if config.final_test
        else None
    )
    is_graph = isinstance(model, ConnectomePolicy)
    result = RunResult(
        model=name,
        seed=seed,
        graph_sha256=graph_sha256(model.connectome) if is_graph else None,
        changed_target_edges=(
            int(np.count_nonzero(model.connectome.target_indices != graph.target_indices))
            if isinstance(model, ConnectomePolicy) and name != "disconnected"
            else None
        ),
        trainable_parameters=sum(parameter.size for parameter in policy_parameters(model)),
        optimizer_updates=config.epochs
        * math.ceil(splits["train"].features.shape[0] / config.batch_size),
        selected_epoch=selected,
        edge_accumulations_per_evaluation=model.steps * model.connectome.edge_count
        if is_graph
        else 0,
        checkpoint=str(checkpoint.relative_to(output)),
        checkpoint_sha256=file_sha256(checkpoint),
        training_seconds=training_seconds,
        peak_traced_bytes=peak_bytes,
        evaluation_seconds_per_example=evaluation_seconds / validation.examples,
        history=history,
        validation=validation,
        test=test,
    )
    _write_json(result_path, asdict(result))
    return result


def _comparisons(runs: list[RunResult], config: ResearchConfig) -> dict[str, Any]:
    lookup = {(run.model, run.seed): run for run in runs}
    comparisons: dict[str, Any] = {}
    for control in MODELS[1:]:
        metrics: dict[str, Any] = {}
        for metric in ("policy_loss", "value_mse"):
            differences = np.asarray(
                [
                    getattr(lookup["male-cns", seed].validation, metric)
                    - getattr(lookup[control, seed].validation, metric)
                    for seed in config.seeds
                ]
            )
            interval = None
            if len(config.seeds) >= 2:
                generator = np.random.default_rng(0)
                means = generator.choice(differences, (2000, len(differences))).mean(axis=1)
                interval = np.quantile(means, [0.025, 0.975]).tolist()
            metrics[metric] = {
                "male_cns_minus_control": differences.tolist(),
                "mean_difference": float(differences.mean()),
                "paired_seed_bootstrap_95": interval,
            }
        comparisons[control] = metrics
    return comparisons


def run_research(
    dataset: Path, graph_path: Path, output: Path, config: ResearchConfig
) -> dict[str, Any]:
    """Run all controls with exclusive output ownership and resumable atomic artifacts."""
    splits = _load_splits(dataset, config)
    graph = load_connectome(graph_path)
    source_directory = Path(__file__).parent
    source_hashes = {path.name: file_sha256(path) for path in sorted(source_directory.glob("*.py"))}
    request = {
        "version": 1,
        "config": asdict(config),
        "models": MODELS,
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
        "graph_file_sha256": file_sha256(graph_path),
        "graph_sha256": graph_sha256(graph),
        "source_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
    }
    # JSON round-trip normalizes tuples so resumed requests compare exactly.
    request = json.loads(json.dumps(request, sort_keys=True))
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request_path = output / "request.json"
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError("Output belongs to a different experiment; choose a new directory")
        _write_json(request_path, request)
        runs = [
            _run_model(name, seed, graph, splits, config, output, identity)
            for seed in config.seeds
            for name in MODELS
        ]
        report = {
            "version": 1,
            "experiment_sha256": identity,
            "request": request,
            "test_evaluated": config.final_test,
            "runs": [asdict(run) for run in runs],
            "validation_comparisons": _comparisons(runs, config),
            "scope": "Supervised screening only; not an Elo or topology-benefit claim",
            "memory_measurement": "Peak traced allocations during training; not process RSS",
            "timing_measurement": (
                "Offline NumPy timings with allocation tracing during training; not browser timings"
            ),
        }
        _write_json(output / "report.json", report)
        return report
