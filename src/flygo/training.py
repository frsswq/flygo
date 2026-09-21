"""Offline policy-value training around the frozen connectome."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flygo.connectome import FrozenConnectome
from flygo.go import BOARD_SIZES
from flygo.model import (
    ConnectomePolicy,
    DensePolicy,
    Policy,
    policy_parameters,
    validate_dynamics,
)

CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class TrainingData:
    features: NDArray[np.float32]
    legal: NDArray[np.bool_]
    policy: NDArray[np.float32]
    value: NDArray[np.float32]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    policy_loss: float
    value_loss: float
    validation_policy_loss: float | None
    validation_value_loss: float | None


def graph_sha256(connectome: FrozenConnectome) -> str:
    digest = hashlib.sha256()
    for array in (
        connectome.node_ids,
        connectome.source_indices,
        connectome.target_indices,
        connectome.weights,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_training_data(path: Path, *, size: int) -> TrainingData:
    """Load and validate a dataset split at the training boundary."""
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        legal = archive["legal"].astype(np.bool_)
        policy = archive["policy"].astype(np.float32)
        value = archive["value"].astype(np.float32)
    if features.ndim != 2:
        raise ValueError("Features must be a matrix")
    rows = features.shape[0]
    feature_count = 2 * size * size + 1
    action_count = size * size + 1
    if features.shape != (rows, feature_count):
        raise ValueError(f"Features must have shape (rows, {feature_count})")
    if legal.shape != (rows, action_count) or policy.shape != (rows, action_count):
        raise ValueError(f"Policy data must have shape (rows, {action_count})")
    if value.shape != (rows,):
        raise ValueError("Values must have one scalar per row")
    if not (
        np.all(np.isfinite(features)) and np.all(np.isfinite(policy)) and np.all(np.isfinite(value))
    ):
        raise ValueError("Training data must be finite")
    if np.any(policy < 0) or np.any(policy[~legal] != 0):
        raise ValueError("Policy targets must be nonnegative and legal")
    if rows and not np.allclose(policy.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Every policy target must sum to one")
    if np.any(np.abs(value) > 1):
        raise ValueError("Value targets must be in [-1, 1]")
    return TrainingData(features, legal, policy, value)


def _transform(matrix: NDArray[Any], symmetry: int) -> NDArray[Any]:
    transformed = np.rot90(matrix, symmetry % 4)
    return np.fliplr(transformed) if symmetry >= 4 else transformed


def apply_symmetry(
    features: NDArray[np.float32],
    legal: NDArray[np.bool_],
    policy: NDArray[np.float32],
    *,
    size: int,
    symmetry: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
    """Apply one of the eight square-board symmetries to one example."""
    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be between 0 and 7")
    points = size * size
    transformed_features = features.copy()
    for plane in range(2):
        start = plane * points
        transformed_features[start : start + points] = _transform(
            features[start : start + points].reshape(size, size), symmetry
        ).reshape(-1)
    transformed_legal = legal.copy()
    transformed_legal[:points] = _transform(legal[:points].reshape(size, size), symmetry).reshape(
        -1
    )
    transformed_policy = policy.copy()
    transformed_policy[:points] = _transform(policy[:points].reshape(size, size), symmetry).reshape(
        -1
    )
    return transformed_features, transformed_legal, transformed_policy


def _augment_batch(
    data: TrainingData,
    indices: NDArray[np.int64],
    *,
    size: int,
    generator: np.random.Generator,
) -> TrainingData:
    features = data.features[indices].copy()
    legal = data.legal[indices].copy()
    policy = data.policy[indices].copy()
    for row, symmetry in enumerate(generator.integers(0, 8, size=indices.size)):
        features[row], legal[row], policy[row] = apply_symmetry(
            features[row], legal[row], policy[row], size=size, symmetry=int(symmetry)
        )
    return TrainingData(features, legal, policy, data.value[indices])


def _recurrent_coefficients(connectome: FrozenConnectome) -> NDArray[np.float32]:
    return connectome.weights / connectome.incoming_strength[connectome.target_indices]


def _forward(
    policy: Policy,
    features: NDArray[np.float32],
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]], NDArray[np.float32]]:
    if isinstance(policy, DensePolicy):
        activity = features if policy.encoder is None else np.tanh(features @ policy.encoder.T)
        return activity, [activity], np.empty(0, dtype=np.float32)
    external = np.tanh(features @ policy.encoder.T).astype(np.float32)
    states = [np.zeros((features.shape[0], policy.connectome.node_count), dtype=np.float32)]
    coefficients = _recurrent_coefficients(policy.connectome)
    for _ in range(policy.steps):
        drive = np.zeros_like(states[-1])
        np.add.at(
            drive.T,
            policy.connectome.target_indices,
            (states[-1][:, policy.connectome.source_indices] * coefficients).T,
        )
        states.append(
            np.tanh(
                policy.retention * states[-1] + policy.recurrent_gain * drive + external
            ).astype(np.float32)
        )
    return external, states, coefficients


def loss_and_gradients(
    model: Policy,
    batch: TrainingData,
    *,
    value_weight: float,
) -> tuple[float, float, tuple[NDArray[np.float32], ...]]:
    """Return mean losses and analytic gradients for numerical verification and training."""
    external, states, coefficients = _forward(model, batch.features)
    activity = states[-1]
    logits = activity @ model.readout.T
    masked = np.where(batch.legal, logits, -1e9)
    shifted = masked - masked.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    epsilon = np.finfo(np.float32).tiny
    policy_loss = float(-np.sum(batch.policy * np.log(np.maximum(probabilities, epsilon))))
    policy_loss /= batch.features.shape[0]

    values = np.tanh(activity @ model.value_readout)
    errors = values - batch.value
    value_loss = float(np.mean(errors * errors))
    rows = batch.features.shape[0]
    logits_gradient = (probabilities - batch.policy) / rows
    readout_gradient = logits_gradient.T @ activity
    state_gradient = logits_gradient @ model.readout
    value_gradient = (2 * value_weight / rows) * errors * (1 - values * values)
    value_readout_gradient = value_gradient @ activity
    state_gradient += value_gradient[:, None] * model.value_readout[None, :]

    if isinstance(model, DensePolicy):
        heads = (readout_gradient, value_readout_gradient)
        gradients = (
            heads
            if model.encoder is None
            else (((state_gradient * (1 - activity * activity)).T @ batch.features), *heads)
        )
        return policy_loss, value_loss, gradients

    external_gradient = np.zeros_like(external)
    for step in range(model.steps - 1, -1, -1):
        activation_gradient = state_gradient * (1 - states[step + 1] * states[step + 1])
        external_gradient += activation_gradient
        previous_gradient = model.retention * activation_gradient
        np.add.at(
            previous_gradient.T,
            model.connectome.source_indices,
            (
                model.recurrent_gain
                * activation_gradient[:, model.connectome.target_indices]
                * coefficients
            ).T,
        )
        state_gradient = previous_gradient
    encoder_gradient = (external_gradient * (1 - external * external)).T @ batch.features
    return (
        policy_loss,
        value_loss,
        (
            encoder_gradient.astype(np.float32),
            readout_gradient.astype(np.float32),
            value_readout_gradient.astype(np.float32),
        ),
    )


@dataclass(frozen=True)
class EvaluationMetrics:
    examples: int
    policy_loss: float
    value_mse: float
    target_top1: float
    target_top3: float
    legal_action_rate: float


def evaluate_policy(
    model: Policy, data: TrainingData, *, batch_size: int = 128
) -> EvaluationMetrics:
    """Evaluate in bounded batches with no backward pass and stable action-index tie breaks."""
    rows = data.features.shape[0]
    if rows == 0 or batch_size < 1:
        raise ValueError("Evaluation needs nonempty data and a positive batch size")
    totals = np.zeros(5, dtype=np.float64)
    for start in range(0, rows, batch_size):
        selected = slice(start, start + batch_size)
        _, states, _ = _forward(model, data.features[selected])
        activity = states[-1]
        logits = activity @ model.readout.T
        values = np.tanh(activity @ model.value_readout)
        legal = data.legal[selected]
        targets = data.policy[selected]
        masked = np.where(legal, logits, -1e9)
        shifted = masked - masked.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        ranked = np.argsort(-masked, axis=1, kind="stable")[:, :3]
        expected = targets.argmax(axis=1)
        row_indices = np.arange(logits.shape[0])
        totals += [
            -np.sum(targets * log_probabilities, dtype=np.float64),
            np.sum((values - data.value[selected]) ** 2, dtype=np.float64),
            np.sum(ranked[:, 0] == expected),
            np.sum(np.any(ranked == expected[:, None], axis=1)),
            np.sum(legal[row_indices, logits.argmax(axis=1)]),
        ]
    return EvaluationMetrics(rows, *(float(value / rows) for value in totals))


def evaluate_loss(model: Policy, data: TrainingData) -> tuple[float, float]:
    metrics = evaluate_policy(model, data)
    return metrics.policy_loss, metrics.value_mse


def selected_epoch(history: list[EpochMetrics], *, value_weight: float = 1.0) -> int:
    """Return the selected validation epoch, or the last epoch without validation."""
    scored = [
        (epoch.validation_policy_loss + value_weight * epoch.validation_value_loss, epoch.epoch)
        for epoch in history
        if epoch.validation_policy_loss is not None and epoch.validation_value_loss is not None
    ]
    return min(scored)[1] if scored else history[-1].epoch


def fit_policy(
    model: Policy,
    train: TrainingData,
    *,
    validation: TrainingData | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    value_weight: float = 1.0,
    seed: int = 7,
) -> list[EpochMetrics]:
    """Fit in place with shared Adam and augmentation; restore the best validation epoch."""
    if train.features.shape[0] == 0:
        raise ValueError("Training data is empty")
    if (
        epochs < 1
        or batch_size < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not np.isfinite(value_weight)
        or value_weight < 0
    ):
        raise ValueError("Training hyperparameters must be positive")
    if validation is not None and validation.features.shape[0] == 0:
        raise ValueError("Validation data is empty")
    parameters = policy_parameters(model)
    best_parameters = [parameter.copy() for parameter in parameters]
    best_loss = float("inf")
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    generator = np.random.default_rng(seed)
    update = 0
    history: list[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        permutation = generator.permutation(train.features.shape[0])
        policy_total = 0.0
        value_total = 0.0
        for start in range(0, permutation.size, batch_size):
            indices = permutation[start : start + batch_size]
            batch = _augment_batch(train, indices, size=model.size, generator=generator)
            policy_loss, value_loss, gradients = loss_and_gradients(
                model, batch, value_weight=value_weight
            )
            update += 1
            for parameter, gradient, first, second in zip(
                parameters, gradients, first_moments, second_moments, strict=True
            ):
                np.clip(gradient, -5, 5, out=gradient)
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient * gradient
                corrected_first = first / (1 - 0.9**update)
                corrected_second = second / (1 - 0.999**update)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            policy_total += policy_loss * indices.size
            value_total += value_loss * indices.size
        validation_losses = None
        if validation is not None:
            validation_losses = evaluate_loss(model, validation)
            combined = validation_losses[0] + value_weight * validation_losses[1]
            if combined < best_loss:
                best_loss = combined
                best_parameters = [parameter.copy() for parameter in parameters]
        history.append(
            EpochMetrics(
                epoch,
                policy_total / train.features.shape[0],
                value_total / train.features.shape[0],
                validation_losses[0] if validation_losses else None,
                validation_losses[1] if validation_losses else None,
            )
        )
    if validation is not None:
        for parameter, best in zip(parameters, best_parameters, strict=True):
            parameter[:] = best
    return history


def save_policy(
    path: Path,
    policy: Policy,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a portable checkpoint atomically, binding connectome models to their graph."""
    architecture = (
        {
            "kind": "connectome",
            "steps": policy.steps,
            "retention": policy.retention,
            "recurrent_gain": policy.recurrent_gain,
            "graph_sha256": graph_sha256(policy.connectome),
        }
        if isinstance(policy, ConnectomePolicy)
        else {"kind": "linear" if policy.encoder is None else "mlp"}
    )
    payload = {
        **(metadata or {}),
        "version": CHECKPOINT_VERSION,
        "size": policy.size,
        **architecture,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            np.savez_compressed(
                temporary,
                encoder=(
                    np.empty((0, 0), dtype=np.float32) if policy.encoder is None else policy.encoder
                ),
                readout=policy.readout,
                value_readout=policy.value_readout,
                metadata=np.asarray(json.dumps(payload)),
            )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_policy(
    path: Path, connectome: FrozenConnectome | None = None
) -> tuple[Policy, dict[str, Any]]:
    """Load a checkpoint only when its graph and array shapes match."""
    with np.load(path, allow_pickle=False) as archive:
        encoder = archive["encoder"].astype(np.float32)
        readout = archive["readout"].astype(np.float32)
        value_readout = archive["value_readout"].astype(np.float32)
        metadata = json.loads(str(archive["metadata"]))
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported policy checkpoint version")
    size = metadata.get("size")
    if type(size) is not int or size not in BOARD_SIZES:
        raise ValueError("Unsupported checkpoint board size")
    kind = metadata.get("kind", "connectome")
    if kind not in {"connectome", "linear", "mlp"}:
        raise ValueError("Unsupported checkpoint model kind")
    if not all(np.all(np.isfinite(array)) for array in (encoder, readout, value_readout)):
        raise ValueError("Checkpoint weights must be finite")
    feature_count = 2 * size * size + 1
    action_count = size * size + 1
    if kind == "connectome":
        if connectome is None or metadata.get("graph_sha256") != graph_sha256(connectome):
            raise ValueError("Policy checkpoint was trained on a different graph")
        width = connectome.node_count
    else:
        width = feature_count if kind == "linear" else encoder.shape[0]
    expected_encoder_shape = (0, 0) if kind == "linear" else (width, feature_count)
    if width < 1 or encoder.shape != expected_encoder_shape:
        raise ValueError("Checkpoint encoder shape is invalid")
    if readout.shape != (action_count, width):
        raise ValueError("Checkpoint policy readout shape is invalid")
    if value_readout.shape != (width,):
        raise ValueError("Checkpoint value readout shape is invalid")
    if kind != "connectome":
        return DensePolicy(
            None if kind == "linear" else encoder, readout, value_readout, size
        ), metadata
    if connectome is None:
        raise ValueError("Connectome checkpoint needs its frozen graph")
    steps = metadata.get("steps")
    retention = metadata.get("retention")
    recurrent_gain = metadata.get("recurrent_gain")
    dynamics = (retention, recurrent_gain)
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in dynamics):
        raise ValueError("Checkpoint must record numeric recurrent dynamics")
    validate_dynamics(steps=steps, retention=retention, recurrent_gain=recurrent_gain)
    return ConnectomePolicy(
        connectome,
        encoder,
        readout,
        value_readout,
        size,
        steps,
        retention,
        recurrent_gain,
    ), metadata
