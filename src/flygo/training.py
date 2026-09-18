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
from flygo.model import ConnectomePolicy

CHECKPOINT_VERSION = 1


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
    policy: ConnectomePolicy,
    features: NDArray[np.float32],
    *,
    steps: int,
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]], NDArray[np.float32]]:
    external = np.tanh(features @ policy.encoder.T).astype(np.float32)
    states = [np.zeros((features.shape[0], policy.connectome.node_count), dtype=np.float32)]
    coefficients = _recurrent_coefficients(policy.connectome)
    for _ in range(steps):
        drive = np.zeros_like(states[-1])
        np.add.at(
            drive.T,
            policy.connectome.target_indices,
            (states[-1][:, policy.connectome.source_indices] * coefficients).T,
        )
        states.append(np.tanh(0.35 * states[-1] + 0.9 * drive + external).astype(np.float32))
    return external, states, coefficients


def _loss_and_gradients(
    model: ConnectomePolicy,
    batch: TrainingData,
    *,
    steps: int,
    value_weight: float,
) -> tuple[float, float, tuple[NDArray[np.float32], ...]]:
    external, states, coefficients = _forward(model, batch.features, steps=steps)
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

    external_gradient = np.zeros_like(external)
    for step in range(steps - 1, -1, -1):
        activation_gradient = state_gradient * (1 - states[step + 1] * states[step + 1])
        external_gradient += activation_gradient
        previous_gradient = 0.35 * activation_gradient
        np.add.at(
            previous_gradient.T,
            model.connectome.source_indices,
            (0.9 * activation_gradient[:, model.connectome.target_indices] * coefficients).T,
        )
        state_gradient = previous_gradient
    encoder_gradient = ((external_gradient * (1 - external * external)).T @ batch.features) / rows
    return (
        policy_loss,
        value_loss,
        (
            encoder_gradient.astype(np.float32),
            readout_gradient.astype(np.float32),
            value_readout_gradient.astype(np.float32),
        ),
    )


def evaluate_loss(
    model: ConnectomePolicy,
    data: TrainingData,
    *,
    steps: int = 8,
    value_weight: float = 1.0,
) -> tuple[float, float]:
    if data.features.shape[0] == 0:
        raise ValueError("Evaluation data is empty")
    policy_loss, value_loss, _ = _loss_and_gradients(
        model, data, steps=steps, value_weight=value_weight
    )
    return policy_loss, value_loss


def train_policy_value(
    connectome: FrozenConnectome,
    train: TrainingData,
    *,
    size: int = 19,
    validation: TrainingData | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    steps: int = 8,
    value_weight: float = 1.0,
    seed: int = 7,
) -> tuple[ConnectomePolicy, list[EpochMetrics]]:
    """Train the encoder and two heads with Adam and dihedral augmentation."""
    if train.features.shape[0] == 0:
        raise ValueError("Training data is empty")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or steps < 1:
        raise ValueError("Training hyperparameters must be positive")
    model = ConnectomePolicy.initialize(connectome, size=size, seed=seed)
    parameters = (model.encoder, model.readout, model.value_readout)
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
            batch = _augment_batch(train, indices, size=size, generator=generator)
            policy_loss, value_loss, gradients = _loss_and_gradients(
                model, batch, steps=steps, value_weight=value_weight
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
        if validation is not None and validation.features.shape[0]:
            validation_losses = evaluate_loss(
                model, validation, steps=steps, value_weight=value_weight
            )
        history.append(
            EpochMetrics(
                epoch,
                policy_total / train.features.shape[0],
                value_total / train.features.shape[0],
                validation_losses[0] if validation_losses else None,
                validation_losses[1] if validation_losses else None,
            )
        )
    return model, history


def save_policy(
    path: Path,
    policy: ConnectomePolicy,
    *,
    steps: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a portable, graph-bound policy checkpoint atomically."""
    payload = {
        "version": CHECKPOINT_VERSION,
        "size": policy.size,
        "steps": steps,
        "graph_sha256": graph_sha256(policy.connectome),
        **(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            np.savez_compressed(
                temporary,
                encoder=policy.encoder,
                readout=policy.readout,
                value_readout=policy.value_readout,
                metadata=np.asarray(json.dumps(payload)),
            )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_policy(
    path: Path, connectome: FrozenConnectome
) -> tuple[ConnectomePolicy, dict[str, Any]]:
    """Load a checkpoint only when its graph and array shapes match."""
    with np.load(path, allow_pickle=False) as archive:
        encoder = archive["encoder"].astype(np.float32)
        readout = archive["readout"].astype(np.float32)
        value_readout = archive["value_readout"].astype(np.float32)
        metadata = json.loads(str(archive["metadata"]))
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported policy checkpoint version")
    if metadata.get("graph_sha256") != graph_sha256(connectome):
        raise ValueError("Policy checkpoint was trained on a different graph")
    size = int(metadata["size"])
    feature_count = 2 * size * size + 1
    action_count = size * size + 1
    if encoder.shape != (connectome.node_count, feature_count):
        raise ValueError("Checkpoint encoder shape is invalid")
    if readout.shape != (action_count, connectome.node_count):
        raise ValueError("Checkpoint policy readout shape is invalid")
    if value_readout.shape != (connectome.node_count,):
        raise ValueError("Checkpoint value readout shape is invalid")
    return ConnectomePolicy(connectome, encoder, readout, value_readout, size), metadata
