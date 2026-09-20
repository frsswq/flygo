"""Trainable boundaries around a frozen connectome."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from flygo.connectome import FrozenConnectome
from flygo.go import DEFAULT_BOARD_SIZE, Position

DEFAULT_RETENTION = 0.35
DEFAULT_RECURRENT_GAIN = 0.9


def validate_dynamics(*, steps: int, retention: float, recurrent_gain: float) -> None:
    """Reject simulation settings that cannot produce a rate-coded state."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Simulation steps must be a positive integer")
    for name, value in (("retention", retention), ("recurrent_gain", recurrent_gain)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class ConnectomePolicy:
    """A small encoder and readout around immutable recurrent wiring."""

    connectome: FrozenConnectome
    encoder: NDArray[np.float32]
    readout: NDArray[np.float32]
    value_readout: NDArray[np.float32]
    size: int = DEFAULT_BOARD_SIZE
    steps: int = 8
    retention: float = DEFAULT_RETENTION
    recurrent_gain: float = DEFAULT_RECURRENT_GAIN

    def __post_init__(self) -> None:
        validate_dynamics(
            steps=self.steps,
            retention=self.retention,
            recurrent_gain=self.recurrent_gain,
        )

    @classmethod
    def initialize(
        cls,
        connectome: FrozenConnectome,
        *,
        size: int = DEFAULT_BOARD_SIZE,
        seed: int = 7,
        steps: int = 8,
        retention: float = DEFAULT_RETENTION,
        recurrent_gain: float = DEFAULT_RECURRENT_GAIN,
    ) -> ConnectomePolicy:
        validate_dynamics(steps=steps, retention=retention, recurrent_gain=recurrent_gain)
        generator = np.random.default_rng(seed)
        feature_count = 2 * size * size + 1
        encoder = generator.normal(0, 0.15, (connectome.node_count, feature_count))
        readout = generator.normal(0, 0.05, (size * size + 1, connectome.node_count))
        value_readout = generator.normal(0, 0.05, connectome.node_count)
        return cls(
            connectome,
            encoder.astype(np.float32),
            readout.astype(np.float32),
            value_readout.astype(np.float32),
            size,
            steps,
            retention,
            recurrent_gain,
        )

    @property
    def action_count(self) -> int:
        return self.size * self.size + 1

    def activity(self, position: Position) -> NDArray[np.float32]:
        if position.size != self.size:
            raise ValueError(f"This policy plays {self.size}x{self.size}, not {position.size}")
        external_input = np.tanh(self.encoder @ position.features()).astype(np.float32)
        return self.connectome.run(
            external_input,
            steps=self.steps,
            retention=self.retention,
            recurrent_gain=self.recurrent_gain,
        )

    def logits(self, position: Position) -> NDArray[np.float32]:
        return self.readout @ self.activity(position)

    def evaluate(self, position: Position) -> tuple[NDArray[np.float32], float]:
        """Return policy logits and value from the current player's perspective."""
        activity = self.activity(position)
        logits = self.readout @ activity
        value = float(np.tanh(self.value_readout @ activity))
        return logits, value

    def choose_legal_action(self, position: Position) -> int:
        logits = self.logits(position)
        legal = position.legal_actions()
        return max(legal, key=lambda action: float(logits[action]))

    def fit_readout(
        self,
        activities: NDArray[np.float32],
        labels: NDArray[np.int64],
        *,
        regularization: float = 1.0,
    ) -> None:
        """Fit a deterministic ridge classifier while keeping the graph frozen."""
        if activities.ndim != 2 or activities.shape[1] != self.connectome.node_count:
            raise ValueError("Activities must have one column per connectome node")
        if labels.ndim != 1 or labels.shape[0] != activities.shape[0]:
            raise ValueError("Labels must provide one action for each activity row")
        if labels.size == 0:
            raise ValueError("At least one training example is required")
        if np.any(labels < 0) or np.any(labels >= self.action_count):
            raise ValueError(f"Labels must be between 0 and {self.action_count - 1}")
        targets = np.eye(self.action_count, dtype=np.float32)[labels]
        gram = activities @ activities.T
        system = gram + regularization * np.eye(gram.shape[0], dtype=np.float32)
        self.readout = (targets.T @ np.linalg.solve(system, activities)).astype(np.float32)


def linear_baseline_logits(
    features: NDArray[np.float32],
    weights: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Conventional baseline with no recurrent topology."""
    return weights @ features


def comparison_table(scores: dict[str, list[float]]) -> pl.DataFrame:
    """Return tidy per-seed results for MaleCNS, rewired, and ML baselines."""
    return pl.DataFrame(
        {
            "model": [model for model, values in scores.items() for _ in values],
            "seed": [seed for values in scores.values() for seed in range(len(values))],
            "accuracy": [score for values in scores.values() for score in values],
        }
    )


@dataclass
class DensePolicy:
    """A bias-free linear model or one-hidden-layer tanh MLP with the same targets.

    A missing encoder selects the linear model.
    Matching hidden width to the connectome node count exactly matches parameter count.
    """

    encoder: NDArray[np.float32] | None
    readout: NDArray[np.float32]
    value_readout: NDArray[np.float32]
    size: int

    @classmethod
    def initialize(
        cls, *, size: int = DEFAULT_BOARD_SIZE, hidden_count: int | None = None, seed: int = 7
    ) -> DensePolicy:
        if hidden_count is not None and hidden_count < 1:
            raise ValueError("Hidden width must be positive")
        generator = np.random.default_rng(seed)
        features = 2 * size * size + 1
        encoder = (
            None
            if hidden_count is None
            else generator.normal(0, 0.15, (hidden_count, features)).astype(np.float32)
        )
        width = features if hidden_count is None else hidden_count
        return cls(
            encoder,
            generator.normal(0, 0.05, (size * size + 1, width)).astype(np.float32),
            generator.normal(0, 0.05, width).astype(np.float32),
            size,
        )

    def evaluate(self, position: Position) -> tuple[NDArray[np.float32], float]:
        if position.size != self.size:
            raise ValueError("Policy and position board sizes differ")
        features = position.features()
        activity = features if self.encoder is None else np.tanh(self.encoder @ features)
        return self.readout @ activity, float(np.tanh(self.value_readout @ activity))


type Policy = ConnectomePolicy | DensePolicy


def policy_parameters(model: Policy) -> tuple[NDArray[np.float32], ...]:
    heads = (model.readout, model.value_readout)
    return heads if model.encoder is None else (model.encoder, *heads)
