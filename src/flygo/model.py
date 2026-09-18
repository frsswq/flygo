"""Trainable boundaries around a frozen connectome."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from flygo.connectome import FrozenConnectome
from flygo.go import DEFAULT_BOARD_SIZE, Position


@dataclass
class ConnectomePolicy:
    """A small encoder and readout around immutable recurrent wiring."""

    connectome: FrozenConnectome
    encoder: NDArray[np.float32]
    readout: NDArray[np.float32]
    value_readout: NDArray[np.float32]
    size: int = DEFAULT_BOARD_SIZE

    @classmethod
    def initialize(
        cls,
        connectome: FrozenConnectome,
        *,
        size: int = DEFAULT_BOARD_SIZE,
        seed: int = 7,
    ) -> ConnectomePolicy:
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
        )

    @property
    def action_count(self) -> int:
        return self.size * self.size + 1

    def activity(self, position: Position, *, steps: int = 8) -> NDArray[np.float32]:
        if position.size != self.size:
            raise ValueError(f"This policy plays {self.size}x{self.size}, not {position.size}")
        external_input = np.tanh(self.encoder @ position.features()).astype(np.float32)
        return self.connectome.run(external_input, steps=steps)

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
