"""A lightweight simulation layer over frozen MaleCNS wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FrozenConnectome:
    """A compact directed graph whose biological edge weights cannot be trained."""

    node_ids: IntArray
    source_indices: IntArray
    target_indices: IntArray
    weights: FloatArray
    incoming_strength: FloatArray

    @property
    def node_count(self) -> int:
        return self.node_ids.size

    @property
    def edge_count(self) -> int:
        return self.weights.size

    def step(
        self,
        state: FloatArray,
        external_input: FloatArray | None = None,
        *,
        retention: float = 0.35,
        recurrent_gain: float = 0.9,
    ) -> FloatArray:
        """Advance rate-coded activity by one documented modeling step."""
        if state.shape != (self.node_count,):
            raise ValueError(f"Expected state shape {(self.node_count,)}, got {state.shape}")
        drive = self.weights * state[self.source_indices]
        recurrent = np.bincount(
            self.target_indices,
            weights=drive,
            minlength=self.node_count,
        ).astype(np.float32, copy=False)
        recurrent /= self.incoming_strength
        total = retention * state + recurrent_gain * recurrent
        if external_input is not None:
            if external_input.shape != state.shape:
                raise ValueError("External input and state must have equal shapes")
            total = total + external_input
        return np.tanh(total).astype(np.float32, copy=False)

    def run(self, external_input: FloatArray, *, steps: int = 8) -> FloatArray:
        if steps < 1:
            raise ValueError("Simulation steps must be positive")
        state = np.zeros(self.node_count, dtype=np.float32)
        for _ in range(steps):
            state = self.step(state, external_input)
        return state

    def randomized(self, *, seed: int) -> FrozenConnectome:
        """Make a directed degree-preserving target rewire for a control experiment."""
        generator = np.random.default_rng(seed)
        targets = generator.permutation(self.target_indices)
        return _freeze(
            self.node_ids.copy(), self.source_indices.copy(), targets, self.weights.copy()
        )


def _freeze(
    node_ids: IntArray, sources: IntArray, targets: IntArray, weights: FloatArray
) -> FrozenConnectome:
    incoming = np.bincount(targets, weights=weights, minlength=node_ids.size).astype(
        np.float32, copy=False
    )
    incoming[incoming == 0] = 1
    arrays = (node_ids, sources, targets, weights, incoming)
    for array in arrays:
        array.flags.writeable = False
    return FrozenConnectome(*arrays)


def from_frame(edges: pl.DataFrame) -> FrozenConnectome:
    """Build a frozen graph from normalized pre, post, and weight columns."""
    required = {"pre", "post", "weight"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing graph columns: {', '.join(sorted(missing))}")
    if edges.is_empty():
        raise ValueError("The graph has no edges")

    nodes = (
        pl.concat(
            [edges.select(pl.col("pre").alias("id")), edges.select(pl.col("post").alias("id"))]
        )
        .unique()
        .sort("id")
    )
    lookup = nodes.with_row_index("index")
    indexed = edges.join(lookup, left_on="pre", right_on="id").rename({"index": "source"})
    indexed = indexed.join(lookup, left_on="post", right_on="id").rename({"index": "target"})
    return _freeze(
        nodes["id"].to_numpy().astype(np.int64, copy=False),
        indexed["source"].to_numpy().astype(np.int64, copy=False),
        indexed["target"].to_numpy().astype(np.int64, copy=False),
        indexed["weight"].to_numpy().astype(np.float32, copy=False),
    )


def load_connectome(path: Path) -> FrozenConnectome:
    return from_frame(pl.read_parquet(path))
