"""Deterministic circuit selection for frozen-connectome experiments.

One fixed rule selects every circuit, so a circuit size cannot be chosen from
validation or test results.
The rule depends only on the prepared graph.

The rule grows a connected circuit from the strongest neuron.
It keeps the biological wiring inside one component, so every selected neuron
has at least one connection to the rest of the circuit.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from flygo.official_data import file_sha256

RULE = "strongest-first connected growth"
RULE_VERSION = 1
DEFAULT_NODE_COUNTS = (250, 500, 1000)

type Strength = dict[int, float]
type Neighbours = dict[int, dict[int, float]]


@dataclass(frozen=True)
class SelectionReport:
    """Diagnostics for one selected circuit."""

    requested_nodes: int
    node_count: int
    edge_count: int
    seed_node: int
    largest_component: int
    nodes_without_outgoing: int


def validate_edges(edges: pl.DataFrame) -> None:
    """Reject a frame that cannot describe a directed weighted graph."""
    missing = {"pre", "post", "weight"}.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing graph columns: {', '.join(sorted(missing))}")
    for column in ("pre", "post", "weight"):
        if not edges.schema[column].is_numeric():
            raise ValueError(f"Graph column {column} must be numeric")
    if edges.is_empty():
        raise ValueError("The graph has no edges")
    if edges.select(pl.col("pre").is_null().any() | pl.col("post").is_null().any()).item():
        raise ValueError("Graph endpoints must not be null")
    if not edges.select((pl.col("weight").is_finite() & (pl.col("weight") > 0)).all()).item():
        raise ValueError("Graph weights must be finite and positive")


def _adjacency(edges: pl.DataFrame) -> tuple[Strength, Neighbours]:
    """Return incident strength and summed neighbour weight for every node."""
    strength: Strength = {}
    neighbours: Neighbours = {}
    sources = edges["pre"].to_list()
    targets = edges["post"].to_list()
    weights: list[float] = edges["weight"].cast(pl.Float64).to_list()
    for source, target, value in zip(sources, targets, weights, strict=True):
        strength[source] = strength.get(source, 0.0) + value
        strength[target] = strength.get(target, 0.0) + value
        outgoing = neighbours.setdefault(source, {})
        outgoing[target] = outgoing.get(target, 0.0) + value
        incoming = neighbours.setdefault(target, {})
        incoming[source] = incoming.get(source, 0.0) + value
    return strength, neighbours


def prepare(edges: pl.DataFrame) -> tuple[Strength, Neighbours]:
    """Validate the frame and measure it once, for repeated selection."""
    validate_edges(edges)
    return _adjacency(edges)


def _strongest(strength: Strength) -> int:
    return min(strength, key=lambda node: (-strength[node], node))


def grow(
    strength: Strength,
    neighbours: Neighbours,
    *,
    limit: int | None = None,
    allow_fallback: bool = True,
) -> tuple[list[int], int | None]:
    """Grow the selection order and report where the first component ends.

    Return the first ``limit`` nodes of the growth order, or every node when
    ``limit`` is ``None``, together with the connected core size.
    With ``allow_fallback`` the walk restarts from the strongest unselected
    neuron after a component ends, so the order covers every node.
    Without it the walk stops at the component edge, which reports the
    connected core size and keeps a circuit inside one component.
    The core size is ``None`` when the limit stops the walk inside one
    component, because the full core was never reached.
    """
    seed = _strongest(strength)
    remaining = set(strength)
    remaining.discard(seed)
    fallback: list[tuple[float, int]] = []
    if allow_fallback:
        fallback = [(-strength[node], node) for node in remaining]
        heapq.heapify(fallback)
    scores: dict[int, float] = {}
    frontier: list[tuple[float, float, int]] = []

    def push(node: int, score: float) -> None:
        scores[node] = score
        heapq.heappush(frontier, (-score, -strength[node], node))

    for neighbour, weight in neighbours[seed].items():
        if neighbour in remaining:
            push(neighbour, weight)

    order = [seed]
    core: int | None = None
    while remaining and (limit is None or len(order) < limit):
        node: int | None = None
        while frontier:
            negative_score, _, candidate = frontier[0]
            if candidate not in remaining or scores.get(candidate) != -negative_score:
                heapq.heappop(frontier)
                continue
            node = candidate
            heapq.heappop(frontier)
            break
        if node is None:
            if core is None:
                core = len(order)
            if not allow_fallback:
                break
            while fallback:
                candidate = fallback[0][1]
                if candidate in remaining:
                    node = candidate
                    break
                heapq.heappop(fallback)
            if node is None:
                break
        order.append(node)
        remaining.discard(node)
        scores.pop(node, None)
        for neighbour, weight in neighbours[node].items():
            if neighbour in remaining:
                push(neighbour, scores.get(neighbour, 0.0) + weight)
    if core is None and not remaining:
        core = len(order)
    return order, core


def seed_node(edges: pl.DataFrame) -> int:
    """Return the strongest neuron, the one with the highest incident weight."""
    strength, _ = prepare(edges)
    return _strongest(strength)


def connected_core_size(edges: pl.DataFrame) -> int:
    """Return the neuron count of the connected component that holds the seed."""
    strength, neighbours = prepare(edges)
    _, core = grow(strength, neighbours, allow_fallback=False)
    if core is None:
        raise ValueError("The connected core size is unavailable")
    return core


def growth_order(edges: pl.DataFrame) -> list[int]:
    """Return every node in the order the selection rule adds it.

    Start from the strongest neuron, then repeatedly add the unselected neuron
    with the strongest connection to the current set.
    Break ties by incident strength and then by the smallest body ID.
    When a component is exhausted, restart from the strongest remaining
    neuron, so the order also covers disconnected components.
    The order does not depend on a target size, so larger circuits extend
    smaller ones.
    """
    strength, neighbours = prepare(edges)
    order, _ = grow(strength, neighbours)
    return order


def largest_component(edges: pl.DataFrame) -> int:
    """Return the node count of the largest strongly connected component."""
    if edges.is_empty():
        return 0
    sources = edges["pre"].to_list()
    targets = edges["post"].to_list()
    nodes = sorted(set(sources) | set(targets))
    index = {node: position for position, node in enumerate(nodes)}
    count = len(nodes)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    reverse: list[list[int]] = [[] for _ in range(count)]
    for source, target in zip(sources, targets, strict=True):
        adjacency[index[source]].append(index[target])
        reverse[index[target]].append(index[source])

    seen = [False] * count
    finish: list[int] = []
    for root in range(count):
        if seen[root]:
            continue
        seen[root] = True
        stack = [(root, 0)]
        while stack:
            node, position = stack[-1]
            if position < len(adjacency[node]):
                neighbour = adjacency[node][position]
                stack[-1] = (node, position + 1)
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append((neighbour, 0))
            else:
                finish.append(node)
                stack.pop()

    seen = [False] * count
    largest = 0
    for node in reversed(finish):
        if seen[node]:
            continue
        seen[node] = True
        size = 0
        pending = [node]
        while pending:
            current = pending.pop()
            size += 1
            for neighbour in reverse[current]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    pending.append(neighbour)
        largest = max(largest, size)
    return largest


def _circuit_frame(edges: pl.DataFrame, kept: set[int]) -> pl.DataFrame:
    members = sorted(kept)
    return edges.filter(pl.col("pre").is_in(members) & pl.col("post").is_in(members))


def _report(
    selected: pl.DataFrame,
    kept: set[int],
    *,
    requested_nodes: int,
    seed: int,
) -> SelectionReport:
    sources = set(selected["pre"].to_list())
    return SelectionReport(
        requested_nodes=requested_nodes,
        node_count=len(kept),
        edge_count=selected.height,
        seed_node=seed,
        largest_component=largest_component(selected),
        nodes_without_outgoing=sum(1 for node in kept if node not in sources),
    )


def select_circuit(edges: pl.DataFrame, *, node_count: int) -> tuple[pl.DataFrame, SelectionReport]:
    """Return one connected circuit of ``node_count`` neurons and its diagnostics."""
    if node_count < 2:
        raise ValueError("A circuit needs at least two nodes")
    strength, neighbours = prepare(edges)
    seed = _strongest(strength)
    order, core = grow(strength, neighbours, limit=node_count, allow_fallback=False)
    if len(order) < node_count:
        raise ValueError(f"The connected core of {seed} holds {core} nodes, not {node_count}")
    kept = set(order)
    selected = _circuit_frame(edges, kept)
    realised = set(selected["pre"].to_list()) | set(selected["post"].to_list())
    if realised != kept:
        raise ValueError("The selected node set is not connected")
    return selected, _report(
        selected,
        kept,
        requested_nodes=node_count,
        seed=seed,
    )


def write_circuits(
    graph: Path,
    output: Path,
    *,
    node_counts: Sequence[int] = DEFAULT_NODE_COUNTS,
) -> dict[str, Any]:
    """Write one circuit file per requested size and a selection manifest."""
    if not node_counts:
        raise ValueError("At least one circuit size is required")
    if min(node_counts) < 2:
        raise ValueError("A circuit needs at least two nodes")
    edges = pl.read_parquet(graph)
    strength, neighbours = prepare(edges)
    seed = _strongest(strength)
    sizes = sorted(set(node_counts))
    order, core = grow(strength, neighbours, limit=max(sizes), allow_fallback=False)
    if len(order) < max(sizes):
        raise ValueError(f"The connected core of {seed} holds {core} nodes, not {max(sizes)}")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for node_count in sizes:
        kept = set(order[:node_count])
        selected = _circuit_frame(edges, kept)
        path = output / f"circuit-{node_count}.parquet"
        selected.write_parquet(path)
        report = _report(selected, kept, requested_nodes=node_count, seed=seed)
        entries.append({"file": path.name, "sha256": file_sha256(path)} | asdict(report))
    manifest: dict[str, Any] = {
        "rule": RULE,
        "rule_version": RULE_VERSION,
        "source": {"file": graph.name, "sha256": file_sha256(graph)},
        "circuits": entries,
    }
    (output / "selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
