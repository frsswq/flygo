"""Deterministic circuit selection for frozen-connectome experiments.

One fixed rule selects every circuit, so a circuit size cannot be chosen from
validation or test results.
The rule depends only on the prepared graph.

The rule grows a connected circuit from the strongest neuron.
It keeps the biological wiring inside one component, so every selected neuron
has at least one connection to the rest of the circuit.
"""

from __future__ import annotations

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


def _adjacency(
    edges: pl.DataFrame,
) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """Return incident strength and summed neighbour weight for every node."""
    strength: dict[int, float] = {}
    neighbours: dict[int, dict[int, float]] = {}
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


def seed_node(edges: pl.DataFrame) -> int:
    """Return the strongest neuron, the one with the highest incident weight."""
    validate_edges(edges)
    strength, _ = _adjacency(edges)
    return min(strength, key=lambda node: (-strength[node], node))


def connected_core_size(edges: pl.DataFrame) -> int:
    """Return the neuron count of the connected component that holds the seed."""
    validate_edges(edges)
    _, neighbours = _adjacency(edges)
    start = seed_node(edges)
    seen = {start}
    pending = [start]
    while pending:
        node = pending.pop()
        for neighbour in neighbours[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                pending.append(neighbour)
    return len(seen)


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
    validate_edges(edges)
    strength, neighbours = _adjacency(edges)
    remaining = set(strength)
    scores: dict[int, float] = {}
    seed = seed_node(edges)
    order = [seed]
    remaining.discard(seed)
    for neighbour, weight in neighbours[seed].items():
        if neighbour in remaining:
            scores[neighbour] = weight
    while remaining:
        if scores:
            node = min(scores, key=lambda node: (-scores[node], -strength[node], node))
        else:
            node = min(remaining, key=lambda node: (-strength[node], node))
        order.append(node)
        remaining.discard(node)
        scores.pop(node, None)
        for neighbour, weight in neighbours[node].items():
            if neighbour in remaining:
                scores[neighbour] = scores.get(neighbour, 0.0) + weight
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


def select_circuit(edges: pl.DataFrame, *, node_count: int) -> tuple[pl.DataFrame, SelectionReport]:
    """Return one connected circuit of ``node_count`` neurons and its diagnostics."""
    if node_count < 2:
        raise ValueError("A circuit needs at least two nodes")
    core = connected_core_size(edges)
    if node_count > core:
        raise ValueError(
            f"The connected core of {seed_node(edges)} holds {core} nodes, not {node_count}"
        )
    kept = set(growth_order(edges)[:node_count])
    members = sorted(kept)
    selected = edges.filter(pl.col("pre").is_in(members) & pl.col("post").is_in(members))
    realised = set(selected["pre"].to_list()) | set(selected["post"].to_list())
    if realised != kept:
        raise ValueError("The selected node set is not connected")
    sources = set(selected["pre"].to_list())
    report = SelectionReport(
        requested_nodes=node_count,
        node_count=len(kept),
        edge_count=selected.height,
        seed_node=seed_node(edges),
        largest_component=largest_component(selected),
        nodes_without_outgoing=sum(1 for node in kept if node not in sources),
    )
    return selected, report


def write_circuits(
    graph: Path,
    output: Path,
    *,
    node_counts: Sequence[int] = DEFAULT_NODE_COUNTS,
) -> dict[str, Any]:
    """Write one circuit file per requested size and a selection manifest."""
    if not node_counts:
        raise ValueError("At least one circuit size is required")
    edges = pl.read_parquet(graph)
    validate_edges(edges)
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for node_count in sorted(set(node_counts)):
        selected, report = select_circuit(edges, node_count=node_count)
        path = output / f"circuit-{node_count}.parquet"
        selected.write_parquet(path)
        entries.append({"file": path.name, "sha256": file_sha256(path)} | asdict(report))
    manifest: dict[str, Any] = {
        "rule": RULE,
        "rule_version": RULE_VERSION,
        "source": {"file": graph.name, "sha256": file_sha256(graph)},
        "circuits": entries,
    }
    (output / "selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
