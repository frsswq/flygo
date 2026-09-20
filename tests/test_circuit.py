"""Deterministic circuit selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from flygo.circuit import (
    RULE,
    RULE_VERSION,
    connected_core_size,
    growth_order,
    largest_component,
    seed_node,
    select_circuit,
    write_circuits,
)
from flygo.official_data import file_sha256


def graph_frame() -> pl.DataFrame:
    """One connected chain of four neurons with a clear strength ordering."""
    return pl.DataFrame(
        {
            "pre": [1, 2, 2, 3, 4],
            "post": [2, 1, 3, 4, 3],
            "weight": [5, 1, 1, 1, 1],
        }
    )


def split_frame() -> pl.DataFrame:
    """Two disconnected pairs."""
    return pl.DataFrame(
        {
            "pre": [1, 2, 3, 4],
            "post": [2, 1, 4, 3],
            "weight": [5, 1, 1, 1],
        }
    )


def test_growth_starts_from_the_strongest_neuron_and_is_deterministic() -> None:
    edges = graph_frame()

    assert seed_node(edges) == 2
    assert growth_order(edges) == [2, 1, 3, 4]
    assert growth_order(edges) == growth_order(edges)


def test_largest_circuits_extend_smaller_ones() -> None:
    edges = graph_frame()

    small, _ = select_circuit(edges, node_count=2)
    large, _ = select_circuit(edges, node_count=3)

    small_members = set(small["pre"].to_list()) | set(small["post"].to_list())
    large_members = set(large["pre"].to_list()) | set(large["post"].to_list())
    assert small_members <= large_members


def test_selection_keeps_only_internal_edges() -> None:
    edges = graph_frame()

    selected, report = select_circuit(edges, node_count=3)

    members = set(selected["pre"].to_list()) | set(selected["post"].to_list())
    assert members == {1, 2, 3}
    assert report.edge_count == selected.height == 3
    assert report.node_count == report.requested_nodes == 3
    assert report.seed_node == 2
    assert report.largest_component == 2
    assert report.nodes_without_outgoing == 1


def test_connected_core_stops_at_the_component_edge() -> None:
    edges = split_frame()

    assert connected_core_size(edges) == 2
    with pytest.raises(ValueError, match="connected core of 1 holds 2 nodes, not 3"):
        select_circuit(edges, node_count=3)
    selected, report = select_circuit(edges, node_count=2)
    assert report.edge_count == selected.height == 2


def test_largest_component_counts_the_strongest_cycle() -> None:
    cycle = pl.DataFrame(
        {
            "pre": [1, 2, 3, 4, 5],
            "post": [2, 1, 4, 5, 3],
            "weight": [1, 1, 1, 1, 1],
        }
    )

    assert largest_component(cycle) == 3
    assert largest_component(cycle.filter(pl.col("pre") == 1)) == 1


def test_selection_rejects_bad_input() -> None:
    edges = graph_frame()

    with pytest.raises(ValueError, match="at least two"):
        select_circuit(edges, node_count=1)
    with pytest.raises(ValueError, match="connected core"):
        select_circuit(edges, node_count=9)
    with pytest.raises(ValueError, match="Missing graph columns"):
        select_circuit(pl.DataFrame({"pre": [1], "post": [2]}), node_count=2)
    with pytest.raises(ValueError, match="finite and positive"):
        select_circuit(edges.with_columns(pl.lit(0).alias("weight")), node_count=2)
    with pytest.raises(ValueError, match="no edges"):
        select_circuit(
            pl.DataFrame(schema={"pre": pl.Int64, "post": pl.Int64, "weight": pl.Int64}),
            node_count=2,
        )


def test_write_circuits_publishes_sizes_and_hashes(tmp_path: Path) -> None:
    graph = tmp_path / "graph.parquet"
    graph_frame().write_parquet(graph)
    output = tmp_path / "circuits"

    manifest = write_circuits(graph, output, node_counts=[2, 4])

    assert [entry["node_count"] for entry in manifest["circuits"]] == [2, 4]
    assert manifest["rule"] == RULE
    assert manifest["rule_version"] == RULE_VERSION
    assert manifest["source"] == {"file": "graph.parquet", "sha256": file_sha256(graph)}
    published = json.loads((output / "selection.json").read_text())
    assert published == manifest
    for entry in manifest["circuits"]:
        path = output / str(entry["file"])
        assert entry["sha256"] == file_sha256(path)
        assert pl.read_parquet(path).height == entry["edge_count"]


def test_write_circuits_is_repeatable(tmp_path: Path) -> None:
    graph = tmp_path / "graph.parquet"
    graph_frame().write_parquet(graph)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = write_circuits(graph, first, node_counts=[3])
    second_manifest = write_circuits(graph, second, node_counts=[3])

    assert first_manifest == second_manifest
    assert (first / "circuit-3.parquet").read_bytes() == (second / "circuit-3.parquet").read_bytes()
