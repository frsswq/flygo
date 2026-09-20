import numpy as np
import polars as pl

from flygo.connectome import from_frame


def graph_frame() -> pl.DataFrame:
    return pl.DataFrame({"pre": [10, 20, 20], "post": [20, 10, 30], "weight": [2, 1, 3]})


def test_connectome_is_frozen_and_propagates_activity() -> None:
    graph = from_frame(graph_frame())
    state = np.asarray([1, 0, 0], dtype=np.float32)

    next_state = graph.step(state, retention=0, recurrent_gain=1)

    assert graph.node_count == 3
    assert graph.edge_count == 3
    assert next_state[1] == np.float32(np.tanh(1))
    assert not graph.weights.flags.writeable


def test_randomized_control_preserves_directed_degrees() -> None:
    graph = from_frame(graph_frame())
    randomized = graph.randomized(seed=12)

    np.testing.assert_array_equal(
        np.bincount(graph.source_indices), np.bincount(randomized.source_indices)
    )
    np.testing.assert_array_equal(
        np.bincount(graph.target_indices), np.bincount(randomized.target_indices)
    )


def test_controls_keep_nodes_and_never_modify_the_original_graph() -> None:
    graph = from_frame(graph_frame())
    original_weights = graph.weights.copy()
    shuffled = graph.shuffled_weights(seed=7)
    disconnected = graph.without_connections()

    for control in (shuffled, disconnected):
        np.testing.assert_array_equal(control.node_ids, graph.node_ids)
        assert not control.weights.flags.writeable
        assert not control.source_indices.flags.writeable
        assert not control.target_indices.flags.writeable
    np.testing.assert_array_equal(shuffled.source_indices, graph.source_indices)
    np.testing.assert_array_equal(shuffled.target_indices, graph.target_indices)
    np.testing.assert_array_equal(np.sort(shuffled.weights), np.sort(original_weights))
    np.testing.assert_array_equal(shuffled.weights, graph.shuffled_weights(seed=7).weights)
    np.testing.assert_array_equal(graph.weights, original_weights)
    assert disconnected.edge_count == 0
    state = np.asarray([0.2, -0.3, 0.4], dtype=np.float32)
    external = np.asarray([0.5, -0.1, 0.2], dtype=np.float32)
    np.testing.assert_allclose(disconnected.step(state, external), np.tanh(0.35 * state + external))


def test_rewiring_preserves_simple_directed_graph_and_is_reproducible() -> None:
    sources = np.repeat(np.arange(12), 2)
    targets = np.asarray([(source + hop) % 12 for source in range(12) for hop in (1, 3)])
    graph = from_frame(pl.DataFrame({"pre": sources, "post": targets, "weight": np.arange(1, 25)}))
    control = graph.randomized(seed=7)
    np.testing.assert_array_equal(control.source_indices, graph.source_indices)
    np.testing.assert_array_equal(control.weights, graph.weights)
    np.testing.assert_array_equal(np.sort(control.target_indices), np.sort(graph.target_indices))
    np.testing.assert_array_equal(control.target_indices, graph.randomized(seed=7).target_indices)
    assert not np.array_equal(control.target_indices, graph.target_indices)
    assert np.all(control.source_indices != control.target_indices)
    assert len(set(zip(control.source_indices, control.target_indices, strict=True))) == 24
