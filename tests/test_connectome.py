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
