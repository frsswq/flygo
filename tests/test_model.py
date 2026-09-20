import numpy as np
import polars as pl
import pytest

from flygo.connectome import from_frame
from flygo.go import Position
from flygo.model import ConnectomePolicy


def policy() -> ConnectomePolicy:
    graph = from_frame(pl.DataFrame({"pre": [1], "post": [2], "weight": [1]}))
    return ConnectomePolicy.initialize(graph, size=5)


@pytest.mark.parametrize("label", [-1, 26])
def test_fit_readout_rejects_labels_outside_the_action_space(label: int) -> None:
    model = policy()
    activities = np.zeros((1, model.connectome.node_count), dtype=np.float32)
    labels = np.asarray([label], dtype=np.int64)

    with pytest.raises(ValueError, match="Labels must be between 0 and 25"):
        model.fit_readout(activities, labels)


def test_fit_readout_requires_one_label_per_activity() -> None:
    model = policy()
    activities = np.zeros((2, model.connectome.node_count), dtype=np.float32)
    labels = np.asarray([0], dtype=np.int64)

    with pytest.raises(ValueError, match="one action for each activity row"):
        model.fit_readout(activities, labels)


def test_evaluate_returns_policy_logits_and_bounded_value() -> None:
    model = policy()

    logits, value = model.evaluate(Position.empty(5))

    assert logits.shape == (26,)
    assert -1 <= value <= 1


def test_dynamics_change_the_recurrent_state() -> None:
    graph = from_frame(pl.DataFrame({"pre": [1], "post": [2], "weight": [1]}))

    slow = ConnectomePolicy.initialize(graph, size=5, seed=3, retention=0.0, recurrent_gain=0.0)
    fast = ConnectomePolicy.initialize(graph, size=5, seed=3, retention=0.9, recurrent_gain=0.0)

    np.testing.assert_array_equal(slow.encoder, fast.encoder)
    assert not np.allclose(slow.activity(Position.empty(5)), fast.activity(Position.empty(5)))


@pytest.mark.parametrize(
    ("steps", "retention", "recurrent_gain"),
    [(0, 0.35, 0.9), (8, -0.1, 0.9), (8, 0.35, float("nan"))],
)
def test_invalid_dynamics_are_rejected(steps: int, retention: float, recurrent_gain: float) -> None:
    graph = from_frame(pl.DataFrame({"pre": [1], "post": [2], "weight": [1]}))

    with pytest.raises(ValueError):
        ConnectomePolicy.initialize(
            graph,
            size=5,
            steps=steps,
            retention=retention,
            recurrent_gain=recurrent_gain,
        )
