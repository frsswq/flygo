import numpy as np
import polars as pl
import pytest

from flygo.connectome import from_frame
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
