from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygo.connectome import from_frame
from flygo.training import (
    TrainingData,
    apply_symmetry,
    evaluate_loss,
    load_policy,
    save_policy,
    train_policy_value,
)


def graph():
    return from_frame(
        pl.DataFrame(
            {
                "pre": [1, 2, 3],
                "post": [2, 3, 1],
                "weight": [2, 3, 1],
            }
        )
    )


def examples(rows: int = 8) -> TrainingData:
    features = np.zeros((rows, 51), dtype=np.float32)
    features[:, -1] = 1
    legal = np.ones((rows, 26), dtype=np.bool_)
    policy = np.zeros((rows, 26), dtype=np.float32)
    policy[:, 12] = 1
    value = np.ones(rows, dtype=np.float32)
    return TrainingData(features, legal, policy, value)


def test_dihedral_symmetry_moves_features_policy_and_legality_together() -> None:
    features = np.zeros(51, dtype=np.float32)
    features[0] = 1
    features[25 + 4] = 1
    legal = np.zeros(26, dtype=np.bool_)
    legal[[0, 25]] = True
    policy = np.zeros(26, dtype=np.float32)
    policy[0] = 0.75
    policy[25] = 0.25

    transformed_features, transformed_legal, transformed_policy = apply_symmetry(
        features, legal, policy, size=5, symmetry=1
    )

    assert transformed_features[20] == 1
    assert transformed_features[25] == 1
    assert np.flatnonzero(transformed_legal).tolist() == [20, 25]
    assert transformed_policy[[20, 25]].tolist() == [0.75, 0.25]


def test_training_reduces_policy_and_value_loss() -> None:
    data = examples()
    initial, _ = train_policy_value(graph(), data, size=5, epochs=1, batch_size=8, steps=2)
    initial_loss = evaluate_loss(initial, data, steps=2)

    trained, history = train_policy_value(
        graph(),
        data,
        size=5,
        epochs=30,
        batch_size=8,
        learning_rate=0.02,
        steps=2,
    )
    trained_loss = evaluate_loss(trained, data, steps=2)

    assert len(history) == 30
    assert trained_loss[0] < initial_loss[0]
    assert trained_loss[1] < initial_loss[1]


def test_checkpoint_is_bound_to_the_frozen_graph(tmp_path: Path) -> None:
    connectome = graph()
    trained, _ = train_policy_value(connectome, examples(), size=5, epochs=1, steps=2)
    path = tmp_path / "policy.npz"

    save_policy(path, trained, steps=2, metadata={"dataset": "fixture"})
    restored, metadata = load_policy(path, connectome)

    assert metadata["dataset"] == "fixture"
    assert metadata["steps"] == 2
    np.testing.assert_array_equal(restored.encoder, trained.encoder)
    np.testing.assert_array_equal(restored.readout, trained.readout)
    np.testing.assert_array_equal(restored.value_readout, trained.value_readout)

    other = from_frame(pl.DataFrame({"pre": [1], "post": [2], "weight": [9]}))
    with pytest.raises(ValueError, match="different graph"):
        load_policy(path, other)
