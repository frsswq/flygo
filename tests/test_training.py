from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygo.connectome import from_frame
from flygo.model import ConnectomePolicy
from flygo.training import (
    TrainingData,
    apply_symmetry,
    evaluate_loss,
    fit_policy,
    load_policy,
    loss_and_gradients,
    save_policy,
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
    initial = ConnectomePolicy.initialize(graph(), size=5, steps=2)
    fit_policy(initial, data, epochs=1, batch_size=8)
    initial_loss = evaluate_loss(initial, data)

    trained = ConnectomePolicy.initialize(graph(), size=5, steps=2)
    history = fit_policy(
        trained,
        data,
        epochs=30,
        batch_size=8,
        learning_rate=0.02,
    )
    trained_loss = evaluate_loss(trained, data)

    assert len(history) == 30
    assert trained_loss[0] < initial_loss[0]
    assert trained_loss[1] < initial_loss[1]


def test_checkpoint_is_bound_to_the_frozen_graph(tmp_path: Path) -> None:
    connectome = graph()
    trained = ConnectomePolicy.initialize(
        connectome, size=5, steps=2, retention=0.5, recurrent_gain=0.25
    )
    fit_policy(trained, examples(), epochs=1)
    path = tmp_path / "policy.npz"

    save_policy(path, trained, metadata={"dataset": "fixture"})
    restored, metadata = load_policy(path, connectome)

    assert metadata["dataset"] == "fixture"
    assert metadata["steps"] == 2
    assert metadata["retention"] == 0.5
    assert metadata["recurrent_gain"] == 0.25
    assert isinstance(restored, ConnectomePolicy)
    assert restored.retention == 0.5
    assert restored.recurrent_gain == 0.25
    np.testing.assert_array_equal(restored.encoder, trained.encoder)
    np.testing.assert_array_equal(restored.readout, trained.readout)
    np.testing.assert_array_equal(restored.value_readout, trained.value_readout)

    other = from_frame(pl.DataFrame({"pre": [1], "post": [2], "weight": [9]}))
    with pytest.raises(ValueError, match="different graph"):
        load_policy(path, other)


def test_mean_loss_gradients_match_finite_differences_for_every_parameter() -> None:
    model = ConnectomePolicy.initialize(graph(), size=5, steps=3)
    data = examples(rows=4)
    value_weight = 0.7
    _, _, gradients = loss_and_gradients(model, data, value_weight=value_weight)
    for parameter, gradient in zip(
        (model.encoder, model.readout, model.value_readout), gradients, strict=True
    ):
        index = np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
        original = parameter[index].copy()
        epsilon = 0.005
        losses = []
        for offset in (epsilon, -epsilon):
            parameter[index] = original + offset
            policy_loss, value_loss = evaluate_loss(model, data)
            losses.append(policy_loss + value_weight * value_loss)
        parameter[index] = original
        numerical_gradient = (losses[0] - losses[1]) / (2 * epsilon)
        np.testing.assert_allclose(gradient[index], numerical_gradient, rtol=0.01, atol=1e-4)


def test_validation_selects_best_epoch_instead_of_last_epoch() -> None:
    model = ConnectomePolicy.initialize(graph(), size=5, steps=2)
    train = examples()
    validation = examples()
    validation.value[:] = -1
    history = fit_policy(
        model,
        train,
        validation=validation,
        epochs=10,
        batch_size=8,
        learning_rate=0.05,
        value_weight=10,
    )
    scores = [
        epoch.validation_policy_loss + 10 * epoch.validation_value_loss
        for epoch in history
        if epoch.validation_policy_loss is not None and epoch.validation_value_loss is not None
    ]
    policy_loss, value_loss = evaluate_loss(model, validation)
    assert scores[-1] > min(scores)
    assert policy_loss + 10 * value_loss == pytest.approx(min(scores), abs=1e-6)


@pytest.mark.parametrize("learning_rate", [float("nan"), float("inf"), 0, -1])
def test_fit_rejects_invalid_learning_rate(learning_rate: float) -> None:
    model = ConnectomePolicy.initialize(graph(), size=5)
    with pytest.raises(ValueError, match="hyperparameters"):
        fit_policy(model, examples(), learning_rate=learning_rate)
