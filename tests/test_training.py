from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygo.connectome import from_frame
from flygo.go import Position
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


def position_examples() -> tuple[list[Position], TrainingData]:
    positions = [Position.empty(5), Position.empty(5).play(0), Position.empty(5).play(12)]
    features = np.stack([position.features() for position in positions])
    legal = np.zeros((len(positions), 26), dtype=np.bool_)
    for row, position in enumerate(positions):
        legal[row, position.legal_actions()] = True
    policy = np.zeros((len(positions), 26), dtype=np.float32)
    policy[:, 25] = 1
    value = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    return positions, TrainingData(features, legal, policy, value)


def inference_loss(
    model: ConnectomePolicy,
    positions: list[Position],
    data: TrainingData,
    *,
    value_weight: float,
) -> float:
    policy_loss = 0.0
    value_loss = 0.0
    for row, position in enumerate(positions):
        logits, value = model.evaluate(position)
        masked = np.where(data.legal[row], logits, -1e9)
        shifted = masked - masked.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        policy_loss -= float(
            np.sum(data.policy[row, data.legal[row]] * np.log(probabilities[data.legal[row]]))
        )
        value_loss += (value - data.value[row]) ** 2
    return (policy_loss + value_weight * value_loss) / len(positions)


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


def test_mean_loss_gradients_match_configured_inference_for_every_parameter() -> None:
    model = ConnectomePolicy.initialize(graph(), size=5, steps=3, retention=0.6, recurrent_gain=0.2)
    positions, data = position_examples()
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
            losses.append(inference_loss(model, positions, data, value_weight=value_weight))
        parameter[index] = original
        numerical_gradient = (losses[0] - losses[1]) / (2 * epsilon)
        np.testing.assert_allclose(gradient[index], numerical_gradient, rtol=0.01, atol=1e-4)


@pytest.mark.parametrize(
    ("retention", "recurrent_gain"),
    [(0.35, 0.9), (0.0, 0.0), (0.6, 0.2)],
)
def test_evaluation_matches_configured_policy_inference(
    retention: float, recurrent_gain: float
) -> None:
    model = ConnectomePolicy.initialize(
        graph(), size=5, steps=3, retention=retention, recurrent_gain=recurrent_gain
    )
    positions, data = position_examples()

    policy_loss, value_loss = evaluate_loss(model, data)
    expected_combined = inference_loss(model, positions, data, value_weight=1.0)

    assert policy_loss + value_loss == pytest.approx(expected_combined, abs=1e-6)


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
