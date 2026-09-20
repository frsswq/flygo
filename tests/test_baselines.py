import numpy as np
import pytest

from flygo.go import Position
from flygo.model import DensePolicy
from flygo.training import TrainingData, evaluate_loss, fit_policy, loss_and_gradients


@pytest.mark.parametrize("hidden_count", [None, 4])
def test_dense_baselines_train_with_verified_gradients(hidden_count: int | None) -> None:
    model = DensePolicy.initialize(size=5, hidden_count=hidden_count, seed=7)
    data = TrainingData(
        np.tile(Position.empty(5).features(), (4, 1)),
        np.ones((4, 26), dtype=np.bool_),
        np.tile(np.eye(26, dtype=np.float32)[12], (4, 1)),
        np.ones(4, dtype=np.float32),
    )
    _, _, gradients = loss_and_gradients(model, data, value_weight=0.7)
    parameters = (model.readout, model.value_readout)
    if model.encoder is not None:
        parameters = (model.encoder, *parameters)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        index = np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
        original = parameter[index].copy()
        losses = []
        for offset in (0.005, -0.005):
            parameter[index] = original + offset
            policy_loss, value_loss = evaluate_loss(model, data)
            losses.append(policy_loss + 0.7 * value_loss)
        parameter[index] = original
        np.testing.assert_allclose(
            gradient[index], (losses[0] - losses[1]) / 0.01, atol=1e-4, rtol=0.01
        )
    initial = evaluate_loss(model, data)
    history = fit_policy(model, data, epochs=15, batch_size=4, learning_rate=0.03, seed=7)
    final = evaluate_loss(model, data)
    assert len(history) == 15
    assert final[0] < initial[0]
    assert final[1] < initial[1]
    logits, value = model.evaluate(Position.empty(5))
    assert logits.argmax() == 12
    assert value > 0
