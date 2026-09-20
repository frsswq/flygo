import math

import numpy as np

from flygo.model import DensePolicy
from flygo.training import TrainingData, evaluate_policy


def test_evaluation_reports_unmasked_legality_and_masked_target_agreement() -> None:
    model = DensePolicy.initialize(size=5)
    model.readout.fill(0)
    model.value_readout.fill(0)
    legal = np.zeros((2, 26), dtype=np.bool_)
    legal[0, [0, 1]] = True
    legal[1, [1, 2]] = True
    data = TrainingData(
        np.zeros((2, 51), dtype=np.float32),
        legal,
        np.eye(26, dtype=np.float32)[[0, 2]],
        np.asarray([1, -1], dtype=np.float32),
    )
    metrics = evaluate_policy(model, data, batch_size=1)
    assert math.isclose(metrics.policy_loss, math.log(2), rel_tol=1e-6)
    assert metrics.value_mse == 1
    assert metrics.target_top1 == 0.5
    assert metrics.target_top3 == 1
    assert metrics.legal_action_rate == 0.5
    assert metrics.examples == 2
    assert metrics == evaluate_policy(model, data, batch_size=2)
