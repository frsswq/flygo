import json
import subprocess
from pathlib import Path

import numpy as np

from flygo.connectome import load_connectome
from flygo.export import ASSET_DIRECTORY, write_web_bundle
from flygo.go import Position
from flygo.policy_fixture import build_policy_conformance
from flygo.training import load_policy


def test_cli_checkpoint_inference_uses_the_training_steps(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    position = Position.empty(5)
    for split in ("train", "validation"):
        np.savez(
            dataset / f"{split}.npz",
            features=position.features()[None, :],
            legal=np.ones((1, 26), dtype=np.bool_),
            policy=np.eye(26, dtype=np.float32)[[12]],
            value=np.ones(1, dtype=np.float32),
        )
    (dataset / "manifest.json").write_text(json.dumps({"fixture": True}))
    checkpoint = tmp_path / "policy.npz"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "flygo",
            "train",
            "--dataset",
            str(dataset),
            "--output",
            str(checkpoint),
            "--size",
            "5",
            "--steps",
            "1",
            "--epochs",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    graph = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    policy, metadata = load_policy(checkpoint, graph)
    # With one step and a zero initial state, no recurrent drive has arrived.
    expected_activity = np.tanh(np.tanh(policy.encoder @ position.features()))
    logits, value = policy.evaluate(position)
    assert metadata["steps"] == 1
    np.testing.assert_allclose(logits, policy.readout @ expected_activity, atol=1e-6)
    np.testing.assert_allclose(value, np.tanh(policy.value_readout @ expected_activity), atol=1e-6)
    bundle = tmp_path / "web"
    write_web_bundle(bundle, policy_checkpoint=checkpoint)
    fixture = build_policy_conformance(bundle)
    empty_case = next(case for case in fixture["cases"] if case["size"] == 5 and not case["moves"])
    assert fixture["steps"] == 1
    np.testing.assert_allclose(empty_case["logits"], logits, atol=1e-6)
    np.testing.assert_allclose(empty_case["value"], value, atol=1e-6)
