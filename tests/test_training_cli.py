import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from flygo.connectome import load_connectome
from flygo.export import ASSET_DIRECTORY, write_web_bundle
from flygo.go import Position
from flygo.model import ConnectomePolicy
from flygo.policy_fixture import build_policy_conformance
from flygo.training import load_policy


def test_cli_checkpoint_inference_uses_the_training_steps(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    position = Position.empty(5)
    files = {}
    for split in ("train", "validation"):
        path = dataset / f"{split}.npz"
        np.savez(
            path,
            features=position.features()[None, :],
            legal=np.ones((1, 26), dtype=np.bool_),
            policy=np.eye(26, dtype=np.float32)[[12]],
            value=np.ones(1, dtype=np.float32),
        )
        files[split] = {
            "file": path.name,
            "examples": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (dataset / "manifest.json").write_text(json.dumps({"fixture": True, "files": files}))
    checkpoint = tmp_path / "policy.npz"
    result = subprocess.run(
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
            "3",
            "--retention",
            "0",
            "--recurrent-gain",
            "0",
            "--epochs",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    graph = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    policy, metadata = load_policy(checkpoint, graph)
    assert isinstance(policy, ConnectomePolicy)
    # With zero retention and gain, each recurrent step depends only on the encoder.
    expected_activity = np.tanh(np.tanh(policy.encoder @ position.features()))
    logits, value = policy.evaluate(position)
    assert metadata["steps"] == 3
    assert metadata["retention"] == 0
    assert metadata["recurrent_gain"] == 0
    np.testing.assert_allclose(logits, policy.readout @ expected_activity, atol=1e-6)
    np.testing.assert_allclose(value, np.tanh(policy.value_readout @ expected_activity), atol=1e-6)
    reported = re.search(r"validation policy=([^,]+), value=([^\s]+)", result.stdout)
    assert reported is not None
    shifted = logits - logits.max()
    expected_policy_loss = float(np.log(np.exp(shifted).sum()) - shifted[12])
    np.testing.assert_allclose(float(reported.group(1)), expected_policy_loss, atol=1e-6)
    np.testing.assert_allclose(float(reported.group(2)), (value - 1) ** 2, atol=1e-6)
    bundle = tmp_path / "web"
    write_web_bundle(bundle, policy_checkpoint=checkpoint)
    fixture = build_policy_conformance(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["dynamics"]["retention"] == 0
    assert manifest["dynamics"]["recurrent_gain"] == 0
    empty_case = next(case for case in fixture["cases"] if case["size"] == 5 and not case["moves"])
    assert fixture["steps"] == 3
    np.testing.assert_allclose(empty_case["logits"], logits, atol=1e-6)
    np.testing.assert_allclose(empty_case["value"], value, atol=1e-6)


def test_cli_train_requires_a_published_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    checkpoint = tmp_path / "policy.npz"

    result = subprocess.run(
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
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not published" in result.stderr
    assert not checkpoint.exists()
