import json
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from flygo.connectome import load_connectome
from flygo.export import ASSET_DIRECTORY, load_web_policies, write_web_bundle
from flygo.go import Position
from flygo.model import ConnectomePolicy
from flygo.training import save_policy


@pytest.mark.parametrize(
    ("normalization", "manifest_value"),
    [("incoming", "incoming weight sum"), ("none", "none")],
)
def test_export_round_trip_preserves_normalization(
    tmp_path: Path,
    normalization: Literal["incoming", "none"],
    manifest_value: str,
) -> None:
    graph = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    if normalization == "none":
        graph = graph.without_normalization()
    policy = ConnectomePolicy.initialize(graph, size=5, seed=7)
    checkpoint = tmp_path / "policy.npz"
    save_policy(checkpoint, policy)
    bundle = tmp_path / "bundle"

    write_web_bundle(bundle, policy_checkpoint=checkpoint, normalization=normalization)

    restored = load_web_policies(bundle)[5]
    position = Position.empty(5).play(0).play(12)
    expected_logits, expected_value = policy.evaluate(position)
    actual_logits, actual_value = restored.evaluate(position)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["dynamics"]["normalization"] == manifest_value
    np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-6)
    np.testing.assert_allclose(actual_value, expected_value, atol=1e-6)


def test_export_rejects_checkpoint_with_different_normalization(tmp_path: Path) -> None:
    graph = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    policy = ConnectomePolicy.initialize(graph.without_normalization(), size=5)
    checkpoint = tmp_path / "policy.npz"
    save_policy(checkpoint, policy)

    with pytest.raises(ValueError, match="different graph"):
        write_web_bundle(tmp_path / "bundle", policy_checkpoint=checkpoint)


def test_export_cli_rejects_unsupported_normalization(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "flygo",
            "export-web",
            "--output",
            str(tmp_path / "bundle"),
            "--normalization",
            "custom",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid choice: 'custom'" in result.stderr
