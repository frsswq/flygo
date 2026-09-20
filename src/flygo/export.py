"""Export the frozen graph and policy weights for browser inference.

The browser cannot import parquet files or regenerate NumPy arrays, so this
module writes explicit little-endian binaries plus a manifest that records the
byte layout, the dynamics constants, and a SHA-256 for every file.

Binary layouts (all integers little endian, all floats IEEE 754 binary32):

``graph.bin``
    uint32 magic "FLYG", uint32 version, uint32 node_count, uint32 edge_count,
    int32[node_count] node_ids, uint32[edge_count] sources,
    uint32[edge_count] targets, float32[edge_count] weights,
    float32[node_count] incoming_strength.

``policy-{size}.bin``
    uint32 magic "FLYP", uint32 version, uint32 size, uint32 node_count,
    uint32 feature_count, uint32 action_count,
    float32[node_count * feature_count] encoder,
    float32[action_count * node_count] readout, float32[node_count] value_readout.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flygo.connectome import FrozenConnectome
from flygo.go import BOARD_SIZES, RULESET
from flygo.model import ConnectomePolicy

PACKAGE_DIRECTORY = Path(__file__).parent
ASSET_DIRECTORY = PACKAGE_DIRECTORY / "assets"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPOSITORY_ROOT / "web" / "public" / "flygo"
GRAPH_FILE = "graph.bin"
MANIFEST_FILE = "manifest.json"
GRAPH_MAGIC = int.from_bytes(b"FLYG", "little")
POLICY_MAGIC = int.from_bytes(b"FLYP", "little")
GRAPH_VERSION = 1
POLICY_VERSION = 2
POLICY_SEED = 7
DEFAULT_STEPS = 8


@dataclass(frozen=True)
class PolicyHeader:
    size: int
    node_count: int
    feature_count: int
    action_count: int


def policy_file(size: int) -> str:
    return f"policy-{size}.bin"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph_bytes(connectome: FrozenConnectome) -> bytes:
    node_count = connectome.node_count
    edge_count = connectome.edge_count
    chunks = [
        struct.pack("<IIII", GRAPH_MAGIC, GRAPH_VERSION, node_count, edge_count),
        connectome.node_ids.astype("<i4", copy=False).tobytes(),
        connectome.source_indices.astype("<u4", copy=False).tobytes(),
        connectome.target_indices.astype("<u4", copy=False).tobytes(),
        connectome.weights.astype("<f4", copy=False).tobytes(),
        connectome.incoming_strength.astype("<f4", copy=False).tobytes(),
    ]
    return b"".join(chunks)


def _policy_bytes(policy: ConnectomePolicy) -> bytes:
    header = PolicyHeader(
        size=policy.size,
        node_count=policy.connectome.node_count,
        feature_count=policy.encoder.shape[1],
        action_count=policy.readout.shape[0],
    )
    if policy.encoder.shape != (header.node_count, header.feature_count):
        raise ValueError("Encoder shape does not match the manifest header")
    if policy.readout.shape != (header.action_count, header.node_count):
        raise ValueError("Readout shape does not match the manifest header")
    if policy.value_readout.shape != (header.node_count,):
        raise ValueError("Value readout shape does not match the manifest header")
    chunks = [
        struct.pack(
            "<IIIIII",
            POLICY_MAGIC,
            POLICY_VERSION,
            header.size,
            header.node_count,
            header.feature_count,
            header.action_count,
        ),
        policy.encoder.astype("<f4", copy=False).tobytes(),
        policy.readout.astype("<f4", copy=False).tobytes(),
        policy.value_readout.astype("<f4", copy=False).tobytes(),
    ]
    return b"".join(chunks)


def build_policies(
    connectome: FrozenConnectome, *, steps: int = DEFAULT_STEPS
) -> dict[int, ConnectomePolicy]:
    return {
        size: ConnectomePolicy.initialize(connectome, size=size, seed=POLICY_SEED, steps=steps)
        for size in BOARD_SIZES
    }


def write_web_bundle(
    output: Path = DEFAULT_BUNDLE,
    *,
    steps: int = DEFAULT_STEPS,
    policy_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Write the browser binaries and manifest, and return the manifest."""
    from flygo.connectome import load_connectome

    connectome = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    policies = build_policies(connectome, steps=steps)
    checkpoint_metadata: dict[str, Any] | None = None
    if policy_checkpoint is not None:
        from flygo.training import load_policy

        trained_policy, checkpoint_metadata = load_policy(policy_checkpoint, connectome)
        if not isinstance(trained_policy, ConnectomePolicy):
            raise ValueError("Browser export requires a connectome policy")
        policies[trained_policy.size] = trained_policy
        steps = int(checkpoint_metadata["steps"])
    output.mkdir(parents=True, exist_ok=True)

    graph_path = output / GRAPH_FILE
    graph_path.write_bytes(_graph_bytes(connectome))

    policy_entries = []
    for size, policy in policies.items():
        path = output / policy_file(size)
        path.write_bytes(_policy_bytes(policy))
        policy_entries.append(
            {
                "size": size,
                "file": path.name,
                "seed": (
                    checkpoint_metadata.get("seed", POLICY_SEED)
                    if checkpoint_metadata is not None and size == checkpoint_metadata["size"]
                    else POLICY_SEED
                ),
                "feature_count": policy.encoder.shape[1],
                "action_count": policy.readout.shape[0],
                "sha256": _sha256(path),
                "trained": checkpoint_metadata is not None and size == checkpoint_metadata["size"],
            }
        )

    manifest: dict[str, Any] = {
        "generated_by": "flygo export-web",
        "graph": {
            "file": graph_path.name,
            "node_count": connectome.node_count,
            "edge_count": connectome.edge_count,
            "sha256": _sha256(graph_path),
        },
        "dynamics": {
            "activation": "tanh",
            "normalization": "incoming weight sum",
            "recurrent_gain": 0.9,
            "retention": 0.35,
            "steps": steps,
        },
        "model_status": (
            "Trained policy-value checkpoint"
            if checkpoint_metadata is not None
            else "Untrained random encoder and readout"
        ),
        "checkpoint": checkpoint_metadata,
        "policies": policy_entries,
        "ruleset": RULESET,
        "source": "Official MaleCNS v1.0 derived sensory-path sample",
    }
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_web_policies(bundle: Path) -> dict[int, ConnectomePolicy]:
    """Read the actual exported weights, rejecting unsupported or damaged bundles."""
    from flygo.connectome import load_connectome

    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    connectome = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    graph_bytes = (bundle / GRAPH_FILE).read_bytes()
    if graph_bytes != _graph_bytes(connectome):
        raise ValueError("Browser bundle has a different graph")
    if hashlib.sha256(graph_bytes).hexdigest() != manifest["graph"]["sha256"]:
        raise ValueError("Browser graph hash does not match the manifest")
    dynamics = manifest["dynamics"]
    steps = dynamics["steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("Browser steps must be a positive integer")
    if dynamics != {
        "activation": "tanh",
        "normalization": "incoming weight sum",
        "retention": 0.35,
        "recurrent_gain": 0.9,
        "steps": steps,
    }:
        raise ValueError("Unsupported browser dynamics")
    policies: dict[int, ConnectomePolicy] = {}
    for entry in manifest["policies"]:
        size = entry["size"]
        if size not in BOARD_SIZES or size in policies:
            raise ValueError("Invalid or duplicate browser board size")
        raw = (bundle / policy_file(size)).read_bytes()
        nodes = connectome.node_count
        features, actions = 2 * size * size + 1, size * size + 1
        expected_header = (POLICY_MAGIC, POLICY_VERSION, size, nodes, features, actions)
        if len(raw) != 24 + 4 * nodes * (features + actions + 1):
            raise ValueError("Invalid browser policy length")
        if struct.unpack_from("<IIIIII", raw) != expected_header:
            raise ValueError("Invalid browser policy header")
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Browser policy hash does not match the manifest")
        weights = np.frombuffer(raw, dtype="<f4", offset=24).copy()
        if not np.all(np.isfinite(weights)):
            raise ValueError("Browser weights must be finite")
        encoder_end = nodes * features
        readout_end = encoder_end + nodes * actions
        policies[size] = ConnectomePolicy(
            connectome,
            weights[:encoder_end].reshape(nodes, features),
            weights[encoder_end:readout_end].reshape(actions, nodes),
            weights[readout_end:],
            size,
            steps,
        )
    if set(policies) != set(BOARD_SIZES):
        raise ValueError("Browser bundle is missing a board size")
    return policies


def estimate_bundle(output: Path = DEFAULT_BUNDLE) -> dict[str, int]:
    sizes = {path.name: path.stat().st_size for path in sorted(output.glob("*")) if path.is_file()}
    sizes["total"] = sum(sizes.values())
    return sizes


def read_graph_header(path: Path) -> tuple[int, int]:
    """Read the header of an exported graph, for tests and tooling."""
    magic, version, node_count, edge_count = struct.unpack("<IIII", path.read_bytes()[:16])
    if magic != GRAPH_MAGIC:
        raise ValueError(f"{path} is not a FlyGo graph bundle")
    if version != GRAPH_VERSION:
        raise ValueError(f"{path} has version {version}, expected {GRAPH_VERSION}")
    return node_count, edge_count
