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
    float32[action_count * node_count] readout.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
POLICY_VERSION = 1
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
    ]
    return b"".join(chunks)


def build_policies(connectome: FrozenConnectome) -> dict[int, ConnectomePolicy]:
    return {
        size: ConnectomePolicy.initialize(connectome, size=size, seed=POLICY_SEED)
        for size in BOARD_SIZES
    }


def write_web_bundle(
    output: Path = DEFAULT_BUNDLE,
    *,
    steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Write the browser binaries and manifest, and return the manifest."""
    from flygo.connectome import load_connectome

    connectome = load_connectome(ASSET_DIRECTORY / "malecns-sample.parquet")
    policies = build_policies(connectome)
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
                "seed": POLICY_SEED,
                "feature_count": policy.encoder.shape[1],
                "action_count": policy.readout.shape[0],
                "sha256": _sha256(path),
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
        "model_status": "Untrained random encoder and readout",
        "policies": policy_entries,
        "ruleset": RULESET,
        "source": "Official MaleCNS v1.0 derived sensory-path sample",
    }
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


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


def arrays_from_policy_file(path: Path) -> tuple[PolicyHeader, list[float], list[float]]:
    """Read an exported policy, for tests and tooling."""
    header_fields = struct.unpack("<IIIIII", path.read_bytes()[:24])
    magic, version, size, node_count, feature_count, action_count = header_fields
    if magic != POLICY_MAGIC:
        raise ValueError(f"{path} is not a FlyGo policy bundle")
    if version != POLICY_VERSION:
        raise ValueError(f"{path} has version {version}, expected {POLICY_VERSION}")
    header = PolicyHeader(size, node_count, feature_count, action_count)
    values = struct.unpack(
        f"<{action_count * node_count + node_count * feature_count}f",
        path.read_bytes()[24:],
    )
    split = node_count * feature_count
    return header, list(values[:split]), list(values[split:])
