"""Policy conformance fixture for the browser dynamics.

The TypeScript dynamics must reproduce the Python dynamics, so this module
records the activity, policy logits, and value of the exported policy on a set
of positions. The exported binaries hold the same encoder and readouts, and the fixture records
their SHA-256 so the browser test can prove it loaded the same weights.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flygo.export import (
    DEFAULT_BUNDLE,
    GRAPH_FILE,
    POLICY_SEED,
    load_web_policies,
    policy_file,
    read_graph_header,
)
from flygo.go import BOARD_SIZES, Position

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_CONFORMANCE = REPOSITORY_ROOT / "shared" / "policy-conformance.json"
RANDOM_SEED = 20240917
GAME_LENGTHS = (0, 5, 14, 25)
DIGITS = 8


def _random_moves(size: int, length: int, seed: int) -> list[int]:
    generator = random.Random(seed)
    position = Position.empty(size)
    moves: list[int] = []
    for _ in range(length):
        choices = [action for action in position.legal_actions() if action != position.pass_action]
        if not choices:
            break
        action = generator.choice(choices)
        position = position.play(action)
        moves.append(action)
    return moves


def _replay(size: int, moves: Sequence[int]) -> Position:
    position = Position.empty(size)
    for action in moves:
        position = position.play(action)
    return position


def _rounded(values: NDArray[np.float32]) -> list[float]:
    return [round(value, DIGITS) for value in values.tolist()]


def build_policy_conformance(
    bundle: Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    """Build the fixture from the Python engine and the exported binaries."""
    policies = load_web_policies(bundle)
    cases: list[dict[str, Any]] = []

    for size in BOARD_SIZES:
        for index, length in enumerate(GAME_LENGTHS):
            moves = _random_moves(size, length, RANDOM_SEED + size * 100 + index)
            position = _replay(size, moves)
            policy = policies[size]
            activity = policy.activity(position)
            logits = policy.logits(position)
            cases.append(
                {
                    "activity": _rounded(activity),
                    "chosen_action": policy.choose_legal_action(position),
                    "logits": _rounded(logits),
                    "moves": moves,
                    "name": f"{size}x{size} after {len(moves)} moves",
                    "size": size,
                    "value": round(float(np.tanh(policy.value_readout @ activity)), DIGITS),
                }
            )

    node_count, edge_count = read_graph_header(bundle / GRAPH_FILE)
    return {
        "cases": cases,
        "edge_count": edge_count,
        "graph_sha256": _sha256(bundle / GRAPH_FILE),
        "node_count": node_count,
        "policies": {
            str(size): {
                "seed": POLICY_SEED,
                "sha256": _sha256(bundle / policy_file(size)),
            }
            for size in BOARD_SIZES
        },
        "steps": policies[BOARD_SIZES[0]].steps,
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_policy_conformance(
    path: Path = DEFAULT_POLICY_CONFORMANCE,
    *,
    bundle: Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    fixture = build_policy_conformance(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture
