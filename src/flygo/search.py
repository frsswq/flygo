"""Deadline-aware PUCT search over the FlyGo policy-value model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from flygo.go import Position
from flygo.model import Policy


@dataclass
class SearchNode:
    """One tree node whose value is from its current player's perspective."""

    position: Position
    consecutive_passes: int
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def terminal(self) -> bool:
        return self.consecutive_passes >= 2


@dataclass(frozen=True)
class SearchResult:
    action: int
    elapsed_seconds: float
    root_value: float
    simulations: int
    visits: dict[int, int]


def _softmax_priors(logits: np.ndarray, legal: tuple[int, ...]) -> dict[int, float]:
    selected = np.asarray([logits[action] for action in legal], dtype=np.float64)
    selected -= selected.max()
    probabilities = np.exp(selected)
    probabilities /= probabilities.sum()
    return {
        action: float(probability) for action, probability in zip(legal, probabilities, strict=True)
    }


def _terminal_value(node: SearchNode) -> float:
    winner = node.position.result().winner
    return float(winner * node.position.to_play)


def _expand(node: SearchNode, policy: Policy) -> float:
    if node.terminal:
        return _terminal_value(node)
    logits, value = policy.evaluate(node.position)
    legal_children = node.position.legal_children()
    priors = _softmax_priors(logits, tuple(action for action, _ in legal_children))
    for action, child_position in legal_children:
        prior = priors[action]
        passes = node.consecutive_passes + 1 if action == node.position.pass_action else 0
        node.children[action] = SearchNode(child_position, passes, prior)
    return value


def _select(node: SearchNode, exploration: float) -> SearchNode:
    scale = math.sqrt(max(1, node.visits))
    _, child = max(
        node.children.items(),
        key=lambda item: (
            -item[1].mean_value + exploration * item[1].prior * scale / (1 + item[1].visits),
            item[1].prior,
            -item[0],
        ),
    )
    return child


def _simulate(root: SearchNode, policy: Policy, exploration: float) -> None:
    path = [root]
    node = root
    while node.children and not node.terminal:
        node = _select(node, exploration)
        path.append(node)
    value = _expand(node, policy)
    for visited in reversed(path):
        visited.visits += 1
        visited.value_sum += value
        value = -value


def search(
    position: Position,
    policy: Policy,
    *,
    consecutive_passes: int = 0,
    time_limit: float | None = 1.0,
    max_simulations: int | None = None,
    exploration: float = 1.5,
) -> SearchResult:
    """Search until either budget expires, with at least one simulation."""
    if position.size != policy.size:
        raise ValueError("Position and policy board sizes differ")
    if time_limit is None and max_simulations is None:
        raise ValueError("A time or simulation budget is required")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if max_simulations is not None and max_simulations < 1:
        raise ValueError("max_simulations must be positive")
    if consecutive_passes >= 2:
        raise ValueError("Cannot search a finished game")
    root = SearchNode(position, consecutive_passes)
    started = time.perf_counter()
    deadline = started + time_limit if time_limit is not None else math.inf
    simulations = 0
    while (max_simulations is None or simulations < max_simulations) and (
        simulations == 0 or time.perf_counter() < deadline
    ):
        _simulate(root, policy, exploration)
        simulations += 1
    if not root.children:
        _expand(root, policy)
    action = max(
        root.children,
        key=lambda candidate: (
            root.children[candidate].visits,
            root.children[candidate].prior,
            -candidate,
        ),
    )
    elapsed = time.perf_counter() - started
    return SearchResult(
        action,
        elapsed,
        root.mean_value,
        simulations,
        {candidate: child.visits for candidate, child in root.children.items()},
    )
