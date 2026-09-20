"""Deterministic paired tournaments and relative Elo estimation."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flygo.go import Position
from flygo.model import Policy
from flygo.search import search

type ActionSelector = Callable[[Position, int], int]


@dataclass(frozen=True)
class Agent:
    name: str
    select_action: ActionSelector


@dataclass(frozen=True)
class GameRecord:
    black: str
    white: str
    winner: int
    moves: tuple[int, ...]
    opening: int
    paired_game: int
    adjudicated: bool

    @property
    def black_score(self) -> float:
        if self.winner == 0:
            return 0.5
        return 1.0 if self.winner == 1 else 0.0


@dataclass(frozen=True)
class Rating:
    elo: float
    lower: float
    upper: float


def random_agent(*, seed: int) -> Agent:
    generator = random.Random(seed)
    return Agent("random", lambda position, _passes: generator.choice(position.legal_actions()))


def greedy_agent(policy: Policy) -> Agent:
    def select(position: Position, _passes: int) -> int:
        logits, _ = policy.evaluate(position)
        return max(position.legal_actions(), key=lambda action: (float(logits[action]), -action))

    return Agent("flygo-greedy", select)


def mcts_agent(
    policy: Policy,
    *,
    time_limit: float | None = None,
    simulations: int | None = None,
    name: str | None = None,
) -> Agent:
    if (time_limit is None) == (simulations is None):
        raise ValueError("MCTS agent needs exactly one search budget")
    label = name or (
        f"flygo-mcts-{simulations}v" if simulations is not None else f"flygo-mcts-{time_limit:g}s"
    )

    def select(position: Position, passes: int) -> int:
        return search(
            position,
            policy,
            consecutive_passes=passes,
            time_limit=time_limit,
            max_simulations=simulations,
        ).action

    return Agent(label, select)


def _replay_opening(size: int, opening: Sequence[int]) -> tuple[Position, int]:
    position = Position.empty(size)
    passes = 0
    for action in opening:
        if passes >= 2:
            raise ValueError("Opening continues after the game ended")
        position = position.play(action)
        passes = passes + 1 if action == position.pass_action else 0
    return position, passes


def play_game(
    black: Agent,
    white: Agent,
    *,
    size: int,
    opening: Sequence[int] = (),
    opening_index: int = 0,
    paired_game: int = 0,
    max_moves: int | None = None,
) -> GameRecord:
    """Play one game and area-adjudicate only if its safety limit is reached."""
    position, passes = _replay_opening(size, opening)
    moves = list(opening)
    limit = max_moves if max_moves is not None else 3 * size * size
    if limit < len(moves):
        raise ValueError("max_moves is shorter than the opening")
    while passes < 2 and len(moves) < limit:
        agent = black if position.to_play == 1 else white
        action = agent.select_action(position, passes)
        if action not in position.legal_actions():
            raise ValueError(f"Agent {agent.name} selected illegal action {action}")
        position = position.play(action)
        moves.append(action)
        passes = passes + 1 if action == position.pass_action else 0
    result = position.result()
    return GameRecord(
        black.name,
        white.name,
        result.winner,
        tuple(moves),
        opening_index,
        paired_game,
        passes < 2,
    )


def paired_tournament(
    agents: Sequence[Agent],
    *,
    size: int,
    openings: Sequence[Sequence[int]],
    rounds: int = 1,
    max_moves: int | None = None,
) -> list[GameRecord]:
    """Play every agent pair with both colors on every opening."""
    if len({agent.name for agent in agents}) != len(agents):
        raise ValueError("Agent names must be unique")
    if rounds < 1 or not openings:
        raise ValueError("Tournament needs a round and at least one opening")
    games: list[GameRecord] = []
    pair = 0
    for _ in range(rounds):
        for opening_index, opening in enumerate(openings):
            for left_index, left in enumerate(agents):
                for right in agents[left_index + 1 :]:
                    games.append(
                        play_game(
                            left,
                            right,
                            size=size,
                            opening=opening,
                            opening_index=opening_index,
                            paired_game=pair,
                            max_moves=max_moves,
                        )
                    )
                    games.append(
                        play_game(
                            right,
                            left,
                            size=size,
                            opening=opening,
                            opening_index=opening_index,
                            paired_game=pair,
                            max_moves=max_moves,
                        )
                    )
                    pair += 1
    return games


def fit_elo(games: Sequence[GameRecord], *, anchor: str | None = None) -> dict[str, float]:
    """Fit regularized Bradley-Terry ratings with one fixed zero-Elo anchor."""
    names = sorted({game.black for game in games} | {game.white for game in games})
    if len(names) < 2:
        raise ValueError("Elo needs at least two agents")
    fixed = anchor or names[0]
    if fixed not in names:
        raise ValueError(f"Unknown Elo anchor {fixed}")
    unknown = [name for name in names if name != fixed]
    index = {name: offset for offset, name in enumerate(unknown)}
    ratings = np.zeros(len(unknown), dtype=np.float64)
    scale = math.log(10) / 400
    prior_variance = 1000.0**2
    for _ in range(50):
        gradient = -ratings / prior_variance
        hessian = -np.eye(len(unknown), dtype=np.float64) / prior_variance
        for game in games:
            black_rating = 0.0 if game.black == fixed else ratings[index[game.black]]
            white_rating = 0.0 if game.white == fixed else ratings[index[game.white]]
            expected = 1 / (1 + math.exp(-scale * (black_rating - white_rating)))
            residual = game.black_score - expected
            weight = scale * scale * expected * (1 - expected)
            black_index = index.get(game.black)
            white_index = index.get(game.white)
            if black_index is not None:
                gradient[black_index] += scale * residual
                hessian[black_index, black_index] -= weight
            if white_index is not None:
                gradient[white_index] -= scale * residual
                hessian[white_index, white_index] -= weight
            if black_index is not None and white_index is not None:
                hessian[black_index, white_index] += weight
                hessian[white_index, black_index] += weight
        step = np.linalg.solve(hessian, gradient)
        ratings -= step
        if np.max(np.abs(step), initial=0) < 1e-7:
            break
    return {name: (0.0 if name == fixed else float(ratings[index[name]])) for name in names}


def elo_with_confidence(
    games: Sequence[GameRecord],
    *,
    anchor: str | None = None,
    bootstrap_samples: int = 500,
    seed: int = 7,
) -> dict[str, Rating]:
    """Fit Elo and bootstrap paired-game blocks for 95% intervals."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    point = fit_elo(games, anchor=anchor)
    blocks: dict[int, list[GameRecord]] = {}
    for game in games:
        blocks.setdefault(game.paired_game, []).append(game)
    generator = random.Random(seed)
    block_ids = sorted(blocks)
    samples = {name: [] for name in point}
    for _ in range(bootstrap_samples):
        sampled = [game for _ in block_ids for game in blocks[generator.choice(block_ids)]]
        fitted = fit_elo(sampled, anchor=anchor)
        for name, rating in fitted.items():
            samples[name].append(rating)
    return {
        name: Rating(
            rating,
            float(np.percentile(samples[name], 2.5)),
            float(np.percentile(samples[name], 97.5)),
        )
        for name, rating in point.items()
    }


def write_tournament_report(
    path: Path,
    games: Sequence[GameRecord],
    ratings: dict[str, Rating],
    *,
    metadata: dict[str, Any],
) -> None:
    """Write all games, ratings, and run settings instead of selected results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "ratings": {name: asdict(rating) for name, rating in ratings.items()},
        "games": [asdict(game) for game in games],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
