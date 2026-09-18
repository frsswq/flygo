"""Shared rules conformance fixture for the Python and TypeScript engines.

Both rule engines must agree on legal actions, captures, superko, and area
score. This module builds one fixture from the Python engine, and the TypeScript
engine replays the same cases. Keep the two engines aligned by regenerating the
fixture with ``flygo conformance`` instead of editing the JSON by hand.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flygo.go import BOARD_SIZES, RULESET, Position, komi_for, position_key

# The fixture is a committed repository artifact, so anchor it to the repository
# root instead of the working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFORMANCE = REPOSITORY_ROOT / "shared" / "rules-conformance.json"
GAME_LENGTHS = (1, 6, 20, 60)
RANDOM_SEED = 20240917
SYMBOLS = {-1: "-", 0: "0", 1: "+"}
STONES = {symbol: stone for stone, symbol in SYMBOLS.items()}


def encode_board(board: Sequence[int]) -> str:
    return "".join(SYMBOLS[stone] for stone in board)


def decode_board(text: str) -> tuple[int, ...]:
    """Parse an encoded board and reject unknown symbols."""
    board: list[int] = []
    for symbol in text:
        if symbol not in STONES:
            raise ValueError(f"Unknown board symbol '{symbol}'")
        board.append(STONES[symbol])
    return tuple(board)


def _keys(boards: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(position_key(board) for board in boards)


def _illegal(
    name: str,
    size: int,
    *,
    board: Sequence[int] | None = None,
    to_play: int = 1,
    history: Sequence[Sequence[int]] = (),
    moves: Sequence[int] = (),
) -> dict[str, Any]:
    start = tuple(board) if board is not None else (0,) * (size * size)
    position = Position(start, to_play, _keys(history), size)
    for action in moves:
        try:
            position = position.play(action)
        except ValueError as error:
            return {
                "name": name,
                "size": size,
                "board": encode_board(start),
                "to_play": to_play,
                "history": [encode_board(previous) for previous in history],
                "moves": list(moves),
                "error": str(error),
            }
    raise AssertionError(f"The case '{name}' was expected to be illegal")


def _case(
    name: str,
    size: int,
    *,
    board: Sequence[int] | None = None,
    to_play: int = 1,
    history: Sequence[Sequence[int]] = (),
    moves: Sequence[int] = (),
) -> dict[str, Any]:
    start = tuple(board) if board is not None else (0,) * (size * size)
    position = Position(start, to_play, _keys(history), size)
    for action in moves:
        position = position.play(action)
    result = position.result()
    return {
        "name": name,
        "size": size,
        "board": encode_board(start),
        "to_play": to_play,
        "history": [encode_board(previous) for previous in history],
        "moves": list(moves),
        "final_board": encode_board(position.board),
        "final_to_play": position.to_play,
        "legal_actions": sorted(position.legal_actions()),
        "black_area": result.black_area,
        "white_area": result.white_area,
        "komi": result.komi,
        "winner": result.winner,
        "margin": result.margin,
        "label": result.label,
    }


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


def _filling_moves(size: int) -> list[int]:
    position = Position.empty(size)
    moves: list[int] = []
    while position.legal_actions() and len(moves) < size * size:
        choices = [action for action in position.legal_actions() if action != position.pass_action]
        if not choices:
            break
        position = position.play(choices[0])
        moves.append(choices[0])
    return moves


def _ko_boards() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the board before and after the capture that opens a ko."""

    def point(row: int, column: int) -> int:
        return row * 5 + column

    board = [0] * 25
    for index in (point(0, 1), point(1, 0), point(2, 1)):
        board[index] = 1
    for index in (point(0, 2), point(1, 1), point(1, 3), point(2, 2)):
        board[index] = -1
    before = tuple(board)
    return before, Position(before, 1, (), 5).play(point(1, 2)).board


def _boards(black: Sequence[int], white: Sequence[int], size: int = 5) -> tuple[int, ...]:
    board = [0] * (size * size)
    for index in black:
        board[index] = 1
    for index in white:
        board[index] = -1
    return tuple(board)


def build_conformance() -> dict[str, Any]:
    """Build the fixture from the Python engine."""
    cases: list[dict[str, Any]] = []
    illegal: list[dict[str, Any]] = []

    for size in BOARD_SIZES:
        cases.append(_case(f"empty {size}x{size}", size))

    for size in BOARD_SIZES:
        for index, length in enumerate(GAME_LENGTHS):
            moves = _random_moves(size, length, RANDOM_SEED + size * 100 + index)
            cases.append(
                _case(f"random {size}x{size} game of {len(moves)} moves", size, moves=moves)
            )

    for size in (5,):
        moves = _filling_moves(size)
        cases.append(_case(f"filled {size}x{size} board", size, moves=moves))

    before, after = _ko_boards()
    cases.append(_case("ko leaves the captured point empty", 5, board=after, history=[before]))

    # Black holds 1, 2, and 3. Point 4 is the last liberty of that chain, so
    # playing it clears every black stone on the board.
    cases.append(
        _case(
            "self capture clears the played chain",
            5,
            board=_boards((1, 2, 3), (0, 6, 7, 8, 9)),
            moves=[4],
        )
    )

    # The centre point is enclosed by black, and every other empty point reaches
    # both colors.
    cases.append(
        _case(
            "enclosed empty point scores for black",
            5,
            board=_boards((2, 6, 8, 12), (0, 3, 4, 5, 9, 10, 13, 14, 15, 19, 20, 24)),
        )
    )

    illegal.append(_illegal("occupied point", 5, moves=[0, 0]))
    illegal.append(_illegal("point past the board", 5, moves=[26]))
    illegal.append(
        _illegal("ko recapture", 5, board=after, to_play=-1, history=[before], moves=[6])
    )
    illegal.append(
        _illegal("self capture that repeats the board", 5, board=_boards((), (1, 5)), moves=[0])
    )

    return {
        "ruleset": RULESET,
        "komi": {str(size): komi_for(size) for size in BOARD_SIZES},
        "sizes": list(BOARD_SIZES),
        "cases": cases,
        "illegal": illegal,
    }


def write_conformance(path: Path) -> dict[str, Any]:
    fixture = build_conformance()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture
