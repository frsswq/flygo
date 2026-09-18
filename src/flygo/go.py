"""Tromp-Taylor Go rules for 5x5 validation and standard 19x19 Go.

The ruleset follows Tromp-Taylor: area scoring, positional superko, self-capture
is allowed, and two consecutive passes end the game. See
https://tromp.github.io/go.html for the reference statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_BOARD_SIZE = 19
BOARD_SIZES = (5, 19)
RULESET = "Tromp-Taylor, area scoring, positional superko"

# Komi is 0 on 5x5 so that the published solved result (Black wins by 25 under
# area scoring) holds. Standard 19x19 uses 7.5 komi.
KOMI_BY_SIZE: dict[int, float] = {5: 0.0, 19: 7.5}

_MASK64 = 0xFFFFFFFFFFFFFFFF
_MIX_ADD = 0x9E3779B97F4A7C15
_MIX_MUL_A = 0xBF58476D1CE4E5B9
_MIX_MUL_B = 0x94D049BB133111EB


def komi_for(size: int) -> float:
    """Return the komi this project uses for a board size."""
    try:
        return KOMI_BY_SIZE[size]
    except KeyError as error:
        raise ValueError(f"No komi is defined for a {size}x{size} board") from error


def _mix(value: int) -> int:
    """SplitMix64 finalizer, used to spread point values across the key."""
    value = (value + _MIX_ADD) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX_MUL_A) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX_MUL_B) & _MASK64
    return value ^ (value >> 31)


def position_key(board: Sequence[int]) -> int:
    """Return a 64-bit key for a board coloring.

    The positional superko rule compares these keys instead of whole boards. A
    key collision would reject a legal move, which needs roughly 4e9 positions
    before it becomes likely, so it is safe for game-length histories.
    """
    digest = _mix(len(board))
    for point, stone in enumerate(board):
        digest ^= _mix((point << 2) | (stone + 1))
    return digest


@dataclass(frozen=True)
class GameResult:
    """Final area count under Tromp-Taylor scoring."""

    black_area: int
    white_area: int
    komi: float
    winner: int

    @property
    def margin(self) -> float:
        return abs(self.black_area - self.white_area - self.komi)

    @property
    def label(self) -> str:
        if self.winner == 0:
            return "Draw"
        name = "Black" if self.winner == 1 else "White"
        return f"{name} by {self.margin:g}"


@dataclass(frozen=True)
class Position:
    """An immutable position where ``to_play`` is 1 for black and -1 for white.

    ``history`` holds the keys of every board coloring seen before this one, so
    that ``play`` can enforce positional superko. A position built without a
    history only forbids an immediate repetition of its own board.
    """

    board: tuple[int, ...]
    to_play: int = 1
    history: tuple[int, ...] = ()
    size: int = DEFAULT_BOARD_SIZE

    def __post_init__(self) -> None:
        if self.size not in BOARD_SIZES:
            supported = " or ".join(str(size) for size in BOARD_SIZES)
            raise ValueError(f"Board size must be {supported}")
        expected = self.points
        if len(self.board) != expected or any(stone not in {-1, 0, 1} for stone in self.board):
            raise ValueError(
                f"A {self.size}x{self.size} board needs {expected} values from -1, 0, and 1"
            )
        if self.to_play not in {-1, 1}:
            raise ValueError("to_play must be 1 for black or -1 for white")
        if any(not isinstance(key, int) or not 0 <= key <= _MASK64 for key in self.history):
            raise ValueError("history must hold 64-bit keys")

    @classmethod
    def empty(cls, size: int = DEFAULT_BOARD_SIZE, to_play: int = 1) -> Position:
        return cls((0,) * (size * size), to_play, (), size)

    @property
    def points(self) -> int:
        return self.size * self.size

    @property
    def pass_action(self) -> int:
        return self.points

    def features(self) -> NDArray[np.float32]:
        values = np.asarray(self.board, dtype=np.int8)
        current = (values == self.to_play).astype(np.float32)
        opponent = (values == -self.to_play).astype(np.float32)
        return np.concatenate([current, opponent, np.asarray([self.to_play], dtype=np.float32)])

    def play(self, action: int) -> Position:
        """Return the position after ``action``, or raise ``ValueError`` if it is illegal."""
        if action == self.pass_action:
            return Position(self.board, -self.to_play, self.history, self.size)
        if not 0 <= action < self.points:
            raise ValueError("Action must be a board point or pass")
        if self.board[action] != 0:
            raise ValueError("That point is occupied")

        board = list(self.board)
        board[action] = self.to_play
        opponent = -self.to_play
        for neighbor in _neighbors(action, self.size):
            if board[neighbor] == opponent:
                group = _group(board, neighbor, self.size)
                if not _liberties(board, group, self.size):
                    for point in group:
                        board[point] = 0
        own_group = _group(board, action, self.size)
        if not _liberties(board, own_group, self.size):
            for point in own_group:
                board[point] = 0

        next_board = tuple(board)
        next_key = position_key(next_board)
        if next_key in self.history or next_key == position_key(self.board):
            raise ValueError("That move repeats an earlier board position")
        history = (*self.history, position_key(self.board))
        return Position(next_board, opponent, history, self.size)

    def legal_actions(self) -> tuple[int, ...]:
        legal = [self.pass_action]
        for action, stone in enumerate(self.board):
            if stone == 0:
                try:
                    self.play(action)
                except ValueError:
                    continue
                legal.append(action)
        return tuple(legal)

    def result(self) -> GameResult:
        """Score the current board by area, leaving dead stones in place."""
        black_area, white_area = area_score(self.board, self.size)
        komi = komi_for(self.size)
        difference = black_area - white_area - komi
        winner = 0 if difference == 0 else (1 if difference > 0 else -1)
        return GameResult(black_area, white_area, komi, winner)


def area_score(board: Sequence[int], size: int) -> tuple[int, int]:
    """Return the black and white area of a board.

    Area scoring counts a player's stones plus every empty point that reaches
    only that player's stones. Empty regions that touch both colors score for
    neither. Stones stay on the board, so no dead-stone agreement is needed.
    """
    black = sum(1 for stone in board if stone == 1)
    white = sum(1 for stone in board if stone == -1)
    seen = [False] * len(board)
    for start, stone in enumerate(board):
        if stone != 0 or seen[start]:
            continue
        region = 0
        touches_black = False
        touches_white = False
        pending = [start]
        seen[start] = True
        while pending:
            point = pending.pop()
            region += 1
            for neighbor in _neighbors(point, size):
                value = board[neighbor]
                if value == 0:
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        pending.append(neighbor)
                elif value == 1:
                    touches_black = True
                else:
                    touches_white = True
        if touches_black and not touches_white:
            black += region
        elif touches_white and not touches_black:
            white += region
    return black, white


def _neighbors(point: int, size: int) -> tuple[int, ...]:
    row, column = divmod(point, size)
    adjacent: list[int] = []
    if row:
        adjacent.append(point - size)
    if row + 1 < size:
        adjacent.append(point + size)
    if column:
        adjacent.append(point - 1)
    if column + 1 < size:
        adjacent.append(point + 1)
    return tuple(adjacent)


def _group(board: list[int], start: int, size: int) -> set[int]:
    color = board[start]
    group = {start}
    pending = [start]
    while pending:
        point = pending.pop()
        for neighbor in _neighbors(point, size):
            if board[neighbor] == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return group


def _liberties(board: list[int], group: set[int], size: int) -> set[int]:
    return {
        neighbor for point in group for neighbor in _neighbors(point, size) if board[neighbor] == 0
    }
