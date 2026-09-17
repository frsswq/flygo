"""Minimal, immutable Go rules for square boards from 5x5 through 9x9."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_BOARD_SIZE = 5
MAX_BOARD_SIZE = 9
DEFAULT_BOARD_SIZE = 9
BOARD_SIZES = tuple(range(MIN_BOARD_SIZE, MAX_BOARD_SIZE + 1))


@dataclass(frozen=True)
class Position:
    """An immutable position where ``to_play`` is 1 for black and -1 for white."""

    board: tuple[int, ...]
    to_play: int = 1
    previous_board: tuple[int, ...] | None = None
    size: int = DEFAULT_BOARD_SIZE

    def __post_init__(self) -> None:
        if not MIN_BOARD_SIZE <= self.size <= MAX_BOARD_SIZE:
            raise ValueError(f"Board size must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}")
        expected = self.points
        if len(self.board) != expected or any(stone not in {-1, 0, 1} for stone in self.board):
            raise ValueError(
                f"A {self.size}x{self.size} board needs {expected} values from -1, 0, and 1"
            )
        if self.to_play not in {-1, 1}:
            raise ValueError("to_play must be 1 for black or -1 for white")
        if self.previous_board is not None and len(self.previous_board) != expected:
            raise ValueError("previous_board must match the board size")

    @classmethod
    def empty(cls, size: int = DEFAULT_BOARD_SIZE, to_play: int = 1) -> Position:
        return cls((0,) * (size * size), to_play, None, size)

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
        if action == self.pass_action:
            return Position(self.board, -self.to_play, self.board, self.size)
        if not 0 <= action < self.points:
            raise ValueError("Action must be a board point or pass")
        if self.board[action] != 0:
            raise ValueError("That point is occupied")

        board = list(self.board)
        board[action] = self.to_play
        for neighbor in _neighbors(action, self.size):
            if board[neighbor] == -self.to_play:
                group = _group(board, neighbor, self.size)
                if not _liberties(board, group, self.size):
                    for point in group:
                        board[point] = 0
        own_group = _group(board, action, self.size)
        if not _liberties(board, own_group, self.size):
            raise ValueError("Suicide is not legal")
        next_board = tuple(board)
        if next_board == self.previous_board:
            raise ValueError("Immediate ko recapture is not legal")
        return Position(next_board, -self.to_play, self.board, self.size)

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
