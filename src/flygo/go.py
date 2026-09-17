"""Minimal, immutable 5x5 Go rules for experiments and the web demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BOARD_SIZE = 5
POINTS = BOARD_SIZE * BOARD_SIZE
PASS = POINTS


@dataclass(frozen=True)
class Position:
    board: tuple[int, ...] = (0,) * POINTS
    to_play: int = 1
    previous_board: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.board) != POINTS or any(stone not in {-1, 0, 1} for stone in self.board):
            raise ValueError("A 5x5 board needs 25 values from -1, 0, and 1")
        if self.to_play not in {-1, 1}:
            raise ValueError("to_play must be 1 for black or -1 for white")

    def features(self) -> NDArray[np.float32]:
        values = np.asarray(self.board, dtype=np.int8)
        current = (values == self.to_play).astype(np.float32)
        opponent = (values == -self.to_play).astype(np.float32)
        return np.concatenate([current, opponent, np.asarray([self.to_play], dtype=np.float32)])

    def play(self, action: int) -> Position:
        if action == PASS:
            return Position(self.board, -self.to_play, self.board)
        if not 0 <= action < POINTS:
            raise ValueError("Action must be a board point or pass")
        if self.board[action] != 0:
            raise ValueError("That point is occupied")

        board = list(self.board)
        board[action] = self.to_play
        for neighbor in _neighbors(action):
            if board[neighbor] == -self.to_play:
                group = _group(board, neighbor)
                if not _liberties(board, group):
                    for point in group:
                        board[point] = 0
        own_group = _group(board, action)
        if not _liberties(board, own_group):
            raise ValueError("Suicide is not legal")
        next_board = tuple(board)
        if next_board == self.previous_board:
            raise ValueError("Immediate ko recapture is not legal")
        return Position(next_board, -self.to_play, self.board)

    def legal_actions(self) -> tuple[int, ...]:
        legal = [PASS]
        for action, stone in enumerate(self.board):
            if stone == 0:
                try:
                    self.play(action)
                except ValueError:
                    continue
                legal.append(action)
        return tuple(legal)


def _neighbors(point: int) -> tuple[int, ...]:
    row, column = divmod(point, BOARD_SIZE)
    adjacent: list[int] = []
    if row:
        adjacent.append(point - BOARD_SIZE)
    if row + 1 < BOARD_SIZE:
        adjacent.append(point + BOARD_SIZE)
    if column:
        adjacent.append(point - 1)
    if column + 1 < BOARD_SIZE:
        adjacent.append(point + 1)
    return tuple(adjacent)


def _group(board: list[int], start: int) -> set[int]:
    color = board[start]
    group = {start}
    pending = [start]
    while pending:
        point = pending.pop()
        for neighbor in _neighbors(point):
            if board[neighbor] == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return group


def _liberties(board: list[int], group: set[int]) -> set[int]:
    return {neighbor for point in group for neighbor in _neighbors(point) if board[neighbor] == 0}
