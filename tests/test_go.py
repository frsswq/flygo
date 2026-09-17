"""Rules for square boards from 5x5 through 9x9."""

import pytest

from flygo.go import BOARD_SIZES, Position


@pytest.mark.parametrize("size", BOARD_SIZES)
def test_empty_board_allows_every_action(size: int) -> None:
    position = Position.empty(size)

    assert set(position.legal_actions()) == set(range(size * size + 1))
    assert position.features().shape == (2 * size * size + 1,)
    assert position.pass_action == size * size


def test_play_captures_surrounded_stone() -> None:
    board = [0] * 25
    board[1] = board[5] = board[11] = 1
    board[6] = -1
    position = Position(tuple(board), 1)

    result = position.play(7)

    assert result.board[6] == 0
    assert result.board[7] == 1
    assert result.to_play == -1


def test_capture_keeps_the_board_size() -> None:
    size = 9
    board = [0] * (size * size)
    board[40] = -1
    board[31] = board[49] = board[41] = 1
    position = Position(tuple(board), 1, None, size)

    result = position.play(39)

    assert result.board[40] == 0
    assert result.board[39] == 1
    assert result.size == size


def test_occupied_point_is_illegal() -> None:
    position = Position((1,) + (0,) * 24, -1)

    with pytest.raises(ValueError, match="occupied"):
        position.play(0)


def test_board_size_is_validated() -> None:
    with pytest.raises(ValueError, match="between"):
        Position.empty(4)


def test_board_must_match_the_size() -> None:
    with pytest.raises(ValueError, match="81 values"):
        Position((0,) * 25, 1, None, 9)
