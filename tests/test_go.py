import pytest

from flygo.go import PASS, Position


def test_empty_board_has_every_action() -> None:
    assert set(Position().legal_actions()) == set(range(PASS + 1))


def test_play_captures_surrounded_stone() -> None:
    board = [0] * 25
    board[1] = board[5] = board[11] = 1
    board[6] = -1
    position = Position(tuple(board), 1)

    result = position.play(7)

    assert result.board[6] == 0
    assert result.board[7] == 1
    assert result.to_play == -1


def test_occupied_point_is_illegal() -> None:
    position = Position((1,) + (0,) * 24, -1)

    with pytest.raises(ValueError, match="occupied"):
        position.play(0)
