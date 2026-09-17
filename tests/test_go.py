"""Tromp-Taylor rules for square boards from 5x5 through 9x9."""

import pytest

from flygo.go import (
    BOARD_SIZES,
    DEFAULT_BOARD_SIZE,
    KOMI_BY_SIZE,
    Position,
    area_score,
    komi_for,
    position_key,
)


def point(row: int, column: int, size: int = 5) -> int:
    return row * size + column


def board_from(black: set[int], white: set[int], size: int = 5) -> tuple[int, ...]:
    board = [0] * (size * size)
    for index in black:
        board[index] = 1
    for index in white:
        board[index] = -1
    return tuple(board)


@pytest.mark.parametrize("size", BOARD_SIZES)
def test_empty_board_allows_every_action(size: int) -> None:
    position = Position.empty(size)

    assert set(position.legal_actions()) == set(range(size * size + 1))
    assert position.features().shape == (2 * size * size + 1,)
    assert position.pass_action == size * size


def test_default_board_size_is_nine() -> None:
    assert DEFAULT_BOARD_SIZE == 9
    assert Position.empty().points == 81


def test_komi_covers_every_supported_size() -> None:
    assert set(KOMI_BY_SIZE) == set(BOARD_SIZES)
    assert komi_for(5) == 0
    assert komi_for(9) == 7.5


def test_komi_rejects_an_unsupported_size() -> None:
    with pytest.raises(ValueError, match="komi"):
        komi_for(19)


def test_play_captures_surrounded_stone() -> None:
    board = board_from({1, 5, 11}, {6})
    position = Position(board, 1, (), 5)

    result = position.play(7)

    assert result.board[6] == 0
    assert result.board[7] == 1
    assert result.to_play == -1


def test_occupied_point_is_illegal() -> None:
    position = Position(board_from({0}, set()), -1, (), 5)

    with pytest.raises(ValueError, match="occupied"):
        position.play(0)


def test_self_capture_clears_the_players_own_chain() -> None:
    # Black holds 1, 2, and 3. Point 4 is the only liberty of that chain, and it
    # is also surrounded, so playing it removes every black stone on the board.
    board = board_from({1, 2, 3}, {0, 6, 7, 8, 9})
    position = Position(board, 1, (), 5)

    result = position.play(4)

    assert all(result.board[index] == 0 for index in (1, 2, 3, 4))
    assert result.to_play == -1


def test_self_capture_that_repeats_the_board_is_illegal() -> None:
    # White surrounds point 0, so a black stone there has no liberty and is
    # removed at once, which would leave the board unchanged.
    position = Position(board_from(set(), {1, 5}), 1, (), 5)

    with pytest.raises(ValueError, match="repeats"):
        position.play(0)


def test_ko_recapture_is_illegal() -> None:
    black = {point(0, 1), point(1, 0), point(2, 1)}
    white = {point(0, 2), point(1, 1), point(1, 3), point(2, 2)}
    position = Position(board_from(black, white), 1, (), 5)

    after_capture = position.play(point(1, 2))

    assert after_capture.board[point(1, 1)] == 0
    with pytest.raises(ValueError, match="repeats"):
        after_capture.play(point(1, 1))


def test_history_records_every_earlier_board() -> None:
    position = Position.empty(5)

    assert position.history == ()
    after_one = position.play(0)
    assert len(after_one.history) == 1
    after_two = after_one.play(1)
    assert len(after_two.history) == 2
    assert after_two.history[0] == position_key(position.board)


def test_pass_keeps_the_board_and_swaps_the_turn() -> None:
    position = Position.empty(5)

    result = position.play(position.pass_action)

    assert result.board == position.board
    assert result.to_play == -1
    assert result.history == position.history


def test_position_key_is_stable_and_sensitive() -> None:
    empty = (0,) * 25

    assert position_key(empty) == position_key(empty)
    assert position_key(empty) != position_key((1,) + (0,) * 24)
    assert 0 <= position_key(empty) < 2**64


def test_area_score_of_an_empty_board_is_neutral() -> None:
    assert area_score((0,) * 25, 5) == (0, 0)


def test_area_score_of_a_full_board_goes_to_one_color() -> None:
    assert area_score((1,) * 25, 5) == (25, 0)
    assert area_score((-1,) * 25, 5) == (0, 25)


def test_area_score_splits_a_board_between_colors() -> None:
    black = {point(row, 0) for row in range(5)}
    white = {point(row, 4) for row in range(5)}

    assert area_score(board_from(black, white), 5) == (5, 5)


def test_area_score_counts_enclosed_empty_points() -> None:
    # Black encircles the centre point 7. Every other empty point reaches both
    # colors, so only point 7 is added to the four black stones.
    black = {2, 6, 8, 12}
    white = {0, 3, 4, 5, 9, 10, 13, 14, 15, 19, 20, 24}

    assert area_score(board_from(black, white), 5) == (5, 12)


def test_area_score_leaves_dead_stones_in_place() -> None:
    black = set(range(1, 25))
    white = {0}

    assert area_score(board_from(black, white), 5) == (24, 1)


def test_result_reports_a_draw_on_an_empty_five_by_five_board() -> None:
    result = Position.empty(5).result()

    assert (result.black_area, result.white_area, result.komi) == (0, 0, 0)
    assert result.winner == 0
    assert result.label == "Draw"


def test_result_gives_white_the_komi_on_a_larger_empty_board() -> None:
    result = Position.empty(9).result()

    assert result.winner == -1
    assert result.margin == 7.5
    assert result.label == "White by 7.5"


def test_result_labels_the_solved_margin_on_a_full_five_by_five_board() -> None:
    result = Position((1,) * 25, -1, (), 5).result()

    assert result.winner == 1
    assert result.label == "Black by 25"


def test_board_size_is_validated() -> None:
    with pytest.raises(ValueError, match="between"):
        Position.empty(4)


def test_board_must_match_the_size() -> None:
    with pytest.raises(ValueError, match="81 values"):
        Position((0,) * 25, 1, (), 9)


def test_history_must_hold_keys() -> None:
    with pytest.raises(ValueError, match="history"):
        Position((0,) * 25, 1, (-1,), 5)
