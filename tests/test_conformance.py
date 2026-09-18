"""The committed fixture and the Python engine must agree, case for case."""

import json

import pytest

from flygo.conformance import (
    DEFAULT_CONFORMANCE,
    build_conformance,
    decode_board,
    encode_board,
)
from flygo.go import RULESET, Position, komi_for, position_key


def load_fixture() -> dict:
    return json.loads(DEFAULT_CONFORMANCE.read_text())


def replay(case: dict) -> Position:
    position = Position(
        decode_board(case["board"]),
        case["to_play"],
        tuple(position_key(decode_board(board)) for board in case["history"]),
        case["size"],
    )
    for action in case["moves"]:
        position = position.play(action)
    return position


def test_committed_fixture_matches_the_engine() -> None:
    assert load_fixture() == build_conformance()


def test_fixture_names_the_ruleset_and_komi() -> None:
    fixture = load_fixture()

    assert fixture["ruleset"] == RULESET
    assert fixture["komi"] == {str(size): komi_for(size) for size in fixture["sizes"]}


def test_every_case_replays_to_the_recorded_state() -> None:
    for case in load_fixture()["cases"]:
        position = replay(case)
        result = position.result()

        assert encode_board(position.board) == case["final_board"], case["name"]
        assert position.to_play == case["final_to_play"], case["name"]
        assert sorted(position.legal_actions()) == case["legal_actions"], case["name"]
        assert result.black_area == case["black_area"], case["name"]
        assert result.white_area == case["white_area"], case["name"]
        assert result.winner == case["winner"], case["name"]
        assert result.margin == case["margin"], case["name"]
        assert result.label == case["label"], case["name"]


def test_every_illegal_case_raises_the_recorded_error() -> None:
    for case in load_fixture()["illegal"]:
        position = Position(
            decode_board(case["board"]),
            case["to_play"],
            tuple(position_key(decode_board(board)) for board in case["history"]),
            case["size"],
        )
        with pytest.raises(ValueError, match=case["error"]) as error:
            for action in case["moves"]:
                position = position.play(action)
        assert str(error.value) == case["error"], case["name"]


def test_fixture_covers_the_hard_rules() -> None:
    fixture = load_fixture()
    names = {case["name"] for case in fixture["cases"]}
    illegal = {case["name"]: case["error"] for case in fixture["illegal"]}

    assert "self capture clears the played chain" in names
    assert "enclosed empty point scores for black" in names
    assert illegal["ko recapture"] == "That move repeats an earlier board position"
    assert (
        illegal["self capture that repeats the board"]
        == "That move repeats an earlier board position"
    )
    assert len(fixture["cases"]) >= 14


def test_board_encoding_round_trips() -> None:
    for board in ((0,) * 25, (1, -1, 0) + (0,) * 22):
        assert decode_board(encode_board(board)) == board


def test_board_encoding_rejects_unknown_symbols() -> None:
    with pytest.raises(ValueError, match="Unknown board symbol"):
        decode_board("x")
