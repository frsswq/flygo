import random
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from flygo.api import PositionRequest, TurnRequest, health, index, play_turn, simulate
from flygo.go import BOARD_SIZES, Position


def test_home_page_is_available() -> None:
    response = index()

    assert response.status_code == 200
    assert Path(response.path).name == "index.html"


def test_health_names_official_dataset_and_ruleset() -> None:
    assert health() == {
        "status": "ok",
        "dataset": "male-cns:v1.0",
        "ruleset": "Tromp-Taylor, area scoring, positional superko",
    }


def test_simulation_uses_bundled_official_subgraph() -> None:
    result = simulate(PositionRequest(size=5, moves=[]))

    assert result.size == 5
    assert result.to_play == 1
    assert result.topology.startswith("Official MaleCNS v1.0")
    assert len(result.activity) == 40
    assert result.recommended_action in result.legal_actions


def test_default_board_size_is_nine() -> None:
    result = simulate(PositionRequest(moves=[]))

    assert result.size == 9
    assert len(result.legal_actions) == 82
    assert result.recommended_action in result.legal_actions


@pytest.mark.parametrize("size", BOARD_SIZES)
def test_every_board_size_can_be_simulated(size: int) -> None:
    result = simulate(PositionRequest(size=size, moves=[]))

    assert result.size == size
    assert len(result.activity) == 40
    assert result.recommended_action in result.legal_actions


def test_simulation_replays_the_move_list() -> None:
    result = simulate(PositionRequest(size=5, moves=[0, 1, 2]))

    assert result.to_play == -1
    assert set(result.legal_actions).isdisjoint({0, 1, 2})
    assert result.recommended_action not in (0, 1, 2)


def test_simulation_rejects_an_illegal_move_list() -> None:
    with pytest.raises(HTTPException) as error:
        simulate(PositionRequest(size=5, moves=[0, 0]))

    assert error.value.status_code == 422
    assert "occupied" in error.value.detail


def test_human_move_gets_automatic_computer_response() -> None:
    result = play_turn(TurnRequest(size=5, moves=[0]))

    assert result.board[0] == 1
    assert result.moves[0] == 0
    assert result.computer_action is not None
    assert result.moves[-1] == result.computer_action
    assert len(result.activity) == 40
    assert result.to_play == 1
    assert not result.game_over
    assert result.score is None
    if result.computer_action != 25:
        assert result.board[result.computer_action] == -1


def test_second_consecutive_pass_ends_game_without_computer_move() -> None:
    result = play_turn(TurnRequest(size=5, moves=[25, 25]))

    assert result.game_over
    assert result.computer_action is None
    assert result.legal_actions == []
    assert result.moves == [25, 25]
    assert result.consecutive_passes == 2


@pytest.mark.parametrize("moves", ([25, 25, 0], [25, 25, 0, 1]))
def test_a_move_after_the_game_ended_is_rejected(moves: list[int]) -> None:
    with pytest.raises(HTTPException) as error:
        play_turn(TurnRequest(size=5, moves=moves))

    assert error.value.status_code == 422
    assert "already over" in error.value.detail


def test_a_scored_five_by_five_game_is_a_draw_at_zero_komi() -> None:
    result = play_turn(TurnRequest(size=5, moves=[25, 25]))

    assert result.score is not None
    assert (result.score.black_area, result.score.white_area, result.score.komi) == (0, 0, 0)
    assert result.score.label == "Draw"


def test_a_scored_nine_by_nine_game_gives_white_the_komi() -> None:
    result = play_turn(TurnRequest(size=9, moves=[81, 81]))

    assert result.score is not None
    assert result.score.label == "White by 7.5"


def test_a_turn_needs_at_least_one_move() -> None:
    with pytest.raises(ValidationError):
        TurnRequest(size=5, moves=[])


def test_action_must_fit_the_board() -> None:
    with pytest.raises(ValidationError):
        TurnRequest(size=5, moves=[26])


def test_a_legal_game_can_exceed_the_old_request_length_limit() -> None:
    generator = random.Random(3)
    position = Position.empty(5)
    moves: list[int] = []
    for _ in range(117):
        non_pass_actions = [
            action for action in position.legal_actions() if action != position.pass_action
        ]
        action = generator.choice(non_pass_actions)
        moves.append(action)
        position = position.play(action)

    result = simulate(PositionRequest(size=5, moves=moves))

    assert result.to_play == -1
