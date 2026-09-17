from pathlib import Path

import pytest
from pydantic import ValidationError

from flygo.api import PositionRequest, TurnRequest, health, index, play_turn, simulate
from flygo.go import BOARD_SIZES


def test_home_page_is_available() -> None:
    response = index()

    assert response.status_code == 200
    assert Path(response.path).name == "index.html"


def test_health_names_official_dataset() -> None:
    assert health() == {"status": "ok", "dataset": "male-cns:v1.0"}


def test_simulation_uses_bundled_official_subgraph() -> None:
    result = simulate(PositionRequest(size=5, board=[0] * 25, to_play=1))

    assert result.size == 5
    assert result.topology.startswith("Official MaleCNS v1.0")
    assert len(result.activity) == 40
    assert result.recommended_action in result.legal_actions


def test_default_board_size_is_nine() -> None:
    result = simulate(PositionRequest(board=[0] * 81, to_play=1))

    assert result.size == 9
    assert result.recommended_action in result.legal_actions


@pytest.mark.parametrize("size", BOARD_SIZES)
def test_every_board_size_can_be_simulated(size: int) -> None:
    result = simulate(PositionRequest(size=size, board=[0] * (size * size), to_play=1))

    assert result.size == size
    assert len(result.activity) == 40
    assert result.recommended_action in result.legal_actions


def test_human_move_gets_automatic_computer_response() -> None:
    result = play_turn(TurnRequest(size=5, board=[0] * 25, action=0))

    assert result.board[0] == 1
    assert result.computer_action is not None
    assert len(result.activity) == 40
    if result.computer_action != 25:
        assert result.board[result.computer_action] == -1


def test_second_consecutive_pass_ends_game_without_computer_move() -> None:
    result = play_turn(TurnRequest(size=5, board=[0] * 25, action=25, consecutive_passes=1))

    assert result.game_over
    assert result.computer_action is None
    assert result.legal_actions == []


def test_board_length_must_match_the_size() -> None:
    with pytest.raises(ValidationError):
        PositionRequest(size=9, board=[0] * 25, to_play=1)


def test_action_must_fit_the_board() -> None:
    with pytest.raises(ValidationError):
        TurnRequest(size=5, board=[0] * 25, action=26)
