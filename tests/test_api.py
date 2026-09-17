from pathlib import Path

from flygo.api import PositionRequest, TurnRequest, health, index, play_turn, simulate
from flygo.go import PASS


def test_home_page_is_available() -> None:
    response = index()

    assert response.status_code == 200
    assert Path(response.path).name == "index.html"


def test_health_names_official_dataset() -> None:
    assert health() == {"status": "ok", "dataset": "male-cns:v1.0"}


def test_simulation_uses_bundled_official_subgraph() -> None:
    result = simulate(PositionRequest(board=[0] * 25, to_play=1))

    assert result.topology.startswith("Official MaleCNS v1.0")
    assert len(result.activity) == 40
    assert result.recommended_action in result.legal_actions


def test_human_move_gets_automatic_computer_response() -> None:
    result = play_turn(TurnRequest(board=[0] * 25, action=0))

    assert result.board[0] == 1
    assert result.computer_action is not None
    assert len(result.activity) == 40
    if result.computer_action != PASS:
        assert result.board[result.computer_action] == -1


def test_second_consecutive_pass_ends_game_without_computer_move() -> None:
    result = play_turn(TurnRequest(board=[0] * 25, action=PASS, consecutive_passes=1))

    assert result.game_over
    assert result.computer_action is None
    assert result.legal_actions == []
