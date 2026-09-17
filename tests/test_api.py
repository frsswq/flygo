from pathlib import Path

from flygo.api import PositionRequest, health, index, simulate


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
