import polars as pl
import pytest

from flygo.arena import (
    Agent,
    GameRecord,
    elo_with_confidence,
    fit_elo,
    paired_tournament,
    play_game,
)
from flygo.connectome import from_frame
from flygo.go import Position
from flygo.model import ConnectomePolicy
from flygo.search import search


def policy() -> ConnectomePolicy:
    connectome = from_frame(pl.DataFrame({"pre": [1], "post": [2], "weight": [1]}))
    return ConnectomePolicy.initialize(connectome, size=5, seed=3)


def passing_agent(name: str) -> Agent:
    return Agent(name, lambda position, _passes: position.pass_action)


def first_legal_agent(name: str) -> Agent:
    return Agent(name, lambda position, _passes: position.legal_actions()[1])


def game(black: str, white: str, winner: int, paired_game: int) -> GameRecord:
    return GameRecord(black, white, winner, (), 0, paired_game, False)


def test_mcts_returns_a_visited_legal_action() -> None:
    position = Position.empty(5)

    result = search(position, policy(), time_limit=None, max_simulations=8)

    assert result.action in position.legal_actions()
    assert result.simulations == 8
    assert result.elapsed_seconds >= 0
    assert -1 <= result.root_value <= 1
    assert sum(result.visits.values()) == 7
    assert result.visits[result.action] > 0


def test_mcts_rejects_a_finished_game() -> None:
    position = Position.empty(5).play(25).play(25)

    try:
        search(position, policy(), consecutive_passes=2, max_simulations=1)
    except ValueError as error:
        assert "finished" in str(error)
    else:
        raise AssertionError("search accepted a finished game")


def test_game_ends_after_two_passes_without_adjudication() -> None:
    game = play_game(passing_agent("black"), passing_agent("white"), size=5)

    assert game.moves == (25, 25)
    assert game.winner == 0
    assert not game.adjudicated


def test_paired_tournament_swaps_colors() -> None:
    games = paired_tournament(
        [passing_agent("pass"), first_legal_agent("play")],
        size=5,
        openings=[[]],
        rounds=1,
        max_moves=4,
    )

    assert [(game.black, game.white) for game in games] == [
        ("pass", "play"),
        ("play", "pass"),
    ]
    assert {game.paired_game for game in games} == {0}


def test_elo_rewards_the_agent_that_wins_every_game() -> None:
    games = [
        play_game(
            first_legal_agent("strong"),
            passing_agent("weak"),
            size=5,
            paired_game=index,
            max_moves=3,
        )
        for index in range(8)
    ]

    ratings = fit_elo(games, anchor="weak")
    confidence = elo_with_confidence(games, anchor="weak", bootstrap_samples=20, seed=1)

    assert ratings["strong"] > ratings["weak"]
    assert confidence["strong"].elo > 0
    assert confidence["strong"].lower <= confidence["strong"].elo
    assert confidence["strong"].upper >= confidence["strong"].elo


def test_elo_bootstrap_keeps_every_matchup_and_fixed_anchor() -> None:
    games = [
        game("random", "greedy", 2, 0),
        game("greedy", "random", 1, 0),
        game("random", "mcts", 2, 1),
        game("mcts", "random", 1, 1),
        game("greedy", "mcts", 2, 2),
        game("mcts", "greedy", 1, 2),
    ]

    for seed in range(10):
        ratings = elo_with_confidence(games, anchor="random", bootstrap_samples=20, seed=seed)
        assert set(ratings) == {"random", "greedy", "mcts"}
        assert ratings["random"].elo == 0
        assert ratings["random"].lower == 0
        assert ratings["random"].upper == 0

    default = elo_with_confidence(games, bootstrap_samples=20, seed=0)
    assert default["greedy"].elo == 0
    assert default["greedy"].lower == 0
    assert default["greedy"].upper == 0


def test_elo_rejects_disconnected_comparison_graph() -> None:
    games = [game("anchor", "peer", 1, 0), game("left", "right", 1, 1)]

    with pytest.raises(ValueError, match="disconnected"):
        fit_elo(games, anchor="anchor")
