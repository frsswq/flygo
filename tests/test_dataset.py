import json
from pathlib import Path

import numpy as np
import pytest

from flygo.dataset import (
    build_dataset,
    load_sgf_games,
    load_teacher_targets,
    parse_sgf_collection,
    sample_id,
    split_of,
)
from flygo.teacher import (
    action_to_gtp,
    gtp_to_action,
    import_katago_analysis,
    write_katago_queries,
)

SGF = b"(;FF[4]GM[1]CA[UTF-8]SZ[19]KM[7.5]RE[B+R];B[pd];W[dd];B[qp])"


def test_sgf_collection_is_validated_and_replayed() -> None:
    (game,) = parse_sgf_collection(SGF)

    assert game.size == 19
    assert game.winner == 1
    assert game.moves == (300, 288, 73)
    assert split_of(game.game_id) in {"train", "validation", "test"}


def test_sgf_rejects_unsupported_or_unscored_games() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_sgf_collection(b"(;SZ[9]RE[B+R];B[aa])")
    with pytest.raises(ValueError, match="winner"):
        parse_sgf_collection(b"(;SZ[19]KM[7.5];B[aa])")


def test_dataset_keeps_complete_games_in_one_split(tmp_path: Path) -> None:
    sgf_path = tmp_path / "game.sgf"
    sgf_path.write_bytes(SGF)
    games = load_sgf_games([sgf_path])
    output = tmp_path / "dataset"

    summary = build_dataset(games, output)

    assert summary.accepted_games == 1
    assert summary.examples == 3
    arrays = [np.load(path) for path in output.glob("*.npz")]
    populated = [array for array in arrays if array["features"].shape[0] > 0]
    assert len(populated) == 1
    assert populated[0]["features"].shape == (3, 723)
    assert populated[0]["policy"].shape == (3, 362)
    assert populated[0]["value"].tolist() == [1, -1, 1]


def test_dataset_deduplicates_positions_across_complete_games(tmp_path: Path) -> None:
    first = parse_sgf_collection(SGF)[0]
    second = parse_sgf_collection(SGF.replace(b"RE[B+R]", b"RE[B+R]C[copy]"))[0]

    summary = build_dataset([first, second], tmp_path / "dataset")

    assert summary.accepted_games == 2
    assert summary.examples == 3
    assert summary.duplicate_examples == 3


def test_teacher_targets_replace_human_policy_and_value(tmp_path: Path) -> None:
    (game,) = parse_sgf_collection(SGF)
    identifier = sample_id(game.game_id, 0)
    teacher_path = tmp_path / "teacher.jsonl"
    teacher_path.write_text(
        json.dumps({"sample_id": identifier, "policy": [[300, 3], [0, 1]], "value": 0.25}) + "\n"
    )

    target = load_teacher_targets(teacher_path)[identifier]

    assert target.policy == {300: 0.75, 0: 0.25}
    assert target.value == 0.25


def test_katago_protocol_round_trip(tmp_path: Path) -> None:
    (game,) = parse_sgf_collection(SGF)
    assert action_to_gtp(300, 19) == "Q4"
    assert gtp_to_action("Q4", 19) == 300
    query_path = tmp_path / "queries.jsonl"
    assert write_katago_queries([game], query_path, visits=64) == 1
    query = json.loads(query_path.read_text())
    assert query["moves"][0] == ["B", "Q4"]
    assert query["maxVisits"] == 64

    analysis_path = tmp_path / "analysis.jsonl"
    responses = [
        {
            "id": game.game_id,
            "turnNumber": 0,
            "rootInfo": {"winrate": 0.75},
            "moveInfos": [
                {"move": "Q4", "visits": 48},
                {"move": "D16", "visits": 16},
            ],
        },
        {
            "id": game.game_id,
            "turnNumber": 1,
            "rootInfo": {"winrate": 0.75},
            "moveInfos": [{"move": "D4", "visits": 64}],
        },
    ]
    analysis_path.write_text("".join(json.dumps(response) + "\n" for response in responses))
    teacher_path = tmp_path / "teacher.jsonl"

    assert import_katago_analysis(analysis_path, teacher_path, [game]) == 2
    targets = load_teacher_targets(teacher_path)
    target = targets[sample_id(game.game_id, 0)]
    assert target.policy == {300: 0.75, 60: 0.25}
    assert target.value == 0.5
    assert targets[sample_id(game.game_id, 1)].value == -0.5
