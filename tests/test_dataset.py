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


def test_teacher_queries_sample_turns_without_losing_move_history(tmp_path: Path) -> None:
    games = parse_sgf_collection(SGF)
    path = tmp_path / "queries.jsonl"
    write_katago_queries(games, path, visits=256, stride=2)
    query = json.loads(path.read_text())
    assert query["analyzeTurns"] == [0, 2]
    assert len(query["moves"]) == 3
    assert query["komi"] == 7.5
    with pytest.raises(ValueError, match="stride"):
        write_katago_queries(games, path, visits=256, stride=0)


def test_teacher_queries_use_validation_board_komi(tmp_path: Path) -> None:
    games = parse_sgf_collection(b"(;SZ[5]KM[0]RE[B+R];B[aa];W[bb])")
    path = tmp_path / "queries.jsonl"
    write_katago_queries(games, path, visits=16)
    assert json.loads(path.read_text())["komi"] == 0


def test_required_teacher_covers_every_sample_before_writing(tmp_path: Path) -> None:
    (game,) = parse_sgf_collection(SGF)
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        "".join(
            json.dumps(
                {"sample_id": sample_id(game.game_id, turn), "policy": [[action, 1]], "value": 0}
            )
            + "\n"
            for turn, action in ((0, 300), (2, 73))
        )
    )
    output = tmp_path / "dataset"
    result = build_dataset([game], output, stride=2, teacher_path=targets, require_teacher=True)
    assert result.examples == 2
    manifest = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Missing teacher targets for 1 sampled positions"):
        build_dataset([game], output, teacher_path=targets, require_teacher=True)
    assert (output / "manifest.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="Missing teacher targets"):
        build_dataset([game], tmp_path / "missing", require_teacher=True)
    assert not (tmp_path / "missing").exists()
