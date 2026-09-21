import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import flygo.dataset as dataset_module
from flygo.dataset import (
    SPLITS,
    build_dataset,
    load_sgf_games,
    load_teacher_targets,
    parse_sgf_collection,
    published_split_paths,
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


def test_sgf_rejects_a_point_outside_the_board() -> None:
    with pytest.raises(ValueError, match="invalid move at node 2"):
        parse_sgf_collection(b"(;FF[4]GM[1]SZ[5]KM[0]RE[B+R];B[gg])")

    with pytest.raises(ValueError, match="invalid move at node 3"):
        parse_sgf_collection(b"(;FF[4]GM[1]SZ[5]KM[0]RE[B+R];B[aa];W[zz])")


@pytest.mark.parametrize(
    ("body", "node_number"),
    [
        (b";B[aa];AB[bb];W[cc]", 3),
        (b";B[aa];AW[bb];W[cc]", 3),
        (b";B[aa];AE[bb];W[cc]", 3),
        (b";B[aa]AB[bb];W[cc]", 2),
        (b";B[aa]AW[bb];W[cc]", 2),
    ],
)
def test_sgf_rejects_setup_stones_after_the_root(body: bytes, node_number: int) -> None:
    payload = b"(;FF[4]GM[1]SZ[5]KM[0]RE[B+R]" + body + b")"

    with pytest.raises(ValueError, match=f"setup stones at node {node_number}"):
        parse_sgf_collection(payload)


def test_sgf_rejects_root_setup_stones() -> None:
    with pytest.raises(ValueError, match="setup stones at node 1"):
        parse_sgf_collection(b"(;FF[4]GM[1]SZ[5]KM[0]RE[B+R]AB[bb];W[cc])")


def test_sgf_without_setup_stones_is_unchanged() -> None:
    (game,) = parse_sgf_collection(b"(;FF[4]GM[1]SZ[5]KM[0]RE[B+R];B[aa];W[cc];B[dd])")

    assert game.moves == (20, 12, 8)


def test_build_dataset_rejects_setup_stones_before_publishing(tmp_path: Path) -> None:
    sgf_path = tmp_path / "setup.sgf"
    sgf_path.write_bytes(b"(;FF[4]GM[1]SZ[5]KM[0]RE[B+R];B[aa];AW[bb];W[cc];B[dd])")
    output = tmp_path / "dataset"

    with pytest.raises(ValueError, match="setup stones at node 3"):
        build_dataset(load_sgf_games([sgf_path]), output, size=5)

    assert not output.exists()


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


def test_teacher_import_uses_final_responses_not_interim_updates(tmp_path: Path) -> None:
    (game,) = parse_sgf_collection(SGF)
    analysis_path = tmp_path / "analysis.jsonl"
    analysis_path.write_text(
        "".join(
            json.dumps(response) + "\n"
            for response in (
                {
                    "id": game.game_id,
                    "turnNumber": 0,
                    "isDuringSearch": True,
                    "rootInfo": {"winrate": 0.25},
                    "moveInfos": [{"move": "pass", "visits": 1}],
                },
                {
                    "id": game.game_id,
                    "turnNumber": 0,
                    "isDuringSearch": False,
                    "rootInfo": {"winrate": 0.75},
                    "moveInfos": [{"move": "Q4", "visits": 64}, {"move": "D16", "visits": 32}],
                },
                {
                    "id": game.game_id,
                    "turnNumber": 1,
                    "isDuringSearch": True,
                    "rootInfo": {"winrate": 0.25},
                    "moveInfos": [{"move": "pass", "visits": 1}],
                },
                {
                    "id": game.game_id,
                    "turnNumber": 1,
                    "rootInfo": {"winrate": 0.75},
                    "moveInfos": [{"move": "D4", "visits": 64}],
                },
            )
        )
    )
    teacher_path = tmp_path / "teacher.jsonl"

    assert import_katago_analysis(analysis_path, teacher_path, [game]) == 2

    targets = load_teacher_targets(teacher_path)
    first = targets[sample_id(game.game_id, 0)]
    assert first.policy == {300: 2 / 3, 60: 1 / 3}
    assert first.value == 0.5
    assert targets[sample_id(game.game_id, 1)].value == -0.5


def test_teacher_import_rejects_provisional_only_analysis(tmp_path: Path) -> None:
    (game,) = parse_sgf_collection(SGF)
    analysis_path = tmp_path / "analysis.jsonl"
    analysis_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": game.game_id,
                    "turnNumber": turn,
                    "isDuringSearch": True,
                    "rootInfo": {"winrate": 0.75},
                    "moveInfos": [{"move": "pass", "visits": 1}],
                }
            )
            + "\n"
            for turn in (0, 1)
        )
    )
    teacher_path = tmp_path / "teacher.jsonl"

    with pytest.raises(ValueError, match="only provisional responses"):
        import_katago_analysis(analysis_path, teacher_path, [game])

    assert not teacher_path.exists()


def test_teacher_import_rejects_duplicate_final_responses(tmp_path: Path) -> None:
    (game,) = parse_sgf_collection(SGF)
    response = {
        "id": game.game_id,
        "turnNumber": 0,
        "isDuringSearch": False,
        "rootInfo": {"winrate": 0.75},
        "moveInfos": [{"move": "Q4", "visits": 64}],
    }
    analysis_path = tmp_path / "analysis.jsonl"
    analysis_path.write_text(f"{json.dumps(response)}\n{json.dumps(response)}\n")

    with pytest.raises(ValueError, match="Duplicate final KataGo response"):
        import_katago_analysis(analysis_path, tmp_path / "teacher.jsonl", [game])


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


def test_published_split_paths_verify_a_published_dataset(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    build_dataset(parse_sgf_collection(SGF), output)

    paths = published_split_paths(output)

    assert set(paths) == set(SPLITS)
    assert paths["train"] == output / "train.npz"


def test_published_split_paths_reject_an_unpublished_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not published"):
        published_split_paths(tmp_path / "missing")


def test_published_split_paths_reject_a_damaged_manifest(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    build_dataset(parse_sgf_collection(SGF), output)
    (output / "manifest.json").write_text("{ not json")

    with pytest.raises(ValueError, match="unreadable manifest"):
        published_split_paths(output)


def test_published_split_paths_reject_a_changed_split(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    build_dataset(parse_sgf_collection(SGF), output)
    with (output / "train.npz").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="train split hash does not match"):
        published_split_paths(output)


def test_build_dataset_refuses_to_replace_a_published_generation(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    first = parse_sgf_collection(SGF)[0]
    second = parse_sgf_collection(SGF.replace(b"RE[B+R]", b"RE[W+R]"))[0]
    build_dataset([first], output)
    published = {path.name: path.read_bytes() for path in output.iterdir()}
    before = published_split_paths(output)

    with pytest.raises(ValueError, match="already published"):
        build_dataset([second], output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == published
    assert published_split_paths(output) == before


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_build_dataset_split_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    output = tmp_path / "dataset"
    game = parse_sgf_collection(SGF)[0]
    original = dataset_module._write_npz
    written = 0

    def failing(path: Path, rows: Sequence[tuple[Any, ...]], size: int) -> None:
        nonlocal written
        if written == fail_at:
            raise OSError("injected split write failure")
        written += 1
        original(path, rows, size)

    monkeypatch.setattr(dataset_module, "_write_npz", failing)
    with pytest.raises(OSError, match="injected split write failure"):
        build_dataset([game], output)
    monkeypatch.setattr(dataset_module, "_write_npz", original)

    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError, match="not published"):
        published_split_paths(output)

    build_dataset([game], output)
    clean = tmp_path / "clean"
    build_dataset([game], clean)

    assert (output / "manifest.json").read_bytes() == (clean / "manifest.json").read_bytes()
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == {
        path.name: path.read_bytes() for path in clean.iterdir() if path.is_file()
    }


def test_build_dataset_manifest_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dataset"
    game = parse_sgf_collection(SGF)[0]

    def failing(_output: Path, _manifest: object) -> None:
        raise OSError("injected manifest failure")

    monkeypatch.setattr(dataset_module, "_write_manifest", failing)
    with pytest.raises(OSError, match="injected manifest failure"):
        build_dataset([game], output)

    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError, match="not published"):
        published_split_paths(output)
