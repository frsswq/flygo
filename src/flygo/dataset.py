"""Build versioned policy-value datasets from complete SGF games."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sgfmill import sgf, sgf_grammar

from flygo.atomic import write_bytes, write_text
from flygo.go import BOARD_SIZES, RULESET, Position, komi_for

DATASET_VERSION = 1
SPLITS = ("train", "validation", "test")
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class GameRecord:
    """One validated main-line game in FlyGo action coordinates."""

    game_id: str
    moves: tuple[int, ...]
    size: int
    winner: int


@dataclass(frozen=True)
class TeacherTarget:
    """A soft policy and value supplied by a versioned external teacher."""

    policy: Mapping[int, float]
    value: float


@dataclass(frozen=True)
class DatasetSummary:
    """Counts and hashes written to a dataset manifest."""

    accepted_games: int
    duplicate_examples: int
    examples: int
    rejected_games: int


def _game_id(payload: bytes, index: int) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}:{index}"


def _action_of(move: tuple[int, int] | None, size: int) -> int:
    if move is None:
        return size * size
    row, column = move
    return row * size + column


def parse_sgf_collection(payload: bytes) -> tuple[GameRecord, ...]:
    """Parse and validate every main-line game in an SGF collection."""
    records: list[GameRecord] = []
    for index, coarse_game in enumerate(sgf_grammar.parse_sgf_collection(payload)):
        game = sgf.Sgf_game.from_coarse_game_tree(coarse_game)
        size = game.get_size()
        if size not in BOARD_SIZES:
            raise ValueError(f"Game {index} uses unsupported board size {size}")
        if game.get_komi() != komi_for(size):
            raise ValueError(f"Game {index} does not use {komi_for(size):g} komi")
        winner_name = game.get_winner()
        if winner_name is None:
            raise ValueError(f"Game {index} has no black or white winner")
        winner = 1 if winner_name == "b" else -1
        position = Position.empty(size)
        moves: list[int] = []
        consecutive_passes = 0
        for node_number, node in enumerate(game.get_main_sequence(), start=1):
            if node.has_setup_stones():
                raise ValueError(
                    f"Game {index} uses setup stones at node {node_number}; "
                    "FlyGo replays move-only main lines"
                )
            try:
                colour, move = node.get_move()
            except ValueError as error:
                raise ValueError(
                    f"Game {index} has an invalid move at node {node_number}"
                ) from error
            if colour is None:
                continue
            expected = "b" if position.to_play == 1 else "w"
            if colour != expected:
                raise ValueError(f"Game {index} has {colour} playing when {expected} is next")
            if consecutive_passes >= 2:
                raise ValueError(f"Game {index} continues after two passes")
            action = _action_of(move, size)
            position = position.play(action)
            moves.append(action)
            consecutive_passes = consecutive_passes + 1 if action == position.pass_action else 0
        if not moves:
            raise ValueError(f"Game {index} has no moves")
        records.append(GameRecord(_game_id(payload, index), tuple(moves), size, winner))
    if not records:
        raise ValueError("The SGF collection contains no games")
    return tuple(records)


def load_sgf_games(paths: Iterable[Path]) -> tuple[GameRecord, ...]:
    """Load SGF files in stable path order."""
    games: list[GameRecord] = []
    for path in sorted(paths):
        games.extend(parse_sgf_collection(path.read_bytes()))
    return tuple(games)


def split_of(game_id: str) -> str:
    """Assign a complete game to a stable 80/10/10 split."""
    bucket = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def sample_id(game_id: str, move_number: int) -> str:
    return f"{game_id}:{move_number}"


def published_split_paths(dataset: Path) -> dict[str, Path]:
    """Resolve a published dataset's split files and verify them against its manifest.

    The manifest is the publication record.
    A dataset directory without one is an incomplete build, not readable evidence.
    """
    manifest_path = dataset / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"Dataset {dataset} has no manifest and is not published")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Dataset {dataset} has an unreadable manifest") from error
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Dataset {dataset} manifest has no split files")
    paths: dict[str, Path] = {}
    for split, entry in files.items():
        try:
            name = entry["file"]
            expected = entry["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Dataset {dataset} has an invalid {split} split entry") from error
        path = dataset / name
        if not path.is_file():
            raise ValueError(f"Dataset {dataset} {split} split file is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset {dataset} {split} split hash does not match the manifest")
        paths[split] = path
    return paths


def load_teacher_targets(path: Path | None) -> dict[str, TeacherTarget]:
    """Read cached soft targets from JSON Lines, keyed by sample ID."""
    if path is None:
        return {}
    targets: dict[str, TeacherTarget] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            identifier = str(payload["sample_id"])
            value = float(payload["value"])
            entries = payload["policy"]
            policy: dict[int, float] = {}
            for action, probability in entries:
                parsed_action = int(action)
                if parsed_action in policy:
                    raise ValueError("duplicate policy action")
                policy[parsed_action] = float(probability)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid teacher target on line {line_number}") from error
        if identifier in targets:
            raise ValueError(f"Duplicate teacher target for {identifier}")
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError(f"Teacher value for {identifier} is outside [-1, 1]")
        if not policy or any(
            action < 0 or not np.isfinite(probability) or probability < 0
            for action, probability in policy.items()
        ):
            raise ValueError(f"Teacher policy for {identifier} is invalid")
        total = sum(policy.values())
        if total <= 0:
            raise ValueError(f"Teacher policy for {identifier} has no probability mass")
        targets[identifier] = TeacherTarget(
            {action: probability / total for action, probability in policy.items()}, value
        )
    return targets


def _examples(
    game: GameRecord,
    teacher: Mapping[str, TeacherTarget],
    stride: int,
) -> Iterator[tuple[str, NDArray[np.int8], NDArray[np.bool_], NDArray[np.float32], float, int]]:
    position = Position.empty(game.size)
    for move_number, action in enumerate(game.moves):
        if move_number % stride == 0:
            identifier = sample_id(game.game_id, move_number)
            features = position.features().astype(np.int8)
            legal = np.zeros(position.points + 1, dtype=np.bool_)
            legal[list(position.legal_actions())] = True
            policy = np.zeros(position.points + 1, dtype=np.float32)
            target = teacher.get(identifier)
            source = 0
            # Both factors are +/-1, so the product is an exact integer.
            value: float = game.winner * position.to_play
            if target is None:
                policy[action] = 1
            else:
                for target_action, probability in target.policy.items():
                    if target_action >= policy.size or not legal[target_action]:
                        raise ValueError(
                            f"Teacher action {target_action} is illegal for {identifier}"
                        )
                    policy[target_action] = probability
                value = target.value
                source = 1
            yield identifier, features, legal, policy, value, source
        position = position.play(action)


def _write_npz(path: Path, rows: Sequence[tuple[Any, ...]], size: int) -> None:
    feature_count = 2 * size * size + 1
    action_count = size * size + 1
    if rows:
        identifiers, features, legal, policies, values, sources = zip(*rows, strict=True)
        arrays = {
            "sample_ids": np.asarray(identifiers),
            "features": np.stack(features),
            "legal": np.stack(legal),
            "policy": np.stack(policies),
            "value": np.asarray(values, dtype=np.float32),
            "source": np.asarray(sources, dtype=np.uint8),
        }
    else:
        arrays = {
            "sample_ids": np.asarray([], dtype="U1"),
            "features": np.empty((0, feature_count), dtype=np.int8),
            "legal": np.empty((0, action_count), dtype=np.bool_),
            "policy": np.empty((0, action_count), dtype=np.float32),
            "value": np.empty(0, dtype=np.float32),
            "source": np.empty(0, dtype=np.uint8),
        }
    write_bytes(path, lambda temporary: np.savez_compressed(temporary, **arrays))


def build_dataset(
    games: Sequence[GameRecord],
    output: Path,
    *,
    size: int = 19,
    stride: int = 1,
    teacher_path: Path | None = None,
    require_teacher: bool = False,
) -> DatasetSummary:
    """Write split NPZ files and a provenance manifest atomically."""
    if size not in BOARD_SIZES:
        raise ValueError(f"Unsupported board size {size}")
    if stride < 1:
        raise ValueError("stride must be positive")
    teacher = load_teacher_targets(teacher_path)
    game_ids = [game.game_id for game in games]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("Duplicate games were supplied")
    if require_teacher:
        expected = {
            sample_id(game.game_id, turn)
            for game in games
            if game.size == size
            for turn in range(0, len(game.moves), stride)
        }
        missing = expected - teacher.keys()
        if missing:
            raise ValueError(f"Missing teacher targets for {len(missing)} sampled positions")
    if (output / MANIFEST_FILE).exists():
        raise ValueError(
            f"Dataset {output} is already published; a published generation is never replaced. "
            "Build the new generation in a separate directory"
        )
    split_rows: dict[str, list[tuple[Any, ...]]] = {name: [] for name in SPLITS}
    seen_positions: set[bytes] = set()
    accepted = 0
    duplicates = 0
    rejected = 0
    for game in sorted(games, key=lambda candidate: candidate.game_id):
        if game.size != size:
            rejected += 1
            continue
        split = split_of(game.game_id)
        for row in _examples(game, teacher, stride):
            position_key = row[1].tobytes()
            if position_key in seen_positions:
                duplicates += 1
                continue
            seen_positions.add(position_key)
            split_rows[split].append(row)
        accepted += 1
    if accepted == 0:
        raise ValueError(f"No {size}x{size} games were supplied")
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split, rows in split_rows.items():
        path = output / f"{split}.npz"
        _write_npz(path, rows, size)
        files[split] = {
            "file": path.name,
            "examples": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "version": DATASET_VERSION,
        "ruleset": RULESET,
        "size": size,
        "stride": stride,
        "split": "sha256(game_id) 80/10/10",
        "symmetry": "dihedral-8 at training time",
        "accepted_games": accepted,
        "duplicate_examples": duplicates,
        "rejected_games": rejected,
        "teacher_targets": len(teacher),
        "require_teacher": require_teacher,
        "game_set_sha256": hashlib.sha256("\n".join(sorted(game_ids)).encode()).hexdigest(),
        "teacher_sha256": (
            hashlib.sha256(teacher_path.read_bytes()).hexdigest() if teacher_path else None
        ),
        "files": files,
    }
    _write_manifest(output, manifest)
    return DatasetSummary(
        accepted,
        duplicates,
        sum(len(rows) for rows in split_rows.values()),
        rejected,
    )


def _write_manifest(output: Path, manifest: Mapping[str, Any]) -> None:
    """Publish the manifest, which is the single commit point for a generation."""
    write_text(
        output / MANIFEST_FILE,
        lambda temporary: temporary.write(json.dumps(manifest, indent=2) + "\n"),
    )
