"""Validate and describe the bounded 19x19 feasibility experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from flygo.research import MODELS, ModelName

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Artifact(StrictModel):
    path: Path
    sha256: Sha256


class PlannedArtifact(StrictModel):
    path: Path
    sha256: Sha256 | None


class SourcePage(StrictModel):
    url: str
    sha256: Sha256
    sampled_game_ids: tuple[str, ...]


def source_archive_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Expected an archive path inside the source directory")
    return value


SourceArchivePath = Annotated[str, AfterValidator(source_archive_path)]


class SourceArchive(StrictModel):
    url: str
    file: SourceArchivePath
    sha256: Sha256


class FeasibilitySources(StrictModel):
    schema_version: Literal[1]
    selection: str
    pages: tuple[SourcePage, ...]
    archives: tuple[SourceArchive, ...]


class CorpusProtocol(StrictModel):
    board_size: Literal[19]
    komi: float
    target_labelled_positions: Literal[10_000]
    visits: Literal[256]
    stride: int = Field(gt=0)
    source_manifest: Artifact
    data_permission: Artifact
    teacher_network: Artifact
    teacher_executable: Artifact
    teacher_configuration: Artifact
    pilot_queries: Artifact
    output: PlannedArtifact
    work_directory: Path
    games_per_shard: Literal[1]
    timed_batch_positions: int = Field(ge=50, le=500)
    timed_batch_report: PlannedArtifact


class CircuitProtocol(StrictModel):
    nodes: Literal[500]
    source_graph: Artifact
    output: PlannedArtifact


class TrainingProtocol(StrictModel):
    models: tuple[ModelName, ...]
    seeds: tuple[int, ...]
    epochs: Literal[10]
    batch_size: Literal[128]
    learning_rate: float
    steps: Literal[8]
    retention: float
    recurrent_gain: float
    normalization: Literal["incoming"]
    weights: Literal["weighted"]
    value_weight: float
    final_test: Literal[False]
    output: Path
    summary: Path


class ApprovalProtocol(StrictModel):
    timed_batch_approved: bool
    full_labelling_approved: bool


class FeasibilityProtocol(StrictModel):
    schema_version: Literal[1]
    name: Literal["feasibility-19-v1"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    claim_scope: Literal["feasibility-only"]
    corpus: CorpusProtocol
    circuit: CircuitProtocol
    training: TrainingProtocol
    approvals: ApprovalProtocol


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_queries(source: Path, target_positions: int) -> list[dict[str, Any]]:
    """Select exactly ``target_positions`` from pinned queries in file order."""
    if target_positions < 1:
        raise ValueError("Timing target must be positive")
    selected: list[dict[str, Any]] = []
    remaining = target_positions
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            identifier = str(row["id"])
            turns = row["analyzeTurns"]
            visits = int(row["maxVisits"])
            if (
                not isinstance(turns, list)
                or not turns
                or any(not isinstance(turn, int) or turn < 0 for turn in turns)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid timing source query on line {line_number}") from error
        chosen = turns[:remaining]
        if any((identifier, turn) in seen for turn in chosen):
            raise ValueError("Timing source contains duplicate requested positions")
        seen.update((identifier, turn) for turn in chosen)
        selected.append({**row, "analyzeTurns": chosen, "maxVisits": visits})
        remaining -= len(chosen)
        if remaining == 0:
            return selected
    raise ValueError(
        f"Timing source has only {target_positions - remaining} of {target_positions} positions"
    )


def verify_timing_analysis(
    analysis: Path,
    queries: list[dict[str, Any]],
    *,
    visits: int,
) -> int:
    """Require one final response at the fixed visit budget for each position."""
    expected: set[tuple[str, int]] = set()
    for query in queries:
        try:
            identifier = str(query["id"])
            turns = query["analyzeTurns"]
            if not isinstance(turns, list):
                raise ValueError
            expected.update((identifier, int(turn)) for turn in turns)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid query passed to timing verification") from error
    completed: set[tuple[str, int]] = set()
    for line_number, line in enumerate(analysis.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (str(row["id"]), int(row["turnNumber"]))
            during_search = row.get("isDuringSearch", False)
            actual_visits = int(row["rootInfo"]["visits"])
            move_infos = row["moveInfos"]
            if (
                not isinstance(during_search, bool)
                or not isinstance(move_infos, list)
                or not move_infos
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid timing response on line {line_number}") from error
        if during_search:
            raise ValueError(f"Timing response is unfinished on line {line_number}")
        if actual_visits != visits:
            raise ValueError(
                f"Timing response visit mismatch on line {line_number}: "
                f"expected {visits}, got {actual_visits}"
            )
        if key not in expected:
            raise ValueError(f"Unexpected timing response on line {line_number}: {key}")
        if key in completed:
            raise ValueError(f"Duplicate timing response on line {line_number}: {key}")
        completed.add(key)
    missing = expected - completed
    if missing:
        raise ValueError(f"Timing analysis is missing {len(missing)} final responses")
    return len(completed)


def _verify_artifact(name: str, artifact: Artifact, blockers: list[str]) -> None:
    if not artifact.path.is_file():
        blockers.append(f"missing {name}: {artifact.path}")
        return
    actual = file_sha256(artifact.path)
    if actual != artifact.sha256:
        raise ValueError(
            f"{name} SHA256 mismatch for {artifact.path}: expected {artifact.sha256}, got {actual}"
        )


def validate_protocol(protocol: FeasibilityProtocol) -> list[str]:
    """Validate fixed feasibility rules and return execution blockers."""
    if protocol.corpus.stride % 2 == 0:
        raise ValueError("Corpus stride must be odd so both players are sampled")
    if protocol.training.models != MODELS:
        raise ValueError(f"Training models must be exactly: {', '.join(MODELS)}")
    if protocol.training.seeds != (7, 17, 27):
        raise ValueError("Training seeds must be exactly 7, 17, and 27")
    fixed_numbers = {
        "komi": (protocol.corpus.komi, 7.5),
        "learning rate": (protocol.training.learning_rate, 0.001),
        "retention": (protocol.training.retention, 0.35),
        "recurrent gain": (protocol.training.recurrent_gain, 0.9),
        "value weight": (protocol.training.value_weight, 1.0),
    }
    for name, (actual, expected) in fixed_numbers.items():
        if actual != expected:
            raise ValueError(f"Feasibility {name} must be {expected}")

    blockers: list[str] = []
    artifacts = {
        "source manifest": protocol.corpus.source_manifest,
        "data permission": protocol.corpus.data_permission,
        "teacher network": protocol.corpus.teacher_network,
        "teacher executable": protocol.corpus.teacher_executable,
        "teacher configuration": protocol.corpus.teacher_configuration,
        "pilot queries": protocol.corpus.pilot_queries,
        "source graph": protocol.circuit.source_graph,
    }
    for name, artifact in artifacts.items():
        _verify_artifact(name, artifact, blockers)

    planned = {
        "timed batch report": protocol.corpus.timed_batch_report,
        "feasibility dataset": protocol.corpus.output,
        "selected circuit": protocol.circuit.output,
    }
    for name, artifact in planned.items():
        if artifact.sha256 is None:
            blockers.append(f"{name} has not been completed: {artifact.path}")
        else:
            _verify_artifact(name, Artifact(path=artifact.path, sha256=artifact.sha256), blockers)

    if not protocol.approvals.timed_batch_approved:
        blockers.append("timed teacher batch is not approved")
    if not protocol.approvals.full_labelling_approved:
        blockers.append("full 10,000-position labelling run awaits the timed-batch estimate")
    return blockers


def dry_run(protocol: FeasibilityProtocol) -> dict[str, Any]:
    """Return the complete planned experiment without starting any work."""
    blockers = validate_protocol(protocol)
    cells = [
        {"model": model, "seed": seed}
        for model in protocol.training.models
        for seed in protocol.training.seeds
    ]
    return {
        "schema_version": protocol.schema_version,
        "protocol": protocol.name,
        "source_revision": protocol.source_revision,
        "claim_scope": protocol.claim_scope,
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "teacher": {
            "board_size": protocol.corpus.board_size,
            "target_labelled_positions": protocol.corpus.target_labelled_positions,
            "visits": protocol.corpus.visits,
            "stride": protocol.corpus.stride,
            "timed_batch_positions": protocol.corpus.timed_batch_positions,
            "work_directory": str(protocol.corpus.work_directory),
            "games_per_shard": protocol.corpus.games_per_shard,
            "output": str(protocol.corpus.output.path),
        },
        "circuit": {
            "nodes": protocol.circuit.nodes,
            "source": str(protocol.circuit.source_graph.path),
            "output": str(protocol.circuit.output.path),
        },
        "training": {
            "cells": cells,
            "cell_count": len(cells),
            "epochs": protocol.training.epochs,
            "batch_size": protocol.training.batch_size,
            "output": str(protocol.training.output),
            "summary": str(protocol.training.summary),
            "final_test": protocol.training.final_test,
        },
    }
