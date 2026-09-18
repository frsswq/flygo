"""FastAPI application for the FlyGo experiment viewer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Self

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from flygo.connectome import FloatArray, from_frame
from flygo.go import (
    BOARD_SIZES,
    DEFAULT_BOARD_SIZE,
    RULESET,
    Position,
)
from flygo.model import ConnectomePolicy

PACKAGE_DIRECTORY = Path(__file__).parent
ASSET_DIRECTORY = PACKAGE_DIRECTORY / "assets"
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"
TOPOLOGY_DESCRIPTION = "Official MaleCNS v1.0 derived sensory-path sample (461 neurons, 605 edges)"
MODEL_STATUS = "Untrained encoder/readout demonstration"
BOARD_SIZE_FIELD = Field(default=DEFAULT_BOARD_SIZE)

_edges = pl.read_parquet(ASSET_DIRECTORY / "malecns-sample.parquet")
_neurons = pl.read_parquet(ASSET_DIRECTORY / "malecns-sample-neurons.parquet")
_connectome = from_frame(_edges)
_policies = {size: ConnectomePolicy.initialize(_connectome, size=size) for size in BOARD_SIZES}
_neuron_lookup = {row["id"]: row for row in _neurons.iter_rows(named=True)}

app = FastAPI(
    title="FlyGo",
    summary="Play Go against a frozen sample of official MaleCNS wiring.",
    version="0.3.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


class MoveSequence(BaseModel):
    """A whole game as the list of actions played so far, oldest first.

    The server replays the list, so the position, the turn, and the ko history
    are derived from one source of truth instead of being sent by the client.
    """

    size: int = BOARD_SIZE_FIELD
    moves: list[int]

    @model_validator(mode="after")
    def check_moves(self) -> Self:
        if self.size not in BOARD_SIZES:
            raise ValueError(f"Board size must be one of {BOARD_SIZES}")
        pass_action = self.size * self.size
        if any(not 0 <= action <= pass_action for action in self.moves):
            raise ValueError("Every action must be a board point or pass")
        return self


class PositionRequest(MoveSequence):
    """A position to evaluate, given as the moves that reached it."""


class TurnRequest(MoveSequence):
    """A game where the last action is the move to apply now."""

    @model_validator(mode="after")
    def check_last_move(self) -> Self:
        if not self.moves:
            raise ValueError("A turn needs at least one action")
        return self


class ActiveNeuron(BaseModel):
    body_id: int
    activity: float
    neuron_type: str | None
    superclass: str | None


class ScorePayload(BaseModel):
    black_area: int
    white_area: int
    komi: float
    winner: int
    margin: float
    label: str


class SimulationResponse(BaseModel):
    size: int
    to_play: int
    ruleset: str
    recommended_action: int
    legal_actions: list[int]
    activity: list[ActiveNeuron]
    topology: str
    model_status: str


class TurnResponse(BaseModel):
    size: int
    moves: list[int]
    board: list[int]
    to_play: int
    ruleset: str
    computer_action: int | None
    legal_actions: list[int]
    consecutive_passes: int
    game_over: bool
    score: ScorePayload | None
    activity: list[ActiveNeuron]


def _strongest_neurons(activity: FloatArray) -> list[ActiveNeuron]:
    strongest = sorted(
        range(activity.size), key=lambda index: abs(float(activity[index])), reverse=True
    )[:40]
    active_neurons = []
    for index in strongest:
        body_id = int(_connectome.node_ids[index])
        metadata = _neuron_lookup.get(body_id, {})
        active_neurons.append(
            ActiveNeuron(
                body_id=body_id,
                activity=float(activity[index]),
                neuron_type=metadata.get("type"),
                superclass=metadata.get("superclass"),
            )
        )
    return active_neurons


def _replay(size: int, moves: Sequence[int]) -> Position:
    position = Position.empty(size)
    consecutive_passes = 0
    for action in moves:
        if consecutive_passes >= 2:
            raise ValueError("That game is already over")
        position = position.play(action)
        consecutive_passes = consecutive_passes + 1 if action == position.pass_action else 0
    return position


def _trailing_passes(moves: Sequence[int], pass_action: int) -> int:
    count = 0
    for action in reversed(moves):
        if action != pass_action:
            break
        count += 1
    return count


def _score(position: Position) -> ScorePayload:
    result = position.result()
    return ScorePayload(
        black_area=result.black_area,
        white_area=result.white_area,
        komi=result.komi,
        winner=result.winner,
        margin=result.margin,
        label=result.label,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "dataset": "male-cns:v1.0", "ruleset": RULESET}


@app.post("/api/simulate")
def simulate(
    request: PositionRequest,
    steps: Annotated[int, Query(ge=1, le=32)] = 8,
) -> SimulationResponse:
    """Report the policy reading of the position that ``request.moves`` reaches."""
    try:
        position = _replay(request.size, request.moves)
        policy = _policies[request.size]
        activity = policy.activity(position, steps=steps)
        action = policy.choose_legal_action(position)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return SimulationResponse(
        size=request.size,
        to_play=position.to_play,
        ruleset=RULESET,
        recommended_action=action,
        legal_actions=list(position.legal_actions()),
        activity=_strongest_neurons(activity),
        topology=TOPOLOGY_DESCRIPTION,
        model_status=MODEL_STATUS,
    )


@app.post("/api/play")
def play_turn(request: TurnRequest) -> TurnResponse:
    """Apply the last action, then let FlyGo answer as White."""
    pass_action = request.size * request.size
    try:
        position = _replay(request.size, request.moves)
        passes = _trailing_passes(request.moves, pass_action)
        game_over = passes >= 2
        moves = list(request.moves)
        computer_action = None
        if not game_over:
            computer_action = _policies[request.size].choose_legal_action(position)
            position = position.play(computer_action)
            moves.append(computer_action)
            passes = _trailing_passes(moves, pass_action)
            game_over = passes >= 2
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    activity = _policies[request.size].activity(position)
    return TurnResponse(
        size=position.size,
        moves=moves,
        board=list(position.board),
        to_play=position.to_play,
        ruleset=RULESET,
        computer_action=computer_action,
        legal_actions=[] if game_over else list(position.legal_actions()),
        consecutive_passes=passes,
        game_over=game_over,
        score=_score(position) if game_over else None,
        activity=_strongest_neurons(activity),
    )
