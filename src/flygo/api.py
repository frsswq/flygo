"""FastAPI application for the FlyGo experiment viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flygo.connectome import FloatArray, from_frame
from flygo.go import PASS, Position
from flygo.model import ConnectomePolicy

PACKAGE_DIRECTORY = Path(__file__).parent
ASSET_DIRECTORY = PACKAGE_DIRECTORY / "assets"
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"
TOPOLOGY_DESCRIPTION = "Official MaleCNS v1.0 derived sensory-path sample (461 neurons, 605 edges)"
MODEL_STATUS = "Untrained encoder/readout demonstration"

_edges = pl.read_parquet(ASSET_DIRECTORY / "malecns-sample.parquet")
_neurons = pl.read_parquet(ASSET_DIRECTORY / "malecns-sample-neurons.parquet")
_connectome = from_frame(_edges)
_policy = ConnectomePolicy.initialize(_connectome)
_neuron_lookup = {row["id"]: row for row in _neurons.iter_rows(named=True)}

app = FastAPI(
    title="FlyGo",
    summary="Explore 5x5 Go through a frozen sample of official MaleCNS wiring.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


class PositionRequest(BaseModel):
    board: list[Literal[-1, 0, 1]] = Field(min_length=25, max_length=25)
    to_play: Literal[-1, 1]


class TurnRequest(BaseModel):
    board: list[Literal[-1, 0, 1]] = Field(min_length=25, max_length=25)
    previous_board: list[Literal[-1, 0, 1]] | None = Field(
        default=None, min_length=25, max_length=25
    )
    action: int = Field(ge=0, le=PASS)
    consecutive_passes: int = Field(default=0, ge=0, le=1)


class ActiveNeuron(BaseModel):
    body_id: int
    activity: float
    neuron_type: str | None
    superclass: str | None


class SimulationResponse(BaseModel):
    recommended_action: int
    legal_actions: list[int]
    activity: list[ActiveNeuron]
    topology: str
    model_status: str


class TurnResponse(BaseModel):
    board: list[int]
    previous_board: list[int] | None
    computer_action: int | None
    legal_actions: list[int]
    consecutive_passes: int
    game_over: bool
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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "dataset": "male-cns:v1.0"}


@app.post("/api/simulate")
def simulate(
    request: PositionRequest,
    steps: Annotated[int, Query(ge=1, le=32)] = 8,
) -> SimulationResponse:
    try:
        position = Position(tuple(request.board), request.to_play)
        legal = position.legal_actions()
        activity = _policy.activity(position, steps=steps)
        action = _policy.choose_legal_action(position)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return SimulationResponse(
        recommended_action=action,
        legal_actions=list(legal),
        activity=_strongest_neurons(activity),
        topology=TOPOLOGY_DESCRIPTION,
        model_status=MODEL_STATUS,
    )


@app.post("/api/play")
def play_turn(request: TurnRequest) -> TurnResponse:
    """Apply one Black move and an automatic FlyGo White response."""
    try:
        previous_board = tuple(request.previous_board) if request.previous_board else None
        position = Position(tuple(request.board), 1, previous_board)
        after_human = position.play(request.action)
        activity = _policy.activity(after_human)

        pass_count = request.consecutive_passes + 1 if request.action == PASS else 0
        game_over = pass_count == 2
        computer_action = None
        result = after_human
        if not game_over:
            computer_action = _policy.choose_legal_action(after_human)
            result = after_human.play(computer_action)
            pass_count = pass_count + 1 if computer_action == PASS else 0
            game_over = pass_count == 2
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return TurnResponse(
        board=list(result.board),
        previous_board=list(result.previous_board) if result.previous_board else None,
        computer_action=computer_action,
        legal_actions=[] if game_over else list(result.legal_actions()),
        consecutive_passes=pass_count,
        game_over=game_over,
        activity=_strongest_neurons(activity),
    )
