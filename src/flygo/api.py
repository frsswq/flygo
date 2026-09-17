"""FastAPI application for the FlyGo experiment viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flygo.connectome import from_frame
from flygo.go import Position
from flygo.model import ConnectomePolicy

PACKAGE_DIRECTORY = Path(__file__).parent
ASSET_DIRECTORY = PACKAGE_DIRECTORY / "assets"
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"

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
    return SimulationResponse(
        recommended_action=action,
        legal_actions=list(legal),
        activity=active_neurons,
        topology="Official MaleCNS v1.0 derived sensory-path sample (461 neurons, 605 edges)",
        model_status="Untrained encoder/readout demonstration",
    )
