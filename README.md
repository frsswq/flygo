# FlyGo

FlyGo tests whether the official MaleCNS fruit-fly connectome provides useful computational structure for learning Go on boards from 5x5 through 9x9.
The biological topology stays frozen while experiments train an encoder and readout around it.

> **Research question:** Can a real biological connectome provide useful computational structure for learning an unrelated task like Go?

This repository uses the official `male-cns:v1.0` release from HHMI Janelia.
It does not use fly.ai or another connectome simulator.
Janelia supplies the wiring map, annotations, and neurotransmitter predictions.
FlyGo supplies the artificial dynamics, Go encoding, training procedure, and evaluation.

## Install

Install Python 3.12 or later, Node.js 20.19 or later, and [uv](https://docs.astral.sh/uv/).
Then install the backend and frontend dependencies:

```bash
uv sync --all-groups
cd web && npm ci
```

## Quick start

Build and start the research viewer:

```bash
cd web && npm run build && cd ..
uv run fastapi dev
```

Open <http://127.0.0.1:8000>.
The bundled visualization uses a small, deterministic subgraph derived from the official release.
You play Black, and FlyGo automatically applies a White response after each legal move.
Two passes in a row end the game, and the status word then reports the area score.
The ruleset is Tromp-Taylor with area scoring, positional superko, and komi 0.0 on 5x5 and 7.5 above.
Its encoder and readout are intentionally untrained at this stage.

## Official data

Download and verify the three official files:

```bash
uv run flygo download
```

The command downloads approximately 1.1 GB into `data/raw/` and verifies committed SHA-256 checksums.
The files are not committed to Git.

Create a reproducible profile and a filtered experiment graph:

```bash
uv run flygo profile
uv run flygo prepare --minimum-weight 5
```

The data profile records measured schemas and counts in [`docs/data-profile.json`](docs/data-profile.json).
The prepared graph includes only connections whose source and target have `status == "Traced"` in the official annotation table.
The default minimum connection strength is five detected synaptic contacts.

The raw release contains segment-to-segment connections, including many untraced segments.
Do not treat every segment ID as an annotated neuron.
The current profile measures 165,122 traced annotations and 25,563,197 connections between traced endpoints before strength filtering.

## Model boundary

FlyGo currently uses this documented rate model:

```text
x[t+1] = tanh(retention * x[t] + recurrent_gain * normalize(W @ x[t]) + input)
```

`W` comes from MaleCNS and remains immutable.
The normalization divides each receiving neuron's aggregate input by its incoming connection strength.
This equation is a computational assumption, not an official biological simulation.

The planned primary comparison uses:

1. The official frozen MaleCNS topology.
2. Directed degree-preserving rewired controls across multiple random seeds.
3. A conventional linear or multilayer perceptron baseline with comparable trainable capacity.

See the [experiment protocol](docs/experiment.md) for controls, metrics, and validity limits.

## Project layout

```text
src/flygo/
├── official_data.py   # Official downloads, compatibility boundary, and profiling
├── connectome.py      # Frozen graph and lightweight dynamics
├── go.py              # Immutable Go rules, scoring, and features
├── model.py           # Trainable encoder/readout and controls
├── api.py             # FastAPI research viewer
├── assets/            # Small official derived subgraph for the viewer
└── static/            # Generated Vite production build
web/                    # React, Vite, Base UI shadcn, and Ultracite source
docs/
├── data-profile.json  # Generated facts about the downloaded release
└── experiment.md      # Evaluation protocol
data/                   # Ignored raw and processed artifacts
tests/                  # Rules, simulation, and API tests
```

Production Python code uses Polars instead of pandas.
PyArrow exists only as a compatibility boundary for one nullable dictionary column in the official annotation Feather file.

## Development

Start the FastAPI and Vite development servers together:

```bash
make dev
```

Open <http://127.0.0.1:5173>.
Stopping `make dev` also stops both child servers.

Run all checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
cd web
npm run check
npm run typecheck
npm test
npm run build
```

For frontend development, run `uv run fastapi dev` and `npm run dev` from `web/` in separate terminals.
Vite proxies `/api` requests to FastAPI on port 8000.
The production build writes into `src/flygo/static/` for FastAPI to serve.

Run the API in production mode:

```bash
uv run fastapi run
```

## Data provenance

- [Official MaleCNS download page](https://male-cns.janelia.org/download/)
- Dataset: `male-cns:v1.0`
- Connection confidence threshold in published filenames: `minconf-0.5`
- Bundled demo graph: 461 annotated neurons and 605 edges selected by deterministic sensory-path expansion from the official files

The MaleCNS data has its own terms and attribution requirements.
This repository's software license does not replace the dataset terms.

## License

FlyGo source code is available under the [MIT License](LICENSE).
