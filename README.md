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

Start the viewer:

```bash
cd web && npm ci && npm run dev
```

Open <http://127.0.0.1:5173>.
The page loads a small, deterministic subgraph derived from the official release and runs it in the browser.
You play Black, and FlyGo automatically applies a White response after each legal move.
Two passes in a row end the game, and the status word then reports the area score.
The ruleset is Tromp-Taylor with area scoring, positional superko, and komi 0.0 on 5x5 and 7.5 above.
Its encoder and readout are intentionally untrained at this stage.

## Deploy

The site is static and all inference runs in the visitor's browser, so no server is required.
Cloudflare serves files only, which is free and unmetered on the free plan.

Reproduce the deployable artifact locally:

```bash
make static
```

That writes `web/dist/`, which is the directory to publish.
It serves `/assets/*` (content hashed) and `/flygo/*` (the graph and one policy per board size, about 470 KB for 9x9).
`web/public/_headers` sets the cache policy for both.

Connect the repository once in the Cloudflare dashboard, then use these settings:

| Setting | Value |
| --- | --- |
| Build command | `cd web && npm ci && npm run build` |
| Output directory | `web/dist` |
| Environment variable | `FLYGO_PUBLIC_BASE` = `/` |
| Environment variable | `FLYGO_OUT_DIR` = `dist` |

`FLYGO_PUBLIC_BASE` switches the built asset URLs from the FastAPI mount at `/static/` to the root.
`FLYGO_OUT_DIR` keeps the deployable build out of the committed FastAPI build.
Cloudflare Pages also accepts the same repository as a Workers project with `npx wrangler deploy`; the artifact is identical.

Check the result with any static file server, for example:

```bash
python3 -m http.server 8099 --directory web/dist
```

## Browser bundle

The viewer runs the frozen graph in the browser, so the exported binaries are committed under `web/public/flygo/`:

```bash
uv run flygo export-web
```

The command writes `graph.bin`, one `policy-{size}.bin` per board size, and a manifest.
The manifest records the byte layout, the dynamics constants, and a SHA-256 for every file.
`src/flygo/export.py` documents each byte offset.

Both rule engines and the dynamics are checked against shared fixtures:

```bash
uv run flygo conformance
uv run flygo policy-conformance
```

`shared/rules-conformance.json` is replayed by `tests/test_conformance.py` and by `web/src/lib/go-rules.test.ts`.
`shared/policy-conformance.json` is replayed by `web/src/lib/policy.test.ts`, which also verifies the recorded hashes.
Regenerate a fixture after changing either engine, and never edit a fixture by hand.

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
├── conformance.py     # Generates the shared rules fixture
├── export.py          # Writes the browser graph and policy binaries
├── policy_fixture.py  # Generates the shared policy fixture
├── api.py             # FastAPI research viewer and reference endpoints
├── assets/            # Small official derived subgraph for the viewer
└── static/            # Generated Vite build for the FastAPI mount
web/                    # React, Vite, Base UI shadcn, and Ultracite source
├── public/flygo/      # Committed browser bundle, written by export-web
└── src/lib/           # Rules engine, browser dynamics, and transport schemas
shared/                 # Conformance fixtures replayed by both languages
docs/
├── data-profile.json  # Generated facts about the downloaded release
└── experiment.md      # Evaluation protocol
data/                   # Ignored raw and processed artifacts
tests/                  # Rules, conformance, simulation, and API tests
```

Production Python code uses Polars instead of pandas.
PyArrow exists only as a compatibility boundary for one nullable dictionary column in the official annotation Feather file.

## Development

Start the viewer. It needs no backend, because the browser loads the exported graph and runs the policy itself:

```bash
make dev
```

Open <http://127.0.0.1:5173>.

Run the FastAPI research server, which serves the built app and the `/api` endpoints:

```bash
make build && make api
```

Open <http://127.0.0.1:8000>.
The API is the Python reference for the same rules and dynamics, not a dependency of the viewer.

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

Or run every check at once:

```bash
make check
```

`make build` writes into `src/flygo/static/` for FastAPI to serve.
`make static` writes the deployable build into `web/dist/`.
Vite still proxies `/api` requests to port 8000 for API work.

## Data provenance

- [Official MaleCNS download page](https://male-cns.janelia.org/download/)
- Dataset: `male-cns:v1.0`
- Connection confidence threshold in published filenames: `minconf-0.5`
- Bundled demo graph: 461 annotated neurons and 605 edges selected by deterministic sensory-path expansion from the official files

The MaleCNS data has its own terms and attribution requirements.
This repository's software license does not replace the dataset terms.

## License

FlyGo source code is available under the [MIT License](LICENSE).
