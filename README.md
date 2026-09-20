# FlyGo

FlyGo tests whether fixed fruit-fly connectome topology can support a policy-value model for standard 19x19 Go.
It provides SGF ingestion, KataGo teacher import, frozen-connectome training, MCTS, paired Elo tournaments, and browser inference.

The repository uses the official `male-cns:v1.0` release from HHMI Janelia.
MaleCNS supplies the wiring map and annotations.
FlyGo supplies all dynamics, Go representations, optimization, search, and evaluation.
This is a topology-transfer experiment, not a biological simulation.

The committed browser weights are an untrained demonstration until you export a trained checkpoint.
The hosted build is at <https://gofly.farissaifuddin.com>.

## Install

Install Python 3.12 or later, Node.js 20.19 or later or 22.12 or later, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cd web && npm ci && cd ..
```

## Quick start

```bash
make dev
```

Open <http://127.0.0.1:5173>.
You play Black on a 19x19 board.
A Web Worker gives FlyGo one second of policy-value MCTS for each reply.
The interface also provides 5x5 for exact rules and solver validation.
The Self-play button makes FlyGo take both colours until two passes end the game.

## Train and benchmark

Build a first human-move dataset from SGF files:

```bash
uv run flygo build-dataset --sgf data/sgf --output data/datasets/human-19
uv run flygo train --dataset data/datasets/human-19 --output data/models/flygo-19.npz
uv run flygo benchmark --policy data/models/flygo-19.npz --output data/reports/flygo-19.json
```

Teacher distillation uses KataGo JSON analysis output before the final dataset build.
See the [training pipeline](docs/pipeline.md) for the complete reproducible workflow.

Export a trained browser bundle and build the static site:

```bash
uv run flygo export-web --policy data/models/flygo-19.npz
make static
```

## Connectome data

```bash
uv run flygo download
uv run flygo profile
uv run flygo prepare --minimum-weight 5
```

The prepared graph keeps connections whose endpoints are traced and which have at least five detected synaptic contacts.
The bundled browser graph is a deterministic sensory-path sample with 461 neurons and 605 edges.

The frozen recurrent update is:

```text
x[t+1] = tanh(retention * x[t] + recurrent_gain * normalize(W @ x[t]) + input)
```

`W` comes from MaleCNS and remains immutable during training.
The encoder, policy readout, and value readout train around it.

## Documentation

- [Training pipeline](docs/pipeline.md): SGF validation, KataGo analysis, training, Elo, and export.
- [Remaining work](docs/plan.md): priority order, workstreams, and gates from here to a research result.
- [Experiment protocol](docs/experiment.md): controls, metrics, and validity limits.
- [Research foundation](docs/research.md): matched controls, reproducible screening, and claim limits.
- [KataGo pilot](docs/teacher-pilot.md): the completed real-data pipeline check.
- [Engineering notes](docs/engineering.md): repository structure, binary formats, fixtures, and checks.
- [Deployment](docs/deploy.md): static Cloudflare deployment.
- [Data profile](docs/data-profile.json): measured facts from the official release.

## Development

```bash
make dev      # browser viewer on port 5173
make build    # FastAPI static build
make static   # deployable build in web/dist
make api      # reference API on port 8000
make check    # all Python and web checks
```

## Data provenance

- [Official MaleCNS download page](https://male-cns.janelia.org/download/)
- Dataset: `male-cns:v1.0`
- Published confidence threshold: `minconf-0.5`

MaleCNS data has its own terms and attribution requirements.
The MIT software license does not replace those terms.

## License

FlyGo source code is available under the [MIT License](LICENSE).
