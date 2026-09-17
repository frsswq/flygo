# FlyGo

FlyGo tests whether the official MaleCNS fruit-fly connectome provides useful computational structure for learning Go on 5x5 through 9x9 boards.
The biological topology stays frozen while an encoder and readout train around it.

This repository uses the official `male-cns:v1.0` release from HHMI Janelia.
It does not use fly.ai or another connectome simulator.
Janelia supplies the wiring map, annotations, and neurotransmitter predictions.
FlyGo supplies the artificial dynamics, Go encoding, training procedure, and evaluation.

## Install

Install Python 3.12 or later, Node.js 20.19 or later, or 22.12 or later, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cd web && npm ci
```

## Quick start

```bash
make dev
```

Open <http://127.0.0.1:5173>.
You play Black, and the frozen graph answers as White after each stone you place.
Passing also passes for White, so two consecutive passes end the game and the status word reports the area score.
The ruleset is Tromp-Taylor with area scoring, positional superko, and komi 0.0 on 5x5 and 7.5 above.
The encoder and readout are intentionally untrained, so the play is weak.

## Data

```bash
uv run flygo download                          # about 1.1 GB into data/raw/, checksums verified
uv run flygo profile                           # measured schemas and counts
uv run flygo prepare --minimum-weight 5        # filtered experiment graph
```

The prepared graph keeps connections whose source and target are both `status == "Traced"`, with at least five detected synaptic contacts.
The raw release contains segment-to-segment connections, including many untraced segments, so do not treat every segment ID as an annotated neuron.

## Model boundary

FlyGo uses this documented rate model:

```text
x[t+1] = tanh(retention * x[t] + recurrent_gain * normalize(W @ x[t]) + input)
```

`W` comes from MaleCNS and remains immutable.
The normalization divides each receiving neuron's aggregate input by its incoming connection strength.
This equation is a computational assumption, not an official biological simulation.

The planned primary comparison places MaleCNS against directed degree-preserving rewired controls and a capacity-matched baseline.
See the [experiment protocol](docs/experiment.md) for controls, metrics, and validity limits.

## Documentation

- [Experiment protocol](docs/experiment.md): controls, metrics, and validity limits.
- [Engineering notes](docs/engineering.md): repository layout, browser bundle, conformance fixtures, and checks.
- [Deployment](docs/deploy.md): static build and Cloudflare settings.
- [Data profile](docs/data-profile.json): measured facts about the downloaded release.

## Development

```bash
make dev      # viewer on port 5173, no backend needed
make build    # FastAPI build into src/flygo/static
make static   # deployable build into web/dist
make api      # FastAPI reference endpoints on port 8000
make check    # every Python and web check
```

`make check` runs Ruff, vulture, BasedPyright, and pytest for Python, and Ultracite, the TypeScript compiler, knip, and Vitest for the web app.

## Data provenance

- [Official MaleCNS download page](https://male-cns.janelia.org/download/)
- Dataset: `male-cns:v1.0`
- Published confidence threshold: `minconf-0.5`
- Bundled viewer graph: 461 annotated neurons and 605 edges, selected by deterministic sensory-path expansion

The MaleCNS data has its own terms and attribution requirements.
This repository's software license does not replace the dataset terms.

## License

FlyGo source code is available under the [MIT License](LICENSE).
