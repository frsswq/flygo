# Remaining work

This document records the work that remains before the project can report a research result and ship a trained browser release.
It follows the priority order agreed during the 2026-09-20 review.
Update the status markers when a unit completes.
Keep the order unless a measured result changes it.

## Current state

- The full pipeline is implemented and checked.
- All repository checks pass: Ruff, BasedPyright, vulture, pytest (112 tests), Ultracite, TypeScript, knip, and Vitest (40 tests).
- The tracked backlog is empty.
- The only real-data run is the [KataGo pilot](teacher-pilot.md): 24 games, 244 labelled positions, 221 training examples.
- The browser bundle is untrained, and `web/public/flygo/manifest.json` reports `"trained": false` for both board sizes.
- The untrained bundle is deployed at <https://gofly.farissaifuddin.com>.
- No research report exists under `data/research/`.
- The matched comparison in [research.md](research.md) has run only on synthetic smoke data.

The code is not the constraint.
The constraints are training data, compute for teacher labelling, and the efficiency of browser inference.

## Known limits of the current artifacts

### Graph

| Property | Bundled sample | Selected circuit, 500 neurons | Selected circuit, 1000 neurons |
| --- | ---: | ---: | ---: |
| Neurons | 461 | 500 | 1000 |
| Directed connections | 605 | 22,502 | 53,332 |
| Neurons with no outgoing connection | 304 | 2 | 4 |
| Largest strongly connected component | 14 | 498 | 996 |

The bundled graph is a deterministic sensory-path sample, and it is unrepresentative.
304 of its 461 neurons cannot send activity onward, and its largest strongly connected component holds 14 neurons.

The prepared graph is far better connected.
`flygo select-circuit` produced the circuit columns above from `data/processed/malecns-traced-w5.parquet` in 25 seconds.
A 500-neuron circuit holds a 498-neuron strongly connected core.

Use a selected circuit for any claim about the recurrent core.
Keep the bundled sample for the browser demonstration and integration tests, where size matters more than connectivity.

Model size is unchanged by circuit selection:

| Property | Value |
| --- | ---: |
| Trainable parameters at 19x19 | 500,646 |
| Recurrent edge accumulations per evaluation | 4,840 |

### Browser compute

The dense encoder and heads need 500,646 multiply-accumulates per evaluation at 19x19.
The recurrent graph needs 4,840 edge accumulations across eight steps.
The dense layers dominate the cost.
Faster graph traversal alone cannot meet the one-second browser budget.
The objective is playing strength per second, not neuron count or simulation count alone.

## Priority order

1. Training and attribution controls.
2. Browser compute efficiency.
3. Circuit selection and dynamics.
4. Scale and release.

## Workstream 1: Training and attribution controls

Status: partly complete.

Done:

- Six matched models: `male-cns`, `rewired`, `weight-shuffled`, `disconnected`, `linear`, and `mlp`.
- The `flygo experiment` runner with matched seeds, validation-selected checkpoints, provenance hashes, atomic output, and resumable runs.
- Encoder gradient correctness and checkpoint dynamics through export.
- Deduplication of board and side-to-move features across splits.

Open:

- Scale the teacher corpus beyond the pilot. [pipeline.md](pipeline.md) asks for millions of diverse positions for a serious result.
- Run the matched screening comparison on that corpus.
- Measure learning curves instead of a single final checkpoint.
- Keep the test split closed until the configuration is locked.

The highest-value first step is teacher distillation.
Diverse positions with soft policy targets and reliable values give the clearest signal per labelled position.
A topology advantage may appear as fewer required examples rather than higher final strength.

A later step is to label positions from FlyGo's own games with KataGo offline.
This targets the mistakes the deployed model actually makes and adds no server inference.

## Workstream 2: Browser compute efficiency

Status: not started.

- Profile the current 19x19 model in a real browser on named devices at the one-second budget.
- Record milliseconds per evaluation and simulations per second.
- Replace dense inference with a faster path if profiling justifies it: WASM with SIMD, or WebGPU.
- Build search children lazily instead of constructing every child position up front.
- Reuse the search tree across moves with correct history handling.
- Tune graph size, recurrent steps, and search budget together, and score strength per second.

Do not optimize before profiling.
The operation counts predict a dense-layer bottleneck, but a device measurement must confirm it.

## Workstream 3: Circuit selection and dynamics

Status: complete except for the feature decision below.

Done:

- `flygo select-circuit` selects circuits with one fixed rule and writes a selection manifest with the rule, its version, the source hash, and per-circuit diagnostics.
- The rule is connectivity-first, so every selected neuron has a connection inside the circuit, and larger circuits extend smaller ones.
- Circuit sizes are comparable with the existing runner, one output directory per size.
- `--normalization` selects incoming-strength normalization or none.
- `--weights` selects recorded weights or equal weights.
- `--steps`, `--retention`, and `--recurrent-gain` set the recurrent update and are recorded in provenance, in checkpoints, and in the browser manifest.

Open:

- Decide whether to add feature variants such as liberties, recent moves, and move history.
  A feature change alters the dataset shape, the policy binary, and the browser feature builder, so it needs a version bump and a conformance rebuild.
  Every model must receive identical features.
- Run the sensitivity grid and the circuit size sweep on a real corpus.
  Neither has run, because both are gated on the corpus.

Selection works on the real prepared graph, not only on the bundled sample.
Circuits of 500 and 1000 neurons keep a strongly connected core of 498 and 996 neurons, and only 2 and 4 neurons lose their outgoing edge.
The bundled sample cannot be used for this claim, because its largest strongly connected component stays at 14 neurons at every size.

The modeling choices remain assumptions.
More recurrent steps can saturate the state or make neuron states too similar.

## Workstream 4: Scale and release

Status: not started.

- Scale distillation to the full corpus.
- Run the final comparison at the locked configuration with `--final-test`.
- Build an external-engine league for a public strength claim.
- Add versioned opening files and paired color-swapped games.
- Measure browser strength versus time on named devices.
- Select a checkpoint, export it, and set the manifest to trained.
- Pass the release gate in [experiment.md](experiment.md).

## Gates

Each gate is a check that must pass before the next stage.

| Gate | Passes when |
| --- | --- |
| A. Screening on real data | All six models train on the same real splits, and validation policy and value metrics are reported for every seed without test access. |
| B. Topology attribution | A topology advantage survives the matched comparison and the circuit selection rule. |
| C. Release | The [release gate](experiment.md) passes and `web/public/flygo/manifest.json` reports `"trained": true`. |
| D. Public strength claim | An external-engine league reports paired color-swapped games, versioned openings, adjudications, and bootstrap intervals. |

A stronger player alone does not show that the biological wiring caused the improvement.
Keep the product result and the research result separate.

## Claim limits

Read [research.md](research.md) before writing any result.
A better MaleCNS result does not prove that biological computation transfers generally.
A worse result does not prove that the connectome lacks useful computation.
The committed untrained bundle is a software demonstration, not an experiment result.
