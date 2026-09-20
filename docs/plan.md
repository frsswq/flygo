# Remaining work

This document records the work that remains before the project can report a research result and ship a trained browser release.
It follows the priority order agreed during the 2026-09-20 review.
Update the status markers when a unit completes.
Keep the order unless a measured result changes it.

## Current state

- The full pipeline is implemented and checked.
- All repository checks pass: Ruff, BasedPyright, vulture, pytest (96 tests), Ultracite, TypeScript, knip, and Vitest (40 tests).
- The tracked backlog is empty.
- The only real-data run is the [KataGo pilot](teacher-pilot.md): 24 games, 244 labelled positions, 221 training examples.
- The browser bundle is untrained, and `web/public/flygo/manifest.json` reports `"trained": false` for both board sizes.
- No research report exists under `data/research/`.
- The matched comparison in [research.md](research.md) has run only on synthetic smoke data.

The code is not the constraint.
The constraints are training data, compute for teacher labelling, and the efficiency of browser inference.

## Known limits of the current artifacts

### Graph

| Property | Value |
| --- | ---: |
| Neurons | 461 |
| Directed connections | 605 |
| Neurons with no outgoing connection | 304 |
| Largest strongly connected component | 14 neurons |
| Trainable parameters at 19x19 | 500,646 |
| Recurrent edge accumulations per evaluation | 4,840 |

The bundled graph is a deterministic sensory-path sample.
304 of 461 neurons cannot send activity onward through connectome edges.
The largest strongly connected component holds 14 neurons.
The recurrent core is therefore small, although every neuron keeps its own state.

This artifact supports a demonstration and integration tests.
It does not support a strong claim about the biological connectome, because most of the graph cannot carry activity forward.

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

Status: not started.

- Replace the bundled demonstration sample with circuits selected by a fixed rule.
- Select circuits by connectivity and size before reading any test result.
- Compare several circuit sizes.
- Give every control the same selection and tuning budget.
- Test sensitivity to unsigned weights, incoming-strength normalization, recurrent steps, retention, and recurrent gain.
- Test added features such as liberties, recent moves, and move history, and give identical features to every model.

The current graph is a narrow test of biological topology.
A circuit with a larger connected core is a fairer test, but it needs a selection rule that cannot leak test information.

The modeling choices are assumptions, not biological requirements.
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
