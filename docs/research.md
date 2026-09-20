# Research foundation

## Objective and claims

Test whether frozen MaleCNS wiring improves Go playing strength, data efficiency, or compute efficiency over matched controls.
Keep the biological edge endpoints and measured weights fixed during optimization.
Do not treat a stronger trained player as evidence that the biological wiring caused the improvement.

The current experiment command provides supervised screening, not a playing-strength result.
It measures learning curves and held-out policy/value quality under the same example and optimizer-update budgets.
It does not yet measure a browser strength-versus-time curve or run an external-engine league.

Before a final run, record the primary outcome, practical effect threshold, graph selection, seeds, and tuning budget.
Use policy cross-entropy as the initial primary supervised outcome and value error as a secondary outcome.
A lower loss against one weak control is not enough to establish a topology advantage.
Use the complete comparison and report negative results.

## Matched models

Every experiment runs all six models for every requested seed.

| Model | Purpose | What stays matched |
| --- | --- | --- |
| `male-cns` | Original biological wiring | Reference model |
| `rewired` | Test endpoint organization | Nodes, directed degrees, edge count, source-associated weights, boundary parameters |
| `weight-shuffled` | Test weight placement | Nodes, endpoints, weight distribution, boundary parameters |
| `disconnected` | Test whether inter-neuron wiring helps at all | Nodes, input at every node, self-retention, recurrent steps, boundary parameters |
| `linear` | Conventional low-capacity baseline | Features, policy/value targets, optimizer, examples, augmentation |
| `mlp` | Conventional capacity-matched baseline | Features, targets, exact trainable parameter count, optimizer, examples, augmentation |

The MLP has one tanh hidden layer and the same hidden width as the connectome node count.
All models have bias-free policy and value heads.
The linear model uses the features directly and is not parameter-count matched.
The topology variants and MLP start with identical encoder and head arrays for each shared seed.
All models use the same minibatch order and symmetry choices for that seed.

Rewiring attempts ten directed edge swaps per edge.
It rejects new self-loops and duplicate directed edges and leaves existing self-loops fixed.
It preserves in-degree, out-degree, and outgoing strength, but not incoming strength.
This is a reproducible constrained sampler, not a uniform sample of all graphs with those degrees.
Inspect `changed_target_edges`; a small or constrained graph can remain unchanged.

Weight shuffling does not preserve each neuron's total input or output strength.
All topology variants recompute incoming-strength normalization from their frozen weights.
For a neuron with only one incoming edge, that normalization removes the effect of its edge magnitude.
A shuffled graph hash alone therefore does not prove that the effective dynamics changed.

The disconnected control has no connectome edges, including no biological self-loops.
Its update still includes the model's intrinsic self-retention and external input.
It is not equivalent to the one-layer MLP.

## Run supervised screening

Build a versioned dataset with nonempty training and validation splits first.
Use `flygo build-dataset --sgf PATH --output PATH` with optional `--teacher PATH` to build the splits.
Use `flygo teacher-queries --help` and `flygo teacher-import --help` for the KataGo analysis boundary.
The dataset builder splits complete games and removes repeated board/side-to-move features.

Run on a POSIX system with `uv` installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run flygo experiment \
  --dataset data/datasets/teacher-19 \
  --output data/research/screen-19 \
  --seeds 7 17 27 \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --steps 8
```

The default graph is the bundled 461-neuron, 605-edge MaleCNS sample.
Use `--graph PATH` to select another prepared graph before examining test results.
Use `--size 5` only for software and solver validation, not for a 19x19 strength claim.

## Select circuits before comparing sizes

The bundled sample is a demonstration, not a designed experiment input.
304 of its 461 neurons have no outgoing connection, and its largest strongly connected component holds 14 neurons.
A circuit fixes both problems by keeping one connected region of the prepared graph.

Select circuits with one fixed rule, and never from validation or test results:

```bash
uv run flygo select-circuit \
  --graph data/processed/malecns-traced-w5.parquet \
  --output data/processed/circuits \
  --nodes 250 500 1000
```

The rule starts at the neuron with the highest incident weight.
It then repeatedly adds the neuron with the strongest connection to the current set.
Ties break by incident strength and then by the smallest body ID.
Every selected neuron has at least one connection inside the circuit.
The rule does not depend on the target size, so larger circuits extend smaller ones.

`selection.json` records the rule, its version, the source hash, and each circuit's size, edge count, strongly connected size, and file hash.
Report that manifest with any result.

Compare sizes by running the same command against each circuit file:

```bash
for circuit in data/processed/circuits/circuit-*.parquet; do
  uv run flygo experiment \
    --dataset data/datasets/teacher-19 \
    --graph "$circuit" \
    --output "data/research/$(basename "$circuit" .parquet)" \
    --seeds 7 17 27
done
```

Give every size the same seeds, epochs, batch size, learning rate, and value weight.
The rule already removes a selection choice, so do not tune the rule after reading results.

Selection improves the wiring but cannot invent cycles.
On the bundled sample the largest strongly connected component stays at 14 neurons for every requested size.
Re-measure it on the real prepared graph before making a claim about recurrent depth.

## Modeling sensitivity

The dynamics and the weight interpretation are modeling choices, not biological facts.
Test them as a separate grid, with the same dataset, seeds, and budgets as the main screen.

```bash
uv run flygo experiment \
  --dataset data/datasets/teacher-19 \
  --output data/research/sensitivity/retention-0.7 \
  --seeds 7 17 27 \
  --retention 0.7 \
  --recurrent-gain 0.9 \
  --steps 8 \
  --normalization incoming \
  --weights weighted
```

`--steps`, `--retention`, and `--recurrent-gain` set the recurrent update.
`--normalization none` removes the division by incoming weight sum, so a neuron with one strong input receives its full drive.
`--weights binary` weights every present connection equally and tests whether connection strength matters beyond the wiring.

Every setting is recorded in `request.json` and changes the graph hash, so a resumed run cannot silently mix configurations.
Keep one output directory per configuration.

Three seeds provide an initial screen, not strong statistical evidence.
Increase the seed count for a serious comparison and keep every configured result.
The same seed selects the control graph, parameter initialization, minibatch order, and augmentation sequence.
The resulting paired-seed distribution includes both topology and optimization variation.
It does not separate those sources of variation.

Adam, gradient clipping, augmentation, batch size, loss weighting, and training epochs are shared across models.
The trainer restores the epoch with the smallest validation policy loss plus `--value-weight` times validation value error.
The default value weight is 1.0.
Equal scores select the earlier epoch.
Every epoch remains in the report even when its checkpoint was not selected.

For tuning, run the same predeclared configuration grid for every model with separate output directories.
Choose settings from validation results only.
Do not give the biological graph more graph-selection trials, seeds, or hyperparameter trials than its controls.

## Final test access

Without `--final-test`, the runner never opens `test.npz`.
It can run even when that file is absent.

After locking the experimental choices, use a new output directory and include `--final-test`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run flygo experiment \
  --dataset data/datasets/teacher-19 \
  --output data/research/final-19 \
  --seeds 7 17 27 \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --steps 8 \
  --final-test
```

This flag authorizes test evaluation for the entire configured comparison.
The test split does not participate in epoch selection.
Do not tune again after reading final-test results.

## Artifacts and repeatability

Each output directory contains:

- `request.json`: configuration, dataset-manifest hash, graph hashes, source hashes, and runtime details.
- `<model>-<seed>/policy.npz`: the selected checkpoint.
- `<model>-<seed>/result.json`: metrics, resource measurements, checkpoint hash, and full learning history.
- `report.json`: every seed result and paired-seed validation comparisons.

The runner verifies split hashes against the dataset manifest before training.
It reloads each saved checkpoint before computing its reported held-out metrics.
Connectome checkpoints bind to the exact derived graph hash.
Keep the source graph and dataset version with the report; the checkpoint does not contain the graph.

Repeating the same command verifies and reuses completed runs.
An interrupted run without a published result restarts deterministically from its seed.
Changed settings, data provenance, graph, source code, or recorded runtime require a new output directory.
A damaged completed checkpoint causes an error instead of silent replacement.

An operating-system file lock serializes writers to the same output directory.
The lock releases when a process exits, including after a crash.
Checkpoints and JSON reports publish through atomic file replacement.
Do not use an output directory on a filesystem without reliable file locks and atomic replacement.

## Metric definitions and limits

Policy cross-entropy uses legal-action masking and the full target distribution.
Top-1 and top-3 agreement compare legal masked predictions with the target's highest-probability action.
Equal scores use the smallest action index first.
These are target-agreement metrics, not necessarily teacher-agreement metrics: a dataset can mix human and KataGo labels.
The legal-action rate measures whether the unmasked top prediction is legal.
Value error is mean squared error from the side-to-move perspective.

Paired differences are MaleCNS minus each control, so negative loss differences favor MaleCNS.
The report resamples paired seeds 2,000 times with bootstrap seed 0.
It omits intervals for a single seed.
These intervals describe seed variation and are not corrected for multiple comparisons or dataset sampling uncertainty.
Small seed counts can give misleadingly narrow intervals.

Trainable parameter counts also give dense multiply-accumulate counts per evaluation for these bias-free architectures.
The report separately counts recurrent edge accumulations.
These operation counts do not include activations, Go rules, or search.
Equal optimizer-update budgets are not equal wall-clock or inference-compute budgets.

Training time includes allocation tracing and validation selection.
Peak traced bytes measure allocations during training, not total process resident memory or browser memory.
Evaluation timing measures offline, batched NumPy inference and metric computation.
It is not browser inference latency or one-second MCTS throughput.
Hardware, BLAS implementation, threading, and load can change timings even when predictions reproduce.

## Evidence still required

The committed browser model remains untrained until a trained checkpoint is explicitly exported.
Synthetic smoke tests prove software behavior, not Go strength or biological benefit.
A useful next result needs a documented, diverse training corpus and a fixed teacher configuration.

A playing-strength claim also needs paired color-swapped games, versioned openings, an external-engine opponent pool, adjudication review, and confidence intervals under fixed budgets.
A browser efficiency claim needs measurements on named target devices with a one-second move budget.
A data-efficiency claim needs matched learning curves across several training-set sizes.
Keep those claims separate from this supervised screening report.
