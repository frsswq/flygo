# Research and trained-release implementation specification

## 1. Purpose and authority

Deliver two separate outputs:

1. A reproducible real-data comparison of frozen MaleCNS wiring with all five controls.
2. A trained 19x19 browser player that passes measured correctness, strength, and resource gates.

A negative research result is a valid output.
A topology advantage is not a prerequisite for a trained release.
A successful release does not establish a topology advantage.

This document specifies remaining work, not work already completed.
`MUST` identifies a required contract.
`MAY` identifies optional work that must not block the required path.
The numerical defaults below are initial protocol and engineering requirements, not measured results or guarantees of statistical power.
Change them only in a versioned protocol before the affected experiment starts.
Never relax a gate after seeing a failure just to declare success.

Use [research.md](research.md) for model and metric definitions, [experiment.md](experiment.md) for scientific scope, and [pipeline.md](pipeline.md) for the existing data commands.
Use this specification for sequencing, new interfaces, and acceptance criteria.
Update those documents with any implemented contract change so they do not contradict this specification.
Use Beads for assignment, status, dependencies, and evidence links.
Milestone IDs here identify requirements, not a second task-status system.

### Implementation-agent rules

- Implement one milestone at a time and verify its acceptance criteria before starting dependent work.
- Inspect the named files and their callers before editing.
- Reuse existing data, training, search, and report functions instead of creating a parallel pipeline.
- Write regression tests before fixing an observed failure.
- Do not replace real-data, real-browser, or external-engine acceptance checks with synthetic tests.
- Do not claim a command exists merely because this specification proposes it.
- Do not run a paid job, start a multi-day job, publish a strength claim, or deploy without owner approval for that action.
- If required compute, data rights, a device, or an approval is missing, record the precise blocker in Beads and continue independent work.
- Do not manually edit generated fixtures, binaries, build output, or trained-status flags.

## 2. Reviewed baseline and corrections

The code review for this specification used revision `ddb4f06`.
The pilot evidence comes from the 2026-09-20 run documented in [teacher-pilot.md](teacher-pilot.md).
Historical check counts and deployment status are not acceptance evidence for a future revision.

| Capability | Existing implementation | Remaining work |
| --- | --- | --- |
| Teacher boundary | `teacher-queries`, `teacher-import`, strict teacher-only dataset option | Larger corpus, durable shard execution, scale measurements |
| Matched comparison | Six models, paired seeds, validation checkpoint selection, hashes, locks, resume | Real-data execution, training-set-size curves, final-test comparison summary |
| Learning history | Every epoch already appears in `result.json` | Curves across nested training-set sizes, not another epoch logger |
| Circuit selection | Deterministic selection and diagnostics in `circuit.py` | Run the real-corpus size and dynamics comparisons |
| Browser inference | TypeScript inference and worker-based PUCT | Real-browser profile, lazy children, safe tree reuse, conditional acceleration |
| Export | Sample-graph connectome checkpoints only | Explicit selected-graph support and end-to-end artifact verification |
| Arena | Python random, greedy, and MCTS league | External engines, browser participant, locked league protocol |
| Release evidence | Untrained committed bundle | Selected trained checkpoint and release report |

The pilot has 24 games and 221 examples after deduplication: 195 train, 18 validation, and eight test.
It is a pipeline fixture, not the screening corpus.
Both entries in the reviewed browser manifest have `trained: false`.
The review found no research output directory under `data/research/`.

### Correct operation counts

At 19x19, there are 723 input features and 362 actions.
For a connectome model with `N` neurons, `E` directed edges, and `S` recurrent steps:

- Trainable parameters and dense multiply-accumulates per evaluation: `N * (723 + 362 + 1) = 1086 * N`.
- Recurrent edge accumulations per evaluation: `S * E`.

| Property | Bundled sample | Selected 500 | Selected 1000 |
| --- | ---: | ---: | ---: |
| Neurons | 461 | 500 | 1000 |
| Edges | 605 | 22,502 | 53,332 |
| No outgoing edge | 304 | 2 | 4 |
| Largest strongly connected component | 14 | 498 | 996 |
| Dense multiply-accumulates | 500,646 | 543,000 | 1,086,000 |
| Edge accumulations at eight steps | 4,840 | 180,016 | 426,656 |

The selected-circuit connectivity counts are prior measurements, not a promise for a different source graph.
M2 MUST regenerate diagnostics and bind them to the source hash.
The old plan incorrectly applied the bundled sample's operation counts to selected circuits.
Neither these counts nor the sample profile proves which component dominates the selected circuit in a browser.

### Integration gaps that MUST NOT be skipped

- `train`, `benchmark`, `write_web_bundle`, and `load_web_policies` currently load `src/flygo/assets/malecns-sample.parquet` directly.
- `experiment --graph` already supports selected circuits, but such a checkpoint cannot currently follow the documented export path.
- Export supports `ConnectomePolicy`, not the `linear` or `mlp` checkpoint types.
- Export has one shared graph and one shared dynamics object for both board sizes.
- The Python bundle loader supports both normalization modes but still assumes the bundled sample graph.
- The browser loader parses binary layouts but does not yet verify every manifest hash and cross-file dimension at runtime.
- `report.json` currently summarizes paired validation differences only, even with `--final-test`.
- Python and browser traversal break equal PUCT scores differently: Python compares priors before action indices, while the browser currently compares action indices directly.
- Dataset construction materializes rows in memory, and training loads full split arrays.
Millions of examples require a resource measurement before assuming this format fits.

## 3. Fixed contracts

### Scientific contract

- Use 19x19, Tromp-Taylor area scoring, positional superko, self-capture, two-pass termination, and komi 7.5 for research and strength claims.
- Keep 5x5 with komi 0.0 for rules and regression checks.
- Freeze the source graph, circuit-selection rule, and circuit files before examining model scores.
- Keep connectome endpoints and weights frozen during training.
- Run `male-cns`, `rewired`, `weight-shuffled`, `disconnected`, `linear`, and `mlp` for every declared seed and configuration.
- Preserve matched initialization, minibatch order, augmentation, optimizer, and update budget within each comparison cell.
- Keep the existing parameter-count match between the connectome variants and MLP.
The linear baseline is not parameter-count matched.
- Keep two occupancy planes relative to the player to move, followed by the signed player-to-move feature.
Do not add liberties, recent moves, or history features in this scope.
- Keep policy targets as legal-action distributions and value targets in `[-1, 1]` from the player-to-move perspective.
- Keep whole games in one split and exact board/side-to-move features unique across the corpus.
This feature deduplication does not make move history irrelevant to Go legality or teacher analysis.
- Dataset integrity verification MAY inspect test arrays and hashes.
Model selection, screening, browser fixture selection, and tuning MUST NOT evaluate test targets.
- Before final test access, verify that no final-test feature row appeared in any earlier training, validation, tuning, or browser benchmark dataset.
An expanded corpus can change which game wins global deduplication, so stable game splitting alone does not prove this invariant.
- Keep all configured results, including failed, negative, and inconclusive runs.
Do not select favorable seeds or silently reduce the grid.

### Artifact and restart contract

New persisted artifacts MUST contain an integer `schema_version`, their input hashes, and the source revision.
Existing versioned artifacts MAY retain their established `version` field.
Reject unsupported versions, invalid counts, non-finite numbers, and inconsistent hashes at load boundaries.
Distinguish source-file SHA-256, effective graph-array SHA-256, checkpoint SHA-256, and exported-binary SHA-256.
They identify different objects and must not be substituted for each other.

Each job owns one output directory under an operating-system lock.
Publish completed files by atomic replacement and publish the completion manifest last.
A directory without a valid completion manifest is incomplete, not reusable evidence.
For a new multi-file corpus or release, stage a complete version in a sibling directory and publish it only after verification.
Do not overwrite an older completed version.

On restart, verify and reuse complete units.
Discard or restart only incomplete units.
A changed configuration or damaged completed artifact MUST fail with its path and mismatch reason.
Do not silently repair completed evidence in place.
The existing research runner's lock and identity checks remain mandatory.
New orchestration MUST also hash its own scripts and protocol, not only `src/flygo/*.py`.

### Initial protocol defaults

| Setting | Required initial value |
| --- | --- |
| Screening training rows | At least 100,000 after deduplication |
| Validation and test rows | At least 5,000 each after deduplication |
| Corpus diversity floor | At least 1,000 accepted games and both players represented in every split |
| Teacher visits and stride | 256 visits, stride 1 |
| Teacher win-rate perspective | Black, converted at import |
| Screening seeds | `7, 17, 27` |
| Confirmatory seeds | `7, 17, 27, 37, 47, 57, 67, 77, 87, 97` |
| Initial circuit | 500 neurons selected from the prepared graph |
| Training | 10 epochs, batch size 128, learning rate 0.001, value weight 1.0 |
| Dynamics | Eight steps, retention 0.35, recurrent gain 0.9, incoming normalization, recorded weights |
| Primary supervised outcome | Legal-masked policy cross-entropy |
| Secondary supervised outcome | Value mean squared error |
| Checkpoint selection | Lowest validation policy loss plus value MSE; earlier epoch wins a tie |
| Practical policy-loss threshold | 0.01 nats per position versus each control |
| Production search | 1,000 ms, PUCT exploration 1.5, no root noise |
| Release asset budget | At most 10 MiB uncompressed for the graph and the active 19x19 policy together |
| Release memory budget | At most 256 MiB attributable to the browser search worker, including its model and retained tree |
| Warm response budget | At most 1,100 ms at p95 from posting the move request to receiving its response |

These are minimum screening requirements, not sufficient evidence of broad generalization.
A final research corpus MUST have at least 1,000,000 deduplicated training rows and a documented diversity profile.
Scale targets do not authorize their cost.

## 4. Delivery order and dependency graph

| ID | Deliverable | Depends on |
| --- | --- | --- |
| M0 | Versioned protocol, resource estimate, and execution verifier | None |
| M1 | Verified screening corpus and resumable teacher execution | M0 |
| M2 | Real-data matched screen and training-set-size curves | M1 |
| M3 | Browser benchmark harness and baseline measurements | M0 |
| M4 | Lazy children and history-safe tree reuse | M3 |
| M5 | Measured inference acceleration, if needed | M3; use M4 for final comparison |
| M6 | Circuit-size and dynamics sensitivity results | M2 |
| M7 | Selected-graph export and artifact conformance | M0; M6 selects the eventual candidate |
| M8 | External-engine and browser league | M3, M7 |
| M9 | Locked final experiment and trained release decision | M1-M8; M5 may be explicitly skipped |

The priority remains training evidence, browser efficiency, circuit sensitivity, then scale and release.
M3 and M7 MAY proceed while teacher labelling runs.
No real-data claim can pass on pilot or synthetic output.
M8 infrastructure can be tested with an untrained bundle, but its release evidence must use the candidate artifact.

## 5. M0: Freeze the protocol and expose blockers

### Deliverables

Add a reviewed protocol at `docs/protocols/screen-19-v1.json` and a validator used by the execution scripts.
The protocol MUST name the corpus recipe, teacher identity, graph source, all defaults above, the M6 grid, and the claim scope.
It MUST also record:

- Source URLs, data-use permissions, source-selection order, exclusions, and sampling seed.
- Exact teacher binary, network, configuration, hashes, perspective, and rule settings.
- Training hardware, BLAS/thread settings, available RAM, disk allowance, and maximum approved job duration or cost.
- One named laptop or desktop and one named lower-power device for browser acceptance.
Record exact model, CPU/GPU, RAM, OS, browser version, and power mode when measured.
- The predeclared seed list, data-size list, stopping rule, practical threshold, and statistical comparison method.
- The paths and hashes of the immutable inputs once they exist.

Unknown required values remain explicit nulls in a draft protocol.
The validator MUST reject a draft for execution rather than infer a device, source, budget, or hash.
Dry runs MAY accept a draft but MUST list unresolved fields as blockers and MUST NOT launch jobs.
Pinning the existing pilot teacher is the default starting choice, not permission to infer its missing local files.

Estimate teacher time from a representative timed batch, not from the pilot's elapsed time alone.
Record completed positions per second, engine startup, hardware, concurrency, peak resident memory, and projected storage.
Run a small data-build and training smoke test to measure memory growth.
If the projected full build or training exceeds 70% of available RAM, stop scaling and implement bounded-memory loading first.
That prerequisite MUST preserve sample IDs, order, hashes, validation selection, and closed-test behavior.
Do not change the dataset format without versioning it and migrating all readers together.

### Acceptance

- Protocol validation rejects a missing hash, duplicate seed, even sampling stride, unsupported rule, and absent compute approval.
- A dry run lists all comparison cells, input identities, output paths, and estimated resources without starting the teacher or training.
- The verifier reports each gate as `pass`, `fail`, or `blocked`, with artifact paths and reasons.
It MUST never interpret a missing result as a pass.

## 6. M1: Build a larger teacher corpus

### Implementation surface

Reuse `dataset.py`, `teacher.py`, and the patterns in `scripts/prepare_teacher_pilot.py` and `scripts/verify_teacher_pilot.py`.
Add corpus-scale orchestration in `scripts/prepare_teacher_corpus.py` and `scripts/verify_teacher_corpus.py`.
Both new scripts MUST accept `--protocol PATH --output PATH`; preparation MUST also support `--dry-run`.
These are new interfaces, not existing commands.

### Required behavior

1. Select eligible complete SGF games using the locked source recipe before teacher analysis.
Record source diversity by archive/release, game length, sampled move number, and player to move.
A count alone is not a diversity report.
2. Reject incompatible games using the existing SGF/rules boundary.
Record rejection counts and reasons; never rewrite moves, komi, or results to make a game pass.
3. Generate full-history queries and partition them into stable shards of at most 100 games.
Use sorted game IDs and preserve original query IDs and turn numbers.
Never partition a game's move history into independent positions.
4. Give each shard its own queries, raw answers, stderr log, input manifest, and completion record.
Require teacher exit code zero, exact requested-position coverage, final answers, and the configured visit budget before publishing completion.
Handle out-of-order answers by game ID and turn number.
Reject conflicting duplicates, missing positions, teacher errors, and unexpected final answers.
5. On restart, reuse verified completed shards and rerun incomplete shards.
Do not append answers from a second engine run to partial output.
Do not promise bit-for-bit teacher regeneration; the saved raw-answer hash identifies the actual labels.
6. Import all verified labels with the pinned perspective and build a teacher-only dataset with `--require-teacher`.
The stride MUST match query generation.
Missing labels MUST fail even when deduplication would later discard that position.
7. Stage and verify the complete dataset before publishing its version directory.
Retain raw answers, logs, license records, and hashes outside Git.
Commit the small recipe and verification summary only when redistribution is permitted.

### Output and acceptance

Publish the screening dataset at `data/datasets/teacher-screen-19-v1/`.
It contains the existing three NPZ splits and manifest, plus a verification report linked to the source and teacher manifests.
The report MUST prove the row/game floors in section 3, nonempty splits, both players in each split, teacher-only targets, finite legal policy distributions, bounded values, and no duplicate feature rows within or across splits.
It MUST report the actual split proportions rather than claim exactly 80/10/10 for a finite corpus.

Rebuild the dataset from saved raw answers in a fresh temporary directory and compare split-file hashes.
Test a corrupt completed shard, missing label, duplicate answer, nonzero engine exit, interrupted publication, and concurrent execution.
All failures MUST leave earlier completed versions unchanged.
Dataset verification MUST NOT compute model scores on test examples.

## 7. M2: Run matched screening and data-size curves

### Existing commands to use

After M1 passes, select circuits from the prepared graph, not the bundled sample:

```bash
uv run flygo select-circuit \
  --graph data/processed/malecns-traced-w5.parquet \
  --output data/processed/circuits-screen-v1 \
  --nodes 250 500 1000

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run flygo experiment \
  --dataset data/datasets/teacher-screen-19-v1 \
  --graph data/processed/circuits-screen-v1/circuit-500.parquet \
  --output data/research/screen-19-v1/full \
  --seeds 7 17 27 --epochs 10 --batch-size 128 \
  --learning-rate 0.001 --value-weight 1.0 \
  --steps 8 --retention 0.35 --recurrent-gain 0.9 \
  --normalization incoming --weights weighted
```

The protocol wrapper MUST validate inputs before invoking these commands.
Never add `--final-test` to screening or sensitivity commands.

### Training-set-size experiment

Add `scripts/build_learning_subsets.py` with `--dataset PATH --output PATH --counts 10000 30000 100000 --seed 7`.
Build subsets from the published training split only, after global deduplication.
Rank training sample IDs by SHA-256 of the UTF-8 string `7:<sample_id>`, breaking a hash tie by sample ID.
Take the first requested number of rows in that order, so smaller sets are strict subsets of larger sets.
Use the same subsets for all model seeds.
Fail if the source has too few rows; do not silently shrink a requested subset.

Copy validation unchanged and retain its exact file hash.
Do not open or copy `test.npz` while creating these screening subsets.
Retain the source test entry as provenance only; mark the subset as screening-only in its additional metadata.
Each subset MUST have a valid dataset manifest for the existing runner plus its parent manifest hash, selection rule, selected-ID hash, and requested count.

Run every subset for all six models and all three seeds with the same ten-epoch settings.
Models within one subset see equal examples and optimizer updates.
Across subset sizes, ten epochs means different update counts.
Report those counts and call the result a fixed-epoch data-size curve, not an equal-compute comparison.
Retain the existing per-epoch learning histories.

### Reports and acceptance

Add `scripts/summarize_research.py --protocol PATH --runs PATH --output PATH` to validate and aggregate completed reports.
It MUST emit machine-readable results and a short Markdown summary with losses, top-1/top-3 target agreement, unmasked legal-action rate, parameter/operation counts, selected epochs, timing, and memory labels.
Do not call traced allocations total RAM or NumPy evaluation time browser latency.

For each configuration and data size, assert exactly one result for every configured model/seed pair, matching budgets and input hashes, finite metrics, and checkpoint hashes that verify on disk.
Report actual rewired endpoint changes and effective weighted/normalized graph differences.
An unchanged control MUST be identified as non-informative, not counted as successful attribution evidence.

Acceptance tests MUST include missing model results, mixed corpus hashes, damaged checkpoints, deterministic subset membership, unchanged validation hashes, and absent `test.npz`.
Run the subset builder and screen successfully with the test file inaccessible.
Use existing research resume tests and add an aggregation test that rejects incomplete runs rather than silently skipping them.
A screen passes when its evidence is complete, regardless of which model wins.

## 8. M3: Measure the real browser

### Harness contract

Add a repeatable browser benchmark entry point and runner under `web/bench/`.
It MUST serve the production static build and invoke the same bundle loader, policy evaluator, and worker search used by the application.
Do not use jsdom, Node timing, or offline NumPy timing as browser evidence.
Use a real browser automation tool and pin its dependency if the runner needs one.

Create a versioned position file from training games only, with full move histories and a file hash.
Include at least 30 legal nonterminal 19x19 positions: ten opening, ten middle-game, and ten late-game positions.
Include captures, ko-history constraints, and a single preceding pass.
Use shared synthetic fixtures separately for terminal and illegal-history checks.

Measure the bundled sample first, then repeat with the selected candidate after M7.
Report cold bundle load/parse/hash-validation time separately from warm search.
For each position, perform five warm-up searches and 20 measured searches with a fresh root.
Benchmark retained-tree sequences separately and report newly completed simulations, not inherited visits, as throughput.

Record:

- Device identity, browser version, power mode, source revision, artifact hashes, position hashes, and backend.
- Per-evaluation time for features/encoder, recurrent steps, policy/value heads, and total inference.
- Search time spent in rules/legal actions, node expansion, and selection/backup.
- Total new simulations, actual elapsed time, and simulations per second.
- Request-to-response p50/p95/max, deadline overshoot, cold-load time, and memory.
- Main-thread long tasks during search and a UI interaction trace while the worker is busy.

Use separate instrumented and uninstrumented runs so profiling overhead does not become the throughput claim.
Record the memory measurement method and scope.
A missing worker-memory measurement is `blocked`, not zero bytes.

### Acceptance and decision

Save raw samples and a summary at `data/browser/<run-id>/`.
Repeat on both named devices from M0.
Show that animation and a reset/board-size interaction remain responsive during a search, with no search-attributable main-thread task over 50 ms.
For the initial profile, a missed resource target is a measured baseline, not a reason to alter the report.

Rank optimization candidates by measured wall time, including rules and allocation costs.
Do not choose WASM or WebGPU from multiply-accumulate counts alone.
M5 is required only if the M4 candidate misses the release budgets or a prototype provides a verified end-to-end benefit.

## 9. M4: Reduce search allocation and reuse valid work

### Lazy child materialization

Change `src/flygo/search.py` and `web/src/lib/mcts.ts` together.
Represent legal outgoing actions with their prior, visit/value statistics, and an optional materialized child.
Expansion MUST compute every legal prior but MUST NOT construct and retain a full child position for every action.
Materialize a child on first selection and reuse it thereafter.
The rules engine may still construct temporary positions while checking legality; do not claim that lazy storage removes that cost.

Preserve legal masking, alternating value perspective, terminal scoring, and PUCT scoring.
Store an edge's value statistics from the resulting child's player-to-move perspective, so selection negates the edge mean at its parent.
Unvisited edges have zero mean value.
Use the same traversal tie rule in Python and TypeScript: highest PUCT score, then highest prior, then smallest action index.
This deliberately resolves the reviewed tie-rule mismatch; add an equal-score/different-prior regression in both languages.
Root selection remains highest visits, then highest prior, then smallest action index.
A one-simulation search MUST return a legal action.
Fixed-simulation fresh-root tests MUST match the pre-change action and visit distribution on non-tied fixtures.
Add exact tie, pass, capture, superko, and two-pass termination cases.

### Worker-owned tree reuse

Keep one search session inside each worker, not in React and not in a global cross-game cache.
A session owns its root, artifact identity, board size, dynamics, rules, exploration setting, complete move prefix, and pass count.
Continue using the complete move list in requests as the authoritative game history.

Reuse a subtree only when the new request extends the stored prefix exactly and every added action has a materialized descendant.
Otherwise rebuild from `replayMoves` with the full history.
Do not key reuse by board occupancy, side to move, or a board hash alone: positional superko depends on prior boards.
Reset, board-size change, artifact change, changed search settings, shorter history, divergent history, or an unavailable descendant MUST discard reuse.
Initialization MUST clear the session even when the board is empty in both games.

When promoting a root, release unreachable ancestors and siblings.
Retain valid node statistics but count only new simulations against the next request's budget.
Return activity for the current root even if it was expanded in an earlier request.
Do not assume that a reused-root search must select the same action as a fresh-root search with fewer total simulations.

Add a request-generation guard so asynchronous bundle loads cannot commit an older session after a newer initialization.
Keep request-ID response rejection in `use-flygo-game.ts`.
A reset or board-size change during a synchronous search MUST terminate and replace the worker, or otherwise cancel work before it can mutate the new session.
Ignoring only the response is not sufficient session isolation.

### Acceptance

Test prefix extension, opponent moves outside the stored subtree, divergent same-board histories, one pass, two passes, reset during search, rapid size changes, and stale initialization.
A controlled evaluator test MUST show fewer repeated evaluations after valid reuse and no reuse after invalidation.
An extended self-play sequence MUST remain within the memory budget without retaining ancestor trees.
Measure fresh-root and retained-tree performance with M3, rather than using a brittle CI timing assertion.
Implement an explicit node/memory cap if measurements show unbounded retained growth; reaching it MUST return a legal result and release the tree before another search.

## 10. M5: Accelerate inference only with evidence

Keep the current TypeScript evaluator as the correctness reference and universal fallback.
Prototype the two plausible accelerated paths, WASM SIMD and WebGPU, against the measured workload before choosing one.
Measure model upload, initialization, batch-size-one latency, synchronization/readback, and worker compatibility, not just matrix kernels.
Only the selected backend belongs in the shipped implementation.
Remove unused prototypes from production dependencies.

An accelerated path MUST:

- Consume the same graph, weights, features, dynamics, and legal-action interface.
- Run outside the main thread and select its backend once per worker session.
- Fall back cleanly when the browser lacks support, initialization fails, or the device is lost.
Discard any partial search before restarting on the reference backend within the remaining budget.
- Keep maximum absolute activity, logit, and value error below `1e-5` on shared fixtures and the M3 corpus.
Do not widen tolerances to hide a failed implementation.
- Preserve deterministic tie rules and chosen actions on the shared policy fixtures.
Search traces can differ when floating-point differences change a close score; record this rather than asserting universal bitwise equality.
- Improve warm end-to-end simulations per second by at least 10% on its target device over M4, without violating memory, response, or UI budgets.
Devices where it does not help MUST retain the reference backend.

If neither prototype qualifies and M4 meets the release gate, record M5 as skipped with measurements.
If no implementation meets the budget, record the release blocker rather than shipping a nominally faster kernel.

## 11. M6: Test circuit size and dynamics

Use the same screening corpus, seeds, optimizer settings, and features as M2.
Use a one-factor-at-a-time grid around the initial 500-neuron configuration:

| Factor | Values |
| --- | --- |
| Neurons | 250, 500, 1000 |
| Steps | 4, 8, 16 |
| Retention | 0.0, 0.35, 0.7 |
| Recurrent gain | 0.45, 0.9, 1.35 |
| Normalization | `incoming`, `none` |
| Weights | `weighted`, `binary` |

This is 11 unique configurations, not a Cartesian product.
Deduplicate the baseline configuration and reuse its verified result when identities match.
Run every configuration for all six models and screening seeds.
Use separate output directories, and label the grid as sensitivity evidence rather than an interaction study.

Record connectivity diagnostics, effective graph hashes, operation counts, and recurrent activity saturation.
Define saturation as the fraction of neuron activations with `abs(state) >= 0.99`, measured after each step on a fixed validation-position sample.
Report identical effective controls explicitly; binary weights can make weight shuffling redundant.
Do not count that redundant control as evidence about weight placement.

Choose the research configuration by the lowest seed-mean MaleCNS validation selection score, with ties broken by fewer parameters, fewer steps, then lexicographic configuration ID.
Apply the chosen configuration to the full matched final comparison.
This selection is exploratory and favors the research candidate; the untouched test comparison is needed for confirmation.
Report the complete grid so baseline tuning effort remains visible.

The release candidate may differ from the research configuration.
Choose it from validation-qualified candidates using paired browser strength at the same time budget, subject to memory and asset limits.
If practical strength is indistinguishable, prefer the smaller and faster candidate.
Lock a deterministic seed-selection rule before selection: use the seed with the lowest validation selection score, then the smaller seed on a tie.
Do not choose the release seed from test or league scores.

Acceptance requires all 11 complete cells, all seed results, selected-circuit manifests, unchanged features, and no test evaluation.
A divergent run is a recorded outcome and prevents that configuration from being selected; it must not disappear from the report.

## 12. M7: Export the actual trained graph

### New CLI contracts

Extend `export-web` and `benchmark` with `--graph PATH`, defaulting to the bundled sample for existing demonstration commands.
Extend `policy-conformance` with `--bundle PATH`, defaulting to `web/public/flygo/`.
Use `experiment` for selected-graph training; adding `--graph` to `train` is not required for this delivery.
Document that limitation instead of presenting `train` as the selected-circuit path.

Refactor `write_web_bundle` and `load_web_policies` so neither silently substitutes the sample for an explicit graph.
The exporter MUST verify the checkpoint against its exact effective graph, including weight mode and normalization.
For an experiment checkpoint at `<run>/<model>-<seed>/policy.npz`, resolve `<run>/request.json` and the adjacent `result.json`, then verify their identities and checkpoint hash before reading graph-transform choices.
If the checkpoint was moved, require an explicit `--experiment PATH` on export and benchmark rather than searching unrelated directories.
For standalone sample-graph checkpoints, retain the existing path only when their metadata is sufficient to verify the unmodified graph and incoming normalization.
Do not guess choices from a filename or use a CLI default that contradicts the checkpoint.
The internal benchmark MUST reconstruct the same effective graph before loading the checkpoint.

Keep one shared graph and dynamics object in a bundle for this release.
Generate the untrained 5x5 companion with exactly that graph and dynamics, and leave its trained flag false.
A trained 19x19 candidate MUST NOT make the 5x5 entry appear trained.
Reject linear and MLP export explicitly; shipping dense baselines is outside this scope.
Report their research performance even when they beat the deployable model.

### Artifact correctness

Update `export.py`, `policy_fixture.py`, `web/src/lib/policy.ts`, and their tests together.
The Python loader MUST reconstruct the graph from the exported artifact rather than compare it only with the bundled sample.
The browser loader MUST verify SHA-256, supported versions, rules, dimensions, finite arrays, edge-index bounds, normalization semantics, and consistency across manifest/graph/policy before inference.
Use browser Web Crypto for hashes.
Reject invalid artifacts with a user-visible worker error rather than returning a move from partial weights.

Preserve exact neuron identity without int32 truncation.
If selected source IDs exceed the current graph layout, version the graph binary and migrate writer, both readers, and fixtures together.
Do not change feature or policy binary versions unless their layouts change.
Version new manifest semantics so older readers reject an unsupported interpretation.

Record the source/circuit manifest hash, effective graph hash, checkpoint hash, dataset hash, experiment identity, selected seed/epoch, dynamics, and each exported binary hash.
Research checkpoints link to the dataset through their experiment identity, so export MUST resolve and verify that chain.
Do not fabricate a missing dataset hash or trained provenance.
The source description MUST identify a selected circuit when one was used.

Export into a staging directory and verify it before replacing the release bundle.
Add a test with non-default steps and retention so shared 5x5/19x19 dynamics cannot disagree silently.
Tests MUST cover wrong graph, wrong normalization, corrupt hash, truncated binary, invalid edge, non-finite weight, unsupported version, and mismatched policy dimensions.
Generate conformance from the actual staged bundle and verify activity, logits, value, and legal chosen action in a browser.
A selected-circuit checkpoint that reloads only in Python is not sufficient acceptance.

## 13. M8: Build a controlled external-engine league

Extend `arena.py` and add a narrow external-engine adapter plus a browser-worker participant.
Keep the existing internal `benchmark` behavior available as an internal diagnostic.
Add `flygo league --protocol PATH --output PATH` as the new versioned league interface.
The protocol MUST describe the candidate bundle, incumbent/comparator, at least two pinned external opponent configurations, opening file, rules, komi, seeds, time controls, move cap, adjudication policy, and bootstrap settings.
Two different visit budgets of one pinned engine MAY form the initial external pool, but must be labelled as such.

### Required behavior

- Drive external engines with an explicit argument array, not a shell command assembled from untrusted text.
- Pin executable, network, configuration, and hashes.
Verify rules support; never substitute an engine's default ko or suicide rules.
- Use protocol command IDs, bounded reads, stderr capture, startup/response deadlines, and guaranteed process cleanup.
- Use a fake engine for timeout, crash, malformed reply, out-of-order reply, and illegal-move tests.
- Run the browser candidate through the production worker, not a NumPy substitute.
Replay the full move history and apply the same validation as application play.
- Use 1,000 ms per FlyGo move and record actual wall time.
Record external visits/time budgets separately; unequal budgets permit opponent-specific strength results, not equal-compute claims.
- Use at least 100 distinct versioned openings chosen without test targets.
Play each matchup twice on each opening, swapping agent colors while retaining the same opening move sequence.
Do not use repeated deterministic games on one opening as independent evidence.
- Default to no resignation and a 1,083-move cap, counting opening moves.
Score two-pass termination with FlyGo's rules.
Mark cap-based area scoring as adjudication, not normal termination.
- Persist every move, actor identity, timing, result, pair/opening ID, termination reason, and adjudication status.
Save SGFs and machine-readable game records.
- Publish and resume complete color-swapped pairs under stable IDs.
If a crash leaves one half incomplete, rerun that incomplete pair and retain the failed attempt log.
Do not combine halves from different protocols.

For release evidence, protocol errors, illegal moves, or timeouts block the run rather than becoming hidden exclusions.
Adjudications above 5% block acceptance pending a predeclared protocol revision and a new run.
Report both all-game and non-adjudicated results, and retain an adjudication review record.

Fit relative Elo within this fixed pool only.
Resample color-swapped pairs together; when openings are reused across matchups or rounds, resample whole opening clusters to preserve their dependence.
Use 10,000 bootstrap samples with seed 0 and report 95% intervals.
For candidate-versus-incumbent improvement, compute the Elo difference within each joint resample instead of subtracting marginal interval endpoints.
Do not map the result to public rank or ratings from another pool.

Acceptance includes a complete small fake-engine tournament, a real external-engine smoke game, a resumed interrupted pair, and a full paired report using the actual browser artifact on the named reference device.
The small smoke tournament validates software only, not strength.

## 14. M9: Final evidence and release

### Lock before test access

Scale M1 to the approved final corpus and repeat validation-only selection on that corpus if required.
Give the final corpus its own version and verify its diversity and resource limits.
Maintain hashed sample-ID and feature-fingerprint inventories for all earlier training, validation, and browser benchmark inputs.
Before publishing the final corpus, deterministically exclude any previously exposed feature row from its test split and record exclusion counts and inventory hashes.
Do not move excluded test examples into training or validation.
Apply this rule before final training or model scoring, and include it in dataset rebuild verification.
If the remaining test set misses the minimum size, obtain more independent games instead of relaxing the exclusion rule.
Do not assume the screening run authorizes a changed corpus, teacher, or hyperparameter grid.
A data-efficiency claim MUST also include the M2 nested-subset method at multiple sizes of the final corpus.

Publish `docs/protocols/final-19-v1.json` before accessing final test scores.
It MUST lock the dataset, graph, configuration, ten seeds, checkpoint/seed selection rules, primary outcome, threshold, statistical method, and all allowed claims.
Store its hash in the final run's provenance.
Only this protocol authorizes `--final-test`, in a new output directory.
The dataset verifier MAY have read test arrays previously for integrity, but no model-score-driven choice may use them.

Extend the research summary to emit test comparisons separately from validation comparisons.
Do not relabel the existing validation comparisons as test evidence.
Use MaleCNS-minus-control differences, so negative loss differences favor MaleCNS.
For the five primary policy-loss comparisons, report paired-seed bootstrap intervals at 99% using 20,000 resamples and seed 0 as a Bonferroni family-wise adjustment.
Also report ordinary 95% descriptive intervals and every seed value.
State that seed resampling does not capture corpus-sampling uncertainty and ten seeds do not guarantee adequate power.

A positive supervised topology result requires the upper adjusted interval to be below `-0.01` nats against every control, informative effective controls, and no increase in seed-mean test value MSE against those controls.
Otherwise report negative or inconclusive evidence without widening the claim.
This result concerns supervised target prediction only; it does not establish browser strength, data efficiency, or general biological computation transfer.
Once test scores are read, any new tuning belongs to a new exploratory study and cannot reuse that test set as untouched confirmation.

### Release gate

Select the artifact using the locked validation rule, not test or league rankings.
Compare it with the frozen incumbent under the same graph/features/search where isolating training improvement is required.
For the first trained release, retain the current untrained bundle as the public incumbent and also evaluate an untrained initialization of the candidate architecture for the validation comparison.
Report these as separate comparisons when their graphs differ.

A trained release MUST satisfy all of the following:

1. Candidate validation policy loss and value MSE are both strictly lower than the untrained same-architecture baseline.
Use the same validation data and seed for that baseline.
2. The M8 candidate-minus-incumbent paired Elo interval has a lower bound above zero under the locked league protocol.
3. The exact candidate export passes Python/browser conformance, hash/provenance checks, and all repository checks.
4. M3 measurements of that export meet the asset, worker-memory, warm response, and UI budgets on both named devices.
A worker uses a deadline and may finish one in-flight simulation after it; the response budget, not an asserted exact 1,000 ms stop, determines acceptance.
5. The 19x19 manifest entry is generated as trained, and its provenance resolves to the selected persisted checkpoint.
The 5x5 entry remains correctly labelled.
6. A release report links the protocol, corpus verification, research result, league, browser measurements, conformance output, artifact hashes, and unresolved claim limits.

The final research result need not be positive for these release conditions to pass.
If a baseline wins the research comparison, say so in the release report.
Do not claim that a deployable connectome model is the best model when a non-exportable baseline performed better.

### Build and deployment verification

After promoting the verified staged bundle, run:

```bash
uv run flygo policy-conformance
make check
make build
make static
```

Regenerate rules fixtures only when rules changed.
`make build` updates the committed FastAPI-mounted output under `src/flygo/static/`.
`make static` builds the root-mounted deployment under `web/dist/`.
Test both mount paths, not only the Vite development server.

In a real browser, verify load, a legal 19x19 reply, pass, reset during search, size switching, self-play, stale-response rejection, and error display for a corrupt bundle.
Check the board and activity display for clipping or stale state while testing responsiveness.

Deploy only with owner approval using [deploy.md](deploy.md).
After deployment, download the live manifest and referenced binaries, compare their hashes with the release report, and play a browser smoke game.
Checking only that the live manifest says `trained: true` is insufficient.
Keep the prior release artifact so a failed smoke check can roll back without retraining.

## 15. Evidence gates and completion report

| Gate | Pass condition | Failure or inconclusive outcome |
| --- | --- | --- |
| A: Real-data screening | M1 and M2 meet all coverage, matching, provenance, and closed-test requirements | Repair missing evidence; do not claim a screen from synthetic data |
| B: Research conclusion | M6 and the locked final comparison are complete and honestly classified | Publish negative or inconclusive results; do not block product work solely on topology |
| C: Trained release | Every M9 release condition passes on the actual artifact | Keep the previous release and report the failed or blocked gate |
| D: Public strength claim | M8 supplies complete external paired games, intervals, budgets, openings, and adjudication review | Restrict the report to software or supervised results |

For each implemented milestone, report changed files, commands run, artifact paths and hashes, acceptance outcomes, and remaining blockers.
A successful unit test is not proof that a corpus was labelled, a league was played, or a device budget was met.
Missing hardware and insufficient compute are acceptable blockers; fabricated measurements are not.

## 16. Explicitly deferred work

- New feature planes or move-history input features.
- KataGo relabelling of FlyGo self-play for active data collection.
- General biological computation-transfer claims.
- Server-side production inference.
- Dense-baseline browser export, signed neurotransmitter models, and full-connectome deployment.
- A Cartesian sensitivity search or unbounded automatic hyperparameter tuning.

Start these only under a separate specification with versioned data, artifact, and conformance changes where required.
