# Experiment protocol

## Question

Can fixed topology from the official MaleCNS connectome provide useful computational structure for standard 19x19 Go policy-value learning?

This is a topology-transfer experiment.
It does not reproduce a fly brain or claim that a fly can play Go.
Use 19x19 for training and Elo.
Keep 5x5 for exact rules, solver, and regression validation.

## Primary objective

Test whether frozen biological wiring improves Go playing strength, data efficiency, or compute efficiency over matched randomized and conventional models.
Separate the strongest browser player from evidence that its biological wiring caused an improvement.
Use the [research foundation protocol](research.md) for matched controls, reproducible screening, and the limits of the resulting claims.

## Fixed evidence and modeling assumptions

The official MaleCNS release determines neuron IDs, directed endpoints, and connection strengths.
The initial graph filters to traced endpoints and connections with at least five detected synaptic contacts.
It treats connection weights as unsigned.

FlyGo defines rate-coded state, incoming-strength normalization, retention, recurrent gain, input placement, simulation steps, and optimization.
Record every choice in experiment metadata.
Do not convert predicted neurotransmitters to excitatory or inhibitory signs without a separate rule and sensitivity analysis.

## Models

### MaleCNS

Keep graph endpoints and weights frozen.
Train the board encoder, policy readout, and value readout through the recurrent dynamics.
The policy predicts 361 points plus pass.
The value predicts the result from the current player's perspective.

### Rewired controls

Randomly permute edge targets while preserving source and target degree sequences.
Run multiple seeds and report the complete distribution.
Add a separate weight-shuffled control when testing the contribution of connection strengths.
Use identical data, optimization budgets, and tuning effort for every topology.

### Conventional baselines

Train a linear model and a capacity-matched multilayer perceptron on the same features and targets.
Report trainable parameter counts, inference cost, and optimization budgets.
Do not claim a topology benefit without these controls.

## Go data

Represent a position as two occupancy planes and one player-to-move feature.
Represent policy targets as 362 probabilities and values in `[-1, 1]`.
Use a fixed KataGo binary, network, configuration, rule set, visit count, and win-rate perspective.
Record hashes and seeds.

Split complete games before extracting positions.
Never place positions from one game in more than one split.
Deduplicate board and side-to-move features globally before writing split files.
Apply the eight square-board symmetries only within training.

Human moves provide one-hot pretraining targets.
KataGo visits provide soft policy targets and KataGo win rates provide value targets.
Keep teacher analysis cached and immutable for a named dataset version.

## Rules

FlyGo uses Tromp-Taylor area scoring, positional superko, self-capture, and two-pass termination.
The implementation lives in `src/flygo/go.py` and is mirrored in `web/src/lib/go-rules.ts`.
Shared conformance fixtures cover capture, self-capture, ko, pass, scoring, komi, and replay.

Komi is 0.0 on 5x5 to preserve the published solved result.
Komi is 7.5 on 19x19.
Area scoring leaves dead stones on the board and therefore requires no dead-stone agreement.

## Training metrics

Report these metrics for every seed:

- Policy cross-entropy.
- Top-1 and top-3 teacher agreement.
- Value mean squared error.
- Legal-action rate before masking.
- Wall-clock training time.
- Peak memory.
- Checkpoint and graph hashes.

Use validation data for model selection.
Read the test split only for a configured final run.

## Playing-strength metrics

Run paired games with colors reversed on versioned openings.
Keep rules, komi, time control, opponents, and software versions fixed.
Report wins, losses, draws, adjudications, relative Elo, and paired-bootstrap 95% intervals.
Publish every game in the report.

The built-in league anchors the random agent at zero Elo and compares greedy and MCTS FlyGo.
Its Elo values are internal and cannot be compared with ratings from another opponent pool.
A public strength claim needs a fixed external-engine league under the same conditions.

Use one second per move for the production benchmark.
Use fixed simulation counts for deterministic regressions.
Do not enable root exploration noise during evaluation.

## Release gate

A production checkpoint must:

- Match the frozen graph hash.
- Improve validation policy and value metrics.
- Improve paired Elo with a confidence interval that supports the claim.
- Pass Python and browser conformance tests.
- Fit the static asset and browser memory budgets.
- Complete one-second searches without blocking the UI.
- Mark the 19x19 manifest entry as trained.

## Validity limits

A better MaleCNS result does not prove that biological computation transfers generally.
A worse result does not prove that the connectome lacks useful computation.
Results depend on graph selection, dynamics, teacher quality, features, optimization, model capacity, search, and baseline fairness.
The browser activity strip shows FlyGo state, not measured activity from a living fly.
The committed untrained bundle is a software demonstration and is not an experiment result.
