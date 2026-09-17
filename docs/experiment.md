# Experiment protocol

## Question

Can the fixed topology of a real biological connectome provide useful computational structure for supervised Go policy learning?

This is a topology-transfer experiment.
It is not an attempt to reproduce a fly brain or claim that a fly can play Go.
Board size is a parameter from 5x5 through 9x9.
The exact 5x5 solver is the primary teacher, because 5x5 is fully solved and its value targets are exact.

## Fixed evidence and modeling assumptions

The official MaleCNS release determines neuron IDs, directed endpoints, and connection strengths.
The first experiment filters the graph to traced endpoints and connections with at least five detected synaptic contacts.
The first experiment treats all connection weights as unsigned.

FlyGo defines rate-coded state, incoming-strength normalization, decay, input placement, simulation steps, and optimization.
Each such choice must appear in experiment metadata.
Predicted neurotransmitters must not be converted to excitatory or inhibitory signs without a separate stated rule and sensitivity analysis.

## Models

### MaleCNS

Keep the prepared edge endpoints and weights frozen.
Train the board encoder and policy readout only.
Use biologically annotated sensory populations for input when coverage permits.
Use descending or motor-related populations for readout experiments when coverage permits.

### Rewired control

Randomly permute edge targets while preserving the source and target degree sequences.
Run multiple seeds.
Report the distribution rather than one selected run.
Add a separate weight-shuffled control if connection strength is part of the claim.

### Conventional baseline

Train a linear classifier and a small multilayer perceptron on the same board features and labels.
Report trainable parameter counts and compute budgets.
Do not claim topology benefit unless the comparison controls capacity and tuning effort.

## Go data

Represent each position as two occupancy planes of `size * size` points and one player-to-move feature.
Represent policy targets as `size * size` board points plus pass.
Generate labels with a fixed, versioned teacher.
Record the ruleset, komi, engine version, search settings, and random seed.

Split complete games before extracting positions.
Never place positions from one game in more than one split.
Deduplicate positions across train, validation, and test sets.

## Rules

The ruleset is Tromp-Taylor: area scoring, positional superko, self-capture allowed, and two consecutive passes end the game.
The implementation lives in `src/flygo/go.py`.
`tests/test_go.py` checks capture, self-capture, ko, pass, area scoring, and komi.
Area scoring counts a player's stones plus every empty point that reaches only that player's stones.
Dead stones stay on the board, so no dead-stone agreement is needed.
Komi is 0.0 on 5x5, so the published solved result of Black +25 at komi 0 holds.
Komi is 7.5 on 6x6 through 9x9, which is the value Chinese rules use on 9x9.
Change `KOMI_BY_SIZE` in `src/flygo/go.py` to test another komi.
The 6x6 entry is untuned and may favour White, because published komi estimates for 6x6 are near 3.
The server replays the full action list of a game on every request, so superko covers the whole game instead of the previous position only.
The server replays the full action list of a game on every request, so superko covers the whole game instead of the previous position only.
The viewer keeps the rules engine unchanged and decides only when White answers.
White answers every stone that Black places.
When Black passes, White passes back, because the untrained readout ranks the pass action near last on 9x9 and the score would otherwise be unreachable.
The two consecutive passes then end the game.

## Metrics

Report these measures across seeds:

- Top-1 teacher-move accuracy.
- Top-3 teacher-move accuracy.
- Legal-action rate before masking.
- Cross-entropy loss.
- Win rate against fixed opponents.
- Wall-clock time and peak memory.

Use confidence intervals for differences between MaleCNS and randomized controls.
Publish all configured runs, including negative results.

## Validity limits

A better result for MaleCNS does not prove that biological computation transfers generally.
A worse result does not show that the connectome lacks useful computation.
Results depend on graph filters, dynamics, input and output populations, teacher quality, optimization, and baseline fairness.
The web viewer lets a human play Black while the frozen graph answers automatically as White.
The activity strip shows FlyGo dynamics, not measured activity from a living fly.
