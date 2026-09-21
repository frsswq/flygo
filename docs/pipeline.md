# Training pipeline

Training and benchmarking run offline.
Cloudflare receives only the static application, graph, and exported policy-value weights.

The [pinned KataGo pilot](teacher-pilot.md) provides a small, real-data recipe for checking this pipeline.

## 1. Collect SGF games

Use legally reusable 19x19 SGF records with 7.5 komi and a black or white result.
FlyGo reads every game in an SGF collection but follows only its main variation.
It rejects setup stones, unsupported sizes, wrong komi, invalid turn order, illegal moves, games without a winner, and moves after two passes.

Keep source licenses and download metadata beside the raw corpus.
Do not commit a corpus unless its license permits redistribution.

## 2. Generate KataGo queries

Create one asynchronous analysis query per complete game:

```bash
uv run flygo teacher-queries \
  --sgf data/sgf \
  --output data/teacher/queries.jsonl \
  --visits 256 \
  --stride 1
```

Run a fixed KataGo binary, network, and analysis configuration:

```bash
katago analysis \
  -config data/teacher/analysis.cfg \
  -model data/teacher/katago.bin.gz \
  < data/teacher/queries.jsonl \
  > data/teacher/analysis.jsonl
```

Record the KataGo version, network hash, configuration hash, GPU type, and command.
For a small pilot, increase `--stride` to analyze fewer positions without removing move history.
Use an odd stride to include both players; an even stride samples only Black to play.
Use the same stride when building the dataset.
Queries use the board's fixed komi: 7.5 for 19x19 and zero for 5x5 validation.
KataGo can return positions out of order, so FlyGo joins results by game ID and turn number.

KataGo reports win rates from the perspective configured by `reportAnalysisWinratesAs`.
Import with the matching value:

```bash
uv run flygo teacher-import \
  --sgf data/sgf \
  --analysis data/teacher/analysis.jsonl \
  --output data/teacher/targets.jsonl \
  --winrate-perspective black
```

Accepted perspective values are `black`, `white`, and `side-to-move`.
The importer converts every value to the current player's perspective.
It converts search visit counts into normalized policy targets.

## 3. Build split datasets

```bash
uv run flygo build-dataset \
  --sgf data/sgf \
  --teacher data/teacher/targets.jsonl \
  --output data/datasets/teacher-19 \
  --size 19 \
  --stride 1 \
  --require-teacher
```

FlyGo assigns complete games to deterministic 80/10/10 train, validation, and test splits.
No position from one game can enter another split.
Each NPZ split contains board features, legal-action masks, policy targets, value targets, sample IDs, and target-source markers.
The manifest records hashes, counts, rules, split logic, and teacher provenance.

A teacher target replaces the human move and game-result value for that position.
With `--require-teacher`, missing labels stop the build before it writes any files.
This check includes sampled positions that deduplication would later remove.
Without this flag, a position without a teacher target keeps a one-hot human policy and final game result.
Training applies one of eight square-board symmetries dynamically.

Use `--stride` to sample every Nth position when building an initial dataset.
A first functional run can use about 100,000 positions.
A serious result should use millions of diverse positions and a documented sampling policy.

## 4. Train the policy-value model

```bash
uv run flygo train \
  --dataset data/datasets/teacher-19 \
  --output data/models/flygo-19.npz \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --steps 8 \
  --seed 7
```

The MaleCNS graph endpoints and weights remain frozen.
Adam updates the board encoder, policy readout, and value readout through all recurrent steps.
The policy loss uses soft cross-entropy.
The value loss predicts the final result from the current player's perspective.
The checkpoint records the graph hash, dataset-manifest hash, hyperparameters, seed, and per-epoch losses.
A checkpoint cannot load against a different graph.

Run multiple seeds and retain every configured result.
Do not select a checkpoint from test-set performance.

## 5. Run paired Elo tournaments

```bash
uv run flygo benchmark \
  --policy data/models/flygo-19.npz \
  --output data/reports/flygo-19.json \
  --seconds 1 \
  --rounds 100 \
  --bootstrap-samples 500 \
  --seed 7
```

The internal league contains random, greedy FlyGo, and MCTS FlyGo agents.
Every opening is played twice with colors reversed.
Provide a versioned JSON array of action arrays with `--openings` to reduce opening and first-player bias.

The report contains every game, all run settings, policy hashes, relative Elo ratings, and paired-bootstrap 95% intervals.
The random agent is fixed at zero Elo.
Elo values are comparable only when board size, rules, opponents, openings, time control, and software versions are identical.
Use `--simulations` instead of `--seconds` for deterministic regression runs.

The arena area-adjudicates games that reach `--max-moves` and marks them as adjudicated.
Review this count before accepting a report.

## 6. Export and verify the browser artifact

```bash
uv run flygo export-web --policy data/models/flygo-19.npz
```

Use `--normalization none` when the checkpoint was trained without incoming-strength normalization.
Export rejects a checkpoint whose graph normalization does not match this option.

```bash
uv run flygo policy-conformance
make static
make check
```

The export includes the frozen graph, encoder, policy readout, value readout, dynamics, hashes, and checkpoint metadata.
The browser loads one board-size policy and runs one-second PUCT search in a Web Worker.
The worker keeps search computation off the UI thread.

Inspect `web/public/flygo/manifest.json` and verify that the 19x19 entry has `"trained": true` before deployment.
The committed bundle remains an untrained demonstration unless a trained checkpoint has replaced it.
