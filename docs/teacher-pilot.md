# KataGo-labelled pilot

This pilot checks the real-data pipeline.
It is too small to establish playing strength, learning efficiency, or an advantage from biological wiring.
It does not replace the browser's untrained weights.

## Completed run

The dataset is available at `data/datasets/teacher-pilot-19/`.
All 244 requested positions received complete teacher answers at 256 visits.
Deduplication removed 23 repeated empty-board examples, leaving 221 training-ready examples.
There are no fallback game-result or recorded-move targets.

| Split | Games | Examples | Black to play | White to play |
| --- | ---: | ---: | ---: | ---: |
| Train | 21 | 195 | 94 | 101 |
| Validation | 2 | 18 | 8 | 10 |
| Test | 1 | 8 | 4 | 4 |

The [verification report](teacher-pilot/verification.json) records the exact input and output hashes.
The verifier rebuilt the dataset from the raw answers and matched every published split file.
Labelling finished with exit code zero on 2026-09-20, from 11:28:53 UTC to 11:54:40 UTC.
The 1,547-second duration includes engine startup but excludes initial tuning and the discarded trial.
Other checks ran concurrently, so this is not a performance benchmark.

## Inputs

The [source manifest](teacher-pilot/sources.json) pins every download, archive member, SGF hash, game ID, split, and analysis setting.
The games come from official KataGo distributed self-play, not human matches.
The source pool uses the teacher's training release and earlier b18 releases.

The selection takes the first 24 eligible games from the listed archives in order, with archive members sorted by filename.
It accepts only normal games, not forks or games started from supplied positions.
Each accepted game has a 19x19 board, 7.5 komi, a known winner, and a complete move sequence that FlyGo can replay.
No SGF moves, results, or komi were changed.
Selection occurred before labelling.
The game split contains 21 training games, two validation games, and one test game.

The original games use several rule sets.
KataGo re-evaluates every sampled position under FlyGo's Tromp-Taylor rules and 7.5 komi.
Original game results and SGF analysis comments are not training targets.

The pilot requests 244 positions at 256 visits each.
It samples turns 0, 33, 66, and so on while retaining the full move history.
The odd interval includes both Black and White to play.
An initial interval of 32 selected only Black, so that incomplete run was stopped and excluded.
The preparation script rejects even intervals.

### Permission and attribution

KataGo's [data-use policy](https://katagotraining.org/privacy/) says generated data can be made "publicly and freely available for download or use by anyone".
The games used here are publicly downloadable training data.
The teacher weights use the [KataGo Neural Network License](https://katagotraining.org/network_license/), not the repository's MIT license or an assumed CC0 license.
The engine uses the [KataGo engine license](https://github.com/lightvector/KataGo/blob/v1.18.1/LICENSE).

The preparation script retains hashed copies of the data-use policy and network license beside the raw inputs.
Downloaded games, weights, engine files, labels, and datasets stay outside Git.
Only the recipe, source hashes, configuration, and verification results belong in Git.

## Teacher and machine

- Engine: KataGo v1.18.1, official Windows x64 OpenCL release.
- Engine revision: `92ee95c0a4b25fec214da00951ab69e97e207729`.
- Network: `kata1-b18c384nbt-s9996604416-d4316597426`.
- Network SHA256: `9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d`.
- Host: WSL Ubuntu on an AMD Ryzen 5 6600H, with six logical processors exposed to WSL.
- Teacher device: Windows AMD Radeon integrated graphics, 2 GB, OpenCL device `gfx1035`, driver `3652.0`.
- Analysis: four simultaneous positions, one search thread per position, maximum neural batch size four.
- Numeric mode observed: FP16 storage and compute, without FP16 tensor cores.

The [analysis configuration](teacher-pilot/analysis.cfg) reports win rates from Black's perspective.
FlyGo converts those values to the player to move.
The script pins the configuration hash and refuses a changed local copy.
First use can spend several minutes tuning OpenCL kernels.
Keep the tuning cache under the extracted engine directory.

## Restore the inputs

Run these commands from the repository root under WSL with working Windows OpenCL support.
The scripts require `curl` and the project's Python environment.
Use a new raw-data directory when changing the source manifest, teacher, or configuration.

```bash
uv sync --frozen
uv run python scripts/prepare_teacher_pilot.py
uv run python -m zipfile -e \
  data/raw/teacher-pilot/katago.zip \
  data/raw/teacher-pilot/engine
chmod +x data/raw/teacher-pilot/engine/katago.exe
```

Preparation restores the selected SGFs, configuration, and `queries.jsonl`.
It verifies cached files before reusing them.
Downloads and new output files publish atomically, and a directory lock prevents simultaneous preparation writers.
It reads individual SGF archive members instead of extracting archive paths onto the filesystem.
An unexpected hash, query file, or extra SGF stops preparation instead of silently mixing inputs.

## Generate labels

Run the engine in the foreground and require a successful exit before importing its output.

```bash
root="$PWD/data/raw/teacher-pilot"
"$root/engine/katago.exe" analysis \
  -config "$(wslpath -w "$root/analysis.cfg")" \
  -model "$(wslpath -w "$root/teacher.bin.gz")" \
  < "$root/queries.jsonl" \
  > "$root/analysis.jsonl" \
  2> "$root/analysis.log"
```

Do not start another engine writer in the same directory.
Keep the raw answers and log after a successful run.
A restart can produce different floating-point answers because search and parallel execution are not guaranteed to be bit-for-bit deterministic.
The raw-answer hash identifies the exact labels used in a dataset.
Restoring the same inputs is not a promise of identical labels from a new analysis run.

## Import, build, and verify

```bash
uv run flygo teacher-import \
  --sgf data/raw/teacher-pilot/sgf \
  --analysis data/raw/teacher-pilot/analysis.jsonl \
  --output data/raw/teacher-pilot/targets.jsonl \
  --winrate-perspective black

uv run flygo build-dataset \
  --sgf data/raw/teacher-pilot/sgf \
  --teacher data/raw/teacher-pilot/targets.jsonl \
  --output data/datasets/teacher-pilot-19 \
  --size 19 --stride 33 --require-teacher

uv run python scripts/verify_teacher_pilot.py
```

The verifier checks exact requested-position coverage, final answers, and the visit budget.
It imports the raw answers again and rebuilds the dataset in a temporary directory.
The published manifest and all split-file hashes must match that fresh build.
Every split must contain positions for both players and only teacher targets.
No exact board features may occur twice across the dataset.
The extracted engine must match the executable inside the pinned release archive.

Dataset integrity checks read all three splits, but do not evaluate a model on the test split.
Keep test-set model scores closed until the research configuration is fixed.
The next stage is the [matched research comparison](research.md), followed by a much larger and more varied corpus before strength claims.
