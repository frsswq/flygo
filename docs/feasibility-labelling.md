# Resumable feasibility labelling

The feasibility corpus contains 31 one-game shards with 10,169 requested positions.
Each shard is independent and publishes `complete.json` only after the teacher exits successfully and every final response passes verification.

## Check progress

```bash
make label-status
```

The status reports completed and remaining shards and positions.
It verifies every completed shard before counting it.

## Complete one shard

```bash
make label
```

This command processes the first incomplete shard and then exits.
Run it again whenever the laptop is available.
Do not start two labelling commands at the same time.
A filesystem lock rejects a concurrent command.

You can stop the current command with `Ctrl+C`.
The runner kills the active teacher process and does not publish that shard as complete.
The next invocation reruns only that incomplete shard.
Previously completed shards remain unchanged.

The first real shard completed 304 positions in 1,243.77 seconds, or 20.73 minutes.
Its completion-manifest SHA-256 is `de000de10bec0c3d3210d3d8af83adbd39578bd59b1ec4348cfed21d33ccb96d`.
Its raw-analysis SHA-256 is `6f6f066eebb14d98f4d0f96835b17a7c9690ea2ceb6a3b2dd7d2e568d78a7bfb`.
The remaining runtime will vary with game length, batching, temperature, and other device use.

## Stored evidence

All SGFs, queries, raw answers, logs, and completion manifests stay outside the main repository Git history under `data/raw/teacher-feasibility-19-v1/corpus/`.
A local-only nested Git repository at `data/` tracks the corpus labels and the published dataset for protection against accidental loss.
That repository has no remote and must never be pushed.
See `data/DATA-REPO.local.md`.
The committed source manifest pins 82 official archive files and the captured source pages by SHA-256.
The prepared selection-manifest SHA-256 is `b7c0946f55961debdff89e297eabe46cc517587db00ac1cd09a7a7e2b78f2b76`.
The selection takes archives in manifest order and SGF members in filename order.
It selected 26 training games, two validation games, and three closed-test games before teacher labelling.

Do not use the test split for feasibility model selection.
Dataset construction and matched training remain blocked until all shards complete and the corpus verifier passes.

## Finalize the dataset

After status reports no remaining shards, run:

```bash
make label-finalize
```

This command refuses incomplete or damaged shards.
It merges verified raw answers, imports teacher targets, builds the dataset in a staging directory, verifies every split, and publishes the directory atomically.
It does not evaluate any model or read test targets for model selection.

The fixed 500-neuron circuit is already selected at `data/processed/circuits-feasibility-v1/circuit-500.parquet`.
Its SHA-256 is `bf50ce0d323d96fe68b37b41326683ed7014e3d99d45607442d770cb222260d2`.

## Result

After the dataset is published, the matched screen and its summary are recorded in the [feasibility result](feasibility-result.md).
Regenerate the summary from the completed runs with:

```bash
make summarize
```
