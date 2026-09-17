# Bundled MaleCNS sample

`malecns-sample.parquet` and `malecns-sample-neurons.parquet` are derived from the official MaleCNS v1.0 files.
They exist only to make the research viewer runnable without a 1.1 GB download.

The sample starts with the first 16 traced neurons by body ID whose superclass is `cb_sensory` or `visual_projection`.
It follows each frontier neuron's four strongest outgoing connections for three rounds in the traced graph with minimum weight five.
The result contains 461 annotated neurons and 605 directed edges.

This sample is suitable for interface demonstration and integration tests.
It is not suitable for reporting experimental results.
