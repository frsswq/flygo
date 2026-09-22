# Feasibility teacher timing

The bounded timing run completed 100 positions at 256 visits each.
It used the pinned KataGo v1.18.1 executable, b18 teacher network, analysis configuration, and pilot queries from `feasibility-19-v1`.
The run started a fresh teacher process and included engine startup time.

| Measurement | Result |
| --- | ---: |
| Completed positions | 100 |
| Elapsed time | 962.49 seconds |
| Throughput | 0.1039 positions per second |
| Projected 10,000-position time | 26.74 hours |
| Projected raw analysis size | 87.81 MB |
| Projected query size | 4.75 MB |
| Peak child RSS reported by WSL | 95.46 MiB |

The memory value comes from `resource.getrusage(RUSAGE_CHILDREN)`.
It does not include Windows GPU memory and is not a complete machine-memory measurement.
The projection assumes similar positions, batch behavior, thermal state, and device availability.
It is sufficient for an approval decision, but it is not a throughput guarantee.

The complete report is stored outside Git at `data/raw/teacher-feasibility-19-v1/timing.json`.
Its SHA-256 is `40f1d63a868cbbb8cb6c1a55c3b99e4cb9b2d0085637da7c14fd16db8a8e6485`.
The raw analysis SHA-256 is `91035398fd77646c4dfcedf9df7a6263eeb7b001e718fe0dd0040c53f9cc9150`.
The query SHA-256 is `375bf86002d4ebe10a903822021c3820ca1681af093ce2a0bc9b7635b368b16a`.

The owner approved the 10,000-position target after reviewing this estimate.
Labelling will use independently completed shards so the owner can run it on and off.
No full labelling process was started during the timing checkpoint.
