# Feasibility supervised screen result

This page records the outcome of the first real-data feasibility screen.
The numbers come from the generated summary.
Regenerate it with:

```bash
uv run python scripts/summarize_research.py --runs data/research/feasibility-19-v1
```

## Verdict

No topology advantage was detected.

- Policy loss does not clear the predeclared 0.01 nat threshold against any control.
- Value error is higher than every control.
- All five controls were informative, so the comparison is usable.

A negative result is a valid outcome for this experiment.
`docs/plan.md` states that a topology advantage is not a prerequisite.

## Setup

- Dataset: `data/datasets/teacher-feasibility-19-v1`.
  It holds 10,105 examples built from 10,169 teacher labels, split into 8,403 train, 656 validation, and 1,046 test.
- Circuit: `data/processed/circuits-feasibility-v1/circuit-500.parquet`.
  It has 500 neurons, 22,502 edges, and a 498-neuron strongly connected core.
- Models: `male-cns`, `rewired`, `weight-shuffled`, `disconnected`, `linear`, and `mlp`.
- Seeds: 7, 17, and 27.
- Budget per run: 10 epochs, 8 recurrent steps, 660 optimizer updates.
- The closed test split was not read.
  Model selection used validation only.

## Held-out metrics at the selected epoch

Means over seeds.
Lower loss and higher agreement are better.

| model | policy loss | value MSE | top-1 | top-3 | legal | parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| male-cns | 5.3449 | 0.9436 | 1.22% | 5.03% | 0.423 | 543,000 |
| rewired | 5.3193 | 0.9197 | 1.78% | 4.07% | 0.374 | 543,000 |
| weight-shuffled | 5.3334 | 0.8758 | 1.52% | 4.01% | 0.378 | 543,000 |
| disconnected | 5.3207 | 0.8849 | 1.78% | 4.73% | 0.358 | 543,000 |
| linear | 5.3604 | 0.5990 | 0.71% | 2.24% | 0.433 | 262,449 |
| mlp | 5.3500 | 0.8493 | 1.88% | 4.32% | 0.347 | 543,000 |

The fly wiring is fourth of six on policy loss and last of six on value error.
The simple linear model is much better at value than every connectome model.

## Paired comparison against each control

`male-cns minus control`.
A positive value means the fly wiring is worse.

| control | policy difference | 95% interval | adjusted interval | clears threshold | value MSE difference |
| --- | ---: | ---: | ---: | :---: | ---: |
| rewired | +0.0256 | -0.0446 to +0.1053 | -0.0446 to +0.1053 | no | +0.0238 |
| weight-shuffled | +0.0114 | -0.0616 to +0.0981 | -0.0616 to +0.0981 | no | +0.0678 |
| disconnected | +0.0242 | -0.0084 to +0.0811 | -0.0084 to +0.0811 | no | +0.0586 |
| linear | -0.0155 | -0.0705 to +0.0442 | -0.0705 to +0.0442 | no | +0.3446 |
| mlp | -0.0051 | -0.0736 to +0.0648 | -0.0736 to +0.0648 | no | +0.0943 |

The adjusted interval is 99 percent with a Bonferroni correction across the five comparisons.
It is the interval used for the threshold decision.

## Control informativeness

An unchanged control cannot support attribution, so each one was checked.

| control | what changed | endpoint changes | graph differs |
| --- | --- | ---: | :---: |
| rewired | endpoints permuted | 22,421 to 22,451 | yes |
| weight-shuffled | weights reshuffled | 0 | yes |
| disconnected | no connectome edges | none | yes |
| linear | no connectome graph | none | n/a |
| mlp | no connectome graph | none | n/a |

## Limits

- This is supervised screening on held-out positions, not playing strength or Elo.
- Comparison is at validation only.
- Absolute quality is weak.
  Policy loss is 5.34 nats against 5.89 for a uniform guess, and top-1 agreement is about one percent.
- Three seeds give wide intervals.
  An unfavourable direction is not proof of harm.
- The locked final experiment requires a stronger teacher corpus, 99 percent intervals, and a fixed external-engine league.

## Provenance

- Experiment: `db0d1403aecf96cee92d2006d79a5e6cd21807daf5a723d5886102878708386c`
- Dataset manifest: `dbc388505ce5d70e591d692602d57c3afa0cafde05b67127f3e5bcc1c999c76e`
- Circuit file: `bf50ce0d323d96fe68b37b41326683ed7014e3d99d45607442d770cb222260d2`
- male-cns graph: `c49879859b16d2adc7a182f7f978359905e479b5e3963487a81ad89a8fdbbfdd`
- disconnected graph: `092795a56479a875` (truncated from the summary)
- `rewired` and `weight-shuffled` used a distinct graph per seed, as designed.

The machine-readable summary and the full per-run reports live under `data/research/feasibility-19-v1/`, outside Git.
