# Current two-link 12-case protocol

The manuscript uses 12 prespecified terrain/length/case checkpoints. All four
executed controller families use their case-specific initial state and the
shared target, termination, work-integration, displacement, failure-handling,
and reset equations. Recovery parameters are selected separately for each
controller and case. The three learned controllers use the complete grid
protocol below; MPC retains its documented coarse-to-fine recovery search.

## Current values

`results/paper2_current/paper2_combined_cases.csv` contains six records per
case: LIPM, TVLQR, discrete active PPO, continuous-torque PPO,
continuous-torque MPC, and the proposed one-sided selector. The four executed
controller means used in the primary comparison are:

| Method | Mean Cmt |
|---|---:|
| Proposed one-sided selector | 0.122346306417 |
| Discrete active PPO | 0.226620152405 |
| Continuous-torque PPO | 0.238107711255 |
| Continuous-torque MPC | 0.198945098769 |

The current primary CSV and Table V builder read the 36 fixed-parameter replay
rows from `results/paper2_push_off_grid_12case/accepted_cases.csv`.

## Learned-controller recovery grid

For every learned controller and case, `r1` and `r2` are evaluated on
`{0.00, 0.01, ..., 1.19} rad/s`. The sequence `[r1, r2, r1]` is applied over
three transitions. This gives 14,400 candidates per controller-case pair and
518,400 candidates in total. NumPy and PyTorch are reset to seed 0 before every
candidate.

Eligibility requires completion of all three transitions, positive COM
displacement, and finite `Cmt`. The eligible minimum on the prescribed grid is
replayed in the corresponding fixed evaluator. All 36 energy, displacement,
and `Cmt` values reproduce within `1e-10`. One proposed-controller pair
(`raised_L1280_r2`, `r2 = 1.19 rad/s`) lies on the upper boundary and is not
represented as a global optimum beyond the searched range.

## Initial-state and reset map

For each target, the compared controllers use the same initial index. In the
raised 0.01 m, nominal 1.145 m, repetition-1 case, the initial index is
`(19, 11, 15, 1)` for the proposed controller, discrete PPO, continuous PPO,
LIPM, TVLQR, and MPC. Controller-specific recovery parameters are part of the
measured `Cmt`; the learned-controller and MPC search grids are documented
separately.

## Reproduction

```bash
python src/paper2/push_off_grid/validate_release.py
python analysis/reproduce_paper2_tables.py
python src/paper2/mpc/validate_release.py
python -m unittest tests.test_paper2_mpc_unified -v
```

Complete MPC candidate tables, traces, and two deterministic replays are
distributed as Supplementary Archive S1.
