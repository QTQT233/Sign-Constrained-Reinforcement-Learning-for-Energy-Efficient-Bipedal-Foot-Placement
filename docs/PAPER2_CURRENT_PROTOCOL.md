# Current two-link 12-case protocol

The manuscript uses 12 prespecified terrain/length/case checkpoints. All four
executed controller families use their case-specific initial state and the
shared target, termination, work-integration, displacement, failure-handling,
and reset equations. Recovery parameters are selected separately for each
controller and case by the documented search protocol.

## Current values

`results/paper2_current/paper2_combined_cases.csv` contains six records per
case: LIPM, TVLQR, discrete active PPO, continuous-torque PPO,
continuous-torque MPC, and the proposed one-sided selector. The four executed
controller means used in the primary comparison are:

| Method | Mean Cmt |
|---|---:|
| Proposed one-sided selector | 0.123472869728 |
| Discrete active PPO | 0.228756976684 |
| Continuous-torque PPO | 0.240916227851 |
| Continuous-torque MPC | 0.198945098769 |

The source-aligned `raised_L1145_r1` values are 0.267040040566968 for discrete
active PPO and 0.2780932427752225 for continuous-torque PPO. The current primary
CSV and Table V builder both use these values.

## Initial-state and reset map

For each target, the compared controllers use the same initial index. In the
raised 0.01 m, nominal 1.145 m, repetition-1 case, the initial index is
`(19, 11, 15, 1)` for the proposed controller, discrete PPO, continuous PPO,
LIPM, TVLQR, and MPC. Controller-specific recovery parameters are selected by
the same search rule and are part of the measured `Cmt`.

## Reproduction

```bash
python analysis/reproduce_paper2_tables.py
python src/paper2/mpc/validate_release.py
python -m unittest tests.test_paper2_mpc_unified -v
```

Complete MPC candidate tables, traces, and two deterministic replays are
distributed as Supplementary Archive S1.
