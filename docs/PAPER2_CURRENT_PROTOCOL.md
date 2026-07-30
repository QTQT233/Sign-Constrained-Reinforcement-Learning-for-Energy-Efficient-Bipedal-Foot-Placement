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
| Proposed Active-default lookup router | 0.122346306417 |
| Discrete active PPO | 0.226620152405 |
| Continuous-torque PPO | 0.238107711255 |
| Continuous-torque MPC | 0.198945098769 |

The current `raised_L1145_r1` values are 0.250980549634431 for discrete active
PPO and 0.258172433323909 for continuous-torque PPO. The primary CSV, Table V
builder, and `results/paper2_push_off_grid_12case/accepted_cases.csv` use the
same accepted recovery-grid results.

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
