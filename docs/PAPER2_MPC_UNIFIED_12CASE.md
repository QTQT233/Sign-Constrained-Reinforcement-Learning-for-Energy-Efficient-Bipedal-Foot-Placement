# Unified two-link MPC 12-case release

## Accepted result

The repository-relative continuous-torque MPC completed all 12 prescribed
three-step cases. The accepted mean `Cmt` is `0.198945098769`, with sample SD
`0.021963452200`. The pooled horizontal foot-placement MAE is
`0.030284639682 m` over 36 transitions, using
`X_foot = l1*cos(theta1) + l2*sin(theta2)` and the corresponding
target-foot position at angular-target entry. Two independent verification
replays reproduced every accepted scientific value exactly, and the validator
reports zero errors. Transition-level values and the metric validation report
are released under `supplementary/S4/landing_metric/`.

| Condition | n | Mean Cmt | Sample SD |
|---|---:|---:|---:|
| flat, nominal 1.280 m | 3 | 0.188705208 | 0.012507092 |
| flat, nominal 1.145 m | 3 | 0.208204140 | 0.016585376 |
| raised 0.01 m, nominal 1.280 m | 3 | 0.178283767 | 0.015569172 |
| raised 0.01 m, nominal 1.145 m | 3 | 0.220587280 | 0.018720893 |

## Search and acceptance rule

At each control update, the finite-horizon optimizer selects continuous torque.
For every case, the recovery search uses a 24 × 24 coarse grid with
0.05 rad s−1 spacing, followed by a 0.01 rad s−1 refinement within
±0.05 rad s−1 of the best coarse pair. If the best coarse point lies on a fine
window boundary, the adjacent fine window is also evaluated. Only complete
three-step trajectories are ranked; successful candidates are ordered by
`Cmt` and then energy. The accepted values are deterministic minima over the
specified grids.

## Current files

- `src/paper2/mpc/`: MPC dynamics, search, case runner, and validator;
- `configs/paper2_mpc_unified_12_cases.json`: 12 case definitions and source
  hashes;
- `results/paper2_mpc_unified_12case/accepted_cases.csv`: accepted case values;
- `results/paper2_mpc_unified_12case/accepted_summary.json`: aggregate result;
- `results/paper2_mpc_unified_12case/formal_validation.json`: integrity and
  replay checks; and
- `results/paper2_mpc_current_12_cases.csv`: manuscript-facing current table.

## Validation

```bash
python src/paper2/mpc/validate_release.py
python -m unittest tests.test_paper2_mpc_unified -v
```

The complete candidate-level search record, solver traces, and replay evidence
are supplied as Supplementary Archive S1.
