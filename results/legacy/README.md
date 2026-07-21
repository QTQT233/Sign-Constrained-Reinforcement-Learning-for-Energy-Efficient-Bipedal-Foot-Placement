# Legacy result boundary

`paper2_mpc_pre_unified_20260720.csv` is the exact pre-sync copy of the former
`results/paper2_mpc_current_12_cases.csv`. It mixes historical case artifacts
that were not all source-aligned and is retained only for provenance and
regression comparison.

Do not use it for current Tables IV–V. Current manuscript-facing MPC values are
derived from `results/paper2_mpc_unified_12case/accepted_cases.csv` and are
labelled **Continuous-torque MPC**.

The three `*_pre_source_aligned_r1_20260721.csv` files preserve the immediately
preceding current-table outputs. They predate the exact source-aligned rerun
records for the raised, nominal-length-1.145, repetition-1 Discrete and
Continuous PPO rows and are provenance snapshots, not current results. Their
Table V landing-error column is also the superseded signed mean; the current
table reports the mean absolute case-level landing error.
