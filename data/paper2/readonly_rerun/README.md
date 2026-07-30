# Historical read-only Paper2 rerun

This directory preserves the read-only baseline rerun and its raw logs for
provenance. `paper2_overall_summary.csv` is not the source of the current
manuscript's learned-controller means.

The current table builder reuses the LIPM records from this directory. The
discrete-PPO, continuous-torque-PPO, and proposed-router values are the accepted
complete recovery-grid minima under
`results/paper2_push_off_grid_12case/`; MPC values are under
`results/paper2_mpc_unified_12case/`. The synchronized manuscript-facing table
is `results/paper2_current/table_v_current.csv`.
