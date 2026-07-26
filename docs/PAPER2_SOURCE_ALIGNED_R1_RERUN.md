# Raised L=1.145, repetition-1 historical provenance

This directory preserves an earlier source-aligned rerun for the raised-terrain,
nominal-length-1.145, repetition-1 case. All executions used initial index
`(19, 11, 15, 1)` and the same case environment.

| Method | Historical Cmt | Time (s) | Historical recovery |
|---|---:|---:|---|
| Unrestricted discrete PPO | 0.267040040566968 | 1.87 | 0.80/1.08 |
| Continuous-torque PPO | 0.2780932427752225 | 1.99 | 1.00/0.89 |
| LIPM COM | 0.301544 | N.A. | 0.95/0.89 |

The raw stdout and frozen learned-controller source snapshots are retained
unchanged for Cmt, runtime, recovery-setting, and source-hash provenance. Their
`Foot_error` output is a deprecated historical reporter and is not the current
manuscript foot-placement metric. The authoritative metric is recomputed in
`supplementary/S4/landing_metric/`.

The current discrete and continuous entrypoints use the recovery settings
selected by the complete 12-case grid and therefore do not reproduce these two
historical learned-controller records. The current LIPM entrypoint remains
source-aligned. The current table generator uses only that LIPM override;
learned-controller rows come from
`results/paper2_push_off_grid_12case/`.
