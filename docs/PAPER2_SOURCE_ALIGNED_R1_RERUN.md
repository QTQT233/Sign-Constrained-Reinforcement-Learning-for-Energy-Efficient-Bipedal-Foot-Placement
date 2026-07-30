# Raised L=1.145, repetition-1 source-aligned rerun

This directory preserves the 2026-07-21 source-alignment rerun for the
raised-terrain, nominal-length-1.145, repetition-1 Discrete PPO,
Continuous-torque PPO, and LIPM cases. All executions used initial index
`(19, 11, 15, 1)` and the same case environment. These records are historical
provenance; they are not the selected minima used by the current manuscript.

| Method | Cmt | Time (s) | Landing error (m) | Recovery |
|---|---:|---:|---:|---|
| Discrete active PPO | 0.267040040566968 | 1.87 | 0.16735289815461774 | 0.80/1.08 |
| Continuous-torque PPO | 0.2780932427752225 | 1.99 | 0.13336367838846308 | 1.00/0.89 |
| LIPM COM | 0.301544 | N.A. | N.A. | 0.95/0.89 |

The Discrete source and repository entrypoint differ only by a terminal
newline, so the repository entrypoint reproduces that record. The Continuous
entrypoint has now been synchronized to the accepted 1.00/0.89 source, and the
LIPM entrypoint to the accepted 0.95/0.89 source. Their repository copies and
accepted local sources are text-identical apart from line-ending normalization.

Machine-readable values, source hashes, logical commands, environment versions,
source snapshots or pinned repository entrypoints, and raw stdout are in
`results/paper2_source_aligned_r1_rerun_20260721/`.

The current manuscript uses the accepted complete push-off-grid minima in
`results/paper2_push_off_grid_12case/` and
`results/paper2_current/paper2_combined_cases.csv`. For `raised_L1145_r1`,
these are 0.250980549634431 for discrete active PPO and 0.258172433323909 for
continuous-torque PPO.
