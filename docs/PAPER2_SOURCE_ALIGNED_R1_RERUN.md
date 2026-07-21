# Raised L=1.145, repetition-1 source-aligned rerun

The current Paper2 tables use exact rerun records for the raised-terrain,
nominal-length-1.145, repetition-1 Discrete PPO, Continuous-torque PPO, and
LIPM cases. All executions used initial index `(19, 11, 15, 1)` and the same
case environment.

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
`results/paper2_source_aligned_r1_rerun_20260721/`. The current table generator
applies only these three explicitly keyed overrides; all other non-MPC rows remain
unchanged.
