# Raised L=1.145, repetition-1 source-aligned rerun

The current Paper2 tables use exact rerun records for the raised-terrain,
nominal-length-1.145, repetition-1 Discrete and Continuous PPO cases. Both
executions used initial index `(19, 11, 15, 1)` and the same released case
environment.

| Method | Cmt | Time (s) | Landing error (m) | Recovery |
|---|---:|---:|---:|---|
| Discrete active PPO | 0.267040040566968 | 1.87 | 0.16735289815461774 | 0.80/1.08 |
| Continuous-torque PPO | 0.2780932427752225 | 1.99 | 0.13336367838846308 | 1.00/0.89 |

The Discrete source and repository entrypoint differ only by a terminal
newline, so the repository entrypoint reproduces that record. The Continuous
record is deliberately a separately named source-aligned snapshot: its source
uses first/second recovery decrements 1.00/0.89, whereas the retained GitHub
entrypoint uses 0.87/0.89. The snapshot therefore does **not** claim that the
retained entrypoint reproduces 0.2780932427752225, and the retained entrypoint
has not been overwritten.

Machine-readable values, source hashes, logical commands, environment versions,
byte-for-byte source snapshots, and raw stdout are in
`results/paper2_source_aligned_r1_rerun_20260721/`. The current table generator
applies only these two explicitly keyed overrides; all other non-MPC rows remain
unchanged.
