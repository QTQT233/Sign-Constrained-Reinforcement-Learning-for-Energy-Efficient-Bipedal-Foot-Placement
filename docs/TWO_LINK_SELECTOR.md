# Two distinct two-link offline tables

The audited code contains two different objects that must not be described as
one generic "selector".

## Direct event-triggered action table

`working_save-ATC-50` is a 50^4 nearest-neighbour **state-action** table. Its
values -1, 0, and +1 are applied directly as -4, 0, and +4 Nm; -2 is a sentinel
that the consumers silently map to zero. The state index is refreshed whenever
the support angle changes by approximately 1.8 degrees, so the controller can
change nonzero torque sign within one transition.

A read-only replay found a successful trajectory that used +4 Nm, then zero,
then -4 Nm before reaching the target. Therefore figures produced by
`Energy_comparison.py`, `PID_tra.py`, or the `Qi_experiment-ATC` drawing scripts
cannot be cited as sign persistence "by construction". A representative trace
may happen not to switch sign, but that is an empirical property of that trace.

## Offline transition-locked policy-routing table

`working_save_passive-10-30` is used differently. The consumer queries it once
at the initial state, routes the transition to either a positive-only or
negative-only PPO policy, and keeps that policy until termination. This path
does enforce transition-level sign persistence.

However, the table is built offline by running both candidate policies and
using their outcomes. It is an outcome-informed initial-state routing map, not
an online comparison of two critic return estimates. The frozen producers do
not share a single success-priority or Cmt-tie rule: the primary flat producer
is negative-first, whereas the raised producer is positive-first and has
different source provenance. These cannot be collapsed into a universal
"lower-Cmt tie breaker." In deployment, only
stored value -1 selects the negative expert; +1, 0, -2, and unrecognized table
values fall through to the positive expert. The retained workflow does not
establish a held-out route-design/test split.

## Required manuscript distinction

Use separate names throughout:

- `direct state-action lookup` for `working_save-ATC-50`;
- `offline transition-locked policy routing` for `working_save_passive-10-30`;
- `legacy action-weight fusion` for its retained HDF5 row; its dual-success
  cells used the in-memory minimum, but single-success/first-action-zero cells
  were corrupted by a failed expert's initialized zero; and
- `learned online transition-onset selector` for the four-link experiment.

For a defensible offline selector, store an unambiguous policy ID and status,
query exactly once per transition, use a held-out evaluation-state manifest,
define uncovered-state fallback in advance, log the full torque sequence, and
assert that the nonzero sign-switch count is zero for every locked rollout.
