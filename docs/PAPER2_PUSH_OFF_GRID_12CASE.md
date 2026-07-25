# Paper2 learned-controller recovery-grid protocol

The current two-link multistep comparison uses 12 prescribed cases and three
learned controllers. For every controller-case pair, the post-impact
recovery-velocity decrements `r1` and `r2` were exhaustively evaluated on
`{0.00, 0.01, ..., 1.19} rad/s`. The sequence `[r1, r2, r1]` was applied across
the three transitions.

This gives:

- 120 by 120 = 14,400 candidates per controller-case pair;
- 36 controller-case searches;
- 518,400 released candidate records;
- seed 0 reset for NumPy and PyTorch before every candidate.

Eligibility requires completion of all three transitions, positive
center-of-mass displacement, and finite `Cmt`. Eligible records are ordered by
`Cmt`; exact ties use the `r1` and then `r2` index. The selected pair is replayed
in the unchanged fixed-parameter evaluator. Energy, displacement, and `Cmt`
match the corresponding search record within `1e-10` for all 36 replays.

The `raised_L1280_r2` proposed-selector optimum has `r2 = 1.19 rad/s` and lies
on the prescribed upper boundary. It is reported as a within-grid minimum and
does not establish global optimality beyond that range.

The learned-controller protocol differs from the MPC recovery search. MPC keeps
its documented 24 by 24 coarse grid and local 0.01-rad/s refinement. The shared
initial state, target sequence, termination rules, work interval, displacement
definition, failure handling, and reset-map equations remain unchanged.

## Verification

```bash
python src/paper2/push_off_grid/validate_release.py
python analysis/reproduce_paper2_tables.py
python -m unittest tests.test_reference_artifacts -v
```

The validator reselects the minimum from every candidate table, checks the 36
accepted replay rows, verifies the fixed entry-point parameters, and verifies
the scoped checksums.
