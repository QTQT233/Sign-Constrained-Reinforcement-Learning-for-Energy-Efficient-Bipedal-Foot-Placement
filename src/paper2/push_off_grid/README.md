# Two-link recovery-grid release

This directory validates the complete 12-case by three-controller recovery-grid
record used by the current manuscript. In the source programs the two positive
parameters are subtracted from the post-impact stance angular velocity. They
are therefore recovery-velocity decrements in `rad/s`, not forces.

For each learned controller and case, both parameters were evaluated on
`0.00, 0.01, ..., 1.19 rad/s`. The stance-tied sequence is `[r1, r2, r1]`
over three transitions. Each of the 36 controller-case searches contains
14,400 candidates, for 518,400 released candidate records in total. NumPy and
PyTorch were reset to seed 0 for every candidate.

A candidate is eligible only if it completes all three transitions, has
positive center-of-mass displacement, and has finite `Cmt`. The selected record
minimizes `Cmt` on the prescribed grid; exact ties are resolved by the `r1`
and then `r2` grid index. This is a within-grid minimum, not a claim of global
optimality. The proposed selector in `raised_L1280_r2` reaches the upper grid
boundary at `r2 = 1.19 rad/s`.

Run the portable validator from the repository root:

```bash
python src/paper2/push_off_grid/validate_release.py
```

The validator checks all 518,400 rows, reselects every minimum, verifies all 36
fixed replays, checks the selected values in the case entry points, and verifies
the scoped checksums. Use `--refresh-metadata` only when intentionally updating
the release.

The released candidate tables and accepted replay values are under
`results/paper2_push_off_grid_12case/`. Machine-specific generated search
scripts are not distributed: they embed workstation and Conda paths and require
additional case artifacts. Their omission does not limit rechecking the
released minimum-selection calculation, but this directory is not advertised
as a standalone model-training or full-dynamics rerun package.
