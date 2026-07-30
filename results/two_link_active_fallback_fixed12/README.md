# Two-link Active-fallback fixed 12-case check

This record applies the manuscript's fixed recovery-velocity pairs to the
transition-start router with three available branches: the non-positive expert,
the non-negative expert, and unrestricted Active PPO as fallback.  All 12
three-transition cases succeeded.  The sign route covered every transition, so
the Active fallback was not invoked and the mean `Cmt` remained
`0.122346306417`.

The controller-specific recovery values were selected on the released
`0.00:0.01:1.19 rad/s` grids.  Full selected-replay and candidate-grid evidence
is in `results/paper2_push_off_grid_12case/`.
