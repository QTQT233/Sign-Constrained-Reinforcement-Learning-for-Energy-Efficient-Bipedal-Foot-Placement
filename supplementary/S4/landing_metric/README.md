# Two-link landing metric

This component verifies the manuscript's reporting-only landing metric from
144 archived terminal-state/target-angle records. It does not rerun controller
actions or alter success, energy, center-of-mass displacement, recovery
parameters, or `Cmt`.

The horizontal swing-foot coordinate and transition residual are

```text
x_foot = l1*cos(q1) + l2*sin(q2)
e_k = x_foot,k - x_foot,k,target
```

where `l1 = 0.521 m` and `l2 = 0.481 m`. The primary result is the pooled
per-transition mean absolute error, which does not permit positive and negative
transition residuals to cancel.

Run from the repository root:

```bash
python supplementary/S4/landing_metric/code/test_foot_geometry.py
python supplementary/S4/landing_metric/code/recompute_landing_metrics.py
```

The verifier checks every recorded coordinate and residual, all 48
controller-case aggregates, and the four controller-level summaries. Generated
validation is written below `landing_metric/_generated/`.

The reported pooled per-transition MAE values are 0.044260 m for discrete PPO,
0.040953 m for continuous PPO, 0.030285 m for MPC, and 0.041389 m for the
proposed selector. These values do not support a best-landing-accuracy claim
for the proposed controller.
