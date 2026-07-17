# Cmt metric used by the two-link action-weight evaluators

## Definition and interpretation

For applied internal hip torque `tau_h` in N m, relative hip angular speed
`omega_h = theta_dot_2 - theta_dot_1` in rad/s, and sample period `dt` in s, the
evaluators accumulate positive commanded mechanical work as

```text
W_h^+ = sum_t max(tau_h,t * omega_h,t, 0) * dt .
```

They then report the dimensionless cost of mechanical transport

```text
Cmt = W_h^+ / ((m_1 + m_2) * g * abs(x_com,end - x_com,start)).
```

The final Appendix D action-weight audit admits a Cmt value only when the center-of-mass displacement is
strictly greater than `0.01 m`. The canonical evaluators save the displacement
arrays as `D_save*` HDF5 files so the denominator screen can be audited without
reconstructing a trajectory.

Thus, for a shared denominator and evaluation protocol, a lower Cmt means less
positive commanded mechanical work per unit body weight and center-of-mass
travel. Cmt is an energy-efficiency metric, but it is not a direct measurement
of battery energy: motor efficiency, regenerative power, electronics, and
negative mechanical work are not included.

## Continuous-controller implementation

The canonical scripts clip the continuous policy output to the physical torque
range and apply that same value in the dynamics and positive-work accumulator:

```python
action_value = np.clip(action_value, -torque, torque)
if action_value * (y[3] - y[1]) > 0:
    Energy_save_active_continuous[...] += (
        abs(action_value) * abs(y[3] - y[1]) * dt
    )
arr = np.array([[-action_value], [action_value]])
```

The completed 2026-07-16 rerun therefore requires no division by four or other
post-hoc scaling.

## Expert fusion

Only status `-2` denotes expert failure. The full evaluator branch is:

```python
if working_save_passive[i, j, k, ll] == 0:
    if working_save0_1[i, j, k, ll] != -2 and working_save01[i, j, k, ll] != -2:
        negative_cmt = Cmt_save0_1[i, j, k, ll]
        positive_cmt = Cmt_save01[i, j, k, ll]
        if np.isfinite(negative_cmt) and np.isfinite(positive_cmt):
            Cmt_save_passive[i, j, k, ll] = min(positive_cmt, negative_cmt)
        elif np.isfinite(negative_cmt):
            Cmt_save_passive[i, j, k, ll] = negative_cmt
        elif np.isfinite(positive_cmt):
            Cmt_save_passive[i, j, k, ll] = positive_cmt
        else:
            Cmt_save_passive[i, j, k, ll] = np.nan
    elif working_save0_1[i, j, k, ll] == -2:
        Cmt_save_passive[i, j, k, ll] = Cmt_save01[i, j, k, ll]
    else:
        Cmt_save_passive[i, j, k, ll] = Cmt_save0_1[i, j, k, ll]
```

The minimum is evaluated only when both experts are valid. If one expert
fails, the successful expert's Cmt is retained. Recomputing the complete branch
over all 12 result cells produced zero mismatches.

## Rerun provenance

The 12 canonical scripts use fixed evaluation seed `20260716`. The negative
expert's HDF5 output now receives `Cmt_save0_1`, not the positive expert array.
`analysis/recompute_action_weight_table_ii.py` reads the resulting HDF5 files,
validates shape, finiteness, expert-file separation, and the full fusion branch,
then computes the Appendix D audit on each condition's three-method common-feasible mask.
The result CSV and JSON manifest include sample counts, distribution summaries,
script hashes, and all HDF5 input hashes.
