# Positive commanded-work metric used by the two-link action-weight evaluators

## Definition

For an internal hip actuator with applied torque `tau_h` in N m, relative hip
angular speed `omega_h = theta_dot_2 - theta_dot_1` in rad/s, and evaluator
sample period `dt` in s, the accumulated positive commanded work is

```text
W_h^+ = sum_t max(tau_h,t * omega_h,t, 0) * dt .
```

`positive_actuator_work_increment()` implements the per-sample term.  Its first
argument is the *applied physical torque* in N m.  A caller must not multiply
that value by the actuator scale again.

The action-weight archive reports the dimensionless ratio

```text
Cmt = W_h^+ / (m_total * g * abs(x_com,end - x_com,start)).
```

when the denominator is positive.  This ratio is a commanded mechanical-work
proxy; it is not electrical energy.

## Deterministic legacy factor-of-four correction

The archived continuous evaluator clipped the sampled policy output to
`[-4, 4]` N m and applied that value directly as the physical torque in the
equations of motion. It then multiplied the same torque by the fixed actuator
bound `T = 4` a second time inside the work accumulator. Under the following
conditions, dividing that archived continuous numerator (and therefore Cmt) by
four is algebraically exact:

1. `T` is the same positive constant, 4, at every sample;
2. the trajectory, positive-power gate, time step, endpoint displacement, and
   aggregation mask are unchanged; and
3. the numerator contains no unscaled additive term.

For every sample satisfying those conditions,

```text
max((tau_h * 4) * omega_h, 0) * dt / 4
    = max(tau_h * omega_h, 0) * dt .
```

The corrected source now accumulates the physical torque once.  Outputs from a
future corrected rerun must **not** be divided by four again.

## Legacy-archive boundary

The code correction does not rewrite any retained HDF5 file.  The 12 archived
action-weight cells remain exploratory records and are unsuitable as a
controlled action-weight ablation because they lack explicit training seeds and
because the audited evaluator snapshot had additional provenance defects:

- 11 of 12 negative-expert Cmt files received the positive-expert Cmt array;
- 3 of 12 evaluator snapshots wrote at least one output outside the nominal
  action-weight directory; and
- the row called `Proposed` is an outcome-aware hindsight envelope over two
  expert rollouts, not the deployed transition-start route.

The evaluator constructs `Cmt_save_passive` in memory from `Cmt_save01` and
`Cmt_save0_1` before writing the envelope to its own file.  The historical
negative-file target error therefore did not alter that in-memory envelope,
but it does prevent an independent reconstruction of the envelope from the two
retained expert Cmt files.

The patched evaluators correct the negative-expert array and nominal output
directory for future reruns.  Existing HDF5 values are intentionally retained
unchanged so that source corrections are not confused with regenerated data.

## Patch contents and verification

The patch includes the shared metric module, 12 canonical action-weight
evaluators, one optimized evaluator, and regression tests.  Every evaluator
accepts the data/model/output root through the `ENERGY_COMPARISON_DATA_ROOT`
environment variable.  The historical Windows path remains only as the
documented default so that the source snapshot retains its provenance.

From `src/two_link/action_weight`, run:

```text
python -m unittest discover -s tests -v
```

The tests verify units and sign gating, the deterministic factor-of-four
identity, use of the shared accumulator in all 13 included evaluators, correct
negative-expert Cmt writes in all 12 canonical evaluators, and nominal output
directory placement.  They also read the published 12-cell CSV and verify all
12 `legacy_raw / 4 = corrected` rows numerically.

For a portable run of the optimized evaluator in a checkout containing the
required checkpoints, set the data root explicitly:

```bash
ENERGY_COMPARISON_DATA_ROOT=/path/to/Energy_Comparison \
python "action_weight=0.02/60,30(0.33m)/Whole_energy_comparison_low_dim_optimized.py"
```
