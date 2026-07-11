# Table II reproducibility audit

The three `action_weight` folders contain four physical conditions, policies,
and 10x30x10x30 HDF5 result arrays. Replaying the final common-completed mask
recovers most proposed/active values, but the published table is not a direct
output of the retained analysis code.

Key conflicts:

- the published continuous-PPO values are approximately the archived raw Cmt
  means divided by four, but no retained script performs or explains that
  transformation;
- the continuous arrays contain extreme outliers, including values above ten
  million in one cell, and no exclusion manifest is retained;
- the 0.06 active-PPO middle conditions align only if two manuscript columns
  are swapped;
- the 0.02 first proposed value is 0.6889 in the archived common set, not 0.48;
- two 0.02 active training scripts warm-start from 0.04 checkpoints.

Therefore Table II should not be described as a controlled independent
training ablation in its current form. Before submission, run one frozen
external evaluator over all archived arrays/checkpoints, define Cmt once in SI
units, state the common-feasible mask and outlier policy in advance, and write a
tidy CSV containing condition, controller, weight, checkpoint hash, n, mean,
median, and uncertainty. If that rerun cannot be completed, remove Table II's
continuous row and causal reward-mechanism language rather than applying an
undocumented scale factor.
