# Table II fixed-seed rerun and audit

## Protocol

The 12 canonical `Whole_energy_comparison_low_dim.py` evaluators cover three
action weights (`0.02`, `0.04`, and `0.06`) and four physical conditions. They
were rerun to completion with evaluation seed `20260716`. Each HDF5 array has
shape `10 x 30 x 10 x 30`.

For each condition, the reported comparison set is exactly

```text
(working_save_passive != -2)
& (working_save_active_discrete != -2)
& (working_save_active_continuous != -6)
```

No trimming, winsorization, or post-hoc rescaling is applied. Means in the
manuscript therefore include every finite Cmt value on the common mask. Because
the distributions are right-skewed, the audit CSV also retains the median,
sample SD, 99th percentile, and maximum.

## Validation results

- all 12 required evaluator files and all 120 HDF5 inputs were readable;
- every array had the expected shape and finite comparison values;
- the negative-expert Cmt arrays were distinct from their positive-expert
  counterparts;
- the complete expert-fusion branch matched every stored fused Cmt value
  (`fusion_bad_n = 0` in all 12 conditions); and
- all manuscript values were regenerated directly from the common masks.

The paper-facing values are in
`results/action_weight_table_ii_seed_20260716.csv`. The wide audit table is in
`results/action_weight_rerun_seed_20260716.csv`, and
`results/action_weight_rerun_seed_20260716_manifest.json` records the protocol,
script hashes, HDF5 hashes, file sizes, and dataset names.

## Interpretation boundary

Table II is a fixed-checkpoint action-weight performance ablation under a fixed
evaluation seed. It supports comparison of the retained controller outputs on
the declared common-feasible states. It is not an estimate of variation across
independently trained random seeds, so causal claims about training robustness
should not be inferred from this table alone.
