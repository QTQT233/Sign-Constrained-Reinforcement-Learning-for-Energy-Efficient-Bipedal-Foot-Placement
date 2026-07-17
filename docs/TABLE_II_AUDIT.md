# Legacy action-weight fixed-seed rerun and audit (final Appendix D)

The filename is retained for compatibility with intermediate-draft links; the
final manuscript moves this material out of main Table II and into Appendix D.

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
& proposed_displacement_eligible
& active_displacement_eligible
& continuous_displacement_eligible
```

Displacement eligibility requires `D > 0.01 m`. For positive Cmt values in the
retained pre-threshold archive, the original displacement is exactly
recoverable from the retained energy and Cmt arrays as `D = E/(W*Cmt)`, where
`W = (1.4122 + 0.0839) * 9.81 N`. Zero-energy/zero-Cmt successes are retained
because they do not create a small-denominator tail. No winsorization or Cmt
rescaling is applied. The audit CSV retains the pre-filter count, exclusion
count, mean, median, sample SD, 99th percentile, and maximum.

## Validation results

- all 12 required evaluator files and all 120 HDF5 inputs were readable;
- every array had the expected shape and finite comparison values;
- the negative-expert Cmt arrays were distinct from their positive-expert
  counterparts;
- the complete expert-fusion branch matched every stored fused Cmt value
  (`fusion_bad_n = 0` in all 12 conditions); and
- all manuscript values were regenerated from the declared common masks after
  the `D > 0.01 m` eligibility filter.

The paper-facing values are in
`results/action_weight_table_ii_seed_20260716.csv`. The wide audit table is in
`results/action_weight_rerun_seed_20260716.csv`, and
`results/action_weight_rerun_seed_20260716_manifest.json` records the protocol,
script hashes, HDF5 hashes, file sizes, and dataset names.
The original unfiltered CSVs and manifest are retained under filenames ending
in `pre_0p01m_filter`.

## Interpretation boundary

Table II is a fixed-checkpoint action-weight performance ablation under a fixed
evaluation seed. It supports comparison of the retained controller outputs on
the declared common-feasible states. It is not an estimate of variation across
independently trained random seeds, so causal claims about training robustness
should not be inferred from this table alone.
