# Table II fixed-checkpoint action-weight protocol

## Evaluation design

The 12 condition-specific `Whole_energy_comparison_low_dim.py` evaluators
cover three action weights (`0.02`, `0.04`, and `0.06`) and four physical
conditions. Each completed run used RNG seed `20260716`; every HDF5 array has
shape `10 × 30 × 10 × 30`. Checkpoint and output directories are configured
with `ACTION_WEIGHT_MODEL_DIR` and `ACTION_WEIGHT_OUTPUT_DIR`. Portable
full reruns use one case-specific checkpoint directory at a time, bound by
`configs/action_weight_checkpoint_manifest.json`. Numerical verification of
the manuscript values uses the S2 HDF5 archive and
`analysis/recompute_action_weight_table_ii.py`.

The manuscript comparison set is

```text
(working_save_passive != -2)
& (working_save_active_discrete != -2)
& (working_save_active_continuous != -6)
& proposed_displacement_eligible
& active_displacement_eligible
& continuous_displacement_eligible
```

Displacement eligibility requires each controller's archived center-of-mass
displacement to be finite and satisfy `D > 0.01 m`. The Table II analysis does
not infer missing displacement from energy and `Cmt`, and zero-work entries
receive no exception. Supplementary Data S2 records how its displacement arrays
were obtained. Its strict primary mask must equal an independent recomputation;
the earlier zero-work-inclusive mask is not distributed in the current release.
No winsorization or `Cmt` rescaling is applied.

## Expert fusion and code corrections

- the negative expert writes to its own `Cmt` array;
- when both experts succeed, the lower finite `Cmt` is used;
- when only one expert succeeds, that expert's finite value is used;
- only status `−2` denotes one-sided-expert failure;
- continuous-controller work is accumulated once from the applied physical
  torque; and
- the current evaluator stores `D_save` arrays and uses a fixed RNG seed.

All 12 current conditions pass the stored-value fusion check with zero
mismatches.

## Released outputs

- `results/action_weight_table_ii_seed_20260716.csv`: manuscript values and
  common-mask counts after the `D > 0.01 m` eligibility rule;
- `results/action_weight_rerun_seed_20260716.csv`: means, medians, sample SDs,
  upper-tail diagnostics, and exclusion counts;
- `results/action_weight_rerun_seed_20260716_manifest.json`: source-archive
  hash, script hashes, and displacement-filter protocol metadata; and
- files containing `pre_0p01m_filter`: the corresponding unfiltered sensitivity
  records.

The complete 12-cell HDF5 package is distributed as Supplementary Data S2.
Table II is a fixed-checkpoint performance ablation, not a set of independent
training repetitions.
