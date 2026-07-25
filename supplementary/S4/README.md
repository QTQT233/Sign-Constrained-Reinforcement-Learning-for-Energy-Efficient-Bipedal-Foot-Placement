# Supplementary Archive S4

This directory contains the compact, repository-hosted form of Supplementary
Archive S4. It supports three analyses reported in the manuscript:

1. pooled per-transition horizontal foot-placement MAE for the 12 two-link
   cases;
2. five additional randomized four-link evaluation batches, comprising 10,800
   matched active-PPO/selector trials; and
3. the frozen-checkpoint transition-start-hold versus per-control-step-requery
   comparison on 2,160 matched cases.

The submission ZIP mirrors these materials and may contain additional execution
logs. The repository copy omits byte-identical model, evaluator, and primary
2,160-case input duplicates and instead references their canonical paths in the
repository. It also omits smoke outputs, obsolete diagnostics, figure-rendering
code, and the separate physical-touchdown diagnostic.

## Directory map

- `landing_metric/`: 144 terminal-state/target-angle records, portable geometry
  verification, case/controller summaries, and validation.
- `fourlink_additional_batches/`: frozen seed configuration, portable batch
  runner, 10,800 matched records, pooled and seed-stream statistics, and paired
  endpoint tests.
- `hold_vs_requery/`: portable single-factor runner, 2,160 paired trial
  records, sign-switch sequences, paired inference, and QA records.

Run `python supplementary/S4/build_manifest.py` from the repository root to
regenerate the scoped `MANIFEST.csv` and `CHECKSUMS.sha256`.

The physical-touchdown diagnostic is not part of S4 and is not used in the
manuscript.
