# Supplementary Archive S4

This directory supports three manuscript analyses:

1. pooled per-transition horizontal foot-placement MAE for the 12 two-link
   cases;
2. five additional randomized four-link evaluation batches comprising 10,800
   matched active-PPO/selector trials; and
3. the frozen transition-start-hold versus per-control-step-requery comparison
   on 2,160 matched four-link cases.

## Directory map

- `landing_metric/`: 144 terminal-state/target-angle records, portable geometry
  verification, case/controller summaries, and validation.
- `fourlink_additional_batches/`: frozen seed configuration, portable batch
  runner, 10,800 matched records, pooled and seed-stream statistics, and paired
  endpoint tests.
- `hold_vs_requery/`: portable single-factor runner, 2,160 paired trial
  records, sign-switch sequences, paired inference, and QA records.

The three directories are the machine-readable sources for the corresponding
main-text and Appendix B results. Run

```bash
python supplementary/S4/build_manifest.py --check
```

from the repository root to verify `MANIFEST.csv` and `CHECKSUMS.sha256`.
Author-generated material is licensed under Apache License 2.0.
