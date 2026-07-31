# Supplementary Archive S4

This directory supports four manuscript analyses:

1. pooled per-transition horizontal foot-placement MAE for the 12 two-link
   cases;
2. the independent five-batch phase-aware three-expert confirmation comprising
   10,800 matched trials over 15 evaluation seed streams;
3. the historical five-batch two-expert evaluation used as a phase-aware
   threshold-development source; and
4. the frozen transition-start-hold versus per-control-step-requery comparison
   on 2,160 matched four-link cases.

## Directory map

- `landing_metric/`: 144 terminal-state/target-angle records, portable geometry
  verification, case/controller summaries, and validation.
- `fourlink_additional_batches/`: frozen seed configuration, portable batch
  runner, historical two-expert matched records, pooled and seed-stream
  statistics, and paired endpoint tests.
- `five_batch_phase_aware_confirmation/`: frozen protocol, parallel rerun
  programs, 10,800-case matched results, 20,000-replicate stratified
  inference, manuscript tables, and Figure 10 regeneration.
- `hold_vs_requery/`: portable single-factor runner, 2,160 paired trial
  records, sign-switch sequences, paired inference, and QA records.

The four directories are the machine-readable sources for the corresponding
main-text and Appendix B results. Run

```bash
python supplementary/S4/build_manifest.py --check
```

from the repository root to verify `MANIFEST.csv` and `CHECKSUMS.sha256`.
Author-generated material is licensed under Apache License 2.0.
