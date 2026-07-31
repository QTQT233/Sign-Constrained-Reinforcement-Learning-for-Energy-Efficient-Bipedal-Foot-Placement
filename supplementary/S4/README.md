# Supplementary Archive S4

This directory contains the compact, repository-hosted form of Supplementary
Archive S4. It supports four analyses reported in the manuscript:

1. pooled per-transition horizontal foot-placement MAE for the 12 two-link
   cases;
2. the primary five-batch, 15-seed, 10,800-case phase-aware three-expert
   evaluation;
3. the matched 2,160-case source archive, its direct-argmax comparison, the
   phase-aware Figure 7 analysis, and the additional randomized four-link
   batches; and
4. the frozen-checkpoint transition-start-hold versus per-control-step-requery
   comparison on 2,160 matched cases.

The journal-supplement ZIP mirrors these materials and may contain additional execution
logs. The repository copy references canonical model and evaluator paths and includes
the compact primary evaluation and supporting records. It omits smoke
outputs, superseded diagnostics, manuscript figure-rendering code, rendered
manuscript assets, and the separate physical-touchdown diagnostic.

## Directory map

- `landing_metric/`: 144 terminal-state/target-angle records, portable geometry
  verification, case/controller summaries, and validation.
- `five_batch_phase_aware_confirmation/`: frozen primary protocol, portable
  route/hard-mask/analysis entry points, 10,800 matched records, aggregate
  metrics, provenance amendments, and checksums.
- `fourlink_additional_batches/`: additional randomized batches for secondary
  analyses.
- `../results/four_link_three_expert_primary_2160/`: matched source rollouts and
  direct-argmax learned-gate comparison.
- `../results/four_link_phase_aware_primary_2160/`: phase-aware three-expert
  matched analysis used for the manuscript summary figure.
- `hold_vs_requery/`: portable single-factor runner, 2,160 paired trial
  records, sign-switch sequences, paired inference, and QA records.

Run `python supplementary/S4/build_manifest.py` from the repository root to
regenerate the scoped `MANIFEST.csv` and `CHECKSUMS.sha256`.
