# Final-manuscript table and figure provenance

| Item | Current numerical source | Reproduction boundary |
|---|---|---|
| Table I | three 30^4 action maps; `analysis/reproduce_table1.py`; `results/table1/table_i.csv` | reports full-grid counts and the 65.9% common feasible fraction |
| Table II(a–c) | action-weight HDF5 package (Supplementary Data S2); `analysis/recompute_action_weight_table_ii.py`; current CSV/JSON summaries | frozen-policy grid evaluation with seed 20260716 and the declared eligibility rule |
| Table III | hardware apparatus record and Zenodo hardware DOI | apparatus parameters only |
| Table IV | primary two-link entry points, complete learned-controller recovery-grid records, and selected MPC package | documents action generation, reset/recovery, metric, denominator, and source |
| Table V | `results/paper2_push_off_grid_12case/`; `results/two_link_primary_12_cases.csv`; `results/paper2_current/`; `supplementary/S4/landing_metric/`; `analysis/synchronize_landing_metric_outputs.py`; `analysis/reproduce_paper2_tables.py` | 12 matched case/checkpoint outputs, four condition means, and pooled per-transition horizontal foot-placement MAE |
| Table VI | `supplementary/S4/five_batch_phase_aware_confirmation/manuscript_metrics/table_controller_summary.csv`; `supplementary/S4/five_batch_phase_aware_confirmation/results/controller_metrics_overall.csv` | primary 15-seed/10,800-case controller summary |
| Table VII | `supplementary/S4/five_batch_phase_aware_confirmation/results/paired_metrics_overall.csv`; `supplementary/S4/five_batch_phase_aware_confirmation/results/final_confirmation_report.json` | primary matched success and common-valid `Cmt` comparisons |
| Table VIII | `supplementary/S4/five_batch_phase_aware_confirmation/manuscript_metrics/five_batch_manuscript_metrics.json` | displacement-threshold sensitivity regenerated from the frozen 10,800-case matched record |
| Appendix A | `configs/paper2_push_off_grid_12_cases.csv`; `results/paper2_push_off_grid_12case/`; `results/paper2_current/paper2_combined_cases.csv`; `supplementary/S4/landing_metric/`; selected MPC records | selected recovery pairs, case-level values, pooled per-transition horizontal foot-placement MAE, and descriptive two-link summaries |
| Appendix B | `supplementary/S4/five_batch_phase_aware_confirmation/`; `results/four_link_three_expert_primary_2160/`; `results/four_link_phase_aware_primary_2160/`; `data/four_link/true_action_mask_scratch_c090_epoch1275/`; `supplementary/S4/fourlink_additional_batches/`; `supplementary/S4/hold_vs_requery/` | primary evaluation, matched 2,160-case comparison, additional controller records, and query-schedule control |
| Supplementary Archive S4 | `supplementary/S4/`; `results/four_link_three_expert_primary_2160/`; `results/four_link_phase_aware_primary_2160/` | compact repository mirror of the horizontal foot-target, primary 10,800-case, matched 2,160-case, additional randomized-batch, and hold-versus-requery evidence |

## Figure boundary

The public repository does not distribute manuscript figure-rendering scripts
or image assets. Figure-level numerical values remain traceable as follows:

- Earlier conceptual, two-link, and hardware figures: manuscript captions and
  the corresponding two-link implementations/Zenodo hardware record;
- Figure 6: a qualitative manuscript illustration of a matched four-link
  controller comparison; it is not used to compute a reported statistic;
- Figure 7: `results/four_link_three_expert_primary_2160/` contains the matched
  source-controller records and the other controller summaries;
  `results/four_link_phase_aware_primary_2160/` contains the phase-aware
  three-expert route assignments and summary values. The primary phase-aware
  five-batch statistics are reported separately in Tables VI--VIII.

Figure-rendering code and source images are outside the public numerical
reproducibility record.

## Controller implementation map

| Mechanism | Current source | Update rule |
|---|---|---|
| Two-link simulation router | `src/two_link/offline_routing/transition_locked_router.py` and the case-level `Qi_multi_passive_sim.py` entry points | select an expert at transition onset and hold it to termination |
| Two-link hardware lookup | `src/two_link/offline_lookup/atc50_event_lookup.py` and the frozen ATC-50 table in Supplementary Data S3 | query the 50^4 action table at approximately 1.8° directed stance-angle events |
| Four-link phase-aware three-expert router (primary) | `src/four_link/evaluation/paper_four_link_three_expert_selector_evaluation.py`; frozen 12--128--64--3 gate; `supplementary/S4/five_batch_phase_aware_confirmation/` | query the learned gate once, apply reset-phase confidence thresholds, use Active PPO when neither one-sided branch passes, and hold the selected expert |
| Direct-argmax three-expert gate (secondary comparison) | same evaluator/gate and `results/four_link_three_expert_primary_2160/` | choose the largest of the positive/negative/active gate outputs once at transition onset |
| Two-expert selector baseline | frozen two-class selector checkpoint and `supplementary/S4/hold_vs_requery/` | choose a one-sided expert and hold or requery according to the comparison schedule |
| True hard mask | scratch true-mask training/evaluation sources | refresh the admissible hip-torque sign at each control step |

## Verified Table I partitions

| Set or exclusive partition | Count | Full-grid rate |
|---|---:|---:|
| active PPO | 630,328 | 77.818% |
| one-sided expert union | 582,861 | 71.958% |
| proposed three-expert route | 679,147 | 83.845% |
| intersection | 534,042 | 65.931% |
| active only | 96,286 | 11.887% |
| one-sided union only | 48,819 | 6.027% |
| neither | 130,853 | 16.155% |
