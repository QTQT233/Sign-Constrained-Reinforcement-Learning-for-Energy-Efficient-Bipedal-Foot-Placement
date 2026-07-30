# Final-manuscript table and figure provenance

| Item | Current numerical source | Reproduction boundary |
|---|---|---|
| Table I | three source-aligned \(30^4\) action maps; `analysis/reproduce_table1_active_fallback.py`; `results/table1/table_i.csv` | reports Active, one-sided-union, Active-default route, and exclusive-partition counts |
| Table II(a–c) | action-weight HDF5 package (Supplementary Data S2); `analysis/recompute_action_weight_table_ii.py`; current CSV/JSON summaries | fixed-checkpoint evaluation with seed 20260716 and the declared eligibility rule |
| Table III | hardware apparatus record and Zenodo hardware DOI | apparatus parameters only |
| Table IV | primary two-link entry points, current source-aligned records, and accepted MPC package | documents action generation, reset/recovery, metric, denominator, and source |
| Table V | `results/two_link_primary_12_cases.csv`; `results/paper2_current/`; `analysis/reproduce_paper2_tables.py` | 12 matched case/checkpoint outputs and four condition means |
| Table VI | `data/four_link/true_action_mask_scratch_c090_epoch1275/controller_summary.csv` | five principal fixed-checkpoint controller rows |
| Table VII | current paired CSVs and `analysis/four_link_true_action_mask_paired_inference.py` | conditional both-valid comparisons under the shared V22_3/V9 evaluator |
| Table VIII | `data/four_link/true_action_mask_scratch_c090_epoch1275/paired_trials.csv`; `analysis/four_link_paired_inference.py`; `results/four_link_statistics/active_vs_selector/displacement_threshold_sensitivity.csv` | active-versus-selector matched-Cmt sensitivity at common signed-displacement thresholds |
| Table IX | `results/four_link_active_default_confirmation/matched_trials.csv.gz`; `analysis/four_link_active_default_confidence_statistics.py` | held-out Active-reference comparison of the Active-default confidence selector |
| Appendix A | `results/paper2_current/paper2_combined_cases.csv`; accepted MPC records | case-level and descriptive two-link values |
| Appendix B | `data/four_link/true_action_mask_scratch_c090_epoch1275/`; current statistical outputs | controller, command-grid, terminal-reason, and training-log diagnostics |

## Figure boundary

The public repository does not distribute manuscript figure-rendering scripts
or image assets. Figure-level numerical values remain traceable as follows:

- Figure 4: the same action-map records as Table I;
- Figures 5–6: the two-link controller implementations and illustrated rollout
  definitions in the manuscript;
- Figures 7–8: the separate hardware evidence record;
- Figure 9: qualitative rollout visualization; its plotting source and selected
  case inputs are retained in the offline submission backup and are not
  represented as a public population-level result;
- Figure 10: the current controller summary and paired four-link analysis.

Drawing code and source images are maintained with the submitted manuscript
files and are not represented as part of the public numerical reproducibility
record.

## Controller implementation map

| Mechanism | Current source | Update rule |
|---|---|---|
| Two-link simulation router | `src/two_link/offline_routing/transition_locked_router.py` and the case-level `Qi_multi_passive_sim.py` entry points | select an expert at transition onset and hold it to termination |
| Two-link hardware lookup | `src/two_link/offline_lookup/atc50_event_lookup.py` and the frozen ATC-50 table in Supplementary Data S3 | query the 50^4 action table at approximately 1.8° directed stance-angle events |
| Four-link online selector | V22_3/V9 evaluator and selector checkpoint | select a one-sided expert at transition onset |
| Four-link Active-default confidence selector | `src/four_link/evaluation/run_active_default_confidence_confirmation.py`; frozen three-class gate | use Active by default; release a sign expert at its fixed confidence threshold and hold it |
| True hard mask | scratch true-mask training/evaluation sources | refresh the admissible hip-torque sign at each control step |

## Verified Table I partitions

| Set or exclusive partition | Count | Full-grid rate |
|---|---:|---:|
| unrestricted Active PPO | 647,160 | 79.896% |
| one-sided expert union | 562,003 | 69.383% |
| Active-default route | 672,382 | 83.010% |
| Active and one-sided feasible | 536,781 | 66.269% |
| Active only | 110,379 | 13.627% |
| one-sided union only | 25,222 | 3.114% |
| neither | 137,618 | 16.990% |
