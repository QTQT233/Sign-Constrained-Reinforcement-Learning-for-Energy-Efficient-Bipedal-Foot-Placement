# Final-manuscript table and figure provenance

| Item | Current numerical source | Reproduction boundary |
|---|---|---|
| Table I | three 30^4 action maps; `analysis/reproduce_table1.py`; `results/table1/table_i.csv` | reports full-grid counts and the 65.9% common feasible fraction |
| Table II(a–c) | action-weight HDF5 package (Supplementary Data S2); `analysis/recompute_action_weight_table_ii.py`; current CSV/JSON summaries | fixed-checkpoint evaluation with seed 20260716 and the declared eligibility rule |
| Table III | hardware apparatus record and Zenodo hardware DOI | apparatus parameters only |
| Table IV | primary two-link entry points, complete learned-controller recovery-grid records, and accepted MPC package | documents action generation, reset/recovery, metric, denominator, and source |
| Table V | `results/paper2_push_off_grid_12case/`; `results/two_link_primary_12_cases.csv`; `results/paper2_current/`; `analysis/reproduce_paper2_tables.py` | 12 matched case/checkpoint outputs and four condition means |
| Table VI | `data/four_link/true_action_mask_scratch_c090_epoch1275/controller_summary.csv` | five principal fixed-checkpoint controller rows |
| Table VII | current paired CSVs and `analysis/four_link_true_action_mask_paired_inference.py` | conditional both-valid comparisons under the shared V22_3/V9 evaluator |
| Appendix A | `configs/paper2_push_off_grid_12_cases.csv`; `results/paper2_push_off_grid_12case/`; `results/paper2_current/paper2_combined_cases.csv`; accepted MPC records | selected recovery pairs, case-level values, and descriptive two-link summaries |
| Appendix B | `data/four_link/true_action_mask_scratch_c090_epoch1275/`; current statistical outputs | controller, command-grid, terminal-reason, and training-log diagnostics |

## Figure boundary

The public repository does not distribute manuscript figure-rendering scripts
or image assets. Figure-level numerical values remain traceable as follows:

- Figure 4: the same action-map records as Table I;
- Figures 5–6: the two-link controller implementations and illustrated rollout
  definitions in the manuscript;
- Figures 7–8: the separate hardware evidence record;
- Figure 9: the formal shared-evaluator record with seed `20260719`, episode
  `1155`, direction `-1`, touchdown reset, 0.30 m command length, and 0.01 m
  command height, selected from the archived 2,160-case comparison for
  qualitative visualization;
- Figure 10: the current controller summary and paired four-link analysis.

The submitted drawing code and source images are stored in the authors'
offline `Paper_store/New` submission backup. Earlier figure candidates are
stored under `Paper_store/Past`. Neither set is represented as part of the
public numerical reproducibility record.

## Controller implementation map

| Mechanism | Current source | Update rule |
|---|---|---|
| Two-link simulation router | `src/two_link/offline_routing/transition_locked_router.py` and the case-level `Qi_multi_passive_sim.py` entry points | select an expert at transition onset and hold it to termination |
| Two-link hardware lookup | `src/two_link/offline_lookup/atc50_event_lookup.py` and the frozen ATC-50 table in Supplementary Data S3 | query the 50^4 action table at approximately 1.8° directed stance-angle events |
| Four-link online selector | V22_3/V9 evaluator and selector checkpoint | select a one-sided expert at transition onset |
| True hard mask | scratch true-mask training/evaluation sources | refresh the admissible hip-torque sign at each control step |

## Verified Table I partitions

| Set or exclusive partition | Count | Full-grid rate |
|---|---:|---:|
| active PPO | 630,328 | 77.818% |
| one-sided expert union | 582,861 | 71.958% |
| intersection | 534,042 | 65.931% |
| active only | 96,286 | 11.887% |
| one-sided union only | 48,819 | 6.027% |
| neither | 130,853 | 16.155% |
