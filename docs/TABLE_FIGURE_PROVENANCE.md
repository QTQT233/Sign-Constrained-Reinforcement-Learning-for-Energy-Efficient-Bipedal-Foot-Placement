# Final-manuscript table and figure provenance map

This map records what is reproducible, what is replayable from archived output,
and what must remain an explicitly bounded secondary result.

| Final item | Canonical source/artifact | Evidence status | Manuscript/release boundary |
|---|---|---|---|
| Table I | three archived 30^4 HDF5 maps; `analysis/reproduce_table1_figure4.py` | exact counts and four mutually exclusive partitions reproduced | report common-set full-grid coverage as 65.9%, not 100% |
| Table II(a-c) | fixed-seed action-weight sources, HDF5 arrays, and `docs/TABLE_II_AUDIT.md` | corrected rerun and common-eligibility counts retained | deterministic archive audit, not independent-seed sensitivity |
| Table III | retained hardware parameter record and Appendix C boundary | nominal fixture properties only | do not infer reliability, success probability, or measured energy |
| Table IV | primary multi-step entrypoints under `src/two_link/`, accepted MPC source, and rerun records | action generation, recovery/reset, metric, and provenance boundaries documented | restrict inference to controller-specific recorded implementations |
| Table V | `results/two_link_primary_12_cases.csv`; source-aligned reruns; accepted unified MPC table; `analysis/reproduce_paper2_tables.py` | all 12 case values and four condition means reproduced | condition means summarize prespecified case/checkpoint outputs, not training repetitions |
| Table VI | `data/four_link/true_action_mask_scratch_c090_epoch1275/controller_summary.csv` | five main fixed-checkpoint controller rows reproduced | historical unrestricted per-step encoding remains in Appendix B and is not relabelled as a hard mask |
| Table VII | direct paired CSVs and `analysis/four_link_true_action_mask_paired_inference.py` | active-selector, active-oracle, and true-mask-selector both-valid statistics reproduced | conditional paired Cmt evidence; no full-domain energy or confirmatory equivalence claim |
| Figure 4 | same three 30^4 action maps | data and deterministic SVG reproduced | use generated `figure4_data.csv` and SVG |
| Figure 5 | several hard-coded candidate plotting scripts | initial-state/source identity remains ambiguous | retain only with a frozen script/config and source-data hash |
| Figure 8 | `analysis/plot_figure08_four_link_diagnostics.py`; final four-link summary CSVs | deterministic SVG/PDF/PNG output | regenerate from the released constants before submission |
| Figure 9 | `analysis/plot_figure9_true_action_mask_case.py`; source-data, case-selection, and QA files under `results/figures/true_action_mask/` | exact selected key and rollout checks retained | post hoc illustrative case only; selection rule must remain in the caption |
| Figure 10 | `analysis/plot_figure10_true_action_mask.py`; formal V22_3/V9 controller and paired CSVs | 600-dpi PNG plus editable-text PDF/SVG and source CSV | fixed-checkpoint method-level comparison, not a one-factor causal ablation |
| Two-link simulation router | `working_save_passive-10-30`; `src/two_link/offline_routing/` | coarse transition-onset expert selection | selected expert is retained to termination |
| Two-link hardware lookup | `working_save-ATC-50`; 50-bin generator family and `src/two_link/offline_lookup/` | same two-expert fusion family, 50 bins on all four axes; event-updated direct action | describe the approximately 1.8-degree update interval; do not call it transition-locked |
| Appendix A references | frozen TVLQR sources and 12 case-tuned MPC records | reference values reproduced but definitions differ | keep out of the primary comparison tables |
| Appendix B diagnostics | legacy V22_3/V9 CSVs plus `data/four_link/true_action_mask_scratch_c090_epoch1275/` | all 10,800 legacy rows reproduced exactly before adding the 2,160 true-mask rows | one scratch run; historical encoding, active-full, and oracle rows are diagnostics, not causal ablations |
| Appendix D action-weight audit | 12 evaluators; `analysis/recompute_action_weight_table_ii.py`; result files prefixed `action_weight_*_seed_20260716` | fusion validation 0 mismatches; `D > 0.01 m` common eligibility recorded | deterministic scale/provenance audit, not a controller sensitivity claim |

## Verified action-map partitions

| Set or exclusive partition | Count | Full-grid rate |
|---|---:|---:|
| active PPO | 630,328 | 77.818% |
| one-sided expert union | 582,861 | 71.958% |
| intersection | 534,042 | 65.931% |
| active only | 96,286 | 11.887% |
| one-sided union only | 48,819 | 6.027% |
| neither | 130,853 | 16.155% |

The old table displayed the common feasible set as 100% by dividing the set by
itself. That conditional normalization is not full-grid coverage.

## Twelve primary two-link case identifiers

- flat, nominal 1.145 m: `1.145`, `1.145-2`, `1.145-3`;
- raised 0.01 m, nominal 1.145 m: `0.01m,1.145`, `...-2`, `...-3`;
- flat, nominal 1.28 m: `1.28`, `1.28-2`, `1.28-3`;
- raised 0.01 m, nominal 1.28 m: `0.01m,1.28`, `...-2`, `...-3`.

Folder suffixes are execution/case identifiers, not independent training seeds
unless a separate manifest proves distinct training runs.
