# Final-manuscript table and figure provenance map

This map records what is reproducible, what is replayable from archived output,
and what must remain an explicitly bounded secondary result.

| Final item | Canonical source/artifact | Evidence status | Manuscript/release boundary |
|---|---|---|---|
| Table I | `docs/TWO_LINK_SELECTOR.md`; simulation/hardware consumers; four-link summaries | three distinct mechanisms documented | keep simulation transition lock, hardware ATC-50 event updates, and the four-link transition-onset selector in separate rows |
| Table II(a/b) | pinned two-link sources, checkpoint filenames, and `docs/TABLE_II_AUDIT.md` | route parameters and legacy action-weight boundaries recovered | do not present the legacy archive as a matched retraining sweep |
| Table III | three archived 30^4 HDF5 maps; `analysis/reproduce_table1_figure4.py` | exact counts and four exclusive partitions reproduced | report common-set full-grid coverage as 65.9%, not 100% |
| Table IV | primary multi-step entrypoints under `src/two_link/` and rerun records | numerator/reset/denominator differences documented | restrict inference to controller-specific recorded aggregates |
| Table V | `results/two_link_primary_12_cases.csv`; `results/paper2_source_aligned_r1_rerun_20260721/rerun_records.csv`; `results/paper2_mpc_unified_12case/accepted_cases.csv`; `analysis/reproduce_paper2_tables.py` | all 12 case values reproduced; two explicitly keyed PPO rows use exact source-aligned records; MPC uses the accepted source-aligned unified rerun | retain case-level rows rather than four condition means only; preserve superseded current-table and MPC CSVs under `results/legacy/` |
| Table VI | `data/four_link/legacy_manuscript/`; `analysis/recompute_four_link_tables.py` | completion, joint eligibility, and reset phases reproduced | keep controller-level diagnostics in Appendix B |
| Table VII | retained hardware record and Appendix C evidence table | nominal fixture properties only | do not infer reliability or measured energy |
| Figure 4 | same three 30^4 action maps | data and deterministic SVG reproduced | use generated `figure4_data.csv` and SVG |
| Figure 5 | several hard-coded candidate plotting scripts | initial-state/source identity remains ambiguous | retain only with a frozen script/config and source-data hash |
| Figure 8 | `analysis/plot_figure08_four_link_diagnostics.py`; final four-link summary CSVs | deterministic SVG/PDF/PNG output | regenerate from the released constants before submission |
| Two-link simulation router | `working_save_passive-10-30`; `src/two_link/offline_routing/` | coarse transition-onset expert selection | selected expert is retained to termination |
| Two-link hardware lookup | `working_save-ATC-50`; 50-bin generator family and `src/two_link/offline_lookup/` | same two-expert fusion family, 50 bins on all four axes; event-updated direct action | describe the approximately 1.8-degree update interval; do not call it transition-locked |
| Appendix A references | frozen TVLQR sources and 12 case-tuned MPC records | reference values reproduced but definitions differ | keep out of the primary comparison tables |
| Appendix B diagnostics | 5 July V22_3/V9 CSVs; `configs/four_link_legacy_manuscript.json` | all 19 outputs retained; reconstruction reproduced all 19 byte-for-byte | continuations and oracle rows are not causal ablations |
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
