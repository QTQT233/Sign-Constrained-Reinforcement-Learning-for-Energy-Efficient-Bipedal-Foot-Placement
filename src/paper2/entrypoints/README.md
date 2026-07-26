# Paper2 source-aligned entry points

This directory contains the current case entrypoints for the five non-MPC
method families across the 12 case directories. The three learned-controller
entrypoints contain the recovery settings selected by the complete 12-case
grid. The raised-1.145-r1 LIPM and TVLQR sources retain their accepted
source-aligned settings; current SHA-256 values and provenance are recorded in
the repository manifests and result documentation.

The TVLQR entry points are self-contained except for four policy checkpoints,
which are released under `models/paper2/tvlqr/`. Run them portably with
`tools/run_frozen_tvlqr.py`; it verifies hashes and changes paths only in a
temporary copy. The other four method families retain source-aligned
provenance entry points whose original case-tree dependencies are not
represented as portable executables. The accepted numerical records and their
portable validators are released under `results/paper2_current/`,
`results/paper2_push_off_grid_12case/`, and
`results/paper2_mpc_unified_12case/`. Supplementary Archive S1 is limited to
the complete MPC candidate searches, solver traces, validation, and independent
verification replays.

The three learned-controller entrypoints contain the selected post-impact
recovery-velocity decrements from the complete 12-case grid record in
`results/paper2_push_off_grid_12case/`. The fixed programs preserve their
original path literals and are not advertised as standalone portable entry
points. The grid-record validator checks both selected values in every entry
point without modifying the sources.

Frozen source snapshots under
`results/paper2_source_aligned_r1_rerun_20260721/` are historical provenance
only. They use earlier recovery settings and a deprecated reporting-only
`Foot_error` calculation; they are not equivalent to the current learned
entrypoints. The current horizontal foot-placement metric is defined and
validated in `supplementary/S4/landing_metric/`.
