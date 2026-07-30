# Paper2 fixed-case entry points

This directory contains provenance-preserving source snapshots for the five
non-MPC method families across the 12 case directories. The three learned
controller entry points contain the fixed post-reset recovery pairs selected by
the complete \(120\times120\) grid searches in
`results/paper2_push_off_grid_12case/`. Their selected values are checked
directly by `src/paper2/push_off_grid/validate_release.py`. LIPM and TVLQR
remain separate reference-based diagnostics.

The entry points resolve their external checkpoints and action maps through
`src/paper2/artifact_paths.py`; no workstation-specific path is required.
Extract Supplementary Archive S1 and set `PAPER2_ARTIFACT_ROOT` to its artifact
directory. This directory must contain separate `flat/` and `raised/`
subdirectories; do not combine them because several repeated filenames have
terrain-specific contents. The four frozen TVLQR checkpoints stored under
`models/paper2/tvlqr/` are discovered automatically.

The released candidate tables, selected minima, fixed replay values, and
validation program are fully portable. A source entry point can be run from
any working directory, for example:

```powershell
$env:PAPER2_ARTIFACT_ROOT = "X:\extracted_S1\paper2_artifacts"
$env:CMT_REPLAY_JSON = "replay.json"
python src/paper2/entrypoints/flat_1.145_r1/Qi_multi_passive_sim.py
```
