# Paper2 source-aligned entry points

This directory contains provenance-preserving source snapshots for the five
non-MPC method families across the 12 case directories. The current raised-1.145-r1
Continuous-torque PPO, LIPM, and TVLQR sources are synchronized to the accepted
source-aligned runs; their current SHA-256 values and provenance are recorded
in the repository manifests and the corresponding result documentation.

The TVLQR entry points are self-contained except for four policy checkpoints,
which are released under `models/paper2/tvlqr/`. Run them portably with
`tools/run_frozen_tvlqr.py`; it verifies hashes and changes paths only in a
temporary copy. The other four method families load additional relative
artifacts from their original case directories. Their complete 12-case trees
must be supplied from the original case bundle under the names in
`configs/paper2_cases.csv` and audited with `tools/audit_paper2_readonly.py`.
They are distinct from the hardware-only Zenodo record; when required for
submission, distribute them through Supplementary Archive S1.

The remaining snapshots preserve original path literals and are not advertised
as standalone portable entry points. The audit launcher is read-only with
respect to these repository sources and does not modify them at runtime.
