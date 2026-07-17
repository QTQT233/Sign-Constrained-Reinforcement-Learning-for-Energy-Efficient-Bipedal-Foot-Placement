# Research-data artifact bundle

Canonical archive: **DOI TO BE MINTED**

The DOI-backed bundle should contain the files listed in `MANIFEST.csv` and
`CHECKSUMS.sha256`, including raw action maps, raw Cmt arrays, the 12 multi-step
case artifacts, four-link per-trial CSVs, selector data, training logs, and the
exact checkpoints loaded by the frozen evaluator.

The prepared submission archive additionally contains
`action_weight_12_cell_arrays.h5` (58,282,799 bytes; SHA-256
`91576b1a6f7f432d9b4c754fae68c637a09bd0e05e7b766aaa27d27f554a545d`),
its Excel/CSV/JSON manifests, and the hardware videos
`Flat_walking.mp4` and `Ueven_foot_placement.mp4`. The HDF5 artifact contains
all 12 action-weight cells with status, positive-work energy, Cmt, recovered
COM displacement, and per-entry displacement-provenance flags.

Small Paper2 audit outputs, all captured stdout/stderr files, and frozen
entry-point copies are included directly in this repository. The DOI bundle is
still required for the entry points' relative model, array, and helper-module
dependencies.

Until the DOI is public, the repository must describe the data as "available
from the authors for reproducibility checking" rather than claiming that the
data are openly available.

Author-owned processed data in this repository and the author-owned hardware
videos when distributed in the submission/archive are licensed under
Apache-2.0. See `../DATA_LICENSE.md` and `../LICENSE`. The future DOI record
must repeat the same license and scope unless the authors publish a documented
superseding version.

The GitHub repository is the canonical code record. The DOI archive is the
canonical immutable data/model record. Each release should cross-reference the
other by Git commit and archive DOI/version.
