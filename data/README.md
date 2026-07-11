# Research-data artifact bundle

Canonical archive: **DOI TO BE MINTED**

The DOI-backed bundle should contain the files listed in `MANIFEST.csv` and
`CHECKSUMS.sha256`, including raw action maps, raw Cmt arrays, the 12 multi-step
case artifacts, four-link per-trial CSVs, selector data, training logs, and the
exact checkpoints loaded by the frozen evaluator.

Small Paper2 audit outputs, all captured stdout/stderr files, and frozen
entry-point copies are included directly in this repository. The DOI bundle is
still required for the entry points' relative model, array, and helper-module
dependencies.

Until the DOI is public, the repository must describe the data as "available
from the authors for reproducibility checking" rather than claiming that the
data are openly available.

The GitHub repository is the canonical code record. The DOI archive is the
canonical immutable data/model record. Each release should cross-reference the
other by Git commit and archive DOI/version.
