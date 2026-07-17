# Paper2 frozen entry points

This directory contains byte-for-byte copies of the five discovered entry
scripts for each of the 12 case directories. Their SHA-256 values are recorded
in `data/paper2/readonly_rerun/paper2_rerun_results.csv`.

The TVLQR entry points are self-contained except for four policy checkpoints,
which are released under `models/paper2/tvlqr/`. Run them portably with
`tools/run_frozen_tvlqr.py`; it verifies hashes and changes paths only in a
temporary copy. The other four method families load additional relative
artifacts from their original case directories. Their complete 12-case trees
must be placed in the DOI bundle under the names in `configs/paper2_cases.csv`
and audited with `tools/audit_paper2_readonly.py`.

No supplied experiment source was edited during this audit.
