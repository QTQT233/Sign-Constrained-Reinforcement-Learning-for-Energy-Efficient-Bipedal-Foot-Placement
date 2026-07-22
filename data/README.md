# Repository data and external hardware archive

The GitHub repository is the canonical executable record for the current
source code, repository-hosted models, simulation inputs and summaries,
accepted MPC results, manifests, and validation scripts. The exact immutable
Git commit used by the manuscript should be reported in the Code Availability
statement.

The unchanged hardware evidence is a separate data record:

- Zenodo DOI: <https://doi.org/10.5281/zenodo.21407986>
- Scope: author-owned hardware evidence and its archive metadata
- License: Apache-2.0 for author-owned material

This DOI must not be described as containing the current GitHub software,
the unified 12-case MPC rerun, simulation checkpoints, or exhaustive MPC
candidate searches unless those files are actually added in a future Zenodo
version. A code-only correction merged into GitHub does not require a new
Zenodo version or a new GitHub Release when the manuscript pins an immutable
Git commit.

The accepted MPC case table and path-free validation digest are under
`../results/paper2_mpc_unified_12case/`. Complete candidate tables, traces,
solve logs, and fresh-process replays are included in Supplementary Archive S1
for the journal submission. Some frozen source-aligned Paper2 entry points require their
original case-directory dependencies; these are distinct from the hardware
DOI and should be supplied as supplementary material or from the authors when
needed.

The 12-cell action-weight HDF5 package is Supplementary Data S2. The frozen
ATC-50 table, its schema, portable event-query implementation, and checksums are
Supplementary Data S3. S3 makes no byte-for-byte table-regeneration claim.
Neither package is part of the hardware-only Zenodo record.

Repository-wide file sizes and SHA-256 values are recorded in
`../MANIFEST.csv` and `../CHECKSUMS.sha256`. The data and video license boundary
is defined in `../DATA_LICENSE.md` and `../LICENSE`.
