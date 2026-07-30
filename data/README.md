# Repository data and external archive

The GitHub repository is the canonical executable record for the current
source code, repository-hosted models, simulation inputs and summaries,
accepted MPC results, manifests, and validation scripts. The exact immutable
Git commit used by the manuscript should be reported in the Code Availability
statement.

Zenodo version 1.0.0 is the released data/model record:

- Zenodo DOI: <https://doi.org/10.5281/zenodo.21407986>
- Scope: the 12-cell action-weight HDF5 archive, ATC-50 and coarse routing
  lookups, selected two-link and legacy four-link records, released model
  checkpoints, manifests, checksums, and two hardware-feasibility videos
- License: Apache-2.0 for author-owned material

The exact manuscript-facing executable code and results added after Zenodo
version 1.0.0 are pinned by the immutable GitHub commit cited in the manuscript.
The exhaustive MPC candidate searches are supplied separately as Supplementary
Archive S1.

The accepted MPC case table and path-free validation digest are under
`../results/paper2_mpc_unified_12case/`. Complete candidate tables, traces,
solve logs, and fresh-process replays are included in Supplementary Archive S1
for the journal submission. The portable Paper2 entry points additionally
resolve their case-specific learned-controller checkpoints and action maps from
the artifact root documented in `../src/paper2/entrypoints/README.md`.

The 12-cell action-weight HDF5 package is designated Supplementary Data S2.
The frozen ATC-50 table, its schema, portable event-query implementation, and
checksums are designated Supplementary Data S3. Both are also present in the
Zenodo version 1.0.0 record. Supplementary Archive S4 contains the released
landing-metric records, additional four-link batches, and hold-versus-requery
analysis.

Repository-wide file sizes and SHA-256 values are recorded in
`../MANIFEST.csv` and `../CHECKSUMS.sha256`. The data and video license boundary
is defined in `../DATA_LICENSE.md` and `../LICENSE`.
