# Data archive checklist for the RAS submission

The public records have separate roles:

- GitHub: corrected executable software, repository-hosted models and
  simulation data, accepted results, provenance, and validators;
- Zenodo DOI <https://doi.org/10.5281/zenodo.21407986>: immutable version 1.0.0
  dataset/model archive, including action-weight arrays, lookup tables,
  two-link records, legacy four-link records, checkpoints, and hardware videos;
- Supplementary Archive S1: exhaustive MPC candidate tables, traces, logs, and
  replay evidence that are impractical or unnecessary to duplicate in Git.
- Supplementary Data S2: the 12-cell action-weight HDF5 package, schema,
  manifest, and checksums.
- Supplementary Data S3: the frozen ATC-50 lookup table, schema, portable
  event-query implementation, and checksums.

## GitHub merge gate

1. Merge the audited pull request into `main` without rewriting history.
2. Record the immutable public commit in the manuscript and supplementary
   README.
3. Run the unit tests, MPC release validator, table-regeneration script, and
   repository manifest/checksum audit.
4. Confirm that portable executables, manifests, and public metadata contain no
   workstation-absolute paths, credentials, or generated cache files. Original
   source snapshots with path literals must be explicitly labeled as
   provenance-only and must not be described as portable entry points.
5. Keep `LICENSE`, `DATA_LICENSE.md`, `CITATION.cff`, and `NOTICE` synchronized.
6. Confirm that no alternate plotting scripts, generated figure assets,
   manuscript drafts, or superseded result tables are tracked.

A new GitHub Release or tag is optional. It is not required when a code-only
correction is merged to `main` and the manuscript cites an exact commit.

## Zenodo dataset audit

1. Confirm that the record title, version (1.0.0), DOI, manifest, and description
   match the deposited dataset/model archive.
2. Verify every deposited file against the file-level SHA-256 manifest and the
   top-level ZIP checksum sidecar.
3. Retain the original videos (`Flat_walking.mp4` and
   `Ueven_foot_placement.mp4`), trial metadata, and any available calibration,
   command, state, encoder, timing, and failure-label records.
4. State the Apache License 2.0 for author-owned material and identify any
   third-party exceptions.
5. Do not describe the legacy four-link files in version 1.0.0 as the current
   true-hard-mask evaluation; cite the exact GitHub commit for the latter.

Merging corrected software into GitHub does not require a new Zenodo version
unless the deposited dataset files or their metadata/checksums are changed.

## Supplementary Archive S1 audit

1. Include the unified 12-case MPC package README, environment specification,
   accepted case table, complete candidate searches, traces, solve logs,
   validation report, and two fresh-process replays per accepted case.
2. Include a machine-readable manifest and an external ZIP SHA-256 sidecar.
3. State that S1 supports the corrected MPC result and is not contained in
   Zenodo dataset version 1.0.0.

## Supplementary Data S2 audit

1. Include all 12 action-weight HDF5 cells and a machine-readable schema.
2. Include file-level SHA-256 checksums and an external ZIP checksum sidecar.
3. Record RNG seed 20260716, the common-success eligibility rule, and the
   center-of-mass displacement threshold.
4. State that S2 supports Table II and is also preserved in Zenodo version
   1.0.0.

## Supplementary Data S3 audit

1. Include `working_save-ATC-50`, its HDF5 dataset name, shape, axis ranges,
   action-code semantics, and 1.8-degree directed event-update rule.
2. Include the portable lookup module and state that the supplement does not
   claim byte-for-byte regeneration of the historical table artifact.
3. Include a file-level manifest and an external ZIP SHA-256 sidecar.
4. State that S3 supports hardware lookup reproducibility, is distinct from the
   action-weight S2 archive, and is also preserved in Zenodo version 1.0.0.

## Limitations to disclose

- Some Paper2 case entry points require their complete relative dependencies;
  these are simulation materials and are not part of the hardware DOI.
- The two-link hardware lookup is event-updated and must not be described as a
  transition-locked expert route.
- A hardware success rate should not be inferred unless every attempted trial
  and its outcome are represented in the archived evidence.
