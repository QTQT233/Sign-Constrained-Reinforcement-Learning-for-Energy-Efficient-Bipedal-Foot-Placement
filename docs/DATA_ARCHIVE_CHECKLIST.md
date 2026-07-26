# Data archive scope and verification record

The public records have separate roles:

- GitHub: executable software, repository-hosted models and simulation data,
  accepted results, provenance, validators, and a compact S4 mirror;
- Zenodo DOI <https://doi.org/10.5281/zenodo.21407986>: version-1.0.0 research
  data/model artifacts and two hardware-feasibility videos;
- Supplementary Archive S1: exhaustive MPC candidate tables, traces, logs, and
  replay evidence that are impractical or unnecessary to duplicate in Git.
- Supplementary Data S2: the submission-supplied strict-\(D>0.01\) 12-cell
  action-weight HDF5 package, schema, manifest, and checksums used for Table II;
  not included in Zenodo version 1.0.0.
- Supplementary Data S3: the frozen ATC-50 lookup table, schema, portable
  event-query implementation, and checksums; the table artifact is in Zenodo
  version 1.0.0.
- Supplementary Archive S4: landing-metric transition records, five
  additional randomized four-link batches, and the hold-versus-requery
  control, mirrored at the cited GitHub commit.

## GitHub release verification

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

The immutable `main` commit cited by the manuscript is the software version of
record.

## Zenodo version-1.0.0 record

1. Confirm that the title, description, README, data dictionary, and manifest
   describe the same data/model/video scope.
2. Verify every deposited file against the file-level SHA-256 manifest.
3. Retain routing lookups, released model artifacts, selected records, and the
   original videos
   (`Flat_walking.mp4` and `Ueven_foot_placement.mp4`).
4. State Apache-2.0 for author-owned material and identify any third-party
   exceptions.
5. Do not describe S1 or S4 as part of Zenodo version 1.0.0.

## Supplementary Archive S1 contents

1. Include the unified 12-case MPC package README, environment specification,
   accepted case table, complete candidate searches, traces, solve logs,
   validation report, and two independent verification replays per accepted
   case.
2. Include a machine-readable manifest and an external ZIP SHA-256 sidecar.
3. State that S1 supports the MPC result and is not included in Zenodo version
   1.0.0.

## Supplementary Data S2 contents

1. Include all 12 action-weight HDF5 cells and a machine-readable schema.
2. Include file-level SHA-256 checksums and an external ZIP checksum sidecar.
3. Record RNG seed 20260716, the common-success eligibility rule, and the
   center-of-mass displacement threshold.
4. State that S2 supports Table II and is not included in Zenodo version 1.0.0.

## Supplementary Data S3 contents

1. Include `working_save-ATC-50`, its HDF5 dataset name, shape, axis ranges,
   action-code semantics, and 1.8-degree directed event-update rule.
2. Include the portable lookup module and state that the supplement does not
   claim byte-for-byte regeneration of the historical table artifact.
3. Include a file-level manifest and an external ZIP SHA-256 sidecar.
4. State that S3 supports hardware lookup reproducibility and identify the
   archived Zenodo table artifact.

## Supplementary Archive S4 contents

1. Include 144 transition-level landing records, current case/controller
   summaries, and the geometry/aggregation validator.
2. Include the 10,800 additional randomized four-link matched trials and the
   2,160-case hold-versus-requery control.
3. Include a scoped manifest and checksums, and verify the compact GitHub
   mirror against the journal supplement.
4. State that S4 is not included in Zenodo version 1.0.0.

## Limitations to disclose

- Some historical Paper2 logs and source snapshots retain the deprecated
  landing reporter as provenance; the manuscript metric is regenerated from
  S4 transition records.
- The two-link hardware lookup is event-updated and must not be described as a
  transition-locked expert route.
- A hardware success rate should not be inferred unless every attempted trial
  and its outcome are represented in the archived evidence.
