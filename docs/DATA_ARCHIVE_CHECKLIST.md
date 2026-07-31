# Data archive scope and verification record

The public records have separate roles:

- GitHub: executable software, repository-hosted models and simulation data,
  selected results, provenance, validators, and a compact S4 mirror;
- Zenodo DOI <https://doi.org/10.5281/zenodo.21407986>: version-1.0.0 research
  data/model artifacts and two hardware-feasibility videos;
- Supplementary Archive S1: exhaustive MPC candidate tables, traces, logs, and
  replay evidence that are impractical or unnecessary to duplicate in Git.
- Supplementary Data S2: the journal-supplement strict-\(D>0.01\) 12-cell
  action-weight HDF5 package, schema, manifest, and checksums used for Table II;
  not included in Zenodo version 1.0.0.
- Supplementary Data S3: the frozen ATC-50 lookup table, schema, portable
  event-query implementation, and checksums; the table artifact is in Zenodo
  version 1.0.0.
- Supplementary Archive S4: horizontal foot-target transition records, the
  primary five-batch phase-aware three-expert evaluation, the direct-argmax
  learned-gate 2,160-case comparison, its phase-aware Figure 7 analysis,
  additional randomized batches, and the
  hold-versus-requery control, mirrored at the cited GitHub commit.

## GitHub version-of-record checks

- The manuscript cites one immutable Git commit.
- Unit tests, the MPC release validator, table-regeneration utilities, and the
  repository manifest/checksum verifier cover the released numerical record.
- Portable executables, manifests, and public metadata contain no credentials,
  generated caches, or workstation-absolute paths. Provenance-only source
  snapshots are identified separately from portable entry points.
- `LICENSE`, `DATA_LICENSE.md`, `CITATION.cff`, and `NOTICE` describe the same
  licensing and citation scope.
- Dedicated manuscript-figure renderers, rendered manuscript assets,
  manuscript source files, and superseded result tables are excluded.

## Zenodo version-1.0.0 record

1. Confirm that the title, description, README, data dictionary, and manifest
   describe the same data/model/video scope.
2. Verify every deposited file against the file-level SHA-256 manifest.
3. Retain routing lookups, released model artifacts, selected records, and the
   original videos
   (`Flat_walking.mp4` and `Ueven_foot_placement.mp4`).
4. State Apache-2.0 for author-owned material and identify any third-party
   exceptions.
5. Do not describe S1, strict S2, or S4 as part of Zenodo version 1.0.0.

## Supplementary Archive S1 contents

1. Include the unified 12-case MPC package README, environment specification,
   selected-case table, complete candidate searches, traces, solve logs,
   validation report, and two verification replays per selected parameter
   pair.
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
2. Include the portable lookup module and identify the frozen table by its
   file-level SHA-256 checksum.
3. Include a file-level manifest and an external ZIP SHA-256 sidecar.
4. State that S3 supports hardware lookup reproducibility and identify the
   archived Zenodo table artifact.

## Supplementary Archive S4 contents

1. Include 144 transition-level horizontal foot-target records, current
   case/controller summaries, and the geometry/aggregation validator.
2. Include the 10,800-case primary phase-aware three-expert matched record,
   the direct-argmax learned-gate 2,160-case comparison, the phase-aware
   Figure 7 analysis, additional randomized batches, and the 2,160-case
   hold-versus-requery control.
3. Include a scoped manifest and checksums, and verify the compact GitHub
   mirror against the journal supplement.
4. State that S4 is not included in Zenodo version 1.0.0.

## Scope boundaries

- Paper2 logs and source snapshots preserve the source calculations; the
  manuscript foot-target metric is regenerated from S4 transition records.
- The two-link hardware lookup is event-updated, whereas the simulation route
  is transition-locked.
- Hardware evidence establishes motion feasibility; no hardware success-rate
  statistic is reported because the archive is not an attempt-complete trial
  ledger.
