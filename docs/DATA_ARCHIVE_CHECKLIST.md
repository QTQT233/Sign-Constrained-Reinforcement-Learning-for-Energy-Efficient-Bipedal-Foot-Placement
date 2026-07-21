# Data archive checklist for the RAS submission

The public records have separate roles:

- GitHub: corrected executable software, repository-hosted models and
  simulation data, accepted results, provenance, and validators;
- Zenodo DOI <https://doi.org/10.5281/zenodo.21407986>: unchanged hardware
  evidence only;
- Supplementary Archive S1: exhaustive MPC candidate tables, traces, logs, and
  replay evidence that are impractical or unnecessary to duplicate in Git.

## GitHub merge gate

1. Merge the audited pull request into `main` without rewriting history.
2. Record the immutable public commit in the manuscript and supplementary
   README.
3. Run the unit tests, MPC release validator, table-regeneration script, and
   repository manifest/checksum audit.
4. Confirm that no workstation-absolute paths, credentials, or generated cache
   files are present.
5. Keep `LICENSE`, `DATA_LICENSE.md`, `CITATION.cff`, and `NOTICE` synchronized.

A new GitHub Release or tag is optional. It is not required when a code-only
correction is merged to `main` and the manuscript cites an exact commit.

## Hardware DOI audit

1. Confirm the Zenodo record title and description state that it is a hardware-
   data archive, not the source-code or MPC-results archive.
2. Verify every deposited hardware file against a file-level SHA-256 manifest.
3. Retain the original videos (`Flat_walking.mp4` and
   `Ueven_foot_placement.mp4`), trial metadata, and any available calibration,
   command, state, encoder, timing, and failure-label records.
4. State Apache-2.0 for author-owned material and identify any third-party
   exceptions.
5. Cite the DOI only for claims actually supported by that hardware record.

Because the hardware files are unchanged, merging corrected software into
GitHub does not require a new Zenodo version. A new Zenodo version is necessary
only if the deposited hardware files or their metadata/checksums are changed.

## Supplementary Archive S1 audit

1. Include the unified 12-case MPC package README, environment specification,
   accepted case table, complete candidate searches, traces, solve logs,
   validation report, and two fresh-process replays per accepted case.
2. Include a machine-readable manifest and an external ZIP SHA-256 sidecar.
3. State that S1 supports the corrected MPC result and is distinct from the
   hardware-only Zenodo DOI.

## Limitations to disclose

- Some legacy Paper2 entry points require original case-directory dependencies
  that are not part of the hardware DOI.
- The two-link hardware lookup is event-updated and must not be described as a
  transition-locked expert route.
- A hardware success rate should not be inferred unless every attempted trial
  and its outcome are represented in the archived evidence.
