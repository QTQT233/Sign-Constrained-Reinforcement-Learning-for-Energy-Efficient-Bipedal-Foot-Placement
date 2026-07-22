# Publishing corrected code and evidence

Target repository:
`QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.

All changes should be reviewed on a branch and merged through a pull request.
Do not force-push or replace the historical repository.

## Code-only correction workflow

1. Inspect the complete diff against `main` and exclude unrelated files.
2. Run:

   ```bash
   python -m unittest discover -s tests -v
   python src/paper2/mpc/validate_release.py
   python analysis/reproduce_paper2_tables.py
   python tools/build_manifest.py
   git diff --check
   ```

3. Re-run the tests after rebuilding `MANIFEST.csv` and `CHECKSUMS.sha256`.
4. Push the audited branch and merge it into `main` without rewriting history.
5. Record the immutable public Git commit in the manuscript's Code
   Availability statement.

A separate GitHub Release or tag is optional. For the current correction, the
public commit on `main` is the authoritative software version pin.

## Zenodo boundary

The unchanged hardware evidence is archived separately at
<https://doi.org/10.5281/zenodo.21407986>. The DOI must not be represented as
containing the corrected software or unified MPC result. Because the hardware
payload is unchanged, this code merge does not require a new Zenodo version.
Create a new Zenodo version only when deposited hardware files or their
metadata/checksums change.

The complete MPC candidate-level record is supplied separately as
Supplementary Archive S1. Its README and checksum sidecar should identify the
exact Git commit used for the submission.

The 12-cell action-weight HDF5 record is supplied separately as Supplementary
Data S2 with its schema, manifest, and checksum sidecar.

The frozen ATC-50 table and portable event-query implementation are supplied
separately as Supplementary Data S3 with an axis/action schema, manifest, and
checksum sidecar. S3 does not claim byte-for-byte regeneration of the
historical table artifact.

## Merge gate

- all unit and reference-artifact tests pass;
- the MPC validator reports 12/12 successful accepted cases and exact replay
  agreement;
- manuscript-facing tables regenerate without an uncommitted diff;
- `MANIFEST.csv` and `CHECKSUMS.sha256` are current;
- repository code and author-owned data are licensed under Apache-2.0;
- portable executables and public metadata contain no private paths,
  credentials, cache files, or bracketed DOI placeholders; original frozen
  source snapshots are explicitly identified as provenance-only rather than
  portable entry points;
- no superseded result tables, manuscript drafts, alternate figure-rendering
  scripts, or generated figure assets are present on the submission branch;
- `tools/verify_manifest.py` confirms complete file-set, size, and SHA-256
  agreement;
- the manuscript distinguishes the GitHub software commit, hardware Zenodo
  DOI, Supplementary Archive S1, Supplementary Data S2, and Supplementary Data
  S3.
