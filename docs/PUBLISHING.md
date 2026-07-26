# Release and archive record

Target repository:
`QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.

The manuscript cites an immutable commit on `main` as the software version of
record.

## Verification

The release is verified from the repository root with:

```bash
python analysis/synchronize_landing_metric_outputs.py
python analysis/reproduce_paper2_tables.py
python supplementary/S4/landing_metric/code/recompute_landing_metrics.py
python src/paper2/push_off_grid/validate_release.py
python src/paper2/mpc/validate_release.py
python -m unittest discover -s tests -v
python supplementary/S4/build_manifest.py
python tools/build_manifest.py
python tools/verify_manifest.py
git diff --check
```

These commands verify the generated numerical files, repository and S4
manifests, and the absence of workstation paths or credentials from portable
files.

## Archive scope

- Zenodo v1.0.0 (`10.5281/zenodo.21407986`) contains released data/model
  artifacts and hardware videos, including the ATC-50 artifact used by S3.
  S1, the strict-\(D>0.01\) S2 package, and S4 are not part of this version.
- S1 contains the complete MPC candidate-level evidence and is supplied with
  the journal submission.
- S2 is the submission-supplied strict-\(D>0.01\) 12-cell action-weight HDF5
  package, schema, manifest, and checksums used for Table II.
- S3 combines the frozen ATC-50 artifact with the schema and portable query
  implementation.
- S4 contains the landing, additional-batch, and hold-versus-requery records,
  is mirrored at the cited GitHub commit, and is not part of Zenodo v1.0.0.
