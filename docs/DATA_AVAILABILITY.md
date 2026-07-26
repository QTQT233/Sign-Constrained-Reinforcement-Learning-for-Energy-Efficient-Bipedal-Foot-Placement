# Data and code availability

Zenodo version 1.0.0
(<https://doi.org/10.5281/zenodo.21407986>) archives research data and model
artifacts supporting this study: the frozen ATC-50 and coarse routing lookups,
selected simulation records, released checkpoints, two hardware-feasibility
videos, manifests, and checksums.

The exact GitHub commit cited in the manuscript is the executable record for
the submission. It contains the source code, repository-hosted checkpoints,
simulation inputs and results, the transition-level foot-placement
metric, validation programs, and table-regeneration utilities.

The journal supplementary files have the following scope:

- S1: complete MPC candidate searches, solver traces, validation, and
  deterministic replays; not included in Zenodo version 1.0.0.
- S2: submission-supplied strict-\(D>0.01\) 12-cell action-weight HDF5 package,
  schema, manifest, and checksums used for Table II; not included in Zenodo
  version 1.0.0.
- S3: frozen ATC-50 table and checksums from Zenodo version 1.0.0, together
  with the schema and portable event-query implementation at the cited GitHub
  commit.
- S4: landing-metric transition records and validation, five additional
  randomized four-link batches, and the hold-versus-requery control; mirrored
  under `supplementary/S4/` at the cited GitHub commit and not included in
  Zenodo version 1.0.0.

Author-generated software and data are distributed under Apache-2.0;
third-party components remain subject to their original terms.
