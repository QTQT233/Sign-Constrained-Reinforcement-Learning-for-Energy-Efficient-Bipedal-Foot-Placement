# Data and archive boundary

The public records supporting the manuscript have complementary roles.

- The immutable GitHub commit cited in the manuscript is the executable record
  for the study: source code, repository-hosted checkpoints, simulation
  inputs and outputs, validation programs, manifests, and checksums.
- Zenodo version 1.0.0
  (<https://doi.org/10.5281/zenodo.21407986>) archives the released research
  data and model artifacts, including the frozen ATC-50 and coarse routing
  lookups, selected simulation records, model checkpoints, two
  hardware-feasibility videos, manifests, and SHA-256 checksums.
- Supplementary Archive S1 contains the complete MPC candidate tables, solver
  traces, validation records, and deterministic replays supplied as a journal
  supplement. These files are not part of Zenodo version 1.0.0.
- Supplementary Data S2 is the strict-\(D>0.01\) 12-cell
  action-weight HDF5 package, schema, manifest, and checksums used for Table II.
  It is not part of Zenodo version 1.0.0.
- Supplementary Data S3 combines the frozen ATC-50 table and checksums archived
  in Zenodo version 1.0.0 with the schema and portable event-query
  implementation at the cited GitHub commit; the frozen table is identified by
  its file-level checksum.
- Supplementary Archive S4 contains the 144 transition-level landing records
  and validation, the primary 15-seed/10,800-case phase-aware three-expert
  confirmation, the direct-argmax learned-gate 2,160-case comparison, its
  phase-aware Figure 7 reanalysis, historical randomized batches, and the
  hold-versus-requery control. A
  compact mirror is tracked under `../supplementary/S4/` and
  `../results/four_link_three_expert_primary_2160/` and
  `../results/four_link_phase_aware_primary_2160/`; S4 is not part of Zenodo
  version 1.0.0.

The MPC result table is under `../results/paper2_mpc_unified_12case/`.
Horizontal foot-placement metrics are derived from the transition
records in `../supplementary/S4/landing_metric/`; historical output logs remain
provenance records and are not used as the manuscript landing metric.

Repository-wide file sizes and SHA-256 values are recorded in
`../MANIFEST.csv` and `../CHECKSUMS.sha256`. Author-owned code and data are
distributed under Apache-2.0; third-party components remain subject to their
original terms.
