# Manuscript data and code statements

## Data and code availability

Zenodo version 1.0.0 at <https://doi.org/10.5281/zenodo.21407986> contains the
12-cell action-weight HDF5 archive, the ATC-50 and coarse routing lookups,
two-link case-level records, legacy four-link trial-level records, model
checkpoints, manifests, checksums, and two supplementary hardware-feasibility
videos. The current executable code, source-aligned two-link routing maps,
Active-default route, corrected fixed-12-case records, fixed V22_3/V9
true-hard-mask comparison, and held-out four-link Active-default confidence
confirmation are available from the GitHub repository at the exact commit cited
in the manuscript. Exhaustive MPC candidate tables, solver traces,
validation records, two deterministic replay records, and the two-link
checkpoint/action-map bundle required by the portable fixed-case entry points
are provided as Supplementary Archive S1. The action-weight archive is designated
Supplementary Data S2, and the ATC-50 table, schema, portable event-query
implementation, and checksums are designated Supplementary Data S3; no
byte-for-byte table-regeneration claim is made. Author-generated software and
released data are licensed under the Apache License 2.0; third-party components
remain subject to their original terms.

## Repository scope

The public GitHub record contains the numerical evidence and executable code
needed to regenerate the manuscript tables and statistical results. Manuscript
figure-rendering scripts and image assets are outside the scope of the public
numerical reproducibility record.

The two-link hardware controller uses an event-updated ATC-50 action lookup.
The two-link simulations use transition-start Active-default expert routing.
The four-link stress test reports the two-expert learned transition-start
selector and a separate Active-default confidence-gated three-expert
confirmation.
