# Manuscript data and code statements

## Data and code availability

The hardware evidence and supplementary videos are available from Zenodo at
<https://doi.org/10.5281/zenodo.21407986>. The source code, released model
checkpoints, two-link case-level outputs, fixed V22_3/V9 four-link trial-level
records, paired analyses, manifests, and checksums supporting the reported
simulation results are available from the GitHub repository at the exact commit
cited in the manuscript. Exhaustive MPC candidate tables, solver traces,
validation records, and two deterministic replay records are provided as
Supplementary Archive S1. The 12-cell action-weight HDF5 archive and its schema,
manifest, and checksums are provided as Supplementary Data S2. The ATC-50
lookup table, schema, portable event-query implementation, and checksums are
provided as Supplementary Data S3; no byte-for-byte table-regeneration claim is
made. Author-generated software and data are licensed under Apache-2.0;
third-party components remain subject to their original terms.

## Repository scope

The public GitHub record contains the numerical evidence and executable code
needed to regenerate the manuscript tables and statistical results. Manuscript
figure-rendering scripts and image assets are maintained in the authors'
offline submission backup and are not cited as public numerical reproducibility
evidence.

The two-link hardware controller uses an event-updated ATC-50 action lookup.
The two-link simulations use transition-start expert routing, and the four-link
stress test uses a learned transition-start selector.
