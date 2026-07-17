# Data archive checklist for the RAS submission

The GitHub repository is the executable code release. The larger immutable data
archive should receive its own DOI (for example, from Zenodo or Mendeley Data)
and should be cited from the manuscript's Data Availability statement.

## Required archive contents

1. `README.md` with the manuscript title, authors, contact, license, software
   versions, directory map, and exact Git commit used for the paper.
2. Table III source maps: the active, negative-expert, and positive-expert HDF5
   files, with SHA-256 hashes and the sampled-grid definition.
3. Appendix D source arrays for all 12 condition/weight cells: controller status,
   mechanical energy, COM displacement, and Cmt arrays. Preserve the pre-filter
   results and the final `D > 0.01 m` result/manifest files.
4. Tables IV-V per-case inputs and outputs: the 12 case definitions, checkpoints,
   captured stdout, timing/error summaries, and source hashes.
5. Table VI and Appendix B four-link trial-level CSV files, configuration manifests,
   controller/checkpoint identifiers, training logs, and paired-inference output.
6. Hardware evidence: a complete trial manifest, commanded footholds, timestamps,
   state/encoder logs, calibration and parameter-identification files, failure
   labels, and the original videos. Do not report a hardware success rate unless
   every attempted trial is represented.
7. Figure source data and deterministic scripts, including
   `analysis/plot_figure08_four_link_diagnostics.py` and its PNG/PDF/SVG outputs.
8. One machine-readable `manifest.csv` containing every archived file's relative
   path, byte size, SHA-256 hash, provenance, and manuscript table/figure link.

## Deposit procedure

1. Freeze the final Git commit and write the hash into the archive README.
2. Copy the items above into a versioned, read-only directory without changing
   filenames or numeric contents.
3. Generate and verify the SHA-256 manifest.
4. Upload the directory to a DOI-granting repository and publish version 1.
5. Add the DOI to the manuscript, cite the dataset in the references, and keep
   later corrections as new archive versions rather than overwriting version 1.

## Current limitations to disclose

- The released Git repository contains processed summaries and replay material,
  but not every large HDF5 source array or complete hardware trial log.
- The Appendix D `D > 0.01 m` values are reproducible from the archived metric
  arrays and recorded recovery rule; a clean rerun should additionally archive
  the newly written `D_save*.h5` files.
- The two-link hardware lookup is event-updated; it must not be described as a
  transition-locked expert route.
