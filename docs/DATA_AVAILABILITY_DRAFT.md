# Draft manuscript statements

## Data availability

The raw and processed data supporting this study, including two-link action
maps, torque-penalty outputs, the 12 multi-step case artifacts, four-link
per-trial evaluation CSVs, selector-training data, training logs, and the exact
model checkpoints used for the reported analyses, will be deposited in a
versioned research-data repository at **[DOI TO BE MINTED]**. Each file is
listed in a machine-readable manifest with a SHA-256 checksum. Until the DOI is
public, the data are available from the corresponding author for the purpose
of reproducibility checking. Author-owned processed data and supplementary
videos are licensed under Apache-2.0 when distributed; the future archive
metadata must repeat that license and the scope in `DATA_LICENSE.md`.

## Code availability

Frozen training, evaluation, and external statistical-analysis code is
available under Apache-2.0 at `https://github.com/QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.
The manuscript identifies the exact public Git commit used for submission.
The release records the mapping from every manuscript table and
figure to the corresponding script, configuration, checkpoints, raw outputs,
and checksums. The two-link experiments use an offline state-to-sign lookup
table, whereas the four-link stress test uses a learned online selector that
chooses a sign at transition onset.

Replace every bracketed placeholder only after the corresponding public record
exists.
