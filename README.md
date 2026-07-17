# Sign-Constrained Reinforcement Learning for Bipedal Foot Placement

Reproducibility materials for the manuscript by Qi Tao, Yiyou Liu, Amir
Degani, and Mingyi Liu.

Canonical code repository:
`QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.

## What is in this repository

- `src/`: versioned experiment and training programs, organized by experiment
  family. The 12 legacy action-weight evaluators used in Appendix D include the documented negative-expert
  output fix and fixed evaluation seed.
- `analysis/`: external read-only scripts that regenerate manuscript-facing
  statistics from frozen CSV outputs.
- `data/`: small derived tables and instructions for obtaining the DOI-backed
  artifact bundle containing raw arrays, evaluation trials, logs, and model
  checkpoints.
- `docs/`: the table/figure provenance map, version audit, data dictionary,
  non-inferiority analysis, and corrected ablation protocol.
- `environment/`: analysis dependencies and the remaining environment-lock
  requirement for training/evaluation.

The older files in the repository root are retained for history. New users
should follow the structured paths above.

Maintainers should follow `docs/PUBLISHING.md` for the branch, draft-PR, release,
and DOI workflow; do not force-push the historical repository.

## Reproducibility status

The manuscript-primary four-link release is the V22_3/V9 fixed-checkpoint output
from 5 July 2026: active PPO succeeded in 975/2160 matched cases (45.1%) and the
online selector in 964/2160 (44.6%). The release contains all 19 trial-level and
summary CSVs required to regenerate Table VI, Appendix B, and the paired inference. The
exact historical extended-evaluator bytes were not retained. A separately named
legacy reconstruction restores the documented V22_3 touchdown bounds and is
never represented as the missing original source. A full 2160-case rerun of
that reconstruction reproduced all 19 archived CSVs byte-for-byte (19/19
matching SHA-256 hashes); see `docs/VERSION_AUDIT.md` and the released
reconstruction-validation record.

## Quick statistical reproduction

```bash
python -m pip install -r environment/requirements-analysis.txt
python analysis/four_link_paired_inference.py \
  --paired data/four_link/legacy_manuscript/paired_trials.csv \
  --terminal data/four_link/legacy_manuscript/terminal_reason_summary.csv \
  --output results/four_link_statistics/legacy_manuscript
python analysis/validate_four_link_pairing.py
python analysis/recompute_four_link_tables.py \
  data/four_link/legacy_manuscript results/four_link_tables_legacy_manuscript
```

The script verifies the input SHA-256, row count, and pair identifiers before
computing the paired success table, paired risk-difference intervals, exact
McNemar test, post-hoc non-inferiority sensitivity analysis, and terminal-reason
transitions.

To verify the reconstructed legacy evaluator, checkpoints, and portable path
launcher before a full 2160-case rerun:

```bash
python tools/run_frozen_four_link_evaluator.py \
  --output results/rerun_legacy_v22_3 --check-only
```

Remove `--check-only` to execute the full evaluator. The launcher changes only
the model/output paths in a temporary copy and verifies the reconstruction hash.
Statistical reproduction from the retained trial-level outputs is exact; the
reconstruction is an output-validated recovery route rather than a claim that
the missing historical source bytes were recovered. The release validation run
completed all 2160 cases and reproduced every one of the 19 archived CSVs
byte-for-byte.

## Reproduce revised Tables IV-V

The initial read-only Paper2 audit completed 59 of 60 non-MPC runs; one
unseeded TVLQR process exceeded the 180-s audit timeout, and no traceable PID
entry point was found. The release now includes all four TVLQR checkpoints and
a portable runner that applies seed 0 only in a temporary copy. This fixed-seed
protocol completed all 12 TVLQR cases and reproduced the manuscript-level
reference trend (mean Cmt 0.232348, sample SD 0.015446). Revised Tables IV-V
therefore report five complete replay families plus retained MPC artifacts:

```bash
python analysis/reproduce_paper2_tables.py
```

This writes a 72-row combined case table and manuscript-facing summaries under
`results/paper2_current/`. The captured stdout/stderr, source hashes, missing
landing residual, timeout, and read-only tree checks are retained under
`data/paper2/readonly_rerun/`. A full rerun additionally requires the DOI case
bundle because the frozen entry points load relative checkpoints and arrays.

TVLQR can be verified and rerun independently from the larger bundle:

```bash
python -m pip install -r environment/requirements-tvlqr.txt
python tools/run_frozen_tvlqr.py --check-only
python tools/run_frozen_tvlqr.py --all --seed 0 --timeout 60 \
  --output results/tvlqr_seed0
```

The 12 frozen source files are not edited. The launcher verifies their hashes,
substitutes the archived workstation model paths in a temporary copy, and
captures per-case results. TVLQR is interpreted as a local reference-tracking
energy comparator, not as an independent global foot-placement controller.
See `docs/TVLQR_RELEASE.md`.

Run the full read-only audit with an activated locked environment as follows:

```bash
python tools/audit_paper2_readonly.py \
  --paper2-root /path/to/paper2_case_bundle \
  --output-root paper2_rerun_logs
```

## Two-link action-weight audit (Appendix D)

The 12 canonical `Whole_energy_comparison_low_dim.py` files under
`src/two_link/action_weight/` use random seed `20260716`, write the negative
expert's Cmt array to the negative-expert file, and accumulate continuous-
controller positive commanded work from the applied physical torque exactly
once. They now require center-of-mass displacement `D > 0.01 m` before a Cmt
value is admitted and archive `D_save` for every independently executed
controller. The expert-status fusion branch remains complete: when both
experts succeed, the lower finite Cmt is retained; when only one expert
succeeds, that expert's finite value is retained. Only status `-2` denotes
expert failure.

The Appendix D audit is recomputed on the condition-specific common-feasible and
common-displacement-eligible mask shared by the Proposed, Active PPO, and
Continuous PPO evaluations. For the retained pre-threshold HDF5 archive,
positive-work displacement is recovered exactly as `D = E/(W*Cmt)`; zero-
energy/zero-Cmt successes are retained because they cannot create a small-
denominator tail. The paper-facing values and counts are in
`results/action_weight_table_ii_seed_20260716.csv`; full mean, median, sample
SD, 99th percentile, maximum, source hashes, and HDF5 hashes are in the
companion audit CSV and JSON manifest. The unfiltered release is preserved in
the three files containing `pre_0p01m_filter` in their names.

```bash
python analysis/recompute_action_weight_table_ii.py \
  --data-root /path/to/Energy_Comparison
cd src/two_link/action_weight
python -m unittest discover -s tests -v
```

The retained archive verifies the deterministic continuous-torque scale
correction and documents provenance/eligibility boundaries. It is not a
matched-retraining sensitivity experiment and does not estimate variability
across independent training seeds. See
`docs/TABLE_II_AUDIT.md`, `docs/CMT_METRIC.md`, and
`docs/ACTION_WEIGHT_LOCAL_SYNC.md` for the exact protocol and provenance.

## Important interpretation boundary

Active PPO is the penalized unrestricted control (`U1`,
`action_value_weight=0.026`). Active-full uses the same 27 physical torque
triples with zero torque penalty (`U0`, `action_value_weight=0`) and is retained
as a single warm-started reward-ablation diagnostic. It is not a matched-seed
from-scratch U0 estimate. `per_step_sign` exposes the same 27 physical torque
triples as U1 and is an encoding diagnostic, not an action-mask baseline. The
online selector is deployable; the oracle one-sided-policy envelope is a
non-deployable hindsight diagnostic. The confirmatory experiment required for
a causal persistence claim is specified in `docs/ABLATION_PROTOCOL.md`.

## Data and model artifacts

Large HDF5 arrays, raw per-trial CSVs, training logs, checkpoints, and the two
hardware videos (`Flat_walking.mp4` and `Ueven_foot_placement.mp4`) should be
deposited as a versioned research-data archive with a DOI. Replace the DOI
placeholder in `data/README.md` and in the manuscript Data Availability
statement only after the archive is public and its checksums have been tested.
Git LFS is optional for model files, but the DOI archive is the canonical data
record.

Use `docs/DATA_ARCHIVE_CHECKLIST.md` to assemble the DOI-backed archive and its
file-level SHA-256 manifest for the Robotics and Autonomous Systems submission.

## License

The source code, analysis code, documentation, and author-owned processed data
in this release are distributed under the Apache License 2.0. See the
repository-root `LICENSE` and `NOTICE` files. `DATA_LICENSE.md` defines the data
and supplementary-video scope and records the third-party-material boundary.
