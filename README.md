# Sign-Constrained Reinforcement Learning for Bipedal Foot Placement

Reproducibility materials for the manuscript by Qi Tao, Yiyou Liu, Amir
Degani, and Mingyi Liu.

Canonical code repository:
`QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.

## What is in this repository

- `src/`: frozen, unmodified experiment and training programs, organized by
  experiment family.
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
summary CSVs required to regenerate Tables VII-XI and the paired inference. The
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

## Two-link action-weight Cmt correction

The continuous-action evaluator had treated the clipped policy output as the
physical hip torque in the equations of motion but multiplied it by the fixed
4 N m torque bound a second time in the positive-work accumulator. The
corrected evaluators under `src/two_link/action_weight/` now accumulate
`max(tau_h * omega_h, 0) * dt` with the applied torque exactly once. They also
correct the negative-expert Cmt output array and keep every write inside its
nominal action-weight condition directory.

The retained HDF5 arrays are not rewritten. Consequently, the archived
action-weight sweep remains an exploratory provenance record rather than a
controlled ablation. The deterministic correction, its validity conditions,
and the known archive limitations are documented in `docs/CMT_METRIC.md`.
Its historical `Proposed` row is a non-deployable two-rollout fusion with a
success-gating defect: 227,935 single-success trajectories had first action
zero, and 216,946 of them lay on the 443,163-state three-method common mask and
were minimized against a failed expert's initialized zero (48.954% of that
mask). The row is retained only for forensic provenance and is not a valid
method-ranking quantity.
The evaluator data root is configurable through
`ENERGY_COMPARISON_DATA_ROOT`; the archived Windows location is retained only
as the default for provenance.
The per-cell fusion counts are preserved in
`results/action_weight_fusion_audit_12_cells.csv`; the non-published local-copy
sync and 16-script verification procedure is in
`docs/ACTION_WEIGHT_LOCAL_SYNC.md`.

```bash
cd src/two_link/action_weight
python -m unittest discover -s tests -v
```

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

Large HDF5 arrays, raw per-trial CSVs, training logs, and checkpoints should be
deposited as a versioned research-data archive with a DOI. Replace the DOI
placeholder in `data/README.md` and in the manuscript Data Availability
statement only after the archive is public and its checksums have been tested.
Git LFS is optional for model files, but the DOI archive is the canonical data
record.

## License

No source or data license has yet been selected. The authors must add explicit
code and data licenses before public release; absence of a license does not
grant reuse permission.
