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
- `data/`: author-generated arrays, retained trials, and derived tables that are
  distributed with the executable repository record.
- `docs/`: the table/figure provenance map, version audit, data dictionary,
  non-inferiority analysis, and corrected ablation protocol.
- `environment/`: analysis dependencies and the remaining environment-lock
  requirement for training/evaluation.

The older files in the repository root are retained for history. New users
should follow the structured paths above.

Maintainers should follow `docs/PUBLISHING.md` for the branch, pull-request,
commit-pinning, and hardware-data DOI workflow; do not force-push the
historical repository.

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
protocol completed all 12 TVLQR cases. After synchronizing the corrected
raised-1.145-r1 source (shared initial state, source-aligned recovery, and the
terminal stance action in the positive-work sum), the full seed-0 rerun gave
mean Cmt 0.234644 and sample SD 0.019875. Revised Tables IV-V
therefore report five complete replay families plus the accepted source-aligned
unified MPC rerun:

```bash
python analysis/reproduce_paper2_tables.py
```

This writes a 72-row combined case table and manuscript-facing summaries under
`results/paper2_current/`. The captured stdout/stderr, source hashes, missing
landing residual, timeout, and read-only tree checks are retained under
`data/paper2/readonly_rerun/`. Some legacy frozen entry points additionally
require their original case-directory dependencies; those dependencies are
not part of the hardware-only Zenodo record and must not be described as such.

The repository-relative MPC release is under `src/paper2/mpc/`, with its
12-case configuration in `configs/paper2_mpc_unified_12_cases.json` and
accepted results in `results/paper2_mpc_unified_12case/`. All 12 cases succeeded;
the accepted mean Cmt is 0.198945099 (sample SD 0.021963452), and two independent
replays matched every accepted scientific result exactly. The superseded MPC
CSV is retained under `results/legacy/`. Validate the release without running
the expensive search using:

```bash
python src/paper2/mpc/validate_release.py
```

See `docs/PAPER2_MPC_UNIFIED_12CASE.md` for the search, initial-state, reset,
edge-closure, and evidence boundaries.

TVLQR can be verified and rerun independently from the larger bundle:

```bash
python -m pip install -r environment/requirements-tvlqr.txt
python tools/run_frozen_tvlqr.py --check-only
python tools/run_frozen_tvlqr.py --all --seed 0 --timeout 60 \
  --output results/tvlqr_seed0
```

The launcher does not edit the repository source files at runtime. It verifies
the recorded hashes of the canonical corrected sources, substitutes archived
workstation model paths in a temporary copy, and captures per-case results.
TVLQR is interpreted as a local reference-tracking energy comparator, not as
an independent global foot-placement controller. See `docs/TVLQR_RELEASE.md`.

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
`action_value_weight=0.026`). In the V1.0.0 evidence archive, Active-full uses
the same 27 physical torque triples with zero torque penalty (`U0`,
`action_value_weight=0`) and is retained as a single warm-started reward
diagnostic, not as a matched-seed from-scratch U0 estimate. The current
main-branch Active-full training entry points are scratch-only: they contain no
policy/critic loading path and refuse a non-empty output directory so that old
checkpoints or appended logs cannot be mixed into a new run. This source change
does not retroactively alter the V1.0.0 checkpoint lineage. `per_step_sign`
exposes the same 27 physical torque triples as U1 and is an encoding diagnostic,
not an action-mask baseline. The
online selector is deployable; the oracle one-sided-policy envelope is a
non-deployable hindsight diagnostic. The confirmatory experiment required for
a causal persistence claim is specified in `docs/ABLATION_PROTOCOL.md`.

## Data, code, and hardware archive boundary

This GitHub repository is the canonical record for the corrected source code,
repository-hosted models and simulation data, accepted MPC results, provenance
files, and executable validation scripts. The manuscript should identify the
exact Git commit used for submission. A GitHub Release is optional: merging an
audited pull request into `main` and citing an immutable commit is sufficient
to pin this software version.

The unchanged hardware evidence is archived separately at
<https://doi.org/10.5281/zenodo.21407986>. That Zenodo record is the hardware-
data record only; it must not be cited as the source of the corrected software
or the unified MPC rerun. Exhaustive MPC candidate tables and process traces
may instead be supplied with the journal submission as Supplementary Archive
S1, while the accepted case table and validators remain in this repository.

Use `docs/DATA_ARCHIVE_CHECKLIST.md` to audit the hardware DOI scope, checksums,
license, and the separation between Zenodo, GitHub, and submission supplements.

The exact source-aligned rerun records for the raised, nominal-length-1.145,
repetition-1 Discrete PPO, Continuous-torque PPO, and LIPM cases are documented
in `docs/PAPER2_SOURCE_ALIGNED_R1_RERUN.md`. The canonical Continuous and LIPM
entrypoints are synchronized to their accepted 1.00/0.89 and 0.95/0.89 recovery
settings, respectively. Source hashes and raw outputs are retained with the
records.

## License

The source code, analysis code, documentation, and author-owned processed data
in this release are distributed under the Apache License 2.0. See the
repository-root `LICENSE` and `NOTICE` files. `DATA_LICENSE.md` defines the data
and supplementary-video scope and records the third-party-material boundary.
