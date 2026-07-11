# Sign-Constrained Reinforcement Learning for Bipedal Foot Placement

Reproducibility materials for the manuscript by Qi Tao, Yiyou Liu, Amir
Degani, and Mingyi Liu.

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

## Reproducibility status

The manuscript originally reported four-link success rates of 45.1% for active
PPO and 44.6% for the online selector. Those values came from the 5 July 2026
output directory, but the exact script revision that produced that directory
was not retained. The reproducible primary release therefore uses the coherent
8 July 2026 V23-success-gate dataset and its matching frozen evaluator. It gives
979/2160 (45.3%) successes for active PPO and 972/2160 (45.0%) for the online
selector. See `docs/VERSION_AUDIT.md` before citing any number.

The manuscript must not mix values, confidence intervals, or failure counts
between these two result versions.

## Quick statistical reproduction

```bash
python -m pip install -r environment/requirements-analysis.txt
python analysis/four_link_paired_inference.py \
  --paired data/four_link/v23_success/paired_trials.csv \
  --terminal data/four_link/v23_success/terminal_reason_summary.csv \
  --output results/four_link_statistics
```

The script verifies the input SHA-256, row count, and pair identifiers before
computing the paired success table, paired risk-difference intervals, exact
McNemar test, post-hoc non-inferiority sensitivity analysis, and terminal-reason
transitions.

To verify the frozen evaluator, checkpoints, and portable path launcher before
a full 2160-case rerun:

```bash
python tools/run_frozen_four_link_evaluator.py \
  --output results/rerun_v23_success --check-only
```

Remove `--check-only` to execute the full evaluator. The launcher changes only
two path assignments in a temporary copy and verifies the archived source hash;
the frozen source file itself remains unmodified.

## Reproduce revised Tables IV-V

The read-only Paper2 audit completed 59 of 60 non-MPC runs. The sole timeout is
a TVLQR case with stochastic sampling and an unbounded success-retry loop; no
traceable PID entry point was found. The revised tables therefore report the
four complete replay families plus the retained MPC artifacts:

```bash
python analysis/reproduce_paper2_tables.py
```

This writes a 60-row combined case table and manuscript-facing summaries under
`results/paper2_current/`. The captured stdout/stderr, source hashes, missing
landing residual, timeout, and read-only tree checks are retained under
`data/paper2/readonly_rerun/`. A full rerun additionally requires the DOI case
bundle because the frozen entry points load relative checkpoints and arrays.

Run the full read-only audit with an activated locked environment as follows:

```bash
python tools/audit_paper2_readonly.py \
  --paper2-root /path/to/paper2_case_bundle \
  --output-root paper2_rerun_logs
```

## Important interpretation boundary

The existing `per_step_sign`, `active_full`, and `bidirectional` scripts do not
form a valid mechanism ablation. `per_step_sign` exposes the same 27 actuator
torque triples as unrestricted active control at every step, and `active_full`
and `bidirectional` use the same normalized torque penalty. Their current runs
are retained only as implementation/checkpoint-sensitivity diagnostics. The
factorial experiment required for a causal persistence claim is specified in
`docs/ABLATION_PROTOCOL.md`.

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
