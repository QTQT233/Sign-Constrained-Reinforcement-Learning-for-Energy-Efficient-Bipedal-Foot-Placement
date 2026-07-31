# Frozen five-batch phase-aware three-expert confirmation

This bundle contains the prespecified independent confirmation of the
phase-aware three-expert four-link controller. The experiment uses 15 new
evaluation seeds in five batches, with 720 matched cases per seed and 10,800
cases per controller. Every controller receives the same initial state,
endpoint command, walking direction, and initialization phase in a matched
case; policy actions are deterministic.

The frozen protocol, controller hashes, route thresholds, statistical
endpoints, stopping rule, and bootstrap design are recorded in
`protocol.json`. `FROZEN_SHA256SUMS.txt` and the execution-amendment files
document the pre-analysis author-workspace source state and the metadata-only
compatibility change used by the route-bank wrapper. `PROVENANCE.md` maps
their hash-only source names to the portable public entry points; the two
frozen ledgers are not current-payload checksum inventories.

The primary router uses the released 12--128--64--3 learned gate. It evaluates
the normalized 12-component initial state and command once at transition
start. The positive and negative gate probabilities are compared with
phase-specific confidence thresholds; the unrestricted Active PPO expert is
used when neither one-sided threshold passes. This calibrated decision rule is
distinct from the direct three-class argmax gate retained in the 2,160-case
diagnostic archive, although both use the same gate checkpoint and the same
three low-level experts.

`GATE_HASH_AMENDMENT_02.json` records the gate hash omitted from the original
frozen protocol and the execution-time versus portable analysis-script hash
mapping. It changes no controller, route, threshold, case, or reported result.

## Contents

- `code/run_route_bank_seed.py`: generate and validate one
  Active/non-negative/non-positive route bank.
- `code/replay_hard_mask.py`: replay the released hard-mask checkpoint on the
  exact route-bank initial states.
- `code/run_five_batches_parallel.py`: bounded parallel orchestration for all
  15 seeds.
- `code/analyze_confirmation.py`: frozen 20,000-replicate stratified-bootstrap
  analysis and exact McNemar tests.
- `code/build_manuscript_metrics.py`: controller, actuator-component,
  target-cell, batch, sensitivity, and three-way common-valid tables.
- `PROVENANCE.md`: mapping from the frozen author-workspace source-hash ledger
  to the portable public entry points.
- `results/`: immutable pooled, batch, seed, target-cell, integrity, and
  matched-case outputs.
- `manuscript_metrics/`: machine-readable values used in Tables VI--VIII and
  Appendix B.

The published `matched_trials_10800.csv.gz` is sufficient to audit the
reported success, valid-\(C_{\mathrm{mt}}\), route, and terminal-reason
statistics without rerunning simulation. A full fresh simulation rerun
requires the repository checkpoints and evaluator plus a Python environment
with NumPy, pandas, SciPy, PyTorch, and tqdm.

## Full rerun

Install the pinned evaluation and analysis dependencies, then run from the
repository root:

```bash
python -m pip install -r environment/requirements-five-batch.txt
python supplementary/S4/five_batch_phase_aware_confirmation/code/run_five_batches_parallel.py route --max-workers 3
python supplementary/S4/five_batch_phase_aware_confirmation/code/run_five_batches_parallel.py hard --max-workers 5
python supplementary/S4/five_batch_phase_aware_confirmation/code/analyze_confirmation.py
python supplementary/S4/five_batch_phase_aware_confirmation/code/build_manuscript_metrics.py
```

The route-bank phase is the more memory-intensive stage; three workers were
used for the reported run. Five hard-mask workers were run concurrently.
Completed seed manifests are validated and skipped, so interruption recovery
occurs at the seed boundary. The analysis refuses to overwrite a nonempty
`reproduced_results/` directory.

## Frozen result

All frozen acceptance criteria passed. In 10,800 cases, the proposed selector
and Active PPO succeeded in 48.583% and 48.417%, respectively. The one-sided
95% lower confidence bound for proposed minus Active was -0.120 percentage
points, above the prespecified -1-point preservation margin. In 2,539
Active/proposed common-valid cases, mean \(C_{\mathrm{mt}}\) was 1.523 for
Active and 0.795 for the proposed selector. Relative to the hard mask, the
proposed selector increased success by 2.009 percentage points and reduced
common-valid mean \(C_{\mathrm{mt}}\) from 1.411 to 0.614
(\(n=2,303\)).

Author-generated software and data in this bundle are licensed under Apache
License 2.0; third-party dependencies and checkpoints retain their stated
terms.
