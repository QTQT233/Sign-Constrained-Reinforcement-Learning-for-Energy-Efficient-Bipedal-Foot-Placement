# Sign-Constrained Reinforcement Learning for Bipedal Foot Placement

Reproducibility materials for *Torque-Sign-Constrained Expert Routing with an
Active Fallback for Energy-Efficient Bipedal Foot Placement* by Qi Tao, Yiyou
Liu, Amir Degani, and Mingyi Liu.

This repository snapshot contains the manuscript-facing simulation code,
repository-hosted model checkpoints, numerical outputs, provenance records, validators,
and checksums.

## Evidence map

- `src/two_link/`: two-link routing, action-weight evaluation, mechanical-cost,
  and portable transition-locked and ATC-50 lookup implementations. The
  source-aligned Table I and executable Active-default route are regenerated
  from the released \(30^4\) HDF5 maps by
  `analysis/reproduce_table1_active_fallback.py`.
- `src/four_link/`: V22_3/V9 four-link training and the shared matched-case
  evaluator, including the scratch-trained true per-step hard-action-mask PPO.
- `src/paper2/mpc/`: repository-relative continuous-torque MPC implementation,
  accepted 12-case results, and validator.
- `data/four_link/true_action_mask_scratch_c090_epoch1275/`: the released
  controller-level and trial-level hard-mask source archive.
- `results/paper2_current/`: the current 72-row two-link comparison and the
  manuscript-facing Table IV–V summaries.
- `results/two_link_active_fallback_fixed12/`: the fixed 12-case
  Active-default routing record, including the unchanged push-off pairs.
- `results/paper2_mpc_unified_12case/`: accepted MPC cases, validation records,
  and deterministic replay checks.
- `results/four_link_statistics/true_action_mask_scratch_c090_epoch1275/`:
  paired fixed-checkpoint inference for the true-mask comparison.
- `results/four_link_active_default_confirmation/`: three held-out seed
  streams from the threshold-development-era Active-default experiment; these
  records are not pooled with the primary confirmation.
- `supplementary/S4/five_batch_phase_aware_confirmation/`: the primary
  five-batch, 15-seed, 10,800-case phase-aware three-expert confirmation,
  including the frozen protocol, matched records, analysis and Figure 10 code,
  manuscript table inputs, and checksums.
- `analysis/`: read-only numerical analysis and table-regeneration programs.
- `docs/`: protocol, metric, provenance, availability, and submission records.

Figure 10 regeneration is included with the five-batch record because the
figure is a direct visualization of released numerical results. Other
manuscript illustration assets are outside this numerical repository.

## Environment

For numerical analysis:

```bash
python -m pip install -r environment/requirements-analysis.txt
```

TVLQR uses the additional requirements recorded in
`environment/requirements-tvlqr.txt`.

Four-link checkpoint loading and evaluation use
`environment/requirements-four-link.txt`; the captured rerun environment and
its provenance boundary are documented under `environment/`.

## Four-link five-batch confirmation

The manuscript's primary four-link result uses five independently seeded
batches, 15 evaluation seed streams, and 10,800 matched trials per controller
on the shared V22_3/V9 command grid. The low-level checkpoints remain frozen.
The proposed router queries its three source experts once at transition start:
it gives priority to an eligible one-sided expert and uses unrestricted Active
PPO when neither one-sided branch is eligible.

Key pooled values are:

- proposed three-expert selector: 5,247/10,800 successes (48.6%);
- Active PPO: 5,229/10,800 successes (48.4%);
- hard mask: 5,030/10,800 successes (46.6%);
- Active versus proposed, 2,539 common-valid pairs: mean `Cmt` 1.523 versus
  0.795, a 47.8% ratio-of-means reduction;
- hard mask versus proposed, 2,303 common-valid pairs: mean `Cmt` 1.411 versus
  0.614, a 56.4% ratio-of-means reduction.

The proposed-minus-Active success difference was +0.167 percentage points.
Its prespecified one-sided 95% lower confidence bound was -0.120 percentage
points, above the -1-point success-preservation margin. Relative to the hard
mask, the proposed selector increased success by 2.009 percentage points.

Regenerate the complete confirmation from the repository root:

```bash
python supplementary/S4/five_batch_phase_aware_confirmation/code/run_five_batches_parallel.py route --max-workers 3
python supplementary/S4/five_batch_phase_aware_confirmation/code/run_five_batches_parallel.py hard --max-workers 5
python supplementary/S4/five_batch_phase_aware_confirmation/code/analyze_confirmation.py
python supplementary/S4/five_batch_phase_aware_confirmation/code/build_manuscript_metrics.py
python supplementary/S4/five_batch_phase_aware_confirmation/code/plot_figure10.py --dpi 400
```

Audit the released numerical record without rerunning the simulations:

```bash
python -m unittest tests.test_five_batch_phase_aware_release tests.test_s4_release -v
python supplementary/S4/build_manifest.py --check
```

The earlier 2,160-case Active-default and transition-persistent records remain
available for provenance and the independent hold-versus-requery
query-schedule control. They are not pooled with the five-batch confirmation.

## Two-link 12-case comparison

The fixed 12 matched multi-step cases use exhaustive controller-specific
post-reset recovery-grid searches while preserving the declared push-off pairs.
Mean `Cmt` is 0.122346306 for the Active-default one-sided route,
0.226620152 for discrete active PPO, 0.238107711 for continuous-torque PPO,
and 0.198945099 for continuous-torque MPC. All four controllers complete all
12 cases. The Active fallback was available to the proposed route but was not
invoked in these 36 transitions because a one-sided expert covered every
transition.

Regenerate Tables IV–V:

```bash
python analysis/reproduce_paper2_tables.py
```

Validate the accepted MPC release:

```bash
python src/paper2/mpc/validate_release.py
python -m unittest tests.test_paper2_mpc_unified -v
```

The full candidate-level MPC search, solver traces, two fresh-process replays,
and the two-link case artifact bundle used by the portable entry points are
provided separately as Supplementary Archive S1.

## Action-weight evaluation

The 12 current `Whole_energy_comparison_low_dim.py` evaluators use RNG
seed `20260716`, write the negative expert's `Cmt` to the negative-expert array,
and accumulate continuous-controller positive commanded work from applied
physical torque once per step. A value is eligible only when center-of-mass
displacement is greater than 0.01 m. The fusion rule remains complete: if both
experts succeed, the lower finite `Cmt` is used; if only one succeeds, that
expert is used. The checkpoint and output roots are configurable through
`ACTION_WEIGHT_MODEL_DIR` and `ACTION_WEIGHT_OUTPUT_DIR`; see
`src/two_link/action_weight/README.md`. Each evaluator uses a separate
case-specific checkpoint directory, with filenames and SHA-256 values pinned in
`configs/action_weight_checkpoint_manifest.json`.

The 12-cell HDF5 package and its schema/checksums are supplied separately as
Supplementary Data S2. Recompute the manuscript table from that package with:

```bash
python analysis/recompute_action_weight_table_ii.py --data-root /path/to/S2
```

## Routing and hardware lookup

The two-link simulations query an Active-default state-to-expert route once at
transition onset and hold the selected expert to termination. The \(30^4\)
source-aligned route covers 672,382/810,000 states, compared with
647,160/810,000 for unrestricted Active PPO and 562,003/810,000 for the
one-sided-expert union. The two-link hardware system
uses a different deployment representation: the 50^4-cell ATC-50 action lookup
is queried again whenever the stance-link angle changes by approximately 1.8°,
and the returned action is held between events. The four-link simulation uses a
learned transition-start selector. These three mechanisms are not
interchangeable.

## Data and archive boundary

- GitHub: current simulation code, model checkpoints, numerical outputs,
  manifests, checksums, and validators. Cite the exact immutable commit used
  for submission.
- Zenodo DOI [10.5281/zenodo.21407986](https://doi.org/10.5281/zenodo.21407986):
  version 1.0.0 data and model archive containing the 12-cell action-weight
  HDF5 package, routing lookups, two-link records, legacy four-link records,
  released model checkpoints, manifests, checksums, and the two hardware
  feasibility videos. The current true-hard-mask comparison and corrected
  executable sources are pinned by the manuscript's GitHub commit.
- Supplementary Archive S1: complete MPC candidate-level evidence and the
  terrain-separated checkpoint/action-map bundle used by the Paper2 fixed-case
  entry points.
- Supplementary Data S2: 12-cell action-weight HDF5 package and schema.
- Supplementary Data S3: frozen ATC-50 lookup table, schema, portable event-
  query implementation, and checksums. No byte-for-byte regeneration claim is
  made for the historical table artifact.
- Supplementary Archive S4: landing-metric records, the independent
  10,800-case phase-aware three-expert confirmation, the historical
  two-expert batches, hold-versus-requery analysis, regeneration programs,
  manifests, and checksums.

This code update does not alter the hardware videos or archived model artifacts
in Zenodo version 1.0.0.

## Release verification

```bash
python -m unittest discover -s tests -v
python src/paper2/mpc/validate_release.py
python analysis/reproduce_paper2_tables.py
python tools/build_manifest.py
python tools/verify_manifest.py
git diff --check
```

## License

Author-generated code and released data are distributed under the Apache
License 2.0. See
`LICENSE`, `DATA_LICENSE.md`, and `NOTICE`. Third-party components remain
subject to their original terms.
