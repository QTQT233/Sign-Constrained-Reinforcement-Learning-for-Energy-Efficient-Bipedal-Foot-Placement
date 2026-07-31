# Energy-Efficient Bipedal Foot Placement via Torque-Sign-Constrained Expert Routing

Reproducibility materials for *Energy-Efficient Bipedal Foot Placement via
Torque-Sign-Constrained Expert Routing* by Qi Tao, Yiyou Liu, Amir Degani,
and Mingyi Liu.

This repository contains the manuscript-facing simulation code,
repository-hosted model checkpoints, numerical outputs, provenance records,
validators, and checksums.

## Evidence map

- `src/two_link/`: two-link routing, action-weight evaluation, mechanical-cost,
  and portable transition-locked and ATC-50 lookup implementations. Table I is
  regenerated from the released 30^4 HDF5 maps by `analysis/reproduce_table1.py`;
  the older 10x30 map-generator snapshots are not part of this release.
- `src/four_link/`: V22_3/V9 four-link training, the shared evaluator, the
  learned 12--128--64--3 three-expert gate, and the scratch-trained true
  per-step hard-action-mask PPO.
- `src/paper2/mpc/`: repository-relative continuous-torque MPC implementation,
  accepted 12-case results, and validator.
- `src/paper2/push_off_grid/`: validator for the complete learned-controller
  recovery-grid record and its selected fixed-parameter replays.
- `data/four_link/true_action_mask_scratch_c090_epoch1275/`: the historical
  controller-level and trial-level true-mask comparison archive.
- `results/four_link_three_expert_primary_2160/`: the matched 2,160-case
  source rollouts and direct-argmax learned-gate comparison.
- `results/four_link_phase_aware_primary_2160/`: the phase-aware reanalysis
  used for the three-expert entries in the four-link summary figure.
- `results/paper2_current/`: the 72-row two-link comparison and the
  manuscript-facing Table IV–V summaries.
- `results/paper2_mpc_unified_12case/`: accepted MPC cases, validation records,
  and deterministic replay checks.
- `results/paper2_push_off_grid_12case/`: 518,400 candidate records, 36 selected
  minima, fixed-parameter replay values, provenance, and scoped checksums.
- `results/four_link_statistics/true_action_mask_scratch_c090_epoch1275/`:
  paired fixed-checkpoint inference for the true-mask comparison.
- `supplementary/S4/`: portable code and machine-readable records for the
  landing metric, the primary 15-seed/10,800-case phase-aware three-expert
  confirmation, the historical randomized batches, and the frozen
  hold-versus-requery query-schedule control.
- `analysis/`: non-simulation analysis, validation, and table-regeneration
  programs.
- `docs/`: protocol, metric, provenance, and availability records.

The repository does not publish manuscript figure-rendering scripts or image
assets because they are not part of the numerical reproducibility record. All
numerical values plotted in the manuscript remain traceable to the released CSV
and JSON records above.

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

## Four-link three-expert confirmation

The primary four-link controller uses the released 12--128--64--3 learned gate
once at transition start. The gate receives the normalized initial state and
command. Its positive and negative probabilities are compared with
reset-phase-specific confidence thresholds; unrestricted Active PPO is the
fallback when neither one-sided threshold passes. The selected expert is held
to termination.

The primary confirmation contains five independently seeded batches, 15 seed
streams, and 10,800 matched cases per controller. Every controller receives the
same sampled initial state, endpoint command, direction, and reset phase.
Frozen low-level checkpoints are unchanged.

Key pooled results are:

- phase-aware three-expert router: 5,247/10,800 successes (48.6%);
- Active PPO: 5,229/10,800 successes (48.4%);
- true hard mask: 5,030/10,800 successes (46.6%);
- Active/proposed common-valid cases (n=2,539): mean `Cmt` 1.523 versus
  0.795;
- hard-mask/proposed common-valid cases (n=2,303): mean `Cmt` 1.411 versus
  0.614.

The separate 2,160-case archive provides the matched source rollouts for both a
direct three-class-argmax comparison and the phase-aware reanalysis used in the
summary figure. These values are reported separately from the primary
five-batch confirmation. The two-expert selector remains an energy-focused
baseline.

The original frozen protocol omitted the three-expert gate from its hash
dictionary even though the analyzer used and recorded it. The original
protocol and result remain unchanged;
`GATE_HASH_AMENDMENT_02.json` binds the gate checkpoint and documents the
execution-time/portable analysis-source boundary without changing any route or
reported statistic.

Verify the released record:

```bash
python -m unittest tests.test_five_batch_phase_aware_release tests.test_figure7_phase_aware_release tests.test_s4_release -v
python supplementary/S4/build_manifest.py --check
```

## Two-link 12-case comparison

The 12 matched multi-step cases use the accepted fixed-parameter replay
records. Mean `Cmt` is 0.122346306 for the transition-start lookup router,
0.226620152 for unrestricted discrete PPO, 0.238107711 for continuous-torque
PPO, and 0.198945099 for continuous-torque MPC. The corresponding
ratio-of-means reductions are 46.0%, 48.6%, and 38.5%. The pooled
per-transition foot-placement MAEs are 0.041389, 0.044260, 0.040953, and
0.030285 m, respectively.

For each learned controller and case, both post-impact recovery-velocity
decrements were evaluated on `0.00:0.01:1.19 rad/s`, giving 14,400 candidates
per controller-case pair and 518,400 released candidate records. Eligibility
requires completion of all three transitions, positive COM displacement, and
finite `Cmt`. The selected within-grid minimum was replayed with the unchanged
fixed-parameter evaluator; all 36 energy, displacement, and `Cmt` values agree
with their search records within `1e-10`.

Regenerate Tables IV–V:

```bash
python src/paper2/push_off_grid/validate_release.py
python analysis/synchronize_landing_metric_outputs.py
python analysis/reproduce_paper2_tables.py
```

Validate the accepted MPC release:

```bash
python src/paper2/mpc/validate_release.py
python -m unittest tests.test_paper2_mpc_unified -v
```

The full candidate-level MPC search, solver traces, and two independent
verification replays per accepted case are supplied as Supplementary Archive S1.

## Action-weight evaluation

The 12 action-weight source snapshots use RNG seed `20260716`, write the
non-positive expert's `Cmt` to the corresponding array, and accumulate
continuous-controller positive commanded work from applied physical torque
once per step. A value is eligible only when center-of-mass displacement is
greater than 0.01 m. The fusion rule remains complete: if both experts succeed,
the lower finite `Cmt` is used; if only one succeeds, that expert is used. The
source snapshots are provenance records; portable regeneration is described in
`src/two_link/action_weight/README.md`.

The 12-cell HDF5 package and its schema/checksums are supplied separately as
Supplementary Data S2. Recompute the manuscript table from that package with:

```bash
python analysis/recompute_action_weight_table_ii.py --data-root /path/to/S2
```

## Routing and hardware lookup

The two-link simulations query a coarse state-to-expert route once at transition
onset and hold the selected expert to termination. The two-link hardware system
uses a different deployment representation: the 50^4-cell ATC-50 action lookup
is queried again whenever the stance-link angle changes by approximately 1.8°,
and the returned action is held between events. The four-link simulation uses a
learned transition-start selector. These three mechanisms are not
interchangeable.

## Data and archive boundary

- GitHub: current simulation code, model checkpoints, numerical outputs,
  manifests, checksums, and validators. Cite the exact immutable Git commit
  used for submission.
- Zenodo DOI [10.5281/zenodo.21407986](https://doi.org/10.5281/zenodo.21407986):
  version-1.0.0 data/model artifacts and supplementary hardware videos. S1, the
  strict-\(D>0.01\) S2 package, and S4 are not part of this Zenodo version.
- Supplementary Archive S1: complete MPC candidate-level evidence supplied
  with the journal submission.
- Supplementary Data S2: submission-supplied strict-\(D>0.01\) 12-cell
  action-weight HDF5 package, schema, manifest, and checksums used for Table II.
- Supplementary Data S3: frozen ATC-50 lookup table and checksums from Zenodo
  version 1.0.0, together with the schema and portable event-query
  implementation. No byte-for-byte regeneration claim is made for the
  historical table artifact.
- Supplementary Archive S4: landing-metric records and validation, the primary
  10,800-case phase-aware three-expert confirmation, historical randomized
  batches, the direct-argmax 2,160-case diagnostic, and the frozen
  hold-versus-requery control. A compact copy is tracked under
  `supplementary/S4/` and is not part of Zenodo version 1.0.0.

## Release verification

```bash
python -m unittest tests.test_five_batch_phase_aware_release tests.test_s4_release -v
python -m unittest discover -s tests -v
python src/paper2/push_off_grid/validate_release.py
python src/paper2/mpc/validate_release.py
python analysis/synchronize_landing_metric_outputs.py
python analysis/reproduce_paper2_tables.py
python supplementary/S4/landing_metric/code/recompute_landing_metrics.py
python supplementary/S4/fourlink_additional_batches/code/analyze_validity.py
python supplementary/S4/hold_vs_requery/analyze_results.py
python supplementary/S4/build_manifest.py
python tools/build_manifest.py
python tools/verify_manifest.py
git diff --check
```

## License

Author-generated code and data are distributed under Apache-2.0. See
`LICENSE`, `DATA_LICENSE.md`, and `NOTICE`. Third-party components remain
subject to their original terms.
