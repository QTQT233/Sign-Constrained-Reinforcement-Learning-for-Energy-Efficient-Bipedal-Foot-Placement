# Sign-Constrained Reinforcement Learning for Bipedal Foot Placement

Reproducibility materials for *Mechanical-Work-Aware Routing of One-Sided PPO
Experts for Planar Bipedal Foot Placement* by Qi Tao, Yiyou Liu, Amir Degani,
and Mingyi Liu.

This `main` branch contains the current manuscript-facing simulation code,
repository-hosted model checkpoints, numerical outputs, provenance records, validators,
and checksums. Superseded public artifacts were removed from the submission
branch and preserved in the authors' offline `Paper_store/Past` archive.

## Evidence map

- `src/two_link/`: two-link routing, action-weight evaluation, mechanical-cost,
  and portable transition-locked and ATC-50 lookup implementations. Table I is
  regenerated from the released 30^4 HDF5 maps by `analysis/reproduce_table1.py`;
  the older 10x30 map-generator snapshots are not part of this release.
- `src/four_link/`: V22_3/V9 four-link training and the shared 2,160-case
  evaluator, including the scratch-trained true per-step hard-action-mask PPO.
- `src/paper2/mpc/`: repository-relative continuous-torque MPC implementation,
  accepted 12-case results, and validator.
- `src/paper2/push_off_grid/`: validator for the complete learned-controller
  recovery-grid record and its selected fixed-parameter replays.
- `data/four_link/true_action_mask_scratch_c090_epoch1275/`: the current
  controller-level and trial-level four-link archive used for Tables VI–VII and
  Appendix B.
- `results/paper2_current/`: the current 72-row two-link comparison and the
  manuscript-facing Table IV–V summaries.
- `results/paper2_mpc_unified_12case/`: accepted MPC cases, validation records,
  and deterministic replay checks.
- `results/paper2_push_off_grid_12case/`: 518,400 candidate records, 36 selected
  minima, fixed-parameter replay values, provenance, and scoped checksums.
- `results/four_link_statistics/true_action_mask_scratch_c090_epoch1275/`:
  paired fixed-checkpoint inference for the true-mask comparison.
- `analysis/`: read-only numerical analysis and table-regeneration programs.
- `docs/`: protocol, metric, provenance, availability, and submission records.

The repository does not publish manuscript figure-rendering scripts or image
assets because they are not part of the numerical reproducibility record. All
numerical values plotted in the manuscript remain traceable to the current CSV
and JSON records above. The plotting sources and submitted image assets are
preserved in the authors' offline `Paper_store/New` submission backup, while
earlier figure candidates are preserved under `Paper_store/Past`.

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

## Four-link command-grid comparison

The current archive evaluates six fixed controller records on the shared
V22_3/V9 command grid. The manuscript's five principal rows are active PPO,
active-full, the scratch-trained true per-step hard mask, the online
transition-persistent selector, and the offline oracle envelope. Each controller
has 2,160 matched trials.

Key values are:

- active PPO: 975/2,160 successes (45.1%) and 458 valid-`Cmt` trials;
- true hard mask: 946/2,160 successes (43.8%) and 430 valid-`Cmt` trials;
- online selector: 964/2,160 successes (44.6%) and 480 valid-`Cmt` trials;
- active versus selector: 419 both-valid pairs, mean `Cmt` 1.417 versus 0.371;
- hard mask versus selector: 395 both-valid pairs, mean `Cmt` 1.303 versus
  0.278, with the selector lower in 354/395 pairs.

The true-mask and selector policies share the formal evaluator and principal
reward coefficients. Their architecture, initialization, soft thresholds, and
training structure are not identical. The comparison is therefore reported as
a method-level ablation: under the evaluated configuration, transition-level
sign persistence is the component associated with the lower conditional `Cmt`.

Regenerate the paired statistics:

```bash
python analysis/four_link_true_action_mask_paired_inference.py
```

Verify the current archive and source/checkpoint hashes:

```bash
python -m unittest tests.test_true_action_mask_release -v
```

## Two-link 12-case comparison

The 12 matched multi-step cases use the current fixed-parameter replay records.
Mean `Cmt` is 0.122346306 for the proposed controller, 0.226620152 for discrete
active PPO, 0.238107711 for continuous-torque PPO, and 0.198945099 for
continuous-torque MPC. The corresponding ratio-of-means reductions are 46.0%,
48.6%, and 38.5%.

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
python analysis/reproduce_paper2_tables.py
```

Validate the accepted MPC release:

```bash
python src/paper2/mpc/validate_release.py
python -m unittest tests.test_paper2_mpc_unified -v
```

The full candidate-level MPC search, solver traces, and two fresh-process
replays are prepared separately as Supplementary Archive S1.

## Action-weight evaluation

The 12 current `Whole_energy_comparison_low_dim.py` source snapshots use RNG
seed `20260716`, write the negative expert's `Cmt` to the negative-expert array,
and accumulate continuous-controller positive commanded work from applied
physical torque once per step. A value is eligible only when center-of-mass
displacement is greater than 0.01 m. The fusion rule remains complete: if both
experts succeed, the lower finite `Cmt` is used; if only one succeeds, that
expert is used. These scripts preserve the workstation paths used for the
completed runs and are explicitly provenance-only; see
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
  manifests, checksums, and validators. Cite the exact `main` commit used for
  submission.
- Zenodo DOI [10.5281/zenodo.21407986](https://doi.org/10.5281/zenodo.21407986):
  version-1.0.0 data/model artifacts and supplementary hardware videos.
- Supplementary Archive S1: complete MPC candidate-level evidence.
- Supplementary Data S2: 12-cell action-weight HDF5 package and schema.
- Supplementary Data S3: frozen ATC-50 lookup table, schema, portable event-
  query implementation, and checksums. No byte-for-byte regeneration claim is
  made for the historical table artifact.

A separate GitHub Release is optional when the manuscript cites the immutable
merged `main` commit. The hardware payload has not changed, so this code update
does not require a new Zenodo version.

## Release verification

```bash
python -m unittest discover -s tests -v
python src/paper2/push_off_grid/validate_release.py
python src/paper2/mpc/validate_release.py
python analysis/reproduce_paper2_tables.py
python tools/build_manifest.py
python tools/verify_manifest.py
git diff --check
```

## License

Author-generated code and data are distributed under Apache-2.0. See
`LICENSE`, `DATA_LICENSE.md`, and `NOTICE`. Third-party components remain
subject to their original terms.
