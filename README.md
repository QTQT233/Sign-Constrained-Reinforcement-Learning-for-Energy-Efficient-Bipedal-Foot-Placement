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
- `src/four_link/`: V22_3/V9 four-link training and the shared 2,160-case
  evaluator, including the scratch-trained true per-step hard-action-mask PPO.
- `src/paper2/mpc/`: repository-relative continuous-torque MPC implementation,
  accepted 12-case results, and validator.
- `data/four_link/true_action_mask_scratch_c090_epoch1275/`: the current
  controller-level and trial-level four-link archive used for Tables VI–VII and
  Appendix B.
- `results/paper2_current/`: the current 72-row two-link comparison and the
  manuscript-facing Table IV–V summaries.
- `results/two_link_active_fallback_fixed12/`: the fixed 12-case
  Active-default routing record, including the unchanged push-off pairs.
- `results/paper2_mpc_unified_12case/`: accepted MPC cases, validation records,
  and deterministic replay checks.
- `results/four_link_statistics/true_action_mask_scratch_c090_epoch1275/`:
  paired fixed-checkpoint inference for the true-mask comparison.
- `results/four_link_active_default_confirmation/`: three held-out seed
  streams, pooled matched trials, and regenerated Active-default confidence
  selector statistics.
- `analysis/`: read-only numerical analysis and table-regeneration programs.
- `docs/`: protocol, metric, provenance, availability, and submission records.

Manuscript figure-rendering scripts and image assets are not part of this
numerical reproducibility repository. All numerical values plotted in the
manuscript remain traceable to the CSV and JSON records above.

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

## Four-link command-grid evidence

The primary two-expert archive evaluates six fixed controller records on the shared
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

A separate Active-default confidence selector retains the same three frozen
low-level experts and queries the learned three-class gate once per transition.
Active PPO is the default; the non-negative and non-positive experts are
released at gate probabilities 0.86 and 0.54, respectively. These thresholds
were selected on development data. On three previously unused evaluation seeds
(2,160 matched trials per controller), Active PPO and the selector achieved
1,061 and 1,063 successes. Among 509 common-valid pairs, mean `Cmt` was
1.504244 and 0.668832, respectively; the paired selector-minus-Active
difference was -0.835412 (stratified-bootstrap 95% CI, -1.064327 to
-0.631652).

Regenerate the confidence-selector statistics:

```bash
python analysis/four_link_active_default_confidence_statistics.py
```

Run a fresh confirmation:

```bash
python src/four_link/evaluation/run_active_default_confidence_confirmation.py \
  --output-dir results/_scratch/active_default_confirmation
```

The released selector was trained from 17,524 labels and validated on 2,880
labels drawn from 75,600 candidate transitions. Label generation is
success-first; when both one-sided experts succeed, valid `Cmt` requires at
least one nonzero-torque step and signed COM displacement greater than 0.006 m,
and differences below 0.01 are omitted as ties. The released checkpoint has
86.5% validation accuracy. Formal controller evaluation separately uses the
shared 0.001 m valid-`Cmt` threshold.

The active-versus-selector displacement-threshold sensitivity table is
regenerated by:

```bash
python analysis/four_link_paired_inference.py \
  --paired data/four_link/true_action_mask_scratch_c090_epoch1275/paired_trials.csv \
  --terminal data/four_link/true_action_mask_scratch_c090_epoch1275/terminal_reason_summary.csv \
  --output results/four_link_statistics/active_vs_selector
```

Regenerate the paired statistics:

```bash
python analysis/four_link_true_action_mask_paired_inference.py
```

Verify the current archive and source/checkpoint hashes:

```bash
python -m unittest tests.test_true_action_mask_release -v
```

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
- Supplementary Archive S1: complete MPC candidate-level evidence.
- Supplementary Data S2: 12-cell action-weight HDF5 package and schema.
- Supplementary Data S3: frozen ATC-50 lookup table, schema, portable event-
  query implementation, and checksums. No byte-for-byte regeneration claim is
  made for the historical table artifact.

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
