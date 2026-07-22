# Frozen true per-step action-mask baseline

## Evidence role

This release adds a genuine per-step action-mask controller (`AM-step1`). At
every simulation step, one learned binary head selects the hip-torque sign. A
second head is then evaluated after masking every nonzero-hip action with the
opposite sign. Zero-hip actions remain valid, so each selected sign admits 18
of the 27 physical torque triples. PPO uses the joint selector and masked-action
log probability during both rollout and update.

This controller replaces the old `per_step_sign` label as the actual action-
mask baseline. The old policy remains an action-index encoding diagnostic: it
exposes all 27 physical actions and must not be relabeled as action masking.

`AM-step1` is not the same intervention as transition-persistent masking
(`AM-P1`). The proposed controller chooses between two fixed-sign experts at
transition onset and retains that choice to termination, whereas `AM-step1`
may select a new sign at every simulation step. Their comparison is therefore
a method-level baseline, not a single-factor causal estimate of persistence.

## Frozen checkpoint

The primary checkpoint is the scratch-trained c090 training-best policy from
epoch 1275. It was frozen before the formal 2,160-case output was inspected.
The retained training snapshot extends through epoch 1307 and verifies
`initialization=scratch`, `checkpoint_source_kind=random_initialization`, a
27-action physical table, 18 valid actions after each sign selection, and
`action_value_weight=0.026`.

The training log records `script=pydevconsole.py`; no fixed training seed or
direct-script launch is claimed. Consequently, this is one retained training
run rather than a training-seed replication study.

Exact paths and SHA-256 values are recorded in
`configs/four_link_true_action_mask_scratch_c090_epoch1275.json`.

## Relationship to the fixed-sign experts

The action-mask run and fixed-sign experts use the same actuator limits,
`action_value_weight=0.026`, entropy coefficient, scalar reward coefficients,
and V22_3 hard success/termination definitions. They are not fully matched
training protocols:

- the action-mask q3 lower soft margin is 14 degrees rather than 18 degrees;
- its clearance soft margin/approach width are 0.100/0.130 m rather than
  0.090/0.115 m;
- it uses one shared hierarchical actor initialized from scratch, whereas the
  proposed controller comprises two separately trained fixed-sign experts and
  a transition-onset selector.

These differences must remain visible when interpreting the comparison.

## Formal evaluation

The public evaluator preserves the archived V22_3 touchdown lower bounds
`[35, 8, -115, 5]` degrees. It uses three evaluation seeds, both walking
directions, a 3 x 3 step-length/height command grid, and 40 cases per
seed/direction/grid cell, giving 2,160 shared cases per controller. Policies
are evaluated deterministically.

The complete non-smoke output is under
`data/four_link/true_action_mask_scratch_c090_epoch1275/`. Its manifest records
the evaluator and model hashes; its local `CHECKSUMS.sha256` covers every CSV
and the manifest.

The retained CSVs were generated with the same self-contained rollout body and
the listed V22_3 runtime gate. The public evaluator subsequently replaced
machine-local path/output wiring and added validation and release metadata.
The release therefore calls it a portable source-aligned evaluator and does
not claim that these CSV bytes were emitted from the exact final public source
bytes. A fresh output should be compared file-by-file before making that
stronger claim.

The retained controller totals are:

| Controller | Success | Valid Cmt | Success rate | Valid-Cmt rate |
|---|---:|---:|---:|---:|
| True action mask, scratch | 946/2160 | 430/2160 | 43.80% | 19.91% |
| Proposed online selector | 964/2160 | 480/2160 | 44.63% | 22.22% |

On the 395 both-valid paired cases, mean Cmt was 1.30296 for the action-mask
controller and 0.27827 for the online selector. The selector had lower Cmt in
354/395 cases (89.62%). The paired success-rate difference, selector minus
action mask, was 0.83 percentage points with a 95% paired Wald interval of
-0.26 to 1.93 percentage points and an exact McNemar p-value of 0.159. These
case-level statistics do not estimate variation across independently trained
models.

For the active-versus-mask diagnostic, both controllers were valid in 404
cases. Their paired-set mean Cmt values were 1.21902 and 1.33084, respectively;
the mask was lower in 146/404 cases (36.14%). This comparison does not support a
claim that generic per-step hard masking lowers Cmt.

The scalar reward coefficients and PPO scalar hyperparameters match the fixed
experts. The scratch action mask nevertheless uses q3/clearance soft thresholds
of 14 degrees, 0.100 m, and 0.130 m rather than 18 degrees, 0.090 m, and 0.115 m;
it also uses a shared 27-action hierarchical actor, random scratch
initialization, the per-step curriculum branch, and a joint sign/action PPO
probability. These are material training differences, not merely notation.

## Reproduction and validation

Verify all frozen policy hashes without executing rollouts:

```bash
python src/four_link/evaluation/paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py --check-only
```

Run a fresh evaluation into the ignored scratch area:

```bash
python src/four_link/evaluation/paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py
```

The evaluator refuses to overwrite a non-empty output directory. Use
`--output-dir PATH` only for a new directory. Regenerate the paired statistics:

```bash
python analysis/four_link_true_action_mask_paired_inference.py
```

Regenerate the two manuscript figures and their source-data exports:

```bash
python analysis/plot_figure9_true_action_mask_case.py
python analysis/plot_figure10_true_action_mask.py
```

Figure 9 is a post hoc illustrative case selected after the aggregate run. The
selection rule and exact selected key are machine-readable in
`results/figures/true_action_mask/figure9_true_action_mask_case_selection.csv`;
the figure must not be used for population inference. Figure 10 reports the
fixed-checkpoint aggregate and paired results. Neither figure is a strict
one-factor causal test.

Validate the release files and all existing reference artifacts:

```bash
python -m unittest tests.test_true_action_mask_release -v
python -m unittest discover -s tests -v
```

The 18-case smoke directories used during development are deliberately absent
from this release and must not be used in the manuscript.
