# Scratch-trained true per-step hard-action-mask PPO

## Controller definition

At each simulation step, the sign head selects the admissible nonzero hip-
torque sign. The mask retains nine zero-hip actions and nine actions with the
selected sign, giving 18 valid actions from the 27-action physical grid. The
masked action head then selects the applied hip/knee torque triple.

The checkpoint was trained from random initialization. The scratch entrypoint
contains no `torch.load` call, does not invoke the inherited standalone trainer,
explicitly sets `base.load_previous_model = False`, and replaces the inherited
checkpoint loader with a function that raises an error. The fixed checkpoint is
the c090 epoch-1275 policy, selected before the 2,160-case formal evaluation.
The associated training log contains epochs 1–1307 and records
`initialization=scratch` throughout.

## Alignment and differences

The true mask and fixed-sign experts use:

- `action_value_weight = 0.026`;
- hip and knee torque limits of 4.0 and 0.2 N m;
- the same principal task and safety reward coefficients; and
- the shared V22_3/V9 formal evaluator and touchdown gate.

The true mask uses one hierarchical actor, whereas the proposed controller uses
two fixed-sign experts plus a transition-start selector. Their soft clearance
thresholds, architecture, initialization lineage, and training structure are
not identical. The comparison is therefore a method-level ablation of sign
masking versus transition-level persistence, not a strict one-factor causal
experiment.

## Formal archive

Current files are under
`data/four_link/true_action_mask_scratch_c090_epoch1275/`. The controller
summary contains 2,160 trials for each of:

- active PPO;
- active-full;
- per-step sign encoding;
- scratch true hard mask;
- online sign selector; and
- offline oracle envelope.

The manuscript principal rows exclude the historical per-step encoding because
it exposes the full 27-action physical union and is not a hard mask.

Key true-mask comparisons are:

- true mask: 946 successes and 430 valid-`Cmt` trials;
- online selector: 964 successes and 480 valid-`Cmt` trials;
- selector-minus-mask success difference: +0.83 percentage points
  (95% CI −0.26 to 1.93; exact McNemar p = 0.159);
- 395 both-valid mask-selector pairs;
- mean `Cmt`: 1.302962 for the mask and 0.278273 for the selector; and
- selector lower in 354/395 pairs (89.6%).

Under the evaluated reward and training configuration, this ablation identifies
transition-level sign persistence, rather than sign masking alone, as the
component associated with the lower conditional `Cmt`.

## Reproduction

Check source and checkpoint availability without a full rollout:

```bash
python src/four_link/evaluation/paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py --check-only
```

Run the evaluator only with a new output directory:

```bash
python src/four_link/evaluation/paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py --output-dir /new/path
```

Regenerate paired statistics:

```bash
python analysis/four_link_true_action_mask_paired_inference.py
```

Validate source, checkpoint, training-log, trial, and statistical hashes:

```bash
python -m unittest tests.test_true_action_mask_release -v
```

Manuscript figure-rendering scripts and image assets are not published in this
repository. Their final versions are stored in the authors' offline submission
backup; manuscript numerical values derive from the current CSV and JSON
records above.
