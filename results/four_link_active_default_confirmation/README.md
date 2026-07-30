# Four-link Active-default confidence confirmation

This directory contains the held-out evaluation used for the manuscript's
Active-default three-expert confirmation.  The controller queries the frozen
three-class gate once at transition start.  Active PPO is used unless the gate
probability for the non-negative expert reaches 0.86 or the probability for
the non-positive expert reaches 0.54; the selected expert is held until
termination.  The thresholds were selected on development data.

The evaluation uses seeds `20261301`, `20261302`, and `20261303`, 40 trials per
command-grid cell and direction, and 2,160 matched trials per controller.

Publication values:

- Active PPO: 1,061/2,160 successes and 529 valid-`Cmt` trials.
- Active-default selector: 1,063/2,160 successes and 552 valid-`Cmt` trials.
- Selector-minus-Active success difference: +0.093 percentage points
  (stratified-bootstrap 95% CI, -0.602 to +0.787).
- Common-valid pairs: 509.
- Mean `Cmt` on the common-valid pairs: 1.504244 for Active PPO and 0.668832
  for the selector.
- Selector-minus-Active paired mean difference: -0.835412
  (stratified-bootstrap 95% CI, -1.064327 to -0.631652).

Files:

- `matched_trials.csv.gz`: all 4,320 controller-level records.
- `per_seed_summary.csv`: compact audit by evaluation seed.
- `publication_summary.json`: manuscript-facing pooled statistics.

Regenerate the summary:

```bash
python analysis/four_link_active_default_confidence_statistics.py
```

Run a fresh evaluation in a new output directory:

```bash
python src/four_link/evaluation/run_active_default_confidence_confirmation.py \
  --output-dir results/_scratch/active_default_confirmation
```

The frozen checkpoints are identified by SHA-256 in the generated evaluation
manifest and by the repository manifest.
