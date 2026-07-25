# Frozen hold-versus-requery comparison

This single-factor control changes only the frozen selector query schedule.
`hold` queries the selector at transition onset and retains the selected
one-sided expert. `requery` applies the same selector before every control
action and switches expert only when the selected sign changes.

The runner references the canonical evaluator, model checkpoints, and primary
2,160-case inputs already present in the repository. Their SHA-256 values are
pinned in `run_hold_vs_requery.py`; no duplicate model or evaluator files are
stored in S4.

Install `environment/requirements-four-link.txt`, then run from the repository
root:

```bash
python supplementary/S4/hold_vs_requery/run_hold_vs_requery.py --stage smoke
python supplementary/S4/hold_vs_requery/run_hold_vs_requery.py --stage formal
python supplementary/S4/hold_vs_requery/analyze_results.py
```

The smoke hold replay must reproduce the released selector records before the
formal stage runs. Generated rollouts are written below `outputs/`; the compact
published trial and inference records remain under `results/`.

## Principal results

- Hold success: 964/2,160 (44.63%).
- Requery success: 893/2,160 (41.34%).
- Hold-minus-requery success difference: +3.29 percentage points
  (95% CI +2.41 to +4.17; exact McNemar `p = 5.39e-13`).
- Hold valid-`Cmt`: 480/2,160 (22.22%).
- Requery valid-`Cmt`: 434/2,160 (20.09%).
- Among 419 both-valid pairs, mean `Cmt` was 0.18761 for hold and 0.23221 for
  requery. The hold-minus-requery interval included zero.

Accordingly, this control supports a success-rate benefit of transition-level
persistence in the frozen evaluation. It does not establish that persistence
reduces conditional `Cmt`. These 419 both-valid hold/requery pairs are distinct
from the 419 both-valid active-PPO/selector pairs in the primary benchmark.
