# Five additional randomized four-link batches

This component evaluates the frozen V22_3/V9 active PPO controller against the
frozen transition-start selector over five additional 2,160-case batches. The
15 evaluation seed streams yield 10,800 matched trials. The primary 2,160-case
benchmark is retained separately and is not pooled into these results.

The runner references the canonical evaluator and checkpoints already stored
under `src/four_link/`, `models/four_link/`, and `data/four_link/`. Their
expected SHA-256 values are pinned in the runner and configuration. No model or
evaluator copies are duplicated in S4.

## Reanalyze the published records

From the repository root:

```bash
python supplementary/S4/fourlink_additional_batches/code/analyze_validity.py
python supplementary/S4/fourlink_additional_batches/code/paired_discordance.py
```

The first command reads the published 10,800-row `matched_records.csv` and
writes recomputed summaries below `_generated/`. It uses 4,000 stratified
bootstrap replicates with seed `2026072504`.

## Regenerate a batch

Install `environment/requirements-four-link.txt`, then run:

```bash
python supplementary/S4/fourlink_additional_batches/code/run_batches.py \
  --batch batch01
```

Available fresh batches are `batch01` through `batch05`. Generated raw evaluator
outputs are written below `_generated/` and are not tracked.

## Principal results

- Active PPO success: 5,143/10,800 (47.620%).
- Selector success: 4,996/10,800 (46.259%).
- Both-valid pairs: 2,244/10,800 (20.778%).
- Both-valid mean `Cmt`: 1.527645 for active PPO and 0.447379 for the selector.
- Paired selector-minus-active mean `Cmt`: -1.080266
  (95% CI -1.189121 to -0.975760).
- The selector had lower `Cmt` in 81.818% of both-valid pairs
  (95% CI 80.168% to 83.364%).

Removing only the nonzero-torque requirement changed no record in these
10,800 cases. The `Cmt` comparison remains conditional on both controllers
producing valid values.
