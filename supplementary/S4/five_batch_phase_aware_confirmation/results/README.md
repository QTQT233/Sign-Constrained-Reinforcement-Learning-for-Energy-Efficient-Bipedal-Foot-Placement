# Frozen five-batch final analysis

This directory was produced by:

```
D:\L-Environment\Anaconda3\envs\Pytorch\python.exe C:\Users\Admin\Documents\Review\four_link_phase_aware_five_batch_20260730\code\analyze_frozen_five_batch_final.py
```

- Matched cases per controller: 10,800
- Bootstrap: 20,000 replicates,
  seed 20263130, stratified by evaluation seed,
  walking direction, commanded horizontal target, and commanded height target.
- Binary paired test: exact two-sided McNemar test.
- All frozen primary acceptance criteria met: **True**

See `RESULTS_SUMMARY.json` for the concise result,
`final_confirmation_report.json` for the complete machine-readable report,
and the `controller_metrics_*` / `paired_metrics_*` CSV files for all
batches, seeds, endpoint cells, and directions. Raw inputs were read only.
