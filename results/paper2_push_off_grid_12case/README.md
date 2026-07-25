# Validated 12-case recovery-grid records

This directory contains the complete candidate tables and accepted
fixed-parameter replays for the three learned two-link controllers.

- `candidates/`: 36 CSV files with 14,400 rows each (518,400 rows total).
- `selected_minima.csv`: the selected eligible minimum from each candidate
  table.
- `accepted_cases.csv`: the fixed-parameter replay metrics for the 36 selected
  pairs.
- `controller_summary.csv`: descriptive controller-level summaries.
- `case_pairwise_cmt_summary.csv`: matched-case comparisons.
- `validation_summary.json`: grid, replay, boundary, and aggregate checks.
- `provenance.json`: repository-relative candidate and entry-point hashes.
- `CHECKSUMS.sha256`: SHA-256 hashes for this result directory.

Controller labels in the source records map as follows: `passive` is the
proposed transition-locked one-sided selector, `discrete` is unrestricted
discrete active PPO, and `continuous` is continuous-torque PPO.

The two parameters are post-impact recovery-velocity decrements in `rad/s`;
they are not push-off forces. All reported optima are minima on the prescribed
`0.00:0.01:1.19 rad/s` grid.
