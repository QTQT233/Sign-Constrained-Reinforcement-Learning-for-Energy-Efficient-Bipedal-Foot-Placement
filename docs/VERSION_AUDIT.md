# Four-link result-version audit

## Manuscript-primary V22_3/V9 output (5 July 2026)

Directory archived as `data/four_link/legacy_manuscript/`:

- active PPO: 975/2160 = 45.1389%;
- online selector: 964/2160 = 44.6296%;
- selector-active paired difference: -0.509 percentage points;
- both-valid active/selector pairs: 419.

All 19 trial-level and summary CSV files are retained. The two controllers have
identical 2160-case keys and initial-state hashes. Tables VII-XI and the paired
inference in the manuscript use this artifact set exclusively.

The exact extended-evaluator source bytes that generated the 5 July directory
were not retained. The repository therefore distinguishes two claims:

1. the manuscript statistics are exactly reproducible from the retained
   trial-level outputs; and
2. the file named `...legacy_reconstructed.py` is an output-validation
   reconstruction, not the missing original source.

The reconstruction restores the V22_3 touchdown lower bounds
`[35, 8, -115, 5] deg`, legacy evaluation label, V22_3 checkpoint paths, and
2160-case grid in the surviving extended evaluator. On 11 July 2026, a full
rerun reproduced all 19 retained CSVs byte-for-byte. The validation record
stores both archived and rerun SHA-256 hashes. This supports end-to-end output
reproduction for the reconstructed program while preserving the distinction
between that reconstruction and the unavailable historical source bytes.

## Non-manuscript personal protocol

Later experiments that relaxed the touchdown knee lower bounds to 0 deg are
personal diagnostics and are not used in the manuscript. Their results, CIs,
failure counts, and both-valid denominator must not be mixed with the 5 July
artifact set.

The separately named `paper_four_link_training_domain_cmt_success_v23.py` also
used a distinct reset/evaluation protocol and a default 1080 shared cases. It is
not the source of the manuscript's 2160-case tables.

## Checkpoint-version finding

Both surviving evaluators and the retained CSV policy labels point to V22_3
checkpoints, including `UniPos c097` and `UniNeg c100`. The V23 uni-positive and
uni-negative training files were created on 8 July, after the manuscript-primary
output was written, so they cannot have generated the 5 July Oracle envelope.

## Release rule

Every table row must resolve to one immutable tuple:

`protocol + source/reconstruction SHA-256 + config + checkpoint SHA-256 + raw-output SHA-256 + analysis-script SHA-256`.

Never replace manuscript data with a later personal protocol merely because its
source history is cleaner. Keep the evidence version and its limitations explicit.
