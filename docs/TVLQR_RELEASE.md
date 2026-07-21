# TVLQR release status

All 12 `LQR.py` entry points and all four checkpoints they load are included in
this repository. Eleven entry points retain the hashes from the initial
read-only audit. The raised-1.145-r1 source was synchronized to the accepted
local source: `init_idx=(19, 11, 15, 1)`, the `1.00/0.89` recovery pair, and the
terminal stance action included in positive work. Its corrected SHA-256 is
pinned explicitly in `configs/tvlqr_release.json`. The portable runner verifies
all source and model hashes, rewrites only the four author-workstation model
paths in a temporary copy, and captures stdout/stderr outside the source tree.

```bash
python -m pip install -r environment/requirements-tvlqr.txt
python tools/run_frozen_tvlqr.py --check-only
python tools/run_frozen_tvlqr.py --case flat_1.145_r1 --seed 0
python tools/run_frozen_tvlqr.py --all --seed 0 --timeout 60 \
  --output results/tvlqr_seed0
```

The initial automated audit retained 11 completed cases and one 180-s timeout;
that timeout is an audit outcome, not a missing source file. The runner uses a
configurable per-case timeout and applies an explicit seed only to a temporary
copy. Under the corrected release protocol (`seed=0`, 600-s per-case timeout),
all 12 cases completed. The reproduced overall Cmt is 0.234644 (sample SD
0.019875; descriptive 95% t interval [0.222016, 0.247272]). The corrected
raised-1.145-r1 value is 0.278501. Per-case logs, the case table, grouped
summary, and run manifest are retained under `results/tvlqr_seed0/`.

The manuscript may retain TVLQR as a local model-based reference and may state
that its reproduced Cmt values are higher than the proposed method in the 12
tested conditions. It should not describe the archived unseeded programs as
intrinsically deterministic or use the earlier timeout as evidence of algorithm
failure. The table and caption should state the Python/package versions, seed
mode, timeout, 12/12 completion status, and the aggregation rule.
