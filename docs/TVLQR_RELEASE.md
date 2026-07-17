# TVLQR release status

All 12 frozen `LQR.py` entry points and all four checkpoints they load are
included in this repository. The original files are unmodified. The portable
runner verifies their SHA-256 values, rewrites only the four author-workstation
model paths in a temporary copy, and captures stdout/stderr outside the source
tree.

```bash
python -m pip install -r environment/requirements-tvlqr.txt
python tools/run_frozen_tvlqr.py --check-only
python tools/run_frozen_tvlqr.py --case flat_1.145_r1 --seed 0
python tools/run_frozen_tvlqr.py --all --seed 0 --timeout 60 \
  --output results/tvlqr_seed0
```

The author reports that the TVLQR programs complete in the local interactive
environment. The automated audit retained 11 completed cases and one 180-s
timeout; that timeout is an audit outcome, not a missing source file. The
released runner therefore uses a configurable per-case timeout and applies an
explicit seed only to a temporary copy. Under the release protocol (`seed=0`,
60-s per-case timeout), all 12 cases completed in 25.7 s. The reproduced overall
Cmt was 0.232348 (sample SD 0.015446; 95% t interval
[0.222534, 0.242162]); per-case logs, the case table, grouped summary, and run
manifest are retained under `results/tvlqr_seed0/`.

The manuscript may retain TVLQR as a local model-based reference and may state
that its reproduced Cmt values are higher than the proposed method in the 12
tested conditions. It should not describe the archived unseeded programs as
intrinsically deterministic or use the earlier timeout as evidence of algorithm
failure. The table and caption should state the Python/package versions, seed
mode, timeout, 12/12 completion status, and the aggregation rule.
