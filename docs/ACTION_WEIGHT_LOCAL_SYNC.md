# Local action-weight source and result sync

The 12 canonical GitHub evaluators are byte-matched copies of the official
local files at

```text
Energy_Comparison/action_weight={0.02,0.04,0.06}/<condition>/
Whole_energy_comparison_low_dim.py
```

Files named `Whole_energy_comparison_low_dim_optimized.py` and historical
`- 副本.py` variants are not canonical Table II sources. The official files
contain two deliberate reproducibility changes:

1. the negative-expert HDF5 file receives `Cmt_save0_1`; and
2. Python, NumPy, PyTorch, and CUDA evaluation generators use seed `20260716`.

The existing complete Cmt fusion branch is retained unchanged. It compares
negative and positive expert Cmt only when both expert statuses are not `-2`,
and otherwise selects the successful expert.

After a local rerun, regenerate and validate the repository outputs with:

```bash
python analysis/recompute_action_weight_table_ii.py \
  --data-root /path/to/Energy_Comparison
```

The JSON manifest written under `results/` binds each of the 12 public scripts
and 120 local HDF5 inputs to a SHA-256 digest. This is the authoritative
source-to-result link for Table II.
