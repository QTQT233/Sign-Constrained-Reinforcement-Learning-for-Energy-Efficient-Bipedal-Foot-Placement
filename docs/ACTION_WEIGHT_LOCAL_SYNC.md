# Local action-weight sync manifest

The public patch contains 12 canonical evaluators, one optimized evaluator,
the shared Cmt/fusion module, regression tests, and two read-only 12-cell audit
tables.  The historical local tree additionally contains three files named
`Whole_energy_comparison_low_dim - 副本.py`.  Those copies are not published,
but they must receive the same prospective code correction.

Run the mechanical sync from an account that can write the local project:

```powershell
powershell -ExecutionPolicy Bypass -File tools\sync_action_weight_patch_to_local.ps1 `
  -LocalRoot "D:\L&S\Mas\Project\Paper1\Energy_Comparison" `
  -PythonExe "D:\L-Environment\Anaconda3\envs\Pytorch\python.exe"
```

The script copies `cmt_metrics.py`, all 12 canonical evaluators, and the
optimized evaluator.  It deliberately does **not** overwrite the three files
matching `Whole_energy_comparison_low_dim - *.py`: those historical variants
have substantive source differences.  Patch each copy in place by adding
`select_successful_expert_by_cmt` to its existing `cmt_metrics` import and by
replacing only the legacy `working_save_passive`/`Cmt_save_passive`
construction loops with:

```python
working_save_passive, Cmt_save_passive = select_successful_expert_by_cmt(
    working_save0_1,
    Cmt_save0_1,
    working_save01,
    Cmt_save01,
    tie_break="positive",
)
```

Do not change any other line in an archival copy.  After those three surgical
edits, the sync script requires one wildcard match per weight, performs byte
hash checks for the 13 public files, verifies both shared functions in all 16
local scripts, and executes a scalar fusion truth table.  It does not open or
rewrite any HDF5 file.
