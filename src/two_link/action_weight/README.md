# Table II evaluator and regeneration entry points

The 12 condition-specific `Whole_energy_comparison_low_dim.py` files are the
fixed-seed evaluators associated with Table II. They
contain the negative-expert output correction, complete expert-fusion branch,
direct `D_save` capture, strict `D > 0.01 m` eligibility rule, and RNG seed
`20260716`.

Each evaluator resolves its four case-specific checkpoints from the directory
given by `ACTION_WEIGHT_MODEL_DIR`; the default is a `models` subdirectory
beside that evaluator. The environment variable must identify one case only:
checkpoint directories from different cases must not be flattened or shared.
The exact filenames, byte sizes, and SHA-256 values are listed in
`configs/action_weight_checkpoint_manifest.json`.

Generated HDF5 files are written to `ACTION_WEIGHT_OUTPUT_DIR`, which defaults
to a `generated` subdirectory beside the evaluator. Use a separate output
directory for each case because the output basenames repeat. For example, in
PowerShell:

```powershell
$env:ACTION_WEIGHT_MODEL_DIR = "X:\action_weight_checkpoints\case_name"
$env:ACTION_WEIGHT_OUTPUT_DIR = "X:\table_ii_regeneration\case_name"
python .\Whole_energy_comparison_low_dim.py
```

The fixed-checkpoint evaluators require the case checkpoint bundle in addition
to the repository. Independent numerical verification of the reported table
starts directly from the released HDF5 arrays in Supplementary Data S2:

```bash
python analysis/recompute_action_weight_table_ii.py --data-root /path/to/S2
```

That program validates the consolidated HDF5 schema, strict common mask, and
expert-fusion result, then regenerates the manuscript CSV. S2 also contains the
logical source-file manifest and the status-common sensitivity summaries.
