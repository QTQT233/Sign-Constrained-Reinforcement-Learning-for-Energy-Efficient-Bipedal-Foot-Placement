# Table II evaluator source boundary

The 12 condition-specific `Whole_energy_comparison_low_dim.py` files are the
current fixed-seed evaluator source snapshots associated with Table II. They
contain the negative-expert output correction, complete expert-fusion branch,
direct `D_save` capture, strict `D > 0.01 m` eligibility rule, and RNG seed
`20260716`.

These files preserve the workstation paths used for the completed evaluations.
They are provenance snapshots, not portable entry points: a fresh execution
requires the case-specific policy checkpoints and input/output directories
named by each source file. The public repository does not imply that those
absolute paths exist on another machine.

Portable verification of the reported values starts from Supplementary Data S2:

```bash
python analysis/recompute_action_weight_table_ii.py --data-root /path/to/S2
```

That program validates the consolidated HDF5 schema, strict common mask, and
expert-fusion result, then regenerates the manuscript CSV. S2 also contains the
logical source-file manifest and the status-common sensitivity summaries.
