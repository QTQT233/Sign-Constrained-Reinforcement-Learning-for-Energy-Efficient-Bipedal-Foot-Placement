# Phase-aware 2,160-case Figure 7 record

This directory contains the phase-aware three-expert reanalysis used for the
three-expert entries in manuscript Figure 7. The reanalysis applies the same
reset-phase confidence thresholds as the five-batch primary protocol to the
already executed Active PPO, non-negative, and non-positive source rollouts in
`../four_link_three_expert_primary_2160/`; it does not rerun the dynamics.

Reproduce the record from the repository root with:

```text
python analysis/reconstruct_phase_aware_primary_2160.py
```

Outputs:

- `route_assignments.csv`: route, gate probabilities, and selected outcome for
  every matched case;
- `controller_summary.json`: success, valid-\(C_{mt}\), and actuator-resolved
  mean values for the phase-aware three-expert selector;
- `active_pair_summary.json`: Active--three-expert common-valid comparison;
- `CHECKSUMS.sha256`: file-level integrity inventory.

The other Figure 7 controller values remain in
`../four_link_three_expert_primary_2160/controller_summary.csv`,
`active_reference_summary.csv`, and `paired_summary.csv`.
