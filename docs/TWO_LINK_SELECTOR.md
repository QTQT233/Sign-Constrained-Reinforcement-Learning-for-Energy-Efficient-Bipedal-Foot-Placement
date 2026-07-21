# Relationship between the two-link offline tables

`working_save_passive-10-30` is the coarse `10 x 30 x 10 x 30` offline result
map obtained from the sign-restricted expert evaluations. Its entries encode
the selected sign/action outcome after considering expert feasibility and, when
both experts are valid, their Cmt values.

In the two-link simulations, this map is queried once at transition onset. It
selects the non-positive or non-negative expert, and the chosen expert remains
active until transition termination. This is the manuscript's **coarse
transition-locked expert routing** implementation.

`working_save-ATC-50` is the `50^4` higher-resolution deployment table that
implements the same offline mapping concept for the hardware controller. It is
not an interpolation of the released coarse array. The author-confirmed
generator follows the same two-expert evaluation/fusion logic used for the
`60,30(0.33m)` / `working_save_passive-10-30` family, with all four state axes
discretized to 50 points. The repository contains the corresponding 50-bin
generator-family source and hardware consumers, while the large generated
table belongs in the journal supplementary archive. The hardware-only Zenodo
record is not the source of this simulation table. The table is not an online comparison
of critic return estimates. Values `-1`, `0`, and `+1` are converted to
negative, zero, and positive hip-torque commands. The `-2` value is the
uncovered/failure sentinel handled by the deployment fallback.

## Deployment update rule

The hardware controller discretizes the measured two-link state and queries
`working_save-ATC-50`. It keeps the selected action between lookup events.
The indices are updated when the stance-link angle changes by about
`1.8 degrees`, after which the refined table is queried again. This explains
how the offline map is connected to the state trajectories in the manuscript
without incorrectly describing the hardware implementation as transition-
locked or as an online return-based selector.

## Manuscript terminology

Use the following names consistently:

- `coarse transition-locked expert routing` for the simulation use of
  `working_save_passive-10-30`;
- `ATC-50 event-updated lookup` for the hardware use of
  `working_save-ATC-50`; and
- `learned transition-onset selector` only for the separately implemented
  four-link experiment.

The legacy action-weight appendix is evaluated from the coarse expert-result
arrays on the declared common-feasible mask. Figure and hardware descriptions
may refer to the refined lookup, but the figures do not need to duplicate
GitHub file links when the Methods and Code Availability sections provide the
shared repository provenance.
