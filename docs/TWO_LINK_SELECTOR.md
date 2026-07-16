# Relationship between the two-link offline tables

`working_save_passive-10-30` is the coarse `10 x 30 x 10 x 30` offline result
map obtained from the sign-restricted expert evaluations. Its entries encode
the selected sign/action outcome after considering expert feasibility and, when
both experts are valid, their Cmt values.

`working_save-ATC-50` is the `50^4` refined deployment table derived from the
same offline mapping concept. It provides finer state resolution for the
hardware/controller lookup; it is not an online comparison of critic return
estimates. Values `-1`, `0`, and `+1` are converted to the corresponding
negative, zero, and positive hip-torque commands. The `-2` value is the
uncovered/failure sentinel handled by the deployment fallback.

## Deployment update rule

The controller discretizes the measured two-link state and queries
`working_save-ATC-50`. It keeps the selected route/action between lookup events.
The indices are updated when the stance-link angle changes by about
`1.8 degrees`, after which the refined table is queried again. This explains
how the offline map is connected to the state trajectories in the manuscript
without incorrectly describing the implementation as an online return-based
selector.

## Manuscript terminology

Use the following names consistently:

- `coarse offline expert-result map` for `working_save_passive-10-30`;
- `refined event-updated deployment lookup` for `working_save-ATC-50`; and
- `learned transition-onset selector` only for the separately implemented
  four-link experiment.

The Table II `Proposed` row is evaluated from the coarse expert-result arrays
on the declared common-feasible mask. Figure and hardware descriptions may
refer to the refined lookup, but the figures do not need to duplicate GitHub
file links when the Methods and Code Availability sections provide the shared
repository provenance.
