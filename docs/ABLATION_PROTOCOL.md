# Four-link sign-persistence ablation protocol

## Compared controller structures

- **Active PPO:** unrestricted hip-torque sign at every step.
- **Active-full:** the same physical action grid with zero torque penalty; this
  row is a reward/reference diagnostic.
- **True per-step hard mask (scratch):** a learned sign head refreshes the
  admissible nonzero hip-torque sign at every step, while zero-hip actions
  remain available.
- **Online selector:** one one-sided expert is selected at transition onset and
  held until termination.
- **Oracle envelope:** an offline lower-`Cmt` envelope over the two one-sided
  experts; it is not deployable.

All reported four-link rows use the shared V22_3/V9 2,160-case evaluator. The
true mask and the fixed-sign experts use the same principal scalar reward
coefficients, including `action_value_weight = 0.026`. The true-mask network is
trained from scratch. Its architecture, curriculum branch, soft clearance
thresholds, and initialization differ from the two-expert selector system.

## Interpretation

The hard mask tests whether sign masking at each step reproduces the selector's
energy pattern. It does not: selector and mask success rates show no detectable
paired difference, while the selector has lower `Cmt` on 354 of 395 both-valid
matched pairs and reduces mean `Cmt` from 1.303 to 0.278.

Accordingly, this is described as an ablation within the evaluated reward and
training configuration. It identifies transition-level sign persistence,
rather than sign masking alone, as the component associated with the lower
conditional `Cmt`. The comparison is not presented as a strict one-factor
causal experiment because the two controller families do not share an
identical architecture and training lineage.

## Active-full source boundary

The current active-full training entry points are scratch-only. They contain no
policy/critic loading code and reject a non-empty output directory, preventing
checkpoint reuse or log appending. The fixed checkpoint represented in the
current evaluation archive is used only as a diagnostic row and is not the
basis of the main selector-versus-active or selector-versus-mask conclusions.

## Terminology

The one-sided experts constrain hip-torque sign; they do not enforce the power
inequality `tau_hip * qdot_hip <= 0`. The manuscript therefore uses
“one-sided,” “sign-constrained,” and “transition-persistent” rather than
claiming passive control.
