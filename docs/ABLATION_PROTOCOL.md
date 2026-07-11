# Corrected mechanism-ablation protocol

## Why the current comparison is not causal

The current `per_step_sign` action indices map one-to-one to the same 27 actual
torque triples as unrestricted active control: hip torque is -4, 0, or +4 Nm
and each knee torque is -0.2, 0, or +0.2 Nm. There is no mask, dwell time, or
locked sign. The two controllers therefore admit the same torque sequences.

The current `active_full` program sets `action_value_weight = 0`, whereas the
`bidirectional` program uses 0.026. Because the penalty is subtracted after all
task-reward branches, the difference propagates through the stored reward,
one-step TD target, GAE advantage, critic loss, and PPO actor loss. At the reward
definition level, the two programs therefore implement U0 versus U1.

However, Active-full is warm-started from the U1 c090 checkpoint, while the U1
run has a different training lineage and budget; training seeds were not fixed.
The available Active-full checkpoint is consequently a zero-penalty continuation
diagnostic, not a matched-seed from-scratch U0 estimate.

The current runs may be retained as encoding/checkpoint-sensitivity diagnostics
but cannot identify a persistence or reward mechanism.

## Mapping the current methods to factorial cells

Define `U/P` as unrestricted versus transition-persistent sign control and
`0/1` as torque-penalty coefficient 0 versus 0.026. The current mapping is:

| Current label | Correct role | Factorial status |
|---|---|---|
| Active PPO / bidirectional | U1 | available, but needs matched-seed retraining |
| Active-full | U0 at reward-definition level | available only as a single warm-started continuation diagnostic |
| Online sign selector | deployable P1 system | available, but architecture/configuration differ from U1 |
| Oracle envelope | P1-oracle upper bound | not a deployable P1 cell |
| Per-step sign | U1-equivalent action encoding | not an action mask |

Thus a provisional single-checkpoint U0 diagnostic exists, but matched-seed U0
confirmation and the P0 cell are absent. The online selector, not the oracle
envelope, is the deployable P1 result. The oracle should remain only as a
hindsight two-expert envelope.

## Terminology boundary

The one-sided experts constrain hip-torque sign but do not impose the power
inequality `tau_hip * qdot_hip <= 0`. A fixed-sign actuator may still perform
positive mechanical work when torque and angular velocity have the same sign.
Without an explicit storage-element and energy-balance model, the manuscript
therefore uses `one-sided` or `sign-constrained actuation`, not strict `passive
control`. The oracle is named `Oracle one-sided-policy envelope (hindsight,
non-deployable)`.

## Interpreting the proposed comparisons

The intended distinction is scientifically reasonable, but the current labels
do not implement it. If a "per-step sign" controller may choose either
`{-4,0}` or `{0,+4}` anew at every control step, the union available at that
step is exactly `{-4,0,+4}`. Across a trajectory it can generate every torque
sequence available to active PPO. A binary sign head plus a magnitude head may
change optimization or parameterization, but it is not a control constraint
unless the chosen sign is held for more than one step.

Use one hierarchical policy for every persistence treatment: a selector chooses
the sign at a specified boundary, and an otherwise identical low-level policy
chooses zero or the magnitude allowed by that sign. At `K=1`, the selector is
updated every step; at `K=transition`, it is sampled once and stored until the
terminal event. This common architecture makes the update interval, rather
than network topology, the manipulated variable.

"Active-full versus bidirectional" is a torque-penalty contrast at the reward
definition level because the frozen programs use coefficients 0 and 0.026,
respectively. The present checkpoints still differ in training lineage and
budget, so the existing result must remain a continuation diagnostic. For
confirmatory evidence, train explicit `penalty=0` and `penalty=0.026` cells from
matched initializations and seeds. If branching from a common checkpoint is
unavoidable, branch every cell from the same checkpoint, give equal post-branch
environment steps, and state that the result is a fine-tuning experiment rather
than de novo learning.

## Minimum 2 x 2 factorial experiment

| Cell | Sign update | Hip-torque set | Torque penalty |
|---|---|---|---:|
| U0 | every step | {-4, 0, +4} Nm | 0 |
| U1 | every step | {-4, 0, +4} Nm | 0.026 |
| P0 | choose once and hold to transition end | {0,+4} or {-4,0} Nm | 0 |
| P1 | choose once and hold to transition end | {0,+4} or {-4,0} Nm | 0.026 |

This design separates the persistence main effect, torque-penalty main effect,
and their interaction. A stronger dose-response experiment uses a common
hierarchical architecture and changes only the sign update interval
`K = 1, 5, 10, 25, 50, transition`.

## Baselines needed to distinguish the method from action masking

The most important new baseline is `AM-P1`: one shared 27-output actor with a
transition-persistent action mask. At transition onset, a selector stores
`z in {-1,+1}`. Hip-zero actions remain valid; nonzero hip actions opposite to
`z` receive `-inf` logits. Behavior log-probability, PPO update
log-probability, and entropy must all use the same masked distribution.

`AM-P1` and the proposed two-expert P1 system then admit the same torque
sequences and share the same transition-level commitment. Their comparison
tests two-expert routing versus a shared masked actor. Add `AM-step1`, which
reselects the sign and mask every step, to test transition commitment versus
ordinary per-step masking. The minimum mechanism table should therefore report:

| Method | Purpose |
|---|---|
| U1 | unrestricted penalized active PPO |
| P1-deployable | proposed online selector plus two one-sided experts |
| AM-P1 | same transition commitment implemented by a shared masked actor |
| AM-step1 | per-step action-mask control |
| P1-oracle | non-deployable upper bound only |
| U0 | zero-penalty unrestricted control; existing result is diagnostic and requires matched-seed confirmation |
| P0 | missing zero-penalty persistent-sign cell needed for the interaction |

Because two experts have more parameters and may consume more training
interactions than one actor, report both compute-matched and parameter-matched
comparisons. Train the action-mask selector from labels generated by its own
conditional policy; do not reuse labels that privilege the proposed experts.

## Fairness and reproducibility requirements

- Use at least five, preferably ten, independent matched training seeds.
- Fix Python, NumPy, PyTorch, and CUDA seeds and record deterministic settings.
- Give every cell the same environment-step budget, initialization protocol,
  curriculum schedule, network capacity, and checkpoint-selection rule.
- Evaluate every trained policy on the same frozen case manifest with common
  random numbers.
- Record sign-switch count, sign dwell time, hip/knee work, Cmt, displacement,
  success, validity, and terminal reason for every trial.
- Treat any within-transition sign switch in a persistent cell as an
  implementation failure.
- Define a sign switch between consecutive nonzero hip commands; zeros neither
  create nor erase the stored transition sign. Also record the selected sign
  directly so that an all-zero trajectory is not misclassified as persistent.

For success/validity, use paired risk differences or a mixed-effects logistic
model with evaluation case and training seed effects. For Cmt, analyze validity
first and then the matched both-valid outcome, with training-seed-aware
bootstrap or mixed effects. Do not treat evaluation trials as independent
training replications.
