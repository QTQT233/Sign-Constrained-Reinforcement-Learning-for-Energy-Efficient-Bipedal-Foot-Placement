# Success non-inferiority and equivalence

## Manuscript-primary values: 44.6% versus 45.1%

The old 5 July output contains 975 active-PPO successes and 964 selector
successes among the same 2160 cases. Its paired table is:

| Active PPO | Selector success | Selector failure | Total |
|---|---:|---:|---:|
| Success | 906 | 69 | 975 |
| Failure | 58 | 1127 | 1185 |
| Total | 964 | 1196 | 2160 |

Thus `p(selector)-p(active) = -0.509` percentage points, with paired 95% CI
[-1.532, 0.513], paired 90% CI [-1.367, 0.349], and exact two-sided McNemar
`p=0.375`. This supports the wording "no paired success difference was
detected." It does not prove equality.

Had an absolute 2-percentage-point margin been justified and prespecified, the
one-sided 95% lower bound (-1.367 points) would satisfy non-inferiority, and the
two-sided 90% interval would lie inside [-2, +2], satisfying a symmetric
equivalence rule for this fixed-checkpoint experiment. Because the margin was
chosen after seeing the results, both are sensitivity analyses, not
confirmatory claims.

## Primary paired endpoint

Use all 2160 shared evaluation cases and define
`d = p(selector success) - p(active success)`. The paired table above, rather
than a later personal test protocol, is the only success endpoint used in the
manuscript. Failure to reject a difference is not proof of equivalence.

The one-sided 95% lower confidence bound is -1.367 percentage points. If an
absolute non-inferiority margin of 2 percentage points had been justified and
prespecified before evaluation, this fixed-checkpoint result would meet that
criterion. Because the margin was selected after viewing the data, it must be
reported as a post-hoc sensitivity analysis unless the protocol is locked and
repeated on a new test manifest.

Algorithm-level non-inferiority also requires independent training seeds. The
three seeds in the current CSV generate evaluation initial states; they are not
three independently trained policies.

## Same 2160 attempts, smaller paired Cmt cohort

Active PPO and the online selector already receive the same 2160 initial
states. Their case-key sets and all recorded initial-state fields match. No
control-logic change is required to make the success endpoint paired.

Cmt has a different denominator because it is declared valid only after task
success, nonzero actuation, and signed forward COM displacement greater than
0.001 m. The manuscript-primary analysis therefore contains 419 jointly valid
Cmt pairs. State the two estimands separately:

- success non-inferiority on all 2160 paired attempts;
- conditional paired Cmt comparison on 419 jointly eligible transitions.

Do not compare each controller's separate valid-only mean or impute Cmt=0 for
failed/ineligible cases. For stronger auditability, write a stable `case_id`,
initial-state SHA-256, `cmt_exclusion_reason`, checkpoint hash, and evaluation
configuration hash for every controller-case row, then assert equality of the
two 2160-case key and initial-state-hash sets.

## What the terminal-reason table can and cannot show

The terminal-reason table describes how controllers fail and can identify a
new severe failure mode. It cannot establish success non-inferiority because
it contains marginal counts, not the paired 2x2 success transitions. Similar
margins can hide different sets of successful cases. Report the paired success
analysis first, then use a paired terminal-reason transition matrix as a
secondary diagnostic.
