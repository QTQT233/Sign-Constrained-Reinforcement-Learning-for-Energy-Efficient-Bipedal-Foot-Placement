# Draft manuscript statements

## Data availability

The unchanged author-owned hardware evidence is available from Zenodo at
<https://doi.org/10.5281/zenodo.21407986>. This DOI is limited to the hardware-
data record and is not the source of the corrected software or the unified
12-case MPC results. Author-generated simulation inputs, retained evaluation
outputs, accepted MPC case results, and machine-readable provenance files that
are distributed publicly are available in the GitHub repository at the exact
commit identified in the Code Availability statement. This includes the frozen
scratch true-action-mask checkpoint and training log, all 2,160-case V22_3/V9
trial outputs, paired analyses, figure source data, plotting scripts, manifests,
and checksums. V23 diagnostic code is not used for the manuscript comparison.
Exhaustive MPC candidate
tables, solve traces, and fresh-process replay records are supplied with the
submission as Supplementary Archive S1. Author-owned data and videos are
licensed under Apache-2.0; third-party materials retain their original terms.

## Code availability

Frozen training, evaluation, MPC, and statistical-analysis code is available
under Apache-2.0 at
`https://github.com/QTQT233/Sign-Constrained-Reinforcement-Learning-for-Energy-Efficient-Bipedal-Foot-Placement`.
The manuscript identifies the exact public Git commit used for submission. An
immutable commit on `main` is the software version pin; a separate GitHub
Release is optional and is not required for this code-only correction. The
repository records the mapping from manuscript tables and figures to scripts,
configurations, retained outputs, and checksums. The two-link hardware
controller uses an event-updated offline lookup, whereas the four-link stress
test uses a learned online selector at transition onset.
