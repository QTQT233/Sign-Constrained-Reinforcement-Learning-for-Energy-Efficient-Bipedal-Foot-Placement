# Training and evaluation environment provenance

The original training folders did not contain a complete package lock. The
author-workstation environment that loads the released checkpoints and was used
for release verification is recorded in `local-audit-environment.txt`; its
principal four-link dependencies are also listed in
`requirements-four-link.txt`. This captured rerun environment supports loading
and evaluating the released artifacts but does not establish that every
historical checkpoint was trained with byte-identical library versions.

The released fixed-checkpoint evaluations are protected by source, model, data,
and output SHA-256 hashes. Random seeds are reported where they were recorded;
the scratch true-action-mask training seed was not recorded and is not
reconstructed retrospectively.
