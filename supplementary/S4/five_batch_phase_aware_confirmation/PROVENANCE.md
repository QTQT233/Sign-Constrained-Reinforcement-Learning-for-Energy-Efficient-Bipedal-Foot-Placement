# Frozen-source provenance and public entry-point mapping

`FROZEN_SHA256SUMS.txt` and `EXECUTION_AMENDMENT_01_SHA256SUMS.txt` are
time-stamped source-hash ledgers. They preserve the names and SHA-256 values
of the author-workspace files that were frozen before the five-batch analysis;
they are not checksum inventories of the current repository payload.

The public release uses repository-relative, validation-hardened entry points.
The mapping is:

| Frozen author-workspace name | SHA-256 in the frozen ledger | Public release path |
|---|---|---|
| `FROZEN_PROTOCOL.json` | `3ee4fc567e0ae922fd9379ff0eb7cd3eb2bb5dd850cc26e199230ea45bda642f` | `protocol.json` |
| `code/run_new_seed_route_bank.py` | `9e03172f4948d2d9eca68ff35c3ca6b2c3aa513ac36704a43ab9ecc0066bb932` | `code/run_route_bank_seed.py` |
| `code/replay_hard_mask_on_confirmation_states.py` | `4b12d9927851ab62d359e8bf9b962ac36136b32e8d6c65a0213784bca30f4c7a` | `code/replay_hard_mask.py` |
| `code/run_validated_route_bank.py` | `f2d6279b374f57b65952752fea54ffe5ac6da1c4aacf4a976e8febbb9795770d` | `code/run_route_bank_seed.py` |

The released `protocol.json` is byte-identical to the frozen protocol and
therefore has the same SHA-256 value. The released Python entry points are not
claimed to be byte-identical to the author-workspace wrappers. They replace
workspace path handling with repository-root discovery, combine the
route-bank validation amendment with the seed runner, and expose portable
command-line interfaces. They do not replace the scientific evaluator:
the evaluator and all controller checkpoints are separately hash-pinned in
`protocol.json`, and their hashes are checked by
`tests/test_five_batch_phase_aware_release.py`.

For verification of the files actually distributed in this repository, use
`results/SHA256SUMS.txt`, `supplementary/S4/MANIFEST.csv`,
`supplementary/S4/CHECKSUMS.sha256`, and the repository-level
`MANIFEST.csv` and `CHECKSUMS.sha256`.
