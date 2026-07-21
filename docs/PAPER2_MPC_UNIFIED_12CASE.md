# Unified two-link MPC 12-case release

This repository release replaces the earlier heterogeneous MPC result table
with one source-aligned, repository-relative coarse-to-fine workflow. The
historical CSV is retained under `results/legacy/`; it is not silently
overwritten or represented as the unified result.

## Accepted result

All 12 cases completed three walking steps. Every accepted scientific result
matched two independent fresh-process replays exactly, and the formal post-run
validator reported zero errors. The accepted overall Cmt is
`0.198945098769` with sample SD `0.021963452200`.

The accepted case table preserves the signed evaluator output as
`foot_error_m` and adds `absolute_foot_error_m`. The 12-case mean absolute
landing error is `0.06382515890169439 m`; manuscript summaries must not use the
signed mean (`0.061671879977 m`) as a mean absolute error.

| Condition | n | Mean Cmt | Sample SD |
|---|---:|---:|---:|
| flat, nominal 1.280 m | 3 | 0.188705208 | 0.012507092 |
| flat, nominal 1.145 m | 3 | 0.208204140 | 0.016585376 |
| raised 0.01 m, nominal 1.280 m | 3 | 0.178283767 | 0.015569172 |
| raised 0.01 m, nominal 1.145 m | 3 | 0.220587280 | 0.018720893 |

These values are deterministic minima within the evaluated 24-by-24 coarse
grid, the case-specific fine grid, and the documented adjacent-window closure.
They are not a proof of global optimality over a continuous recovery domain.

## Code and evidence map

- `src/paper2/mpc/two_link_mpc_multistep_compare.py`: pinned two-link plant,
  event, reset-energy, success, distance, and Cmt evaluator.
- `src/paper2/mpc/mpc_ppo_aligned_common.py`: pinned MPC controller and shared
  three-step simulation path.
- `src/paper2/mpc/mpc_ppo_aligned_search_recovery_coarse_to_fine.py`: retained
  historical search-driver source used in the provenance pin set.
- `src/paper2/mpc/run_unified_12case.py`: portable 12-case runner and static
  case-entrypoint audit.
- `src/paper2/mpc/extend_fine_edge.py`: predeclared adjacent-window closure.
- `src/paper2/mpc/validate_release.py`: path-free release validator.
- `configs/paper2_mpc_unified_12_cases.json`: targets, initial grid indices,
  MPC settings, source hashes, and accepted-file map.
- `results/paper2_mpc_unified_12case/`: accepted case rows and the portable
  formal-validation digest.

The three accepted source hashes before path porting are retained in the case
configuration. Their repository copies differ only in default/import/output
path declarations: workstation-specific absolute paths were replaced with
paths derived from `__file__`. The plant, controller, objective, event/reset,
search, and metric logic are unchanged. The runner pins the portable repository
bytes, while the accepted pre-port hashes preserve the link to the local
accepted-source package prepared for a subsequent Zenodo version.

The exhaustive candidate tables, traces, solve logs, process replays, and the
full validator report are in a prepared local supplementary/Zenodo-version
candidate archive. The currently published DOI does not carry this full rerun
bundle until a new Zenodo version is uploaded and published. Public repository
release files contain repository-relative paths only.

## Initial-state correction and reset boundary

The unified case manifest requires the compared controllers for a case to use
the same target-dependent initial index. Five repository copies were corrected
to the accepted source-aligned configuration:

- `flat_1.145_r1/Qi_multi_passive_sim.py`: `(19, 11, 15, 7)`;
- `raised_1.145_r1/Multi_continuous.py`: `(19, 11, 15, 1)`;
- `raised_1.145_r1/Qi_multi_ac_discrete_sim.py`: `(19, 11, 15, 1)`;
- `raised_1.145_r1/LIPM.py`: `(19, 11, 15, 1)`;
- `raised_1.145_r1/LQR.py`: `(19, 11, 15, 1)`.

The raised `Qi_multi_passive_sim.py` already used `(19, 11, 15, 1)`.
Thus all five raised-1.145-r1 controller entry points now provide direct code
evidence for the paper's same-physical-initial-state comparison.

The unified MPC runner parses the two entry-point families only to audit
`BASE_PARAMS`, `PARAMS1`, `PARAMS2`, and `init_idx`. It does not import or run
`Multi_continuous.py`; the actual MPC call chain imports only the three pinned
MPC files and injects the case configuration into the shared evaluator.
Consequently, controller-specific recovery/reset values in
`Multi_continuous.py` belong to the Continuous-torque PPO baseline and are not
MPC inputs. In particular, the raised-1.145-r1 baseline retains its archived
`0.87/0.89` recovery pair; it was not replaced by the accepted-source file's
`1.00/0.89` pair.

## Case-ID mapping

The accepted MPC artifact and current combined table use canonical IDs. Older
standalone MPC tables used decimal length labels, while the historical PPO
audit used `L128` for the same nominal 1.28-m group. These names map one-to-one:

| Canonical/current prefix | Legacy decimal MPC prefix | Legacy PPO audit prefix |
|---|---|---|
| `flat_L1145_` | `flat_1.145_` | `flat_L1145_` |
| `raised_L1145_` | `raised_1.145_` | `raised_L1145_` |
| `flat_L1280_` | `flat_1.28_` | `flat_L128_` |
| `raised_L1280_` | `raised_1.28_` | `raised_L128_` |

The suffix `r1`, `r2`, or `r3` is preserved. Thus, for example,
`flat_L1280_r2`, legacy `flat_1.28_r2`, and legacy `flat_L128_r2` identify the
same nominal condition/case slot. The generator asserts that every one of the
12 canonical cases has exactly one row for each of the six methods.

## Search-boundary resolution

The retained formal report contains three warnings from the original full
search. For `flat_L1145_r3`, an additional 100-candidate adjacent r2 window
closed the artificial fine-window edge; the accepted r2 value `-0.98` is
interior and r1 `-1.19` is the physical lower bound. For
`raised_L1280_r3`, the complete physically clipped 10-by-11 fine grid produced
the interior accepted pair r1 `-1.11`, r2 `-0.94`.

## Reproduction and validation

From the repository root:

```bash
python -m pip install -r environment/requirements-mpc.txt
python src/paper2/mpc/run_unified_12case.py --audit --output-root /tmp/mpc-audit
python src/paper2/mpc/validate_release.py
python analysis/reproduce_paper2_tables.py
python -m unittest discover -s tests -v
```

A new numerical rerun should use a new output directory. Do not write generated
candidate data into `results/paper2_mpc_unified_12case/`, which is the accepted
release record.
