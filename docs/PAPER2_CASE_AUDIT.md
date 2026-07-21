# Tables IV-V: 12-case read-only audit

Audit date: 2026-07-11. The five discovered non-MPC entry points were executed
in each of the 12 supplied case directories. Before and after every process,
the complete source directory was snapshotted by relative path, byte size, and
modification time. All 60 source trees were unchanged.

## Result status

| Manuscript family | Entry point | Cmt records | Status |
|---|---|---:|---|
| Offline transition-start route | `Qi_multi_passive_sim.py` | 12/12 | complete; the legacy filename is retained; landing residual is missing in one case |
| Discrete active PPO | `Qi_multi_ac_discrete_sim.py` | 12/12 | complete |
| Continuous-torque PPO | `Multi_continuous.py` | 12/12 | complete |
| LIPM COM | `LIPM.py` | 12/12 | complete Cmt reference; no time/error fields |
| TVLQR tracking | `LQR.py` | 12/12 | fixed-seed portable release completed all cases |
| PID tracking | none found | 0/12 | unsupported; removed from revised tables |

Three archived flat/1.145-m TVLQR scripts call `dist.sample()` without an
internal RNG seed; one exceeded the initial 180-s audit timeout. The released
wrapper applies seed 0 only to a temporary copy, redirects the four archived
checkpoint paths, and retains the original source hashes. Under this declared
protocol all 12 cases completed. The earlier timeout is preserved as audit
history, while the seeded 12-case release is the manuscript source.

## Frozen route and metric boundaries

The 12 route records are outputs of the retained scripts, not a clean
fixed-horizon comparison of actuator work. Only route value `-1` selects the
negative expert; `+1`, `0`, `-2`, NaN table values, and other unrecognized table
values all fall through to the positive expert. Four coordinate-wise nearest
grid lookups are used without interpolation. Finite out-of-range inputs map to
an endpoint, and equal-distance ties inherit NumPy's first-index rule. The
consumer has no explicit out-of-distribution or NaN-state safety check.

The flat and raised route-table producers also use different dual-success
priority rules: the retained flat producer selects the negative expert first,
whereas the raised producer selects the positive expert first. Neither frozen
rule is a general lower-Cmt tie breaker. Changing these rules would require a
new route build and rerun; the present package instead discloses the archived
behavior.

Although the entry points define `simu_time = 3.2`, their multi-step `while`
loops do not enforce that horizon or a maximum retry count. They continue until
three target-box events have accumulated. Failed attempts still contribute to
the numerator and denominator, and deterministic repeated failure can in
principle produce an infinite loop. The released read-only audit completed all
12 supplied cases, but that completion does not create a source-level horizon.

For the route, discrete PPO, and continuous PPO entry points, the reported
ratio is a **script-level transition aggregate**. Its numerator combines
positive commanded hip work with a case- and controller-specific post-impact
kinetic-energy increment induced by a prescribed velocity reset. The latter is
not measured recovery-actuator work, is not constrained to be positive, and
uses different reset magnitudes across cases and controllers. The denominator
is accumulated absolute endpoint COM displacement over all terminated
attempts. The two numerator components and the denominator were not retained
separately, so the archived ratios cannot be decomposed without a rerun.

## Current complete summaries

The five complete replay families and the accepted source-aligned unified MPC
release give:

| Method | n | Mean Cmt | Sample SD | 95% Student-t CI | Mean time (s) | Mean absolute terminal landing error (m) |
|---|---:|---:|---:|---:|---:|---:|
| LIPM COM | 12 | 0.232807 | 0.030205 | [0.213616, 0.251999] | N.A. | N.A. |
| TVLQR tracking | 12 | 0.232348 | 0.015446 | [0.222534, 0.242162] | Reference only | N.A. |
| Discrete active PPO | 12 | 0.228757 | 0.017414 | [0.217693, 0.239821] | 2.680 | 0.145132 |
| Continuous-torque PPO | 12 | 0.240916 | 0.018083 | [0.229427, 0.252405] | 3.405 | 0.125820 |
| Continuous-torque MPC | 12 | 0.198945 | 0.021963 | [0.184990, 0.212900] | 2.641 | 0.063825 |
| Offline transition-start route | 12 | 0.123473 | 0.010531 | [0.116782, 0.130164] | 2.797 | 0.082750 (n=11) |

All reported landing-error means use the absolute value of each case-level
terminal residual before averaging. The proposed landing-error mean must retain
its `n=11` label; the missing
case must not be silently imputed. The manuscript's former 0.080-m value is not
reproduced by the available 11 records.

## Table IV drift

For the 19 complete non-MPC method-by-condition cells in the old table, only
seven agree under ordinary three-decimal rounding. The largest non-TVLQR
absolute difference is 0.001211. The directionally consistent small drift is
compatible with an earlier code/model version or manual truncation, but its
cause is not documented. The revised table is regenerated directly from the
released case CSV and uses round-to-nearest.

## Released evidence

- `data/paper2/readonly_rerun/paper2_rerun_results.csv`: one row per attempted
  case and method, including status, parsed metrics, source SHA-256, log paths,
  and source-tree change status;
- `data/paper2/readonly_rerun/*.stdout.txt` and `*.stderr.txt`: complete captured
  process output;
- `src/paper2/entrypoints/`: historical entry-point copies. The read-only audit
  retains the original hashes; five `init_idx` assignments were subsequently
  corrected to the accepted source-aligned MPC case configuration and are
  explicitly tested and documented in `docs/PAPER2_MPC_UNIFIED_12CASE.md`;
- `models/paper2/tvlqr/`: all four checkpoints required by the TVLQR sources;
- `tools/run_frozen_tvlqr.py` and `results/tvlqr_seed0/`: portable fixed-seed
  runner, 12 structured records, and complete captured logs;
- `analysis/reproduce_paper2_tables.py`: deterministic current-table builder;
- `results/paper2_source_aligned_r1_rerun_20260721/`: exact Discrete and
  Continuous PPO records for `raised_L1145_r1`, including raw stdout, hashes,
  environment metadata, and the explicit boundary between the Continuous
  1.00/0.89 snapshot and the retained 0.87/0.89 entrypoint;
- `tools/audit_paper2_readonly.py`: parameterized read-only runner for the DOI
  case bundle.
- `src/paper2/mpc/`, `configs/paper2_mpc_unified_12_cases.json`, and
  `results/paper2_mpc_unified_12case/`: repository-relative unified MPC code,
  accepted 12-case values, and path-free validation digest. The superseded MPC
  CSV is retained under `results/legacy/`.

The TVLQR family is self-contained in GitHub. The other Paper2 methods still
require the complete relative dependencies, checkpoints, and arrays to be
deposited as a versioned DOI artifact.
