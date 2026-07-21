"""Isolated, auditable 12-case rerun harness for the paper's two-link MPC.

This file never writes to the archived Paper2 entry points. It imports the
repository's SHA-256-pinned PPO-aligned MPC implementation, injects the
case-specific targets and initial state, and writes every result below the
requested output root.

Examples:
    python src/paper2/mpc/run_unified_12case.py --audit
    python src/paper2/mpc/run_unified_12case.py --mode smoke --case raised_L1145_r1
    python src/paper2/mpc/run_unified_12case.py --all --workers 2 --case-concurrency 12

The full protocol is the historical 24 x 24 coarse recovery search followed
by an at-most 11 x 11 fine search.  Recovery is tied by stance as [r1,r2,r1].
Failed trajectories retain diagnostic energy/distance in raw files, but are
never assigned a valid/reportable Cmt by this harness.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Prevent one scipy process from creating additional BLAS/OpenMP workers.  The
# requested parallelism is explicit and controlled by --workers.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
import scipy


MPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MPC_DIR.parents[2]
ENTRYPOINT_ROOT = REPO_ROOT / "src" / "paper2" / "entrypoints"
CONFIG_PATH = REPO_ROOT / "configs" / "paper2_mpc_unified_12_cases.json"
CORE_SOURCE = MPC_DIR / "two_link_mpc_multistep_compare.py"
COMMON_SOURCE = MPC_DIR / "mpc_ppo_aligned_common.py"
SEARCH_SOURCE = MPC_DIR / "mpc_ppo_aligned_search_recovery_coarse_to_fine.py"

# These hashes freeze the exact source reviewed on 2026-07-20. If a pinned
# repository file changes, the harness stops instead of silently changing the
# controller or evaluator.
PINNED_SHA256 = {
    CORE_SOURCE: "F8DC01BAC479F549FE6E7AAB6667771BE8B6B3DEAB215BA5406D510093A07760",
    COMMON_SOURCE: "452791D164605F9A38E8F46BEBA4C407A9A6F409B9DC4B79D3935A7B62ABBBFE",
    SEARCH_SOURCE: "D778C44E345896C516CCD686A64F7CC9604CBCEAB7435FEA4AC8F4351C0E5E32",
}

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "paper2_mpc_unified_12case_rerun"


BASE_COMMON = {
    "g": 9.8,
    "dt": 0.01,
    "max_torque": 4.0,
    "theta1_range": 90.0,
    "theta2_range": 90.0,
    "speed_range": 2.0,
    "settle": 5.0,
    "reward_scale": 5.0,
    "action_value_weight": 0.03,
}

MODEL_STANCE1 = {
    "m1": 1.5981,
    "m2": 0.6503,
    "l1": 0.521,
    "l2": 0.481,
    "l1_prime": 0.4114,
    "l2_prime": 0.218,
    "J1": 0.5,
    "J2": 0.105,
    "c1": 0.00047630386389081107,
    "c2": 0.0001,
}

MODEL_STANCE2 = {
    "m1": 0.6503,
    "m2": 1.5981,
    "l1": 0.521,
    "l2": 0.481,
    "l1_prime": 0.30300000000000005,
    "l2_prime": 0.05916,
    "J1": 0.34,
    "J2": 0.1,
    "c1": 0.001,
    "c2": 0.0001,
}

MPC_SETTINGS = {
    "action_mode": "continuous",
    "cost_mode": "cmt",
    "target_mode": "primary",
    "total_walking_steps": 3,
    "control_hold_steps": 3,
    "num_knots": 55,
    "max_replans_per_walking_step": 3,
    "maxiter": 180,
    "ftol": 1e-5,
    "terminal_angle_weight": 420.0,
    "terminal_velocity_weight": 0.0,
    "running_target_weight": 0.08,
    "predicted_work_weight": 1.0,
    "reward_action_weight": 0.04,
    "torque_abs_weight": 0.004,
    "torque_smooth_weight": 0.012,
    "fall_soft_weight": 1000.0,
    "discrete_cem_samples": 96,
    "discrete_cem_elite": 16,
    "discrete_cem_iters": 4,
    "random_seed": 7,
    "recovery_min": -1.19,
    "recovery_max": 0.0,
    "coarse_step": 0.05,
    "fine_radius": 0.05,
    "fine_step": 0.01,
    "tie_recovery_by_stance": True,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    source_dir_name: str
    terrain: str
    length_label: str
    replicate: int
    targets_deg: Tuple[float, float, float, float]
    initial_idx: Tuple[int, int, int, int]

    @property
    def source_dir(self) -> Path:
        return ENTRYPOINT_ROOT / self.source_dir_name


def _case(case_id: str, dirname: str, terrain: str, length: str, replicate: int,
          targets: Sequence[float], idx: Sequence[int]) -> Case:
    return Case(case_id, dirname, terrain, length, replicate,
                tuple(float(v) for v in targets), tuple(int(v) for v in idx))


FLAT = (60.0, 30.0, 120.0, -30.0)
RAISED = (61.8, 31.7, 118.2, -31.7)

CASES: Dict[str, Case] = {
    c.case_id: c for c in [
        _case("flat_L1145_r1", "flat_1.145_r1", "flat", "1.145", 1, FLAT, (19, 11, 15, 7)),
        _case("flat_L1145_r2", "flat_1.145_r2", "flat", "1.145", 2, FLAT, (19, 14, 15, 5)),
        _case("flat_L1145_r3", "flat_1.145_r3", "flat", "1.145", 3, FLAT, (19, 19, 15, 4)),
        _case("flat_L1280_r1", "flat_1.28_r1", "flat", "1.280", 1, FLAT, (27, 0, 15, 13)),
        _case("flat_L1280_r2", "flat_1.28_r2", "flat", "1.280", 2, FLAT, (27, 1, 15, 10)),
        _case("flat_L1280_r3", "flat_1.28_r3", "flat", "1.280", 3, FLAT, (27, 3, 15, 9)),
        _case("raised_L1145_r1", "raised_1.145_r1", "raised_0.01m", "1.145", 1, RAISED, (19, 11, 15, 1)),
        _case("raised_L1145_r2", "raised_1.145_r2", "raised_0.01m", "1.145", 2, RAISED, (19, 14, 15, 8)),
        _case("raised_L1145_r3", "raised_1.145_r3", "raised_0.01m", "1.145", 3, RAISED, (19, 19, 15, 4)),
        _case("raised_L1280_r1", "raised_1.28_r1", "raised_0.01m", "1.280", 1, RAISED, (27, 0, 15, 21)),
        _case("raised_L1280_r2", "raised_1.28_r2", "raised_0.01m", "1.280", 2, RAISED, (27, 1, 15, 25)),
        _case("raised_L1280_r3", "raised_1.28_r3", "raised_0.01m", "1.280", 3, RAISED, (27, 3, 15, 9)),
    ]
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def repo_relative(path: Path) -> str:
    """Return a portable repository-relative POSIX path for manifests."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def verify_pinned_sources() -> Dict[str, str]:
    actual: Dict[str, str] = {}
    for path, expected in PINNED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(f"Pinned source missing: {path}")
        value = sha256(path)
        actual[repo_relative(path)] = value
        if value != expected:
            raise RuntimeError(
                f"Pinned source changed; refusing to run.\n  file: {path}\n"
                f"  expected: {expected}\n  actual:   {value}"
            )
    return actual


def _literal_source_config(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    found: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"BASE_PARAMS", "init_idx"}:
                try:
                    found[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "update"
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in {"PARAMS1", "PARAMS2"}):
            found[node.value.func.value.id] = ast.literal_eval(node.value.args[0])
    return found


def expected_base(case: Case) -> Dict[str, float]:
    base = dict(BASE_COMMON)
    base.update(dict(zip(("target1", "target2", "target3", "target4"), case.targets_deg)))
    return base


def audit_case(case: Case) -> Dict[str, Any]:
    expected = {
        "BASE_PARAMS": expected_base(case),
        "PARAMS1": MODEL_STANCE1,
        "PARAMS2": MODEL_STANCE2,
        "init_idx": case.initial_idx,
    }
    sources: Dict[str, Any] = {}
    for filename in ("Qi_multi_passive_sim.py", "Multi_continuous.py"):
        path = case.source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Case source missing: {path}")
        actual = _literal_source_config(path)
        for key, wanted in expected.items():
            got = actual.get(key)
            if key == "init_idx" and got is not None:
                got = tuple(got)
            if got != wanted:
                raise RuntimeError(
                    f"Case manifest mismatch: {case.case_id} / {filename} / {key}\n"
                    f"expected={wanted!r}\nactual={got!r}"
                )
        sources[filename] = {"path": repo_relative(path), "sha256": sha256(path)}
    return sources


def verify_release_config() -> Dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Unified case config missing: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema") != "paper2-mpc-unified-12case-v1":
        raise RuntimeError(f"Unexpected unified case config schema: {config.get('schema')!r}")
    configured = {item["case_id"]: item for item in config.get("cases", [])}
    if set(configured) != set(CASES):
        raise RuntimeError("Unified case config does not contain exactly the runner's 12 cases")
    for case_id, case in CASES.items():
        item = configured[case_id]
        expected = {
            "entrypoint_dir": case.source_dir_name,
            "terrain": case.terrain,
            "length_label": case.length_label,
            "replicate": case.replicate,
            "targets_deg": list(case.targets_deg),
            "initial_idx": list(case.initial_idx),
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise RuntimeError(
                    f"Unified case config mismatch: {case_id}/{key}: "
                    f"expected={value!r}, actual={item.get(key)!r}"
                )
    if config.get("mpc_settings") != MPC_SETTINGS:
        raise RuntimeError("Unified case config MPC settings differ from the runner")
    return config


def initial_state(case: Case) -> np.ndarray:
    n_angle, n_speed = 30, 60
    t1 = np.linspace(case.targets_deg[0], case.targets_deg[2], n_angle) * np.pi / 180.0
    dt1 = np.linspace(-2.0, 2.0, n_speed)
    t2 = np.linspace(-60.0, 60.0, n_angle) * np.pi / 180.0
    dt2 = np.linspace(-2.0, 2.0, n_speed)
    i, j, k, ll = case.initial_idx
    return np.array([t1[i], dt1[j], t2[k], dt2[ll]], dtype=float)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_and_configure(case: Case):
    verify_pinned_sources()
    sys.modules.pop("mpc_ppo_aligned_common", None)
    sys.modules.pop("two_link_mpc_multistep_compare", None)
    core = _load_module("two_link_mpc_multistep_compare", CORE_SOURCE)
    common = _load_module("mpc_ppo_aligned_common", COMMON_SOURCE)

    base = expected_base(case)
    core.BASE_PARAMS.clear()
    core.BASE_PARAMS.update(base)
    core.PARAMS1.clear()
    core.PARAMS1.update(base)
    core.PARAMS1.update(MODEL_STANCE1)
    core.PARAMS2.clear()
    core.PARAMS2.update(base)
    core.PARAMS2.update(MODEL_STANCE2)
    core.W = (core.PARAMS1["m1"] + core.PARAMS1["m2"]) * core.PARAMS1["g"]
    core.INITIAL_GRID_INDEX = tuple(case.initial_idx)
    core.TOTAL_WALKING_STEPS = int(MPC_SETTINGS["total_walking_steps"])
    core.MAX_INNER_STEPS_PER_WALKING_STEP = 360

    # Force all paths through the target-dependent grid, including any default
    # call inside the imported reference modules.
    def case_initial_state(index: Sequence[int] = case.initial_idx) -> np.ndarray:
        local = Case(case.case_id, case.source_dir_name, case.terrain, case.length_label,
                     case.replicate, case.targets_deg, tuple(int(v) for v in index))
        return initial_state(local)

    core.initial_state_from_grid = case_initial_state
    common.core = core
    return core, common


def build_controller(common):
    s = MPC_SETTINGS
    return common.PPOAlignedMPCController(
        action_mode=common.ACTION_MODE_CONTINUOUS,
        cost_mode=common.COST_MODE_CMT,
        target_mode=common.TARGET_MODE_PRIMARY,
        control_hold_steps=s["control_hold_steps"],
        num_knots=s["num_knots"],
        max_replans_per_walking_step=s["max_replans_per_walking_step"],
        maxiter=s["maxiter"],
        ftol=s["ftol"],
        terminal_angle_weight=s["terminal_angle_weight"],
        terminal_velocity_weight=s["terminal_velocity_weight"],
        running_target_weight=s["running_target_weight"],
        predicted_work_weight=s["predicted_work_weight"],
        reward_action_weight=s["reward_action_weight"],
        torque_abs_weight=s["torque_abs_weight"],
        torque_smooth_weight=s["torque_smooth_weight"],
        fall_soft_weight=s["fall_soft_weight"],
        discrete_cem_samples=s["discrete_cem_samples"],
        discrete_cem_elite=s["discrete_cem_elite"],
        discrete_cem_iters=s["discrete_cem_iters"],
        random_seed=s["random_seed"],
    )


_WORKER: Dict[str, Any] = {}


def worker_initialize(case_id: str) -> None:
    case = CASES[case_id]
    core, common = load_and_configure(case)
    _WORKER.clear()
    _WORKER.update(case=case, core=core, common=common, initial_state=initial_state(case))


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def rank_from_result(result) -> Tuple[int, float, float]:
    cmt = float(result.cmt) if np.isfinite(result.cmt) else float("inf")
    energy = float(result.energy_j) if np.isfinite(result.energy_j) else float("inf")
    return (0 if bool(result.success) else 1, cmt, energy)


def evaluate_candidate(task: Tuple[int, Tuple[float, ...]]) -> Dict[str, Any]:
    candidate_index, sequence = task
    common = _WORKER["common"]
    start = time.perf_counter()
    controller = build_controller(common)
    result = common.run_three_step_simulation_ppo_aligned(
        controller,
        method_name="unified_mpc_candidate",
        initial_state=_WORKER["initial_state"],
        recovery_sequence=sequence,
        target_mode=common.TARGET_MODE_PRIMARY,
        total_steps=MPC_SETTINGS["total_walking_steps"],
    )
    row = common.result_summary_row(result, {
        "candidate_index": candidate_index,
        "candidate_recovery_sequence": ";".join(f"{v:.6g}" for v in sequence),
        "candidate_recovery_stance1": sequence[0],
        "candidate_recovery_stance2": sequence[1],
        "tie_recovery_by_stance": True,
        "candidate_elapsed_seconds": time.perf_counter() - start,
        "reportable_cmt": float(result.cmt) if result.success and np.isfinite(result.cmt) else "",
    })
    return {
        "candidate_index": candidate_index,
        "sequence": list(sequence),
        "rank": list(rank_from_result(result)),
        "success": bool(result.success),
        "row": _json_value(row),
    }


def sequence_product(grid1: Sequence[float], grid2: Sequence[float]) -> List[Tuple[float, ...]]:
    return [(float(r1), float(r2), float(r1)) for r1 in grid1 for r2 in grid2]


def coarse_grid() -> np.ndarray:
    stride = max(1, int(round(MPC_SETTINGS["coarse_step"] / 0.01)))
    values = -np.arange(0, 120, stride, dtype=float) * 0.01
    return np.unique(np.round(np.clip(values, MPC_SETTINGS["recovery_min"], MPC_SETTINGS["recovery_max"]), 10))


def fine_grid(center: float) -> np.ndarray:
    lo = max(MPC_SETTINGS["recovery_min"], center - MPC_SETTINGS["fine_radius"])
    hi = min(MPC_SETTINGS["recovery_max"], center + MPC_SETTINGS["fine_radius"])
    step = MPC_SETTINGS["fine_step"]
    count = int(np.floor((hi - lo) / step + 0.5)) + 1
    return np.unique(np.round(np.clip(lo + np.arange(count) * step,
                                      MPC_SETTINGS["recovery_min"], MPC_SETTINGS["recovery_max"]), 10))


def load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            rows[int(payload["candidate_index"])] = payload
        except Exception as exc:
            raise RuntimeError(f"Invalid checkpoint {path}:{line_number}: {exc}") from exc
    return rows


def append_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()


def replay_best(case_id: str, sequence: Sequence[float], method_name: str):
    if _WORKER.get("case", None) is None or _WORKER["case"].case_id != case_id:
        worker_initialize(case_id)
    common = _WORKER["common"]
    controller = build_controller(common)
    result = common.run_three_step_simulation_ppo_aligned(
        controller,
        method_name=method_name,
        initial_state=_WORKER["initial_state"],
        recovery_sequence=tuple(float(v) for v in sequence),
        target_mode=common.TARGET_MODE_PRIMARY,
        total_steps=MPC_SETTINGS["total_walking_steps"],
    )
    return result, controller


def run_parallel_stage(case: Case, stage_name: str, grid1: Sequence[float], grid2: Sequence[float],
                       case_output: Path, workers: int, resume: bool) -> Tuple[Any, Tuple[float, ...]]:
    stage_dir = case_output / stage_name
    prefix = f"mpc_{stage_name}_recovery_continuous_cmt"
    checkpoint = stage_dir / f"{prefix}_checkpoint.jsonl"
    sequences = sequence_product(grid1, grid2)
    existing = load_checkpoint(checkpoint) if resume else {}
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Checkpoint exists; use --resume or choose a new output root: {checkpoint}")
    pending = [(idx, seq) for idx, seq in enumerate(sequences) if idx not in existing]
    print(f"[{case.case_id}/{stage_name}] total={len(sequences)}, completed={len(existing)}, "
          f"pending={len(pending)}, workers={workers}", flush=True)
    start = time.perf_counter()
    errors: List[str] = []
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=worker_initialize, initargs=(case.case_id,)) as pool:
            futures = {pool.submit(evaluate_candidate, item): item for item in pending}
            completed_now = 0
            for future in as_completed(futures):
                idx, sequence = futures[future]
                try:
                    payload = future.result()
                    append_checkpoint(checkpoint, payload)
                    existing[idx] = payload
                except Exception as exc:
                    errors.append(f"candidate {idx} {sequence}: {exc!r}")
                completed_now += 1
                if completed_now == 1 or completed_now % 10 == 0 or completed_now == len(pending):
                    elapsed = time.perf_counter() - start
                    rate = elapsed / completed_now
                    eta = rate * (len(pending) - completed_now)
                    successes = sum(1 for item in existing.values() if item.get("success"))
                    print(f"[{case.case_id}/{stage_name}] {len(existing)}/{len(sequences)}; "
                          f"success={successes}; elapsed={elapsed/3600:.2f}h; ETA={eta/3600:.2f}h", flush=True)
    if errors:
        error_path = stage_dir / "worker_errors.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(errors)} candidate workers failed; see {error_path}")
    if len(existing) != len(sequences):
        raise RuntimeError(f"Incomplete stage: {len(existing)}/{len(sequences)} candidates")
    successful = [item for item in existing.values() if item.get("success")]
    if not successful:
        raise RuntimeError(
            f"{case.case_id}/{stage_name}: zero successful candidates. "
            "No finite partial-trajectory Cmt is valid or reportable."
        )
    best_payload = min(successful, key=lambda item: tuple(float(v) for v in item["rank"]))
    best_sequence = tuple(float(v) for v in best_payload["sequence"])
    result, controller = replay_best(case.case_id, best_sequence, prefix)
    if not result.success:
        raise RuntimeError("Deterministic best-candidate replay did not reproduce success.")

    ordered_payloads = [existing[idx] for idx in range(len(sequences))]
    candidate_rows = [item["row"] for item in ordered_payloads]
    core = _WORKER["core"]
    common = _WORKER["common"]
    core.write_csv(stage_dir / f"{prefix}_all_recovery_candidates.csv", candidate_rows)
    total_elapsed = sum(float(row["row"].get("candidate_elapsed_seconds") or 0.0) for row in ordered_payloads)
    summary_extra = {
        "recovery_search_candidates": len(sequences),
        "successful_candidates": len(successful),
        "best_recovery_sequence": ";".join(f"{v:.6g}" for v in best_sequence),
        "best_recovery_stance1": best_sequence[0],
        "best_recovery_stance2": best_sequence[1],
        "tie_recovery_by_stance": True,
        "summed_candidate_cpu_seconds": total_elapsed,
        "reportable_cmt": float(result.cmt),
    }
    result.extra.update(summary_extra)
    common.write_result_bundle(stage_dir, prefix, result, controller, summary_extra)
    return result, best_sequence


def environment_metadata() -> Dict[str, Any]:
    return {
        "python_executable_name": Path(sys.executable).name,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "thread_limits": {name: os.environ.get(name) for name in
                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }


def write_manifest(case: Case, output_dir: Path, mode: str, workers: int) -> None:
    verify_release_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "unified-mpc-rerun-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "case": asdict(case),
        "targets_deg": list(case.targets_deg),
        "target_dependent_theta1_grid_deg": [case.targets_deg[0], case.targets_deg[2], 30],
        "speed_grid_rad_s": [-2.0, 2.0, 60],
        "theta2_grid_deg": [-60.0, 60.0, 30],
        "initial_state": initial_state(case).tolist(),
        "model_stance1": MODEL_STANCE1,
        "model_stance2": MODEL_STANCE2,
        "mpc_settings": MPC_SETTINGS,
        "mode": mode,
        "candidate_workers": workers,
        "release_config": repo_relative(CONFIG_PATH),
        "pinned_sources": verify_pinned_sources(),
        "case_sources": audit_case(case),
        "environment": environment_metadata(),
        "internal_plan_selection_rule": (
            "Pinned raised-0.01m-L1.145 canonical common.py: heuristic/optimizer plans are selected "
            "by the original cost-only rule. The later flat-L1.145 target-success-first internal "
            "plan selector is intentionally excluded because adopting it would change the MPC algorithm."
        ),
        "outer_recovery_candidate_ranking": (
            "Three-step success first, then Cmt, then energy. This is result acceptance/ranking only "
            "and does not alter the MPC controller or its internal plan selection."
        ),
        "failure_rule": "Cmt is reportable only when success=True; stages with zero successes fail.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def run_case(case: Case, mode: str, output_root: Path, workers: int, resume: bool,
             smoke_recovery: Tuple[float, float]) -> Dict[str, Any]:
    case_output = output_root / mode / case.case_id
    write_manifest(case, case_output, mode, workers)
    worker_initialize(case.case_id)
    start = time.perf_counter()
    if mode == "smoke":
        result, sequence = run_parallel_stage(
            case, "smoke", [smoke_recovery[0]], [smoke_recovery[1]], case_output, workers, resume
        )
    else:
        coarse_values = coarse_grid()
        _, coarse_sequence = run_parallel_stage(
            case, "coarse", coarse_values, coarse_values, case_output, workers, resume
        )
        fine1 = fine_grid(coarse_sequence[0])
        fine2 = fine_grid(coarse_sequence[1])
        result, sequence = run_parallel_stage(
            case, "fine", fine1, fine2, case_output, workers, resume
        )
    elapsed = time.perf_counter() - start
    final = {
        "case_id": case.case_id,
        "success": bool(result.success),
        "reportable_cmt": float(result.cmt) if result.success and np.isfinite(result.cmt) else None,
        "energy_j": float(result.energy_j),
        "distance_m": float(result.distance_m),
        "time_s": float(result.time_s),
        "foot_error_m": float(result.foot_error_m),
        "terminal_reason": result.terminal_reason,
        "best_recovery_sequence": list(sequence),
        "wall_elapsed_seconds": elapsed,
    }
    (case_output / "final_result.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, allow_nan=False), flush=True)
    return final


def audit_all(output_root: Path) -> None:
    report = {
        "release_config": {
            "path": repo_relative(CONFIG_PATH),
            "sha256": sha256(CONFIG_PATH),
            "validated": bool(verify_release_config()),
        },
        "pinned_sources": verify_pinned_sources(),
        "environment": environment_metadata(),
        "cases": {},
    }
    for case in CASES.values():
        report["cases"][case.case_id] = {
            "case_sources": audit_case(case),
            "targets_deg": list(case.targets_deg),
            "initial_idx": list(case.initial_idx),
            "initial_state": initial_state(case).tolist(),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "configuration_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Audit passed for {len(CASES)} cases: {path}")


def launch_all(args) -> int:
    log_dir = args.output_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(CASES)
    running: Dict[str, Tuple[subprocess.Popen, Any]] = {}
    failures: Dict[str, int] = {}
    while pending or running:
        while pending and len(running) < args.case_concurrency:
            case_id = pending.pop(0)
            log_path = log_dir / f"{case_id}.log"
            handle = log_path.open("a" if args.resume else "w", encoding="utf-8")
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--mode", "full", "--case", case_id,
                "--workers", str(args.workers),
                "--output-root", str(args.output_root),
            ]
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[case_id] = (process, handle)
            print(f"Started {case_id}: PID={process.pid}, log={log_path}", flush=True)
        time.sleep(2.0)
        for case_id, (process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[case_id]
            if code != 0:
                failures[case_id] = code
            print(f"Finished {case_id}: exit={code}", flush=True)
    if failures:
        print(f"Failed cases: {failures}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Audit all 12 source cases without simulating.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--case", choices=tuple(CASES), default="raised_L1145_r1")
    parser.add_argument("--all", action="store_true", help="Run all 12 full cases as isolated subprocesses.")
    parser.add_argument("--workers", type=int, default=2, help="Candidate workers per case (recommended: 2).")
    parser.add_argument("--case-concurrency", type=int, default=12,
                        help="Concurrent case coordinators used with --all (recommended: 12 on 24 logical cores).")
    parser.add_argument("--resume", action="store_true", help="Resume from per-candidate JSONL checkpoints.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-r1", type=float, default=-0.95)
    parser.add_argument("--smoke-r2", type=float, default=-1.15)
    args = parser.parse_args()
    if args.workers < 1 or args.case_concurrency < 1:
        parser.error("--workers and --case-concurrency must be positive")
    if args.all and args.mode == "smoke":
        parser.error("--all is reserved for the full protocol; smoke one explicit --case")
    return args


def main() -> int:
    multiprocessing.freeze_support()
    args = parse_args()
    if args.audit:
        audit_all(args.output_root)
        return 0
    if args.all:
        return launch_all(args)
    run_case(CASES[args.case], args.mode, args.output_root, args.workers, args.resume,
             (args.smoke_r1, args.smoke_r2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
