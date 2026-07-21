"""Run PPO-aligned MPC with automatic coarse-to-fine recovery search.

This is a faster alternative to the full 120 x 120 recovery traversal.  It keeps
the PPO-matched convention that one recovery value is shared by the same stance:
for three walking steps the sequence is [r1, r2, r1].

Search stages:
    1. Coarse search over 0, -0.05, ..., -1.15.
    2. Fine search around the best coarse (r1, r2) using 0.01 spacing.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import two_link_mpc_multistep_compare as core
from mpc_ppo_aligned_common import (
    ACTION_MODE_CONTINUOUS,
    ACTION_MODE_DISCRETE_THREE,
    ACTION_MODE_NEGATIVE_CONTINUOUS,
    ACTION_MODE_NEGATIVE_DISCRETE,
    ACTION_MODE_POSITIVE_CONTINUOUS,
    ACTION_MODE_POSITIVE_DISCRETE,
    COST_MODE_CMT,
    COST_MODE_REWARD_MATCHED,
    TARGET_MODE_EITHER,
    TARGET_MODE_PRIMARY,
    PPOAlignedMPCController,
    format_duration,
    result_summary_row,
    run_recovery_grid_search,
)


# =============================================================================
# User-editable settings
# =============================================================================

REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "paper2_mpc_coarse_to_fine"

INITIAL_GRID_INDEX = (19, 11, 15, 1)

# Choose one:
#   continuous, positive_continuous, negative_continuous,
#   discrete_three, positive_discrete, negative_discrete
ACTION_MODE = ACTION_MODE_CONTINUOUS

# Main paper baseline should use COST_MODE_CMT.  COST_MODE_REWARD_MATCHED is an
# optional ablation matching PPO's -action_weight * abs(torque) regularizer.
COST_MODE = COST_MODE_CMT

# Forward multi-step Cmt table usually uses TARGET_MODE_PRIMARY.
TARGET_MODE = TARGET_MODE_PRIMARY

TOTAL_WALKING_STEPS = 3
TIE_RECOVERY_BY_STANCE = True

# PPO full search is 0, -0.01, ..., -1.19.  The coarse-to-fine version first
# scans every 0.05, then refines near the best coarse point.
RECOVERY_MIN = -1.19
RECOVERY_MAX = 0.0
COARSE_STEP = 0.05
FINE_RADIUS = 0.05
FINE_STEP = 0.01

# Optional caps for quick tests.  Set to None for normal runs.
MAX_COARSE_COMBINATIONS = None
MAX_FINE_COMBINATIONS = None

# Progress/ETA printing.
PROGRESS_INTERVAL = 20
PROGRESS_MIN_SECONDS = 600.0

# MPC horizon and optimization settings.
MPC_CONTROL_HOLD_STEPS = 3
MPC_NUM_KNOTS = 55
MPC_MAX_REPLANS_PER_WALKING_STEP = 3
MPC_MAXITER = 180
MPC_FTOL = 1e-5

# Cmt-aligned objective weights.
MPC_TERMINAL_ANGLE_WEIGHT = 420.0
MPC_TERMINAL_VELOCITY_WEIGHT = 0.0
MPC_RUNNING_TARGET_WEIGHT = 0.08
MPC_PREDICTED_WORK_WEIGHT = 1.0
MPC_REWARD_ACTION_WEIGHT = 0.04
MPC_TORQUE_ABS_WEIGHT = 0.004
MPC_TORQUE_SMOOTH_WEIGHT = 0.012
MPC_FALL_SOFT_WEIGHT = 1000.0

# Used only for discrete action modes.
DISCRETE_CEM_SAMPLES = 96
DISCRETE_CEM_ELITE = 16
DISCRETE_CEM_ITERS = 4
RANDOM_SEED = 7


def build_controller() -> PPOAlignedMPCController:
    return PPOAlignedMPCController(
        action_mode=ACTION_MODE,
        cost_mode=COST_MODE,
        target_mode=TARGET_MODE,
        control_hold_steps=MPC_CONTROL_HOLD_STEPS,
        num_knots=MPC_NUM_KNOTS,
        max_replans_per_walking_step=MPC_MAX_REPLANS_PER_WALKING_STEP,
        maxiter=MPC_MAXITER,
        ftol=MPC_FTOL,
        terminal_angle_weight=MPC_TERMINAL_ANGLE_WEIGHT,
        terminal_velocity_weight=MPC_TERMINAL_VELOCITY_WEIGHT,
        running_target_weight=MPC_RUNNING_TARGET_WEIGHT,
        predicted_work_weight=MPC_PREDICTED_WORK_WEIGHT,
        reward_action_weight=MPC_REWARD_ACTION_WEIGHT,
        torque_abs_weight=MPC_TORQUE_ABS_WEIGHT,
        torque_smooth_weight=MPC_TORQUE_SMOOTH_WEIGHT,
        fall_soft_weight=MPC_FALL_SOFT_WEIGHT,
        discrete_cem_samples=DISCRETE_CEM_SAMPLES,
        discrete_cem_elite=DISCRETE_CEM_ELITE,
        discrete_cem_iters=DISCRETE_CEM_ITERS,
        random_seed=RANDOM_SEED,
    )


def recovery_grid(start: float, stop: float, step: float) -> np.ndarray:
    lo = max(float(RECOVERY_MIN), min(float(start), float(stop)))
    hi = min(float(RECOVERY_MAX), max(float(start), float(stop)))
    count = int(np.floor((hi - lo) / float(step) + 0.5)) + 1
    values = lo + np.arange(count, dtype=float) * float(step)
    values = np.clip(values, RECOVERY_MIN, RECOVERY_MAX)
    return np.unique(np.round(values, 10))


def coarse_recovery_grid() -> np.ndarray:
    stride = max(1, int(round(float(COARSE_STEP) / 0.01)))
    values = -np.arange(0, 120, stride, dtype=float) * 0.01
    return np.unique(np.round(np.clip(values, RECOVERY_MIN, RECOVERY_MAX), 10))


def best_stance_recovery(result) -> Tuple[float, float]:
    seq = result.extra.get("recovery_sequence", "")
    if seq:
        parts = [float(item) for item in str(seq).split(";") if item != ""]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    best_seq = result.extra.get("best_recovery_sequence", "")
    parts = [float(item) for item in str(best_seq).split(";") if item != ""]
    if len(parts) < 2:
        raise RuntimeError("Could not infer best recovery pair from coarse result.")
    return float(parts[0]), float(parts[1])


def run_stage(
    stage_name: str,
    grid_by_stance: Dict[int, Sequence[float]],
    initial_state: np.ndarray,
    max_combinations,
):
    prefix = f"mpc_{stage_name}_recovery_{ACTION_MODE}_{COST_MODE}"
    stage_dir = OUTPUT_DIR / stage_name
    print("")
    print(f"=== {stage_name.upper()} RECOVERY SEARCH ===")
    print(f"stance 1 grid size={len(grid_by_stance[1])}, range=({min(grid_by_stance[1]):.3f}, {max(grid_by_stance[1]):.3f})")
    print(f"stance 2 grid size={len(grid_by_stance[2])}, range=({min(grid_by_stance[2]):.3f}, {max(grid_by_stance[2]):.3f})")
    return run_recovery_grid_search(
        controller_factory=build_controller,
        output_dir=stage_dir,
        prefix=prefix,
        recovery_grid_by_stance=grid_by_stance,
        initial_state=initial_state,
        target_mode=TARGET_MODE,
        total_steps=TOTAL_WALKING_STEPS,
        max_combinations=max_combinations,
        tie_recovery_by_stance=TIE_RECOVERY_BY_STANCE,
        progress_interval=PROGRESS_INTERVAL,
        progress_min_seconds=PROGRESS_MIN_SECONDS,
    )


def main():
    start_time = time.perf_counter()
    initial_state = core.initial_state_from_grid(INITIAL_GRID_INDEX)
    coarse_values = coarse_recovery_grid()

    coarse_grid = {
        1: coarse_values,
        2: coarse_values,
    }
    coarse_result = run_stage("coarse", coarse_grid, initial_state, MAX_COARSE_COMBINATIONS)
    coarse_r1, coarse_r2 = best_stance_recovery(coarse_result)

    fine_grid = {
        1: recovery_grid(coarse_r1 - FINE_RADIUS, coarse_r1 + FINE_RADIUS, FINE_STEP),
        2: recovery_grid(coarse_r2 - FINE_RADIUS, coarse_r2 + FINE_RADIUS, FINE_STEP),
    }
    fine_result = run_stage("fine", fine_grid, initial_state, MAX_FINE_COMBINATIONS)

    elapsed = time.perf_counter() - start_time
    print("")
    print("Coarse-to-fine recovery MPC final result:")
    print(result_summary_row(fine_result))
    print(f"Coarse best recovery: stance1={coarse_r1:.4f}, stance2={coarse_r2:.4f}")
    print(f"Elapsed time: {format_duration(elapsed)} ({elapsed:.2f} s)")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
