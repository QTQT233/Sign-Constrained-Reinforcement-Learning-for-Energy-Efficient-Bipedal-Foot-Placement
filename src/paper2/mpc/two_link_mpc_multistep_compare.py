"""Three-step NMPC comparison for the two-link Table-V model.

This script is self-contained and deliberately does not import the original
Paper2 scripts, because those files execute simulations at import time.  The
model, collision map, push-off recovery and Cmt accounting are copied into
functions so MPC can be compared with PPO/LQR/LIPM under the same physical
accounting.

Main output:
    two_link_mpc_outputs/mpc_three_step_summary.csv
    two_link_mpc_outputs/mpc_three_step_trace.csv
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


# =============================================================================
# User-editable settings
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "results" / "paper2_mpc_single_case"

# Same initial state used by Multi_continuous.py / LQR.py / LIPM.py in this case.
INITIAL_GRID_INDEX = (19, 11, 15, 1)  # theta1, dtheta1, theta2, dtheta2

TOTAL_WALKING_STEPS = 3
MAX_INNER_STEPS_PER_WALKING_STEP = 360

# MPC is solved once per walking step and re-planned if the previous plan is
# exhausted before touchdown.
MPC_CONTROL_HOLD_STEPS = 3
MPC_NUM_KNOTS = 55
MPC_MAX_REPLANS_PER_WALKING_STEP = 3
MPC_MAXITER = 180
MPC_FTOL = 1e-5

MPC_TERMINAL_ANGLE_WEIGHT = 320.0
MPC_TERMINAL_VELOCITY_WEIGHT = 3.0
MPC_RUNNING_TARGET_WEIGHT = 0.2
MPC_TORQUE_WEIGHT = 0.012
MPC_SMOOTH_WEIGHT = 0.018
MPC_FALL_SOFT_WEIGHT = 800.0
MPC_SUCCESS_BONUS_WEIGHT = 0.0

# Use the same Table-V active push-off recovery as Multi_continuous.py.
RECOVERY_DTHETA1_BY_STANCE = {1: -1.06, 2: -0.68}


# =============================================================================
# Two-link model parameters, copied from the 60,30(0.33m),1.145 scripts
# =============================================================================

BASE_PARAMS = {
    "g": 9.8,
    "dt": 0.01,
    "max_torque": 4.0,
    'target1': 61.8,
    'target2': 31.7,
    'target3': 118.2,
    'target4': -31.7,
    "theta1_range": 90.0,
    "theta2_range": 90.0,
    "speed_range": 2.0,
    "settle": 5.0,
    "reward_scale": 5.0,
    "action_value_weight": 0.03,
}

PARAMS1 = BASE_PARAMS.copy()
PARAMS1.update(
    {
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
)

PARAMS2 = BASE_PARAMS.copy()
PARAMS2.update(
    {
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
)

W = (PARAMS1["m1"] + PARAMS1["m2"]) * PARAMS1["g"]


def params_for_stance(stance_leg: int) -> Dict[str, float]:
    return PARAMS1 if int(stance_leg) == 1 else PARAMS2


def initial_state_from_grid(index: Sequence[int] = INITIAL_GRID_INDEX) -> np.ndarray:
    n1, n2 = 30, 60
    # Match the PPO/LIPM/TVLQR entry points exactly: their theta1 grid is
    # constructed from the case-specific primary and secondary targets,
    # rather than from a generic 60--120 degree range.
    tht1s = np.linspace(PARAMS1["target1"], PARAMS1["target3"], n1) * np.pi / 180.0
    dtht1s = np.linspace(-2.0, 2.0, n2)
    tht2s = np.linspace(-60.0, 60.0, n1) * np.pi / 180.0
    dtht2s = np.linspace(-2.0, 2.0, n2)
    i, j, k, ll = [int(v) for v in index]
    return np.array([tht1s[i], dtht1s[j], tht2s[k], dtht2s[ll]], dtype=float)


def target_state(param: Dict[str, float], primary: bool = True) -> np.ndarray:
    if primary:
        return np.array([np.deg2rad(param["target1"]), np.deg2rad(param["target2"])], dtype=float)
    return np.array([np.deg2rad(param["target3"]), np.deg2rad(param["target4"])], dtype=float)


def angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def com_x(theta1: float, theta2: float, env: "TwoLinkEnv") -> float:
    return float(
        (
            env.m1 * env.l1_s * np.cos(theta1)
            + env.m2 * (env.l1 * np.cos(theta1) + env.l2_s * np.sin(theta2))
        )
        / (env.m1 + env.m2)
    )


def foot_distance(theta1: float, theta2: float, env: "TwoLinkEnv") -> float:
    return float(env.l1 * (np.cos(theta1) + np.sin(theta2)))


def collision_dynamics_full(theta1: float, theta2: float, dtheta1_pre: float, dtheta2_pre: float,
                            params_pre: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    m1, m2 = params_pre["m1"], params_pre["m2"]
    l1, l2 = params_pre["l1"], params_pre["l2"]
    l1_prime, l2_prime = params_pre["l1_prime"], params_pre["l2_prime"]
    j1_root, j2_cm = params_pre["J1"], params_pre["J2"]
    j1_cm = j1_root - m1 * l1_prime ** 2
    qdot_minus = np.array([0.0, 0.0, dtheta1_pre, dtheta2_pre], dtype=float)
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    mass = np.zeros((4, 4), dtype=float)
    mass[0, 0] = m1 + m2
    mass[1, 1] = m1 + m2
    mass[0, 2] = -m1 * l1_prime * s1 - m2 * l1 * s1
    mass[1, 2] = m1 * l1_prime * c1 + m2 * l1 * c1
    mass[0, 3] = m2 * l2_prime * c2
    mass[1, 3] = m2 * l2_prime * s2
    mass[2, 2] = m1 * l1_prime ** 2 + j1_cm + m2 * (l1 ** 2)
    mass[3, 3] = m2 * l2_prime ** 2 + j2_cm
    mass[2, 3] = m2 * l1 * l2_prime * np.sin(theta2 - theta1)
    mass[3, 2] = mass[2, 3]
    mass[1, 0] = mass[0, 1]
    mass[2, 0] = mass[0, 2]
    mass[2, 1] = mass[1, 2]
    mass[3, 0] = mass[0, 3]
    mass[3, 1] = mass[1, 3]

    jac = np.array([[1.0, 0.0, -l1 * s1, l2 * c2], [0.0, 1.0, l1 * c1, l2 * s2]], dtype=float)
    a_mat = np.zeros((6, 6), dtype=float)
    a_mat[0:4, 0:4] = mass
    a_mat[0:4, 4:6] = -jac.T
    a_mat[4:6, 0:4] = jac
    b_vec = np.zeros(6, dtype=float)
    b_vec[0:4] = mass @ qdot_minus
    try:
        sol = np.linalg.inv(a_mat) @ b_vec
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(a_mat) @ b_vec

    theta1_new = np.arctan2(l2 * c2, -l2 * s2)
    theta2_new = theta1 - np.pi / 2.0
    dtheta1_new = sol[3]
    dtheta2_new = sol[2]
    return theta1_new, theta2_new, dtheta1_new, dtheta2_new, sol[0], sol[1]


def kinetic_energy(theta1: float, theta2: float, dtheta1: float, dtheta2: float,
                   params_fun: Dict[str, float]) -> float:
    m2 = params_fun["m2"]
    l1 = params_fun["l1"]
    l2_prime = params_fun["l2_prime"]
    j1 = params_fun["J1"]
    j2 = params_fun["J2"]
    t_rot1 = 0.5 * j1 * dtheta1 ** 2
    v2_sq = (
        (dtheta1 * l1) ** 2
        + (dtheta2 * l2_prime) ** 2
        + 2.0 * dtheta1 * l1 * dtheta2 * l2_prime * np.sin(theta2 - theta1)
    )
    t_trans2 = 0.5 * m2 * v2_sq
    t_rot2 = 0.5 * j2 * dtheta2 ** 2
    return float(t_rot1 + t_trans2 + t_rot2)


def handle_collision_with_recovery(state: np.ndarray, stance_leg: int, params_fun: Dict[str, float],
                                   energy: float,
                                   recovery_map: Dict[int, float] = RECOVERY_DTHETA1_BY_STANCE
                                   ) -> Tuple[np.ndarray, float, float]:
    theta1_new, theta2_new, dtheta1_new, dtheta2_new, _, _ = collision_dynamics_full(
        state[0], state[2], state[1], state[3], params_fun
    )
    energy_pre = kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params_fun)
    dtheta1_new += float(recovery_map[int(stance_leg)])
    energy_new = kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params_fun)
    delta = float(energy_new - energy_pre)
    return np.array([theta1_new, dtheta1_new, theta2_new, dtheta2_new], dtype=float), energy + delta, delta


class TwoLinkEnv:
    def __init__(self, param: Dict[str, float]):
        self.param = dict(param)
        self.max_torque = float(param["max_torque"])
        self.dt = float(param["dt"])
        self.m1 = float(param["m1"])
        self.m2 = float(param["m2"])
        self.l1 = float(param["l1"])
        self.l2 = float(param["l2"])
        self.l1_s = float(param["l1_prime"])
        self.l2_s = float(param["l2_prime"])
        self.J1 = float(param["J1"])
        self.J2 = float(param["J2"])
        self.c1 = float(param["c1"])
        self.c2 = float(param["c2"])
        self.g = float(param.get("g", 9.8))
        self.settle = np.deg2rad(float(param["settle"]))
        self.settle2 = self.settle / 5.0
        self.target1_rad = np.deg2rad(float(param["target1"]))
        self.target2_rad = np.deg2rad(float(param["target2"]))
        self.target3_rad = np.deg2rad(float(param["target3"]))
        self.target4_rad = np.deg2rad(float(param["target4"]))
        self.rad_theta1_range = np.deg2rad(float(param["theta1_range"]))
        self.rad_theta2_range = np.deg2rad(float(param["theta2_range"]))
        self.speed_range = float(param["speed_range"])
        self.state = np.zeros(4, dtype=float)
        self.next_state = np.zeros(4, dtype=float)
        self.reward = 0.0
        self.over = False
        self.terminal_reason = "running"

    def define(self, state: Sequence[float]) -> np.ndarray:
        self.state = np.array(state, dtype=float)
        self.next_state = self.state.copy()
        self.over = False
        self.terminal_reason = "running"
        return self.state

    def step(self, action_value: float) -> Tuple[np.ndarray, float, bool]:
        y = self.state.copy()
        action_value = float(np.clip(action_value, -self.max_torque, self.max_torque))
        a_mat = np.array(
            [
                [self.J1 + self.m2 * self.l1 ** 2, self.m2 * self.l1 * self.l2_s * np.sin(y[2] - y[0])],
                [self.m2 * self.l1 * self.l2_s * np.sin(y[2] - y[0]), self.J2 + self.m2 * self.l2_s ** 2],
            ],
            dtype=float,
        )
        b_vec = np.array(
            [
                [
                    -self.m2 * y[1] * self.l1 * y[3] * self.l2_s * np.cos(y[2] - y[0])
                    - self.m1 * self.g * self.l1_s * np.cos(y[0])
                    - self.m2 * self.g * self.l1 * np.cos(y[0])
                    - self.m2 * self.l1 * self.l2_s * y[3] * np.cos(y[2] - y[0]) * (y[3] - y[1])
                ],
                [
                    -self.m2 * self.l1 * self.l2_s * y[1] * np.cos(y[2] - y[0]) * (y[3] - y[1])
                    + self.m2 * y[1] * self.l1 * y[3] * self.l2_s * np.cos(y[2] - y[0])
                    - self.m2 * self.g * self.l2_s * np.sin(y[2])
                ],
            ],
            dtype=float,
        )
        tau_vec = np.array([[-action_value], [action_value]], dtype=float)
        rhs = (b_vec + tau_vec).reshape(2)
        det = a_mat[0, 0] * a_mat[1, 1] - a_mat[0, 1] * a_mat[1, 0]
        if abs(det) < 1e-12:
            ddq = (np.linalg.pinv(a_mat) @ rhs.reshape(2, 1)).reshape(2)
        else:
            ddq = np.array(
                [
                    (rhs[0] * a_mat[1, 1] - a_mat[0, 1] * rhs[1]) / det,
                    (a_mat[0, 0] * rhs[1] - rhs[0] * a_mat[1, 0]) / det,
                ],
                dtype=float,
            )
        y[1] += (ddq[0] - self.c1 * y[1]) * self.dt
        y[0] += y[1] * self.dt
        y[3] += (ddq[1] - self.c2 * y[3]) * self.dt
        y[2] += y[3] * self.dt

        primary = self.primary_target_success(y)
        secondary = self.secondary_target_success(y)
        fall = self.fall_condition(y)
        if primary:
            self.reward = self.param["reward_scale"] * 3.0
            self.over = True
            self.terminal_reason = "primary_target"
        elif secondary:
            self.reward = self.param["reward_scale"] * 2.0
            self.over = True
            self.terminal_reason = "secondary_target"
        elif fall:
            self.reward = -self.param["reward_scale"] * 5.0
            self.over = True
            self.terminal_reason = "fall"
        else:
            self.reward = 0.0
            self.over = False
            self.terminal_reason = "running"
        self.reward += -abs(action_value) * self.param["action_value_weight"]
        self.next_state = y
        self.state = y.copy()
        return self.next_state.copy(), float(self.reward), bool(self.over)

    def primary_target_success(self, state: np.ndarray) -> bool:
        return bool(abs(state[0] - self.target1_rad) < self.settle and abs(state[2] - self.target2_rad) < self.settle)

    def secondary_target_success(self, state: np.ndarray) -> bool:
        return bool(abs(state[0] - self.target3_rad) < self.settle2 and abs(state[2] - self.target4_rad) < self.settle2)

    def fall_condition(self, state: np.ndarray) -> bool:
        return bool(abs(state[0] - np.pi / 2.0) > self.rad_theta1_range or abs(state[2]) > self.rad_theta2_range)


@dataclass
class TraceRow:
    method: str
    global_step_index: int
    time_s: float
    walking_step_index: int
    stance_leg: int
    theta1_rad: float
    dtheta1_rad_s: float
    theta2_rad: float
    dtheta2_rad_s: float
    torque_nm: float
    positive_work_j: float
    collision_recovery_j: float
    cumulative_energy_j: float
    cumulative_distance_m: float
    cumulative_cmt: float
    terminal_reason: str


@dataclass
class SimulationResult:
    method: str
    success: bool
    cmt: float
    energy_j: float
    distance_m: float
    time_s: float
    foot_error_m: float
    steps: int
    terminal_reason: str
    trace: List[TraceRow]
    extra: Dict[str, float]


class ControllerBase:
    name = "controller"

    def begin_walking_step(self, walking_step_index: int, stance_leg: int, state: np.ndarray,
                           env: TwoLinkEnv) -> None:
        return None

    def action(self, state: np.ndarray, env: TwoLinkEnv, walking_step_index: int, stance_leg: int) -> float:
        raise NotImplementedError


class PIDSeedController(ControllerBase):
    name = "pid_seed"

    def __init__(self, kp: float = 18.0, kd: float = 2.0):
        self.kp = float(kp)
        self.kd = float(kd)

    def action(self, state: np.ndarray, env: TwoLinkEnv, walking_step_index: int, stance_leg: int) -> float:
        tgt = target_state(env.param, primary=True)
        err = angle_error(state[[0, 2]], tgt)
        rel_err = err[1] - err[0]
        rel_vel = state[3] - state[1]
        return float(np.clip(self.kp * rel_err - self.kd * rel_vel, -env.max_torque, env.max_torque))


class TwoLinkMPCController(ControllerBase):
    name = "mpc"

    def __init__(self):
        self.plan: List[float] = []
        self.solve_log: List[Dict[str, float]] = []
        self.replans_this_step = 0
        self.last_solution: Optional[np.ndarray] = None

    def begin_walking_step(self, walking_step_index: int, stance_leg: int, state: np.ndarray,
                           env: TwoLinkEnv) -> None:
        self.plan = []
        self.replans_this_step = 0
        self.last_solution = None

    def action(self, state: np.ndarray, env: TwoLinkEnv, walking_step_index: int, stance_leg: int) -> float:
        if not self.plan:
            if self.replans_this_step >= MPC_MAX_REPLANS_PER_WALKING_STEP:
                return PIDSeedController().action(state, env, walking_step_index, stance_leg)
            self.plan = self.solve_plan(state, env, walking_step_index, stance_leg)
            self.replans_this_step += 1
        return float(self.plan.pop(0))

    def rollout_open_loop(self, state: np.ndarray, env: TwoLinkEnv, knot_torques: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        sim_env = TwoLinkEnv(env.param)
        sim_env.define(state)
        expanded = np.repeat(np.asarray(knot_torques, dtype=float), MPC_CONTROL_HOLD_STEPS)
        states = [state.copy()]
        reasons = []
        for torque in expanded:
            next_state, _, done = sim_env.step(float(torque))
            states.append(next_state.copy())
            reasons.append(sim_env.terminal_reason)
            if done:
                break
        return np.asarray(states), expanded[: max(0, len(states) - 1)], reasons

    def seed_guess(self, state: np.ndarray, env: TwoLinkEnv) -> np.ndarray:
        pid = PIDSeedController()
        sim_env = TwoLinkEnv(env.param)
        sim_env.define(state)
        torques = []
        current = state.copy()
        for _ in range(MPC_NUM_KNOTS):
            torque = pid.action(current, sim_env, 0, 1)
            torques.append(torque)
            for _ in range(MPC_CONTROL_HOLD_STEPS):
                current, _, done = sim_env.step(torque)
                if done:
                    break
            if done:
                torques.extend([0.0] * (MPC_NUM_KNOTS - len(torques)))
                break
        return np.asarray(torques[:MPC_NUM_KNOTS], dtype=float)

    def cost(self, knot_torques: np.ndarray, state: np.ndarray, env: TwoLinkEnv) -> float:
        states, expanded, _ = self.rollout_open_loop(state, env, knot_torques)
        q = states[:, [0, 2]]
        dq = states[:, [1, 3]]
        tgt = target_state(env.param, primary=True)
        terminal_err = angle_error(q[-1], tgt)
        running_err = angle_error(q, tgt)
        fall_violation = np.maximum(np.abs(states[:, 0] - np.pi / 2.0) - env.rad_theta1_range, 0.0)
        fall_violation += np.maximum(np.abs(states[:, 2]) - env.rad_theta2_range, 0.0)
        torque_smooth = np.diff(knot_torques) if len(knot_torques) > 1 else np.zeros(0)
        success_bonus = 0.0
        if env.primary_target_success(states[-1]):
            success_bonus = -MPC_SUCCESS_BONUS_WEIGHT
        return float(
            MPC_TERMINAL_ANGLE_WEIGHT * np.sum(terminal_err ** 2)
            + MPC_TERMINAL_VELOCITY_WEIGHT * np.sum(dq[-1] ** 2)
            + MPC_RUNNING_TARGET_WEIGHT * np.sum(running_err ** 2)
            + MPC_TORQUE_WEIGHT * np.sum(np.asarray(knot_torques) ** 2)
            + MPC_SMOOTH_WEIGHT * np.sum(torque_smooth ** 2)
            + MPC_FALL_SOFT_WEIGHT * np.sum(fall_violation ** 2)
            + success_bonus
        )

    def final_target_constraint(self, knot_torques: np.ndarray, state: np.ndarray, env: TwoLinkEnv) -> float:
        states, _, _ = self.rollout_open_loop(state, env, knot_torques)
        final = states[-1]
        err = np.abs(angle_error(final[[0, 2]], target_state(env.param, primary=True)))
        return float(env.settle - np.max(err))

    def state_limit_constraint(self, knot_torques: np.ndarray, state: np.ndarray, env: TwoLinkEnv) -> float:
        states, _, _ = self.rollout_open_loop(state, env, knot_torques)
        theta1_margin = env.rad_theta1_range - np.abs(states[:, 0] - np.pi / 2.0)
        theta2_margin = env.rad_theta2_range - np.abs(states[:, 2])
        return float(np.min(np.concatenate([theta1_margin, theta2_margin])))

    def solve_plan(self, state: np.ndarray, env: TwoLinkEnv, walking_step_index: int, stance_leg: int) -> List[float]:
        guess = self.seed_guess(state, env)
        if self.last_solution is not None and len(self.last_solution) == MPC_NUM_KNOTS:
            guess = np.r_[self.last_solution[1:], self.last_solution[-1]]
        bounds = [(-env.max_torque, env.max_torque)] * MPC_NUM_KNOTS
        constraints = [
            {"type": "ineq", "fun": lambda u, s=state.copy(), e=env: self.state_limit_constraint(u, s, e)},
            {"type": "ineq", "fun": lambda u, s=state.copy(), e=env: self.final_target_constraint(u, s, e)},
        ]
        result = minimize(
            lambda u, s=state.copy(), e=env: self.cost(u, s, e),
            guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": MPC_MAXITER, "ftol": MPC_FTOL, "disp": False},
        )
        candidates = [guess]
        if result.x is not None:
            candidates.append(np.asarray(result.x, dtype=float))
        zero = np.zeros(MPC_NUM_KNOTS, dtype=float)
        candidates.append(zero)
        best = min(candidates, key=lambda u: self.cost(u, state, env))
        self.last_solution = best.copy()
        states, expanded, reasons = self.rollout_open_loop(state, env, best)
        self.solve_log.append(
            {
                "walking_step_index": float(walking_step_index),
                "stance_leg": float(stance_leg),
                "optimizer_success": float(bool(result.success)),
                "optimizer_cost": float(result.fun) if np.isfinite(result.fun) else float("nan"),
                "selected_cost": float(self.cost(best, state, env)),
                "predicted_steps": float(len(expanded)),
                "predicted_final_reason": str(reasons[-1] if reasons else "empty"),
            }
        )
        return [float(v) for v in expanded]


def positive_work(torque: float, state: np.ndarray, dt: float) -> float:
    rel_vel = float(state[3] - state[1])
    if torque * rel_vel > 0.0:
        return float(abs(torque) * abs(rel_vel) * dt)
    return 0.0


def run_three_step_simulation(controller: ControllerBase,
                              method_name: Optional[str] = None,
                              initial_state: Optional[np.ndarray] = None,
                              recovery_map: Dict[int, float] = RECOVERY_DTHETA1_BY_STANCE,
                              total_steps: int = TOTAL_WALKING_STEPS) -> SimulationResult:
    method = method_name or controller.name
    state = initial_state_from_grid() if initial_state is None else np.array(initial_state, dtype=float)
    stance_leg = 1
    walking_step = 0
    global_step = 0
    energy = 0.0
    distance = 0.0
    foot_d = 0.0
    initial_theta1 = float(state[0])
    initial_theta2 = float(state[2])
    trace: List[TraceRow] = []
    terminal_reason = "not_started"

    while walking_step < total_steps:
        env = TwoLinkEnv(params_for_stance(stance_leg))
        env.define(state)
        controller.begin_walking_step(walking_step + 1, stance_leg, state.copy(), env)
        step_start_theta1 = initial_theta1
        step_start_theta2 = initial_theta2
        terminal_reason = "max_inner_steps"
        collision_delta = 0.0

        for _ in range(MAX_INNER_STEPS_PER_WALKING_STEP):
            old_state = env.state.copy()
            torque = float(np.clip(controller.action(old_state.copy(), env, walking_step + 1, stance_leg),
                                   -env.max_torque, env.max_torque))
            work = positive_work(torque, old_state, env.dt)
            energy += work
            next_state, _, done = env.step(torque)
            current_center = com_x(next_state[0], next_state[2], env)
            start_center = com_x(step_start_theta1, step_start_theta2, env)
            running_distance = distance + abs(current_center - start_center)
            cmt = energy / (W * running_distance) if running_distance > 1e-12 else float("nan")
            trace.append(
                TraceRow(
                    method=method,
                    global_step_index=global_step,
                    time_s=global_step * env.dt,
                    walking_step_index=walking_step + 1,
                    stance_leg=stance_leg,
                    theta1_rad=float(next_state[0]),
                    dtheta1_rad_s=float(next_state[1]),
                    theta2_rad=float(next_state[2]),
                    dtheta2_rad_s=float(next_state[3]),
                    torque_nm=torque,
                    positive_work_j=work,
                    collision_recovery_j=0.0,
                    cumulative_energy_j=energy,
                    cumulative_distance_m=running_distance,
                    cumulative_cmt=cmt,
                    terminal_reason=env.terminal_reason,
                )
            )
            global_step += 1
            if done:
                terminal_reason = env.terminal_reason
                break

        state = env.state.copy()
        if not env.primary_target_success(state):
            break

        final_theta1 = float(state[0])
        final_theta2 = float(state[2])
        initial_center = com_x(initial_theta1, initial_theta2, env)
        final_center = com_x(final_theta1, final_theta2, env)
        distance += abs(final_center - initial_center)
        foot_d += foot_distance(final_theta1, final_theta2, env)

        state, energy, collision_delta = handle_collision_with_recovery(state, stance_leg, env.param, energy, recovery_map)
        initial_theta1 = float(state[0])
        initial_theta2 = float(state[2])
        if trace:
            trace[-1].collision_recovery_j = collision_delta
            trace[-1].cumulative_energy_j = energy
            trace[-1].cumulative_distance_m = distance
            trace[-1].cumulative_cmt = energy / (W * distance) if distance > 1e-12 else float("nan")
            trace[-1].terminal_reason = "collision_recovery"
        stance_leg = 2 if stance_leg == 1 else 1
        walking_step += 1

    success = walking_step >= total_steps
    # Manuscript convention: failed three-step cases are not Cmt-valid.
    # Keep their energy and distance as diagnostics, but never expose a
    # partial-trajectory Cmt that could be mistaken for a completed result.
    cmt = energy / (W * distance) if success and distance > 1e-12 else float("nan")
    foot_error = total_steps * PARAMS1["l1"] - foot_d
    extra: Dict[str, float] = {}
    if isinstance(controller, TwoLinkMPCController):
        extra["mpc_num_solves"] = float(len(controller.solve_log))
        extra["mpc_failed_solves"] = float(sum(1 for item in controller.solve_log if item["optimizer_success"] < 0.5))
    return SimulationResult(
        method=method,
        success=success,
        cmt=float(cmt),
        energy_j=float(energy),
        distance_m=float(distance),
        time_s=float(global_step * PARAMS1["dt"]),
        foot_error_m=float(foot_error),
        steps=int(global_step),
        terminal_reason=terminal_reason,
        trace=trace,
        extra=extra,
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def result_summary_row(result: SimulationResult) -> Dict[str, object]:
    row = {
        "method": result.method,
        "success": result.success,
        "cmt": result.cmt,
        "energy_j": result.energy_j,
        "distance_m": result.distance_m,
        "time_s": result.time_s,
        "foot_error_m": result.foot_error_m,
        "steps": result.steps,
        "terminal_reason": result.terminal_reason,
    }
    row.update(result.extra)
    return row


def run_mpc_only() -> SimulationResult:
    controller = TwoLinkMPCController()
    result = run_three_step_simulation(controller, method_name="MPC_NMPC")
    summary_rows = [result_summary_row(result)]
    trace_rows = [asdict(row) for row in result.trace]
    write_csv(OUTPUT_DIR / "mpc_three_step_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "mpc_three_step_trace.csv", trace_rows)
    if controller.solve_log:
        write_csv(OUTPUT_DIR / "mpc_solve_log.csv", controller.solve_log)
    print("MPC result:")
    print(summary_rows[0])
    print(f"Wrote outputs to {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    run_mpc_only()
