"""PPO-aligned MPC utilities for the two-link walking comparison.

This module extends ``two_link_mpc_multistep_compare.py`` without modifying the
original file.  It keeps the same multi-step physical parameters and Cmt
accounting, while adding:

* action spaces matched to PPO variants;
* Cmt-aligned or reward-matched MPC objective;
* PPO-style target success check;
* fixed or searched push-off recovery.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

import two_link_mpc_multistep_compare as core


# =============================================================================
# Defaults. Entry scripts override these values at the top of their files.
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "paper2_mpc_ppo_aligned"

ACTION_MODE_CONTINUOUS = "continuous"  # tau in [-T, +T]
ACTION_MODE_POSITIVE_CONTINUOUS = "positive_continuous"  # tau in [0, +T]
ACTION_MODE_NEGATIVE_CONTINUOUS = "negative_continuous"  # tau in [-T, 0]
ACTION_MODE_DISCRETE_THREE = "discrete_three"  # tau in {-T, 0, +T}
ACTION_MODE_POSITIVE_DISCRETE = "positive_discrete"  # tau in {0, +T}
ACTION_MODE_NEGATIVE_DISCRETE = "negative_discrete"  # tau in {-T, 0}

COST_MODE_CMT = "cmt"
COST_MODE_REWARD_MATCHED = "reward_matched"

TARGET_MODE_PRIMARY = "primary"
TARGET_MODE_EITHER = "either"


def is_discrete_action_mode(action_mode: str) -> bool:
    return action_mode in {
        ACTION_MODE_DISCRETE_THREE,
        ACTION_MODE_POSITIVE_DISCRETE,
        ACTION_MODE_NEGATIVE_DISCRETE,
    }


def continuous_bounds_for_mode(action_mode: str, max_torque: float) -> Tuple[float, float]:
    if action_mode == ACTION_MODE_CONTINUOUS:
        return -max_torque, max_torque
    if action_mode == ACTION_MODE_POSITIVE_CONTINUOUS:
        return 0.0, max_torque
    if action_mode == ACTION_MODE_NEGATIVE_CONTINUOUS:
        return -max_torque, 0.0
    raise ValueError(f"{action_mode!r} is not a continuous action mode")


def discrete_values_for_mode(action_mode: str, max_torque: float) -> np.ndarray:
    if action_mode == ACTION_MODE_DISCRETE_THREE:
        return np.array([-max_torque, 0.0, max_torque], dtype=float)
    if action_mode == ACTION_MODE_POSITIVE_DISCRETE:
        return np.array([0.0, max_torque], dtype=float)
    if action_mode == ACTION_MODE_NEGATIVE_DISCRETE:
        return np.array([-max_torque, 0.0], dtype=float)
    raise ValueError(f"{action_mode!r} is not a discrete action mode")


def quantize_actions(values: np.ndarray, action_set: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    action_set = np.asarray(action_set, dtype=float)
    idx = np.argmin(np.abs(values.reshape(-1, 1) - action_set.reshape(1, -1)), axis=1)
    return action_set[idx]


def clip_or_quantize(values: np.ndarray, action_mode: str, max_torque: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if is_discrete_action_mode(action_mode):
        return quantize_actions(values, discrete_values_for_mode(action_mode, max_torque))
    lo, hi = continuous_bounds_for_mode(action_mode, max_torque)
    return np.clip(values, lo, hi)


def target_error_candidates(q: np.ndarray, env: core.TwoLinkEnv, target_mode: str) -> List[np.ndarray]:
    q = np.asarray(q, dtype=float)
    candidates = [core.angle_error(q, core.target_state(env.param, primary=True))]
    if target_mode == TARGET_MODE_EITHER:
        candidates.append(core.angle_error(q, core.target_state(env.param, primary=False)))
    elif target_mode != TARGET_MODE_PRIMARY:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    return candidates


def target_error_sq(q: np.ndarray, env: core.TwoLinkEnv, target_mode: str) -> float:
    return float(min(np.sum(err ** 2) for err in target_error_candidates(q, env, target_mode)))


def target_margin(q: np.ndarray, env: core.TwoLinkEnv, target_mode: str) -> float:
    best_max_abs_err = min(np.max(np.abs(err)) for err in target_error_candidates(q, env, target_mode))
    return float(env.settle - best_max_abs_err)


def ppo_style_target_success(state: np.ndarray, env: core.TwoLinkEnv, target_mode: str) -> bool:
    q = np.asarray(state, dtype=float)[[0, 2]]
    return target_margin(q, env, target_mode) > 0.0


def ppo_style_target_reason(state: np.ndarray, env: core.TwoLinkEnv, target_mode: str) -> str:
    q = np.asarray(state, dtype=float)[[0, 2]]
    primary_margin = float(env.settle - np.max(np.abs(core.angle_error(q, core.target_state(env.param, True)))))
    if target_mode == TARGET_MODE_PRIMARY:
        return "primary_target" if primary_margin > 0.0 else "running"
    secondary_margin = float(env.settle - np.max(np.abs(core.angle_error(q, core.target_state(env.param, False)))))
    if primary_margin >= secondary_margin and primary_margin > 0.0:
        return "primary_target"
    if secondary_margin > 0.0:
        return "secondary_target"
    return "running"


def predicted_positive_work(states: np.ndarray, expanded_torques: np.ndarray, dt: float) -> float:
    n = min(len(expanded_torques), max(0, len(states) - 1))
    work = 0.0
    for i in range(n):
        work += core.positive_work(float(expanded_torques[i]), states[i], dt)
    return float(work)


def solve_linear_system_gauss(a_mat: np.ndarray, b_vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Small dense linear solve without numpy.linalg, avoiding local LAPACK crashes."""
    a = np.asarray(a_mat, dtype=float).copy()
    b = np.asarray(b_vec, dtype=float).copy()
    n = int(b.shape[0])
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(a[col:, col])))
        if pivot != col:
            a[[col, pivot], :] = a[[pivot, col], :]
            b[[col, pivot]] = b[[pivot, col]]
        if abs(a[col, col]) < eps:
            a[col, col] = eps if a[col, col] >= 0.0 else -eps
        pivot_value = a[col, col]
        a[col, :] /= pivot_value
        b[col] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = a[row, col]
            if factor == 0.0:
                continue
            a[row, :] -= factor * a[col, :]
            b[row] -= factor * b[col]
    return b


def collision_dynamics_full_safe(
    theta1: float,
    theta2: float,
    dtheta1_pre: float,
    dtheta2_pre: float,
    params_pre: Dict[str, float],
) -> Tuple[float, float, float, float, float, float]:
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
    sol = solve_linear_system_gauss(a_mat, b_vec)

    theta1_new = np.arctan2(l2 * c2, -l2 * s2)
    theta2_new = theta1 - np.pi / 2.0
    dtheta1_new = sol[3]
    dtheta2_new = sol[2]
    return theta1_new, theta2_new, dtheta1_new, dtheta2_new, sol[0], sol[1]


def handle_collision_with_delta(
    state: np.ndarray,
    stance_leg: int,
    params_fun: Dict[str, float],
    energy: float,
    recovery_delta_dtheta1: float,
) -> Tuple[np.ndarray, float, float]:
    theta1_new, theta2_new, dtheta1_new, dtheta2_new, _, _ = collision_dynamics_full_safe(
        state[0], state[2], state[1], state[3], params_fun
    )
    energy_pre = core.kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params_fun)
    dtheta1_new += float(recovery_delta_dtheta1)
    energy_new = core.kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params_fun)
    delta = float(energy_new - energy_pre)
    next_state = np.array([theta1_new, dtheta1_new, theta2_new, dtheta2_new], dtype=float)
    return next_state, float(energy + delta), delta


class PPOAlignedMPCController(core.ControllerBase):
    """MPC controller with PPO-matched action spaces and optional Cmt objective."""

    name = "ppo_aligned_mpc"

    def __init__(
        self,
        action_mode: str = ACTION_MODE_CONTINUOUS,
        cost_mode: str = COST_MODE_CMT,
        target_mode: str = TARGET_MODE_EITHER,
        control_hold_steps: int = core.MPC_CONTROL_HOLD_STEPS,
        num_knots: int = core.MPC_NUM_KNOTS,
        max_replans_per_walking_step: int = core.MPC_MAX_REPLANS_PER_WALKING_STEP,
        maxiter: int = core.MPC_MAXITER,
        ftol: float = core.MPC_FTOL,
        terminal_angle_weight: float = 420.0,
        terminal_velocity_weight: float = 0.0,
        running_target_weight: float = 0.08,
        predicted_work_weight: float = 1.0,
        reward_action_weight: float = 0.04,
        torque_abs_weight: float = 0.004,
        torque_smooth_weight: float = 0.012,
        fall_soft_weight: float = 1000.0,
        discrete_cem_samples: int = 96,
        discrete_cem_elite: int = 16,
        discrete_cem_iters: int = 4,
        random_seed: int = 7,
    ):
        self.action_mode = action_mode
        self.cost_mode = cost_mode
        self.target_mode = target_mode
        self.control_hold_steps = int(control_hold_steps)
        self.num_knots = int(num_knots)
        self.max_replans_per_walking_step = int(max_replans_per_walking_step)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.terminal_angle_weight = float(terminal_angle_weight)
        self.terminal_velocity_weight = float(terminal_velocity_weight)
        self.running_target_weight = float(running_target_weight)
        self.predicted_work_weight = float(predicted_work_weight)
        self.reward_action_weight = float(reward_action_weight)
        self.torque_abs_weight = float(torque_abs_weight)
        self.torque_smooth_weight = float(torque_smooth_weight)
        self.fall_soft_weight = float(fall_soft_weight)
        self.discrete_cem_samples = int(discrete_cem_samples)
        self.discrete_cem_elite = int(discrete_cem_elite)
        self.discrete_cem_iters = int(discrete_cem_iters)
        self.rng = np.random.default_rng(int(random_seed))
        self.plan: List[float] = []
        self.solve_log: List[Dict[str, object]] = []
        self.replans_this_step = 0
        self.last_solution: Optional[np.ndarray] = None

    def begin_walking_step(
        self, walking_step_index: int, stance_leg: int, state: np.ndarray, env: core.TwoLinkEnv
    ) -> None:
        self.plan = []
        self.replans_this_step = 0
        self.last_solution = None

    def action(self, state: np.ndarray, env: core.TwoLinkEnv, walking_step_index: int, stance_leg: int) -> float:
        if not self.plan:
            if self.replans_this_step >= self.max_replans_per_walking_step:
                return core.PIDSeedController().action(state, env, walking_step_index, stance_leg)
            self.plan = self.solve_plan(state, env, walking_step_index, stance_leg)
            self.replans_this_step += 1
        return float(self.plan.pop(0))

    def rollout_open_loop(
        self, state: np.ndarray, env: core.TwoLinkEnv, knot_torques: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        sim_env = core.TwoLinkEnv(env.param)
        sim_env.define(state)
        knot_torques = clip_or_quantize(np.asarray(knot_torques, dtype=float), self.action_mode, env.max_torque)
        expanded = np.repeat(knot_torques, self.control_hold_steps)
        states = [np.asarray(state, dtype=float).copy()]
        reasons: List[str] = []
        for torque in expanded:
            next_state, _, done = sim_env.step(float(torque))
            reason = sim_env.terminal_reason
            if ppo_style_target_success(next_state, sim_env, self.target_mode):
                done = True
                reason = ppo_style_target_reason(next_state, sim_env, self.target_mode)
            elif sim_env.fall_condition(next_state):
                done = True
                reason = "fall"
            states.append(next_state.copy())
            reasons.append(reason)
            if done:
                break
        return np.asarray(states), expanded[: max(0, len(states) - 1)], reasons

    def seed_guess(self, state: np.ndarray, env: core.TwoLinkEnv) -> np.ndarray:
        pid = core.PIDSeedController()
        sim_env = core.TwoLinkEnv(env.param)
        sim_env.define(state)
        torques: List[float] = []
        current = state.copy()
        done = False
        for _ in range(self.num_knots):
            torque = pid.action(current, sim_env, 0, 1)
            torque = float(clip_or_quantize(np.array([torque]), self.action_mode, env.max_torque)[0])
            torques.append(torque)
            for _ in range(self.control_hold_steps):
                current, _, done = sim_env.step(torque)
                if done or ppo_style_target_success(current, sim_env, self.target_mode):
                    done = True
                    break
            if done:
                torques.extend([0.0] * (self.num_knots - len(torques)))
                break
        if len(torques) < self.num_knots:
            torques.extend([0.0] * (self.num_knots - len(torques)))
        return clip_or_quantize(np.asarray(torques[: self.num_knots]), self.action_mode, env.max_torque)

    def cost(self, knot_torques: np.ndarray, state: np.ndarray, env: core.TwoLinkEnv) -> float:
        knot_torques = clip_or_quantize(np.asarray(knot_torques, dtype=float), self.action_mode, env.max_torque)
        states, expanded, _ = self.rollout_open_loop(state, env, knot_torques)
        q = states[:, [0, 2]]
        dq = states[:, [1, 3]]
        terminal_err_sq = target_error_sq(q[-1], env, self.target_mode)
        running_err_sq = sum(target_error_sq(item, env, self.target_mode) for item in q)
        fall_violation = np.maximum(np.abs(states[:, 0] - np.pi / 2.0) - env.rad_theta1_range, 0.0)
        fall_violation += np.maximum(np.abs(states[:, 2]) - env.rad_theta2_range, 0.0)
        torque_smooth = np.diff(knot_torques) if len(knot_torques) > 1 else np.zeros(0)

        if self.cost_mode == COST_MODE_CMT:
            effort = self.predicted_work_weight * predicted_positive_work(states, expanded, env.dt)
            effort += self.torque_abs_weight * np.sum(np.abs(expanded))
        elif self.cost_mode == COST_MODE_REWARD_MATCHED:
            effort = self.reward_action_weight * np.sum(np.abs(expanded))
        else:
            raise ValueError(f"Unknown cost_mode: {self.cost_mode}")

        return float(
            self.terminal_angle_weight * terminal_err_sq
            + self.terminal_velocity_weight * np.sum(dq[-1] ** 2)
            + self.running_target_weight * running_err_sq
            + effort
            + self.torque_smooth_weight * np.sum(torque_smooth ** 2)
            + self.fall_soft_weight * np.sum(fall_violation ** 2)
        )

    def final_target_constraint(self, knot_torques: np.ndarray, state: np.ndarray, env: core.TwoLinkEnv) -> float:
        states, _, _ = self.rollout_open_loop(state, env, knot_torques)
        return target_margin(states[-1][[0, 2]], env, self.target_mode)

    def state_limit_constraint(self, knot_torques: np.ndarray, state: np.ndarray, env: core.TwoLinkEnv) -> float:
        states, _, _ = self.rollout_open_loop(state, env, knot_torques)
        theta1_margin = env.rad_theta1_range - np.abs(states[:, 0] - np.pi / 2.0)
        theta2_margin = env.rad_theta2_range - np.abs(states[:, 2])
        return float(np.min(np.concatenate([theta1_margin, theta2_margin])))

    def solve_plan(self, state: np.ndarray, env: core.TwoLinkEnv, walking_step_index: int, stance_leg: int) -> List[float]:
        if is_discrete_action_mode(self.action_mode):
            return self.solve_discrete_plan(state, env, walking_step_index, stance_leg)
        return self.solve_continuous_plan(state, env, walking_step_index, stance_leg)

    def solve_continuous_plan(
        self, state: np.ndarray, env: core.TwoLinkEnv, walking_step_index: int, stance_leg: int
    ) -> List[float]:
        guess = self.seed_guess(state, env)
        if self.last_solution is not None and len(self.last_solution) == self.num_knots:
            guess = np.r_[self.last_solution[1:], self.last_solution[-1]]
            guess = clip_or_quantize(guess, self.action_mode, env.max_torque)
        lo, hi = continuous_bounds_for_mode(self.action_mode, env.max_torque)
        bounds = [(lo, hi)] * self.num_knots
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
            options={"maxiter": self.maxiter, "ftol": self.ftol, "disp": False},
        )
        candidates = [
            guess,
            np.zeros(self.num_knots, dtype=float),
            np.full(self.num_knots, lo, dtype=float),
            np.full(self.num_knots, hi, dtype=float),
        ]
        if result.x is not None:
            candidates.append(np.asarray(result.x, dtype=float))
        best = min(candidates, key=lambda u: self.cost(u, state, env))
        best = clip_or_quantize(best, self.action_mode, env.max_torque)
        self.last_solution = best.copy()
        states, expanded, reasons = self.rollout_open_loop(state, env, best)
        self.solve_log.append(
            {
                "walking_step_index": float(walking_step_index),
                "stance_leg": float(stance_leg),
                "action_mode": self.action_mode,
                "cost_mode": self.cost_mode,
                "optimizer": "SLSQP",
                "optimizer_success": float(bool(result.success)),
                "optimizer_cost": float(result.fun) if np.isfinite(result.fun) else float("nan"),
                "selected_cost": float(self.cost(best, state, env)),
                "predicted_steps": float(len(expanded)),
                "predicted_final_reason": str(reasons[-1] if reasons else "empty"),
            }
        )
        return [float(v) for v in expanded]

    def solve_discrete_plan(
        self, state: np.ndarray, env: core.TwoLinkEnv, walking_step_index: int, stance_leg: int
    ) -> List[float]:
        action_set = discrete_values_for_mode(self.action_mode, env.max_torque)
        n_actions = len(action_set)
        probs = np.full((self.num_knots, n_actions), 1.0 / n_actions, dtype=float)
        seeds = [
            quantize_actions(self.seed_guess(state, env), action_set),
            np.zeros(self.num_knots, dtype=float),
        ]
        if self.last_solution is not None and len(self.last_solution) == self.num_knots:
            seeds.append(quantize_actions(np.r_[self.last_solution[1:], self.last_solution[-1]], action_set))

        best: Optional[np.ndarray] = None
        best_cost = float("inf")
        for seed in seeds:
            c = self.cost(seed, state, env)
            if c < best_cost:
                best = np.asarray(seed, dtype=float).copy()
                best_cost = c

        elite_count = max(1, min(self.discrete_cem_elite, self.discrete_cem_samples))
        for _ in range(self.discrete_cem_iters):
            samples = np.empty((self.discrete_cem_samples, self.num_knots), dtype=float)
            for k in range(self.num_knots):
                idx = self.rng.choice(n_actions, size=self.discrete_cem_samples, p=probs[k])
                samples[:, k] = action_set[idx]
            costs = np.array([self.cost(sample, state, env) for sample in samples], dtype=float)
            elite_idx = np.argsort(costs)[:elite_count]
            elites = samples[elite_idx]
            if costs[elite_idx[0]] < best_cost:
                best = elites[0].copy()
                best_cost = float(costs[elite_idx[0]])
            new_probs = np.zeros_like(probs)
            for k in range(self.num_knots):
                for a_idx, action_value in enumerate(action_set):
                    new_probs[k, a_idx] = np.mean(elites[:, k] == action_value)
            probs = 0.25 * probs + 0.75 * new_probs
            probs = np.maximum(probs, 1e-4)
            probs /= probs.sum(axis=1, keepdims=True)

        if best is None:
            best = np.zeros(self.num_knots, dtype=float)
        best = quantize_actions(best, action_set)
        self.last_solution = best.copy()
        states, expanded, reasons = self.rollout_open_loop(state, env, best)
        self.solve_log.append(
            {
                "walking_step_index": float(walking_step_index),
                "stance_leg": float(stance_leg),
                "action_mode": self.action_mode,
                "cost_mode": self.cost_mode,
                "optimizer": "categorical_cem",
                "optimizer_success": float(ppo_style_target_success(states[-1], env, self.target_mode)),
                "optimizer_cost": float(best_cost),
                "selected_cost": float(self.cost(best, state, env)),
                "predicted_steps": float(len(expanded)),
                "predicted_final_reason": str(reasons[-1] if reasons else "empty"),
            }
        )
        return [float(v) for v in expanded]


def run_three_step_simulation_ppo_aligned(
    controller: core.ControllerBase,
    method_name: Optional[str] = None,
    initial_state: Optional[np.ndarray] = None,
    recovery_map: Dict[int, float] = core.RECOVERY_DTHETA1_BY_STANCE,
    recovery_sequence: Optional[Sequence[float]] = None,
    target_mode: str = TARGET_MODE_EITHER,
    total_steps: int = core.TOTAL_WALKING_STEPS,
) -> core.SimulationResult:
    method = method_name or controller.name
    state = core.initial_state_from_grid() if initial_state is None else np.array(initial_state, dtype=float)
    stance_leg = 1
    walking_step = 0
    global_step = 0
    energy = 0.0
    distance = 0.0
    foot_d = 0.0
    initial_theta1 = float(state[0])
    initial_theta2 = float(state[2])
    trace: List[core.TraceRow] = []
    terminal_reason = "not_started"
    recovery_used: List[float] = []
    recovery_energy_used: List[float] = []

    while walking_step < total_steps:
        env = core.TwoLinkEnv(core.params_for_stance(stance_leg))
        env.define(state)
        controller.begin_walking_step(walking_step + 1, stance_leg, state.copy(), env)
        step_start_theta1 = initial_theta1
        step_start_theta2 = initial_theta2
        terminal_reason = "max_inner_steps"
        collision_delta = 0.0

        for _ in range(core.MAX_INNER_STEPS_PER_WALKING_STEP):
            old_state = env.state.copy()
            torque = float(
                np.clip(
                    controller.action(old_state.copy(), env, walking_step + 1, stance_leg),
                    -env.max_torque,
                    env.max_torque,
                )
            )
            work = core.positive_work(torque, old_state, env.dt)
            energy += work
            next_state, _, env_done = env.step(torque)
            target_done = ppo_style_target_success(next_state, env, target_mode)
            fall_done = env.fall_condition(next_state)
            if target_done:
                terminal_reason = ppo_style_target_reason(next_state, env, target_mode)
            elif fall_done:
                terminal_reason = "fall"
            elif env_done:
                terminal_reason = env.terminal_reason
            else:
                terminal_reason = "running"

            current_center = core.com_x(next_state[0], next_state[2], env)
            start_center = core.com_x(step_start_theta1, step_start_theta2, env)
            running_distance = distance + abs(current_center - start_center)
            cmt = energy / (core.W * running_distance) if running_distance > 1e-12 else float("nan")
            trace.append(
                core.TraceRow(
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
                    terminal_reason=terminal_reason,
                )
            )
            global_step += 1
            if target_done or fall_done or env_done:
                break

        state = env.state.copy()
        if not ppo_style_target_success(state, env, target_mode):
            break

        final_theta1 = float(state[0])
        final_theta2 = float(state[2])
        initial_center = core.com_x(initial_theta1, initial_theta2, env)
        final_center = core.com_x(final_theta1, final_theta2, env)
        distance += abs(final_center - initial_center)
        foot_d += core.foot_distance(final_theta1, final_theta2, env)

        if recovery_sequence is None:
            recovery_delta = float(recovery_map[int(stance_leg)])
        else:
            recovery_delta = float(recovery_sequence[walking_step])
        state, energy, collision_delta = handle_collision_with_delta(
            state, stance_leg, env.param, energy, recovery_delta
        )
        recovery_used.append(recovery_delta)
        recovery_energy_used.append(collision_delta)
        initial_theta1 = float(state[0])
        initial_theta2 = float(state[2])
        if trace:
            trace[-1].collision_recovery_j = collision_delta
            trace[-1].cumulative_energy_j = energy
            trace[-1].cumulative_distance_m = distance
            trace[-1].cumulative_cmt = energy / (core.W * distance) if distance > 1e-12 else float("nan")
            trace[-1].terminal_reason = f"collision_recovery_{terminal_reason}"
        stance_leg = 2 if stance_leg == 1 else 1
        walking_step += 1

    success = walking_step >= total_steps
    cmt = energy / (core.W * distance) if distance > 1e-12 else float("nan")
    foot_error = total_steps * core.PARAMS1["l1"] - foot_d
    extra: Dict[str, float] = {
        "recovery_sequence": ";".join(f"{v:.6g}" for v in recovery_used),
        "recovery_energy_sequence": ";".join(f"{v:.6g}" for v in recovery_energy_used),
        "recovery_energy_total_j": float(np.sum(recovery_energy_used)) if recovery_energy_used else 0.0,
    }
    if isinstance(controller, PPOAlignedMPCController):
        extra.update(
            {
                "action_mode": controller.action_mode,
                "cost_mode": controller.cost_mode,
                "target_mode": controller.target_mode,
                "mpc_num_solves": float(len(controller.solve_log)),
                "mpc_failed_solves": float(
                    sum(1 for item in controller.solve_log if float(item["optimizer_success"]) < 0.5)
                ),
            }
        )
    return core.SimulationResult(
        method=method,
        success=success,
        cmt=float(cmt),
        energy_j=float(energy),
        distance_m=float(distance),
        time_s=float(global_step * core.PARAMS1["dt"]),
        foot_error_m=float(foot_error),
        steps=int(global_step),
        terminal_reason=terminal_reason,
        trace=trace,
        extra=extra,
    )


def result_summary_row(result: core.SimulationResult, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    row = core.result_summary_row(result)
    row.update(result.extra)
    if extra:
        row.update(extra)
    return row


def write_result_bundle(
    output_dir: Path,
    prefix: str,
    result: core.SimulationResult,
    controller: Optional[PPOAlignedMPCController] = None,
    extra_summary: Optional[Dict[str, object]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    core.write_csv(output_dir / f"{prefix}_summary.csv", [result_summary_row(result, extra_summary)])
    core.write_csv(output_dir / f"{prefix}_trace.csv", [asdict(row) for row in result.trace])
    if controller is not None and controller.solve_log:
        core.write_csv(output_dir / f"{prefix}_solve_log.csv", controller.solve_log)


def recovery_sequence_product(
    total_steps: int,
    grid_by_stance: Dict[int, Sequence[float]],
    tie_by_stance: bool = True,
) -> Iterable[Tuple[float, ...]]:
    stance_sequence = [1 if step % 2 == 0 else 2 for step in range(total_steps)]
    if not tie_by_stance:
        grids = [list(grid_by_stance[int(stance)]) for stance in stance_sequence]
        return itertools.product(*grids)

    unique_stances = []
    for stance in stance_sequence:
        if stance not in unique_stances:
            unique_stances.append(stance)

    def tied_sequences() -> Iterable[Tuple[float, ...]]:
        stance_grids = [list(grid_by_stance[int(stance)]) for stance in unique_stances]
        for values in itertools.product(*stance_grids):
            by_stance = {int(stance): float(value) for stance, value in zip(unique_stances, values)}
            yield tuple(by_stance[int(stance)] for stance in stance_sequence)

    return tied_sequences()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def recovery_candidate_count(
    total_steps: int,
    grid_by_stance: Dict[int, Sequence[float]],
    tie_by_stance: bool = True,
    max_combinations: Optional[int] = None,
) -> int:
    stance_sequence = [1 if step % 2 == 0 else 2 for step in range(total_steps)]
    if tie_by_stance:
        unique_stances: List[int] = []
        for stance in stance_sequence:
            if stance not in unique_stances:
                unique_stances.append(stance)
        count = 1
        for stance in unique_stances:
            count *= len(grid_by_stance[int(stance)])
    else:
        count = 1
        for stance in stance_sequence:
            count *= len(grid_by_stance[int(stance)])
    if max_combinations is not None:
        count = min(count, int(max_combinations))
    return int(count)


def run_recovery_grid_search(
    controller_factory: Callable[[], PPOAlignedMPCController],
    output_dir: Path,
    prefix: str,
    recovery_grid_by_stance: Dict[int, Sequence[float]],
    initial_state: Optional[np.ndarray] = None,
    target_mode: str = TARGET_MODE_EITHER,
    total_steps: int = core.TOTAL_WALKING_STEPS,
    max_combinations: Optional[int] = None,
    tie_recovery_by_stance: bool = True,
    progress_interval: int = 10,
    progress_min_seconds: float = 30.0,
) -> core.SimulationResult:
    rows: List[Dict[str, object]] = []
    best_result: Optional[core.SimulationResult] = None
    best_controller: Optional[PPOAlignedMPCController] = None
    best_sequence: Optional[Tuple[float, ...]] = None
    total_candidates = recovery_candidate_count(
        total_steps,
        recovery_grid_by_stance,
        tie_by_stance=tie_recovery_by_stance,
        max_combinations=max_combinations,
    )
    start_time = time.perf_counter()
    last_progress_time = start_time
    progress_interval = max(1, int(progress_interval))
    print(
        "Recovery search candidates: "
        f"{total_candidates} | tie_by_stance={tie_recovery_by_stance} | "
        "ETA will update after the first candidate."
    )

    for idx, sequence in enumerate(
        recovery_sequence_product(total_steps, recovery_grid_by_stance, tie_by_stance=tie_recovery_by_stance)
    ):
        if max_combinations is not None and idx >= max_combinations:
            break
        controller = controller_factory()
        result = run_three_step_simulation_ppo_aligned(
            controller,
            method_name=f"{prefix}_candidate",
            initial_state=initial_state,
            recovery_sequence=sequence,
            target_mode=target_mode,
            total_steps=total_steps,
        )
        row = result_summary_row(
            result,
            {
                "candidate_index": idx,
                "candidate_recovery_sequence": ";".join(f"{v:.6g}" for v in sequence),
                "candidate_recovery_stance1": sequence[0] if len(sequence) > 0 else "",
                "candidate_recovery_stance2": sequence[1] if len(sequence) > 1 else "",
                "tie_recovery_by_stance": tie_recovery_by_stance,
            },
        )
        rows.append(row)

        def rank_key(item: core.SimulationResult) -> Tuple[int, float, float]:
            cmt = item.cmt if np.isfinite(item.cmt) else float("inf")
            energy = item.energy_j if np.isfinite(item.energy_j) else float("inf")
            return (0 if item.success else 1, cmt, energy)

        if best_result is None or rank_key(result) < rank_key(best_result):
            best_result = result
            best_controller = controller
            best_sequence = tuple(sequence)

        completed = idx + 1
        now = time.perf_counter()
        should_print = (
            completed == 1
            or completed == total_candidates
            or completed % progress_interval == 0
            or (now - last_progress_time) >= progress_min_seconds
        )
        if should_print:
            elapsed = now - start_time
            avg = elapsed / max(1, completed)
            remaining = avg * max(0, total_candidates - completed)
            total_est = avg * total_candidates
            best_cmt = best_result.cmt if best_result is not None else float("nan")
            print(
                "Recovery search progress: "
                f"{completed}/{total_candidates} | "
                f"elapsed {format_duration(elapsed)} | "
                f"avg {avg:.2f}s/candidate | "
                f"ETA {format_duration(remaining)} | "
                f"estimated total {format_duration(total_est)} | "
                f"best_success={best_result.success if best_result is not None else False} | "
                f"best_cmt={best_cmt:.6g}"
            )
            last_progress_time = now

    if best_result is None:
        raise RuntimeError("No recovery candidates were evaluated.")

    best_result.method = prefix
    elapsed_total = time.perf_counter() - start_time
    avg_candidate_seconds = elapsed_total / max(1, len(rows))
    final_extra = {
        "recovery_search_candidates": len(rows),
        "best_recovery_sequence": ";".join(f"{v:.6g}" for v in best_sequence or ()),
        "best_recovery_stance1": best_sequence[0] if best_sequence and len(best_sequence) > 0 else "",
        "best_recovery_stance2": best_sequence[1] if best_sequence and len(best_sequence) > 1 else "",
        "tie_recovery_by_stance": tie_recovery_by_stance,
        "elapsed_seconds": elapsed_total,
        "elapsed_human": format_duration(elapsed_total),
        "avg_candidate_seconds": avg_candidate_seconds,
    }
    best_result.extra.update(final_extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    core.write_csv(output_dir / f"{prefix}_all_recovery_candidates.csv", rows)
    write_result_bundle(
        output_dir,
        prefix,
        best_result,
        best_controller,
        final_extra,
    )
    return best_result
