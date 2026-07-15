from pathlib import Path
import os
import sys
from time import perf_counter
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


_ENERGY_COMPARISON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ENERGY_COMPARISON_ROOT))
from cmt_metrics import positive_actuator_work_increment


# Preserve the archived location as the default while allowing portable reruns.
ENERGY_COMPARISON_DATA_ROOT = os.environ.get(
    "ENERGY_COMPARISON_DATA_ROOT",
    "D:/L&S/Mas/Project/Paper1/Energy_Comparison",
).rstrip("/\\")


# =========================
# User editable settings
# =========================

# Change these four paths when you want to read different trained models.
MODEL_PATHS = {
    "passive_01": Path(
        ENERGY_COMPARISON_DATA_ROOT
    ) / "action_weight=0.02/60,30(0.33m)/Policy_Net_Pytorch(1,0)_846.pth",
    "passive_minus10": Path(
        ENERGY_COMPARISON_DATA_ROOT
    ) / "action_weight=0.02/60,30(0.33m)/Policy_Net_Pytorch(-1,0)_830.pth",
    "active_discrete": Path(
        ENERGY_COMPARISON_DATA_ROOT
    ) / "action_weight=0.02/60,30(0.33m)/Policy_Net_Pytorch(-1,0,1)_1663.pth",
    "active_continuous": Path(
        ENERGY_COMPARISON_DATA_ROOT
    ) / "action_weight=0.02/60,30(0.33m)/Policy_Net_Pytorch(-1,0,1)_1421_continuous.pth",
}

# Change this directory and these names when you want different output files.
OUTPUT_DIR = Path(ENERGY_COMPARISON_DATA_ROOT) / "action_weight=0.02/60,30(0.33m)"
OUTPUT_FILES = {
    "working_01": "working_save(0,1)-10-30",
    "cmt_01": "Cmt_save(0,1)-10-30",
    "energy_01": "Energy_save(0,1)-10-30",
    "working_minus10": "working_save(-1,0)-10-30",
    "cmt_minus10": "Cmt_save(-1,0)-10-30",
    "energy_minus10": "Energy_save(-1,0)-10-30",
    "working_passive": "working_save_passive-10-30",
    "cmt_passive": "Cmt_save_passive-10-30",
    "working_active_discrete": "working_save_active_discrete-10-30",
    "energy_active_discrete": "Energy_save_active_discrete-10-30",
    "cmt_active_discrete": "Cmt_save_active_discrete(0,1)-10-30",
    "working_active_continuous": "working_save_active_continuous-10-30",
    "energy_active_continuous": "Energy_save_active_continuous-10-30",
    "cmt_active_continuous": "Cmt_save_active_continuous-10-30",
}

# Grid and runtime settings.
N1 = 10
N2 = 30
SPEED_NONDIM = 3.5
SIMU_TIME = 3.2
DT = 0.01
# The source script runs on CPU. Change this to "cuda" manually if you prefer
# maximum speed and can accept tiny policy-output differences near argmax ties.
DEVICE = torch.device("cpu")
TORCH_NUM_THREADS = None  # Set to an integer if you want to limit CPU threads.
RANDOM_SEED = None  # Set to an integer for repeatable continuous-action sampling.


# =========================
# Simulation constants
# =========================

NNN = 512
SETTLE = np.deg2rad(5)
M1 = 1.4122
M2 = 0.0839
L1 = 0.22
L2 = 0.31
L1_R = 0.33
L2_R = 0.31
C1 = 0.0
C2 = 0.0
J1 = M1 * (L1 ** 2)
J2 = M2 * (L2 ** 2)
G = 9.8
TORQUE = 4.0
STEPS = int(SIMU_TIME / DT)

THETA1_TARGET1 = np.deg2rad(60)
THETA1_TARGET2 = np.deg2rad(120)
THETA2_TARGET1 = np.deg2rad(30)
THETA2_TARGET2 = np.deg2rad(-30)
RAD_THETA1_RANGE = np.deg2rad(90)
RAD_THETA2_RANGE = np.deg2rad(90)
SPEED_RANGE = 3.5


def make_discrete_policy(output_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(4, NNN * 2),
        torch.nn.Tanh(),
        torch.nn.Linear(NNN * 2, NNN),
        torch.nn.Tanh(),
        torch.nn.Linear(NNN, output_dim),
        torch.nn.Softmax(dim=1),
    )


def make_continuous_policy() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(4, NNN * 2),
        torch.nn.Tanh(),
        torch.nn.Linear(NNN * 2, NNN),
        torch.nn.Tanh(),
        torch.nn.Linear(NNN, 2),
    )


def load_policy(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    state_dict = torch.load(str(path), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_models() -> Dict[str, torch.nn.Module]:
    return {
        "passive_01": load_policy(make_discrete_policy(2), MODEL_PATHS["passive_01"]),
        "passive_minus10": load_policy(make_discrete_policy(2), MODEL_PATHS["passive_minus10"]),
        "active_discrete": load_policy(make_discrete_policy(3), MODEL_PATHS["active_discrete"]),
        "active_continuous": load_policy(make_continuous_policy(), MODEL_PATHS["active_continuous"]),
    }


def torch_inference_context():
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


def build_initial_states() -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    tht1s = np.linspace(60, 120, N1, dtype=np.float64) * np.pi / 180
    dtht1s = np.linspace(-SPEED_NONDIM, SPEED_NONDIM, N2, dtype=np.float64)
    tht2s = np.linspace(-60, 60, N1, dtype=np.float64) * np.pi / 180
    dtht2s = np.linspace(-SPEED_NONDIM, SPEED_NONDIM, N2, dtype=np.float64)

    mesh = np.meshgrid(tht1s, dtht1s, tht2s, dtht2s, indexing="ij")
    states = np.stack(mesh, axis=-1).reshape(-1, 4)
    return states, (N1, N2, N1, N2)


def center_x(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    return (
        M1 * L1 * np.cos(theta1)
        + M2 * (L1_R * np.cos(theta1) + L2 * np.sin(theta2))
    ) / (M1 + M2)


def success_mask(states: np.ndarray) -> np.ndarray:
    return (
        (np.abs(states[:, 0] - THETA1_TARGET1) < SETTLE)
        & (np.abs(states[:, 2] - THETA2_TARGET1) < SETTLE)
    ) | (
        (np.abs(states[:, 0] - THETA1_TARGET2) < SETTLE)
        & (np.abs(states[:, 2] - THETA2_TARGET2) < SETTLE)
    )


def in_bounds(states: np.ndarray) -> np.ndarray:
    return (
        (states[:, 0] >= 0.0)
        & (states[:, 0] <= np.pi)
        & (states[:, 2] >= -np.pi / 2)
        & (states[:, 2] <= np.pi / 2)
    )


def normalized_states(states: np.ndarray) -> np.ndarray:
    out = np.empty((states.shape[0], 4), dtype=np.float32)
    out[:, 0] = (states[:, 0] - np.pi / 2) / RAD_THETA1_RANGE
    out[:, 1] = states[:, 1] / SPEED_RANGE
    out[:, 2] = states[:, 2] / RAD_THETA2_RANGE
    out[:, 3] = states[:, 3] / SPEED_RANGE
    return out


def dynamics_step(states: np.ndarray, applied_torque: np.ndarray) -> None:
    theta1 = states[:, 0]
    dtheta1 = states[:, 1]
    theta2 = states[:, 2]
    dtheta2 = states[:, 3]

    diff = theta2 - theta1
    sin_diff = np.sin(diff)
    cos_diff = np.cos(diff)

    a00 = J1 + M2 * L1_R ** 2
    a11 = J2 + M2 * L2 ** 2
    a01 = M2 * L1_R * L2 * sin_diff

    b0 = (
        -M2 * dtheta1 * L1_R * dtheta2 * L2 * cos_diff
        - M1 * G * L1 * np.cos(theta1)
        - M2 * G * L1_R * np.cos(theta1)
        - M2 * L1_R * dtheta2 * L2 * cos_diff * (dtheta2 - dtheta1)
    )
    b1 = (
        M2 * dtheta1 * L1_R * dtheta2 * L2 * cos_diff
        - M2 * G * L2 * np.sin(theta2)
        - M2 * L1_R * dtheta1 * L2 * cos_diff * (dtheta2 - dtheta1)
    )

    rhs0 = b0 - applied_torque
    rhs1 = b1 + applied_torque
    det = a00 * a11 - a01 * a01
    acc1 = (a11 * rhs0 - a01 * rhs1) / det
    acc2 = (-a01 * rhs0 + a00 * rhs1) / det

    states[:, 1] += acc1 * DT - C1 * states[:, 0]
    states[:, 0] += states[:, 1] * DT
    states[:, 3] += acc2 * DT - C2 * states[:, 2]
    states[:, 2] += states[:, 3] * DT


def finalize_success(
    flat_indices: np.ndarray,
    states: np.ndarray,
    working: np.ndarray,
    energy: np.ndarray,
    cmt: np.ndarray,
    first_action: np.ndarray,
    initial_center: np.ndarray,
) -> None:
    working[flat_indices] = first_action[flat_indices]
    final_center = center_x(states[:, 0], states[:, 2])
    distance = np.abs(final_center - initial_center[flat_indices])
    denominator = (M1 + M2) * G * distance
    cmt[flat_indices] = np.divide(
        energy[flat_indices],
        denominator,
        out=np.zeros_like(distance),
        where=denominator > 0,
    )


def choose_discrete_actions(
    policy: torch.nn.Module,
    states: np.ndarray,
    action_table: np.ndarray,
) -> np.ndarray:
    net_input = torch.from_numpy(normalized_states(states)).to(DEVICE)
    with torch_inference_context():
        action_indices = torch.argmax(policy(net_input), dim=1).cpu().numpy()
    return action_table[action_indices]


def choose_continuous_actions(policy: torch.nn.Module, states: np.ndarray) -> np.ndarray:
    net_input = torch.from_numpy(normalized_states(states)).to(DEVICE)
    with torch_inference_context():
        output = policy(net_input)
        mean, log_std = output.chunk(2, dim=1)
        actions = mean + torch.exp(log_std) * torch.randn_like(mean)
        actions = torch.clamp(actions, -TORQUE, TORQUE).squeeze(1).cpu().numpy()
    return actions.astype(np.float64, copy=False)


def simulate_policy(
    name: str,
    policy: torch.nn.Module,
    initial_states: np.ndarray,
    grid_shape: Tuple[int, int, int, int],
    *,
    action_table: Optional[np.ndarray] = None,
    fail_value: float = -2.0,
    continuous: bool = False,
    check_success_before_step: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = initial_states.copy()
    total = states.shape[0]
    active = np.ones(total, dtype=bool)
    has_first_action = np.zeros(total, dtype=bool)
    first_action = np.zeros(total, dtype=np.float64)
    working = np.full(total, fail_value, dtype=np.float64)
    energy = np.zeros(total, dtype=np.float64)
    cmt = np.zeros(total, dtype=np.float64)
    initial_center = center_x(states[:, 0], states[:, 2])

    progress = tqdm(range(STEPS), desc=name, unit="step")
    for _ in progress:
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break

        if check_success_before_step:
            active_states = states[active_indices]
            reached = success_mask(active_states)
            if np.any(reached):
                reached_indices = active_indices[reached]
                finalize_success(
                    reached_indices,
                    active_states[reached],
                    working,
                    energy,
                    cmt,
                    first_action,
                    initial_center,
                )
                active[reached_indices] = False

        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        active_states = states[active_indices]

        if continuous:
            actions = choose_continuous_actions(policy, active_states)
            applied_torque = actions
        else:
            actions = choose_discrete_actions(policy, active_states, action_table)
            applied_torque = actions * TORQUE

        first_mask = ~has_first_action[active_indices]
        if np.any(first_mask):
            first_indices = active_indices[first_mask]
            first_action[first_indices] = actions[first_mask]
            has_first_action[first_indices] = True

        rel_speed = active_states[:, 3] - active_states[:, 1]
        step_work = positive_actuator_work_increment(applied_torque, rel_speed, DT)
        positive_work = step_work > 0
        if np.any(positive_work):
            energy_indices = active_indices[positive_work]
            energy[energy_indices] += step_work[positive_work]

        dynamics_step(active_states, applied_torque)
        states[active_indices] = active_states

        valid = in_bounds(active_states)
        if np.any(~valid):
            active[active_indices[~valid]] = False

        valid_indices = active_indices[valid]
        if valid_indices.size == 0:
            continue

        valid_states = active_states[valid]
        reached_after_step = success_mask(valid_states)
        if np.any(reached_after_step):
            reached_indices = valid_indices[reached_after_step]
            finalize_success(
                reached_indices,
                valid_states[reached_after_step],
                working,
                energy,
                cmt,
                first_action,
                initial_center,
            )
            active[reached_indices] = False

        progress.set_postfix(active=int(active.sum()), refresh=False)

    return (
        working.reshape(grid_shape),
        energy.reshape(grid_shape),
        cmt.reshape(grid_shape),
    )


def combine_passive_results(
    working_01: np.ndarray,
    cmt_01: np.ndarray,
    working_minus10: np.ndarray,
    cmt_minus10: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    working_passive = np.zeros_like(working_01)

    remaining = np.ones_like(working_01, dtype=bool)

    mask = remaining & (working_minus10 == -1) & (working_01 != 0)
    working_passive[mask] = -1
    remaining[mask] = False

    mask = remaining & (working_minus10 != 0) & (working_01 == 1)
    working_passive[mask] = 1
    remaining[mask] = False

    mask = remaining & ((working_minus10 == 0) | (working_01 == 0))
    working_passive[mask] = 0
    remaining[mask] = False

    # Keep the original if/elif order exactly. This branch is unreachable for
    # (-1, 1), because the first branch has already consumed those points.
    mask = remaining & (working_minus10 == -1) & (working_01 == 1)
    working_passive[mask] = np.where(cmt_01[mask] > cmt_minus10[mask], -1, 1)
    remaining[mask] = False

    mask = remaining & (working_minus10 == -2) & (working_01 == -2)
    working_passive[mask] = -2

    cmt_passive = np.full_like(cmt_01, -2.0)
    mask = working_passive == -1
    cmt_passive[mask] = cmt_minus10[mask]
    mask = working_passive == 1
    cmt_passive[mask] = cmt_01[mask]
    mask = working_passive == 0
    cmt_passive[mask] = np.minimum(cmt_01[mask], cmt_minus10[mask])

    return working_passive, cmt_passive


def save_h5(file_key: str, dataset_name: str, data: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / OUTPUT_FILES[file_key]
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset(dataset_name, data=data)


def summarize(
    working_passive: np.ndarray,
    cmt_passive: np.ndarray,
    working_active_discrete: np.ndarray,
    cmt_active_discrete: np.ndarray,
    working_active_continuous: np.ndarray,
    cmt_active_continuous: np.ndarray,
) -> None:
    valid_passive = working_passive != -2
    valid_active_discrete = working_active_discrete != -2
    valid_active_continuous = working_active_continuous != -6
    valid_all = valid_passive & valid_active_discrete & valid_active_continuous

    count = int(np.count_nonzero(valid_all))
    if count == 0:
        print("No shared valid samples.")
        return

    passive_energy = float(np.sum(cmt_passive[valid_all]) / count)
    active_discrete_energy = float(np.sum(cmt_active_discrete[valid_all]) / count)
    active_continuous_energy = float(np.sum(cmt_active_continuous[valid_all]) / count)

    error1 = cmt_active_discrete - cmt_passive
    idx = np.indices(cmt_passive.shape)
    candidate = (
        valid_all
        & (error1 > 0)
        & (idx[0] > 5)
        & (idx[0] < 35)
        & (idx[1] > 5)
        & (idx[1] < 45)
        & (idx[2] > 5)
        & (idx[2] < 35)
        & (idx[3] > 5)
        & (idx[3] < 45)
    )
    error_save = np.zeros(4)
    if np.any(candidate):
        flat_position = int(np.argmax(np.where(candidate, error1, -np.inf)))
        error_save = np.array(np.unravel_index(flat_position, cmt_passive.shape), dtype=float)

    print("Passive_energy")
    print(passive_energy)
    print("Active_energy_discrete")
    print(active_discrete_energy)
    print("Active_energy_continuous")
    print(active_continuous_energy)
    print(error_save)


def main() -> None:
    if TORCH_NUM_THREADS is not None:
        torch.set_num_threads(TORCH_NUM_THREADS)
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")
    start = perf_counter()
    models = load_models()
    initial_states, grid_shape = build_initial_states()

    working_01, energy_01, cmt_01 = simulate_policy(
        "passive (0,1)",
        models["passive_01"],
        initial_states,
        grid_shape,
        action_table=np.array([0.0, 1.0], dtype=np.float64),
    )
    save_h5("working_01", "working_save", working_01)
    save_h5("cmt_01", "Cmt_save", cmt_01)
    save_h5("energy_01", "Energy_save", energy_01)

    working_minus10, energy_minus10, cmt_minus10 = simulate_policy(
        "passive (-1,0)",
        models["passive_minus10"],
        initial_states,
        grid_shape,
        action_table=np.array([-1.0, 0.0], dtype=np.float64),
        check_success_before_step=False,
    )
    save_h5("working_minus10", "working_save", working_minus10)
    save_h5("cmt_minus10", "Cmt_save", cmt_minus10)
    save_h5("energy_minus10", "Energy_save", energy_minus10)

    working_passive, cmt_passive = combine_passive_results(
        working_01, cmt_01, working_minus10, cmt_minus10
    )
    save_h5("working_passive", "working_save_passive", working_passive)
    save_h5("cmt_passive", "Cmt_save_passive", cmt_passive)

    working_active_discrete, energy_active_discrete, cmt_active_discrete = simulate_policy(
        "active discrete",
        models["active_discrete"],
        initial_states,
        grid_shape,
        action_table=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        check_success_before_step=False,
    )
    save_h5("working_active_discrete", "working_save_active_discrete", working_active_discrete)
    save_h5("energy_active_discrete", "Energy_save_active_discrete", energy_active_discrete)
    save_h5("cmt_active_discrete", "Cmt_save", cmt_active_discrete)

    working_active_continuous, energy_active_continuous, cmt_active_continuous = simulate_policy(
        "active continuous",
        models["active_continuous"],
        initial_states,
        grid_shape,
        fail_value=-6.0,
        continuous=True,
        check_success_before_step=False,
    )
    save_h5(
        "working_active_continuous",
        "working_save_active_continuous",
        working_active_continuous,
    )
    save_h5("energy_active_continuous", "Energy_save_active_continuous", energy_active_continuous)
    save_h5("cmt_active_continuous", "Cmt_save", cmt_active_continuous)

    summarize(
        working_passive,
        cmt_passive,
        working_active_discrete,
        cmt_active_discrete,
        working_active_continuous,
        cmt_active_continuous,
    )
    print(f"Finished in {perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
