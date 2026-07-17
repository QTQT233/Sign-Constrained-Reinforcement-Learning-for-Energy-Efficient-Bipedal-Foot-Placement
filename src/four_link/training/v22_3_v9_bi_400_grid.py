import os
import random
import re
import csv
import json

import numpy as np
import torch
from tqdm import tqdm


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

print(f"Training script: {os.path.basename(__file__)}")


# ------------------------- PPO hyperparameters -------------------------
hidden_size = 512
gamma = 0.99
gamma_gae = 0.95
dt = 0.01

episode = 50000
critic_training_times = 10
critic_training_steps = 10
actor_training_times = 10
playing_times = 500
concentrated_sample_times = 3
batch_size = 5000

reward_scale = 5.0
forward_success_reward_bonus = reward_scale * 2.0
backward_success_reward_bonus = 0.0
action_value_weight = 0.026
policy_entropy_coefficient = 0.001
ppo_clip = 0.2
progress_reward_weight = 0.5
state_error_penalty_weight = 0.005
step_penalty = 0.001
joint_limit_penalty_weight = 0.05
speed_soft_penalty_weight = 0.140
stance_angle_penalty_weight = 0.18
q1_low_soft_penalty_weight = 0.22
q3_low_soft_penalty_weight = 0.14
hip_height_penalty_weight = 0.08
clearance_soft_penalty_weight = 0.26
ground_soft_penalty_weight_start = 0.06
ground_soft_penalty_weight_final = 0.18
ground_soft_penalty_weight = ground_soft_penalty_weight_final
precontact_penalty_weight = 0.12
phase_step_progress_reward_weight = 1.35
phase_lift_progress_reward_weight = 0.22
phase_crossing_reward_bonus = 0.40
phase_ready_reward_bonus = 0.20
precontact_descent_penalty_weight = 0.16
precontact_descent_scale = 0.025
forward_phase_reward_multiplier = 1.25
forward_mid_cross_clearance_hold_weight = 0.025
forward_mid_cross_low_foot_penalty_weight = 0.10
touchdown_descent_penalty_weight = 0.100
touchdown_descent_speed_soft_limit = 0.35
touchdown_descent_height_window = 0.035
touchdown_reset_step_error_scale = 2.5
touchdown_reset_height_error_scale = 5.0
touchdown_reset_min_step_error = 0.07
touchdown_reset_min_height_error = 0.035


# ------------------------- Robot parameters -------------------------
g = 9.8

# Both legs use the same mass and length distribution. The upper body is modeled
# as a point mass at the hip.
m_leg_total = 0.6458
m_hip = 1.2546

m_shank = m_leg_total * 1.0 / 3.2
m_thigh = m_leg_total * 2.2 / 3.2

l_leg_total = 0.49
l_thigh = l_leg_total * 1.0 / 1.8
l_shank = l_leg_total * 0.8 / 1.8

m_support_shank = m_shank
m_support_thigh = m_thigh
m_swing_thigh = m_thigh
m_swing_shank = m_shank

l_support_shank = l_shank
l_support_thigh = l_thigh
l_swing_thigh = l_thigh
l_swing_shank = l_shank

masses = np.array([m_support_shank, m_support_thigh, m_swing_thigh, m_swing_shank], dtype=float)
lengths = np.array([l_support_shank, l_support_thigh, l_swing_thigh, l_swing_shank], dtype=float)
com_lengths = 0.5 * lengths
inertias = masses * lengths ** 2 / 12.0
damping = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

# q = [stance shank absolute angle, stance knee flexion, hip opening from thigh overlap, swing knee flexion]
# phi1 = q1, phi2 = q1 + q2, phi3 = phi2 - pi + q3, phi4 = phi3 - q4.
final_angle_settle = np.deg2rad(5.0)
final_step_settle = 0.025
final_height_settle = 0.008
final_distance_settle = final_step_settle
angle_settle = final_angle_settle
distance_settle = final_distance_settle
height_settle = final_height_settle
initial_posture_error_weight = 0.25
final_posture_error_weight = 0.02
posture_error_weight = initial_posture_error_weight
landing_side_margin = 0.005
swing_ground_tolerance = 1e-4

# Touchdown target: use a signed walking direction, a reasonable signed step
# length, swing-foot ground height, and a safe posture region instead of the old
# mirrored shank-line angles plus whole-leg endpoint distance.
target_swing_foot_height = 0.0
commanded_step_length_options = np.array([0.30, 0.35, 0.40], dtype=float)
commanded_step_height_options = np.array([0.0, 0.01, 0.02], dtype=float)
commanded_step_height_scale = 0.05
target_forward_q = np.deg2rad(np.array([46.1560, 22.8901, 61.4918, 60.7154], dtype=float))
target_backward_q = np.deg2rad(np.array([71.8281, 59.8427, -58.4970, 29.7468], dtype=float))
directional_touchdown_q_targets = {
    1.0: target_forward_q,
    -1.0: target_backward_q,
}

# Target postures matched to the commanded 3x3 step-length/height grid.
# They are used only for shaping/reset centers; success is still judged by
# commanded step length, commanded height, safe posture, thigh-side separation,
# and ground/clearance constraints.
directional_touchdown_q_grid_targets = {
    1.0: {
        (0.30, 0.00): np.deg2rad(np.array([43.3786, 26.5533, 56.4416, 67.6725], dtype=float)),
        (0.30, 0.01): np.deg2rad(np.array([45.1679, 25.8115, 57.1672, 68.6083], dtype=float)),
        (0.30, 0.02): np.deg2rad(np.array([52.7170, 20.5080, 55.6573, 60.8219], dtype=float)),
        (0.35, 0.00): np.deg2rad(np.array([46.1112, 22.8671, 61.5772, 60.8745], dtype=float)),
        (0.35, 0.01): np.deg2rad(np.array([48.4143, 19.7042, 63.8974, 63.0626], dtype=float)),
        (0.35, 0.02): np.deg2rad(np.array([48.8822, 24.2161, 60.9525, 60.4900], dtype=float)),
        (0.40, 0.00): np.deg2rad(np.array([42.4668, 22.9475, 70.7697, 65.0258], dtype=float)),
        (0.40, 0.01): np.deg2rad(np.array([41.3672, 28.2999, 68.8295, 65.0100], dtype=float)),
        (0.40, 0.02): np.deg2rad(np.array([47.2767, 23.1808, 67.4182, 58.6520], dtype=float)),
    },
    -1.0: {
        (0.30, 0.00): np.deg2rad(np.array([70.7318, 58.2823, -47.6056, 38.6240], dtype=float)),
        (0.30, 0.01): np.deg2rad(np.array([61.9022, 62.0780, -54.4717, 24.9323], dtype=float)),
        (0.30, 0.02): np.deg2rad(np.array([69.2348, 51.8407, -49.9630, 24.1316], dtype=float)),
        (0.35, 0.00): np.deg2rad(np.array([71.8279, 59.8567, -58.4699, 29.8289], dtype=float)),
        (0.35, 0.01): np.deg2rad(np.array([73.5698, 57.9370, -54.7936, 37.4559], dtype=float)),
        (0.35, 0.02): np.deg2rad(np.array([69.8914, 61.8963, -55.3007, 42.1653], dtype=float)),
        (0.40, 0.00): np.deg2rad(np.array([69.7807, 66.9374, -71.2930, 24.2330], dtype=float)),
        (0.40, 0.01): np.deg2rad(np.array([75.9404, 60.1900, -63.7136, 35.6327], dtype=float)),
        (0.40, 0.02): np.deg2rad(np.array([73.4133, 61.1507, -64.8776, 33.8921], dtype=float)),
    },
}

touchdown_safe_q_low = np.deg2rad(np.array([35.0, 8.0, -115.0, 5.0], dtype=float))
touchdown_safe_q_high = np.deg2rad(np.array([145.0, 105.0, 115.0, 110.0], dtype=float))
reset_safe_q_low = np.deg2rad(np.array([45.0, 12.0, -105.0, 8.0], dtype=float))
reset_safe_q_high = np.deg2rad(np.array([135.0, 100.0, 105.0, 105.0], dtype=float))


def absolute_angles_from_q(q):
    # Coordinate-fix convention: q3 is the signed hip opening from the thigh-overlap line,
    # and q4 is positive swing-knee flexion.
    return np.array(
        [
            q[0],
            q[0] + q[1],
            q[0] + q[1] - np.pi + q[2],
            q[0] + q[1] - np.pi + q[2] - q[3],
        ],
        dtype=float,
    )


def kinematic_points_from_q(q):
    phi = absolute_angles_from_q(q)
    support_foot = np.array([0.0, 0.0], dtype=float)
    support_knee = support_foot + l_support_shank * np.array([np.cos(phi[0]), np.sin(phi[0])])
    hip = support_knee + l_support_thigh * np.array([np.cos(phi[1]), np.sin(phi[1])])
    swing_knee = hip + l_swing_thigh * np.array([np.cos(phi[2]), np.sin(phi[2])])
    swing_foot = swing_knee + l_swing_shank * np.array([np.cos(phi[3]), np.sin(phi[3])])
    return {
        "support_foot": support_foot,
        "support_knee": support_knee,
        "hip": hip,
        "swing_knee": swing_knee,
        "swing_foot": swing_foot,
    }


target_forward_step_length = abs(kinematic_points_from_q(target_forward_q)["swing_foot"][0])
target_backward_step_length = abs(kinematic_points_from_q(target_backward_q)["swing_foot"][0])
commanded_step_length_scale = l_leg_total

hip_angle_limits = np.deg2rad(np.array([-130.0, 130.0], dtype=float))
q_ranges = np.deg2rad(np.array([90.0, 65.0, 130.0, 65.0], dtype=float))
q_centers = np.deg2rad(np.array([90.0, 65.0, 0.0, 65.0], dtype=float))
PHI_Q_JACOBIAN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, -1.0],
    ],
    dtype=float,
)
speed_range = 3.5
initial_speed_range = 1.0
terminal_speed_limit = 25.0
max_episode_steps = 500
joint_limit_soft_margin = np.deg2rad(np.array([16.0, 4.0, 8.0, 4.0], dtype=float))
stance_angle_safe_low = np.deg2rad(40.0)
stance_angle_safe_high = np.deg2rad(140.0)
q1_low_soft_margin = np.deg2rad(35.0)
q3_low_soft_margin = np.deg2rad(14.0)
hip_height_soft_min = 0.36
hip_height_soft_width = 0.08
speed_soft_limit = speed_range * 2.0

# Human-like knee limits. With the V10 coordinate convention, both knees use
# positive flexion and zero is the straight mechanical stop.
support_knee_limits = np.deg2rad(np.array([0.0, 130.0], dtype=float))
swing_knee_limits = np.deg2rad(np.array([0.0, 130.0], dtype=float))
min_swing_foot_clearance = 0.005

# Curriculum: start with a wider touchdown target and more near-target initial
# states, then gradually return to the final task definition and broad starts.
curriculum_enabled = True
curriculum_epochs = 80
curriculum_adaptive = True
curriculum_progress_value = 0.0
curriculum_success_target_start = 0.22
curriculum_success_target_final = 0.14
curriculum_success_regress_threshold_start = 0.08
curriculum_success_regress_threshold_final = 0.03
curriculum_success_ema_alpha = 0.35
curriculum_success_rate_ema = None
curriculum_failure_ema_alpha = 0.30
curriculum_q1_failure_rate_ema = None
curriculum_q3_failure_rate_ema = None
curriculum_clearance_failure_rate_ema = None
curriculum_ground_failure_rate_ema = None
curriculum_ground_failure_rate_ema_delta = 0.0
curriculum_pre_ground_failure_rate_ema = None
curriculum_mid_ground_failure_rate_ema = None
curriculum_touchdown_ground_failure_rate_ema = None
curriculum_pre_success_rate_ema = None
curriculum_mid_success_rate_ema = None
curriculum_touchdown_success_rate_ema = None
curriculum_speed_failure_rate_ema = None
curriculum_last_progress_debug = {}
curriculum_progress_step = 0.0045
curriculum_regress_step = 0.0012
curriculum_progress_deadband = 0.010
curriculum_q1_failure_gate_start = 0.44
curriculum_q1_failure_gate_final = 0.32
curriculum_clearance_failure_gate_start = 0.42
curriculum_clearance_failure_gate_final = 0.31
curriculum_ground_failure_gate_start = 0.58
curriculum_ground_failure_gate_final = 0.40
curriculum_q3_failure_pause_gate_start = 0.32
curriculum_q3_failure_pause_gate_final = 0.20
curriculum_speed_failure_gate_start = 0.18
curriculum_speed_failure_gate_final = 0.08
curriculum_ground_failure_hard_gate_start = 0.70
curriculum_ground_failure_hard_gate_final = 0.58
curriculum_pre_ground_failure_gate_start = 0.95
curriculum_pre_ground_failure_gate_final = 0.82
curriculum_mid_ground_failure_gate_start = 0.94
curriculum_mid_ground_failure_gate_final = 0.78
curriculum_touchdown_ground_failure_gate_start = 0.48
curriculum_touchdown_ground_failure_gate_final = 0.38
curriculum_pre_ground_gate_weight = 0.15
curriculum_mid_ground_gate_weight = 0.25
curriculum_touchdown_ground_gate_weight = 0.60
curriculum_min_phase_ground_gate_samples = 20
curriculum_ground_trend_tolerance = 0.012
curriculum_failure_gate_deadband = 0.015
curriculum_failure_progress_margin = 0.055
curriculum_min_progress_step = 0.0020
curriculum_phase_ground_soft_margin = 0.06
curriculum_phase_ground_slowdown_scale = 0.18
curriculum_phase_ground_max_slowdown = 0.65
curriculum_phase_success_slowdown_scale = 0.12
curriculum_pre_success_gate_start = 0.00
curriculum_pre_success_gate_final = 0.03
curriculum_mid_success_gate_start = 0.02
curriculum_mid_success_gate_final = 0.06
curriculum_touchdown_success_gate_start = 0.45
curriculum_touchdown_success_gate_final = 0.35
reset_curriculum_power = 1.6
curriculum_start_angle_settle = np.deg2rad(12.0)
curriculum_start_distance_settle = 0.08
curriculum_start_height_settle = 0.06
curriculum_start_easy_reset_probability = 0.90
curriculum_final_easy_reset_probability = 0.10
curriculum_start_easy_reset_noise = np.deg2rad(np.array([6.0, 5.0, 8.0, 6.0], dtype=float))
curriculum_final_easy_reset_noise = np.deg2rad(np.array([18.0, 14.0, 20.0, 18.0], dtype=float))
curriculum_start_challenge_reset_noise = np.deg2rad(np.array([8.0, 6.0, 10.0, 8.0], dtype=float))
curriculum_final_challenge_reset_noise = np.deg2rad(np.array([30.0, 22.0, 25.0, 30.0], dtype=float))
reset_phase_names = np.array(["pre_cross", "mid_cross", "touchdown"], dtype=object)
reset_phase_quota_start_progress = 0.35
reset_phase_quota_full_progress = 1.00
reset_phase_quota_forward_final = {"pre_cross": 30, "mid_cross": 60}
reset_phase_quota_backward_final = {"pre_cross": 24, "mid_cross": 50}
reset_phase_quota_remaining = {}
reset_phase_quota_targets = {}
curriculum_start_reset_phase_probabilities = np.array([0.0, 0.0, 1.0], dtype=float)
curriculum_forward_final_reset_phase_probabilities = np.array([0.12, 0.38, 0.50], dtype=float)
curriculum_backward_final_reset_phase_probabilities = np.array([0.12, 0.28, 0.60], dtype=float)
curriculum_final_reset_phase_probabilities = 0.5 * (
    curriculum_forward_final_reset_phase_probabilities
    + curriculum_backward_final_reset_phase_probabilities
)
current_easy_reset_probability = 0.0
current_easy_reset_noise = curriculum_final_easy_reset_noise.copy()
current_challenge_reset_noise = curriculum_final_challenge_reset_noise.copy()
current_reset_phase_probabilities = curriculum_final_reset_phase_probabilities.copy()
current_forward_reset_phase_probabilities = curriculum_forward_final_reset_phase_probabilities.copy()
current_backward_reset_phase_probabilities = curriculum_backward_final_reset_phase_probabilities.copy()
initial_reset_rejection_attempts = 100
initial_swing_ground_margin_touchdown = 0.006
initial_swing_ground_margin_non_touchdown = 0.025
initial_swing_knee_ground_margin = 0.030
swing_foot_ground_soft_margin = 0.025
swing_knee_ground_soft_margin = 0.030
ground_soft_touchdown_step_scale = 1.5
ground_soft_touchdown_height_scale = 1.5
precontact_step_scale = 1.25
precontact_foot_height_margin = 0.025
forward_ground_soft_multiplier = 1.35
clearance_soft_margin = 0.100
clearance_approach_width = 0.130

torque_hip = 4.00
torque_knee = 0.20
hip_direction_mode = "bidirectional"  # Bidirectional hip torque: -torque, 0, or +torque.
walk_direction_mode = "random"  # "random", "forward", or "backward"

# Active V22_3 uses the same effective curriculum gates as the passive V22_3 scripts.
if hip_direction_mode != "bidirectional":
    curriculum_success_target_start = 0.22
    curriculum_success_target_final = 0.14
    curriculum_ground_failure_gate_start = 0.58
    curriculum_ground_failure_gate_final = 0.40
    curriculum_ground_failure_hard_gate_start = 0.70
    curriculum_ground_failure_hard_gate_final = 0.58
    curriculum_progress_step = 0.0045
    curriculum_regress_step = 0.0012
    curriculum_min_progress_step = 0.0020


def build_discrete_actions():
    actions = []
    for tau_hip in (-torque_hip, 0.0, torque_hip):
        for tau_support_knee in (-torque_knee, 0.0, torque_knee):
            for tau_swing_knee in (-torque_knee, 0.0, torque_knee):
                actions.append([tau_hip, tau_support_knee, tau_swing_knee])
    return np.array(actions, dtype=float)


ACTIONS = build_discrete_actions()
raw_state_dim = 8
state_dim = 12
action_dim = len(ACTIONS)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "v22_3_v9_bi_400_grid")
checkpoint_prefix = "V22_3V9Bi400Grid"
curriculum_save_bucket_width = 0.01

training_log_enabled = True
training_log_csv_path = os.path.join(output_dir, f"{checkpoint_prefix}_training_log.csv")
training_log_jsonl_path = os.path.join(output_dir, f"{checkpoint_prefix}_training_log.jsonl")

# V22_3 can warm-start from a matching V22_2 checkpoint copied into this output directory as *_old.pth.
load_previous_model = True
load_policy_path = os.path.join(output_dir, f"V22_2V9Bi400Grid_Policy_old.pth")
load_critic_path = os.path.join(output_dir, f"V22_2V9Bi400Grid_Critic_old.pth")
success = []
terminal_reasons = []
directional_terminal_reasons = []
directional_phase_terminal_reasons = []
reset_readiness_records = []
experience_buffer_for_policy = []
experience_buffer_for_value = []


model = torch.nn.Sequential(
    torch.nn.Linear(state_dim, hidden_size * 2),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size * 2, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
)

policy = torch.nn.Sequential(
    torch.nn.Linear(state_dim, hidden_size * 2),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_size * 2, hidden_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_size, action_dim),
    torch.nn.Softmax(dim=1),
)

for net in (model, policy):
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight)
            torch.nn.init.zeros_(module.bias)

model.to(device)
policy.to(device)
model.train()
policy.train()

optimizer_value = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


def curriculum_bucket(progress):
    max_bucket = int(round(1.0 / curriculum_save_bucket_width))
    bucket = int(np.floor(float(progress) / curriculum_save_bucket_width + 0.5))
    return int(np.clip(bucket, 0, max_bucket))


def curriculum_from_bucket(bucket):
    return bucket * curriculum_save_bucket_width


def curriculum_bucket_label(bucket):
    return f"c{bucket:03d}"


def curriculum_checkpoint_paths(bucket, success_count):
    label = curriculum_bucket_label(bucket)
    policy_path = os.path.join(output_dir, f"{checkpoint_prefix}_{label}_Policy_{success_count}.pth")
    critic_path = os.path.join(output_dir, f"{checkpoint_prefix}_{label}_Critic_{success_count}.pth")
    return policy_path, critic_path


def curriculum_best_checkpoint_paths(bucket):
    label = curriculum_bucket_label(bucket)
    policy_path = os.path.join(output_dir, f"{checkpoint_prefix}_{label}_Policy_best.pth")
    critic_path = os.path.join(output_dir, f"{checkpoint_prefix}_{label}_Critic_best.pth")
    return policy_path, critic_path


def load_existing_curriculum_bests():
    best_success_by_curriculum = {}
    if not os.path.isdir(output_dir):
        return best_success_by_curriculum

    pattern = re.compile(rf"{re.escape(checkpoint_prefix)}_c(\d{{3}})_Policy_(\d+)\.pth$")
    for file_name in os.listdir(output_dir):
        match = pattern.match(file_name)
        if not match:
            continue
        bucket = int(match.group(1))
        success_count = int(match.group(2))
        best_success_by_curriculum[bucket] = max(
            best_success_by_curriculum.get(bucket, -1),
            success_count,
        )
    return best_success_by_curriculum


def load_state_with_optional_input_expansion(net, checkpoint_path):
    previous_state = torch.load(checkpoint_path, map_location=device)
    current_state = net.state_dict()
    skipped_keys = []

    for key, value in previous_state.items():
        if key not in current_state:
            skipped_keys.append(key)
            continue

        current_value = current_state[key]
        if current_value.shape == value.shape:
            current_state[key] = value
            continue

        can_expand_input = (
            current_value.ndim == 2
            and value.ndim == 2
            and current_value.shape[0] == value.shape[0]
            and current_value.shape[1] == value.shape[1] + 1
        )
        if can_expand_input:
            expanded = current_value.clone()
            expanded[:, : value.shape[1]] = value
            expanded[:, value.shape[1]:] = 0.0
            current_state[key] = expanded
            continue

        skipped_keys.append(key)

    net.load_state_dict(current_state)
    return skipped_keys


def load_previous_model_if_requested():
    if not load_previous_model:
        return

    if not os.path.exists(load_policy_path) or not os.path.exists(load_critic_path):
        print(
            "load_previous_model=True, but previous checkpoint was not found; "
            "training will start from the current initialization. "
            f"Expected: {os.path.basename(load_policy_path)}, {os.path.basename(load_critic_path)}"
        )
        return

    skipped_policy_keys = load_state_with_optional_input_expansion(policy, load_policy_path)
    skipped_critic_keys = load_state_with_optional_input_expansion(model, load_critic_path)
    print(
        "Loaded previous V22_2 checkpoint with commanded-step/height input expansion: "
        f"{os.path.basename(load_policy_path)}, {os.path.basename(load_critic_path)}"
    )
    if skipped_policy_keys:
        print(f"Skipped incompatible policy keys: {skipped_policy_keys}")
    if skipped_critic_keys:
        print(f"Skipped incompatible critic keys: {skipped_critic_keys}")


def curriculum_training_score(success_count, reason_counts):
    ground_count = reason_count_exact_token(reason_counts, "clearance_ground")
    q_limit_count = (
        reason_count_with_prefix(reason_counts, "q1_")
        + reason_count_with_prefix(reason_counts, "q3_")
    )
    return success_count - 0.5 * ground_count - 0.2 * q_limit_count


def save_curriculum_best(bucket, success_count, best_success_by_curriculum, reason_counts=None):
    score = curriculum_training_score(success_count, reason_counts or {})
    old_best = best_success_by_curriculum.get(bucket, -float("inf"))
    if score < old_best:
        return False

    policy_path, critic_path = curriculum_checkpoint_paths(bucket, success_count)
    best_policy_path, best_critic_path = curriculum_best_checkpoint_paths(bucket)
    torch.save(policy.state_dict(), policy_path)
    torch.save(model.state_dict(), critic_path)
    torch.save(policy.state_dict(), best_policy_path)
    torch.save(model.state_dict(), best_critic_path)

    label = curriculum_bucket_label(bucket)
    pattern = re.compile(rf"{re.escape(checkpoint_prefix)}_{label}_(Policy|Critic)_(\d+)\.pth$")
    for file_name in os.listdir(output_dir):
        match = pattern.match(file_name)
        if not match:
            continue
        if int(match.group(2)) == success_count:
            continue
        os.remove(os.path.join(output_dir, file_name))

    best_success_by_curriculum[bucket] = score
    print(
        f"saved curriculum {curriculum_from_bucket(bucket):.2f} best: "
        f"{success_count}/{playing_times}, score {score:.1f}"
    )
    return True


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def compact_json(value):
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)


def append_training_log(row):
    if not training_log_enabled:
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_row = {
        key: compact_json(value) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
        for key, value in row.items()
    }
    write_header = not os.path.exists(training_log_csv_path)
    with open(training_log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
    with open(training_log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(compact_json(row) + "\n")


def angle_error(q, target):
    return np.arctan2(np.sin(q - target), np.cos(q - target))


def line_angle_error(angle, target):
    return 0.5 * np.arctan2(np.sin(2.0 * (angle - target)), np.cos(2.0 * (angle - target)))


def update_curriculum(epoch, progress_override=None):
    global angle_settle, distance_settle, height_settle, current_easy_reset_probability
    global ground_soft_penalty_weight
    global current_easy_reset_noise, current_challenge_reset_noise
    global current_reset_phase_probabilities, current_forward_reset_phase_probabilities
    global current_backward_reset_phase_probabilities
    global posture_error_weight

    if not curriculum_enabled:
        angle_settle = final_angle_settle
        distance_settle = final_distance_settle
        height_settle = final_height_settle
        ground_soft_penalty_weight = ground_soft_penalty_weight_final
        current_easy_reset_probability = 0.0
        current_easy_reset_noise = curriculum_final_easy_reset_noise.copy()
        current_challenge_reset_noise = curriculum_final_challenge_reset_noise.copy()
        current_forward_reset_phase_probabilities = curriculum_forward_final_reset_phase_probabilities.copy()
        current_backward_reset_phase_probabilities = curriculum_backward_final_reset_phase_probabilities.copy()
        current_reset_phase_probabilities = curriculum_final_reset_phase_probabilities.copy()
        posture_error_weight = final_posture_error_weight
        return 1.0

    if progress_override is None:
        progress = min(1.0, epoch / max(1, curriculum_epochs - 1))
    else:
        progress = float(np.clip(progress_override, 0.0, 1.0))
    reset_progress = progress ** reset_curriculum_power
    angle_settle = (1.0 - progress) * curriculum_start_angle_settle + progress * final_angle_settle
    distance_settle = (1.0 - progress) * curriculum_start_distance_settle + progress * final_distance_settle
    height_settle = (1.0 - progress) * curriculum_start_height_settle + progress * final_height_settle
    ground_soft_penalty_weight = (
        (1.0 - progress) * ground_soft_penalty_weight_start
        + progress * ground_soft_penalty_weight_final
    )
    posture_error_weight = (
        (1.0 - progress) * initial_posture_error_weight
        + progress * final_posture_error_weight
    )
    current_easy_reset_probability = (
        (1.0 - reset_progress) * curriculum_start_easy_reset_probability
        + reset_progress * curriculum_final_easy_reset_probability
    )
    current_easy_reset_noise = (
        (1.0 - reset_progress) * curriculum_start_easy_reset_noise
        + reset_progress * curriculum_final_easy_reset_noise
    )
    current_challenge_reset_noise = (
        (1.0 - reset_progress) * curriculum_start_challenge_reset_noise
        + reset_progress * curriculum_final_challenge_reset_noise
    )
    current_forward_reset_phase_probabilities = (
        (1.0 - reset_progress) * curriculum_start_reset_phase_probabilities
        + reset_progress * curriculum_forward_final_reset_phase_probabilities
    )
    current_forward_reset_phase_probabilities /= np.sum(current_forward_reset_phase_probabilities)
    current_backward_reset_phase_probabilities = (
        (1.0 - reset_progress) * curriculum_start_reset_phase_probabilities
        + reset_progress * curriculum_backward_final_reset_phase_probabilities
    )
    current_backward_reset_phase_probabilities /= np.sum(current_backward_reset_phase_probabilities)
    current_reset_phase_probabilities = 0.5 * (
        current_forward_reset_phase_probabilities + current_backward_reset_phase_probabilities
    )
    current_reset_phase_probabilities /= np.sum(current_reset_phase_probabilities)
    return progress


def curriculum_lerp(start, final, progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return (1.0 - progress) * start + progress * final


def curriculum_success_target_for_progress(progress):
    return curriculum_lerp(
        curriculum_success_target_start,
        curriculum_success_target_final,
        progress,
    )


def curriculum_regress_threshold_for_progress(progress):
    return curriculum_lerp(
        curriculum_success_regress_threshold_start,
        curriculum_success_regress_threshold_final,
        progress,
    )


def curriculum_q1_failure_gate_for_progress(progress):
    return curriculum_lerp(
        curriculum_q1_failure_gate_start,
        curriculum_q1_failure_gate_final,
        progress,
    )


def curriculum_clearance_failure_gate_for_progress(progress):
    return curriculum_lerp(
        curriculum_clearance_failure_gate_start,
        curriculum_clearance_failure_gate_final,
        progress,
    )


def curriculum_q3_failure_pause_gate_for_progress(progress):
    return curriculum_lerp(
        curriculum_q3_failure_pause_gate_start,
        curriculum_q3_failure_pause_gate_final,
        progress,
    )


def curriculum_speed_failure_gate_for_progress(progress):
    return curriculum_lerp(
        curriculum_speed_failure_gate_start,
        curriculum_speed_failure_gate_final,
        progress,
    )


def curriculum_ground_failure_gate_for_progress(progress):
    return curriculum_lerp(
        curriculum_ground_failure_gate_start,
        curriculum_ground_failure_gate_final,
        progress,
    )


def curriculum_ground_failure_hard_gate_for_progress(progress):
    return curriculum_lerp(
        curriculum_ground_failure_hard_gate_start,
        curriculum_ground_failure_hard_gate_final,
        progress,
    )


def curriculum_phase_ground_failure_gates(progress):
    return {
        "pre_cross": curriculum_lerp(
            curriculum_pre_ground_failure_gate_start,
            curriculum_pre_ground_failure_gate_final,
            progress,
        ),
        "mid_cross": curriculum_lerp(
            curriculum_mid_ground_failure_gate_start,
            curriculum_mid_ground_failure_gate_final,
            progress,
        ),
        "touchdown": curriculum_lerp(
            curriculum_touchdown_ground_failure_gate_start,
            curriculum_touchdown_ground_failure_gate_final,
            progress,
        ),
    }


def reason_count_with_prefix(reason_counts, prefix):
    count = 0
    for reason, value in reason_counts.items():
        tokens = str(reason).split("+")
        if any(token.startswith(prefix) for token in tokens):
            count += value
    return count


def reason_count_exact_token(reason_counts, token_name):
    count = 0
    for reason, value in reason_counts.items():
        tokens = str(reason).split("+")
        if token_name in tokens:
            count += value
    return count



def phase_reason_count(phase_reason_counts, phase, ground_only=False):
    marker = f"_{phase}_"
    count = 0
    for reason, value in phase_reason_counts.items():
        reason_text = str(reason)
        if marker not in reason_text:
            continue
        if not ground_only:
            count += value
            continue
        terminal_reason = reason_text.split(marker, 1)[1]
        if "clearance_ground" in terminal_reason.split("+"):
            count += value
    return count


def curriculum_phase_ground_failure_rates(phase_reason_counts):
    rates = {}
    for phase in ("pre_cross", "mid_cross", "touchdown"):
        total = phase_reason_count(phase_reason_counts, phase, ground_only=False)
        if total < curriculum_min_phase_ground_gate_samples:
            rates[phase] = None
        else:
            rates[phase] = phase_reason_count(phase_reason_counts, phase, ground_only=True) / total
    return rates


def phase_reason_success_count(phase_reason_counts, phase):
    marker = f"_{phase}_"
    count = 0
    for reason, value in phase_reason_counts.items():
        reason_text = str(reason)
        if marker not in reason_text:
            continue
        terminal_reason = reason_text.split(marker, 1)[1]
        if "success" in terminal_reason.split("+"):
            count += value
    return count


def curriculum_phase_success_rates(phase_reason_counts):
    rates = {}
    for phase in ("pre_cross", "mid_cross", "touchdown"):
        total = phase_reason_count(phase_reason_counts, phase, ground_only=False)
        if total < curriculum_min_phase_ground_gate_samples:
            rates[phase] = None
        else:
            rates[phase] = phase_reason_success_count(phase_reason_counts, phase) / total
    return rates


def curriculum_phase_success_gates(progress):
    return {
        "pre_cross": curriculum_lerp(
            curriculum_pre_success_gate_start,
            curriculum_pre_success_gate_final,
            progress,
        ),
        "mid_cross": curriculum_lerp(
            curriculum_mid_success_gate_start,
            curriculum_mid_success_gate_final,
            progress,
        ),
        "touchdown": curriculum_lerp(
            curriculum_touchdown_success_gate_start,
            curriculum_touchdown_success_gate_final,
            progress,
        ),
    }


def curriculum_phase_progress_slowdown(phase_ground_gates, phase_success_gates):
    phase_items = (
        (
            "pre_cross",
            curriculum_pre_ground_failure_rate_ema,
            curriculum_pre_ground_gate_weight,
            curriculum_pre_success_rate_ema,
            0.25,
        ),
        (
            "mid_cross",
            curriculum_mid_ground_failure_rate_ema,
            curriculum_mid_ground_gate_weight,
            curriculum_mid_success_rate_ema,
            0.35,
        ),
        (
            "touchdown",
            curriculum_touchdown_ground_failure_rate_ema,
            curriculum_touchdown_ground_gate_weight,
            curriculum_touchdown_success_rate_ema,
            0.25,
        ),
    )
    slowdown = 0.0
    for phase, ground_ema, ground_weight, success_ema, success_weight in phase_items:
        if ground_ema is not None:
            ground_excess = max(
                ground_ema - phase_ground_gates[phase] - curriculum_phase_ground_soft_margin,
                0.0,
            )
            slowdown = max(
                slowdown,
                ground_weight * ground_excess / max(curriculum_phase_ground_slowdown_scale, 1e-8),
            )
        if success_ema is not None:
            success_shortfall = max(phase_success_gates[phase] - success_ema, 0.0)
            slowdown = max(
                slowdown,
                success_weight * success_shortfall / max(curriculum_phase_success_slowdown_scale, 1e-8),
            )
    return min(curriculum_phase_ground_max_slowdown, slowdown)


def update_optional_failure_ema(current_ema, raw_rate):
    if raw_rate is None:
        return current_ema
    if current_ema is None:
        return raw_rate
    return curriculum_failure_ema_alpha * raw_rate + (1.0 - curriculum_failure_ema_alpha) * current_ema


def curriculum_phase_ground_excess(phase_ground_gates, phase_ground_rates):
    excess_values = [0.0]
    if phase_ground_rates['pre_cross'] is not None and curriculum_pre_ground_failure_rate_ema is not None:
        excess_values.append(
            curriculum_pre_ground_gate_weight
            * (curriculum_pre_ground_failure_rate_ema - phase_ground_gates['pre_cross'])
        )
    if phase_ground_rates['mid_cross'] is not None and curriculum_mid_ground_failure_rate_ema is not None:
        excess_values.append(
            curriculum_mid_ground_gate_weight
            * (curriculum_mid_ground_failure_rate_ema - phase_ground_gates['mid_cross'])
        )
    if phase_ground_rates['touchdown'] is not None and curriculum_touchdown_ground_failure_rate_ema is not None:
        excess_values.append(
            curriculum_touchdown_ground_gate_weight
            * (curriculum_touchdown_ground_failure_rate_ema - phase_ground_gates['touchdown'])
        )
    return max(excess_values)


def format_optional_rate(value):
    return "n/a" if value is None else f"{value:.3f}"


def curriculum_failure_rates(reason_counts):
    q1_rate = reason_count_with_prefix(reason_counts, "q1_") / playing_times
    q3_rate = reason_count_with_prefix(reason_counts, "q3_") / playing_times
    ground_rate = reason_count_exact_token(reason_counts, "clearance_ground") / playing_times
    clearance_rate = (
        reason_count_with_prefix(reason_counts, "clearance")
        - reason_count_exact_token(reason_counts, "clearance_ground")
    ) / playing_times
    speed_rate = reason_count_with_prefix(reason_counts, "speed") / playing_times
    return q1_rate, q3_rate, clearance_rate, ground_rate, speed_rate


def update_adaptive_curriculum_progress(success_rate, reason_counts, phase_reason_counts=None):
    global curriculum_progress_value, curriculum_success_rate_ema
    global curriculum_q1_failure_rate_ema, curriculum_q3_failure_rate_ema
    global curriculum_clearance_failure_rate_ema, curriculum_ground_failure_rate_ema
    global curriculum_ground_failure_rate_ema_delta, curriculum_speed_failure_rate_ema
    global curriculum_pre_ground_failure_rate_ema, curriculum_mid_ground_failure_rate_ema
    global curriculum_touchdown_ground_failure_rate_ema
    global curriculum_pre_success_rate_ema, curriculum_mid_success_rate_ema
    global curriculum_touchdown_success_rate_ema, curriculum_last_progress_debug

    if not curriculum_enabled or not curriculum_adaptive:
        return success_rate

    phase_reason_counts = {} if phase_reason_counts is None else phase_reason_counts

    if curriculum_success_rate_ema is None:
        curriculum_success_rate_ema = success_rate
    else:
        curriculum_success_rate_ema = (
            curriculum_success_ema_alpha * success_rate
            + (1.0 - curriculum_success_ema_alpha) * curriculum_success_rate_ema
        )

    success_gate = curriculum_success_target_for_progress(curriculum_progress_value)
    regress_gate = curriculum_regress_threshold_for_progress(curriculum_progress_value)
    q1_failure_rate, q3_failure_rate, clearance_failure_rate, ground_failure_rate, speed_failure_rate = curriculum_failure_rates(reason_counts)
    phase_ground_rates = curriculum_phase_ground_failure_rates(phase_reason_counts)
    phase_success_rates = curriculum_phase_success_rates(phase_reason_counts)
    if curriculum_q1_failure_rate_ema is None:
        curriculum_q1_failure_rate_ema = q1_failure_rate
    else:
        curriculum_q1_failure_rate_ema = (
            curriculum_failure_ema_alpha * q1_failure_rate
            + (1.0 - curriculum_failure_ema_alpha) * curriculum_q1_failure_rate_ema
        )
    if curriculum_q3_failure_rate_ema is None:
        curriculum_q3_failure_rate_ema = q3_failure_rate
    else:
        curriculum_q3_failure_rate_ema = (
            curriculum_failure_ema_alpha * q3_failure_rate
            + (1.0 - curriculum_failure_ema_alpha) * curriculum_q3_failure_rate_ema
        )
    if curriculum_clearance_failure_rate_ema is None:
        curriculum_clearance_failure_rate_ema = clearance_failure_rate
    else:
        curriculum_clearance_failure_rate_ema = (
            curriculum_failure_ema_alpha * clearance_failure_rate
            + (1.0 - curriculum_failure_ema_alpha) * curriculum_clearance_failure_rate_ema
        )
    previous_ground_ema = curriculum_ground_failure_rate_ema
    if curriculum_ground_failure_rate_ema is None:
        curriculum_ground_failure_rate_ema = ground_failure_rate
        curriculum_ground_failure_rate_ema_delta = 0.0
    else:
        curriculum_ground_failure_rate_ema = (
            curriculum_failure_ema_alpha * ground_failure_rate
            + (1.0 - curriculum_failure_ema_alpha) * curriculum_ground_failure_rate_ema
        )
        curriculum_ground_failure_rate_ema_delta = curriculum_ground_failure_rate_ema - previous_ground_ema
    if curriculum_speed_failure_rate_ema is None:
        curriculum_speed_failure_rate_ema = speed_failure_rate
    else:
        curriculum_speed_failure_rate_ema = (
            curriculum_failure_ema_alpha * speed_failure_rate
            + (1.0 - curriculum_failure_ema_alpha) * curriculum_speed_failure_rate_ema
        )
    curriculum_pre_ground_failure_rate_ema = update_optional_failure_ema(
        curriculum_pre_ground_failure_rate_ema,
        phase_ground_rates['pre_cross'],
    )
    curriculum_mid_ground_failure_rate_ema = update_optional_failure_ema(
        curriculum_mid_ground_failure_rate_ema,
        phase_ground_rates['mid_cross'],
    )
    curriculum_touchdown_ground_failure_rate_ema = update_optional_failure_ema(
        curriculum_touchdown_ground_failure_rate_ema,
        phase_ground_rates['touchdown'],
    )
    curriculum_pre_success_rate_ema = update_optional_failure_ema(
        curriculum_pre_success_rate_ema,
        phase_success_rates['pre_cross'],
    )
    curriculum_mid_success_rate_ema = update_optional_failure_ema(
        curriculum_mid_success_rate_ema,
        phase_success_rates['mid_cross'],
    )
    curriculum_touchdown_success_rate_ema = update_optional_failure_ema(
        curriculum_touchdown_success_rate_ema,
        phase_success_rates['touchdown'],
    )
    q1_failure_gate = curriculum_q1_failure_gate_for_progress(curriculum_progress_value)
    q3_failure_gate = curriculum_q3_failure_pause_gate_for_progress(curriculum_progress_value)
    clearance_failure_gate = curriculum_clearance_failure_gate_for_progress(curriculum_progress_value)
    ground_failure_gate = curriculum_ground_failure_gate_for_progress(curriculum_progress_value)
    ground_failure_hard_gate = curriculum_ground_failure_hard_gate_for_progress(curriculum_progress_value)
    speed_failure_gate = curriculum_speed_failure_gate_for_progress(curriculum_progress_value)
    phase_ground_gates = curriculum_phase_ground_failure_gates(curriculum_progress_value)
    phase_success_gates = curriculum_phase_success_gates(curriculum_progress_value)
    phase_ground_excess = curriculum_phase_ground_excess(phase_ground_gates, phase_ground_rates)
    phase_progress_slowdown = curriculum_phase_progress_slowdown(phase_ground_gates, phase_success_gates)
    ground_gate_excess = curriculum_ground_failure_rate_ema - ground_failure_gate
    ground_hard_excess = curriculum_ground_failure_rate_ema - ground_failure_hard_gate
    single_direction_curriculum = hip_direction_mode != "bidirectional"
    ground_trend_allowed = curriculum_ground_failure_rate_ema_delta <= curriculum_ground_trend_tolerance
    ground_excess_for_regress = ground_hard_excess
    if single_direction_curriculum and ground_hard_excess <= 0.0 and ground_trend_allowed:
        ground_excess_for_progress = 0.0
    else:
        ground_excess_for_progress = ground_gate_excess

    failure_excess = max(
        curriculum_q1_failure_rate_ema - q1_failure_gate,
        curriculum_clearance_failure_rate_ema - clearance_failure_gate,
        ground_excess_for_regress,
        curriculum_speed_failure_rate_ema - speed_failure_gate,
    )
    q3_pause_excess = curriculum_q3_failure_rate_ema - q3_failure_gate
    curriculum_last_progress_debug = {
        "decision": "hold",
        "step": 0.0,
        "phase_ground_excess": phase_ground_excess,
        "phase_slowdown": phase_progress_slowdown,
        "failure_excess": failure_excess,
        "q3_pause_excess": q3_pause_excess,
    }

    if curriculum_success_rate_ema < regress_gate:
        adaptive_step = curriculum_regress_step
        curriculum_progress_value = max(0.0, curriculum_progress_value - adaptive_step)
        curriculum_last_progress_debug.update({"decision": "regress_success", "step": -adaptive_step})
        return curriculum_success_rate_ema
    if failure_excess > curriculum_failure_progress_margin:
        adaptive_step = curriculum_regress_step * min(1.0, failure_excess / 0.10)
        curriculum_progress_value = max(0.0, curriculum_progress_value - adaptive_step)
        curriculum_last_progress_debug.update({"decision": "regress_failure", "step": -adaptive_step})
        return curriculum_success_rate_ema

    success_error = curriculum_success_rate_ema - success_gate
    if abs(success_error) <= curriculum_progress_deadband:
        curriculum_last_progress_debug.update({"decision": "hold_deadband"})
        return curriculum_success_rate_ema

    if success_error > 0.0:
        progress_failure_excess = max(
            curriculum_q1_failure_rate_ema - q1_failure_gate,
            curriculum_clearance_failure_rate_ema - clearance_failure_gate,
            ground_excess_for_progress,
            curriculum_speed_failure_rate_ema - speed_failure_gate,
        )
        if progress_failure_excess > curriculum_failure_progress_margin or q3_pause_excess > curriculum_failure_gate_deadband:
            curriculum_last_progress_debug.update(
                {
                    "decision": "hold_failure_gate",
                    "failure_excess": progress_failure_excess,
                }
            )
            return curriculum_success_rate_ema
        adaptive_step = curriculum_progress_step * min(1.5, success_error / 0.10)
        slowdown_multiplier = max(1.0 - phase_progress_slowdown, 0.35)
        adaptive_step *= slowdown_multiplier
        if progress_failure_excess <= 0.0 and q3_pause_excess <= 0.0:
            adaptive_step = max(
                adaptive_step,
                curriculum_min_progress_step * max(slowdown_multiplier, 0.50),
            )
        curriculum_progress_value = min(1.0, curriculum_progress_value + adaptive_step)
        curriculum_last_progress_debug.update(
            {
                "decision": "progress",
                "step": adaptive_step,
                "failure_excess": progress_failure_excess,
                "slowdown_multiplier": slowdown_multiplier,
            }
        )
    else:
        adaptive_step = curriculum_regress_step * min(1.0, -success_error / 0.10)
        curriculum_progress_value = max(0.0, curriculum_progress_value - adaptive_step)
        curriculum_last_progress_debug.update({"decision": "regress_success_error", "step": -adaptive_step})

    return curriculum_success_rate_ema


def normalize_state(
    state,
    hip_direction=0.0,
    walk_direction=0.0,
    commanded_step_length=None,
    commanded_step_height=None,
):
    if commanded_step_length is None:
        commanded_step_length = commanded_step_length_for_direction(walk_direction)
    if commanded_step_height is None:
        commanded_step_height = commanded_step_height_for_direction(walk_direction)
    q = state[0::2]
    dq = state[1::2]
    normalized = np.empty(state_dim, dtype=float)
    normalized[:8:2] = (q - q_centers) / q_ranges
    normalized[1:8:2] = dq / speed_range
    normalized[8] = hip_direction
    normalized[9] = walk_direction
    normalized[10] = commanded_step_length / max(commanded_step_length_scale, 1e-8)
    normalized[11] = commanded_step_height / max(commanded_step_height_scale, 1e-8)
    return normalized


def nearest_command_value(value, options):
    options = np.asarray(options, dtype=float)
    return float(options[int(np.argmin(np.abs(options - float(value))))])


def target_q_for_command(walk_direction, commanded_step_length=None, commanded_step_height=None):
    direction_key = 1.0 if walk_direction >= 0.0 else -1.0
    if commanded_step_length is None:
        commanded_step_length = target_step_length_for_direction(walk_direction)
    if commanded_step_height is None:
        commanded_step_height = target_swing_foot_height
    step_key = round(nearest_command_value(commanded_step_length, commanded_step_length_options), 2)
    height_key = round(nearest_command_value(commanded_step_height, commanded_step_height_options), 2)
    return directional_touchdown_q_grid_targets.get(direction_key, {}).get(
        (step_key, height_key),
        directional_touchdown_q_targets[direction_key],
    )


def target_q_for_direction(walk_direction):
    return target_q_for_command(walk_direction)


def target_step_length_for_direction(walk_direction):
    if walk_direction >= 0.0:
        return target_forward_step_length
    return target_backward_step_length


def commanded_step_length_for_direction(walk_direction):
    return target_step_length_for_direction(walk_direction)


def commanded_step_height_for_direction(walk_direction):
    return target_swing_foot_height


def sample_commanded_step_length():
    return float(np.random.choice(commanded_step_length_options))


def sample_commanded_step_height():
    return float(np.random.choice(commanded_step_height_options))


def select_hip_direction():
    if hip_direction_mode == "positive":
        return 1.0
    if hip_direction_mode == "negative":
        return -1.0
    if hip_direction_mode == "random":
        return float(np.random.choice([-1.0, 1.0]))
    return 0.0


def select_walk_direction(hip_direction):
    if walk_direction_mode == "forward":
        return 1.0
    if walk_direction_mode == "backward":
        return -1.0
    return float(np.random.choice([-1.0, 1.0]))


def q_range_violation(q, q_low, q_high):
    return np.maximum(q_low - q, 0.0) + np.maximum(q - q_high, 0.0)


def q_limit_soft_violation(q):
    q_low = np.array(
        [
            np.pi / 2 - q_ranges[0],
            support_knee_limits[0],
            hip_angle_limits[0],
            swing_knee_limits[0],
        ],
        dtype=float,
    )
    q_high = np.array(
        [
            np.pi / 2 + q_ranges[0],
            support_knee_limits[1],
            hip_angle_limits[1],
            swing_knee_limits[1],
        ],
        dtype=float,
    )
    lower = np.maximum((q_low + joint_limit_soft_margin - q) / joint_limit_soft_margin, 0.0)
    upper = np.maximum((q - (q_high - joint_limit_soft_margin)) / joint_limit_soft_margin, 0.0)
    return lower + upper


def speed_soft_violation(dq):
    soft_width = max(terminal_speed_limit - speed_soft_limit, 1e-8)
    return np.maximum(np.abs(dq) - speed_soft_limit, 0.0) / soft_width


def stance_angle_soft_violation(q):
    low_violation = max((stance_angle_safe_low - q[0]) / max(stance_angle_safe_low, 1e-8), 0.0)
    high_width = max(np.pi - stance_angle_safe_high, 1e-8)
    high_violation = max((q[0] - stance_angle_safe_high) / high_width, 0.0)
    return low_violation + high_violation


def q1_low_soft_violation(q):
    return max((q1_low_soft_margin - q[0]) / max(q1_low_soft_margin, 1e-8), 0.0)


def q3_low_soft_violation(q):
    q3_low = hip_angle_limits[0]
    return max((q3_low + q3_low_soft_margin - q[2]) / max(q3_low_soft_margin, 1e-8), 0.0)


def hip_height_soft_violation(metrics):
    if metrics is None:
        return 0.0
    hip_height = metrics.get("hip_height", hip_height_soft_min)
    return max((hip_height_soft_min - hip_height) / max(hip_height_soft_width, 1e-8), 0.0)


def soft_safety_penalty(q, dq, metrics=None):
    q_violation = q_limit_soft_violation(q)
    joint_penalty = np.sum(q_violation ** 2)
    stance_penalty = stance_angle_soft_violation(q) ** 2
    q1_low_penalty = q1_low_soft_violation(q) ** 2
    q3_low_penalty = q3_low_soft_violation(q) ** 2
    hip_height_penalty = hip_height_soft_violation(metrics) ** 2
    speed_penalty = np.sum(speed_soft_violation(dq) ** 2)
    clearance_penalty = 0.0
    ground_soft_penalty = 0.0
    precontact_penalty = 0.0
    touchdown_descent_penalty = 0.0
    if metrics is not None:
        clearance_penalty = metrics.get("clearance_soft_violation", 0.0) ** 2
        ground_multiplier = metrics.get("ground_penalty_multiplier", 1.0)
        ground_soft_penalty = ground_multiplier * metrics.get("ground_soft_violation", 0.0) ** 2
        precontact_penalty = ground_multiplier * metrics.get("precontact_violation", 0.0) ** 2
        touchdown_descent_penalty = metrics.get("touchdown_descent_violation", 0.0) ** 2
    return (
        joint_limit_penalty_weight * joint_penalty
        + stance_angle_penalty_weight * stance_penalty
        + q1_low_soft_penalty_weight * q1_low_penalty
        + q3_low_soft_penalty_weight * q3_low_penalty
        + hip_height_penalty_weight * hip_height_penalty
        + speed_soft_penalty_weight * speed_penalty
        + clearance_soft_penalty_weight * clearance_penalty
        + ground_soft_penalty_weight * ground_soft_penalty
        + precontact_penalty_weight * precontact_penalty
        + touchdown_descent_penalty_weight * touchdown_descent_penalty
    )


def phase_progress_shaping_reward(reset_phase, walk_direction, previous_metrics, metrics):
    if reset_phase == "touchdown" or previous_metrics is None or metrics is None:
        return 0.0

    target_step = max(metrics.get("target_step_length", target_forward_step_length), 1e-8)
    previous_step = previous_metrics.get("signed_step_length", 0.0)
    current_step = metrics.get("signed_step_length", previous_step)
    step_delta = current_step - previous_step
    remaining_step = max(target_step - max(previous_step, 0.0), 0.0)
    useful_step_delta = min(max(step_delta, 0.0), remaining_step)
    step_progress = useful_step_delta / target_step

    previous_foot_height = previous_metrics.get("swing_foot_height", 0.0)
    current_foot_height = metrics.get("swing_foot_height", previous_foot_height)
    foot_quality = np.clip(
        current_foot_height / max(precontact_foot_height_margin, 1e-8),
        0.0,
        1.0,
    )
    knee_quality = np.clip(
        metrics.get("swing_knee_height", 0.0) / max(swing_knee_ground_soft_margin, 1e-8),
        0.0,
        1.0,
    )
    clearance_quality = min(foot_quality, knee_quality)
    direction_multiplier = forward_phase_reward_multiplier if walk_direction >= 0.0 else 1.0
    reward = direction_multiplier * phase_step_progress_reward_weight * step_progress * clearance_quality

    foot_lift_delta = max(current_foot_height - previous_foot_height, 0.0)
    lift_progress = min(foot_lift_delta / max(precontact_foot_height_margin, 1e-8), 1.0)
    reward += direction_multiplier * phase_lift_progress_reward_weight * lift_progress * knee_quality

    if step_progress > 0.0 and reset_phase == "pre_cross":
        reward += direction_multiplier * phase_lift_progress_reward_weight * step_progress * foot_quality
    if walk_direction >= 0.0 and reset_phase == "mid_cross":
        reward += forward_mid_cross_clearance_hold_weight * clearance_quality
        foot_deficit = max(precontact_foot_height_margin - current_foot_height, 0.0)
        reward -= forward_mid_cross_low_foot_penalty_weight * (foot_deficit / max(precontact_foot_height_margin, 1e-8)) ** 2
    if previous_step < 0.0 <= current_step:
        reward += direction_multiplier * phase_crossing_reward_bonus * clearance_quality
    if metrics.get("landing_ready_for_contact", False):
        reward += direction_multiplier * phase_ready_reward_bonus
    else:
        descent = max(previous_foot_height - current_foot_height, 0.0)
        low_height_gate = np.clip(
            (2.0 * precontact_foot_height_margin - current_foot_height)
            / max(2.0 * precontact_foot_height_margin, 1e-8),
            0.0,
            1.0,
        )
        reward -= precontact_descent_penalty_weight * (descent / max(precontact_descent_scale, 1e-8)) ** 2 * low_height_gate
    return reward


def q3_bounds_for_reset_phase(walk_direction, phase):
    if phase == "pre_cross":
        if walk_direction >= 0.0:
            return np.deg2rad(np.array([-45.0, -5.0], dtype=float))
        return np.deg2rad(np.array([5.0, 45.0], dtype=float))
    if phase == "mid_cross":
        if walk_direction >= 0.0:
            return np.deg2rad(np.array([-12.0, 25.0], dtype=float))
        return np.deg2rad(np.array([-25.0, 12.0], dtype=float))
    return None


def reset_phase_center_q(walk_direction, phase, commanded_step_length=None, commanded_step_height=None):
    if phase == "touchdown":
        return target_q_for_command(walk_direction, commanded_step_length, commanded_step_height).copy()

    if walk_direction >= 0.0:
        if phase == "pre_cross":
            return np.deg2rad(np.array([62.0, 38.0, -28.0, 72.0], dtype=float))
        if phase == "mid_cross":
            return np.deg2rad(np.array([60.0, 42.0, 6.0, 82.0], dtype=float))
    else:
        if phase == "pre_cross":
            return np.deg2rad(np.array([72.0, 50.0, 28.0, 62.0], dtype=float))
        if phase == "mid_cross":
            return np.deg2rad(np.array([68.0, 54.0, -6.0, 72.0], dtype=float))

    return target_q_for_command(walk_direction, commanded_step_length, commanded_step_height).copy()



def reset_quota_scale(progress):
    width = max(reset_phase_quota_full_progress - reset_phase_quota_start_progress, 1e-8)
    return float(np.clip((progress - reset_phase_quota_start_progress) / width, 0.0, 1.0))


def reset_epoch_phase_quotas(progress):
    global reset_phase_quota_remaining, reset_phase_quota_targets
    scale = reset_quota_scale(progress)
    targets = {}
    for phase, final_count in reset_phase_quota_forward_final.items():
        quota = int(round(final_count * scale))
        if quota > 0:
            targets[(1.0, phase)] = quota
    for phase, final_count in reset_phase_quota_backward_final.items():
        quota = int(round(final_count * scale))
        if quota > 0:
            targets[(-1.0, phase)] = quota
    reset_phase_quota_targets = dict(targets)
    reset_phase_quota_remaining = dict(targets)
    return reset_phase_quota_targets


def choose_reset_quota_task():
    if not reset_phase_quota_remaining:
        return None
    keys = [key for key, value in reset_phase_quota_remaining.items() if value > 0]
    if not keys:
        return None
    weights = np.asarray([reset_phase_quota_remaining[key] for key in keys], dtype=float)
    weights /= np.sum(weights)
    key = keys[int(np.random.choice(len(keys), p=weights))]
    reset_phase_quota_remaining[key] -= 1
    return key


def format_reset_phase_quota_targets():
    labels = {}
    for (walk_direction, phase), value in sorted(reset_phase_quota_targets.items(), key=lambda item: (item[0][0], item[0][1])):
        direction_label = "forward" if walk_direction >= 0.0 else "backward"
        labels[f"{direction_label}_{phase}"] = value
    return labels




def log_reset_readiness(reset_phase, metrics):
    low_not_ready = (
        not metrics["landing_ready_for_contact"]
        and metrics["swing_foot_height"] < precontact_foot_height_margin
    )
    reset_readiness_records.append(
        {
            "phase": reset_phase,
            "ready": bool(metrics["landing_ready_for_contact"]),
            "near": bool(metrics["near_touchdown_region"]),
            "low_not_ready": bool(low_not_ready),
        }
    )


def format_reset_readiness_records():
    if not reset_readiness_records:
        return "none"
    phases = ["pre_cross", "mid_cross", "touchdown"]
    parts = []
    for phase in phases:
        records = [record for record in reset_readiness_records if record["phase"] == phase]
        if not records:
            continue
        n = len(records)
        ready = sum(record["ready"] for record in records) / n
        near = sum(record["near"] for record in records) / n
        low_not_ready = sum(record["low_not_ready"] for record in records) / n
        parts.append(
            f"{phase}:n={n},ready={ready:.2f},near={near:.2f},low_not_ready={low_not_ready:.2f}"
        )
    return "; ".join(parts) if parts else "none"


def choose_reset_phase(walk_direction=None):
    if walk_direction is None:
        probabilities = np.asarray(current_reset_phase_probabilities, dtype=float)
    elif walk_direction >= 0.0:
        probabilities = np.asarray(current_forward_reset_phase_probabilities, dtype=float)
    else:
        probabilities = np.asarray(current_backward_reset_phase_probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    return str(np.random.choice(reset_phase_names, p=probabilities))


def sample_reset_q(walk_direction, noise, phase="touchdown", commanded_step_length=None, commanded_step_height=None):
    center = reset_phase_center_q(walk_direction, phase, commanded_step_length, commanded_step_height)
    q3_bounds = q3_bounds_for_reset_phase(walk_direction, phase)
    for _ in range(100):
        q = center + np.random.uniform(-noise, noise)
        if q3_bounds is not None:
            q[2] = np.random.uniform(q3_bounds[0], q3_bounds[1])
        if np.all(q >= reset_safe_q_low) and np.all(q <= reset_safe_q_high):
            return q
    q = np.clip(center + np.random.uniform(-noise, noise), reset_safe_q_low, reset_safe_q_high)
    if q3_bounds is not None:
        q[2] = np.clip(q[2], q3_bounds[0], q3_bounds[1])
    return q


class FourLinkBiped:
    """Four-link compass-like biped with stance knee, hip and swing knee motors."""

    def __init__(
        self,
        masses_=masses,
        lengths_=lengths,
        com_lengths_=com_lengths,
        inertias_=inertias,
        hip_mass_=m_hip,
        damping_=damping,
        dt_=dt,
        g_=g,
        actions_=ACTIONS,
    ):
        self.n = 4
        self.m = np.asarray(masses_, dtype=float)
        self.l = np.asarray(lengths_, dtype=float)
        self.lc = np.asarray(com_lengths_, dtype=float)
        self.I = np.asarray(inertias_, dtype=float)
        self.m_hip = hip_mass_
        self.damping = np.asarray(damping_, dtype=float)
        self.dt = dt_
        self.g = g_
        self.actions = np.asarray(actions_, dtype=float)
        self.state = None
        self.reward = None
        self.over = None
        self.check_time = 0
        self.hip_direction = 1.0
        self.walk_direction = 1.0
        self.previous_normalized_error = None
        self.terminal_reason = None
        self.reset_phase = "touchdown"
        self.commanded_step_length = target_forward_step_length
        self.commanded_step_height = target_swing_foot_height

    def _absolute_angles(self, q):
        return absolute_angles_from_q(q)

    def joint_positions(self, q):
        phi = self._absolute_angles(q)
        support_foot = np.array([0.0, 0.0], dtype=float)
        support_knee = support_foot + self.l[0] * np.array([np.cos(phi[0]), np.sin(phi[0])])
        hip = support_knee + self.l[1] * np.array([np.cos(phi[1]), np.sin(phi[1])])
        swing_knee = hip + self.l[2] * np.array([np.cos(phi[2]), np.sin(phi[2])])
        swing_foot = swing_knee + self.l[3] * np.array([np.cos(phi[3]), np.sin(phi[3])])
        return {
            "support_foot": support_foot,
            "support_knee": support_knee,
            "hip": hip,
            "swing_knee": swing_knee,
            "swing_foot": swing_foot,
        }

    def _hip_jacobian(self, q):
        phi = self._absolute_angles(q)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        A = PHI_Q_JACOBIAN
        Jv = np.zeros((2, self.n), dtype=float)
        for k in range(self.n):
            Jv[:, k] += self.l[0] * A[0, k] * np.array([-sin_phi[0], cos_phi[0]])
            Jv[:, k] += self.l[1] * A[1, k] * np.array([-sin_phi[1], cos_phi[1]])
        return Jv

    def touchdown_metrics(self, q):
        points = self.joint_positions(q)
        clearance = self.ground_clearance_metrics(points)

        signed_step_length = self.walk_direction * (
            points["swing_foot"][0] - points["support_foot"][0]
        )
        target_step_length = self.commanded_step_length
        target_step_height = self.commanded_step_height
        step_error = abs(signed_step_length - target_step_length)

        hip_height = points["hip"][1] - points["support_foot"][1]
        swing_foot_height = points["swing_foot"][1] - points["support_foot"][1]
        swing_knee_height = points["swing_knee"][1] - points["support_foot"][1]
        height_error = abs(swing_foot_height - target_step_height)

        q_target = target_q_for_command(self.walk_direction, target_step_length, target_step_height)
        posture_error = np.linalg.norm(angle_error(q, q_target))
        safe_q_violation = q_range_violation(q, touchdown_safe_q_low, touchdown_safe_q_high)
        safe_q_error = np.linalg.norm(safe_q_violation)
        safe_posture = safe_q_error <= 1e-10

        hip_x = points["hip"][0]
        support_thigh_offset = points["support_knee"][0] - hip_x
        swing_thigh_offset = points["swing_knee"][0] - hip_x
        support_left_swing_right = (
            support_thigh_offset < -landing_side_margin
            and swing_thigh_offset > landing_side_margin
        )
        support_right_swing_left = (
            support_thigh_offset > landing_side_margin
            and swing_thigh_offset < -landing_side_margin
        )
        hip_line_separates_thighs = support_left_swing_right or support_right_swing_left
        landing_ready_for_contact = (
            step_error < precontact_step_scale * distance_settle
            and safe_posture
            and hip_line_separates_thighs
        )
        near_touchdown_region = (
            landing_ready_for_contact
            and height_error < ground_soft_touchdown_height_scale * height_settle
        )
        foot_ground_soft_violation = 0.0
        knee_ground_soft_violation = 0.0
        if not near_touchdown_region:
            foot_ground_soft_violation = max(
                (swing_foot_ground_soft_margin - swing_foot_height)
                / max(swing_foot_ground_soft_margin, 1e-8),
                0.0,
            )
            knee_ground_soft_violation = max(
                (swing_knee_ground_soft_margin - swing_knee_height)
                / max(swing_knee_ground_soft_margin, 1e-8),
                0.0,
            )
        ground_soft_violation = np.sqrt(
            foot_ground_soft_violation ** 2 + 0.5 * knee_ground_soft_violation ** 2
        )
        precontact_violation = 0.0
        if not landing_ready_for_contact:
            precontact_violation = max(
                (precontact_foot_height_margin - swing_foot_height)
                / max(precontact_foot_height_margin, 1e-8),
                0.0,
            )
        ground_penalty_multiplier = forward_ground_soft_multiplier if self.walk_direction >= 0.0 else 1.0
        normalized_error = np.sqrt(
            (step_error / distance_settle) ** 2
            + (height_error / height_settle) ** 2
            + posture_error_weight * (posture_error / angle_settle) ** 2
            + (safe_q_error / angle_settle) ** 2
        )

        return {
            "step_error": step_error,
            "height_error": height_error,
            "distance_error": max(step_error, height_error),
            "signed_step_length": signed_step_length,
            "target_step_length": target_step_length,
            "target_step_height": target_step_height,
            "hip_height": hip_height,
            "swing_foot_height": swing_foot_height,
            "swing_knee_height": swing_knee_height,
            "forward_touchdown": points["swing_foot"][0] > points["support_foot"][0] + landing_side_margin,
            "posture_error": posture_error,
            "safe_q_error": safe_q_error,
            "safe_posture": safe_posture,
            "hip_line_separates_thighs": hip_line_separates_thighs,
            "landing_ready_for_contact": landing_ready_for_contact,
            "near_touchdown_region": near_touchdown_region,
            "ground_soft_violation": ground_soft_violation,
            "precontact_violation": precontact_violation,
            "ground_penalty_multiplier": ground_penalty_multiplier,
            "foot_ground_soft_violation": foot_ground_soft_violation,
            "knee_ground_soft_violation": knee_ground_soft_violation,
            "clearance_ok": clearance["clearance_ok"],
            "clearance_violation": clearance["clearance_violation"],
            "clearance_gap": clearance["clearance_gap"],
            "clearance_soft_violation": clearance["clearance_soft_violation"],
            "swing_foot_crossing_stance": clearance["swing_foot_crossing_stance"],
            "swing_leg_ground_failed": clearance["swing_leg_ground_failed"],
            "swing_leg_ground_violation": clearance["swing_leg_ground_violation"],
            "swing_lowest_height": clearance["swing_lowest_height"],
            "normalized_error": normalized_error,
        }


    def touchdown_success_from_metrics(self, metrics):
        clearance_failed = metrics["swing_foot_crossing_stance"] and not metrics["clearance_ok"]
        return (
            metrics["step_error"] < distance_settle
            and metrics["height_error"] < height_settle
            and metrics["safe_posture"]
            and metrics["hip_line_separates_thighs"]
            and not clearance_failed
            and not metrics["swing_leg_ground_failed"]
        )

    def swing_ground_heights(self, q):
        points = self.joint_positions(q)
        support_y = points["support_foot"][1]
        foot_height = points["swing_foot"][1] - support_y
        knee_height = points["swing_knee"][1] - support_y
        return foot_height, knee_height

    def ground_crossing_alpha(self, previous_height, current_height):
        if previous_height > 0.0 and current_height <= 0.0 and previous_height > current_height:
            return float(np.clip(previous_height / (previous_height - current_height), 0.0, 1.0))
        return None

    def add_touchdown_descent_metric(self, metrics, previous_state, current_state):
        previous_foot_height, _ = self.swing_ground_heights(previous_state[0::2])
        current_foot_height = metrics["swing_foot_height"]
        descent_speed = max(
            (previous_foot_height - current_foot_height) / max(self.dt, 1e-8),
            0.0,
        )
        touchdown_descent_violation = 0.0
        if (
            metrics["landing_ready_for_contact"]
            and current_foot_height < touchdown_descent_height_window
        ):
            touchdown_descent_violation = max(
                (descent_speed - touchdown_descent_speed_soft_limit)
                / max(touchdown_descent_speed_soft_limit, 1e-8),
                0.0,
            )
        metrics["touchdown_descent_speed"] = descent_speed
        metrics["touchdown_descent_violation"] = touchdown_descent_violation
        return metrics

    def classify_ground_failure(self, metrics):
        if (
            metrics["swing_knee_height"] < -swing_ground_tolerance
            and metrics["swing_knee_height"] <= metrics["swing_foot_height"]
        ):
            return "clearance_ground+knee_ground"
        if metrics["landing_ready_for_contact"]:
            return "clearance_ground+ground_overshoot"
        return "clearance_ground+foot_ground_before_ready"

    def detect_ground_event(self, previous_state, current_state):
        previous_q = previous_state[0::2]
        current_q = current_state[0::2]
        previous_foot_height, previous_knee_height = self.swing_ground_heights(previous_q)
        current_foot_height, current_knee_height = self.swing_ground_heights(current_q)

        events = []
        foot_alpha = self.ground_crossing_alpha(previous_foot_height, current_foot_height)
        if foot_alpha is not None:
            events.append(("foot", foot_alpha))
        knee_alpha = self.ground_crossing_alpha(previous_knee_height, current_knee_height)
        if knee_alpha is not None:
            events.append(("knee", knee_alpha))
        if not events:
            return None

        contact_part, alpha = min(events, key=lambda item: item[1])
        event_state = previous_state + alpha * (current_state - previous_state)
        event_metrics = self.touchdown_metrics(event_state[0::2])
        self.add_touchdown_descent_metric(event_metrics, previous_state, event_state)
        final_ground_violation = max(
            0.0,
            -min(current_foot_height, current_knee_height) - swing_ground_tolerance,
        )

        if contact_part == "knee":
            return {
                "success": False,
                "state": event_state,
                "metrics": event_metrics,
                "reason": "clearance_ground+knee_ground",
                "ground_violation": final_ground_violation,
            }

        clearance_failed = event_metrics["swing_foot_crossing_stance"] and not event_metrics["clearance_ok"]
        if self.touchdown_success_from_metrics(event_metrics):
            return {
                "success": True,
                "state": event_state,
                "metrics": event_metrics,
                "reason": "success+touchdown_event",
                "ground_violation": 0.0,
            }

        if clearance_failed:
            reason = "clearance_ground+foot_ground_bad_clearance"
        elif event_metrics["landing_ready_for_contact"]:
            reason = "clearance_ground+foot_ground_after_ready"
        else:
            reason = "clearance_ground+foot_ground_before_ready"
        return {
            "success": False,
            "state": event_state,
            "metrics": event_metrics,
            "reason": reason,
            "ground_violation": final_ground_violation,
        }
    def initial_q_in_touchdown_region(self, q):
        metrics = self.touchdown_metrics(q)
        clearance_failed = (
            metrics["swing_foot_crossing_stance"] and not metrics["clearance_ok"]
        ) or metrics["swing_leg_ground_failed"]
        return (
            metrics["step_error"] < distance_settle
            and metrics["height_error"] < height_settle
            and metrics["safe_posture"]
            and metrics["hip_line_separates_thighs"]
            and not clearance_failed
        )

    def initial_q_valid_for_reset_phase(self, q, phase="touchdown"):
        if self.initial_q_clearance_failed(q, phase):
            return False
        metrics = self.touchdown_metrics(q)
        if phase != "touchdown":
            return not self.initial_q_in_touchdown_region(q)

        if self.initial_q_in_touchdown_region(q):
            return False
        step_limit = max(touchdown_reset_step_error_scale * distance_settle,
                         touchdown_reset_min_step_error)
        height_limit = max(touchdown_reset_height_error_scale * height_settle,
                           touchdown_reset_min_height_error)
        return (
            metrics["landing_ready_for_contact"]
            and metrics["step_error"] < step_limit
            and metrics["height_error"] < height_limit
        )

    def initial_q_clearance_failed(self, q, phase="touchdown"):
        metrics = self.touchdown_metrics(q)
        points = self.joint_positions(q)
        support_y = points["support_foot"][1]
        swing_foot_height = points["swing_foot"][1] - support_y
        swing_knee_height = points["swing_knee"][1] - support_y
        required_foot_margin = (
            initial_swing_ground_margin_touchdown
            if phase == "touchdown"
            else initial_swing_ground_margin_non_touchdown
        )
        too_close_to_ground = (
            swing_foot_height < required_foot_margin
            or swing_knee_height < initial_swing_knee_ground_margin
        )
        return (
            metrics["swing_foot_crossing_stance"] and not metrics["clearance_ok"]
        ) or metrics["swing_leg_ground_failed"] or too_close_to_ground

    def nudge_initial_q_out_of_touchdown_region(self, q, phase="touchdown"):
        for delta in np.deg2rad(np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0], dtype=float)):
            for index in (2, 0, 3, 1):
                for sign in (-1.0, 1.0):
                    q_trial = q.copy()
                    q_trial[index] = np.clip(
                        q_trial[index] + sign * delta,
                        reset_safe_q_low[index],
                        reset_safe_q_high[index],
                    )
                    if self.initial_q_clearance_failed(q_trial, phase):
                        continue
                    if self.initial_q_valid_for_reset_phase(q_trial, phase):
                        return q_trial
        q_fallback = q.copy()
        q_fallback[2] = np.clip(
            q_fallback[2] - self.walk_direction * np.deg2rad(30.0),
            reset_safe_q_low[2],
            reset_safe_q_high[2],
        )
        if self.initial_q_valid_for_reset_phase(q_fallback, phase):
            return q_fallback
        for hip_delta in np.deg2rad(np.array([45.0, 60.0, 75.0, 90.0], dtype=float)):
            for knee_delta in np.deg2rad(np.array([8.0, 16.0, 24.0, 32.0, 40.0], dtype=float)):
                q_trial = q.copy()
                q_trial[2] = np.clip(
                    q_trial[2] - self.walk_direction * hip_delta,
                    reset_safe_q_low[2],
                    reset_safe_q_high[2],
                )
                q_trial[3] = np.clip(
                    q_trial[3] + knee_delta,
                    reset_safe_q_low[3],
                    reset_safe_q_high[3],
                )
                if self.initial_q_clearance_failed(q_trial, phase):
                    continue
                if self.initial_q_valid_for_reset_phase(q_trial, phase):
                    return q_trial
        return q_fallback

    def sample_initial_q_away_from_touchdown(self, sampler, phase="touchdown"):
        best_q = None
        best_error = -np.inf
        for _ in range(initial_reset_rejection_attempts):
            q = sampler()
            metrics = self.touchdown_metrics(q)
            if self.initial_q_clearance_failed(q, phase):
                continue
            if self.initial_q_valid_for_reset_phase(q, phase):
                return q
            candidate_score = (
                -metrics["normalized_error"]
                if phase == "touchdown"
                else metrics["normalized_error"]
            )
            if candidate_score > best_error:
                best_error = candidate_score
                best_q = q
        if best_q is not None:
            return self.nudge_initial_q_out_of_touchdown_region(best_q, phase)
        return self.nudge_initial_q_out_of_touchdown_region(
            target_q_for_command(
                self.walk_direction,
                self.commanded_step_length,
                self.commanded_step_height,
            ).copy(),
            phase,
        )


    def ground_clearance_metrics(self, points):
        support_foot = points["support_foot"]
        hip = points["hip"]
        swing_foot = points["swing_foot"]
        swing_knee = points["swing_knee"]

        x_low = min(support_foot[0], hip[0]) - landing_side_margin
        x_high = max(support_foot[0], hip[0]) + landing_side_margin
        swing_foot_crossing_stance = x_low <= swing_foot[0] <= x_high
        distance_to_stance_span = max(x_low - swing_foot[0], swing_foot[0] - x_high, 0.0)
        clearance_approach_weight = max(
            0.0,
            1.0 - distance_to_stance_span / max(clearance_approach_width, 1e-8),
        )

        support_vertical_height = hip[1] - support_foot[1]
        swing_vertical_height = hip[1] - swing_foot[1]
        clearance_gap = support_vertical_height - swing_vertical_height
        clearance_ok = (
            not swing_foot_crossing_stance
            or clearance_gap > min_swing_foot_clearance
        )
        clearance_violation = max(
            0.0,
            min_swing_foot_clearance - clearance_gap,
        )
        clearance_soft_violation = max(
            0.0,
            min_swing_foot_clearance + clearance_soft_margin - clearance_gap,
        ) / max(min_swing_foot_clearance + clearance_soft_margin, 1e-8)
        clearance_soft_violation *= clearance_approach_weight
        swing_foot_height = swing_foot[1] - support_foot[1]
        swing_knee_height = swing_knee[1] - support_foot[1]
        swing_lowest_height = min(swing_foot_height, swing_knee_height)
        swing_leg_ground_violation = max(
            0.0,
            -swing_lowest_height - swing_ground_tolerance,
        )
        swing_leg_ground_failed = swing_leg_ground_violation > 0.0

        return {
            "swing_foot_crossing_stance": swing_foot_crossing_stance,
            "support_vertical_height": support_vertical_height,
            "swing_vertical_height": swing_vertical_height,
            "clearance_gap": clearance_gap,
            "clearance_ok": clearance_ok,
            "clearance_violation": clearance_violation if swing_foot_crossing_stance else 0.0,
            "clearance_soft_violation": clearance_soft_violation,
            "clearance_approach_weight": clearance_approach_weight,
            "swing_lowest_height": swing_lowest_height,
            "swing_leg_ground_violation": swing_leg_ground_violation,
            "swing_leg_ground_failed": swing_leg_ground_failed,
        }

    def _com_jacobians(self, q):
        phi = self._absolute_angles(q)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        A = PHI_Q_JACOBIAN
        jacobians = []

        for i in range(self.n):
            Jv = np.zeros((2, self.n), dtype=float)
            for k in range(self.n):
                for j in range(i):
                    Jv[:, k] += self.l[j] * A[j, k] * np.array([-sin_phi[j], cos_phi[j]])
                Jv[:, k] += self.lc[i] * A[i, k] * np.array([-sin_phi[i], cos_phi[i]])
            Jw = A[i, :].copy()
            jacobians.append((Jv, Jw))

        return jacobians

    def mass_matrix(self, q):
        M = np.zeros((self.n, self.n), dtype=float)
        for i, (Jv, Jw) in enumerate(self._com_jacobians(q)):
            for a in range(self.n):
                for b in range(self.n):
                    M[a, b] += self.m[i] * (Jv[0, a] * Jv[0, b] + Jv[1, a] * Jv[1, b])
                    M[a, b] += self.I[i] * Jw[a] * Jw[b]
        J_hip = self._hip_jacobian(q)
        for a in range(self.n):
            for b in range(self.n):
                M[a, b] += self.m_hip * (J_hip[0, a] * J_hip[0, b] + J_hip[1, a] * J_hip[1, b])
        return 0.5 * (M + M.T)

    def gravity_vector(self, q):
        G = np.zeros(self.n, dtype=float)
        for i, (Jv, _Jw) in enumerate(self._com_jacobians(q)):
            G += self.m[i] * self.g * Jv[1, :]
        G += self.m_hip * self.g * self._hip_jacobian(q)[1, :]
        return G

    def coriolis_vector(self, q, dq):
        eps = 1e-6
        dM = np.zeros((self.n, self.n, self.n), dtype=float)

        for k in range(self.n):
            delta = np.zeros(self.n, dtype=float)
            delta[k] = eps
            dM[:, :, k] = (self.mass_matrix(q + delta) - self.mass_matrix(q - delta)) / (2.0 * eps)

        C = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            total = 0.0
            for j in range(self.n):
                for k in range(self.n):
                    christoffel = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                    total += christoffel * dq[j] * dq[k]
            C[i] = total

        return C

    def generalized_force(self, tau_hip, tau_support_knee, tau_swing_knee):
        # q1 is unactuated. q2, q3 and q4 are exactly the three motor coordinates.
        return np.array([0.0, tau_support_knee, tau_hip, tau_swing_knee], dtype=float)

    def solve_mass_matrix(self, M, rhs):
        A = np.array(M, dtype=float, copy=True)
        b = np.array(rhs, dtype=float, copy=True)
        n = len(b)

        for i in range(n):
            A[i, i] += 1e-10

        for col in range(n):
            pivot = col
            pivot_abs = abs(A[col, col])
            for row in range(col + 1, n):
                candidate_abs = abs(A[row, col])
                if candidate_abs > pivot_abs:
                    pivot = row
                    pivot_abs = candidate_abs

            if pivot_abs < 1e-12:
                A[col, col] += 1e-8
                pivot = col

            if pivot != col:
                A[[col, pivot], :] = A[[pivot, col], :]
                b[col], b[pivot] = b[pivot], b[col]

            pivot_value = A[col, col]
            for row in range(col + 1, n):
                factor = A[row, col] / pivot_value
                A[row, col] = 0.0
                for c in range(col + 1, n):
                    A[row, c] -= factor * A[col, c]
                b[row] -= factor * b[col]

        x = np.zeros(n, dtype=float)
        for row in range(n - 1, -1, -1):
            total = b[row]
            for col in range(row + 1, n):
                total -= A[row, col] * x[col]
            x[row] = total / A[row, row]

        return x

    def solve_constrained_mass_matrix(self, M, rhs, constrained_indices):
        constrained_indices = list(constrained_indices)
        if not constrained_indices:
            return self.solve_mass_matrix(M, rhs)

        constrained = set(constrained_indices)
        free = [index for index in range(self.n) if index not in constrained]
        ddq = np.zeros(self.n, dtype=float)
        if free:
            M_free = M[np.ix_(free, free)]
            rhs_free = rhs[free]
            ddq_free = self.solve_mass_matrix(M_free, rhs_free)
            ddq[free] = ddq_free
        return ddq

    def projected_knee_stop_velocity(self, q, dq):
        qdot = dq.copy()
        eps = 1e-9
        if q[1] <= support_knee_limits[0] + eps and qdot[1] < 0.0:
            qdot[1] = 0.0
        elif q[1] >= support_knee_limits[1] - eps and qdot[1] > 0.0:
            qdot[1] = 0.0

        if q[3] <= swing_knee_limits[0] + eps and qdot[3] < 0.0:
            qdot[3] = 0.0
        elif q[3] >= swing_knee_limits[1] - eps and qdot[3] > 0.0:
            qdot[3] = 0.0
        return qdot

    def active_knee_stop_constraints(self, q, ddq_free):
        eps = 1e-9
        active = []
        if q[1] <= support_knee_limits[0] + eps and ddq_free[1] < 0.0:
            active.append(1)
        elif q[1] >= support_knee_limits[1] - eps and ddq_free[1] > 0.0:
            active.append(1)

        if q[3] <= swing_knee_limits[0] + eps and ddq_free[3] < 0.0:
            active.append(3)
        elif q[3] >= swing_knee_limits[1] - eps and ddq_free[3] > 0.0:
            active.append(3)
        return active

    def dynamics(self, _t, y, tau_hip=0.0, tau_support_knee=0.0, tau_swing_knee=0.0):
        q = y[0::2].copy()
        dq = y[1::2].copy()
        q[1] = np.clip(q[1], support_knee_limits[0], support_knee_limits[1])
        q[3] = np.clip(q[3], swing_knee_limits[0], swing_knee_limits[1])
        qdot = self.projected_knee_stop_velocity(q, dq)

        M = self.mass_matrix(q)
        C = self.coriolis_vector(q, qdot)
        G = self.gravity_vector(q)
        D = self.damping * qdot
        Q = self.generalized_force(tau_hip, tau_support_knee, tau_swing_knee)

        rhs = Q - C - G - D
        ddq_free = self.solve_mass_matrix(M, rhs)
        active_constraints = self.active_knee_stop_constraints(q, ddq_free)
        ddq = self.solve_constrained_mass_matrix(M, rhs, active_constraints)

        dydt = np.empty_like(y)
        dydt[0::2] = qdot
        dydt[1::2] = ddq
        return dydt

    def _enforce_active_knee_stops_in_derivative(self, q, dq, ddq):
        eps = 1e-9

        support_at_extension = q[1] <= support_knee_limits[0] + eps
        support_at_flexion_limit = q[1] >= support_knee_limits[1] - eps
        if support_at_extension and dq[1] < 0.0:
            dq[1] = 0.0
        if support_at_extension and ddq[1] < 0.0:
            ddq[1] = 0.0
        if support_at_flexion_limit and dq[1] > 0.0:
            dq[1] = 0.0
        if support_at_flexion_limit and ddq[1] > 0.0:
            ddq[1] = 0.0

        swing_at_extension = q[3] <= swing_knee_limits[0] + eps
        swing_at_flexion_limit = q[3] >= swing_knee_limits[1] - eps
        if swing_at_extension and dq[3] < 0.0:
            dq[3] = 0.0
        if swing_at_extension and ddq[3] < 0.0:
            ddq[3] = 0.0
        if swing_at_flexion_limit and dq[3] > 0.0:
            dq[3] = 0.0
        if swing_at_flexion_limit and ddq[3] > 0.0:
            ddq[3] = 0.0

    def _rk4(self, y, tau_hip, tau_support_knee, tau_swing_knee):
        k1 = self.dynamics(0.0, y, tau_hip, tau_support_knee, tau_swing_knee)
        k2 = self.dynamics(0.0, y + 0.5 * self.dt * k1, tau_hip, tau_support_knee, tau_swing_knee)
        k3 = self.dynamics(0.0, y + 0.5 * self.dt * k2, tau_hip, tau_support_knee, tau_swing_knee)
        k4 = self.dynamics(0.0, y + self.dt * k3, tau_hip, tau_support_knee, tau_swing_knee)
        return y + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _apply_joint_limits(self):
        q = self.state[0::2]
        dq = self.state[1::2]
        limit_reasons = []

        limits = [
            (np.pi / 2 - q_ranges[0], np.pi / 2 + q_ranges[0]),
            (support_knee_limits[0], support_knee_limits[1]),
            (hip_angle_limits[0], hip_angle_limits[1]),
            (swing_knee_limits[0], swing_knee_limits[1]),
        ]

        for i, (low, high) in enumerate(limits):
            clipped = np.clip(q[i], low, high)
            if clipped != q[i]:
                low_side = q[i] < low
                q[i] = clipped
                dq[i] = 0.0
                if i == 0:
                    limit_reasons.append("q1_low_limit" if low_side else "q1_high_limit")
                elif i == 2:
                    limit_reasons.append("q3_low_limit" if low_side else "q3_high_limit")

        if q[1] <= support_knee_limits[0] and dq[1] < 0.0:
            dq[1] = 0.0
        elif q[1] >= support_knee_limits[1] and dq[1] > 0.0:
            dq[1] = 0.0

        if q[3] <= swing_knee_limits[0] and dq[3] < 0.0:
            dq[3] = 0.0
        elif q[3] >= swing_knee_limits[1] and dq[3] > 0.0:
            dq[3] = 0.0

        self.state[0::2] = q
        self.state[1::2] = dq
        return "+".join(limit_reasons) if limit_reasons else None

    def step(self, act_index):
        tau_hip, tau_support_knee, tau_swing_knee = self.actions[act_index]
        self.check_time += 1

        previous_state = self.state.copy()
        previous_metrics = self.touchdown_metrics(previous_state[0::2])
        self.state = self._rk4(self.state, tau_hip, tau_support_knee, tau_swing_knee)
        clipped_by_limit = self._apply_joint_limits()

        ground_event = self.detect_ground_event(previous_state, self.state)
        if ground_event is not None:
            self.state = ground_event["state"].copy()
            q = self.state[0::2]
            dq = self.state[1::2]
            metrics = ground_event["metrics"]
        else:
            q = self.state[0::2]
            dq = self.state[1::2]
            metrics = self.touchdown_metrics(q)
            self.add_touchdown_descent_metric(metrics, previous_state, self.state)

        clearance_failed = metrics["swing_foot_crossing_stance"] and not metrics["clearance_ok"]
        swing_ground_failed = metrics["swing_leg_ground_failed"]
        previous_error = (
            metrics["normalized_error"]
            if self.previous_normalized_error is None
            else self.previous_normalized_error
        )
        progress = previous_error - metrics["normalized_error"]

        if ground_event is not None:
            touchdown_success = ground_event["success"]
            success_reason = ground_event["reason"]
        else:
            touchdown_success = self.touchdown_success_from_metrics(metrics)
            success_reason = "success"
        speed_failed = np.any(np.abs(dq) > terminal_speed_limit)
        out_of_range = clipped_by_limit is not None or speed_failed

        if touchdown_success:
            direction_bonus = (
                forward_success_reward_bonus
                if metrics["forward_touchdown"]
                else backward_success_reward_bonus
            )
            self.reward = (
                reward_scale * 6.0
                + direction_bonus
                + max(0.0, progress)
                - self.check_time / max_episode_steps
            )
            self.over = True
            self.terminal_reason = success_reason
            success.append(1)
            directional_terminal_reasons.append((self.walk_direction, success_reason))
            directional_phase_terminal_reasons.append((self.walk_direction, self.reset_phase, success_reason))
        elif ground_event is not None:
            ground_violation = max(
                ground_event.get("ground_violation", 0.0),
                metrics.get("swing_leg_ground_violation", 0.0),
            )
            self.reward = -reward_scale * 4.0 - 20.0 * ground_violation
            self.over = True
            self.terminal_reason = ground_event["reason"]
        elif swing_ground_failed:
            self.reward = -reward_scale * 4.0 - 20.0 * metrics["swing_leg_ground_violation"]
            self.over = True
            self.terminal_reason = self.classify_ground_failure(metrics)
        elif clearance_failed:
            self.reward = -reward_scale * 4.0 - 20.0 * metrics["clearance_violation"]
            self.over = True
            self.terminal_reason = "clearance"
        elif out_of_range:
            self.reward = -reward_scale * 6.0
            self.over = True
            if clipped_by_limit is not None and speed_failed:
                self.terminal_reason = f"{clipped_by_limit}+speed"
            elif speed_failed:
                self.terminal_reason = "speed"
            else:
                self.terminal_reason = clipped_by_limit
        elif self.check_time > max_episode_steps:
            self.reward = -reward_scale * 3.0
            self.over = True
            self.terminal_reason = "timeout"
        else:
            self.reward = (
                progress_reward_weight * progress
                + phase_progress_shaping_reward(self.reset_phase, self.walk_direction, previous_metrics, metrics)
                - state_error_penalty_weight * metrics["normalized_error"]
                - soft_safety_penalty(q, dq, metrics)
                - step_penalty
            )
            self.over = False
            self.terminal_reason = None

        self.previous_normalized_error = metrics["normalized_error"]
        if self.over:
            terminal_reasons.append(self.terminal_reason)
            if not str(self.terminal_reason).startswith("success"):
                directional_terminal_reasons.append((self.walk_direction, self.terminal_reason))
                directional_phase_terminal_reasons.append((self.walk_direction, self.reset_phase, self.terminal_reason))

        torque_cost = (
            abs(tau_hip) / max(torque_hip, 1e-8)
            + abs(tau_support_knee) / max(torque_knee, 1e-8)
            + abs(tau_swing_knee) / max(torque_knee, 1e-8)
        )
        self.reward -= action_value_weight * torque_cost

        return self.state, self.reward, self.over

    def reset(self):
        self.hip_direction = select_hip_direction()
        quota_task = choose_reset_quota_task()
        if quota_task is not None:
            self.walk_direction, reset_phase = quota_task
        else:
            self.walk_direction = select_walk_direction(self.hip_direction)
            if np.random.random() < current_easy_reset_probability:
                reset_phase = "touchdown"
            else:
                reset_phase = choose_reset_phase(self.walk_direction)

        self.commanded_step_length = sample_commanded_step_length()
        self.commanded_step_height = sample_commanded_step_height()
        reset_noise = (
            current_easy_reset_noise
            if reset_phase == "touchdown" and quota_task is None
            else current_challenge_reset_noise
        )
        q_initial = self.sample_initial_q_away_from_touchdown(
            lambda: sample_reset_q(
                self.walk_direction,
                reset_noise,
                reset_phase,
                self.commanded_step_length,
                self.commanded_step_height,
            ),
            reset_phase,
        )
        dq_initial = np.array(
            [
                np.random.uniform(-initial_speed_range, initial_speed_range),
                np.random.uniform(-initial_speed_range, initial_speed_range),
                np.random.uniform(-initial_speed_range, initial_speed_range),
                np.random.uniform(-initial_speed_range / 2.0, initial_speed_range / 2.0),
            ],
            dtype=float,
        )

        self.check_time = 0
        self.state = np.empty(raw_state_dim, dtype=float)
        self.state[0::2] = q_initial
        self.state[1::2] = dq_initial
        self.previous_normalized_error = self.touchdown_metrics(q_initial)["normalized_error"]
        self.terminal_reason = None
        self.reset_phase = reset_phase
        log_reset_readiness(self.reset_phase, self.touchdown_metrics(q_initial))
        return self.state


    def define(
        self,
        q1, dq1, q2, dq2, q3, dq3, q4, dq4,
        hip_direction=1.0,
        walk_direction=1.0,
        commanded_step_length=None,
        commanded_step_height=None,
    ):
        self.state = np.array([q1, dq1, q2, dq2, q3, dq3, q4, dq4], dtype=float)
        self.hip_direction = 0.0 if hip_direction_mode == "bidirectional" else (1.0 if hip_direction >= 0.0 else -1.0)
        self.walk_direction = 1.0 if walk_direction >= 0.0 else -1.0
        self.commanded_step_length = (
            commanded_step_length_for_direction(self.walk_direction)
            if commanded_step_length is None else float(commanded_step_length)
        )
        self.commanded_step_height = (
            commanded_step_height_for_direction(self.walk_direction)
            if commanded_step_height is None else float(commanded_step_height)
        )
        self.check_time = 0
        self.previous_normalized_error = self.touchdown_metrics(self.state[0::2])["normalized_error"]
        self.terminal_reason = None
        self.reset_phase = "defined"
        return self.state


env = FourLinkBiped()


def play():
    global experience_buffer_for_policy, experience_buffer_for_value

    state_raw = env.reset()
    over = False
    experience_buffer = []

    while not over:
        state_net = normalize_state(state_raw, env.hip_direction, env.walk_direction, env.commanded_step_length, env.commanded_step_height)
        with torch.no_grad():
            prob = policy(torch.FloatTensor(state_net).reshape(1, state_dim).to(device))[0]
        prob_np = prob.cpu().numpy()
        action_index = np.random.choice(action_dim, p=prob_np)

        next_state_raw, reward, over = env.step(action_index)
        action_prob = prob_np[action_index]
        next_state_net = normalize_state(next_state_raw, env.hip_direction, env.walk_direction, env.commanded_step_length, env.commanded_step_height)

        experience_buffer.append([state_net, action_index, reward, next_state_net, over, action_prob, 0.0])
        state_raw = np.copy(next_state_raw)

    with torch.no_grad():
        next_states = torch.FloatTensor(np.array([item[3] for item in experience_buffer])).to(device)
        states = torch.FloatTensor(np.array([item[0] for item in experience_buffer])).to(device)
        rewards = torch.FloatTensor(np.array([item[2] for item in experience_buffer]).reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(np.array([item[4] for item in experience_buffer]).reshape(-1, 1)).to(device)

        td_target = rewards + gamma * (1.0 - dones) * model(next_states)
        td_delta = (td_target - model(states)).reshape(-1).cpu().numpy()

    advantages = []
    running_advantage = 0.0
    for delta in td_delta[::-1]:
        running_advantage = delta + gamma * gamma_gae * running_advantage
        advantages.append(running_advantage)
    advantages.reverse()

    for i, advantage in enumerate(advantages):
        experience_buffer[i][-1] = advantage

    experience_buffer_for_policy.extend(experience_buffer)
    experience_buffer_for_value.extend(experience_buffer)

    for _ in range(concentrated_sample_times):
        experience_buffer_for_value.append(experience_buffer[-1].copy())


def train():
    global success, terminal_reasons, directional_terminal_reasons, directional_phase_terminal_reasons, reset_readiness_records
    global experience_buffer_for_policy, experience_buffer_for_value

    os.makedirs(output_dir, exist_ok=True)
    load_previous_model_if_requested()
    best_success_by_curriculum = load_existing_curriculum_bests()

    for epoch in range(episode):
        curriculum_progress = update_curriculum(
            epoch,
            curriculum_progress_value if curriculum_adaptive else None,
        )
        reset_epoch_phase_quotas(curriculum_progress)
        success = []
        terminal_reasons = []
        directional_terminal_reasons = []
        directional_phase_terminal_reasons = []
        reset_readiness_records = []
        experience_buffer_for_policy = []
        experience_buffer_for_value = []

        for _ in tqdm(range(playing_times)):
            play()

        print(f"the {epoch + 1} episode")
        print(
            "curriculum "
            f"{curriculum_progress:.2f}, angle {np.rad2deg(angle_settle):.2f} deg, "
            f"tol_step {distance_settle:.3f} m, tol_height {height_settle:.3f} m, "
            f"easy_reset {current_easy_reset_probability:.2f}"
        )
        print(
            f"target grid step {commanded_step_length_options.tolist()} m, "
            f"height {commanded_step_height_options.tolist()} m"
        )
        print(f"reset phase quota targets {format_reset_phase_quota_targets()}")
        print(f"reset readiness {format_reset_readiness_records()}")
        print(f"success {len(success)} times for {playing_times} agents")
        reason_counts = {}
        for reason in terminal_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print(f"terminal reasons {reason_counts}")
        directional_reason_counts = {}
        for walk_direction_value, reason in directional_terminal_reasons:
            direction_label = "forward" if walk_direction_value >= 0.0 else "backward"
            key = f"{direction_label}_{reason}"
            directional_reason_counts[key] = directional_reason_counts.get(key, 0) + 1
        print(f"directional terminal reasons {directional_reason_counts}")
        directional_phase_reason_counts = {}
        for walk_direction_value, reset_phase, reason in directional_phase_terminal_reasons:
            direction_label = "forward" if walk_direction_value >= 0.0 else "backward"
            key = f"{direction_label}_{reset_phase}_{reason}"
            directional_phase_reason_counts[key] = directional_phase_reason_counts.get(key, 0) + 1
        print(f"directional phase terminal reasons {directional_phase_reason_counts}")
        success_count = len(success)
        success_rate = success_count / playing_times
        success_gate = curriculum_success_target_for_progress(curriculum_progress)
        regress_gate = curriculum_regress_threshold_for_progress(curriculum_progress)
        q1_failure_rate, q3_failure_rate, clearance_failure_rate, ground_failure_rate, speed_failure_rate = curriculum_failure_rates(reason_counts)
        q1_failure_gate = curriculum_q1_failure_gate_for_progress(curriculum_progress)
        q3_failure_gate = curriculum_q3_failure_pause_gate_for_progress(curriculum_progress)
        clearance_failure_gate = curriculum_clearance_failure_gate_for_progress(curriculum_progress)
        ground_failure_gate = curriculum_ground_failure_gate_for_progress(curriculum_progress)
        speed_failure_gate = curriculum_speed_failure_gate_for_progress(curriculum_progress)
        ground_hard_gate = curriculum_ground_failure_hard_gate_for_progress(curriculum_progress)
        phase_ground_rates = curriculum_phase_ground_failure_rates(directional_phase_reason_counts)
        phase_success_rates = curriculum_phase_success_rates(directional_phase_reason_counts)
        phase_ground_gates = curriculum_phase_ground_failure_gates(curriculum_progress)
        phase_success_gates = curriculum_phase_success_gates(curriculum_progress)
        bucket = curriculum_bucket(curriculum_progress)

        if epoch > 1:
            save_curriculum_best(bucket, success_count, best_success_by_curriculum, reason_counts)

        smoothed_success_rate = update_adaptive_curriculum_progress(success_rate, reason_counts, directional_phase_reason_counts)
        print(
            "curriculum gate "
            f"raw {success_rate:.3f}, ema {smoothed_success_rate:.3f}, "
            f"target {success_gate:.3f}, regress {regress_gate:.3f}"
        )
        print(
            "failure gate "
            f"q1 raw {q1_failure_rate:.3f}, ema {curriculum_q1_failure_rate_ema:.3f}/{q1_failure_gate:.3f}, "
            f"q3 raw {q3_failure_rate:.3f}, ema {curriculum_q3_failure_rate_ema:.3f}/{q3_failure_gate:.3f}, "
            f"clearance raw {clearance_failure_rate:.3f}, ema {curriculum_clearance_failure_rate_ema:.3f}/{clearance_failure_gate:.3f}, "
            f"ground raw {ground_failure_rate:.3f}, ema {curriculum_ground_failure_rate_ema:.3f}/{ground_failure_gate:.3f}, "
            f"speed raw {speed_failure_rate:.3f}, ema {curriculum_speed_failure_rate_ema:.3f}/{speed_failure_gate:.3f}"
        )
        print(
            "ground curriculum detail "
            f"hard {ground_hard_gate:.3f}, trend {curriculum_ground_failure_rate_ema_delta:.4f}, "
            f"soft_weight {ground_soft_penalty_weight:.3f}"
        )
        print(
            "phase ground gate "
            f"pre raw {format_optional_rate(phase_ground_rates['pre_cross'])}, "
            f"ema {format_optional_rate(curriculum_pre_ground_failure_rate_ema)}/{phase_ground_gates['pre_cross']:.3f}, "
            f"mid raw {format_optional_rate(phase_ground_rates['mid_cross'])}, "
            f"ema {format_optional_rate(curriculum_mid_ground_failure_rate_ema)}/{phase_ground_gates['mid_cross']:.3f}, "
            f"touch raw {format_optional_rate(phase_ground_rates['touchdown'])}, "
            f"ema {format_optional_rate(curriculum_touchdown_ground_failure_rate_ema)}/{phase_ground_gates['touchdown']:.3f}"
        )
        print(
            "phase success gate "
            f"pre raw {format_optional_rate(phase_success_rates['pre_cross'])}, "
            f"ema {format_optional_rate(curriculum_pre_success_rate_ema)}/{phase_success_gates['pre_cross']:.3f}, "
            f"mid raw {format_optional_rate(phase_success_rates['mid_cross'])}, "
            f"ema {format_optional_rate(curriculum_mid_success_rate_ema)}/{phase_success_gates['mid_cross']:.3f}, "
            f"touch raw {format_optional_rate(phase_success_rates['touchdown'])}, "
            f"ema {format_optional_rate(curriculum_touchdown_success_rate_ema)}/{phase_success_gates['touchdown']:.3f}"
        )
        print(
            "curriculum progress detail "
            f"decision {curriculum_last_progress_debug.get('decision', 'n/a')}, "
            f"step {curriculum_last_progress_debug.get('step', 0.0):.4f}, "
            f"phase_slowdown {curriculum_last_progress_debug.get('phase_slowdown', 0.0):.3f}, "
            f"phase_ground_excess {curriculum_last_progress_debug.get('phase_ground_excess', 0.0):.3f}"
        )
        append_training_log(
            {
                "epoch": epoch + 1,
                "script": os.path.basename(__file__),
                "checkpoint_prefix": checkpoint_prefix,
                "curriculum_progress": float(curriculum_progress),
                "curriculum_bucket": int(bucket),
                "angle_settle_deg": float(np.rad2deg(angle_settle)),
                "distance_settle_m": float(distance_settle),
                "height_settle_m": float(height_settle),
                "easy_reset_probability": float(current_easy_reset_probability),
                "target_step_grid_m": commanded_step_length_options,
                "target_height_grid_m": commanded_step_height_options,
                "reset_phase_quota_targets": reset_phase_quota_targets,
                "reset_readiness_summary": format_reset_readiness_records(),
                "playing_times": int(playing_times),
                "success_count": int(success_count),
                "success_rate": float(success_rate),
                "success_rate_ema": float(smoothed_success_rate),
                "success_gate": float(success_gate),
                "regress_gate": float(regress_gate),
                "q1_failure_rate": float(q1_failure_rate),
                "q1_failure_rate_ema": float(curriculum_q1_failure_rate_ema),
                "q1_failure_gate": float(q1_failure_gate),
                "q3_failure_rate": float(q3_failure_rate),
                "q3_failure_rate_ema": float(curriculum_q3_failure_rate_ema),
                "q3_failure_gate": float(q3_failure_gate),
                "clearance_failure_rate": float(clearance_failure_rate),
                "clearance_failure_rate_ema": float(curriculum_clearance_failure_rate_ema),
                "clearance_failure_gate": float(clearance_failure_gate),
                "ground_failure_rate": float(ground_failure_rate),
                "ground_failure_rate_ema": float(curriculum_ground_failure_rate_ema),
                "ground_failure_gate": float(ground_failure_gate),
                "ground_failure_hard_gate": float(ground_hard_gate),
                "ground_failure_ema_delta": float(curriculum_ground_failure_rate_ema_delta),
                "speed_failure_rate": float(speed_failure_rate),
                "speed_failure_rate_ema": float(curriculum_speed_failure_rate_ema),
                "speed_failure_gate": float(speed_failure_gate),
                "ground_soft_penalty_weight": float(ground_soft_penalty_weight),
                "phase_ground_rates": phase_ground_rates,
                "phase_ground_gates": phase_ground_gates,
                "phase_success_rates": phase_success_rates,
                "phase_success_gates": phase_success_gates,
                "terminal_reasons": reason_counts,
                "directional_terminal_reasons": directional_reason_counts,
                "directional_phase_terminal_reasons": directional_phase_reason_counts,
                "curriculum_decision": curriculum_last_progress_debug.get("decision", "n/a"),
                "curriculum_step": float(curriculum_last_progress_debug.get("step", 0.0)),
                "curriculum_phase_slowdown": float(curriculum_last_progress_debug.get("phase_slowdown", 0.0)),
                "curriculum_phase_ground_excess": float(curriculum_last_progress_debug.get("phase_ground_excess", 0.0)),
                "curriculum_failure_excess": float(curriculum_last_progress_debug.get("failure_excess", 0.0)),
                "curriculum_slowdown_multiplier": float(curriculum_last_progress_debug.get("slowdown_multiplier", 1.0)),
            }
        )

        for _ in range(critic_training_times):
            sample_size = min(batch_size, len(experience_buffer_for_value))
            experience_buffer_proxy = random.sample(experience_buffer_for_value, sample_size)

            state = torch.FloatTensor(np.array([item[0] for item in experience_buffer_proxy])).to(device)
            reward = torch.FloatTensor(np.array([item[2] for item in experience_buffer_proxy]).reshape(-1, 1)).to(device)
            next_state = torch.FloatTensor(np.array([item[3] for item in experience_buffer_proxy])).to(device)
            over = torch.FloatTensor(np.array([item[4] for item in experience_buffer_proxy]).reshape(-1, 1)).to(device)

            for _ in range(critic_training_steps):
                value = model(state)
                with torch.no_grad():
                    target = reward + gamma * (1.0 - over) * model(next_state)
                loss = loss_fn(value, target)
                print(f"\r{loss.item():.6f}", end=" ")
                loss.backward()
                optimizer_value.step()
                optimizer_value.zero_grad()

        state = torch.FloatTensor(np.array([item[0] for item in experience_buffer_for_policy])).to(device)
        old_prob = torch.FloatTensor(np.array([item[5] for item in experience_buffer_for_policy]).reshape(-1, 1)).to(device)
        action_current = torch.LongTensor(np.array([item[1] for item in experience_buffer_for_policy]).reshape(-1, 1)).to(device)

        advantages = np.array([item[6] for item in experience_buffer_for_policy], dtype=float).reshape(-1, 1)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = torch.FloatTensor(advantages).to(device)

        for _ in range(actor_training_times):
            probs = policy(state)
            new_prob = probs.gather(dim=1, index=action_current)
            ratio = new_prob / (old_prob + 1e-10)
            loss1 = ratio * advantages
            loss2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            loss = -torch.min(loss1, loss2).mean() - policy_entropy_coefficient * entropy

            loss.backward()
            optimizer_policy.step()
            optimizer_policy.zero_grad()

        torch.save(policy.state_dict(), os.path.join(output_dir, f"{checkpoint_prefix}_Policy.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir, f"{checkpoint_prefix}_Critic.pth"))


if __name__ == "__main__":
    train()





