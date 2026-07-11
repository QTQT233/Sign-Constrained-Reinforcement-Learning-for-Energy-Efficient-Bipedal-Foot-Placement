"""Output-validated reconstruction of the legacy V22_3 four-link evaluation.

This script is intentionally self-contained. It does not import any training
script at runtime, so the paper evaluation is not silently affected when a
training file changes. All parameters that are likely to be edited for a paper
comparison are collected in the first block below. The exact 5 July source was
not retained; this reconstruction restores the V22_3 touchdown posture bounds
and legacy output label in the surviving extended evaluator. Release validation
must compare its results with the archived 5 July trial files.
"""

import csv
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# User-editable paper evaluation settings
# =============================================================================
# Evaluation curriculum. The default is the full training domain.
CURRICULUM_PROGRESS = 1.0

# Actuator limits used by this evaluation. For the 4 Nm comparison keep
# HIP_TORQUE_NM = 4.0 and KNEE_TORQUE_NM = 0.20.
HIP_TORQUE_NM = 4.0
KNEE_TORQUE_NM = 0.20

# Optional command overrides. Set either value to a float to evaluate one
# command; leave both None to evaluate the full V22_3 3x3 command grid below.
COMMAND_STEP_LENGTH_OVERRIDE_M = None
COMMAND_STEP_HEIGHT_OVERRIDE_M = None

# V22_3 trained command grid. These values are part of the policy input.
COMMAND_STEP_LENGTH_GRID_M = [0.30, 0.35, 0.40]
COMMAND_STEP_HEIGHT_GRID_M = [0.00, 0.01, 0.02]

# Model files. Edit only these paths/names when switching checkpoints.
MODEL_ROOT = Path(r"D:\L&S\Mas\Project\Paper1\Knee\Four_link")
ACTIVE_POLICY_NAME = "v22_3_active_bidirectional_4Nm_grid"
ACTIVE_POLICY_PATH = MODEL_ROOT / "v22_3_v9_bi_400_grid" / "V22_3V9Bi400Grid_c090_Policy_best.pth"
ACTIVE_FULL_POLICY_NAME = "v22_3_active_full_4Nm_grid"
ACTIVE_FULL_POLICY_PATH = MODEL_ROOT / "v22_3_v9_active_full_400_grid" / "V22_3V9ActiveFull400Grid_c090_Policy_best.pth"
PER_STEP_SIGN_POLICY_NAME = "v22_3_per_step_sign_4Nm_grid"
PER_STEP_SIGN_POLICY_PATH = MODEL_ROOT / "v22_3_v9_per_step_sign_400_grid" / "V22_3V9PerStepSign400Grid_c091_Policy_258.pth"
PASSIVE_POS_POLICY_NAME = "v22_3_passive_positive_4Nm_grid"
PASSIVE_POS_POLICY_PATH = MODEL_ROOT / "v22_3_v9_uni_pos_400_grid" / "V22_3V9UniPos400Grid_c097_Policy_best.pth"
PASSIVE_NEG_POLICY_NAME = "v22_3_passive_negative_4Nm_grid"
PASSIVE_NEG_POLICY_PATH = MODEL_ROOT / "v22_3_v9_uni_neg_400_grid" / "V22_3V9UniNeg400Grid_c100_Policy_best.pth"
PASSIVE_SIGN_SELECTOR_NAME = "v22_3_passive_sign_selector_4Nm_grid"
PASSIVE_SIGN_SELECTOR_PATH = MODEL_ROOT / "paper_four_link_passive_sign_selector_v22_3_grid.pth"

# Extra active-style controllers to include in the same sampled initial states.
# They are compared against ACTIVE_POLICY_PATH in active_reference_* outputs.
INCLUDE_ACTIVE_FULL_POLICY = True
INCLUDE_PER_STEP_SIGN_POLICY = True

# Passive comparisons to run from the same sampled initial states.
#   "sign_selector":    online selector choosing positive/negative passive policy
#                       from the initial normalized state.
#   "oracle_best_cmt":  offline lower-bound envelope; runs positive and negative
#                       passive policies and reports the successful lower-Cmt one.
#   "positive":         online positive-only passive controller.
#   "negative":         online negative-only passive controller.
PASSIVE_POLICY_MODES = ["sign_selector", "oracle_best_cmt"]

# Kept only as a fallback if PASSIVE_POLICY_MODES is accidentally empty.
PASSIVE_POLICY_MODE = "oracle_best_cmt"

# Sampling/evaluation controls.
OUTPUT_DIR = MODEL_ROOT / "paper_four_link_reachability_cmt_v22_3_grid_4Nm_extended_legacy_reconstructed"
EPISODES_PER_GRID_CELL_DIRECTION = 40
RANDOM_SEEDS = [20260702, 20260719, 20260803]
WALK_DIRECTIONS = [1.0, -1.0]
RESET_PHASE_MODE = "random"  # "random", "pre_cross", "mid_cross", "touchdown"
MAX_STEPS = 500
DETERMINISTIC_POLICY = True
MIN_NONZERO_TORQUE_STEPS = 1
MIN_COM_DISPLACEMENT_M = 0.001
REQUIRE_NONZERO_TORQUE_FOR_CMT = True
REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT = True
CMT_EPS_DISPLACEMENT_M = 1e-8
USE_CUDA_IF_AVAILABLE = True

# Reporting label stored in the CSV outputs.
EVALUATION_LABEL = "4DoF_V22_3_3x3_grid_4Nm_curriculum1_extended_active_passive_selector"

# Representative initial-state export. The score is active_cmt - passive_cmt,
# so larger positive values mean the passive controller used less positive work.
REPRESENTATIVE_STATE_COUNT = 80
SUCCESSFUL_STATE_EXPORT_COUNT = 160
REPRESENTATIVE_MIN_COM_DISPLACEMENT_M = 0.030
REPRESENTATIVE_MIN_DISPLACEMENT_BALANCE_RATIO = 0.60
REPRESENTATIVE_REQUIRE_PASSIVE_LOWER_CMT = True


# =============================================================================
# Fixed model/environment parameters copied explicitly into this paper script
# =============================================================================
hidden_size = 512
dt = 0.01
raw_state_dim = 8
state_dim = 12

# These reward-related values are retained only because curriculum/reset helper
# functions reference them. Cmt evaluation does not use PPO rewards.
playing_times = 500
joint_limit_penalty_weight = 0.05
speed_soft_penalty_weight = 0.140
stance_angle_penalty_weight = 0.18
q1_low_soft_penalty_weight = 0.22
q3_low_soft_penalty_weight = 0.05
hip_height_penalty_weight = 0.08
clearance_soft_penalty_weight = 0.28
ground_soft_penalty_weight_start = 0.06
ground_soft_penalty_weight_final = 0.18
ground_soft_penalty_weight = ground_soft_penalty_weight_final
precontact_penalty_weight = 0.12
phase_step_progress_reward_weight = 1.35
phase_lift_progress_reward_weight = 0.22
phase_crossing_reward_bonus = 0.40
phase_ready_reward_bonus = 0.20
precontact_descent_penalty_weight = 0.16
forward_phase_reward_multiplier = 1.25
forward_mid_cross_clearance_hold_weight = 0.025
forward_mid_cross_low_foot_penalty_weight = 0.10
touchdown_descent_penalty_weight = 0.100
touchdown_reset_step_error_scale = 2.5
touchdown_reset_height_error_scale = 5.0
touchdown_reset_min_step_error = 0.07
touchdown_reset_min_height_error = 0.035
touchdown_descent_speed_soft_limit = 0.35
touchdown_descent_height_window = 0.035
precontact_descent_scale = 0.025
reset_readiness_records = []

DEVICE = torch.device("cuda:0" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")

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
final_step_settle = 0.04
final_height_settle = 0.018
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
commanded_step_length_options = np.array(COMMAND_STEP_LENGTH_GRID_M, dtype=float)
commanded_step_height_options = np.array(COMMAND_STEP_HEIGHT_GRID_M, dtype=float)
commanded_step_height_scale = 0.05
target_forward_q = np.deg2rad(np.array([46.1560, 22.8901, 61.4918, 60.7154], dtype=float))
target_backward_q = np.deg2rad(np.array([71.8281, 59.8427, -58.4970, 29.7468], dtype=float))
directional_touchdown_q_targets = {
    1.0: target_forward_q,
    -1.0: target_backward_q,
}

# Target postures matched to the commanded 3x3 step-length/height grid.
# They are used for reset centers and posture-error shaping. Success uses the
# commanded task-space step length/height plus safety/contact constraints.
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
curriculum_success_target_start = 0.26
curriculum_success_target_final = 0.18
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
curriculum_progress_step = 0.006
curriculum_regress_step = 0.0015
curriculum_progress_deadband = 0.010
curriculum_q1_failure_gate_start = 0.48
curriculum_q1_failure_gate_final = 0.38
curriculum_clearance_failure_gate_start = 0.42
curriculum_clearance_failure_gate_final = 0.31
curriculum_ground_failure_gate_start = 0.56
curriculum_ground_failure_gate_final = 0.40
curriculum_q3_failure_pause_gate_start = 0.24
curriculum_q3_failure_pause_gate_final = 0.14
curriculum_speed_failure_gate_start = 0.18
curriculum_speed_failure_gate_final = 0.10
curriculum_ground_failure_hard_gate_start = 0.68
curriculum_ground_failure_hard_gate_final = 0.56
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
curriculum_min_progress_step = 0.0025
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



# Actuator action tables are explicit here, not imported from a training file.
torque_hip = HIP_TORQUE_NM
torque_knee = KNEE_TORQUE_NM
hip_direction_mode = "bidirectional"
walk_direction_mode = "random"


def build_active_actions():
    actions = []
    for tau_hip in (-torque_hip, 0.0, torque_hip):
        for tau_support_knee in (-torque_knee, 0.0, torque_knee):
            for tau_swing_knee in (-torque_knee, 0.0, torque_knee):
                actions.append([tau_hip, tau_support_knee, tau_swing_knee])
    return np.asarray(actions, dtype=float)


def build_passive_actions():
    actions = []
    for hip_activation in (0.0, 1.0):
        for tau_support_knee in (-torque_knee, 0.0, torque_knee):
            for tau_swing_knee in (-torque_knee, 0.0, torque_knee):
                actions.append([hip_activation, tau_support_knee, tau_swing_knee])
    return np.asarray(actions, dtype=float)


def build_per_step_sign_actions():
    actions = []
    for hip_sign, hip_activation in ((-1.0, 1.0), (0.0, 0.0), (1.0, 1.0)):
        for tau_support_knee in (-torque_knee, 0.0, torque_knee):
            for tau_swing_knee in (-torque_knee, 0.0, torque_knee):
                actions.append([hip_sign, hip_activation, tau_support_knee, tau_swing_knee])
    return np.asarray(actions, dtype=float)


ACTIVE_ACTIONS = build_active_actions()
PASSIVE_ACTIONS = build_passive_actions()
PER_STEP_SIGN_ACTIONS = build_per_step_sign_actions()
ACTIONS = ACTIVE_ACTIONS
action_dim = len(ACTIONS)

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


def command_grid_values():
    if COMMAND_STEP_LENGTH_OVERRIDE_M is None:
        lengths = list(map(float, commanded_step_length_options))
    else:
        lengths = [float(COMMAND_STEP_LENGTH_OVERRIDE_M)]
    if COMMAND_STEP_HEIGHT_OVERRIDE_M is None:
        heights = list(map(float, commanded_step_height_options))
    else:
        heights = [float(COMMAND_STEP_HEIGHT_OVERRIDE_M)]
    return lengths, heights


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
        controller_mode_="active",
        hip_sign_=0.0,
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
        self.controller_mode = str(controller_mode_)
        self.hip_sign = 0.0 if hip_sign_ == 0.0 else (1.0 if hip_sign_ > 0.0 else -1.0)
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

    def action_to_torques(self, act_index):
        action = self.actions[int(act_index)]
        if self.controller_mode == "active":
            tau_hip, tau_support_knee, tau_swing_knee = action
        elif self.controller_mode == "passive":
            hip_activation, tau_support_knee, tau_swing_knee = action
            tau_hip = self.hip_sign * torque_hip * hip_activation
        elif self.controller_mode == "per_step_sign":
            hip_sign, hip_activation, tau_support_knee, tau_swing_knee = action
            tau_hip = hip_sign * torque_hip * hip_activation
        else:
            raise ValueError(f"Unknown controller_mode: {self.controller_mode}")
        return np.asarray([tau_support_knee, tau_hip, tau_swing_knee], dtype=float)

    def step(self, act_index):
        tau_support_knee, tau_hip, tau_swing_knee = self.action_to_torques(act_index)
        self.check_time += 1

        previous_state = self.state.copy()
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

        if ground_event is not None:
            touchdown_success = bool(ground_event["success"])
            success_reason = str(ground_event["reason"])
        else:
            touchdown_success = bool(self.touchdown_success_from_metrics(metrics))
            success_reason = "success"

        clearance_failed = metrics["swing_foot_crossing_stance"] and not metrics["clearance_ok"]
        swing_ground_failed = metrics["swing_leg_ground_failed"]
        speed_failed = bool(np.any(np.abs(dq) > terminal_speed_limit))
        out_of_range = clipped_by_limit is not None or speed_failed

        if touchdown_success:
            self.over = True
            self.terminal_reason = success_reason
        elif ground_event is not None:
            self.over = True
            self.terminal_reason = str(ground_event["reason"])
        elif swing_ground_failed:
            self.over = True
            self.terminal_reason = self.classify_ground_failure(metrics)
        elif clearance_failed:
            self.over = True
            self.terminal_reason = "clearance"
        elif out_of_range:
            self.over = True
            if clipped_by_limit is not None and speed_failed:
                self.terminal_reason = f"{clipped_by_limit}+speed"
            elif speed_failed:
                self.terminal_reason = "speed"
            else:
                self.terminal_reason = clipped_by_limit
        elif self.check_time > max_episode_steps:
            self.over = True
            self.terminal_reason = "timeout"
        else:
            self.over = False
            self.terminal_reason = None

        self.previous_normalized_error = metrics["normalized_error"]
        self.reward = 0.0
        return self.state, self.reward, self.over

    def reset(self):
        self.hip_direction = select_hip_direction()
        quota_task = choose_reset_quota_task()
        if quota_task is not None:
            self.walk_direction, reset_phase = quota_task
            self.commanded_step_length = sample_commanded_step_length()
            self.commanded_step_height = sample_commanded_step_height()
            q_initial = self.sample_initial_q_away_from_touchdown(
                lambda: sample_reset_q(
                    self.walk_direction,
                    current_challenge_reset_noise,
                    reset_phase,
                    self.commanded_step_length,
                    self.commanded_step_height,
                ),
                reset_phase,
            )
        else:
            self.walk_direction = select_walk_direction(self.hip_direction)
            self.commanded_step_length = sample_commanded_step_length()
            self.commanded_step_height = sample_commanded_step_height()
            if np.random.random() < current_easy_reset_probability:
                reset_phase = "touchdown"
                q_initial = self.sample_initial_q_away_from_touchdown(
                    lambda: sample_reset_q(
                        self.walk_direction,
                        current_easy_reset_noise,
                        reset_phase,
                        self.commanded_step_length,
                        self.commanded_step_height,
                    ),
                    reset_phase,
                )
            else:
                reset_phase = choose_reset_phase(self.walk_direction)
                q_initial = self.sample_initial_q_away_from_touchdown(
                    lambda: sample_reset_q(
                        self.walk_direction,
                        current_challenge_reset_noise,
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


    def define(self, q1, dq1, q2, dq2, q3, dq3, q4, dq4,
               hip_direction=0.0, walk_direction=1.0, commanded_step_length=None,
               commanded_step_height=None, reset_phase="defined"):
        self.state = np.array([q1, dq1, q2, dq2, q3, dq3, q4, dq4], dtype=float)
        self.hip_direction = (
            0.0
            if self.controller_mode in ("active", "per_step_sign")
            else (1.0 if hip_direction >= 0.0 else -1.0)
        )
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
        self.reset_phase = reset_phase
        self.reward = 0.0
        self.over = False
        return self.state




# =============================================================================
# Paper Cmt evaluator
# =============================================================================

def eval_commanded_step_length(walk_direction):
    if COMMAND_STEP_LENGTH_OVERRIDE_M is not None:
        return float(COMMAND_STEP_LENGTH_OVERRIDE_M)
    return commanded_step_length_for_direction(walk_direction)


def eval_commanded_step_height(walk_direction):
    if COMMAND_STEP_HEIGHT_OVERRIDE_M is not None:
        return float(COMMAND_STEP_HEIGHT_OVERRIDE_M)
    return commanded_step_height_for_direction(walk_direction)


def make_policy(action_count):
    return torch.nn.Sequential(
        torch.nn.Linear(state_dim, hidden_size * 2),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size * 2, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, action_count),
        torch.nn.Softmax(dim=1),
    )


def make_sign_selector():
    return torch.nn.Sequential(
        torch.nn.Linear(state_dim, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 2),
    )


def load_policy(path, action_count, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} policy: {path}")
    policy = make_policy(action_count).to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_sign_selector(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    selector = make_sign_selector().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    selector.load_state_dict(state_dict)
    selector.eval()
    return selector


def choose_action(policy, state_net):
    with torch.no_grad():
        probs = policy(torch.as_tensor(state_net, dtype=torch.float32, device=DEVICE).reshape(1, state_dim))[0]
    probs_np = probs.detach().cpu().numpy()
    if DETERMINISTIC_POLICY:
        return int(np.argmax(probs_np)), float(np.max(probs_np))
    action_index = int(np.random.choice(len(probs_np), p=probs_np))
    return action_index, float(probs_np[action_index])


def choose_passive_sign(selector, initial_state, walk_direction, commanded_step_length, commanded_step_height):
    state_net = normalize_state(
        initial_state,
        0.0,
        walk_direction,
        commanded_step_length,
        commanded_step_height,
    )
    with torch.no_grad():
        logits = selector(torch.as_tensor(state_net, dtype=torch.float32, device=DEVICE).reshape(1, state_dim))[0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    label = int(np.argmax(probs))
    if label == 0:
        return "positive", float(probs[0])
    return "negative", float(probs[1])


def center_of_mass(q):
    """COM in the current stance-foot frame.

    For this single-step paper metric, D is the directional COM displacement in
    the same stance frame between the shared initial state and terminal state.
    It is not intended to be a multi-step world-frame odometry estimate.
    """
    phi = absolute_angles_from_q(q)
    support_foot = np.array([0.0, 0.0], dtype=float)
    support_knee = support_foot + l_support_shank * np.array([np.cos(phi[0]), np.sin(phi[0])])
    hip = support_knee + l_support_thigh * np.array([np.cos(phi[1]), np.sin(phi[1])])
    swing_knee = hip + l_swing_thigh * np.array([np.cos(phi[2]), np.sin(phi[2])])

    support_shank_com = support_foot + 0.5 * l_support_shank * np.array([np.cos(phi[0]), np.sin(phi[0])])
    support_thigh_com = support_knee + 0.5 * l_support_thigh * np.array([np.cos(phi[1]), np.sin(phi[1])])
    swing_thigh_com = hip + 0.5 * l_swing_thigh * np.array([np.cos(phi[2]), np.sin(phi[2])])
    swing_shank_com = swing_knee + 0.5 * l_swing_shank * np.array([np.cos(phi[3]), np.sin(phi[3])])

    weighted = (
        m_support_shank * support_shank_com
        + m_support_thigh * support_thigh_com
        + m_swing_thigh * swing_thigh_com
        + m_swing_shank * swing_shank_com
        + m_hip * hip
    )
    return weighted / total_mass()


def total_mass():
    return float(np.sum(masses) + m_hip)


def projected_motor_rates(env, state):
    """Return [support knee, hip, swing knee] rates after stop projection.

    The generalized coordinates are q=[q1, q2, q3, q4]. Motors apply generalized
    torques [q2, q3, q4], so their relative angular rates are [dq2, dq3, dq4].
    Near a mechanical stop, the dynamics projects blocked knee velocities to
    zero; using the same projection avoids counting artificial motor work while
    the stop is clamped.
    """
    q = state[0::2].copy()
    dq = state[1::2].copy()
    qdot = env.projected_knee_stop_velocity(q, dq)
    return qdot[[1, 2, 3]]


@dataclass
class RolloutResult:
    eval_label: str
    seed: int
    episode_index: int
    walk_direction: float
    reset_phase: str
    controller: str
    policy_name: str
    passive_mode: str
    passive_selected_from: str
    curriculum_progress: float
    hip_torque_nm: float
    knee_torque_nm: float
    commanded_step_length_m: float
    commanded_step_height_m: float
    success: bool
    valid_for_cmt: bool
    terminal_reason: str
    steps: int
    nonzero_torque_steps: int
    energy_j: float
    support_knee_energy_j: float
    hip_energy_j: float
    swing_knee_energy_j: float
    signed_com_displacement_m: float
    cmt: float
    support_knee_cmt: float
    hip_cmt: float
    swing_knee_cmt: float
    initial_q1_deg: float
    initial_q2_deg: float
    initial_q3_deg: float
    initial_q4_deg: float
    initial_dq1_rad_s: float
    initial_dq2_rad_s: float
    initial_dq3_rad_s: float
    initial_dq4_rad_s: float
    final_q1_deg: float
    final_q2_deg: float
    final_q3_deg: float
    final_q4_deg: float


def make_env(controller, hip_sign=0.0):
    if controller in ("active", "active_full"):
        return FourLinkBiped(actions_=ACTIVE_ACTIONS, controller_mode_="active", hip_sign_=0.0)
    if controller == "per_step_sign":
        return FourLinkBiped(actions_=PER_STEP_SIGN_ACTIONS, controller_mode_="per_step_sign", hip_sign_=0.0)
    return FourLinkBiped(actions_=PASSIVE_ACTIONS, controller_mode_="passive", hip_sign_=hip_sign)


def rollout_policy(
    initial_state,
    walk_direction,
    reset_phase,
    commanded_step_length,
    commanded_step_height,
    policy,
    controller,
    policy_name,
    seed,
    episode_index,
    passive_selected_from="",
    controller_label=None,
    passive_mode_label=None,
):
    hip_sign = 0.0
    if controller == "passive_positive":
        hip_sign = 1.0
    elif controller == "passive_negative":
        hip_sign = -1.0

    if controller in ("active", "active_full"):
        env_controller = "active"
    elif controller == "per_step_sign":
        env_controller = "per_step_sign"
    else:
        env_controller = "passive"

    env = make_env(env_controller, hip_sign=hip_sign)
    env.define(*initial_state, hip_direction=hip_sign, walk_direction=walk_direction,
               commanded_step_length=commanded_step_length,
               commanded_step_height=commanded_step_height,
               reset_phase=reset_phase)

    initial_q = env.state[0::2].copy()
    initial_dq = env.state[1::2].copy()
    initial_com = center_of_mass(initial_q)
    joint_energy = np.zeros(3, dtype=float)  # [support knee, hip, swing knee]
    nonzero_torque_steps = 0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        state_net = normalize_state(
            env.state,
            env.hip_direction,
            env.walk_direction,
            env.commanded_step_length,
            env.commanded_step_height,
        )
        action_index, _ = choose_action(policy, state_net)
        torques = env.action_to_torques(action_index)
        previous_state = env.state.copy()
        previous_rates = projected_motor_rates(env, previous_state)

        next_state, _, done = env.step(action_index)
        next_rates = projected_motor_rates(env, next_state)
        motor_rates = 0.5 * (previous_rates + next_rates)
        motor_power = torques * motor_rates
        joint_energy += np.maximum(motor_power, 0.0) * env.dt
        if np.any(np.abs(torques) > 1e-12):
            nonzero_torque_steps += 1
        steps += 1

    if not done:
        env.over = True
        env.terminal_reason = "timeout"

    final_q = env.state[0::2].copy()
    final_com = center_of_mass(final_q)
    signed_com_displacement = float(walk_direction * (final_com[0] - initial_com[0]))
    success_flag = str(env.terminal_reason).startswith("success")
    torque_valid_for_cmt = (
        (not REQUIRE_NONZERO_TORQUE_FOR_CMT)
        or nonzero_torque_steps >= MIN_NONZERO_TORQUE_STEPS
    )
    displacement_valid_for_cmt = (
        (not REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT)
        or signed_com_displacement > MIN_COM_DISPLACEMENT_M
    )
    valid_flag = (
        success_flag
        and torque_valid_for_cmt
        and displacement_valid_for_cmt
    )
    denom = total_mass() * g * max(signed_com_displacement, CMT_EPS_DISPLACEMENT_M)
    energy = float(np.sum(joint_energy))
    cmt_value = float(energy / denom) if valid_flag else float("nan")
    joint_cmt = joint_energy / denom if valid_flag else np.full(3, np.nan, dtype=float)

    return RolloutResult(
        eval_label=EVALUATION_LABEL,
        seed=int(seed),
        episode_index=int(episode_index),
        walk_direction=float(walk_direction),
        reset_phase=str(reset_phase),
        controller=str(controller_label or controller),
        policy_name=policy_name,
        passive_mode=str(passive_mode_label or ("" if controller == "active" else PASSIVE_POLICY_MODE)),
        passive_selected_from=passive_selected_from,
        curriculum_progress=float(CURRICULUM_PROGRESS),
        hip_torque_nm=float(HIP_TORQUE_NM),
        knee_torque_nm=float(KNEE_TORQUE_NM),
        commanded_step_length_m=float(commanded_step_length),
        commanded_step_height_m=float(commanded_step_height),
        success=bool(success_flag),
        valid_for_cmt=bool(valid_flag),
        terminal_reason=str(env.terminal_reason),
        steps=int(steps),
        nonzero_torque_steps=int(nonzero_torque_steps),
        energy_j=float(energy),
        support_knee_energy_j=float(joint_energy[0]),
        hip_energy_j=float(joint_energy[1]),
        swing_knee_energy_j=float(joint_energy[2]),
        signed_com_displacement_m=float(signed_com_displacement),
        cmt=float(cmt_value),
        support_knee_cmt=float(joint_cmt[0]),
        hip_cmt=float(joint_cmt[1]),
        swing_knee_cmt=float(joint_cmt[2]),
        initial_q1_deg=float(np.rad2deg(initial_q[0])),
        initial_q2_deg=float(np.rad2deg(initial_q[1])),
        initial_q3_deg=float(np.rad2deg(initial_q[2])),
        initial_q4_deg=float(np.rad2deg(initial_q[3])),
        initial_dq1_rad_s=float(initial_dq[0]),
        initial_dq2_rad_s=float(initial_dq[1]),
        initial_dq3_rad_s=float(initial_dq[2]),
        initial_dq4_rad_s=float(initial_dq[3]),
        final_q1_deg=float(np.rad2deg(final_q[0])),
        final_q2_deg=float(np.rad2deg(final_q[1])),
        final_q3_deg=float(np.rad2deg(final_q[2])),
        final_q4_deg=float(np.rad2deg(final_q[3])),
    )


def choose_eval_reset_phase(walk_direction):
    if RESET_PHASE_MODE != "random":
        return RESET_PHASE_MODE
    return choose_reset_phase(walk_direction)


def sample_initial_state(walk_direction, reset_phase, commanded_step_length, commanded_step_height):
    sampler_env = make_env("active")
    sampler_env.walk_direction = float(walk_direction)
    sampler_env.commanded_step_length = float(commanded_step_length)
    sampler_env.commanded_step_height = float(commanded_step_height)
    if reset_phase == "touchdown" and np.random.random() < current_easy_reset_probability:
        noise = current_easy_reset_noise
    else:
        noise = current_challenge_reset_noise
    q_initial = sampler_env.sample_initial_q_away_from_touchdown(
        lambda: sample_reset_q(
            walk_direction,
            noise,
            reset_phase,
            commanded_step_length,
            commanded_step_height,
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
    state = np.empty(raw_state_dim, dtype=float)
    state[0::2] = q_initial
    state[1::2] = dq_initial
    return state


def select_passive_result(
    initial_state,
    walk_direction,
    reset_phase,
    commanded_step_length,
    commanded_step_height,
    policies,
    seed,
    episode_index,
    candidate_rows,
    passive_policy_mode,
):
    candidates = []
    if passive_policy_mode == "positive":
        candidates.append(("positive", "passive_positive", PASSIVE_POS_POLICY_NAME, policies["passive_pos"]))
    elif passive_policy_mode == "negative":
        candidates.append(("negative", "passive_negative", PASSIVE_NEG_POLICY_NAME, policies["passive_neg"]))
    elif passive_policy_mode == "sign_selector":
        sign_name, sign_confidence = choose_passive_sign(
            policies["sign_selector"],
            initial_state,
            walk_direction,
            commanded_step_length,
            commanded_step_height,
        )
        if sign_name == "positive":
            candidates.append((
                f"selector_positive_p{sign_confidence:.3f}",
                "passive_positive",
                PASSIVE_POS_POLICY_NAME,
                policies["passive_pos"],
            ))
        else:
            candidates.append((
                f"selector_negative_p{sign_confidence:.3f}",
                "passive_negative",
                PASSIVE_NEG_POLICY_NAME,
                policies["passive_neg"],
            ))
    elif passive_policy_mode == "oracle_best_cmt":
        candidates.extend(
            [
                ("positive", "passive_positive", PASSIVE_POS_POLICY_NAME, policies["passive_pos"]),
                ("negative", "passive_negative", PASSIVE_NEG_POLICY_NAME, policies["passive_neg"]),
            ]
        )
    else:
        raise ValueError(f"Unsupported passive policy mode: {passive_policy_mode}")

    results = []
    controller_label = {
        "sign_selector": "passive_sign_selector",
        "oracle_best_cmt": "passive_oracle_envelope",
        "positive": "passive_positive_only",
        "negative": "passive_negative_only",
    }.get(passive_policy_mode, f"passive_{passive_policy_mode}")
    for selection_name, controller_name, policy_name, policy in candidates:
        result = rollout_policy(
            initial_state, walk_direction, reset_phase, commanded_step_length, commanded_step_height,
            policy, controller_name, policy_name, seed, episode_index,
            passive_selected_from=selection_name,
            controller_label=controller_label if passive_policy_mode != "oracle_best_cmt" else "passive_oracle_candidate",
            passive_mode_label=passive_policy_mode,
        )
        results.append(result)
        candidate_row = asdict(result)
        candidate_row["candidate_for_mode"] = passive_policy_mode
        candidate_rows.append(candidate_row)

    if passive_policy_mode != "oracle_best_cmt":
        return results[0]

    valid = [r for r in results if r.valid_for_cmt]
    if valid:
        selected = min(valid, key=lambda r: r.cmt)
    else:
        successful = [r for r in results if r.success]
        if successful:
            selected = max(successful, key=lambda r: r.signed_com_displacement_m)
        else:
            selected = max(results, key=lambda r: (r.signed_com_displacement_m, -r.steps))
    selected.controller = controller_label
    selected.passive_mode = passive_policy_mode
    return selected


def finite_mean(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def finite_median(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def reason_tokens(reason):
    tokens = [token for token in str(reason).split("+") if token]
    return tokens if tokens else ["unknown"]


def format_float_key(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def summarize_terminal_reasons(rows):
    summary_rows = []
    controllers = sorted({str(r["controller"]) for r in rows})
    for controller in controllers:
        subset = [r for r in rows if str(r["controller"]) == controller]
        failures = [r for r in subset if not bool(r["success"])]
        successes = [r for r in subset if bool(r["success"])]
        exact_failure_counts = {}
        token_failure_counts = {}
        success_reason_counts = {}
        for row in failures:
            reason = str(row.get("terminal_reason", "unknown"))
            exact_failure_counts[reason] = exact_failure_counts.get(reason, 0) + 1
            for token in reason_tokens(reason):
                token_failure_counts[token] = token_failure_counts.get(token, 0) + 1
        for row in successes:
            reason = str(row.get("terminal_reason", "success"))
            success_reason_counts[reason] = success_reason_counts.get(reason, 0) + 1

        for reason, count in sorted(exact_failure_counts.items(), key=lambda item: (-item[1], item[0])):
            summary_rows.append(
                {
                    "eval_label": EVALUATION_LABEL,
                    "controller": controller,
                    "reason_group": "failure_exact_terminal_reason",
                    "reason": reason,
                    "count": int(count),
                    "fraction_of_all": count / max(1, len(subset)),
                    "fraction_of_failures": count / max(1, len(failures)),
                    "fraction_of_successes": 0.0,
                    "n_all": len(subset),
                    "n_failures": len(failures),
                    "n_successes": len(successes),
                }
            )
        for reason, count in sorted(token_failure_counts.items(), key=lambda item: (-item[1], item[0])):
            summary_rows.append(
                {
                    "eval_label": EVALUATION_LABEL,
                    "controller": controller,
                    "reason_group": "failure_reason_token",
                    "reason": reason,
                    "count": int(count),
                    "fraction_of_all": count / max(1, len(subset)),
                    "fraction_of_failures": count / max(1, len(failures)),
                    "fraction_of_successes": 0.0,
                    "n_all": len(subset),
                    "n_failures": len(failures),
                    "n_successes": len(successes),
                }
            )
        for reason, count in sorted(success_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            summary_rows.append(
                {
                    "eval_label": EVALUATION_LABEL,
                    "controller": controller,
                    "reason_group": "success_terminal_reason",
                    "reason": reason,
                    "count": int(count),
                    "fraction_of_all": count / max(1, len(subset)),
                    "fraction_of_failures": 0.0,
                    "fraction_of_successes": count / max(1, len(successes)),
                    "n_all": len(subset),
                    "n_failures": len(failures),
                    "n_successes": len(successes),
                }
            )
    return summary_rows


def summarize_success_cases(rows):
    summary_rows = []
    controllers = sorted({str(r["controller"]) for r in rows})
    for controller in controllers:
        subset = [r for r in rows if str(r["controller"]) == controller]
        successes = [r for r in subset if bool(r["success"])]
        category_values = {
            "reset_phase": lambda r: str(r.get("reset_phase", "")),
            "walk_direction": lambda r: "forward" if float(r.get("walk_direction", 0.0)) >= 0.0 else "backward",
            "commanded_step_length_m": lambda r: format_float_key(r.get("commanded_step_length_m", "")),
            "commanded_step_height_m": lambda r: format_float_key(r.get("commanded_step_height_m", "")),
            "command_pair": lambda r: (
                f"L{format_float_key(r.get('commanded_step_length_m', ''))}_"
                f"H{format_float_key(r.get('commanded_step_height_m', ''))}"
            ),
            "valid_for_cmt": lambda r: str(bool(r.get("valid_for_cmt", False))),
            "terminal_reason": lambda r: str(r.get("terminal_reason", "")),
            "passive_selected_from": lambda r: str(r.get("passive_selected_from", "")) or "none",
        }
        for category, value_getter in category_values.items():
            counts = {}
            for row in successes:
                value = value_getter(row)
                counts[value] = counts.get(value, 0) + 1
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                summary_rows.append(
                    {
                        "eval_label": EVALUATION_LABEL,
                        "controller": controller,
                        "category": category,
                        "value": value,
                        "count": int(count),
                        "fraction_of_successes": count / max(1, len(successes)),
                        "fraction_of_all": count / max(1, len(subset)),
                        "n_successes": len(successes),
                        "n_all": len(subset),
                    }
                )
    return summary_rows


def summarize_controller(rows, controller):
    subset = [r for r in rows if r["controller"] == controller]
    valid = [r for r in subset if r["valid_for_cmt"]]
    return {
        "eval_label": EVALUATION_LABEL,
        "controller": controller,
        "n_all": len(subset),
        "n_success": sum(bool(r["success"]) for r in subset),
        "n_valid_cmt": len(valid),
        "success_rate_all": sum(bool(r["success"]) for r in subset) / max(1, len(subset)),
        "valid_cmt_rate_all": len(valid) / max(1, len(subset)),
        "mean_energy_all_j": finite_mean(r["energy_j"] for r in subset),
        "mean_energy_valid_j": finite_mean(r["energy_j"] for r in valid),
        "mean_support_knee_energy_all_j": finite_mean(r["support_knee_energy_j"] for r in subset),
        "mean_support_knee_energy_valid_j": finite_mean(r["support_knee_energy_j"] for r in valid),
        "mean_hip_energy_all_j": finite_mean(r["hip_energy_j"] for r in subset),
        "mean_hip_energy_valid_j": finite_mean(r["hip_energy_j"] for r in valid),
        "mean_swing_knee_energy_all_j": finite_mean(r["swing_knee_energy_j"] for r in subset),
        "mean_swing_knee_energy_valid_j": finite_mean(r["swing_knee_energy_j"] for r in valid),
        "mean_displacement_all_m": finite_mean(r["signed_com_displacement_m"] for r in subset),
        "mean_displacement_valid_m": finite_mean(r["signed_com_displacement_m"] for r in valid),
        "mean_cmt_valid": finite_mean(r["cmt"] for r in valid),
        "median_cmt_valid": finite_median(r["cmt"] for r in valid),
        "mean_support_knee_cmt_valid": finite_mean(r["support_knee_cmt"] for r in valid),
        "mean_hip_cmt_valid": finite_mean(r["hip_cmt"] for r in valid),
        "mean_swing_knee_cmt_valid": finite_mean(r["swing_knee_cmt"] for r in valid),
    }


def summarize_pairs(pair_rows, passive_controller=None):
    if passive_controller is not None:
        pair_rows = [p for p in pair_rows if p.get("passive_controller") == passive_controller]
    passive_modes = sorted({str(p.get("passive_policy_mode", "")) for p in pair_rows if p.get("passive_policy_mode", "")})
    both_valid = [p for p in pair_rows if p["active_valid"] and p["passive_valid"]]
    active_only = [p for p in pair_rows if p["active_valid"] and not p["passive_valid"]]
    passive_only = [p for p in pair_rows if p["passive_valid"] and not p["active_valid"]]
    neither = [p for p in pair_rows if not p["active_valid"] and not p["passive_valid"]]
    active_cmts = [p["active_cmt"] for p in both_valid]
    passive_cmts = [p["passive_cmt"] for p in both_valid]
    deltas = [p["passive_cmt"] - p["active_cmt"] for p in both_valid]
    ratios = [p["passive_cmt"] / p["active_cmt"] for p in both_valid if p["active_cmt"] > 0]
    joint_names = ("support_knee", "hip", "swing_knee")
    joint_summary = {}
    for joint in joint_names:
        joint_summary[f"active_{joint}_mean_cmt_on_both_valid"] = finite_mean(
            p[f"active_{joint}_cmt"] for p in both_valid
        )
        joint_summary[f"passive_{joint}_mean_cmt_on_both_valid"] = finite_mean(
            p[f"passive_{joint}_cmt"] for p in both_valid
        )
        joint_summary[f"mean_passive_minus_active_{joint}_cmt"] = finite_mean(
            p[f"passive_{joint}_cmt"] - p[f"active_{joint}_cmt"] for p in both_valid
        )
    return {
        "eval_label": EVALUATION_LABEL,
        "passive_controller": passive_controller or "all",
        "passive_policy_mode": "+".join(passive_modes) if passive_modes else "",
        "total_pairs": len(pair_rows),
        "both_valid_pairs": len(both_valid),
        "active_valid_only_pairs": len(active_only),
        "passive_valid_only_pairs": len(passive_only),
        "neither_valid_pairs": len(neither),
        "both_valid_rate": len(both_valid) / max(1, len(pair_rows)),
        "active_valid_rate": (len(both_valid) + len(active_only)) / max(1, len(pair_rows)),
        "passive_valid_rate": (len(both_valid) + len(passive_only)) / max(1, len(pair_rows)),
        "active_mean_cmt_on_both_valid": finite_mean(active_cmts),
        "passive_mean_cmt_on_both_valid": finite_mean(passive_cmts),
        "active_median_cmt_on_both_valid": finite_median(active_cmts),
        "passive_median_cmt_on_both_valid": finite_median(passive_cmts),
        "mean_passive_minus_active_cmt": finite_mean(deltas),
        "median_passive_minus_active_cmt": finite_median(deltas),
        "mean_passive_over_active_cmt": finite_mean(ratios),
        "fraction_passive_lower_cmt": sum(p["passive_cmt"] < p["active_cmt"] for p in both_valid) / max(1, len(both_valid)),
        **joint_summary,
    }


def select_representative_initial_states(pair_rows, passive_controller=None):
    if passive_controller is not None:
        pair_rows = [p for p in pair_rows if p.get("passive_controller") == passive_controller]
    selected = []
    for row in pair_rows:
        if not (row["active_valid"] and row["passive_valid"]):
            continue
        active_cmt = float(row["active_cmt"])
        passive_cmt = float(row["passive_cmt"])
        if not (np.isfinite(active_cmt) and np.isfinite(passive_cmt)):
            continue
        active_disp = float(row["active_displacement_m"])
        passive_disp = float(row["passive_displacement_m"])
        min_disp = min(active_disp, passive_disp)
        max_disp = max(active_disp, passive_disp, CMT_EPS_DISPLACEMENT_M)
        balance_ratio = min_disp / max_disp
        if min_disp < REPRESENTATIVE_MIN_COM_DISPLACEMENT_M:
            continue
        if balance_ratio < REPRESENTATIVE_MIN_DISPLACEMENT_BALANCE_RATIO:
            continue
        active_minus_passive_cmt = active_cmt - passive_cmt
        if REPRESENTATIVE_REQUIRE_PASSIVE_LOWER_CMT and active_minus_passive_cmt <= 0.0:
            continue
        representative_score = active_minus_passive_cmt + 0.02 * balance_ratio + 0.01 * min_disp
        selected.append(
            {
                **row,
                "active_minus_passive_cmt": active_minus_passive_cmt,
                "min_pair_displacement_m": min_disp,
                "displacement_balance_ratio": balance_ratio,
                "representative_score": representative_score,
            }
        )
    selected.sort(
        key=lambda r: (
            r["active_minus_passive_cmt"],
            r["displacement_balance_ratio"],
            r["min_pair_displacement_m"],
        ),
        reverse=True,
    )
    return selected[:REPRESENTATIVE_STATE_COUNT]


def select_successful_initial_states(rows, controller):
    subset = [
        r for r in rows
        if str(r.get("controller", "")) == controller and bool(r.get("success", False))
    ]
    subset.sort(
        key=lambda r: (
            bool(r.get("valid_for_cmt", False)),
            float(r.get("signed_com_displacement_m", 0.0)),
            -float(r.get("cmt", 1e9)) if np.isfinite(float(r.get("cmt", float("nan")))) else -1e9,
        ),
        reverse=True,
    )
    return subset[:SUCCESSFUL_STATE_EXPORT_COUNT]


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_active_passive_pair_row(active_result, passive_result):
    return {
        "eval_label": EVALUATION_LABEL,
        "seed": int(active_result.seed),
        "episode_index": int(active_result.episode_index),
        "walk_direction": float(active_result.walk_direction),
        "reset_phase": str(active_result.reset_phase),
        "commanded_step_length_m": float(active_result.commanded_step_length_m),
        "commanded_step_height_m": float(active_result.commanded_step_height_m),
        "passive_controller": passive_result.controller,
        "passive_policy_mode": passive_result.passive_mode,
        "passive_selected_from": passive_result.passive_selected_from,
        "active_success": active_result.success,
        "passive_success": passive_result.success,
        "active_valid": active_result.valid_for_cmt,
        "passive_valid": passive_result.valid_for_cmt,
        "active_cmt": active_result.cmt,
        "passive_cmt": passive_result.cmt,
        "active_support_knee_cmt": active_result.support_knee_cmt,
        "active_hip_cmt": active_result.hip_cmt,
        "active_swing_knee_cmt": active_result.swing_knee_cmt,
        "passive_support_knee_cmt": passive_result.support_knee_cmt,
        "passive_hip_cmt": passive_result.hip_cmt,
        "passive_swing_knee_cmt": passive_result.swing_knee_cmt,
        "active_energy_j": active_result.energy_j,
        "passive_energy_j": passive_result.energy_j,
        "active_support_knee_energy_j": active_result.support_knee_energy_j,
        "active_hip_energy_j": active_result.hip_energy_j,
        "active_swing_knee_energy_j": active_result.swing_knee_energy_j,
        "passive_support_knee_energy_j": passive_result.support_knee_energy_j,
        "passive_hip_energy_j": passive_result.hip_energy_j,
        "passive_swing_knee_energy_j": passive_result.swing_knee_energy_j,
        "active_displacement_m": active_result.signed_com_displacement_m,
        "passive_displacement_m": passive_result.signed_com_displacement_m,
        "initial_q1_deg": active_result.initial_q1_deg,
        "initial_q2_deg": active_result.initial_q2_deg,
        "initial_q3_deg": active_result.initial_q3_deg,
        "initial_q4_deg": active_result.initial_q4_deg,
        "initial_dq1_rad_s": active_result.initial_dq1_rad_s,
        "initial_dq2_rad_s": active_result.initial_dq2_rad_s,
        "initial_dq3_rad_s": active_result.initial_dq3_rad_s,
        "initial_dq4_rad_s": active_result.initial_dq4_rad_s,
        "active_terminal_reason": active_result.terminal_reason,
        "passive_terminal_reason": passive_result.terminal_reason,
    }


def build_reference_comparison_pair_row(reference_result, comparison_result):
    return {
        "eval_label": EVALUATION_LABEL,
        "seed": int(reference_result.seed),
        "episode_index": int(reference_result.episode_index),
        "walk_direction": float(reference_result.walk_direction),
        "reset_phase": str(reference_result.reset_phase),
        "commanded_step_length_m": float(reference_result.commanded_step_length_m),
        "commanded_step_height_m": float(reference_result.commanded_step_height_m),
        "reference_controller": reference_result.controller,
        "comparison_controller": comparison_result.controller,
        "reference_policy_name": reference_result.policy_name,
        "comparison_policy_name": comparison_result.policy_name,
        "reference_success": reference_result.success,
        "comparison_success": comparison_result.success,
        "reference_valid": reference_result.valid_for_cmt,
        "comparison_valid": comparison_result.valid_for_cmt,
        "reference_cmt": reference_result.cmt,
        "comparison_cmt": comparison_result.cmt,
        "reference_support_knee_cmt": reference_result.support_knee_cmt,
        "reference_hip_cmt": reference_result.hip_cmt,
        "reference_swing_knee_cmt": reference_result.swing_knee_cmt,
        "comparison_support_knee_cmt": comparison_result.support_knee_cmt,
        "comparison_hip_cmt": comparison_result.hip_cmt,
        "comparison_swing_knee_cmt": comparison_result.swing_knee_cmt,
        "reference_energy_j": reference_result.energy_j,
        "comparison_energy_j": comparison_result.energy_j,
        "reference_displacement_m": reference_result.signed_com_displacement_m,
        "comparison_displacement_m": comparison_result.signed_com_displacement_m,
        "initial_q1_deg": reference_result.initial_q1_deg,
        "initial_q2_deg": reference_result.initial_q2_deg,
        "initial_q3_deg": reference_result.initial_q3_deg,
        "initial_q4_deg": reference_result.initial_q4_deg,
        "initial_dq1_rad_s": reference_result.initial_dq1_rad_s,
        "initial_dq2_rad_s": reference_result.initial_dq2_rad_s,
        "initial_dq3_rad_s": reference_result.initial_dq3_rad_s,
        "initial_dq4_rad_s": reference_result.initial_dq4_rad_s,
        "reference_terminal_reason": reference_result.terminal_reason,
        "comparison_terminal_reason": comparison_result.terminal_reason,
    }


def summarize_reference_pairs(pair_rows, comparison_controller=None):
    if comparison_controller is not None:
        pair_rows = [p for p in pair_rows if p.get("comparison_controller") == comparison_controller]
    both_valid = [p for p in pair_rows if p["reference_valid"] and p["comparison_valid"]]
    reference_only = [p for p in pair_rows if p["reference_valid"] and not p["comparison_valid"]]
    comparison_only = [p for p in pair_rows if p["comparison_valid"] and not p["reference_valid"]]
    neither = [p for p in pair_rows if not p["reference_valid"] and not p["comparison_valid"]]
    deltas = [p["comparison_cmt"] - p["reference_cmt"] for p in both_valid]
    ratios = [p["comparison_cmt"] / p["reference_cmt"] for p in both_valid if p["reference_cmt"] > 0]
    return {
        "eval_label": EVALUATION_LABEL,
        "reference_controller": "active",
        "comparison_controller": comparison_controller or "all",
        "total_pairs": len(pair_rows),
        "both_valid_pairs": len(both_valid),
        "reference_valid_only_pairs": len(reference_only),
        "comparison_valid_only_pairs": len(comparison_only),
        "neither_valid_pairs": len(neither),
        "both_valid_rate": len(both_valid) / max(1, len(pair_rows)),
        "reference_valid_rate": (len(both_valid) + len(reference_only)) / max(1, len(pair_rows)),
        "comparison_valid_rate": (len(both_valid) + len(comparison_only)) / max(1, len(pair_rows)),
        "reference_mean_cmt_on_both_valid": finite_mean(p["reference_cmt"] for p in both_valid),
        "comparison_mean_cmt_on_both_valid": finite_mean(p["comparison_cmt"] for p in both_valid),
        "mean_comparison_minus_reference_cmt": finite_mean(deltas),
        "median_comparison_minus_reference_cmt": finite_median(deltas),
        "mean_comparison_over_reference_cmt": finite_mean(ratios),
        "fraction_comparison_lower_cmt": (
            sum(p["comparison_cmt"] < p["reference_cmt"] for p in both_valid)
            / max(1, len(both_valid))
        ),
    }


def build_selector_oracle_row(active_result, selector_result, oracle_result):
    row = build_active_passive_pair_row(active_result, oracle_result)
    row["selector_success"] = selector_result.success
    row["selector_valid"] = selector_result.valid_for_cmt
    row["selector_cmt"] = selector_result.cmt
    row["selector_selected_from"] = selector_result.passive_selected_from
    row["selector_energy_j"] = selector_result.energy_j
    row["selector_displacement_m"] = selector_result.signed_com_displacement_m
    row["selector_terminal_reason"] = selector_result.terminal_reason
    row["oracle_success"] = oracle_result.success
    row["oracle_valid"] = oracle_result.valid_for_cmt
    row["oracle_cmt"] = oracle_result.cmt
    row["oracle_selected_from"] = oracle_result.passive_selected_from
    row["oracle_energy_j"] = oracle_result.energy_j
    row["oracle_displacement_m"] = oracle_result.signed_com_displacement_m
    row["oracle_terminal_reason"] = oracle_result.terminal_reason
    if selector_result.valid_for_cmt and oracle_result.valid_for_cmt:
        row["selector_minus_oracle_cmt"] = selector_result.cmt - oracle_result.cmt
        row["selector_over_oracle_cmt"] = selector_result.cmt / max(oracle_result.cmt, CMT_EPS_DISPLACEMENT_M)
    else:
        row["selector_minus_oracle_cmt"] = float("nan")
        row["selector_over_oracle_cmt"] = float("nan")
    row["selector_matches_oracle_sign"] = (
        str(selector_result.passive_selected_from).split("_")[1]
        if str(selector_result.passive_selected_from).startswith("selector_")
        else str(selector_result.passive_selected_from)
    ) in str(oracle_result.passive_selected_from)
    return row


def run_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEEDS[0])
    np.random.seed(RANDOM_SEEDS[0])
    torch.manual_seed(RANDOM_SEEDS[0])
    update_curriculum(0, progress_override=CURRICULUM_PROGRESS)
    passive_modes = list(PASSIVE_POLICY_MODES) if PASSIVE_POLICY_MODES else [PASSIVE_POLICY_MODE]
    supported_modes = {"sign_selector", "oracle_best_cmt", "positive", "negative"}
    unsupported_modes = [mode for mode in passive_modes if mode not in supported_modes]
    if unsupported_modes:
        raise ValueError(f"Unsupported PASSIVE_POLICY_MODES: {unsupported_modes}")

    policies = {
        "active": load_policy(ACTIVE_POLICY_PATH, len(ACTIVE_ACTIONS), ACTIVE_POLICY_NAME),
        "passive_pos": load_policy(PASSIVE_POS_POLICY_PATH, len(PASSIVE_ACTIONS), PASSIVE_POS_POLICY_NAME),
        "passive_neg": load_policy(PASSIVE_NEG_POLICY_PATH, len(PASSIVE_ACTIONS), PASSIVE_NEG_POLICY_NAME),
    }
    if INCLUDE_ACTIVE_FULL_POLICY:
        policies["active_full"] = load_policy(
            ACTIVE_FULL_POLICY_PATH,
            len(ACTIVE_ACTIONS),
            ACTIVE_FULL_POLICY_NAME,
        )
    if INCLUDE_PER_STEP_SIGN_POLICY:
        policies["per_step_sign"] = load_policy(
            PER_STEP_SIGN_POLICY_PATH,
            len(PER_STEP_SIGN_ACTIONS),
            PER_STEP_SIGN_POLICY_NAME,
        )
    if "sign_selector" in passive_modes:
        policies["sign_selector"] = load_sign_selector(PASSIVE_SIGN_SELECTOR_PATH, PASSIVE_SIGN_SELECTOR_NAME)

    rollout_rows = []
    candidate_rows = []
    pair_rows = []
    active_reference_pair_rows = []
    selector_oracle_rows = []
    episode_index = 0

    command_lengths, command_heights = command_grid_values()
    for seed in RANDOM_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        for walk_direction in WALK_DIRECTIONS:
            for commanded_step_length in command_lengths:
                for commanded_step_height in command_heights:
                    for _ in range(EPISODES_PER_GRID_CELL_DIRECTION):
                        episode_index += 1
                        reset_phase = choose_eval_reset_phase(walk_direction)
                        initial_state = sample_initial_state(
                            walk_direction,
                            reset_phase,
                            commanded_step_length,
                            commanded_step_height,
                        )

                        active_result = rollout_policy(
                            initial_state, walk_direction, reset_phase,
                            commanded_step_length, commanded_step_height,
                            policies["active"], "active", ACTIVE_POLICY_NAME,
                            seed, episode_index,
                        )
                        extra_active_results = []
                        if "active_full" in policies:
                            extra_active_results.append(
                                rollout_policy(
                                    initial_state, walk_direction, reset_phase,
                                    commanded_step_length, commanded_step_height,
                                    policies["active_full"], "active_full", ACTIVE_FULL_POLICY_NAME,
                                    seed, episode_index,
                                    controller_label="active_full",
                                )
                            )
                        if "per_step_sign" in policies:
                            extra_active_results.append(
                                rollout_policy(
                                    initial_state, walk_direction, reset_phase,
                                    commanded_step_length, commanded_step_height,
                                    policies["per_step_sign"], "per_step_sign", PER_STEP_SIGN_POLICY_NAME,
                                    seed, episode_index,
                                    controller_label="per_step_sign",
                                )
                            )
                        passive_result = select_passive_result(
                            initial_state, walk_direction, reset_phase,
                            commanded_step_length, commanded_step_height,
                            policies, seed, episode_index, candidate_rows,
                            passive_modes[0],
                        )
                        rollout_rows.append(asdict(active_result))
                        for extra_result in extra_active_results:
                            rollout_rows.append(asdict(extra_result))
                            active_reference_pair_rows.append(
                                build_reference_comparison_pair_row(active_result, extra_result)
                            )
                        passive_results = {passive_modes[0]: passive_result}
                        rollout_rows.append(asdict(passive_result))
                        pair_rows.append(build_active_passive_pair_row(active_result, passive_result))

                        for passive_mode in passive_modes[1:]:
                            passive_result = select_passive_result(
                                initial_state, walk_direction, reset_phase,
                                commanded_step_length, commanded_step_height,
                                policies, seed, episode_index, candidate_rows,
                                passive_mode,
                            )
                            passive_results[passive_mode] = passive_result
                            rollout_rows.append(asdict(passive_result))
                            pair_rows.append(build_active_passive_pair_row(active_result, passive_result))

                        if "sign_selector" in passive_results and "oracle_best_cmt" in passive_results:
                            selector_oracle_rows.append(
                                build_selector_oracle_row(
                                    active_result,
                                    passive_results["sign_selector"],
                                    passive_results["oracle_best_cmt"],
                                )
                            )

    controller_labels = sorted({row["controller"] for row in rollout_rows})
    passive_controller_labels = sorted(
        {row["controller"] for row in rollout_rows if str(row["controller"]).startswith("passive")}
    )
    active_comparison_labels = sorted(
        {row["comparison_controller"] for row in active_reference_pair_rows}
    )
    controller_summary_rows = [
        summarize_controller(rollout_rows, controller_label)
        for controller_label in controller_labels
    ]
    paired_summary_rows = [
        summarize_pairs(pair_rows, controller_label)
        for controller_label in passive_controller_labels
    ]
    active_reference_summary_rows = [
        summarize_reference_pairs(active_reference_pair_rows, controller_label)
        for controller_label in active_comparison_labels
    ]
    representative_by_controller = {
        controller_label: select_representative_initial_states(pair_rows, controller_label)
        for controller_label in passive_controller_labels
    }
    successful_initial_states_by_controller = {
        controller_label: select_successful_initial_states(rollout_rows, controller_label)
        for controller_label in controller_labels
    }
    terminal_reason_summary_rows = summarize_terminal_reasons(rollout_rows)
    success_case_summary_rows = summarize_success_cases(rollout_rows)

    write_csv(OUTPUT_DIR / "rollouts_active_vs_passive.csv", rollout_rows)
    write_csv(OUTPUT_DIR / "paired_trials.csv", pair_rows)
    write_csv(OUTPUT_DIR / "active_reference_comparison_trials.csv", active_reference_pair_rows)
    write_csv(OUTPUT_DIR / "selector_vs_oracle_trials.csv", selector_oracle_rows)
    write_csv(OUTPUT_DIR / "controller_summary.csv", controller_summary_rows)
    write_csv(OUTPUT_DIR / "paired_summary.csv", paired_summary_rows)
    write_csv(OUTPUT_DIR / "active_reference_summary.csv", active_reference_summary_rows)
    write_csv(OUTPUT_DIR / "terminal_reason_summary.csv", terminal_reason_summary_rows)
    write_csv(OUTPUT_DIR / "success_case_breakdown.csv", success_case_summary_rows)
    for controller_label, representative_rows in representative_by_controller.items():
        write_csv(OUTPUT_DIR / f"representative_initial_states_{controller_label}.csv", representative_rows)
    all_successful_initial_states = []
    for controller_label, initial_state_rows in successful_initial_states_by_controller.items():
        write_csv(OUTPUT_DIR / f"successful_initial_states_{controller_label}.csv", initial_state_rows)
        all_successful_initial_states.extend(initial_state_rows)
    write_csv(OUTPUT_DIR / "successful_initial_states_all_controllers.csv", all_successful_initial_states)
    default_representative_controller = (
        "passive_oracle_envelope"
        if "passive_oracle_envelope" in representative_by_controller
        else (passive_controller_labels[0] if passive_controller_labels else None)
    )
    if default_representative_controller is not None:
        write_csv(
            OUTPUT_DIR / "representative_initial_states.csv",
            representative_by_controller[default_representative_controller],
        )
    write_csv(OUTPUT_DIR / "passive_candidate_audit.csv", candidate_rows)

    print(f"Evaluation label: {EVALUATION_LABEL}")
    print(f"Device: {DEVICE}")
    print(f"Curriculum progress: {CURRICULUM_PROGRESS:.3f}")
    print(f"Hip/knee torque: {HIP_TORQUE_NM:.3f} Nm / {KNEE_TORQUE_NM:.3f} Nm")
    print(f"Command grid lengths/heights: {command_lengths} m / {command_heights} m")
    print(f"Passive modes: {passive_modes}")
    print(f"Controllers: {controller_labels}")
    for controller_label, representative_rows in representative_by_controller.items():
        print(f"Representative initial states {controller_label}: {len(representative_rows)}")
    for controller_label, initial_state_rows in successful_initial_states_by_controller.items():
        print(f"Successful initial states {controller_label}: {len(initial_state_rows)}")
    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print("Controller summary:")
    for row in controller_summary_rows:
        print(row)
    print("Paired summary:")
    if paired_summary_rows:
        print(paired_summary_rows[0])
    if active_reference_summary_rows:
        print("Active reference summary:")
        print(active_reference_summary_rows[0])


if __name__ == "__main__":
    run_evaluation()
