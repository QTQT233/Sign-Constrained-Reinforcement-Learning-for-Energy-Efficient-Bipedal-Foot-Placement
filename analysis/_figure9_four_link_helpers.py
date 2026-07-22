# -*- coding: utf-8 -*-
"""Compare five four-link controllers from the same initial state.

The initial q/dq, command, direction and reset phase are read from
simulate_policy_four_link_biped_v21_v22_2_static_figue_paper_v4_no_path.py.
Dynamics, action spaces and Cmt accounting are reused from the paper Cmt script
so the energy curves match the paper evaluation convention.
"""

import csv
import importlib.util
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# User-editable settings
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
FOUR_LINK_DIR = REPO_ROOT / "src" / "four_link"
SIM_CONFIG_SCRIPT = (
    FOUR_LINK_DIR
    / "evaluation"
    / "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
)
CMT_SCRIPT = SIM_CONFIG_SCRIPT

OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "true_action_mask"
OUTPUT_TAG = "same_initial_active_passive_cmt"

# Leave these as None to inherit from SIM_CONFIG_SCRIPT.
INITIAL_Q_DEG_OVERRIDE = None
INITIAL_DQ_OVERRIDE = None
COMMAND_STEP_LENGTH_OVERRIDE = None
COMMAND_STEP_HEIGHT_OVERRIDE = None
WALK_DIRECTION_OVERRIDE = None  # "forward" or "backward"
RESET_PHASE_OVERRIDE = None
CURRICULUM_OVERRIDE = None
MAX_STEPS_OVERRIDE = None
DETERMINISTIC_POLICY = True
RANDOM_SEED = 20260704

# Leave paths as None to inherit the explicit paths in CMT_SCRIPT.
ACTIVE_POLICY_PATH_OVERRIDE = None
ACTIVE_FULL_POLICY_PATH_OVERRIDE = None
PER_STEP_SIGN_POLICY_PATH_OVERRIDE = None
PASSIVE_POS_POLICY_PATH_OVERRIDE = None
PASSIVE_NEG_POLICY_PATH_OVERRIDE = None
SIGN_SELECTOR_PATH_OVERRIDE = None

FIGURE_DPI = 180
SAVE_PDF = False
STATIC_NUM_POSES = 7
STATIC_KEEP_FIRST_POSE = True
STATIC_DISTANCE_SAMPLING = True
# ===== Static robot panel style =====
MAIN_TITLE_COMBINED = "Controller comparison from the same initial state"
MAIN_TITLE_SPLIT = "Controller comparison with separated torques and Cmt"
STATIC_PANEL_TITLE = "Active PPO Trajectory"

STATIC_SHOW_AXES = False
STATIC_SHOW_STATUS_TEXT = True
STATIC_SHOW_TARGET = True
STATIC_SHOW_SWING_PATH = False
STATIC_EQUAL_ASPECT = True
CMT_PLOT_MIN_DISPLACEMENT_M = 1e-4


# =============================================================================
# Module loading
# =============================================================================
def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sim_cfg = load_module(SIM_CONFIG_SCRIPT, "four_link_sim_config_for_compare")
cmt = load_module(CMT_SCRIPT, "four_link_cmt_for_compare")


def set_seed(seed):
    if seed is None:
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def inherited(value, fallback):
    return fallback if value is None else value


def walk_direction_value(name):
    return 1.0 if str(name).lower().startswith("f") else -1.0


def initial_state_from_sim_config():
    q_deg = inherited(INITIAL_Q_DEG_OVERRIDE, sim_cfg.INITIAL_Q_DEG)
    dq = inherited(INITIAL_DQ_OVERRIDE, sim_cfg.INITIAL_DQ)
    if q_deg is None or dq is None:
        raise ValueError("INITIAL_Q_DEG and INITIAL_DQ must be fixed in the simulation config or overrides.")
    q = np.deg2rad(np.asarray(q_deg, dtype=float))
    dq = np.asarray(dq, dtype=float)
    state = np.empty(cmt.raw_state_dim, dtype=float)
    state[0::2] = q
    state[1::2] = dq
    return state


def selected_path(path_override, default_path):
    return Path(default_path if path_override is None else path_override)


def load_all_policies():
    return {
        "active": cmt.load_policy(
            selected_path(ACTIVE_POLICY_PATH_OVERRIDE, cmt.ACTIVE_POLICY_PATH),
            len(cmt.ACTIVE_ACTIONS),
            cmt.ACTIVE_POLICY_NAME,
        ),
        "active_full": cmt.load_policy(
            selected_path(ACTIVE_FULL_POLICY_PATH_OVERRIDE, cmt.ACTIVE_FULL_POLICY_PATH),
            len(cmt.ACTIVE_ACTIONS),
            cmt.ACTIVE_FULL_POLICY_NAME,
        ),
        "per_step_sign": cmt.load_policy(
            selected_path(PER_STEP_SIGN_POLICY_PATH_OVERRIDE, cmt.PER_STEP_SIGN_POLICY_PATH),
            len(cmt.PER_STEP_SIGN_ACTIONS),
            cmt.PER_STEP_SIGN_POLICY_NAME,
        ),
        "passive_pos": cmt.load_policy(
            selected_path(PASSIVE_POS_POLICY_PATH_OVERRIDE, cmt.PASSIVE_POS_POLICY_PATH),
            len(cmt.PASSIVE_ACTIONS),
            cmt.PASSIVE_POS_POLICY_NAME,
        ),
        "passive_neg": cmt.load_policy(
            selected_path(PASSIVE_NEG_POLICY_PATH_OVERRIDE, cmt.PASSIVE_NEG_POLICY_PATH),
            len(cmt.PASSIVE_ACTIONS),
            cmt.PASSIVE_NEG_POLICY_NAME,
        ),
        "sign_selector": cmt.load_sign_selector(
            selected_path(SIGN_SELECTOR_PATH_OVERRIDE, cmt.PASSIVE_SIGN_SELECTOR_PATH),
            cmt.PASSIVE_SIGN_SELECTOR_NAME,
        ),
    }


# =============================================================================
# Rollout tracing
# =============================================================================
class Trace:
    def __init__(self, label, env, frames, torques, motor_rates, powers, joint_energy, cmt_values, raw_cmt_values):
        self.label = label
        self.env = env
        self.frames = frames
        self.torques = np.asarray(torques, dtype=float)
        self.motor_rates = np.asarray(motor_rates, dtype=float)
        self.powers = np.asarray(powers, dtype=float)
        self.joint_energy = np.asarray(joint_energy, dtype=float)
        self.cmt_values = np.asarray(cmt_values, dtype=float)
        self.raw_cmt_values = np.asarray(raw_cmt_values, dtype=float)
        self.terminal_reason = str(env.terminal_reason)
        self.success = self.terminal_reason.startswith("success")
        self.steps = max(0, len(frames) - 1)
        self.t_state = np.arange(len(frames), dtype=float) * env.dt
        self.t_action = np.arange(max(0, len(frames) - 1), dtype=float) * env.dt
        self.final_raw_cmt = self._last_finite(self.raw_cmt_values)
        self.final_valid_cmt = self._last_finite(self.cmt_values)

    @staticmethod
    def _last_finite(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(finite[-1]) if finite.size else float("nan")


def frame_from_env(env):
    q = env.state[0::2].copy()
    dq = env.state[1::2].copy()
    return {
        "state": env.state.copy(),
        "q": q,
        "dq": dq,
        "points": env.joint_positions(q),
        "metrics": env.touchdown_metrics(q),
        "com": cmt.center_of_mass(q),
        "terminal_reason": str(env.terminal_reason),
    }


def cmt_from_energy(joint_energy, displacement, valid_success):
    denom = cmt.total_mass() * cmt.g * max(float(displacement), cmt.CMT_EPS_DISPLACEMENT_M)
    raw = float(np.sum(joint_energy) / denom)
    if not valid_success or displacement <= cmt.MIN_COM_DISPLACEMENT_M:
        return float("nan"), raw
    return raw, raw


def rollout_trace(label, initial_state, walk_direction, reset_phase, command_length, command_height,
                  policy, controller, hip_sign=0.0):
    if controller in ("active", "active_full"):
        env_controller = "active"
    elif controller == "per_step_sign":
        env_controller = "per_step_sign"
    else:
        env_controller = "passive"

    env = cmt.make_env(env_controller, hip_sign=hip_sign)
    env.define(
        *initial_state,
        hip_direction=hip_sign,
        walk_direction=walk_direction,
        commanded_step_length=command_length,
        commanded_step_height=command_height,
        reset_phase=reset_phase,
    )

    initial_com = cmt.center_of_mass(env.state[0::2])
    frames = [frame_from_env(env)]
    joint_energy = np.zeros(3, dtype=float)
    cumulative_joint_energy = [joint_energy.copy()]
    cmt_values = [float("nan")]
    raw_cmt_values = [float("nan")]
    torques = []
    motor_rates = []
    powers = []
    nonzero_torque_steps = 0

    done = False
    steps = 0
    max_steps = int(inherited(MAX_STEPS_OVERRIDE, getattr(sim_cfg, "MAX_STEPS", cmt.MAX_STEPS)))
    while not done and steps < max_steps:
        state_net = cmt.normalize_state(
            env.state,
            env.hip_direction,
            env.walk_direction,
            env.commanded_step_length,
            env.commanded_step_height,
        )
        action_index, _ = cmt.choose_action(policy, state_net)
        tau = env.action_to_torques(action_index)  # [support knee, hip, swing knee]
        previous_state = env.state.copy()
        previous_rates = cmt.projected_motor_rates(env, previous_state)
        next_state, _, done = env.step(action_index)
        next_rates = cmt.projected_motor_rates(env, next_state)
        rates = 0.5 * (previous_rates + next_rates)
        power = tau * rates
        joint_energy += np.maximum(power, 0.0) * env.dt

        if np.any(np.abs(tau) > 1e-12):
            nonzero_torque_steps += 1

        torques.append(tau.copy())
        motor_rates.append(rates.copy())
        powers.append(power.copy())
        cumulative_joint_energy.append(joint_energy.copy())
        frames.append(frame_from_env(env))

        displacement = walk_direction * (frames[-1]["com"][0] - initial_com[0])
        success_now = str(env.terminal_reason).startswith("success")
        valid_cmt, raw_cmt = cmt_from_energy(joint_energy, displacement, success_now)
        if displacement <= CMT_PLOT_MIN_DISPLACEMENT_M:
            raw_cmt = float("nan")
            valid_cmt = float("nan")
        raw_cmt_values.append(raw_cmt)
        cmt_values.append(valid_cmt)
        steps += 1

    if not done:
        env.over = True
        env.terminal_reason = "timeout"
        frames[-1]["terminal_reason"] = "timeout"

    # Final Cmt validity follows the paper script: success, nonzero torque, and
    # sufficient forward COM displacement.
    final_displacement = walk_direction * (frames[-1]["com"][0] - initial_com[0])
    final_success = str(env.terminal_reason).startswith("success")
    valid_final = (
        final_success
        and ((not cmt.REQUIRE_NONZERO_TORQUE_FOR_CMT) or nonzero_torque_steps >= cmt.MIN_NONZERO_TORQUE_STEPS)
        and ((not cmt.REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT) or final_displacement > cmt.MIN_COM_DISPLACEMENT_M)
    )
    final_valid_cmt, final_raw_cmt = cmt_from_energy(joint_energy, final_displacement, valid_final)
    if len(cmt_values):
        cmt_values[-1] = final_valid_cmt
        raw_cmt_values[-1] = final_raw_cmt if final_displacement > CMT_PLOT_MIN_DISPLACEMENT_M else float("nan")

    return Trace(label, env, frames, torques, motor_rates, powers, cumulative_joint_energy, cmt_values, raw_cmt_values)


def choose_oracle_trace(pos_trace, neg_trace):
    candidates = [pos_trace, neg_trace]
    valid = [t for t in candidates if np.isfinite(t.final_valid_cmt)]
    if valid:
        return min(valid, key=lambda t: t.final_valid_cmt)
    successful = [t for t in candidates if t.success]
    if successful:
        return max(successful, key=lambda t: t.frames[-1]["com"][0] - t.frames[0]["com"][0])
    return max(candidates, key=lambda t: t.frames[-1]["com"][0] - t.frames[0]["com"][0])


# =============================================================================
# Plotting
# =============================================================================
COLORS = {
    "Active PPO": "#1f77b4",
    # "Active-full reference": "#d62728",
    "Per-step sign reference": "#9467bd",
    "Online sign selector": "#2ca02c",
    "Oracle envelope": "#ff7f0e",
}

TRACE_ORDER = [
    "Active PPO",
    # "Active-full reference",
    "Per-step sign reference",
    "Online sign selector",
    "Oracle envelope",
]


def sample_static_indices(frames):
    n = len(frames)
    if n <= STATIC_NUM_POSES:
        return list(range(n))
    if not STATIC_DISTANCE_SAMPLING:
        idx = np.linspace(0, n - 1, STATIC_NUM_POSES)
        return sorted(set(int(round(i)) for i in idx))
    features = np.vstack([
        np.r_[f["points"]["hip"], f["points"]["swing_foot"], f["points"]["swing_knee"]]
        for f in frames
    ])
    distances = np.zeros(n, dtype=float)
    if n > 1:
        step_dist = np.linalg.norm(np.diff(features, axis=0), axis=1)
        distances[1:] = np.cumsum(step_dist)
    if distances[-1] <= 1e-12:
        idx = np.linspace(0, n - 1, STATIC_NUM_POSES)
    else:
        targets = np.linspace(0.0, distances[-1], STATIC_NUM_POSES)
        idx = [int(np.argmin(np.abs(distances - t))) for t in targets]
    idx = sorted(set(idx))
    if STATIC_KEEP_FIRST_POSE and 0 not in idx:
        idx = [0] + idx
    if n - 1 not in idx:
        idx.append(n - 1)
    return idx


def draw_robot_pose(ax, points, color, alpha=1.0, lw=1.4, marker_size=9):
    support = points["support_foot"]
    support_knee = points["support_knee"]
    hip = points["hip"]
    swing_knee = points["swing_knee"]
    swing = points["swing_foot"]
    ax.plot([support[0], support_knee[0], hip[0]], [support[1], support_knee[1], hip[1]],
            color=color, alpha=alpha, lw=lw)
    ax.plot([hip[0], swing_knee[0], swing[0]], [hip[1], swing_knee[1], swing[1]],
            color=color, alpha=alpha, lw=lw)
    ax.scatter(
        [support[0], support_knee[0], hip[0], swing_knee[0], swing[0]],
        [support[1], support_knee[1], hip[1], swing_knee[1], swing[1]],
        s=marker_size,
        color=color,
        alpha=min(1.0, alpha + 0.08),
        zorder=3,
    )


def apply_fixed_ticks(ax, x_count=5, y_count=5):
    """Use simple fixed ticks to avoid backend auto-layout/tick crashes."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if np.isfinite(x_min) and np.isfinite(x_max) and abs(x_max - x_min) > 1e-12:
        ax.set_xticks(np.linspace(x_min, x_max, int(x_count)))
    if np.isfinite(y_min) and np.isfinite(y_max) and abs(y_max - y_min) > 1e-12:
        ax.set_yticks(np.linspace(y_min, y_max, int(y_count)))


def draw_manual_legend(fig, x0=0.48, y0=0.955):
    x = x0
    for label in TRACE_ORDER:
        fig.text(
            x,
            y0,
            label,
            color=COLORS.get(label, "#111111"),
            fontsize=8,
            ha="left",
            va="center",
        )
        x += 0.095 if len(label) <= 11 else 0.14


def draw_active_static(ax, active_trace, command_length, command_height, walk_direction):
    frames = active_trace.frames
    all_points = []
    for f in frames:
        for value in f["points"].values():
            all_points.append(np.asarray(value, dtype=float))
    all_points = np.vstack(all_points)
    target_x = walk_direction * command_length
    target_y = command_height

    indices = sample_static_indices(frames)
    for order, idx in enumerate(indices):
        frac = order / max(1, len(indices) - 1)
        draw_robot_pose(
            ax,
            frames[idx]["points"],
            COLORS["Active PPO"],
            alpha=0.22 + 0.70 * frac,
            lw=1.0 + 0.8 * frac,
            marker_size=8 + 8 * frac,
        )

    swing_path = np.vstack([f["points"]["swing_foot"] for f in frames])
    ax.plot(swing_path[:, 0], swing_path[:, 1], color="#111111", lw=0.9, alpha=0.45, label="swing foot path")
    ax.scatter([target_x], [target_y], marker="x", color="#d62728", s=42, lw=1.5, label="target")

    x_min = min(float(all_points[:, 0].min()), target_x) - 0.08
    x_max = max(float(all_points[:, 0].max()), target_x) + 0.08
    y_min = min(float(all_points[:, 1].min()), target_y, 0.0) - 0.04
    y_max = max(float(all_points[:, 1].max()), target_y) + 0.06
    ax.plot([x_min, x_max], [0.0, 0.0], color="#444444", lw=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # Avoid set_aspect("equal") here. In this Windows/Pytorch Matplotlib build,
    # equal-aspect axes inside a mixed GridSpec can trigger a native transform
    # crash during savefig.
    ax.set_aspect("auto")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Trajectory\n",
        # f"{active_trace.terminal_reason}, {active_trace.steps} steps",
        fontsize=10,
    )
    ax.grid(False, alpha=0.18)
    apply_fixed_ticks(ax, x_count=4, y_count=5)


def trace_label(trace):
    cmt_text = "nan" if not np.isfinite(trace.final_valid_cmt) else f"{trace.final_valid_cmt:.3f}"
    return f"{trace.label} ({trace.terminal_reason}, Cmt={cmt_text})"


def plot_time_series(ax, traces, value_getter, ylabel, title=None, step=True):
    for trace in traces:
        color = COLORS.get(trace.label, None)
        x, y = value_getter(trace)
        if len(x) == 0:
            continue
        ax.plot(
            x,
            y,
            label=trace_label(trace),
            color=color,
            lw=1.5,
            drawstyle="steps-post" if step else "default",
            alpha=0.95,
        )
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.22)
    apply_fixed_ticks(ax, x_count=5, y_count=5)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    print(f"Saved figure: {path}")
    if SAVE_PDF:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path)
        print(f"Saved figure: {pdf_path}")


def pil_font(size, bold=False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_SMALL = None
FONT_NORMAL = None
FONT_TITLE = None


def get_fonts():
    global FONT_SMALL, FONT_NORMAL, FONT_TITLE
    if FONT_SMALL is None:
        FONT_SMALL = pil_font(18)
        FONT_NORMAL = pil_font(22)
        FONT_TITLE = pil_font(28, bold=True)
    return FONT_SMALL, FONT_NORMAL, FONT_TITLE


def text_size(draw, text, font):
    """兼容不同 Pillow 版本的文字尺寸测量。"""
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def draw_centered_text(draw, text, x_min, x_max, y, font, fill="#111111"):
    """在 [x_min, x_max] 范围内水平居中绘制文字。"""
    w, _ = text_size(draw, text, font)
    x = 0.5 * (x_min + x_max) - 0.5 * w
    draw.text((x, y), text, fill=fill, font=font)


def hex_to_rgb(hex_color):
    text = str(hex_color).lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def draw_alpha_line(image, points, color, width=3, alpha=255):
    if len(points) < 2:
        return
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.line(points, fill=tuple(color) + (int(alpha),), width=int(width), joint="curve")
    image.alpha_composite(overlay)


def draw_alpha_circle(image, center, radius, color, alpha=255):
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    x, y = center
    r = float(radius)
    overlay_draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(color) + (int(alpha),))
    image.alpha_composite(overlay)


def plot_area_from_box(box, left_pad=70, right_pad=20, top_pad=55, bottom_pad=55):
    left, top, right, bottom = box
    return (left + left_pad, top + top_pad, right - right_pad, bottom - bottom_pad)


def make_transform(plot_area, x_min, x_max, y_min, y_max):
    left, top, right, bottom = plot_area
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    def transform(x, y):
        px = left + (float(x) - x_min) / (x_max - x_min) * width
        py = bottom - (float(y) - y_min) / (y_max - y_min) * height
        return (px, py)

    return transform


def draw_axes(draw, box, x_min, x_max, y_min, y_max, title, ylabel, xlabel="", ticks=5):
    font_small, font_normal, font_title = get_fonts()
    plot_area = plot_area_from_box(box)
    left, top, right, bottom = plot_area
    draw.rectangle(box, outline="#c8c8c8", width=2)
    draw.text((box[0] + 12, box[1] + 12), title, fill="#111111", font=font_normal)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    for i in range(ticks):
        frac = i / max(1, ticks - 1)
        x = left + frac * (right - left)
        y = bottom - frac * (bottom - top)
        xv = x_min + frac * (x_max - x_min)
        yv = y_min + frac * (y_max - y_min)
        draw.line((x, bottom, x, bottom + 6), fill="#333333", width=1)
        draw.text((x - 24, bottom + 10), f"{xv:.2g}", fill="#333333", font=font_small)
        draw.line((left - 6, y, left, y), fill="#333333", width=1)
        draw.text((box[0] + 8, y - 10), f"{yv:.2g}", fill="#333333", font=font_small)
    draw.text((box[0] + 10, box[1] + 68), ylabel, fill="#333333", font=font_small)
    if xlabel:
        draw.text(((left + right) / 2 - 38, bottom + 30), xlabel, fill="#333333", font=font_small)
    return plot_area


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def padded_limits(values, include_zero=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if include_zero:
        v_min = min(v_min, 0.0)
        v_max = max(v_max, 0.0)
    if abs(v_max - v_min) < 1e-12:
        pad = max(1.0, abs(v_max) * 0.1)
    else:
        pad = 0.08 * (v_max - v_min)
    return v_min - pad, v_max + pad


def draw_timeseries_pil(image, box, traces, value_getter, title, ylabel, xlabel="", step=True):
    draw = ImageDraw.Draw(image)
    series = []
    all_x = []
    all_y = []
    for trace in traces:
        x, y = value_getter(trace)
        x, y = finite_xy(x, y)
        if x.size == 0:
            continue
        series.append((trace.label, x, y))
        all_x.append(x)
        all_y.append(y)
    if not series:
        all_x = [np.asarray([0.0, 1.0])]
        all_y = [np.asarray([0.0, 1.0])]
    x_min, x_max = padded_limits(np.concatenate(all_x), include_zero=True)
    y_min, y_max = padded_limits(np.concatenate(all_y), include_zero=True)
    plot_area = draw_axes(draw, box, x_min, x_max, y_min, y_max, title, ylabel, xlabel=xlabel)
    transform = make_transform(plot_area, x_min, x_max, y_min, y_max)
    left, top, right, bottom = plot_area
    for frac in np.linspace(0.25, 0.75, 3):
        y = top + frac * (bottom - top)
        draw.line((left, y, right, y), fill="#eeeeee", width=1)
    for frac in np.linspace(0.25, 0.75, 3):
        x = left + frac * (right - left)
        draw.line((x, top, x, bottom), fill="#eeeeee", width=1)

    for label, x, y in series:
        color = hex_to_rgb(COLORS.get(label, "#111111"))
        if step and x.size > 1:
            pts = []
            for i in range(x.size - 1):
                pts.append(transform(x[i], y[i]))
                pts.append(transform(x[i + 1], y[i]))
            pts.append(transform(x[-1], y[-1]))
        else:
            pts = [transform(xi, yi) for xi, yi in zip(x, y)]
        draw_alpha_line(image, pts, color, width=4, alpha=230)


def expand_limits_for_equal_aspect(x_min, x_max, y_min, y_max, plot_area):
    """让机器人静态图尽量保持真实比例，避免左侧画布变宽后机器人被横向拉伸。"""
    left, top, right, bottom = plot_area
    plot_w = max(1.0, right - left)
    plot_h = max(1.0, bottom - top)
    plot_aspect = plot_w / plot_h

    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    data_aspect = x_span / y_span

    if data_aspect < plot_aspect:
        new_x_span = plot_aspect * y_span
        cx = 0.5 * (x_min + x_max)
        x_min = cx - 0.5 * new_x_span
        x_max = cx + 0.5 * new_x_span
    else:
        new_y_span = x_span / plot_aspect
        cy = 0.5 * (y_min + y_max)
        y_min = cy - 0.5 * new_y_span
        y_max = cy + 0.5 * new_y_span

    return x_min, x_max, y_min, y_max


def draw_static_pil(image, box, active_trace, command_length, command_height, walk_direction):
    draw = ImageDraw.Draw(image)
    font_small, font_normal, font_title = get_fonts()

    frames = active_trace.frames
    all_points = []
    for f in frames:
        for value in f["points"].values():
            all_points.append(np.asarray(value, dtype=float))
    all_points = np.vstack(all_points)

    target_x = walk_direction * command_length
    target_y = command_height

    x_min = min(float(all_points[:, 0].min()), target_x) - 0.08
    x_max = max(float(all_points[:, 0].max()), target_x) + 0.08
    y_min = min(float(all_points[:, 1].min()), target_y, 0.0) - 0.04
    y_max = max(float(all_points[:, 1].max()), target_y) + 0.06

    # 左图标题和状态文字：均在左图 box 范围内居中
    title_text = STATIC_PANEL_TITLE
    status_text = f"{active_trace.terminal_reason}, {active_trace.steps} steps"

    title_y = box[1] + 8
    status_y = box[1] + 40

    if title_text:
        draw_centered_text(
            draw,
            title_text,
            box[0],
            box[2],
            title_y,
            font_normal,
            fill="#111111",
        )

    if STATIC_SHOW_STATUS_TEXT:
        draw_centered_text(
            draw,
            status_text,
            box[0],
            box[2],
            status_y,
            font_small,
            fill="#333333",
        )

    # 隐藏左图坐标系时，给标题和状态文字留出顶部空间
    if STATIC_SHOW_AXES:
        plot_area = draw_axes(
            draw,
            box,
            x_min,
            x_max,
            y_min,
            y_max,
            "",
            "y (m)",
            xlabel="x (m)",
        )
    else:
        plot_area = plot_area_from_box(
            box,
            left_pad=24,
            right_pad=24,
            top_pad=82,
            bottom_pad=24,
        )

    if STATIC_EQUAL_ASPECT:
        x_min, x_max, y_min, y_max = expand_limits_for_equal_aspect(
            x_min, x_max, y_min, y_max, plot_area
        )

    transform = make_transform(plot_area, x_min, x_max, y_min, y_max)

    left, top, right, bottom = plot_area

    # 地面线保留，但不作为坐标轴
    ground_y = transform(0.0, 0.0)[1]
    if top <= ground_y <= bottom:
        draw.line((left, ground_y, right, ground_y), fill="#888888", width=2)

    if STATIC_SHOW_TARGET:
        tx, ty = transform(target_x, target_y)
        draw.line((tx - 10, ty - 10, tx + 10, ty + 10), fill="#d62728", width=3)
        draw.line((tx - 10, ty + 10, tx + 10, ty - 10), fill="#d62728", width=3)

    if STATIC_SHOW_SWING_PATH:
        swing_path = [transform(*f["points"]["swing_foot"]) for f in frames]
        draw.line(swing_path, fill="#444444", width=2)

    indices = sample_static_indices(frames)
    color = hex_to_rgb(COLORS["Active PPO"])

    for order, idx in enumerate(indices):
        frac = order / max(1, len(indices) - 1)

        alpha = int(80 + 165 * frac)
        width = int(3 + 3 * frac)
        radius = 4 + 4 * frac

        points = frames[idx]["points"]
        support = transform(*points["support_foot"])
        support_knee = transform(*points["support_knee"])
        hip = transform(*points["hip"])
        swing_knee = transform(*points["swing_knee"])
        swing = transform(*points["swing_foot"])

        draw_alpha_line(image, [support, support_knee, hip], color, width=width, alpha=alpha)
        draw_alpha_line(image, [hip, swing_knee, swing], color, width=width, alpha=alpha)

        for p in [support, support_knee, hip, swing_knee, swing]:
            draw_alpha_circle(image, p, radius, color, alpha=min(255, alpha + 20))


def text_size(draw, text, font):
    """兼容不同 Pillow 版本的文字尺寸测量。"""
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def draw_centered_text(draw, text, x_min, x_max, y, font, fill="#111111"):
    """在 [x_min, x_max] 范围内水平居中绘制文字。"""
    w, h = text_size(draw, text, font)
    x = 0.5 * (x_min + x_max) - 0.5 * w
    draw.text((x, y), text, fill=fill, font=font)


def text_width(draw, text, font):
    """兼容不同 Pillow 版本的文字宽度测量。"""
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def draw_pil_legend(draw, x, y, max_x=2520):
    """
    自动排布图例，避免文字遮住颜色线段；
    若当前行放不下，则自动换到下一行。
    """
    font_small, font_normal, font_title = get_fonts()

    line_len = 36
    text_gap = 10
    item_gap = 42
    row_gap = 30

    px = x
    py = y

    for label in TRACE_ORDER:
        color = hex_to_rgb(COLORS[label])
        label_w, _ = text_size(draw, label, font_small)
        item_w = line_len + text_gap + label_w + item_gap

        if px != x and px + item_w > max_x:
            px = x
            py += row_gap

        draw.line(
            (px, py + 12, px + line_len, py + 12),
            fill=color,
            width=5,
        )
        draw.text(
            (px + line_len + text_gap, py),
            label,
            fill="#111111",
            font=font_small,
        )

        px += item_w


def save_combined_knee_pil(path, traces, active_trace, command_length, command_height, walk_direction):
    image = Image.new("RGBA", (2600, 1400), "white")
    draw = ImageDraw.Draw(image)
    _, _, font_title = get_fonts()

    # 整张图总标题：水平居中
    draw_centered_text(
        draw,
        MAIN_TITLE_COMBINED,
        0,
        image.size[0],
        18,
        font_title,
        fill="#111111",
    )

    # 图例：放在总标题下方，只占右侧区域
    draw_pil_legend(draw, 1400, 72, max_x=2520)

    # 左图：整体下移，避免和总标题/图例冲突
    draw_static_pil(
        image,
        (70, 150, 1280, 1320),
        active_trace,
        command_length,
        command_height,
        walk_direction,
    )

    draw_timeseries_pil(
        image,
        (1400, 150, 2520, 500),
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 1] if tr.torques.size else np.asarray([])),
        "Hip torque",
        "Nm",
    )

    draw_timeseries_pil(
        image,
        (1400, 545, 2520, 895),
        traces,
        lambda tr: (
            tr.t_action,
            np.sqrt(0.5 * (tr.torques[:, 0] ** 2 + tr.torques[:, 2] ** 2)) if tr.torques.size else np.asarray([]),
        ),
        "Two-knee RMS torque",
        "Nm",
    )

    draw_timeseries_pil(
        image,
        (1400, 940, 2520, 1300),
        traces,
        lambda tr: (tr.t_state, tr.raw_cmt_values),
        "Cmt over time",
        "Cmt",
        xlabel="time (s)",
        step=False,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    print(f"Saved figure: {path}")


def save_split_knee_pil(path, traces, active_trace, command_length, command_height, walk_direction):
    image = Image.new("RGBA", (2600, 1600), "white")
    draw = ImageDraw.Draw(image)
    _, _, font_title = get_fonts()

    # 整张图总标题：水平居中
    draw_centered_text(
        draw,
        MAIN_TITLE_SPLIT,
        0,
        image.size[0],
        18,
        font_title,
        fill="#111111",
    )

    # 图例：下移，避免和总标题冲突
    draw_pil_legend(draw, 1400, 72, max_x=2520)

    draw_static_pil(
        image,
        (70, 150, 1280, 1500),
        active_trace,
        command_length,
        command_height,
        walk_direction,
    )

    draw_timeseries_pil(
        image,
        (1400, 150, 2520, 450),
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 1] if tr.torques.size else np.asarray([])),
        "Hip torque",
        "Nm",
    )

    draw_timeseries_pil(
        image,
        (1400, 495, 2520, 795),
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 0] if tr.torques.size else np.asarray([])),
        "Support-knee torque",
        "Nm",
    )

    draw_timeseries_pil(
        image,
        (1400, 840, 2520, 1140),
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 2] if tr.torques.size else np.asarray([])),
        "Swing-knee torque",
        "Nm",
    )

    draw_timeseries_pil(
        image,
        (1400, 1185, 2520, 1500),
        traces,
        lambda tr: (tr.t_state, tr.raw_cmt_values),
        "Cmt over time",
        "Cmt",
        xlabel="time (s)",
        step=False,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    print(f"Saved figure: {path}")


def make_figure_combined_knee(traces, active_trace, command_length, command_height, walk_direction):
    fig = plt.figure(figsize=(14.5, 7.6), constrained_layout=False)
    ax_static = fig.add_axes([0.055, 0.11, 0.34, 0.76])
    draw_active_static(ax_static, active_trace, command_length, command_height, walk_direction)

    axes = [
        fig.add_axes([0.48, 0.68, 0.49, 0.22]),
        fig.add_axes([0.48, 0.39, 0.49, 0.22]),
        fig.add_axes([0.48, 0.10, 0.49, 0.22]),
    ]
    plot_time_series(
        axes[0],
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 1] if tr.torques.size else np.asarray([])),
        "hip torque (Nm)",
        "Hip torque",
    )
    plot_time_series(
        axes[1],
        traces,
        lambda tr: (
            tr.t_action,
            np.sqrt(0.5 * (tr.torques[:, 0] ** 2 + tr.torques[:, 2] ** 2)) if tr.torques.size else np.asarray([]),
        ),
        "knee RMS torque (Nm)",
        "Two-knee RMS torque",
    )
    plot_time_series(
        axes[2],
        traces,
        lambda tr: (tr.t_state, tr.raw_cmt_values),
        "cumulative Cmt",
        "Cmt over time",
        step=False,
    )
    axes[-1].set_xlabel("time (s)")
    draw_manual_legend(fig, x0=0.48, y0=0.955)
    fig.suptitle("Five-controller comparison from the same initial state", fontsize=12, y=0.99)
    return fig


def make_figure_split_knee(traces, active_trace, command_length, command_height, walk_direction):
    fig = plt.figure(figsize=(14.5, 8.8), constrained_layout=False)
    ax_static = fig.add_axes([0.055, 0.10, 0.34, 0.78])
    draw_active_static(ax_static, active_trace, command_length, command_height, walk_direction)

    axes = [
        fig.add_axes([0.48, 0.75, 0.49, 0.16]),
        fig.add_axes([0.48, 0.54, 0.49, 0.16]),
        fig.add_axes([0.48, 0.33, 0.49, 0.16]),
        fig.add_axes([0.48, 0.11, 0.49, 0.16]),
    ]
    plot_time_series(
        axes[0],
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 1] if tr.torques.size else np.asarray([])),
        "hip torque (Nm)",
        "Hip torque",
    )
    plot_time_series(
        axes[1],
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 0] if tr.torques.size else np.asarray([])),
        "support knee torque (Nm)",
        "Support-knee torque",
    )
    plot_time_series(
        axes[2],
        traces,
        lambda tr: (tr.t_action, tr.torques[:, 2] if tr.torques.size else np.asarray([])),
        "swing knee torque (Nm)",
        "Swing-knee torque",
    )
    plot_time_series(
        axes[3],
        traces,
        lambda tr: (tr.t_state, tr.raw_cmt_values),
        "cumulative Cmt",
        "Cmt over time",
        step=False,
    )
    axes[-1].set_xlabel("time (s)")
    draw_manual_legend(fig, x0=0.48, y0=0.955)
    fig.suptitle("Five-controller comparison with separated knee torques", fontsize=12, y=0.99)
    return fig


def write_summary_csv(path, traces, sign_selector_choice, oracle_choice):
    rows = []
    for trace in traces:
        metrics = trace.frames[-1]["metrics"]
        rows.append(
            {
                "controller": trace.label,
                "terminal_reason": trace.terminal_reason,
                "success": trace.success,
                "steps": trace.steps,
                "final_valid_cmt": trace.final_valid_cmt,
                "final_raw_cmt": trace.final_raw_cmt,
                "final_step_error": metrics.get("step_error", np.nan),
                "final_height_error": metrics.get("height_error", np.nan),
                "safe_posture": metrics.get("safe_posture", False),
                "hip_line_separates_thighs": metrics.get("hip_line_separates_thighs", False),
                "sign_selector_choice": sign_selector_choice if trace.label == "Online sign selector" else "",
                "oracle_choice": oracle_choice if trace.label == "Oracle envelope" else "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {path}")


def main():
    print("Loading settings and policies...", flush=True)
    set_seed(RANDOM_SEED)
    cmt.DETERMINISTIC_POLICY = bool(DETERMINISTIC_POLICY)
    if MAX_STEPS_OVERRIDE is not None:
        cmt.MAX_STEPS = int(MAX_STEPS_OVERRIDE)

    curriculum = float(inherited(CURRICULUM_OVERRIDE, sim_cfg.CURRICULUM))
    cmt.update_curriculum(0, progress_override=curriculum)

    initial_state = initial_state_from_sim_config()
    command_length = float(inherited(COMMAND_STEP_LENGTH_OVERRIDE, sim_cfg.COMMAND_STEP_LENGTH))
    command_height = float(inherited(COMMAND_STEP_HEIGHT_OVERRIDE, sim_cfg.COMMAND_STEP_HEIGHT))
    walk_direction = walk_direction_value(inherited(WALK_DIRECTION_OVERRIDE, sim_cfg.WALK_DIRECTION))
    reset_phase = str(inherited(RESET_PHASE_OVERRIDE, sim_cfg.INITIAL_RESET_PHASE))
    policies = load_all_policies()

    print("Rolling out Active PPO...", flush=True)
    active = rollout_trace(
        "Active PPO",
        initial_state,
        walk_direction,
        reset_phase,
        command_length,
        command_height,
        policies["active"],
        "active",
        hip_sign=0.0,
    )
    # print("Rolling out Active-full reference...", flush=True)
    # active_full = rollout_trace(
    #     "Active-full reference",
    #     initial_state,
    #     walk_direction,
    #     reset_phase,
    #     command_length,
    #     command_height,
    #     policies["active_full"],
    #     "active_full",
    #     hip_sign=0.0,
    # )
    print("Rolling out Per-step sign reference...", flush=True)
    per_step = rollout_trace(
        "Per-step sign reference",
        initial_state,
        walk_direction,
        reset_phase,
        command_length,
        command_height,
        policies["per_step_sign"],
        "per_step_sign",
        hip_sign=0.0,
    )

    print("Choosing online passive sign...", flush=True)
    sign_name, sign_confidence = cmt.choose_passive_sign(
        policies["sign_selector"],
        initial_state,
        walk_direction,
        command_length,
        command_height,
    )
    selector_policy = policies["passive_pos"] if sign_name == "positive" else policies["passive_neg"]
    selector_hip_sign = 1.0 if sign_name == "positive" else -1.0
    selector_controller = "passive_positive" if sign_name == "positive" else "passive_negative"
    print(f"Rolling out Online sign selector ({sign_name})...", flush=True)
    online_selector = rollout_trace(
        "Online sign selector",
        initial_state,
        walk_direction,
        reset_phase,
        command_length,
        command_height,
        selector_policy,
        selector_controller,
        hip_sign=selector_hip_sign,
    )

    print("Rolling out Oracle positive/negative candidates...", flush=True)
    oracle_pos = rollout_trace(
        "Oracle positive candidate",
        initial_state,
        walk_direction,
        reset_phase,
        command_length,
        command_height,
        policies["passive_pos"],
        "passive_positive",
        hip_sign=1.0,
    )
    oracle_neg = rollout_trace(
        "Oracle negative candidate",
        initial_state,
        walk_direction,
        reset_phase,
        command_length,
        command_height,
        policies["passive_neg"],
        "passive_negative",
        hip_sign=-1.0,
    )
    oracle = choose_oracle_trace(oracle_pos, oracle_neg)
    oracle.label = "Oracle envelope"
    oracle_choice = "positive" if oracle is oracle_pos else "negative"

    # traces = [active, active_full, per_step, online_selector, oracle]
    traces = [active, per_step, online_selector, oracle]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Saving combined-knee figure...", flush=True)
    save_combined_knee_pil(
        OUTPUT_DIR / f"{OUTPUT_TAG}_combined_knee.png",
        traces,
        active,
        command_length,
        command_height,
        walk_direction,
    )

    print("Saving split-knee figure...", flush=True)
    save_split_knee_pil(
        OUTPUT_DIR / f"{OUTPUT_TAG}_split_knees.png",
        traces,
        active,
        command_length,
        command_height,
        walk_direction,
    )

    write_summary_csv(
        OUTPUT_DIR / f"{OUTPUT_TAG}_summary.csv",
        traces,
        f"{sign_name}_p{sign_confidence:.3f}",
        oracle_choice,
    )

    print("Initial q deg:", np.rad2deg(initial_state[0::2]))
    print("Initial dq:", initial_state[1::2])
    print(f"Command length/height: {command_length:.3f} m / {command_height:.3f} m")
    print(f"Direction/reset phase: {'forward' if walk_direction >= 0 else 'backward'} / {reset_phase}")
    print(f"Online selector chose: {sign_name} (p={sign_confidence:.3f})")
    print(f"Oracle envelope chose: {oracle_choice}")
    for trace in traces:
        print(
            f"{trace.label}: reason={trace.terminal_reason}, steps={trace.steps}, "
            f"raw_cmt={trace.final_raw_cmt:.4g}, valid_cmt={trace.final_valid_cmt:.4g}"
        )


if __name__ == "__main__":
    main()
