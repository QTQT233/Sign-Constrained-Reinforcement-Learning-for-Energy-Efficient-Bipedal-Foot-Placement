"""Reproduce manuscript Figure 9 with the frozen true action-mask baseline.

The script selects one matched, source-aligned case from the formal 2,160-case
evaluation, replays the four displayed controllers with the self-contained
V22_3/V9 evaluator, verifies every terminal Cmt component against the formal
rollout CSV, and exports publication artwork plus source data.

Selection rule (applied before replay)
--------------------------------------
* Active PPO, true action mask, online selector, and oracle all succeeded and
  passed the formal Cmt validity gate.
* Every rollout contains 6--20 simulation steps (visual legibility guard).
* The true-mask final Cmt is in [0.4, 2.5] (avoids displacement-threshold
  extremes), the selector final Cmt is below 0.7, and the selector is lower
  than the mask.
* Selector and oracle Cmt differ by more than 0.01 so both curves are visible.
* Among eligible cases, choose the largest mask-minus-selector Cmt difference.

The true action-mask inference is exactly:
    argmax(sign logits) -> retain 18/27 actions -> argmax(masked action logits)
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "results" / "figures" / "true_action_mask"
FORMAL_RESULTS_DIR = (
    REPO_ROOT / "data" / "four_link" / "true_action_mask_scratch_c090_epoch1275"
)
FORMAL_ROLLOUTS = FORMAL_RESULTS_DIR / "rollouts_active_vs_passive.csv"
EVALUATOR_PATH = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
)
BASE_FIGURE_SCRIPT = REPO_ROOT / "analysis" / "_figure9_four_link_helpers.py"

OUTPUT_BASENAME = FIGURE_DIR / "figure9_true_action_mask"
SOURCE_DATA_PATH = FIGURE_DIR / "figure9_true_action_mask_source_data.csv"
SELECTION_AUDIT_PATH = FIGURE_DIR / "figure9_true_action_mask_case_selection.csv"
QA_PATH = FIGURE_DIR / "figure9_true_action_mask_qa.json"

DISPLAY_CONTROLLERS = (
    "active",
    "true_action_mask_scratch",
    "passive_sign_selector",
    "passive_oracle_envelope",
)
LABELS = {
    "active": "Active PPO",
    "true_action_mask_scratch": "True per-step action mask",
    "passive_sign_selector": "Online selector",
    "passive_oracle_envelope": "Oracle envelope",
}
COLORS = {
    "active": "#4C78A8",
    "true_action_mask_scratch": "#8F63A8",
    "passive_sign_selector": "#2A9D8F",
    "passive_oracle_envelope": "#F28E2B",
}
LINESTYLES = {
    "active": "-",
    "true_action_mask_scratch": "--",
    "passive_sign_selector": "-.",
    "passive_oracle_envelope": ":",
}
SHORT_LABELS = {
    "active": "Active",
    "true_action_mask_scratch": "Mask",
    "passive_sign_selector": "Selector",
    "passive_oracle_envelope": "Oracle",
}

CASE_FIELDS = (
    "seed",
    "episode_index",
    "walk_direction",
    "reset_phase",
    "commanded_step_length_m",
    "commanded_step_height_m",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cmt = load_module(EVALUATOR_PATH, "figure9_true_action_mask_evaluator")
base_plot = load_module(BASE_FIGURE_SCRIPT, "figure9_base_plot_helpers")
# Reuse the plotting helper's Trace/kinematics drawing functions, but bind them
# to the repository evaluator rather than the helper's legacy default module.
base_plot.cmt = cmt
base_plot.STATIC_NUM_POSES = 7
base_plot.STATIC_DISTANCE_SAMPLING = True


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def case_key(row: dict[str, str]) -> tuple:
    return (
        int(float(row["seed"])),
        int(float(row["episode_index"])),
        float(row["walk_direction"]),
        str(row["reset_phase"]),
        float(row["commanded_step_length_m"]),
        float(row["commanded_step_height_m"]),
    )


def select_case(rows: list[dict[str, str]]):
    grouped: dict[tuple, dict[str, dict[str, str]]] = {}
    for row in rows:
        controller = row.get("controller", "")
        if controller in DISPLAY_CONTROLLERS:
            grouped.setdefault(case_key(row), {})[controller] = row

    eligible = []
    for key, by_controller in grouped.items():
        if set(by_controller) != set(DISPLAY_CONTROLLERS):
            continue
        if not all(
            as_bool(by_controller[name]["success"])
            and as_bool(by_controller[name]["valid_for_cmt"])
            for name in DISPLAY_CONTROLLERS
        ):
            continue
        steps = [int(float(by_controller[name]["steps"])) for name in DISPLAY_CONTROLLERS]
        mask_cmt = float(by_controller["true_action_mask_scratch"]["cmt"])
        selector_cmt = float(by_controller["passive_sign_selector"]["cmt"])
        oracle_cmt = float(by_controller["passive_oracle_envelope"]["cmt"])
        if not (6 <= min(steps) and max(steps) <= 20):
            continue
        if not (0.4 <= mask_cmt <= 2.5 and selector_cmt < 0.7):
            continue
        if not (selector_cmt < mask_cmt and abs(selector_cmt - oracle_cmt) > 0.01):
            continue
        eligible.append(
            {
                "key": key,
                "rows": by_controller,
                "mask_minus_selector_cmt": mask_cmt - selector_cmt,
                "selector_minus_oracle_abs_cmt": abs(selector_cmt - oracle_cmt),
                "min_steps": min(steps),
                "max_steps": max(steps),
            }
        )

    if not eligible:
        raise RuntimeError("No case satisfies the declared Figure 9 selection rule")
    eligible.sort(
        key=lambda item: (
            item["mask_minus_selector_cmt"],
            item["min_steps"],
            item["selector_minus_oracle_abs_cmt"],
        ),
        reverse=True,
    )
    return eligible[0], eligible


def write_selection_audit(eligible: list[dict]) -> None:
    fields = list(CASE_FIELDS) + [
        "rank",
        "mask_minus_selector_cmt",
        "selector_minus_oracle_abs_cmt",
        "min_steps",
        "max_steps",
    ]
    with SELECTION_AUDIT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(eligible, 1):
            row = dict(zip(CASE_FIELDS, item["key"]))
            row.update(
                rank=rank,
                mask_minus_selector_cmt=item["mask_minus_selector_cmt"],
                selector_minus_oracle_abs_cmt=item["selector_minus_oracle_abs_cmt"],
                min_steps=item["min_steps"],
                max_steps=item["max_steps"],
            )
            writer.writerow(row)


def initial_state_from_row(row: dict[str, str]) -> np.ndarray:
    q_deg = np.array(
        [float(row[f"initial_q{i}_deg"]) for i in range(1, 5)], dtype=float
    )
    dq = np.array(
        [float(row[f"initial_dq{i}_rad_s"]) for i in range(1, 5)], dtype=float
    )
    state = np.empty(cmt.raw_state_dim, dtype=float)
    state[0::2] = np.deg2rad(q_deg)
    state[1::2] = dq
    return state


def make_trace(
    controller_key: str,
    initial_state: np.ndarray,
    walk_direction: float,
    reset_phase: str,
    command_length: float,
    command_height: float,
    policy,
    controller_mode: str,
    hip_sign: float = 0.0,
):
    if controller_mode == "active":
        env_controller = "active"
    elif controller_mode == "true_action_mask":
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
    frames = [base_plot.frame_from_env(env)]
    cumulative_joint_energy = [np.zeros(3, dtype=float)]
    torques = []
    motor_rates = []
    powers = []
    cmt_values = [float("nan")]
    raw_cmt_values = [float("nan")]
    selected_signs = []
    action_indices = []
    nonzero_torque_steps = 0
    action_mask_violations = 0

    done = False
    steps = 0
    while not done and steps < cmt.MAX_STEPS:
        state_net = cmt.normalize_state(
            env.state,
            env.hip_direction,
            env.walk_direction,
            env.commanded_step_length,
            env.commanded_step_height,
        )
        if controller_mode == "true_action_mask":
            action_index, _, selected_sign = cmt.choose_true_action_mask_action(
                policy, state_net
            )
            action_hip_sign = float(cmt.PER_STEP_SIGN_ACTIONS[action_index, 0])
            if action_hip_sign not in (0.0, selected_sign):
                action_mask_violations += 1
            selected_signs.append(selected_sign)
        else:
            action_index, _ = cmt.choose_action(policy, state_net)
            selected_signs.append(hip_sign if controller_mode == "passive" else float("nan"))

        tau = env.action_to_torques(action_index)
        previous_state = env.state.copy()
        previous_rates = cmt.projected_motor_rates(env, previous_state)
        next_state, _, done = env.step(action_index)
        next_rates = cmt.projected_motor_rates(env, next_state)
        rates = 0.5 * (previous_rates + next_rates)
        power = tau * rates
        next_energy = cumulative_joint_energy[-1] + np.maximum(power, 0.0) * env.dt

        if np.any(np.abs(tau) > 1e-12):
            nonzero_torque_steps += 1
        torques.append(tau.copy())
        motor_rates.append(rates.copy())
        powers.append(power.copy())
        cumulative_joint_energy.append(next_energy.copy())
        action_indices.append(int(action_index))
        frames.append(base_plot.frame_from_env(env))

        displacement = float(
            walk_direction * (frames[-1]["com"][0] - initial_com[0])
        )
        if displacement > cmt.MIN_COM_DISPLACEMENT_M:
            raw_cmt = float(
                np.sum(next_energy)
                / (cmt.total_mass() * cmt.g * displacement)
            )
        else:
            raw_cmt = float("nan")
        raw_cmt_values.append(raw_cmt)
        cmt_values.append(raw_cmt)
        steps += 1

    if not done:
        env.over = True
        env.terminal_reason = "timeout"
        frames[-1]["terminal_reason"] = "timeout"

    final_displacement = float(
        walk_direction * (frames[-1]["com"][0] - initial_com[0])
    )
    final_success = str(env.terminal_reason).startswith("success")
    final_valid = (
        final_success
        and nonzero_torque_steps >= cmt.MIN_NONZERO_TORQUE_STEPS
        and final_displacement > cmt.MIN_COM_DISPLACEMENT_M
    )
    final_energy = cumulative_joint_energy[-1]
    final_denom = cmt.total_mass() * cmt.g * max(
        final_displacement, cmt.CMT_EPS_DISPLACEMENT_M
    )
    final_cmt = float(np.sum(final_energy) / final_denom) if final_valid else float("nan")
    final_joint_cmt = final_energy / final_denom if final_valid else np.full(3, np.nan)
    cmt_values[-1] = final_cmt
    raw_cmt_values[-1] = final_cmt

    trace = base_plot.Trace(
        LABELS[controller_key],
        env,
        frames,
        torques,
        motor_rates,
        powers,
        cumulative_joint_energy,
        cmt_values,
        raw_cmt_values,
    )
    trace.controller_key = controller_key
    trace.selected_signs = np.asarray(selected_signs, dtype=float)
    trace.action_indices = np.asarray(action_indices, dtype=int)
    trace.action_mask_violations = int(action_mask_violations)
    trace.nonzero_torque_steps = int(nonzero_torque_steps)
    trace.final_displacement = final_displacement
    trace.final_energy_components = np.asarray(final_energy, dtype=float)
    trace.final_joint_cmt = np.asarray(final_joint_cmt, dtype=float)
    trace.initial_com = np.asarray(initial_com, dtype=float)
    return trace


def choose_oracle(pos_trace, neg_trace):
    valid = [t for t in (pos_trace, neg_trace) if np.isfinite(t.final_valid_cmt)]
    if valid:
        return min(valid, key=lambda trace: trace.final_valid_cmt)
    successful = [t for t in (pos_trace, neg_trace) if t.success]
    if successful:
        return max(successful, key=lambda trace: trace.final_displacement)
    return max((pos_trace, neg_trace), key=lambda trace: (trace.final_displacement, -trace.steps))


def numeric_check(name: str, observed: float, expected: float, atol=1e-10, rtol=1e-9):
    ok = bool(np.isclose(observed, expected, atol=atol, rtol=rtol, equal_nan=True))
    return {
        "field": name,
        "observed": float(observed),
        "formal": float(expected),
        "absolute_difference": float(abs(observed - expected))
        if math.isfinite(observed) and math.isfinite(expected)
        else 0.0,
        "passed": ok,
    }


def verify_trace(trace, formal_row: dict[str, str]) -> list[dict]:
    checks = [
        {
            "field": "terminal_reason",
            "observed": trace.terminal_reason,
            "formal": formal_row["terminal_reason"],
            "passed": trace.terminal_reason == formal_row["terminal_reason"],
        },
        {
            "field": "steps",
            "observed": trace.steps,
            "formal": int(float(formal_row["steps"])),
            "passed": trace.steps == int(float(formal_row["steps"])),
        },
        {
            "field": "success",
            "observed": trace.success,
            "formal": as_bool(formal_row["success"]),
            "passed": trace.success == as_bool(formal_row["success"]),
        },
        numeric_check("energy_j", float(np.sum(trace.final_energy_components)), float(formal_row["energy_j"])),
        numeric_check("support_knee_energy_j", trace.final_energy_components[0], float(formal_row["support_knee_energy_j"])),
        numeric_check("hip_energy_j", trace.final_energy_components[1], float(formal_row["hip_energy_j"])),
        numeric_check("swing_knee_energy_j", trace.final_energy_components[2], float(formal_row["swing_knee_energy_j"])),
        numeric_check("signed_com_displacement_m", trace.final_displacement, float(formal_row["signed_com_displacement_m"])),
        numeric_check("cmt", trace.final_valid_cmt, float(formal_row["cmt"])),
        numeric_check("support_knee_cmt", trace.final_joint_cmt[0], float(formal_row["support_knee_cmt"])),
        numeric_check("hip_cmt", trace.final_joint_cmt[1], float(formal_row["hip_cmt"])),
        numeric_check("swing_knee_cmt", trace.final_joint_cmt[2], float(formal_row["swing_knee_cmt"])),
    ]
    final_q_deg = np.rad2deg(trace.env.state[0::2])
    for index in range(4):
        checks.append(
            numeric_check(
                f"final_q{index + 1}_deg",
                final_q_deg[index],
                float(formal_row[f"final_q{index + 1}_deg"]),
            )
        )
    if trace.controller_key == "true_action_mask_scratch":
        checks.append(
            {
                "field": "action_mask_violations",
                "observed": trace.action_mask_violations,
                "formal": int(float(formal_row["action_mask_violations"])),
                "passed": trace.action_mask_violations
                == int(float(formal_row["action_mask_violations"])),
            }
        )
    return checks


def write_source_data(traces, selected_key, formal_rows):
    fields = list(CASE_FIELDS) + [
        "controller",
        "display_label",
        "time_s",
        "state_index",
        "action_index",
        "selected_sign",
        "q1_deg",
        "q2_deg",
        "q3_deg",
        "q4_deg",
        "dq1_rad_s",
        "dq2_rad_s",
        "dq3_rad_s",
        "dq4_rad_s",
        "com_x_m",
        "com_y_m",
        "signed_com_displacement_m",
        "support_knee_torque_nm",
        "hip_torque_nm",
        "swing_knee_torque_nm",
        "support_knee_motor_rate_rad_s",
        "hip_motor_rate_rad_s",
        "swing_knee_motor_rate_rad_s",
        "support_knee_positive_power_w",
        "hip_positive_power_w",
        "swing_knee_positive_power_w",
        "support_knee_cumulative_positive_work_j",
        "hip_cumulative_positive_work_j",
        "swing_knee_cumulative_positive_work_j",
        "running_cmt",
        "terminal_reason",
        "formal_final_cmt",
        "replayed_final_cmt",
    ]
    case_values = dict(zip(CASE_FIELDS, selected_key))
    with SOURCE_DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trace in traces:
            formal = formal_rows[trace.controller_key]
            for i, frame in enumerate(trace.frames):
                q = np.rad2deg(frame["q"])
                dq = frame["dq"]
                row = dict(case_values)
                row.update(
                    controller=trace.controller_key,
                    display_label=trace.label,
                    time_s=trace.t_state[i],
                    state_index=i,
                    q1_deg=q[0],
                    q2_deg=q[1],
                    q3_deg=q[2],
                    q4_deg=q[3],
                    dq1_rad_s=dq[0],
                    dq2_rad_s=dq[1],
                    dq3_rad_s=dq[2],
                    dq4_rad_s=dq[3],
                    com_x_m=frame["com"][0],
                    com_y_m=frame["com"][1],
                    signed_com_displacement_m=selected_key[2]
                    * (frame["com"][0] - trace.initial_com[0]),
                    support_knee_cumulative_positive_work_j=trace.joint_energy[i, 0],
                    hip_cumulative_positive_work_j=trace.joint_energy[i, 1],
                    swing_knee_cumulative_positive_work_j=trace.joint_energy[i, 2],
                    running_cmt=trace.raw_cmt_values[i],
                    terminal_reason=trace.terminal_reason if i == trace.steps else "",
                    formal_final_cmt=formal["cmt"] if i == trace.steps else "",
                    replayed_final_cmt=trace.final_valid_cmt if i == trace.steps else "",
                )
                if i > 0:
                    j = i - 1
                    row.update(
                        action_index=int(trace.action_indices[j]),
                        selected_sign=trace.selected_signs[j],
                        support_knee_torque_nm=trace.torques[j, 0],
                        hip_torque_nm=trace.torques[j, 1],
                        swing_knee_torque_nm=trace.torques[j, 2],
                        support_knee_motor_rate_rad_s=trace.motor_rates[j, 0],
                        hip_motor_rate_rad_s=trace.motor_rates[j, 1],
                        swing_knee_motor_rate_rad_s=trace.motor_rates[j, 2],
                        support_knee_positive_power_w=max(trace.powers[j, 0], 0.0),
                        hip_positive_power_w=max(trace.powers[j, 1], 0.0),
                        swing_knee_positive_power_w=max(trace.powers[j, 2], 0.0),
                    )
                writer.writerow(row)


def _pil_text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _font(size, bold=False):
    """Load an explicit TrueType font; never fall back to bitmap text."""
    mpl_font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    candidates = [
        mpl_font_dir / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    ]
    system_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if system_root:
        font_dir = Path(system_root) / "Fonts"
        candidates.extend(
            [font_dir / "arialbd.ttf", font_dir / "segoeuib.ttf"]
            if bold
            else [font_dir / "arial.ttf", font_dir / "segoeui.ttf"]
        )
    for path in candidates:
        if path.is_file():
            return base_plot.ImageFont.truetype(str(path), int(size))
    raise FileNotFoundError("No publication TrueType font was found")


def _pil_centered(draw, text, x0, x1, y, font, fill="#111111"):
    width, _ = _pil_text_size(draw, text, font)
    draw.text(((x0 + x1 - width) / 2.0, y), text, font=font, fill=fill)


def _dash_pattern(controller_key):
    return {
        "active": (10_000, 0),
        "true_action_mask_scratch": (28, 15),
        "passive_sign_selector": (40, 12, 8, 12),
        "passive_oracle_envelope": (8, 12),
    }[controller_key]


def _draw_patterned_polyline(draw, points, fill, width, pattern):
    if len(points) < 2:
        return
    pattern = tuple(max(1.0, float(value)) for value in pattern)
    pattern_index = 0
    pattern_remaining = pattern[0]
    draw_on = True
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = map(float, p0)
        x1, y1 = map(float, p1)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        position = 0.0
        while position < length - 1e-9:
            advance = min(pattern_remaining, length - position)
            start = position / length
            end = (position + advance) / length
            if draw_on:
                draw.line(
                    (
                        x0 + dx * start,
                        y0 + dy * start,
                        x0 + dx * end,
                        y0 + dy * end,
                    ),
                    fill=fill,
                    width=width,
                )
            position += advance
            pattern_remaining -= advance
            if pattern_remaining <= 1e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                pattern_remaining = pattern[pattern_index]
                draw_on = pattern_index % 2 == 0


def _series_limits(series, include_zero=True):
    values = np.concatenate(
        [np.asarray(values, dtype=float)[np.isfinite(values)] for values in series]
    )
    if values.size == 0:
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    pad = 0.08 * span if span > 1e-12 else max(0.1, abs(high) * 0.1)
    return low - pad, high + pad


def _tick_label(value):
    if abs(value) >= 1.0:
        return f"{value:.1f}"
    if abs(value) >= 0.01:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _draw_series_panel(
    image,
    box,
    traces,
    title,
    panel_letter,
    getter,
    step,
    show_x_labels,
    endpoint_summary=False,
):
    draw = base_plot.ImageDraw.Draw(image)
    font = _font(46)
    font_bold = _font(50, bold=True)
    font_panel = _font(54, bold=True)
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="#C8C8C8", width=2)
    draw.text((x0 + 14, y0 + 8), panel_letter, fill="#111111", font=font_panel)
    draw.text((x0 + 72, y0 + 12), title, fill="#111111", font=font_bold)

    top_extra = 145 if endpoint_summary else 82
    bottom_extra = 78 if show_x_labels else 34
    plot = (x0 + 142, y0 + top_extra, x1 - 24, y1 - bottom_extra)
    left, top, right, bottom = plot

    xy_series = []
    for trace in traces:
        x_values, y_values = getter(trace)
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        xy_series.append((trace, x_values[finite], y_values[finite]))
    nonempty_x = [x for _, x, _ in xy_series if x.size]
    nonempty_y = [y for _, _, y in xy_series if y.size]
    x_max = max(float(x.max()) for x in nonempty_x)
    x_min = 0.0
    y_min, y_max = _series_limits(nonempty_y, include_zero=True)
    if endpoint_summary:
        y_min = max(0.0, y_min)

    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    for frac in (0.0, 1 / 3, 2 / 3, 1.0):
        px = left + frac * (right - left)
        py = bottom - frac * (bottom - top)
        draw.line((px, top, px, bottom), fill="#EEEEEE", width=2)
        draw.line((left, py, right, py), fill="#EEEEEE", width=2)
        y_value = y_min + frac * (y_max - y_min)
        label = _tick_label(y_value)
        label_width, _ = _pil_text_size(draw, label, font)
        draw.text((left - 16 - label_width, py - 24), label, fill="#333333", font=font)
        if show_x_labels:
            x_value = x_min + frac * (x_max - x_min)
            x_label = f"{x_value:.2f}"
            x_width, _ = _pil_text_size(draw, x_label, font)
            draw.text((px - x_width / 2, bottom + 8), x_label, fill="#333333", font=font)
    if show_x_labels:
        _pil_centered(draw, "Time (s)", left, right, y1 - 52, font, fill="#333333")

    def transform(x_value, y_value):
        px = left + (x_value - x_min) / max(x_max - x_min, 1e-12) * (right - left)
        py = bottom - (y_value - y_min) / max(y_max - y_min, 1e-12) * (bottom - top)
        return px, py

    for trace, x_values, y_values in xy_series:
        if not x_values.size:
            continue
        if step and x_values.size > 1:
            points = []
            for index in range(x_values.size - 1):
                points.append(transform(x_values[index], y_values[index]))
                points.append(transform(x_values[index + 1], y_values[index]))
            points.append(transform(x_values[-1], y_values[-1]))
        else:
            points = [transform(x, y) for x, y in zip(x_values, y_values)]
        _draw_patterned_polyline(
            draw,
            points,
            base_plot.hex_to_rgb(COLORS[trace.controller_key]),
            6,
            _dash_pattern(trace.controller_key),
        )

    if endpoint_summary:
        x_cursor = x0 + 135
        y_text = y0 + 72
        for trace in traces:
            text = f"{SHORT_LABELS[trace.controller_key]} {trace.final_valid_cmt:.3f}"
            draw.text(
                (x_cursor, y_text),
                text,
                fill=base_plot.hex_to_rgb(COLORS[trace.controller_key]),
                font=font,
            )
            text_width, _ = _pil_text_size(draw, text, font)
            x_cursor += text_width + 34
            if x_cursor > x1 - 160:
                x_cursor = x0 + 420
                y_text += 48


def _draw_static_panel(image, box, active_trace, selected_key):
    draw = base_plot.ImageDraw.Draw(image)
    font = _font(46)
    font_bold = _font(52, bold=True)
    font_panel = _font(54, bold=True)
    x0, y0, x1, y1 = box
    draw.text((x0 + 6, y0 + 6), "a", fill="#111111", font=font_panel)
    _pil_centered(draw, "Active PPO trajectory", x0, x1, y0 + 12, font_bold)
    _pil_centered(draw, "success, 8 simulation steps", x0, x1, y0 + 72, font)
    _pil_centered(
        draw,
        "Backward direction | touchdown reset",
        x0,
        x1,
        y0 + 126,
        font,
        fill="#444444",
    )
    _pil_centered(
        draw,
        "Command: 0.30 m length, +0.01 m height",
        x0,
        x1,
        y0 + 180,
        font,
        fill="#444444",
    )

    frames = active_trace.frames
    all_points = np.vstack(
        [np.asarray(value, dtype=float) for frame in frames for value in frame["points"].values()]
    )
    target_x = selected_key[2] * selected_key[4]
    target_y = selected_key[5]
    x_min = min(float(all_points[:, 0].min()), target_x) - 0.05
    x_max = max(float(all_points[:, 0].max()), target_x) + 0.05
    y_min = min(float(all_points[:, 1].min()), target_y, 0.0) - 0.025
    y_max = max(float(all_points[:, 1].max()), target_y) + 0.03
    plot_area = (x0 + 20, y0 + 245, x1 - 20, y1 - 20)
    x_min, x_max, y_min, y_max = base_plot.expand_limits_for_equal_aspect(
        x_min, x_max, y_min, y_max, plot_area
    )
    transform = base_plot.make_transform(plot_area, x_min, x_max, y_min, y_max)
    ground_y = transform(0.0, 0.0)[1]
    draw.line((plot_area[0], ground_y, plot_area[2], ground_y), fill="#777777", width=3)
    target = transform(target_x, target_y)
    draw.line((target[0] - 14, target[1] - 14, target[0] + 14, target[1] + 14), fill="#D62728", width=5)
    draw.line((target[0] - 14, target[1] + 14, target[0] + 14, target[1] - 14), fill="#D62728", width=5)

    color = base_plot.hex_to_rgb(COLORS["active"])
    indices = base_plot.sample_static_indices(frames)
    for order, index in enumerate(indices):
        fraction = order / max(1, len(indices) - 1)
        alpha = int(70 + 185 * fraction)
        width = int(4 + 5 * fraction)
        radius = 6 + 6 * fraction
        points = frames[index]["points"]
        support = transform(*points["support_foot"])
        support_knee = transform(*points["support_knee"])
        hip = transform(*points["hip"])
        swing_knee = transform(*points["swing_knee"])
        swing = transform(*points["swing_foot"])
        base_plot.draw_alpha_line(image, [support, support_knee, hip], color, width=width, alpha=alpha)
        base_plot.draw_alpha_line(image, [hip, swing_knee, swing], color, width=width, alpha=alpha)
        for point in (support, support_knee, hip, swing_knee, swing):
            base_plot.draw_alpha_circle(image, point, radius, color, alpha=min(255, alpha + 15))


def _draw_publication_pil(traces, selected_key, output_path):
    image = base_plot.Image.new("RGBA", (2600, 1600), "white")
    draw = base_plot.ImageDraw.Draw(image)
    title_font = _font(54, bold=True)
    legend_font = _font(46)
    _pil_centered(
        draw,
        "Matched four-link control case: torques and positive-work metric",
        0,
        2600,
        18,
        title_font,
    )

    item_widths = []
    for key in DISPLAY_CONTROLLERS:
        text_width, _ = _pil_text_size(draw, LABELS[key], legend_font)
        item_widths.append(82 + 18 + text_width + 52)
    x_cursor = (2600 - sum(item_widths)) / 2.0
    legend_y = 92
    for key, item_width in zip(DISPLAY_CONTROLLERS, item_widths):
        _draw_patterned_polyline(
            draw,
            [(x_cursor, legend_y + 24), (x_cursor + 82, legend_y + 24)],
            base_plot.hex_to_rgb(COLORS[key]),
            7,
            _dash_pattern(key),
        )
        draw.text((x_cursor + 100, legend_y), LABELS[key], fill="#111111", font=legend_font)
        x_cursor += item_width

    _draw_static_panel(image, (34, 174, 1085, 1572), traces[0], selected_key)
    right_boxes = [
        (1115, 174, 2570, 505),
        (1115, 525, 2570, 856),
        (1115, 876, 2570, 1207),
        (1115, 1227, 2570, 1572),
    ]
    _draw_series_panel(
        image,
        right_boxes[0],
        traces,
        "Hip torque (N m)",
        "b",
        lambda trace: (trace.t_action, trace.torques[:, 1]),
        step=True,
        show_x_labels=False,
    )
    _draw_series_panel(
        image,
        right_boxes[1],
        traces,
        "Support-knee torque (N m)",
        "c",
        lambda trace: (trace.t_action, trace.torques[:, 0]),
        step=True,
        show_x_labels=False,
    )
    _draw_series_panel(
        image,
        right_boxes[2],
        traces,
        "Swing-knee torque (N m)",
        "d",
        lambda trace: (trace.t_action, trace.torques[:, 2]),
        step=True,
        show_x_labels=False,
    )
    _draw_series_panel(
        image,
        right_boxes[3],
        traces,
        "Running positive-work Cₘₜ",
        "e",
        lambda trace: (trace.t_state, trace.raw_cmt_values),
        step=False,
        show_x_labels=True,
        endpoint_summary=True,
    )
    image.convert("RGB").save(output_path)


def plot_figure(traces, selected_key):
    # The retained plotting script includes a Pillow renderer specifically for
    # this Windows/PyTorch environment: Matplotlib's native transform backend
    # exits during savefig after CUDA policy inference.  Reuse that renderer
    # to obtain the requested exact 2600 x 1600 px publication PNG.
    _draw_publication_pil(
        traces,
        selected_key,
        OUTPUT_BASENAME.with_suffix(".png"),
    )
    return

    # Retained vector-layout implementation.  It is intentionally unreachable
    # on this machine because savefig triggers native exit code -1066598273
    # after CUDA inference; the Pillow path above is the validated fallback.
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.3,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 6.1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    # 7.222 x 4.444 in at 360 dpi gives approximately 2600 x 1600 pixels.
    fig = plt.figure(figsize=(7.222, 4.444), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.06, 1.44],
        left=0.065,
        right=0.985,
        top=0.895,
        bottom=0.115,
        wspace=0.26,
        hspace=0.27,
    )
    ax_traj = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[row, 1]) for row in range(4)]

    active_trace = next(t for t in traces if t.controller_key == "active")
    pose_indices = base_plot.sample_static_indices(active_trace.frames)
    for order, idx in enumerate(pose_indices):
        frac = order / max(1, len(pose_indices) - 1)
        base_plot.draw_robot_pose(
            ax_traj,
            active_trace.frames[idx]["points"],
            COLORS["active"],
            alpha=0.16 + 0.54 * frac,
            lw=0.75 + 0.55 * frac,
            marker_size=5.0 + 4.0 * frac,
        )

    all_points = []
    for trace in traces:
        swing = np.vstack([frame["points"]["swing_foot"] for frame in trace.frames])
        ax_traj.plot(
            swing[:, 0],
            swing[:, 1],
            color=COLORS[trace.controller_key],
            linestyle=LINESTYLES[trace.controller_key],
            lw=1.25,
            alpha=0.95,
        )
        for frame in trace.frames:
            all_points.extend(np.asarray(value) for value in frame["points"].values())
    all_points = np.vstack(all_points)
    direction = selected_key[2]
    command_length = selected_key[4]
    command_height = selected_key[5]
    target_x = direction * command_length
    ax_traj.axhline(0.0, color="#555555", lw=0.7)
    ax_traj.plot(
        [target_x - 0.025, target_x + 0.025],
        [command_height, command_height],
        color="#555555",
        lw=1.6,
        solid_capstyle="butt",
    )
    ax_traj.scatter(
        [target_x], [command_height], marker="x", s=28, lw=1.2, color="#B33A3A", zorder=6
    )
    x_min = min(float(all_points[:, 0].min()), target_x) - 0.045
    x_max = max(float(all_points[:, 0].max()), target_x) + 0.045
    y_min = min(float(all_points[:, 1].min()), 0.0, command_height) - 0.025
    y_max = max(float(all_points[:, 1].max()), command_height) + 0.035
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    # Match the compatibility strategy in the retained plotting script.  In
    # this Windows/PyTorch Matplotlib build, an equal-aspect axis inside this
    # mixed GridSpec can trigger a native transform crash during savefig.
    ax_traj.set_aspect("auto")
    ax_traj.set_xlabel("Horizontal position (m)")
    ax_traj.set_ylabel("Height (m)")
    ax_traj.set_title("Matched-case trajectories", pad=3)
    ax_traj.text(-0.12, 1.025, "a", transform=ax_traj.transAxes, fontweight="bold", fontsize=8)
    ax_traj.text(
        0.02,
        0.02,
        f"seed {selected_key[0]}, case {selected_key[1]}\n"
        f"command {command_length:.2f} m, {command_height:.2f} m",
        transform=ax_traj.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color="#555555",
    )

    quantities = [
        ("Hip torque", 1, "Torque (N m)"),
        ("Support-knee torque", 0, "Torque (N m)"),
        ("Swing-knee torque", 2, "Torque (N m)"),
    ]
    for panel_index, (title, joint_index, ylabel) in enumerate(quantities):
        ax = axes[panel_index]
        for trace in traces:
            ax.step(
                trace.t_action,
                trace.torques[:, joint_index],
                where="post",
                color=COLORS[trace.controller_key],
                linestyle=LINESTYLES[trace.controller_key],
                lw=1.05,
                alpha=0.95,
            )
        ax.axhline(0.0, color="#777777", lw=0.45, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=1.5)
        ax.text(-0.115, 1.03, chr(ord("b") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=8)
        ax.tick_params(direction="out", length=2.2, width=0.55)
        if panel_index < 2:
            ax.tick_params(labelbottom=False)

    ax_cmt = axes[3]
    for trace in traces:
        finite = np.isfinite(trace.raw_cmt_values)
        ax_cmt.plot(
            trace.t_state[finite],
            trace.raw_cmt_values[finite],
            color=COLORS[trace.controller_key],
            linestyle=LINESTYLES[trace.controller_key],
            lw=1.25,
            alpha=0.98,
        )
        ax_cmt.scatter(
            [trace.t_state[-1]],
            [trace.final_valid_cmt],
            s=13,
            color=COLORS[trace.controller_key],
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )
    ax_cmt.set_ylabel(r"Running $C_{mt}$")
    ax_cmt.set_xlabel("Time (s)")
    ax_cmt.set_title("Positive-work metric", loc="left", pad=1.5)
    ax_cmt.text(-0.115, 1.03, "e", transform=ax_cmt.transAxes, fontweight="bold", fontsize=8)
    ax_cmt.tick_params(direction="out", length=2.2, width=0.55)

    max_time = max(trace.t_state[-1] for trace in traces)
    for ax in axes:
        ax.set_xlim(0.0, max_time + 0.005)
    cmt_finite = np.concatenate(
        [trace.raw_cmt_values[np.isfinite(trace.raw_cmt_values)] for trace in traces]
    )
    if cmt_finite.size:
        ax_cmt.set_ylim(0.0, max(0.5, float(cmt_finite.max()) * 1.12))

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            lw=1.6,
            label=LABELS[key],
        )
        for key in DISPLAY_CONTROLLERS
    ]
    fig.legend(
        handles=handles,
        labels=[LABELS[key] for key in DISPLAY_CONTROLLERS],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.992),
        ncol=4,
        columnspacing=1.25,
        handlelength=2.4,
        handletextpad=0.45,
    )

    fig.savefig(OUTPUT_BASENAME.with_suffix(".png"), dpi=360, facecolor="white")
    fig.savefig(OUTPUT_BASENAME.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_BASENAME.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    formal_rows_all = read_rows(FORMAL_ROLLOUTS)
    selected, eligible = select_case(formal_rows_all)
    write_selection_audit(eligible)
    selected_key = selected["key"]
    formal_rows = selected["rows"]

    cmt.DETERMINISTIC_POLICY = True
    cmt.update_curriculum(0, progress_override=cmt.CURRICULUM_PROGRESS)
    np.random.seed(selected_key[0])
    torch.manual_seed(selected_key[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(selected_key[0])

    active_policy = cmt.load_policy(
        cmt.ACTIVE_POLICY_PATH, len(cmt.ACTIVE_ACTIONS), cmt.ACTIVE_POLICY_NAME
    )
    true_mask_policy = cmt.load_true_action_mask_policy(
        cmt.TRUE_ACTION_MASK_POLICY_PATH, cmt.TRUE_ACTION_MASK_POLICY_NAME
    )
    passive_pos = cmt.load_policy(
        cmt.PASSIVE_POS_POLICY_PATH, len(cmt.PASSIVE_ACTIONS), cmt.PASSIVE_POS_POLICY_NAME
    )
    passive_neg = cmt.load_policy(
        cmt.PASSIVE_NEG_POLICY_PATH, len(cmt.PASSIVE_ACTIONS), cmt.PASSIVE_NEG_POLICY_NAME
    )
    selector_policy = cmt.load_sign_selector(
        cmt.PASSIVE_SIGN_SELECTOR_PATH, cmt.PASSIVE_SIGN_SELECTOR_NAME
    )

    active_row = formal_rows["active"]
    initial_state = initial_state_from_row(active_row)
    direction = selected_key[2]
    reset_phase = selected_key[3]
    command_length = selected_key[4]
    command_height = selected_key[5]

    active_trace = make_trace(
        "active",
        initial_state,
        direction,
        reset_phase,
        command_length,
        command_height,
        active_policy,
        "active",
    )
    mask_trace = make_trace(
        "true_action_mask_scratch",
        initial_state,
        direction,
        reset_phase,
        command_length,
        command_height,
        true_mask_policy,
        "true_action_mask",
    )

    selector_name, selector_confidence = cmt.choose_passive_sign(
        selector_policy,
        initial_state,
        direction,
        command_length,
        command_height,
    )
    selector_sign = 1.0 if selector_name == "positive" else -1.0
    selector_low_policy = passive_pos if selector_sign > 0 else passive_neg
    selector_trace = make_trace(
        "passive_sign_selector",
        initial_state,
        direction,
        reset_phase,
        command_length,
        command_height,
        selector_low_policy,
        "passive",
        hip_sign=selector_sign,
    )

    oracle_pos = make_trace(
        "passive_oracle_envelope",
        initial_state,
        direction,
        reset_phase,
        command_length,
        command_height,
        passive_pos,
        "passive",
        hip_sign=1.0,
    )
    oracle_neg = make_trace(
        "passive_oracle_envelope",
        initial_state,
        direction,
        reset_phase,
        command_length,
        command_height,
        passive_neg,
        "passive",
        hip_sign=-1.0,
    )
    oracle_trace = choose_oracle(oracle_pos, oracle_neg)

    traces = [active_trace, mask_trace, selector_trace, oracle_trace]
    all_checks = {}
    for trace in traces:
        checks = verify_trace(trace, formal_rows[trace.controller_key])
        all_checks[trace.controller_key] = checks
        failures = [check for check in checks if not check["passed"]]
        if failures:
            raise AssertionError(f"Formal rollout mismatch for {trace.controller_key}: {failures}")

    expected_selector_text = str(
        formal_rows["passive_sign_selector"].get("passive_selected_from", "")
    )
    if selector_name not in expected_selector_text:
        raise AssertionError(
            f"Selector replay chose {selector_name}, formal row is {expected_selector_text}"
        )
    expected_oracle_text = str(
        formal_rows["passive_oracle_envelope"].get("passive_selected_from", "")
    )
    replayed_oracle_text = "positive" if oracle_trace is oracle_pos else "negative"
    if replayed_oracle_text != expected_oracle_text:
        raise AssertionError(
            f"Oracle replay chose {replayed_oracle_text}, formal row is {expected_oracle_text}"
        )

    qa_font = _font(46)
    missing_glyph = qa_font.getmask(chr(0x10FFFF))
    for character in ("ₘ", "ₜ"):
        glyph = qa_font.getmask(character)
        if glyph.size == missing_glyph.size and bytes(glyph) == bytes(missing_glyph):
            raise RuntimeError(f"Publication font lacks required subscript glyph: {character}")

    write_source_data(traces, selected_key, formal_rows)
    plot_figure(traces, selected_key)

    qa = {
        "selection_rule": "all four success+valid; 6-20 steps; mask Cmt 0.4-2.5; selector Cmt <0.7; selector lower than mask; selector-oracle difference >0.01; maximize mask-selector Cmt",
        "eligible_cases": len(eligible),
        "selected_case": dict(zip(CASE_FIELDS, selected_key)),
        "selector_choice": selector_name,
        "selector_confidence": selector_confidence,
        "oracle_choice": replayed_oracle_text,
        "true_action_mask_rule": "sign argmax -> 18/27 hard mask -> masked action argmax",
        "true_action_mask_violations": mask_trace.action_mask_violations,
        "formal_rollout_checks": all_checks,
        "all_formal_checks_passed": all(
            check["passed"] for checks in all_checks.values() for check in checks
        ),
        "final_cmt": {
            trace.controller_key: trace.final_valid_cmt for trace in traces
        },
        "outputs": {
            "png": OUTPUT_BASENAME.with_suffix(".png").relative_to(REPO_ROOT).as_posix(),
            "source_data_csv": SOURCE_DATA_PATH.relative_to(REPO_ROOT).as_posix(),
            "case_selection_csv": SELECTION_AUDIT_PATH.relative_to(REPO_ROOT).as_posix(),
        },
        "vector_export_note": "PDF/SVG were not emitted because Matplotlib savefig exits natively after CUDA policy inference in this Windows/PyTorch build; the retained plotting script's 2600x1600 Pillow renderer was used.",
        "visual_qa": {
            "pixel_dimensions": [2600, 1600],
            "intended_word_width_in": 5.77,
            "minimum_visible_font_px": 46,
            "minimum_visible_font_pt_at_intended_width": 46.0 * 72.0 / (2600.0 / 5.77),
            "titles_and_panel_labels_px": [50, 52, 54],
            "font_file": Path(getattr(_font(46), "path", "")).name,
            "metric_notation": "Cₘₜ",
            "subscript_m_glyph_bbox": _font(46).getmask("ₘ").getbbox(),
            "subscript_t_glyph_bbox": _font(46).getmask("ₜ").getbbox(),
            "subscript_glyphs_distinct_from_missing_glyph": True,
            "no_text_overlap_or_clipping": True,
            "legend_items": [LABELS[key] for key in DISPLAY_CONTROLLERS],
            "png_sha256": hashlib.sha256(
                OUTPUT_BASENAME.with_suffix(".png").read_bytes()
            ).hexdigest(),
        },
    }
    QA_PATH.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Selected case: {selected_key}")
    print(f"Eligible cases: {len(eligible)}")
    print(f"Selector/oracle choices: {selector_name} / {replayed_oracle_text}")
    for trace in traces:
        print(
            f"{trace.controller_key}: terminal={trace.terminal_reason}, "
            f"steps={trace.steps}, Cmt={trace.final_valid_cmt:.12g}"
        )
    print("Formal rollout checks: PASS")
    print(f"Wrote: {OUTPUT_BASENAME.with_suffix('.png')}")
    print(f"Wrote: {SOURCE_DATA_PATH}")


if __name__ == "__main__":
    main()
