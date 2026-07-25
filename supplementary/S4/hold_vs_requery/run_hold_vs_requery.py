"""Frozen V22_3/V9 inference-time hold-versus-requery experiment.

This program changes only the selector query schedule:

* ``hold`` queries the frozen selector at the transition start and keeps the
  chosen one-sided expert for the complete rollout.
* ``requery`` queries the same frozen selector at every control step and uses
  the corresponding frozen one-sided expert for that step.

The dynamics, reset states, commands, success gate, work integration, Cmt
validity rule, torque limits, deterministic action selection, selector,
positive expert, and negative expert are shared.  The 2,160 initial conditions
are read from the released formal V22_3/V9 rollout archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
FORMAL_EVALUATOR = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
)
FORMAL_ROLLOUTS = (
    REPO_ROOT
    / "data"
    / "four_link"
    / "true_action_mask_scratch_c090_epoch1275"
    / "rollouts_active_vs_passive.csv"
)
FORMAL_PAIRED = (
    REPO_ROOT
    / "data"
    / "four_link"
    / "true_action_mask_scratch_c090_epoch1275"
    / "paired_trials.csv"
)
SELECTOR_PATH = (
    REPO_ROOT
    / "models"
    / "four_link"
    / "paper_four_link_passive_sign_selector_v22_3_grid.pth"
)
POSITIVE_EXPERT_PATH = (
    REPO_ROOT
    / "models"
    / "four_link"
    / "v22_3_v9_uni_pos_400_grid"
    / "V22_3V9UniPos400Grid_c097_Policy_best.pth"
)
NEGATIVE_EXPERT_PATH = (
    REPO_ROOT
    / "models"
    / "four_link"
    / "v22_3_v9_uni_neg_400_grid"
    / "V22_3V9UniNeg400Grid_c100_Policy_best.pth"
)

EXPECTED_SHA256 = {
    "src/four_link/evaluation/"
    "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py":
        "c13c0c45b092fc43aa75fc091e870a14e4d8b6730cda5b2b8659c21a2d2bb186",
    "models/four_link/paper_four_link_passive_sign_selector_v22_3_grid.pth":
        "99fdebbe1ecdccfdfd1d36902ff4dd6936712488065aa0e6cf5c8391b24979e1",
    "models/four_link/v22_3_v9_uni_pos_400_grid/"
    "V22_3V9UniPos400Grid_c097_Policy_best.pth":
        "521962b8a3deea4411dc1b1c801cf9d8fcecfafa51d240f6752c07952ec1926a",
    "models/four_link/v22_3_v9_uni_neg_400_grid/"
    "V22_3V9UniNeg400Grid_c100_Policy_best.pth":
        "4752c0c4632d5958ee64c7f3004a76e9a7bd7f3204dd914fdb7003bce9a92d7c",
    "data/four_link/true_action_mask_scratch_c090_epoch1275/"
    "rollouts_active_vs_passive.csv":
        "39557858bdee443dcfbb837b15a2c337570eb8b8828721fd5e4dfa00d42f6ade",
    "data/four_link/true_action_mask_scratch_c090_epoch1275/"
    "paired_trials.csv":
        "cab0475943a31109c386910f815e1a361fc68594c3ff1b61c872a51356db1f4d",
}

EXPECTED_CASE_COUNT = 2160
SMOKE_EPISODE_INDICES = [
    1, 40, 41, 120, 241, 360, 361, 540, 720, 721, 900, 1080,
    1081, 1260, 1440, 1441, 1620, 1800, 1921, 2040, 2121, 2160,
]
NUMERIC_ABS_TOL = 1e-10
NUMERIC_REL_TOL = 1e-9
ENERGY_DISPLACEMENT_ABS_TOL = 1e-7
ENERGY_DISPLACEMENT_REL_TOL = 1e-7
CMT_ABS_TOL = 1e-7
CMT_REL_TOL = 1e-7
FINAL_ANGLE_ABS_TOL_DEG = 1e-6
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPLICATES = 10000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_formal_module():
    spec = importlib.util.spec_from_file_location("formal_v22_3_v9_evaluator", FORMAL_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create a module specification for the formal evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError("Cannot parse Boolean value {!r}".format(value))


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def validate_inputs():
    observed = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = REPO_ROOT / Path(relative)
        if not path.exists():
            raise FileNotFoundError("Missing required input: {}".format(path))
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            raise RuntimeError(
                "Input checksum mismatch for {}: expected {}, observed {}".format(
                    relative, expected, observed[relative]
                )
            )
    return observed


def selector_reference_rows():
    rows = [
        row for row in read_csv(FORMAL_ROLLOUTS)
        if row["controller"] == "passive_sign_selector"
    ]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if len(rows) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            "Expected {} selector reference rows, found {}".format(
                EXPECTED_CASE_COUNT, len(rows)
            )
        )
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(1, EXPECTED_CASE_COUNT + 1)):
        raise RuntimeError("Formal selector episode indices are not exactly 1..2160")
    keys = {
        (
            row["seed"],
            row["episode_index"],
            row["walk_direction"],
            row["reset_phase"],
            row["commanded_step_length_m"],
            row["commanded_step_height_m"],
        )
        for row in rows
    }
    if len(keys) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Formal selector archive contains duplicate case keys")
    return rows


def initial_state_from_reference(row):
    q_deg = [
        float(row["initial_q1_deg"]),
        float(row["initial_q2_deg"]),
        float(row["initial_q3_deg"]),
        float(row["initial_q4_deg"]),
    ]
    dq = [
        float(row["initial_dq1_rad_s"]),
        float(row["initial_dq2_rad_s"]),
        float(row["initial_dq3_rad_s"]),
        float(row["initial_dq4_rad_s"]),
    ]
    state = np.empty(8, dtype=float)
    state[0::2] = np.deg2rad(np.asarray(q_deg, dtype=float))
    state[1::2] = np.asarray(dq, dtype=float)
    return state


def generate_formal_initial_states(formal):
    """Regenerate the exact pre-serialization initial states in formal loop order."""
    random.seed(formal.RANDOM_SEEDS[0])
    np.random.seed(formal.RANDOM_SEEDS[0])
    torch.manual_seed(formal.RANDOM_SEEDS[0])
    formal.update_curriculum(0, progress_override=formal.CURRICULUM_PROGRESS)
    command_lengths, command_heights = formal.command_grid_values()
    generated = {}
    episode_index = 0
    for seed in formal.RANDOM_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        for walk_direction in formal.WALK_DIRECTIONS:
            for step_length in command_lengths:
                for step_height in command_heights:
                    for _ in range(formal.EPISODES_PER_GRID_CELL_DIRECTION):
                        episode_index += 1
                        reset_phase = formal.choose_eval_reset_phase(walk_direction)
                        initial_state = formal.sample_initial_state(
                            walk_direction,
                            reset_phase,
                            step_length,
                            step_height,
                        )
                        generated[episode_index] = {
                            "state": initial_state.copy(),
                            "seed": int(seed),
                            "walk_direction": float(walk_direction),
                            "reset_phase": str(reset_phase),
                            "commanded_step_length_m": float(step_length),
                            "commanded_step_height_m": float(step_height),
                        }
    if episode_index != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            "Formal loop generated {} cases instead of {}".format(
                episode_index, EXPECTED_CASE_COUNT
            )
        )
    return generated


def validate_generated_cases(generated, references):
    mismatches = []
    for reference in references:
        episode_index = int(reference["episode_index"])
        case = generated[episode_index]
        exact_fields = {
            "seed": int(reference["seed"]),
            "walk_direction": float(reference["walk_direction"]),
            "reset_phase": reference["reset_phase"],
            "commanded_step_length_m": float(reference["commanded_step_length_m"]),
            "commanded_step_height_m": float(reference["commanded_step_height_m"]),
        }
        for field, expected in exact_fields.items():
            if case[field] != expected:
                mismatches.append(
                    {
                        "episode_index": episode_index,
                        "field": field,
                        "expected": expected,
                        "observed": case[field],
                    }
                )
        state = case["state"]
        q_deg = np.rad2deg(state[0::2])
        dq = state[1::2]
        for index in range(4):
            expected_q = float(reference["initial_q{}_deg".format(index + 1)])
            expected_dq = float(reference["initial_dq{}_rad_s".format(index + 1)])
            if not math.isclose(
                float(q_deg[index]), expected_q, rel_tol=0.0, abs_tol=1e-12
            ):
                mismatches.append(
                    {
                        "episode_index": episode_index,
                        "field": "initial_q{}_deg".format(index + 1),
                        "expected": expected_q,
                        "observed": float(q_deg[index]),
                    }
                )
            if float(dq[index]) != expected_dq:
                mismatches.append(
                    {
                        "episode_index": episode_index,
                        "field": "initial_dq{}_rad_s".format(index + 1),
                        "expected": expected_dq,
                        "observed": float(dq[index]),
                    }
                )
    return mismatches


def sign_symbol(sign):
    return "+" if sign > 0 else "-"


def run_length_encode(signs):
    if not signs:
        return ""
    chunks = []
    current = signs[0]
    count = 1
    for sign in signs[1:]:
        if sign == current:
            count += 1
        else:
            chunks.append("{}x{}".format(sign_symbol(current), count))
            current = sign
            count = 1
    chunks.append("{}x{}".format(sign_symbol(current), count))
    return "|".join(chunks)


def query_selector(formal, selector, state, walk_direction, step_length, step_height):
    state_net = formal.normalize_state(
        state,
        0.0,
        walk_direction,
        step_length,
        step_height,
    )
    with torch.no_grad():
        tensor = torch.as_tensor(
            state_net, dtype=torch.float32, device=formal.DEVICE
        ).reshape(1, formal.state_dim)
        logits = selector(tensor)[0]
        probabilities = torch.softmax(logits, dim=0)
    label = int(torch.argmax(probabilities).item())
    if label == 0:
        return 1.0, float(probabilities[0].item())
    return -1.0, float(probabilities[1].item())


def rollout_variant(
    formal,
    selector,
    positive_expert,
    negative_expert,
    reference,
    initial_state,
    schedule,
):
    if schedule not in {"hold", "requery"}:
        raise ValueError("Unsupported selector schedule: {}".format(schedule))

    initial_state = np.asarray(initial_state, dtype=float).copy()
    walk_direction = float(reference["walk_direction"])
    reset_phase = reference["reset_phase"]
    step_length = float(reference["commanded_step_length_m"])
    step_height = float(reference["commanded_step_height_m"])
    seed = int(reference["seed"])
    episode_index = int(reference["episode_index"])

    initial_sign, initial_confidence = query_selector(
        formal, selector, initial_state, walk_direction, step_length, step_height
    )
    current_sign = initial_sign
    env = formal.make_env("passive", hip_sign=current_sign)
    env.define(
        *initial_state,
        hip_direction=current_sign,
        walk_direction=walk_direction,
        commanded_step_length=step_length,
        commanded_step_height=step_height,
        reset_phase=reset_phase,
    )

    initial_q = env.state[0::2].copy()
    initial_com = formal.center_of_mass(initial_q)
    joint_energy = np.zeros(3, dtype=float)
    nonzero_torque_steps = 0
    signs = []
    confidences = []
    done = False
    steps = 0

    while not done and steps < formal.MAX_STEPS:
        if schedule == "requery" or steps == 0:
            current_sign, confidence = query_selector(
                formal,
                selector,
                env.state,
                walk_direction,
                step_length,
                step_height,
            )
        else:
            confidence = initial_confidence

        # The selected expert and the expert's sign-conditioning coordinate are
        # changed together.  No other environment or evaluator setting changes.
        env.hip_sign = current_sign
        env.hip_direction = current_sign
        policy = positive_expert if current_sign > 0 else negative_expert
        state_net = formal.normalize_state(
            env.state,
            env.hip_direction,
            env.walk_direction,
            env.commanded_step_length,
            env.commanded_step_height,
        )
        action_index, _ = formal.choose_action(policy, state_net)
        torques = env.action_to_torques(action_index)
        previous_state = env.state.copy()
        previous_rates = formal.projected_motor_rates(env, previous_state)
        next_state, _, done = env.step(action_index)
        next_rates = formal.projected_motor_rates(env, next_state)
        motor_rates = 0.5 * (previous_rates + next_rates)
        motor_power = torques * motor_rates
        joint_energy += np.maximum(motor_power, 0.0) * env.dt
        if np.any(np.abs(torques) > 1e-12):
            nonzero_torque_steps += 1
        signs.append(current_sign)
        confidences.append(confidence)
        steps += 1

    if not done:
        env.over = True
        env.terminal_reason = "timeout"

    final_q = env.state[0::2].copy()
    final_com = formal.center_of_mass(final_q)
    signed_displacement = float(
        walk_direction * (final_com[0] - initial_com[0])
    )
    success = str(env.terminal_reason).startswith("success")
    torque_valid = (
        (not formal.REQUIRE_NONZERO_TORQUE_FOR_CMT)
        or nonzero_torque_steps >= formal.MIN_NONZERO_TORQUE_STEPS
    )
    displacement_valid = (
        (not formal.REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT)
        or signed_displacement > formal.MIN_COM_DISPLACEMENT_M
    )
    valid = success and torque_valid and displacement_valid
    denominator = (
        formal.total_mass()
        * formal.g
        * max(signed_displacement, formal.CMT_EPS_DISPLACEMENT_M)
    )
    energy = float(np.sum(joint_energy))
    cmt = float(energy / denominator) if valid else float("nan")
    joint_cmt = (
        joint_energy / denominator
        if valid
        else np.full(3, np.nan, dtype=float)
    )
    switch_count = sum(
        previous != current for previous, current in zip(signs, signs[1:])
    )

    result = {
        "eval_label": "V22_3_V9_frozen_hold_vs_requery_20260725",
        "schedule": schedule,
        "seed": seed,
        "episode_index": episode_index,
        "walk_direction": walk_direction,
        "reset_phase": reset_phase,
        "commanded_step_length_m": step_length,
        "commanded_step_height_m": step_height,
        "success": bool(success),
        "valid_for_cmt": bool(valid),
        "terminal_reason": str(env.terminal_reason),
        "steps": int(steps),
        "nonzero_torque_steps": int(nonzero_torque_steps),
        "initial_selector_sign": sign_symbol(initial_sign),
        "initial_selector_confidence": float(initial_confidence),
        "positive_sign_steps": int(sum(sign > 0 for sign in signs)),
        "negative_sign_steps": int(sum(sign < 0 for sign in signs)),
        "sign_switch_count": int(switch_count),
        "sign_sequence": "".join(sign_symbol(sign) for sign in signs),
        "sign_sequence_rle": run_length_encode(signs),
        "selector_confidence_min": float(min(confidences)) if confidences else float("nan"),
        "selector_confidence_mean": float(statistics.fmean(confidences)) if confidences else float("nan"),
        "selector_confidence_max": float(max(confidences)) if confidences else float("nan"),
        "energy_j": energy,
        "support_knee_energy_j": float(joint_energy[0]),
        "hip_energy_j": float(joint_energy[1]),
        "swing_knee_energy_j": float(joint_energy[2]),
        "signed_com_displacement_m": signed_displacement,
        "cmt": cmt,
        "support_knee_cmt": float(joint_cmt[0]),
        "hip_cmt": float(joint_cmt[1]),
        "swing_knee_cmt": float(joint_cmt[2]),
        "initial_q1_deg": float(np.rad2deg(initial_q[0])),
        "initial_q2_deg": float(np.rad2deg(initial_q[1])),
        "initial_q3_deg": float(np.rad2deg(initial_q[2])),
        "initial_q4_deg": float(np.rad2deg(initial_q[3])),
        "initial_dq1_rad_s": float(initial_state[1]),
        "initial_dq2_rad_s": float(initial_state[3]),
        "initial_dq3_rad_s": float(initial_state[5]),
        "initial_dq4_rad_s": float(initial_state[7]),
        "final_q1_deg": float(np.rad2deg(final_q[0])),
        "final_q2_deg": float(np.rad2deg(final_q[1])),
        "final_q3_deg": float(np.rad2deg(final_q[2])),
        "final_q4_deg": float(np.rad2deg(final_q[3])),
    }
    return result


def close_enough(
    observed,
    expected,
    absolute_tolerance=NUMERIC_ABS_TOL,
    relative_tolerance=NUMERIC_REL_TOL,
):
    if math.isnan(observed) and math.isnan(expected):
        return True
    if not (math.isfinite(observed) and math.isfinite(expected)):
        return observed == expected
    return math.isclose(
        observed,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def compare_hold_to_reference(hold, reference):
    mismatches = []
    exact_fields = {
        "success": parse_bool(reference["success"]),
        "valid_for_cmt": parse_bool(reference["valid_for_cmt"]),
        "terminal_reason": reference["terminal_reason"],
        "steps": int(reference["steps"]),
        "nonzero_torque_steps": int(reference["nonzero_torque_steps"]),
    }
    for field, expected in exact_fields.items():
        if hold[field] != expected:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "observed": hold[field],
                    "difference": "",
                }
            )

    numeric_fields = [
        "energy_j",
        "support_knee_energy_j",
        "hip_energy_j",
        "swing_knee_energy_j",
        "signed_com_displacement_m",
        "cmt",
        "support_knee_cmt",
        "hip_cmt",
        "swing_knee_cmt",
        "final_q1_deg",
        "final_q2_deg",
        "final_q3_deg",
        "final_q4_deg",
    ]
    for field in numeric_fields:
        observed = float(hold[field])
        expected = finite_float(reference[field])
        if field.startswith("final_q"):
            absolute_tolerance = FINAL_ANGLE_ABS_TOL_DEG
            relative_tolerance = NUMERIC_REL_TOL
        elif field.endswith("energy_j") or field == "signed_com_displacement_m":
            absolute_tolerance = ENERGY_DISPLACEMENT_ABS_TOL
            relative_tolerance = ENERGY_DISPLACEMENT_REL_TOL
        elif field == "cmt" or field.endswith("_cmt"):
            absolute_tolerance = CMT_ABS_TOL
            relative_tolerance = CMT_REL_TOL
        else:
            absolute_tolerance = NUMERIC_ABS_TOL
            relative_tolerance = NUMERIC_REL_TOL
        if not close_enough(
            observed,
            expected,
            absolute_tolerance,
            relative_tolerance,
        ):
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "difference": observed - expected,
                }
            )
    return mismatches


def bootstrap_mean_difference(differences):
    if not differences:
        return [float("nan"), float("nan")]
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    means = [
        statistics.fmean(rng.choices(differences, k=n))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    means.sort()

    def quantile(probability):
        position = (len(means) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return means[lower]
        fraction = position - lower
        return means[lower] * (1.0 - fraction) + means[upper] * fraction

    return [quantile(0.025), quantile(0.975)]


def summarize(hold_rows, requery_rows):
    by_case = {
        int(row["episode_index"]): row for row in requery_rows
    }
    pairs = [
        (hold, by_case[int(hold["episode_index"])])
        for hold in hold_rows
    ]
    both_valid = [
        (hold, requery)
        for hold, requery in pairs
        if hold["valid_for_cmt"] and requery["valid_for_cmt"]
    ]
    differences = [
        requery["cmt"] - hold["cmt"] for hold, requery in both_valid
    ]
    requery_lower = sum(
        requery["cmt"] < hold["cmt"] for hold, requery in both_valid
    )
    hold_lower = sum(
        hold["cmt"] < requery["cmt"] for hold, requery in both_valid
    )
    ties = len(both_valid) - requery_lower - hold_lower
    switched = [
        row for row in requery_rows if row["sign_switch_count"] > 0
    ]
    return {
        "case_count": len(pairs),
        "hold_success_count": sum(row["success"] for row in hold_rows),
        "requery_success_count": sum(row["success"] for row in requery_rows),
        "both_success_count": sum(
            hold["success"] and requery["success"] for hold, requery in pairs
        ),
        "hold_only_success_count": sum(
            hold["success"] and not requery["success"] for hold, requery in pairs
        ),
        "requery_only_success_count": sum(
            not hold["success"] and requery["success"] for hold, requery in pairs
        ),
        "neither_success_count": sum(
            not hold["success"] and not requery["success"] for hold, requery in pairs
        ),
        "hold_valid_cmt_count": sum(row["valid_for_cmt"] for row in hold_rows),
        "requery_valid_cmt_count": sum(row["valid_for_cmt"] for row in requery_rows),
        "both_valid_cmt_count": len(both_valid),
        "hold_mean_cmt_all_valid": statistics.fmean(
            row["cmt"] for row in hold_rows if row["valid_for_cmt"]
        ) if any(row["valid_for_cmt"] for row in hold_rows) else float("nan"),
        "requery_mean_cmt_all_valid": statistics.fmean(
            row["cmt"] for row in requery_rows if row["valid_for_cmt"]
        ) if any(row["valid_for_cmt"] for row in requery_rows) else float("nan"),
        "hold_mean_cmt_both_valid": statistics.fmean(
            hold["cmt"] for hold, _ in both_valid
        ) if both_valid else float("nan"),
        "requery_mean_cmt_both_valid": statistics.fmean(
            requery["cmt"] for _, requery in both_valid
        ) if both_valid else float("nan"),
        "mean_requery_minus_hold_cmt_both_valid": statistics.fmean(differences)
        if differences else float("nan"),
        "bootstrap_ci95_mean_requery_minus_hold_cmt": bootstrap_mean_difference(
            differences
        ),
        "requery_lower_cmt_count": requery_lower,
        "hold_lower_cmt_count": hold_lower,
        "cmt_tie_count": ties,
        "requery_lower_cmt_fraction_both_valid": requery_lower / len(both_valid)
        if both_valid else float("nan"),
        "requery_sign_switch_case_count": len(switched),
        "requery_no_switch_case_count": len(requery_rows) - len(switched),
        "requery_total_sign_switches": sum(
            row["sign_switch_count"] for row in requery_rows
        ),
        "requery_mean_sign_switches_per_case": statistics.fmean(
            row["sign_switch_count"] for row in requery_rows
        ),
        "requery_mean_sign_switches_among_switched_cases": statistics.fmean(
            row["sign_switch_count"] for row in switched
        ) if switched else 0.0,
        "hold_terminal_reason_counts": dict(
            Counter(row["terminal_reason"] for row in hold_rows)
        ),
        "requery_terminal_reason_counts": dict(
            Counter(row["terminal_reason"] for row in requery_rows)
        ),
    }


def paired_rows(hold_rows, requery_rows):
    requery_by_case = {
        int(row["episode_index"]): row for row in requery_rows
    }
    rows = []
    for hold in hold_rows:
        requery = requery_by_case[int(hold["episode_index"])]
        both_valid = hold["valid_for_cmt"] and requery["valid_for_cmt"]
        rows.append(
            {
                "seed": hold["seed"],
                "episode_index": hold["episode_index"],
                "walk_direction": hold["walk_direction"],
                "reset_phase": hold["reset_phase"],
                "commanded_step_length_m": hold["commanded_step_length_m"],
                "commanded_step_height_m": hold["commanded_step_height_m"],
                "hold_success": hold["success"],
                "requery_success": requery["success"],
                "hold_valid_for_cmt": hold["valid_for_cmt"],
                "requery_valid_for_cmt": requery["valid_for_cmt"],
                "both_valid_for_cmt": both_valid,
                "hold_cmt": hold["cmt"],
                "requery_cmt": requery["cmt"],
                "requery_minus_hold_cmt": (
                    requery["cmt"] - hold["cmt"]
                    if both_valid else float("nan")
                ),
                "requery_lower_cmt": (
                    requery["cmt"] < hold["cmt"] if both_valid else ""
                ),
                "hold_energy_j": hold["energy_j"],
                "requery_energy_j": requery["energy_j"],
                "hold_displacement_m": hold["signed_com_displacement_m"],
                "requery_displacement_m": requery["signed_com_displacement_m"],
                "hold_terminal_reason": hold["terminal_reason"],
                "requery_terminal_reason": requery["terminal_reason"],
                "hold_sign_switch_count": hold["sign_switch_count"],
                "requery_sign_switch_count": requery["sign_switch_count"],
                "hold_sign_sequence_rle": hold["sign_sequence_rle"],
                "requery_sign_sequence_rle": requery["sign_sequence_rle"],
            }
        )
    return rows


def prepare_output(directory):
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Refusing to overwrite non-empty output directory: {}".format(directory))
    directory.mkdir(parents=True, exist_ok=True)


def write_stop_report(stage, mismatches, input_hashes):
    report_path = ROOT / "STOP_REPORT.md"
    lines = [
        "# STOP REPORT",
        "",
        "The hold implementation did not reproduce the released formal selector.",
        "The full 2,160-case hold-versus-requery evaluation was therefore not authorized.",
        "",
        "- Stage: `{}`".format(stage),
        "- Mismatch count: `{}`".format(len(mismatches)),
        "- Numeric tolerances: absolute `{}`, relative `{}`".format(
            NUMERIC_ABS_TOL, NUMERIC_REL_TOL
        ),
        "",
        "See the stage `hold_reference_mismatches.csv` for case-level details.",
        "",
        "## Verified input hashes",
        "",
    ]
    for relative, digest in sorted(input_hashes.items()):
        lines.append("- `{}`: `{}`".format(relative, digest))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_stage(stage):
    input_hashes = validate_inputs()
    formal = load_formal_module()
    formal.DETERMINISTIC_POLICY = True
    formal.USE_CUDA_IF_AVAILABLE = bool(torch.cuda.is_available())
    torch.set_grad_enabled(False)

    selector = formal.load_sign_selector(
        SELECTOR_PATH, formal.PASSIVE_SIGN_SELECTOR_NAME
    )
    positive = formal.load_policy(
        POSITIVE_EXPERT_PATH, len(formal.PASSIVE_ACTIONS), formal.PASSIVE_POS_POLICY_NAME
    )
    negative = formal.load_policy(
        NEGATIVE_EXPERT_PATH, len(formal.PASSIVE_ACTIONS), formal.PASSIVE_NEG_POLICY_NAME
    )
    references = selector_reference_rows()
    generated_cases = generate_formal_initial_states(formal)
    generated_case_mismatches = validate_generated_cases(generated_cases, references)
    if generated_case_mismatches:
        mismatch_path = ROOT / "generated_case_mismatches.json"
        mismatch_path.write_text(
            json.dumps(generated_case_mismatches, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            "Regenerated formal initial cases do not match the archive; see {}".format(
                mismatch_path
            )
        )
    if stage == "smoke":
        selected = {
            episode for episode in SMOKE_EPISODE_INDICES
        }
        references = [
            row for row in references
            if int(row["episode_index"]) in selected
        ]
        output = ROOT / "outputs" / "smoke"
    else:
        smoke_validation = ROOT / "outputs" / "smoke" / "hold_validation.json"
        if not smoke_validation.exists():
            raise RuntimeError("Smoke validation is missing; run --stage smoke first")
        smoke_status = json.loads(smoke_validation.read_text(encoding="utf-8"))
        if smoke_status.get("status") != "PASS":
            raise RuntimeError("Smoke validation did not pass; full evaluation is blocked")
        output = ROOT / "outputs" / "formal_2160"
    prepare_output(output)

    started = time.time()
    hold_rows = []
    requery_rows = []
    mismatch_rows = []
    for position, reference in enumerate(references, start=1):
        hold = rollout_variant(
            formal,
            selector,
            positive,
            negative,
            reference,
            generated_cases[int(reference["episode_index"])]["state"],
            "hold",
        )
        mismatches = compare_hold_to_reference(hold, reference)
        for mismatch in mismatches:
            mismatch_rows.append(
                {
                    "seed": hold["seed"],
                    "episode_index": hold["episode_index"],
                    **mismatch,
                }
            )
        hold_rows.append(hold)
        if mismatches:
            continue
        requery_rows.append(
            rollout_variant(
                formal,
                selector,
                positive,
                negative,
                reference,
                generated_cases[int(reference["episode_index"])]["state"],
                "requery",
            )
        )
        if position % 25 == 0 or position == len(references):
            elapsed = time.time() - started
            print(
                "[{}] {}/{} cases; {:.1f} s elapsed".format(
                    stage, position, len(references), elapsed
                ),
                flush=True,
            )

    write_csv(output / "hold_rollouts.csv", hold_rows)
    write_csv(output / "hold_reference_mismatches.csv", mismatch_rows)
    validation = {
        "stage": stage,
        "status": "PASS" if not mismatch_rows else "FAIL",
        "case_count": len(references),
        "hold_rows": len(hold_rows),
        "requery_rows": len(requery_rows),
        "mismatch_count": len(mismatch_rows),
        "numeric_abs_tolerance": NUMERIC_ABS_TOL,
        "numeric_rel_tolerance": NUMERIC_REL_TOL,
        "energy_displacement_abs_tolerance": ENERGY_DISPLACEMENT_ABS_TOL,
        "energy_displacement_rel_tolerance": ENERGY_DISPLACEMENT_REL_TOL,
        "cmt_abs_tolerance": CMT_ABS_TOL,
        "cmt_rel_tolerance": CMT_REL_TOL,
        "final_angle_abs_tolerance_deg": FINAL_ANGLE_ABS_TOL_DEG,
        "formal_evaluator_device": str(formal.DEVICE),
        "input_sha256": input_hashes,
        "elapsed_seconds": time.time() - started,
    }
    (output / "hold_validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=True), encoding="utf-8"
    )
    if mismatch_rows:
        stop_path = write_stop_report(stage, mismatch_rows, input_hashes)
        print("HOLD REPRODUCTION FAILED; wrote {}".format(stop_path), flush=True)
        return 2

    write_csv(output / "requery_rollouts.csv", requery_rows)
    write_csv(output / "hold_vs_requery_trials.csv", paired_rows(hold_rows, requery_rows))
    sign_rows = []
    for row in hold_rows + requery_rows:
        sign_rows.append(
            {
                "seed": row["seed"],
                "episode_index": row["episode_index"],
                "schedule": row["schedule"],
                "steps": row["steps"],
                "initial_selector_sign": row["initial_selector_sign"],
                "initial_selector_confidence": row["initial_selector_confidence"],
                "sign_switch_count": row["sign_switch_count"],
                "positive_sign_steps": row["positive_sign_steps"],
                "negative_sign_steps": row["negative_sign_steps"],
                "sign_sequence": row["sign_sequence"],
                "sign_sequence_rle": row["sign_sequence_rle"],
            }
        )
    write_csv(output / "sign_sequences.csv", sign_rows)
    summary = summarize(hold_rows, requery_rows)
    summary.update(
        {
            "stage": stage,
            "comparison": "frozen selector/expert hold versus per-control-step requery",
            "all_non_query_factors_shared": True,
            "formal_hold_reproduction": "PASS",
            "input_sha256": input_hashes,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "elapsed_seconds": time.time() - started,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    flat_summary = [
        {"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    write_csv(output / "summary.csv", flat_summary)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    raise SystemExit(run_stage(args.stage))


if __name__ == "__main__":
    main()
