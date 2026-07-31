"""Reconstruct the phase-aware three-expert route on the 2,160-case diagnostic cohort.

This is an analysis-only recombination of already executed Active, non-negative,
and non-positive rollouts.  It uses the same learned three-class gate and the
same phase-specific thresholds as the frozen five-batch protocol; no simulation
is rerun.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_three_expert_selector_evaluation.py"
)
DATA = REPO_ROOT / "results" / "four_link_three_expert_primary_2160"
PROTOCOL = (
    REPO_ROOT
    / "supplementary"
    / "S4"
    / "five_batch_phase_aware_confirmation"
    / "protocol.json"
)
OUT = REPO_ROOT / "results" / "four_link_phase_aware_primary_2160"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def import_evaluator():
    spec = importlib.util.spec_from_file_location("primary_phase_gate", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DEVICE = torch.device("cpu")
    return module


def wilson(count: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = count / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def controller_summary(name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    valid = [row for row in rows if as_bool(row["valid_for_cmt"])]
    success = sum(as_bool(row["success"]) for row in rows)
    valid_count = len(valid)
    cmt = np.asarray([as_float(row["cmt"]) for row in valid], dtype=float)
    return {
        "controller": name,
        "n_all": len(rows),
        "n_success": success,
        "success_rate": success / len(rows),
        "success_ci95_low": wilson(success, len(rows))[0],
        "success_ci95_high": wilson(success, len(rows))[1],
        "n_valid_cmt": valid_count,
        "valid_cmt_rate": valid_count / len(rows),
        "valid_cmt_ci95_low": wilson(valid_count, len(rows))[0],
        "valid_cmt_ci95_high": wilson(valid_count, len(rows))[1],
        "mean_cmt_valid": float(np.mean(cmt)),
        "median_cmt_valid": float(np.median(cmt)),
        "mean_support_knee_cmt_valid": float(np.mean([as_float(row["support_knee_cmt"]) for row in valid])),
        "mean_hip_cmt_valid": float(np.mean([as_float(row["hip_cmt"]) for row in valid])),
        "mean_swing_knee_cmt_valid": float(np.mean([as_float(row["swing_knee_cmt"]) for row in valid])),
    }


def main() -> None:
    evaluator = import_evaluator()
    selector = evaluator.load_three_expert_selector(
        evaluator.THREE_EXPERT_SELECTOR_PATH,
        evaluator.THREE_EXPERT_SELECTOR_NAME,
    )
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    thresholds = protocol["proposed_rule"]["thresholds"]

    rollout_rows = read_csv(DATA / "rollouts_active_vs_passive.csv")
    candidates = read_csv(DATA / "passive_candidate_audit.csv")
    keys = lambda row: (int(row["seed"]), int(row["episode_index"]))
    source: dict[str, dict[tuple[int, int], dict[str, str]]] = {
        "active": {},
        "positive": {},
        "negative": {},
    }
    for row in rollout_rows:
        if row["controller"] == "active":
            source["active"][keys(row)] = row
    for row in candidates:
        if row["candidate_for_mode"] != "oracle_best_cmt":
            continue
        policy_name = row["policy_name"]
        if "positive" in policy_name:
            source["positive"][keys(row)] = row
        elif "negative" in policy_name:
            source["negative"][keys(row)] = row
    expected = set(source["active"])
    if len(expected) != 2160 or any(set(source[label]) != expected for label in source):
        raise RuntimeError("The three source-controller archives are not matched 2,160-case sets")

    features: list[np.ndarray] = []
    ordered_keys = sorted(expected)
    for key in ordered_keys:
        row = source["active"][key]
        state = np.asarray(
            [
                np.deg2rad(as_float(row["initial_q1_deg"])),
                as_float(row["initial_dq1_rad_s"]),
                np.deg2rad(as_float(row["initial_q2_deg"])),
                as_float(row["initial_dq2_rad_s"]),
                np.deg2rad(as_float(row["initial_q3_deg"])),
                as_float(row["initial_dq3_rad_s"]),
                np.deg2rad(as_float(row["initial_q4_deg"])),
                as_float(row["initial_dq4_rad_s"]),
            ],
            dtype=float,
        )
        features.append(
            np.asarray(
                evaluator.normalize_state(
                    state,
                    0.0,
                    as_float(row["walk_direction"]),
                    as_float(row["commanded_step_length_m"]),
                    as_float(row["commanded_step_height_m"]),
                ),
                dtype=np.float32,
            )
        )
    with torch.no_grad():
        probabilities = torch.softmax(
            selector(torch.as_tensor(np.stack(features), dtype=torch.float32)), dim=1
        ).cpu().numpy()

    selected_rows: list[dict[str, str]] = []
    route_rows: list[dict[str, object]] = []
    for key, probs in zip(ordered_keys, probabilities):
        active = source["active"][key]
        phase = active["reset_phase"]
        threshold = thresholds[phase]
        positive_ok = float(probs[0]) >= float(threshold["non_negative"])
        negative_ok = float(probs[1]) >= float(threshold["non_positive"])
        if positive_ok and not negative_ok:
            route = "positive"
        elif negative_ok and not positive_ok:
            route = "negative"
        elif positive_ok and negative_ok:
            route = "positive" if float(probs[0]) >= float(probs[1]) else "negative"
        else:
            route = "active"
        chosen = dict(source[route][key])
        selected_rows.append(chosen)
        route_rows.append(
            {
                "seed": key[0],
                "episode_index": key[1],
                "reset_phase": phase,
                "route": route,
                "p_positive": float(probs[0]),
                "p_negative": float(probs[1]),
                "p_active": float(probs[2]),
                "success": as_bool(chosen["success"]),
                "valid_for_cmt": as_bool(chosen["valid_for_cmt"]),
                "cmt": as_float(chosen["cmt"]),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    summary = controller_summary("phase_aware_three_expert", selected_rows)
    (OUT / "controller_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "route_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(route_rows[0]))
        writer.writeheader()
        writer.writerows(route_rows)

    active_rows = [source["active"][key] for key in ordered_keys]
    both = [
        (left, right)
        for left, right in zip(active_rows, selected_rows)
        if as_bool(left["valid_for_cmt"]) and as_bool(right["valid_for_cmt"])
    ]
    active_cmt = np.asarray([as_float(left["cmt"]) for left, _ in both])
    proposed_cmt = np.asarray([as_float(right["cmt"]) for _, right in both])
    lower = int(np.sum(proposed_cmt < active_cmt))
    pair = {
        "pair": "active_vs_phase_aware_three_expert",
        "both_valid_n": len(both),
        "active_mean_cmt": float(np.mean(active_cmt)),
        "proposed_mean_cmt": float(np.mean(proposed_cmt)),
        "proposed_lower_count": lower,
        "proposed_lower_fraction": lower / len(both),
        "proposed_lower_ci95": wilson(lower, len(both)),
    }
    (OUT / "active_pair_summary.json").write_text(json.dumps(pair, indent=2), encoding="utf-8")
    payload = sorted(
        path for path in OUT.iterdir()
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    with (OUT / "CHECKSUMS.sha256").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(
            "".join(f"{sha256(path)}  {path.name}\n" for path in payload)
        )
    print(json.dumps({"controller": summary, "active_pair": pair}, indent=2))


if __name__ == "__main__":
    main()
