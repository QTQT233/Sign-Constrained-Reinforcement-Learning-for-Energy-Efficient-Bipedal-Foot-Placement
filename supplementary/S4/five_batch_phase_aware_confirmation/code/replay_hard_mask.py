"""Replay the frozen true hard-mask baseline on confirmation initial states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
EVALUATOR_PATH = (
    REPO
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_three_expert_selector_evaluation.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_evaluator():
    spec = importlib.util.spec_from_file_location(
        "hard_mask_confirmation_replay_evaluator", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_float(value: str) -> float:
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--route-bank-dir", type=Path, required=True)
    args = parser.parse_args()
    input_csv = args.route_bank_dir / "rollouts_active_vs_passive.csv"
    output_csv = args.route_bank_dir / "hard_mask_replay.csv"
    manifest_path = args.route_bank_dir / "hard_mask_replay_manifest.json"
    if output_csv.exists() or manifest_path.exists():
        raise RuntimeError(f"Refusing to overwrite hard-mask replay in {args.route_bank_dir}")

    active_rows = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["controller"] == "active":
                active_rows.append(row)
    active_rows.sort(key=lambda row: int(row["episode_index"]))
    if len(active_rows) != 720:
        raise RuntimeError(f"Expected 720 Active states, found {len(active_rows)}")
    if {int(row["seed"]) for row in active_rows} != {args.seed}:
        raise RuntimeError("Seed mismatch in route bank")

    evaluator = import_evaluator()
    evaluator.DETERMINISTIC_POLICY = True
    evaluator.update_curriculum(0, progress_override=1.0)
    policy = evaluator.load_true_action_mask_policy(
        evaluator.TRUE_ACTION_MASK_POLICY_PATH,
        evaluator.TRUE_ACTION_MASK_POLICY_NAME,
    )
    records = []
    for row in active_rows:
        initial_state = np.asarray(
            [
                np.deg2rad(parse_float(row["initial_q1_deg"])),
                parse_float(row["initial_dq1_rad_s"]),
                np.deg2rad(parse_float(row["initial_q2_deg"])),
                parse_float(row["initial_dq2_rad_s"]),
                np.deg2rad(parse_float(row["initial_q3_deg"])),
                parse_float(row["initial_dq3_rad_s"]),
                np.deg2rad(parse_float(row["initial_q4_deg"])),
                parse_float(row["initial_dq4_rad_s"]),
            ],
            dtype=float,
        )
        result = evaluator.rollout_policy(
            initial_state,
            parse_float(row["walk_direction"]),
            str(row["reset_phase"]),
            parse_float(row["commanded_step_length_m"]),
            parse_float(row["commanded_step_height_m"]),
            policy,
            "true_action_mask",
            evaluator.TRUE_ACTION_MASK_POLICY_NAME,
            int(row["seed"]),
            int(row["episode_index"]),
            controller_label="true_action_mask_scratch",
        )
        records.append(asdict(result))
    if len(records) != 720:
        raise RuntimeError("Incomplete hard-mask replay")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema": "hard-mask-confirmation-replay/v1",
        "seed": args.seed,
        "n_cases": len(records),
        "source_route_bank_sha256": sha256(input_csv),
        "evaluator_sha256": sha256(EVALUATOR_PATH),
        "hard_mask_model_sha256": sha256(
            evaluator.TRUE_ACTION_MASK_POLICY_PATH
        ),
        "output_sha256": sha256(output_csv),
        "deterministic_policy": True,
        "note": (
            "The hard-mask replay uses the exact initial states, commands, "
            "directions, reset phases, seed labels, and episode indices from "
            "the frozen confirmation route bank. It does not change or tune "
            "the proposed router."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
