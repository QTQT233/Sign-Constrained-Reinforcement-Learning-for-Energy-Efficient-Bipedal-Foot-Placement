#!/usr/bin/env python3
"""Assert that controller evaluations use identical four-link case manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


KEY_FIELDS = ["seed", "episode_index"]
STATE_FIELDS = [
    "walk_direction",
    "reset_phase",
    "commanded_step_length_m",
    "commanded_step_height_m",
    "initial_q1_deg",
    "initial_q2_deg",
    "initial_q3_deg",
    "initial_q4_deg",
    "initial_dq1_rad_s",
    "initial_dq2_rad_s",
    "initial_dq3_rad_s",
    "initial_dq4_rad_s",
]


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def initial_state_hash(row: dict[str, str]) -> str:
    payload = "\x1f".join(row[field] for field in STATE_FIELDS).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exclusion_reason(row: dict[str, str]) -> str:
    if not as_bool(row["success"]):
        return "task_failure"
    if int(row["nonzero_torque_steps"]) <= 0:
        return "zero_torque"
    if float(row["signed_com_displacement_m"]) <= 0.001:
        return "insufficient_forward_displacement"
    return "eligible"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=Path(
            "data/four_link/true_action_mask_scratch_c090_epoch1275/"
            "rollouts_active_vs_passive.csv"
        ),
    )
    parser.add_argument("--reference", default="active")
    parser.add_argument("--comparison", default="passive_sign_selector")
    parser.add_argument("--expected", type=int, default=2160)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/four_link_statistics/true_action_mask_scratch_c090_epoch1275/"
            "pairing_validation.json"
        ),
    )
    args = parser.parse_args()

    with args.rollouts.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for controller in (args.reference, args.comparison):
        controller_rows = [row for row in rows if row["controller"] == controller]
        index: dict[tuple[str, str], dict[str, str]] = {}
        for row in controller_rows:
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in index:
                raise RuntimeError(f"duplicate {controller} key: {key}")
            index[key] = row
        if len(index) != args.expected:
            raise RuntimeError(f"{controller}: expected {args.expected} cases, found {len(index)}")
        selected[controller] = index

    reference = selected[args.reference]
    comparison = selected[args.comparison]
    if set(reference) != set(comparison):
        only_reference = sorted(set(reference) - set(comparison))[:10]
        only_comparison = sorted(set(comparison) - set(reference))[:10]
        raise RuntimeError(
            f"case-key mismatch; only reference={only_reference}; only comparison={only_comparison}"
        )

    mismatches = Counter()
    hash_mismatches = 0
    for key in reference:
        left, right = reference[key], comparison[key]
        for field in STATE_FIELDS:
            if left[field] != right[field]:
                mismatches[field] += 1
        if initial_state_hash(left) != initial_state_hash(right):
            hash_mismatches += 1
    if mismatches or hash_mismatches:
        raise RuntimeError(f"initial-state mismatches={dict(mismatches)}, hash mismatches={hash_mismatches}")

    keys = sorted(reference)
    left_success = sum(as_bool(reference[key]["success"]) for key in keys)
    right_success = sum(as_bool(comparison[key]["success"]) for key in keys)
    both_success = sum(
        as_bool(reference[key]["success"]) and as_bool(comparison[key]["success"]) for key in keys
    )
    left_only = sum(
        as_bool(reference[key]["success"]) and not as_bool(comparison[key]["success"]) for key in keys
    )
    right_only = sum(
        not as_bool(reference[key]["success"]) and as_bool(comparison[key]["success"]) for key in keys
    )
    both_valid = sum(
        as_bool(reference[key]["valid_for_cmt"]) and as_bool(comparison[key]["valid_for_cmt"])
        for key in keys
    )
    result = {
        "rollouts": args.rollouts.as_posix(),
        "reference": args.reference,
        "comparison": args.comparison,
        "expected_cases_per_controller": args.expected,
        "unique_cases_per_controller": {name: len(index) for name, index in selected.items()},
        "case_key_sets_equal": True,
        "state_fields_checked": STATE_FIELDS,
        "state_field_mismatches": dict(mismatches),
        "initial_state_hash_mismatches": hash_mismatches,
        "success": {
            args.reference: left_success,
            args.comparison: right_success,
            "both_success": both_success,
            f"{args.reference}_only": left_only,
            f"{args.comparison}_only": right_only,
            "both_failure": args.expected - both_success - left_only - right_only,
        },
        "both_valid_for_cmt": both_valid,
        "cmt_exclusion_reasons": {
            args.reference: dict(Counter(exclusion_reason(row) for row in reference.values())),
            args.comparison: dict(Counter(exclusion_reason(row) for row in comparison.values())),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
