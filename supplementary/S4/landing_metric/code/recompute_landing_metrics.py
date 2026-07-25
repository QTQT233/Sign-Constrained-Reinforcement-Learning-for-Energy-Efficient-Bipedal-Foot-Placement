"""Verify and aggregate the released two-link landing-state records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from foot_geometry import foot_x, foot_y


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSITIONS = COMPONENT_ROOT / "inputs" / "terminal_state_records.csv"
DEFAULT_CASES = COMPONENT_ROOT / "results" / "case_metrics.csv"
DEFAULT_CONTROLLERS = COMPONENT_ROOT / "results" / "controller_summary.csv"
DEFAULT_OUTPUT = COMPONENT_ROOT / "_generated" / "validation.json"
L1_M = 0.521
L2_M = 0.481
ABS_TOL = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=ABS_TOL):
        raise RuntimeError(
            f"{label}: observed {observed:.17g}, expected {expected:.17g}"
        )


def verify(
    transition_path: Path,
    case_path: Path,
    controller_path: Path,
) -> dict[str, object]:
    transitions = read_csv(transition_path)
    cases = read_csv(case_path)
    controllers = read_csv(controller_path)
    if len(transitions) != 144 or len(cases) != 48 or len(controllers) != 4:
        raise RuntimeError(
            "Expected 144 transitions, 48 controller-case rows, and "
            f"4 controller rows; found {len(transitions)}, {len(cases)}, "
            f"and {len(controllers)}"
        )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    controller_abs: dict[str, list[float]] = defaultdict(list)
    max_field_error = 0.0
    for row in transitions:
        q1 = float(row["terminal_theta1_rad"])
        q2 = float(row["terminal_theta2_rad"])
        target_q1 = float(row["target_theta1_rad"])
        target_q2 = float(row["target_theta2_rad"])
        actual_x = foot_x(q1, q2, L1_M, L2_M)
        target_x = foot_x(target_q1, target_q2, L1_M, L2_M)
        residual = actual_x - target_x
        expected_fields = {
            "corrected_foot_x_m": actual_x,
            "corrected_target_foot_x_m": target_x,
            "landing_residual_actual_minus_target_m": residual,
            "landing_residual_abs_m": abs(residual),
            "corrected_foot_y_m": foot_y(q1, q2, L1_M, L2_M),
        }
        for field, expected in expected_fields.items():
            observed = float(row[field])
            max_field_error = max(max_field_error, abs(observed - expected))
            assert_close(observed, expected, f"{row['case_id']}/{field}")
        key = (row["case_id"], row["controller"])
        grouped[key].append(residual)
        controller_abs[row["controller"]].append(abs(residual))

    case_lookup = {(row["case_id"], row["controller"]): row for row in cases}
    if set(grouped) != set(case_lookup):
        raise RuntimeError("Transition and case-summary key sets differ")
    for key, residuals in grouped.items():
        if len(residuals) != 3:
            raise RuntimeError(f"{key}: expected three transitions")
        row = case_lookup[key]
        signed = sum(residuals)
        mae = fmean(abs(value) for value in residuals)
        rmse = math.sqrt(fmean(value * value for value in residuals))
        assert_close(
            float(row["corrected_accumulated_signed_residual_m"]),
            signed,
            f"{key}/signed residual",
        )
        assert_close(
            float(row["corrected_accumulated_abs_residual_m"]),
            abs(signed),
            f"{key}/absolute accumulated residual",
        )
        assert_close(
            float(row["corrected_per_transition_mae_m"]),
            mae,
            f"{key}/MAE",
        )
        assert_close(
            float(row["corrected_per_transition_rmse_m"]),
            rmse,
            f"{key}/RMSE",
        )

    summary_lookup = {row["controller"]: row for row in controllers}
    if set(summary_lookup) != set(controller_abs):
        raise RuntimeError("Transition and controller-summary sets differ")
    pooled_mae: dict[str, float] = {}
    for controller, values in sorted(controller_abs.items()):
        pooled_mae[controller] = fmean(values)
        assert_close(
            float(summary_lookup[controller]["pooled_per_transition_mae_m"]),
            pooled_mae[controller],
            f"{controller}/pooled MAE",
        )

    return {
        "schema": "S4-landing-metric-verification-v1",
        "all_checks_passed": True,
        "formula": {
            "actual_foot_x": "l1*cos(q1)+l2*sin(q2)",
            "target_foot_x": "l1*cos(q1_target)+l2*sin(q2_target)",
            "primary_summary": "mean(abs(actual_foot_x-target_foot_x))",
        },
        "parameters": {"l1_m": L1_M, "l2_m": L2_M},
        "counts": {
            "transitions": len(transitions),
            "controller_case_rows": len(cases),
            "controllers": len(controllers),
        },
        "pooled_per_transition_mae_m": pooled_mae,
        "max_recomputed_field_error_m": max_field_error,
        "reporting_only_invariants": {
            "controller_actions_reexecuted": False,
            "success_changed": False,
            "energy_changed": False,
            "com_displacement_changed": False,
            "cmt_changed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--controllers", type=Path, default=DEFAULT_CONTROLLERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(args.transitions, args.cases, args.controllers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
