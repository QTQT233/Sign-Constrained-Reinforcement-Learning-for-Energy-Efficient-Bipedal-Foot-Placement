#!/usr/bin/env python3
"""Synchronize manuscript-facing landing metrics with the verified S4 record."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
S4_CASES = ROOT / "supplementary" / "S4" / "landing_metric" / "results" / "case_metrics.csv"
PUSH_CASES = ROOT / "results" / "paper2_push_off_grid_12case" / "accepted_cases.csv"
PUSH_SUMMARY = ROOT / "results" / "paper2_push_off_grid_12case" / "controller_summary.csv"
MPC_CURRENT = ROOT / "results" / "paper2_mpc_current_12_cases.csv"
MPC_ACCEPTED = ROOT / "results" / "paper2_mpc_unified_12case" / "accepted_cases.csv"
MPC_SUMMARY = ROOT / "results" / "paper2_mpc_unified_12case" / "accepted_summary.json"
SOURCE_PREFIX = "supplementary/S4/landing_metric/results/case_metrics.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(S4_CASES)
    if len(rows) != 48:
        raise RuntimeError(f"Expected 48 S4 case metrics, found {len(rows)}")
    return {(row["case_id"], row["controller"]): row for row in rows}


def synchronize_push_off(metrics: dict[tuple[str, str], dict[str, str]]) -> None:
    rows = read_csv(PUSH_CASES)
    output: list[dict[str, object]] = []
    removed = {
        "foot_displacement_m",
        "landing_error_m",
        "landing_error_abs_m",
        "landing_error_available",
        "foot_d_reporting_instrumented",
    }
    for row in rows:
        metric = metrics[(row["case_id"], row["controller"])]
        revised = {key: value for key, value in row.items() if key not in removed}
        revised.update(
            {
                "landing_transition_count": 3,
                "foot_placement_mae_per_transition_m": metric[
                    "corrected_per_transition_mae_m"
                ],
                "landing_metric_source": (
                    f"{SOURCE_PREFIX}#{row['case_id']}/{row['controller']}"
                ),
            }
        )
        output.append(revised)
    write_csv(PUSH_CASES, output)

    existing = {row["controller"]: row for row in read_csv(PUSH_SUMMARY)}
    summary: list[dict[str, object]] = []
    for controller in ("continuous", "passive", "discrete"):
        row = existing[controller]
        values = [
            float(metric["corrected_per_transition_mae_m"])
            for (case_id, name), metric in metrics.items()
            if name == controller
            for _ in range(3)
        ]
        revised = {
            key: value
            for key, value in row.items()
            if key not in {"landing_error_available_n", "mean_abs_landing_error_m"}
        }
        revised.update(
            {
                "n_landing_transitions": len(values),
                "pooled_foot_placement_mae_m": repr(fmean(values)),
            }
        )
        summary.append(revised)
    write_csv(PUSH_SUMMARY, summary)


def synchronize_mpc(metrics: dict[tuple[str, str], dict[str, str]]) -> None:
    current_rows = read_csv(MPC_CURRENT)
    current_output: list[dict[str, object]] = []
    for row in current_rows:
        source_case = (
            row["case_id"]
            .replace("_L1145_", "_1.145_")
            .replace("_L1280_", "_1.28_")
        )
        metric = metrics[(source_case, "mpc")]
        revised = {key: value for key, value in row.items() if key != "foot_error_m"}
        revised.update(
            {
                "landing_transition_count": 3,
                "foot_placement_mae_per_transition_m": metric[
                    "corrected_per_transition_mae_m"
                ],
                "landing_metric_source": f"{SOURCE_PREFIX}#{source_case}/mpc",
            }
        )
        current_output.append(revised)
    write_csv(MPC_CURRENT, current_output)

    accepted_rows = read_csv(MPC_ACCEPTED)
    accepted_output: list[dict[str, object]] = []
    transition_values: list[float] = []
    for row in accepted_rows:
        source_case = (
            row["case_id"]
            .replace("_L1145_", "_1.145_")
            .replace("_L1280_", "_1.28_")
        )
        metric = metrics[(source_case, "mpc")]
        case_mae = float(metric["corrected_per_transition_mae_m"])
        transition_values.extend([case_mae] * 3)
        revised = {
            key: value
            for key, value in row.items()
            if key not in {"foot_error_m", "absolute_foot_error_m"}
        }
        revised.update(
            {
                "landing_transition_count": 3,
                "foot_placement_mae_per_transition_m": repr(case_mae),
                "landing_metric_source": f"{SOURCE_PREFIX}#{source_case}/mpc",
            }
        )
        accepted_output.append(revised)
    write_csv(MPC_ACCEPTED, accepted_output)

    summary = json.loads(MPC_SUMMARY.read_text(encoding="utf-8"))
    summary.pop("overall_landing_error", None)
    summary["overall_foot_placement_mae"] = {
        "n_transitions": len(transition_values),
        "definition": (
            "pooled mean absolute horizontal foot-position residual at "
            "angular-target entry"
        ),
        "pooled_per_transition_mae_m": fmean(transition_values),
        "source": SOURCE_PREFIX,
    }
    MPC_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    metrics = metric_lookup()
    synchronize_push_off(metrics)
    synchronize_mpc(metrics)
    print("Synchronized learned-controller and MPC landing-metric outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
