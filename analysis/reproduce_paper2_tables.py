#!/usr/bin/env python3
"""Regenerate the current Paper2 two-link tables from released case records."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


METHODS = [
    "LIPM COM",
    "TVLQR tracking",
    "Unrestricted discrete PPO",
    "Continuous-torque PPO",
    "Continuous-torque MPC",
    "Transition-start lookup router",
]
LEGACY_METHOD_LABELS = {
    "Proposed passive selector": "Transition-start lookup router",
    "Proposed one-sided selector": "Transition-start lookup router",
    "Discrete active PPO": "Unrestricted discrete PPO",
    "Continuous active MPC": "Continuous-torque MPC",
    "Continuous active PPO": "Continuous-torque PPO",
}
GRID_METHOD_LABELS = {
    "passive": "Transition-start lookup router",
    "discrete": "Unrestricted discrete PPO",
    "continuous": "Continuous-torque PPO",
}
LANDING_METHOD_LABELS = {
    "passive": "Transition-start lookup router",
    "discrete": "Unrestricted discrete PPO",
    "continuous": "Continuous-torque PPO",
    "mpc": "Continuous-torque MPC",
}
GROUPS = [
    ("flat", "1.28", "flat_1.280"),
    ("flat", "1.145", "flat_1.145"),
    ("raised_0.01m", "1.28", "raised_0.01_1.280"),
    ("raised_0.01m", "1.145", "raised_0.01_1.145"),
]


def canonical_case_id(case_id: str) -> str:
    replacements = {
        "flat_1.145_": "flat_L1145_",
        "flat_1.28_": "flat_L1280_",
        "raised_1.145_": "raised_L1145_",
        "raised_1.28_": "raised_L1280_",
        "flat_L128_": "flat_L1280_",
        "raised_L128_": "raised_L1280_",
    }
    for prefix, canonical in replacements.items():
        if case_id.startswith(prefix):
            return canonical + case_id[len(prefix):]
    return case_id


def optional_float(value: str | None) -> float | None:
    text = (value or "").strip()
    return float(text) if text else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values: list[float]) -> dict:
    finite = [value for value in values if math.isfinite(value)]
    n = len(finite)
    if n < 2:
        return {"n": n, "mean": finite[0] if n else None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite)
    half_width = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half_width, "hi": mean + half_width}


def load_non_mpc(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = []
    for row in rows:
        method = LEGACY_METHOD_LABELS.get(row["method"], row["method"])
        if method not in METHODS or method in {"Continuous-torque MPC", "TVLQR tracking"}:
            continue
        retained.append(
            {
                "case_id": canonical_case_id(row["case_id"]),
                "terrain": row["terrain"],
                "nominal_length_m": row["nominal_length_m"],
                "method": method,
                "status": row["status"],
                "cmt": optional_float(row["cmt"]),
                "time_s": optional_float(row["time_s"]),
                "landing_transition_count": None,
                "foot_placement_mae_per_transition_m": None,
                "landing_metric_source": "",
                "source_record": row["stdout_log"],
            }
        )
    return retained


def load_grid_learned(path: Path) -> list[dict]:
    """Load the accepted fixed replays selected from the complete recovery grids."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = []
    for row in rows:
        method = GRID_METHOD_LABELS[row["controller"]]
        case_id = canonical_case_id(row["case_id"])
        retained.append(
            {
                "case_id": case_id,
                "terrain": "flat" if case_id.startswith("flat_") else "raised_0.01m",
                "nominal_length_m": "1.145" if "L1145" in case_id else "1.28",
                "method": method,
                "status": "ok",
                "cmt": optional_float(row["cmt"]),
                "time_s": optional_float(row["time_s"]),
                "landing_transition_count": None,
                "foot_placement_mae_per_transition_m": None,
                "landing_metric_source": "",
                "source_record": (
                    f"{path.as_posix()}#{row['case_id']}/{row['controller']}"
                ),
            }
        )
    return retained


def load_tvlqr(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "case_id": canonical_case_id(row["case_id"]),
            "terrain": "flat" if row["step_height_m"] == "0" else "raised_0.01m",
            "nominal_length_m": row["nominal_length_group"],
            "method": "TVLQR tracking",
            "status": row["status"],
            "cmt": optional_float(row["cmt"]),
            "time_s": None,
            "landing_transition_count": None,
            "foot_placement_mae_per_transition_m": None,
            "landing_metric_source": "",
            "source_record": row["stdout"],
        }
        for row in rows
    ]


def load_mpc(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = []
    for row in rows:
        terrain = "flat" if float(row["step_height_m"]) == 0 else "raised_0.01m"
        retained.append(
            {
                "case_id": canonical_case_id(row["case_id"]),
                "terrain": terrain,
                "nominal_length_m": row["nominal_length_group"],
                "method": "Continuous-torque MPC",
                "status": "ok",
                "cmt": optional_float(row["cmt"]),
                "time_s": optional_float(row["time_s"]),
                "landing_transition_count": None,
                "foot_placement_mae_per_transition_m": None,
                "landing_metric_source": "",
                "source_record": str(path.as_posix()),
            }
        )
    return retained


def apply_source_aligned_overrides(
    rows: list[dict], path: Path, allowed_methods: set[str] | None = None
) -> list[dict]:
    """Replace explicitly released case/method records without altering other rows."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        overrides = list(csv.DictReader(handle))
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    for override in overrides:
        if allowed_methods is not None and override["method"] not in allowed_methods:
            continue
        key = (override["case_id"], override["method"])
        if key not in by_key:
            raise RuntimeError(f"source-aligned override has no baseline row: {key}")
        baseline = by_key[key]
        baseline.update(
            {
                "status": override["status"],
                "cmt": optional_float(override["cmt"]),
                "time_s": optional_float(override["time_s"]),
                "source_record": override["source_record"],
            }
        )
    return rows


def apply_landing_metrics(rows: list[dict], path: Path) -> list[dict]:
    """Attach the corrected three-transition case MAE released in S4."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    attached: set[tuple[str, str]] = set()
    for metric in metrics:
        method = LANDING_METHOD_LABELS[metric["controller"]]
        key = (canonical_case_id(metric["case_id"]), method)
        if key not in by_key:
            raise RuntimeError(f"landing metric has no executable-controller row: {key}")
        by_key[key].update(
            {
                "landing_transition_count": 3,
                "foot_placement_mae_per_transition_m": optional_float(
                    metric["corrected_per_transition_mae_m"]
                ),
                "landing_metric_source": (
                    f"{path.as_posix()}#{metric['case_id']}/{metric['controller']}"
                ),
            }
        )
        attached.add(key)
    expected = {
        (row["case_id"], row["method"])
        for row in rows
        if row["method"] in LANDING_METHOD_LABELS.values()
    }
    if attached != expected:
        raise RuntimeError(
            "corrected landing-metric coverage differs from the 48 executable "
            f"controller-case rows: missing={sorted(expected - attached)}, "
            f"extra={sorted(attached - expected)}"
        )
    return rows


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.12f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerun",
        type=Path,
        default=Path("data/paper2/readonly_rerun/paper2_rerun_results.csv"),
    )
    parser.add_argument(
        "--mpc",
        type=Path,
        default=Path("results/paper2_mpc_current_12_cases.csv"),
    )
    parser.add_argument(
        "--tvlqr",
        type=Path,
        default=Path("results/tvlqr_seed0/tvlqr_cases.csv"),
    )
    parser.add_argument(
        "--source-aligned-overrides",
        type=Path,
        default=Path(
            "results/paper2_source_aligned_r1_rerun_20260721/rerun_records.csv"
        ),
    )
    parser.add_argument(
        "--push-off-grid",
        type=Path,
        default=Path(
            "results/paper2_push_off_grid_12case/accepted_cases.csv"
        ),
    )
    parser.add_argument(
        "--landing-metrics",
        type=Path,
        default=Path(
            "supplementary/S4/landing_metric/results/case_metrics.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper2_current"))
    parser.add_argument(
        "--primary-output",
        type=Path,
        default=Path("results/two_link_primary_12_cases.csv"),
    )
    args = parser.parse_args()

    lipm = [
        row for row in load_non_mpc(args.rerun) if row["method"] == "LIPM COM"
    ]
    lipm = apply_source_aligned_overrides(
        lipm, args.source_aligned_overrides, allowed_methods={"LIPM COM"}
    )
    rows = (
        lipm
        + load_grid_learned(args.push_off_grid)
        + load_tvlqr(args.tvlqr)
        + load_mpc(args.mpc)
    )
    rows = apply_landing_metrics(rows, args.landing_metrics)
    retained = [row for row in rows if row["status"] == "ok"]

    expected_cases = {
        f"{terrain}_L{length}_r{replicate}"
        for terrain in ("flat", "raised")
        for length in ("1145", "1280")
        for replicate in (1, 2, 3)
    }
    observed_cases = {row["case_id"] for row in retained}
    if observed_cases != expected_cases:
        raise RuntimeError(
            f"canonical case IDs differ: missing={sorted(expected_cases - observed_cases)}, "
            f"extra={sorted(observed_cases - expected_cases)}"
        )
    case_method_counts = {
        case_id: sum(row["case_id"] == case_id for row in retained)
        for case_id in expected_cases
    }
    if any(count != len(METHODS) for count in case_method_counts.values()):
        raise RuntimeError(f"expected one row per method and canonical case: {case_method_counts}")

    by_method_group: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in retained:
        by_method_group[(row["method"], row["terrain"], row["nominal_length_m"])].append(row)
        by_method[row["method"]].append(row)

    table_iv = []
    for method in METHODS:
        output = {"method": method}
        for terrain, length, label in GROUPS:
            selected = by_method_group[(method, terrain, length)]
            values = [row["cmt"] for row in selected if row["cmt"] is not None]
            if len(values) != 3:
                raise RuntimeError(f"{method} / {label}: expected 3 Cmt values, found {len(values)}")
            output[f"{label}_n"] = len(values)
            output[f"{label}_mean_cmt"] = format_number(statistics.fmean(values))
            output[f"{label}_display_3dp"] = f"{statistics.fmean(values):.3f}"
        table_iv.append(output)

    table_v = []
    for method in METHODS:
        selected = by_method[method]
        cmt = summary([row["cmt"] for row in selected if row["cmt"] is not None])
        timing = [row["time_s"] for row in selected if row["time_s"] is not None]
        landing_rows = [
            row
            for row in selected
            if row["foot_placement_mae_per_transition_m"] is not None
        ]
        landing_transition_count = sum(
            int(row["landing_transition_count"]) for row in landing_rows
        )
        pooled_landing_mae = (
            sum(
                int(row["landing_transition_count"])
                * float(row["foot_placement_mae_per_transition_m"])
                for row in landing_rows
            )
            / landing_transition_count
            if landing_transition_count
            else None
        )
        if cmt["n"] != 12:
            raise RuntimeError(f"{method}: expected 12 Cmt values, found {cmt['n']}")
        table_v.append(
            {
                "method": method,
                "n_cmt": cmt["n"],
                "mean_cmt": format_number(cmt["mean"]),
                "sample_sd": format_number(cmt["sd"]),
                "ci95_low": format_number(cmt["lo"]),
                "ci95_high": format_number(cmt["hi"]),
                "n_time": len(timing),
                "mean_time_s": format_number(statistics.fmean(timing) if timing else None),
                "n_landing_transitions": landing_transition_count,
                "pooled_foot_placement_mae_m": format_number(pooled_landing_mae),
            }
        )

    combined = []
    for row in sorted(retained, key=lambda item: (METHODS.index(item["method"]), item["case_id"])):
        combined.append({key: "" if value is None else value for key, value in row.items()})

    learned_by_case = {
        (row["case_id"], row["method"]): row
        for row in retained
        if row["method"] in GRID_METHOD_LABELS.values()
    }
    primary = []
    for case_id in sorted(expected_cases):
        proposed = learned_by_case[(case_id, "Transition-start lookup router")]
        discrete = learned_by_case[(case_id, "Unrestricted discrete PPO")]
        continuous = learned_by_case[(case_id, "Continuous-torque PPO")]
        proposed_cmt = float(proposed["cmt"])
        discrete_cmt = float(discrete["cmt"])
        continuous_cmt = float(continuous["cmt"])
        primary.append(
            {
                "case_id": case_id,
                "terrain": proposed["terrain"],
                "nominal_length_m": proposed["nominal_length_m"],
                "proposed_selector_cmt": proposed_cmt,
                "discrete_active_ppo_cmt": discrete_cmt,
                "continuous_active_ppo_cmt": continuous_cmt,
                "discrete_minus_proposed": discrete_cmt - proposed_cmt,
                "continuous_minus_proposed": continuous_cmt - proposed_cmt,
                "proposed_lower_than_discrete": proposed_cmt < discrete_cmt,
                "proposed_lower_than_continuous": proposed_cmt < continuous_cmt,
                "proposed_source_record": proposed["source_record"],
                "discrete_source_record": discrete["source_record"],
                "continuous_source_record": continuous["source_record"],
            }
        )

    write_csv(args.output_dir / "paper2_combined_cases.csv", combined)
    write_csv(args.output_dir / "table_iv_current.csv", table_iv)
    write_csv(args.output_dir / "table_v_current.csv", table_v)
    write_csv(args.primary_output, primary)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
