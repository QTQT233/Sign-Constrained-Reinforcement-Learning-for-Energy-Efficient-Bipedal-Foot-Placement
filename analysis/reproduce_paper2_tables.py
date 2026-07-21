#!/usr/bin/env python3
"""Regenerate the revised Paper2 Tables IV and V from released case records."""

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
    "Discrete active PPO",
    "Continuous-torque PPO",
    "Continuous-torque MPC",
    "Proposed one-sided selector",
]
LEGACY_METHOD_LABELS = {
    "Proposed passive selector": "Proposed one-sided selector",
    "Continuous active MPC": "Continuous-torque MPC",
    "Continuous active PPO": "Continuous-torque PPO",
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
                "landing_error_m": optional_float(row["foot_error_m"]),
                "source_record": row["stdout_log"],
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
            "landing_error_m": None,
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
                "landing_error_m": optional_float(row["foot_error_m"]),
                "source_record": str(path.as_posix()),
            }
        )
    return retained


def apply_source_aligned_overrides(rows: list[dict], path: Path) -> list[dict]:
    """Replace explicitly released case/method records without altering other rows."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        overrides = list(csv.DictReader(handle))
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    for override in overrides:
        key = (override["case_id"], override["method"])
        if key not in by_key:
            raise RuntimeError(f"source-aligned override has no baseline row: {key}")
        baseline = by_key[key]
        baseline.update(
            {
                "status": override["status"],
                "cmt": optional_float(override["cmt"]),
                "time_s": optional_float(override["time_s"]),
                "landing_error_m": optional_float(override["landing_error_m"]),
                "source_record": override["source_record"],
            }
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper2_current"))
    args = parser.parse_args()

    non_mpc = apply_source_aligned_overrides(
        load_non_mpc(args.rerun), args.source_aligned_overrides
    )
    rows = non_mpc + load_tvlqr(args.tvlqr) + load_mpc(args.mpc)
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
        absolute_error = [
            abs(row["landing_error_m"])
            for row in selected
            if row["landing_error_m"] is not None
        ]
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
                "n_landing_error": len(absolute_error),
                "mean_absolute_landing_error_m": format_number(
                    statistics.fmean(absolute_error) if absolute_error else None
                ),
            }
        )

    combined = []
    for row in sorted(retained, key=lambda item: (METHODS.index(item["method"]), item["case_id"])):
        combined.append({key: "" if value is None else value for key, value in row.items()})

    write_csv(args.output_dir / "paper2_combined_cases.csv", combined)
    write_csv(args.output_dir / "table_iv_current.csv", table_iv)
    write_csv(args.output_dir / "table_v_current.csv", table_v)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
