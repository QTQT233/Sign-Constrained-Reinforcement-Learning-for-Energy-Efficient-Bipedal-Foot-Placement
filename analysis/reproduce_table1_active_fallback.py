"""Reproduce the source-aligned two-link Table I and route array.

The three source maps use the same 30^4 state grid and symmetric dynamics from
the released PPO training environment.  Route codes are assigned once per
transition:

    -1  uncovered
     0  non-positive expert
     1  non-negative expert
     2  unrestricted Active PPO fallback

Where both sign experts are feasible, the non-positive expert is used as a
deterministic tie rule.  This choice does not change the reachability counts
reported in Table I.  The energy-ranked multi-step route used for Table V is
released separately with its complete recovery-grid records.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


SHAPE = (30, 30, 30, 30)
TOTAL = int(np.prod(SHAPE))
FILENAMES = {
    "active": "candidate_active_30x30x30x30.h5",
    "negative": "negative_30x30x30x30.h5",
    "positive": "positive_30x30x30x30.h5",
}


def load_map(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = handle["working_save"][:]
    if values.shape != SHAPE:
        raise ValueError(f"{path} has shape {values.shape}; expected {SHAPE}")
    return values


def count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def row(label: str, feasible: int) -> list[object]:
    return [label, TOTAL, feasible, feasible / TOTAL, TOTAL - feasible,
            1.0 - feasible / TOTAL]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/two_link/action_map_source_aligned"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/table1"),
    )
    args = parser.parse_args()

    active = load_map(args.data_root / FILENAMES["active"])
    negative = load_map(args.data_root / FILENAMES["negative"])
    positive = load_map(args.data_root / FILENAMES["positive"])

    active_ok = active != -2
    negative_ok = negative != -2
    positive_ok = positive != -2
    sign_ok = negative_ok | positive_ok

    both = active_ok & sign_ok
    active_only = active_ok & ~sign_ok
    sign_only = ~active_ok & sign_ok
    neither = ~active_ok & ~sign_ok
    active_default_ok = active_ok | sign_ok

    route = np.full(SHAPE, -1, dtype=np.int8)
    route[negative_ok] = 0
    route[~negative_ok & positive_ok] = 1
    route[~sign_ok & active_ok] = 2

    expected = {
        "active": 647160,
        "sign_union": 562003,
        "both": 536781,
        "active_only": 110379,
        "sign_only": 25222,
        "neither": 137618,
        "active_default": 672382,
    }
    observed = {
        "active": count(active_ok),
        "sign_union": count(sign_ok),
        "both": count(both),
        "active_only": count(active_only),
        "sign_only": count(sign_only),
        "neither": count(neither),
        "active_default": count(active_default_ok),
    }
    if observed != expected:
        raise RuntimeError(
            "Source-aligned partition mismatch:\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2)
        )
    if count(route >= 0) != expected["active_default"]:
        raise RuntimeError("Route coverage differs from the set union")
    if sum(observed[key] for key in ("both", "active_only", "sign_only", "neither")) != TOTAL:
        raise RuntimeError("Exclusive partition does not sum to the full grid")

    args.data_root.mkdir(parents=True, exist_ok=True)
    route_path = args.data_root / "active_default_route_30x30x30x30.h5"
    with h5py.File(route_path, "w") as handle:
        dataset = handle.create_dataset(
            "route_code",
            data=route,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        dataset.attrs["codes"] = (
            "-1=uncovered,0=non-positive,1=non-negative,2=active"
        )
        handle.attrs["schema"] = "two-link-active-default-route/v1"
        handle.attrs["grid_shape"] = SHAPE
        handle.attrs["dynamics"] = "symmetric PPO-training source"
        handle.attrs["query_schedule"] = "once at transition onset"
        handle.attrs["tie_rule"] = "non-positive when both sign experts are feasible"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row("Unrestricted Active PPO", observed["active"]),
        row("One-sided expert union", observed["sign_union"]),
        row("Active-default three-expert lookup", observed["active_default"]),
        row("Active and one-sided feasible", observed["both"]),
        row("Active only", observed["active_only"]),
        row("One-sided only", observed["sign_only"]),
        row("Neither", observed["neither"]),
    ]
    with (args.output_dir / "table_i.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "controller_or_set",
                "sampled_states",
                "feasible_states",
                "feasible_ratio",
                "outside_set_states",
                "outside_set_ratio",
            ]
        )
        writer.writerows(rows)

    summary = {
        "schema": "table1-source-aligned-active-fallback/v1",
        "grid_shape": list(SHAPE),
        "sampled_states": TOTAL,
        "route_codes": {
            "-1": "uncovered",
            "0": "non-positive expert",
            "1": "non-negative expert",
            "2": "unrestricted Active PPO fallback",
        },
        "counts": observed,
        "active_default_gain_over_active_percentage_points": (
            100.0
            * (observed["active_default"] - observed["active"])
            / TOTAL
        ),
    }
    (args.output_dir / "table_i_source_aligned_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
