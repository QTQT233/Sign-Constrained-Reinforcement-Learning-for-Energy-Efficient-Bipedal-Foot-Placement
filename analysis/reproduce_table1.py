"""Recompute every row of manuscript Table I from the released 30^4 maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


ACTIVE = "working_save(-1,0,1)-30,30_from_2-2"
NEGATIVE = "working_save(-1,0)-30,30"
POSITIVE = "working_save(0,1)-30,30"


def load(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        array = handle["working_save"][:]
    if array.shape != (30, 30, 30, 30):
        raise ValueError(f"Unexpected shape for {path}: {array.shape}")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    active = load(args.data / ACTIVE)
    negative = load(args.data / NEGATIVE)
    positive = load(args.data / POSITIVE)

    # Preserve the released exclusive-partition if/elif order used to combine
    # the two one-sided expert maps.
    constrained = np.select(
        [
            (negative == -1) & (positive != 0),
            (negative != 0) & (positive == 1),
            (negative == 0) | (positive == 0),
            (negative == -2) | (positive == -2),
        ],
        [-1, 1, 0, -2],
        default=np.nan,
    )
    if np.isnan(constrained).any():
        raise ValueError("Unclassified map entries")

    active_ok = active != -2
    one_sided_ok = constrained != -2
    both = active_ok & one_sided_ok
    active_only = active_ok & ~one_sided_ok
    one_sided_only = ~active_ok & one_sided_ok
    neither = ~active_ok & ~one_sided_ok
    proposed_ok = active_ok | one_sided_ok

    total = int(active.size)
    if int(both.sum() + active_only.sum() + one_sided_only.sum() + neither.sum()) != total:
        raise AssertionError("Exclusive reachability partition does not sum to the full grid")
    if int(proposed_ok.sum()) != int(both.sum() + active_only.sum() + one_sided_only.sum()):
        raise AssertionError("Proposed-route union does not match its exclusive partitions")

    rows = [
        ("Unrestricted discrete PPO", active_ok, ~active_ok, "marginal"),
        ("One-sided expert union", one_sided_ok, ~one_sided_ok, "marginal"),
        ("Proposed three-expert route", proposed_ok, neither, "marginal union"),
        (
            "Unrestricted discrete PPO and one-sided-expert union",
            both,
            None,
            "exclusive partition",
        ),
        ("Unrestricted discrete PPO only", active_only, None, "exclusive partition"),
        ("One-sided-expert union only", one_sided_only, None, "exclusive partition"),
        ("Neither", neither, None, "exclusive partition"),
    ]

    output_path = args.output / "table_i.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "controller_or_set",
                "sampled_states",
                "feasible_states",
                "feasible_ratio",
                "outside_set_states",
                "outside_set_ratio",
                "row_role",
            ]
        )
        for name, mask, outside_mask, role in rows:
            count = int(mask.sum())
            writer.writerow(
                [
                    name,
                    total,
                    count,
                    count / total,
                    "" if outside_mask is None else int(outside_mask.sum()),
                    "" if outside_mask is None else float(outside_mask.mean()),
                    role,
                ]
            )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
