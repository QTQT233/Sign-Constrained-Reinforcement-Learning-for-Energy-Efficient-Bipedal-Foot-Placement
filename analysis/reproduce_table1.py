"""Recompute Table I from the released 30^4 action maps.

The unsupported continuous-PPO row is deliberately not synthesized: no
matching 30^4 continuous map is available in the released action-map set.
"""

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

    # Preserve the released exclusive-partition if/elif order.
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

    total = active.size
    active_feasible = int(np.sum(active != -2))
    constrained_feasible = int(np.sum(constrained != -2))
    common = int(np.sum((active != -2) & (constrained != -2)))
    table_rows = [
        [
            "Active PPO",
            total,
            active_feasible,
            active_feasible / total,
            int(np.sum(active == -2)),
            np.mean(active == -2),
        ],
        [
            "Sign-constrained union",
            total,
            constrained_feasible,
            constrained_feasible / total,
            int(np.sum(constrained == -2)),
            np.mean(constrained == -2),
        ],
        [
            "Common feasible intersection",
            total,
            common,
            common / total,
            total - common,
            1.0 - common / total,
        ],
    ]
    with (args.output / "table_i.csv").open("w", newline="", encoding="utf-8") as handle:
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
        writer.writerows(table_rows)


if __name__ == "__main__":
    main()
