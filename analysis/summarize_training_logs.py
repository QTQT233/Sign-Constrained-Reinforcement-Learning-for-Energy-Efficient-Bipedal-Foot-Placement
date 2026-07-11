"""Recompute the single-run Table-XII diagnostics from frozen CSV logs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


FILES = [
    ("Main active / bidirectional baseline", "V22_3V9Bi400Grid_training_log.csv"),
    ("Active-full / U0 zero-penalty continuation diagnostic", "V22_3V9ActiveFull400Grid_training_log.csv"),
    ("Per-step encoding continuation diagnostic", "V22_3V9PerStepSign400Grid_training_log.csv"),
    ("Uni-negative sign", "V22_3V9UniNeg400Grid_training_log.csv"),
    ("Uni-positive sign", "V22_3V9UniPos400Grid_training_log.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for label, name in FILES:
        path = args.logs / name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < 20:
            raise ValueError(f"Fewer than 20 epochs in {path}")
        last = rows[-1]
        tail = rows[-20:]
        success = [float(row["success_rate"]) for row in tail]
        ground = [float(row["ground_failure_rate"]) for row in tail]
        clearance = [float(row["clearance_failure_rate"]) for row in tail]
        output_rows.append({
            "label": label,
            "log_file": name,
            "epochs": int(last["epoch"]),
            "final_curriculum": float(last["curriculum_progress"]),
            "final_epoch_success_rate": float(last["success_rate"]),
            "last20_success_mean": statistics.mean(success),
            "last20_success_sample_sd": statistics.stdev(success),
            "last20_ground_failure_mean": statistics.mean(ground),
            "last20_clearance_failure_mean": statistics.mean(clearance),
            "uncertainty_scope": "sample SD across final 20 epoch-level batches; not across training seeds",
        })
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
