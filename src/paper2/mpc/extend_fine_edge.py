"""Run an isolated adjacent-window MPC refinement for an artificial fine-grid edge.

This helper imports the pinned unified runner without changing it.  It is intended
only for the predeclared closure rule: if a best successful candidate lies on a
non-physical fine-window edge, evaluate the immediately adjacent grid and combine
the results before reporting the optimum.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).with_name("run_unified_12case.py")


def load_runner():
    if not RUNNER.is_file():
        raise RuntimeError(f"Cannot import unified runner: {RUNNER}")
    # A normal import is required so Windows spawn workers can import the
    # module containing evaluate_candidate by the same stable module name.
    return importlib.import_module("run_unified_12case")


def parse_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return [float(v) for v in np.unique(np.round(values, 10))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--grid1", required=True, type=parse_grid)
    parser.add_argument("--grid2", required=True, type=parse_grid)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "results"
        / "paper2_mpc_unified_12case_rerun"
        / "edge_extensions",
    )
    args = parser.parse_args()

    runner = load_runner()
    if args.case not in runner.CASES:
        raise KeyError(f"Unknown case: {args.case}")
    case = runner.CASES[args.case]
    case_output = args.output_root / args.case
    runner.write_manifest(case, case_output, "fine_edge_extension", args.workers)
    runner.worker_initialize(args.case)
    result, sequence = runner.run_parallel_stage(
        case,
        args.stage,
        args.grid1,
        args.grid2,
        case_output,
        args.workers,
        args.resume,
    )
    final = {
        "case_id": args.case,
        "stage": args.stage,
        "success": bool(result.success),
        "reportable_cmt": float(result.cmt),
        "energy_j": float(result.energy_j),
        "distance_m": float(result.distance_m),
        "time_s": float(result.time_s),
        "landing_mae_per_transition_m": float(result.landing_mae_m),
        "terminal_reason": str(result.terminal_reason),
        "best_recovery_sequence": [float(v) for v in sequence],
        "grid1": args.grid1,
        "grid2": args.grid2,
        "rule": "adjacent-window evaluation after a non-physical fine-grid edge hit",
    }
    output = case_output / f"{args.stage}_result.json"
    output.write_text(json.dumps(final, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(final, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
