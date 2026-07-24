"""Paired fixed-checkpoint inference for the four-link selector comparison.

The script is intentionally standard-library only. It never imports, executes,
or modifies a training/evaluation program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


Z95 = 1.959963984540054
Z90 = 1.6448536269514722


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def exact_mcnemar(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    smaller = min(a_only, b_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * lower_tail)


def paired_binary(active: list[bool], selector: list[bool]) -> dict:
    n = len(active)
    both = sum(a and s for a, s in zip(active, selector))
    active_only = sum(a and not s for a, s in zip(active, selector))
    selector_only = sum(not a and s for a, s in zip(active, selector))
    neither = n - both - active_only - selector_only
    difference = (selector_only - active_only) / n
    q = (active_only + selector_only) / n
    se = math.sqrt(max(0.0, (q - difference * difference) / n))
    return {
        "n": n,
        "both_success": both,
        "active_only": active_only,
        "selector_only": selector_only,
        "both_failure": neither,
        "active_rate": sum(active) / n,
        "selector_rate": sum(selector) / n,
        "difference_selector_minus_active": difference,
        "paired_standard_error": se,
        "paired_ci95": [difference - Z95 * se, difference + Z95 * se],
        "paired_ci90": [difference - Z90 * se, difference + Z90 * se],
        "one_sided_95_lower": difference - Z90 * se,
        "exact_mcnemar_two_sided_p": exact_mcnemar(active_only, selector_only),
    }


def displacement_sensitivity(rows: list[dict[str, str]], thresholds: list[float]) -> list[dict]:
    """Summarize matched valid-Cmt pairs after a common displacement threshold."""
    summaries = []
    for threshold in thresholds:
        eligible = [
            row
            for row in rows
            if as_bool(row["active_valid"])
            and as_bool(row["passive_valid"])
            and float(row["active_displacement_m"]) > threshold
            and float(row["passive_displacement_m"]) > threshold
            and math.isfinite(float(row["active_cmt"]))
            and math.isfinite(float(row["passive_cmt"]))
        ]
        active_cmt = [float(row["active_cmt"]) for row in eligible]
        selector_cmt = [float(row["passive_cmt"]) for row in eligible]
        selector_lower = sum(s < a for a, s in zip(active_cmt, selector_cmt))
        active_lower = sum(a < s for a, s in zip(active_cmt, selector_cmt))
        ties = len(eligible) - selector_lower - active_lower
        summaries.append({
            "minimum_signed_displacement_m": threshold,
            "n_both_valid": len(eligible),
            "active_mean_cmt": sum(active_cmt) / len(active_cmt) if active_cmt else math.nan,
            "selector_mean_cmt": sum(selector_cmt) / len(selector_cmt) if selector_cmt else math.nan,
            "mean_active_minus_selector_cmt": (
                sum(a - s for a, s in zip(active_cmt, selector_cmt)) / len(eligible)
                if eligible else math.nan
            ),
            "selector_lower_count": selector_lower,
            "active_lower_count": active_lower,
            "tie_count": ties,
            "selector_lower_fraction": selector_lower / len(eligible) if eligible else math.nan,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True, type=Path)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controller", default="passive_sign_selector")
    parser.add_argument("--margins", default="0.02,0.03,0.05",
                        help="Absolute candidate margins; sensitivity analysis only unless prespecified")
    parser.add_argument(
        "--displacement-thresholds",
        default="0.001,0.005,0.010,0.020",
        help="Common signed COM-displacement thresholds for the matched-Cmt sensitivity table",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.paired.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row.get("passive_controller") == args.controller]
    if not rows:
        raise SystemExit(f"No rows for passive_controller={args.controller!r}")

    keys = [(row["seed"], row["episode_index"]) for row in rows]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise SystemExit(f"Found {duplicates} duplicate (seed, episode_index) keys")

    success = paired_binary(
        [as_bool(row["active_success"]) for row in rows],
        [as_bool(row["passive_success"]) for row in rows],
    )
    validity = paired_binary(
        [as_bool(row["active_valid"]) for row in rows],
        [as_bool(row["passive_valid"]) for row in rows],
    )

    margins = [float(item) for item in args.margins.split(",") if item.strip()]
    displacement_thresholds = [
        float(item) for item in args.displacement_thresholds.split(",") if item.strip()
    ]
    displacement_results = displacement_sensitivity(rows, displacement_thresholds)
    ni = []
    for margin in margins:
        ni.append({
            "candidate_absolute_margin": margin,
            "one_sided_95_lower": success["one_sided_95_lower"],
            "passes_if_margin_had_been_prespecified": success["one_sided_95_lower"] > -margin,
            "status": "post_hoc_sensitivity_not_confirmatory",
        })

    transition_counts = Counter(
        (row["active_terminal_reason"], row["passive_terminal_reason"]) for row in rows
    )
    with (args.output / "terminal_reason_transition_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["active_terminal_reason", "selector_terminal_reason", "count"])
        for (active_reason, selector_reason), count in sorted(transition_counts.items()):
            writer.writerow([active_reason, selector_reason, count])

    terminal_meta = None
    if args.terminal:
        with args.terminal.open(newline="", encoding="utf-8-sig") as handle:
            terminal_rows = list(csv.DictReader(handle))
        terminal_meta = {
            "path": args.terminal.as_posix(),
            "sha256": sha256(args.terminal),
            "rows": len(terminal_rows),
        }

    result = {
        "scope": "one fixed checkpoint per controller on a shared evaluation-case distribution",
        "not_supported": "algorithm-level equivalence or robustness across independent training seeds",
        "paired_input": {
            "path": args.paired.as_posix(),
            "sha256": sha256(args.paired),
            "all_rows": len(all_rows),
            "selected_rows": len(rows),
            "unique_pair_keys": len(set(keys)),
        },
        "terminal_summary_input": terminal_meta,
        "success": success,
        "validity": validity,
        "displacement_threshold_sensitivity": displacement_results,
        "noninferiority_sensitivity": ni,
        "interpretation": [
            "A non-significant McNemar test does not prove equivalence.",
            "A non-inferiority claim requires an engineering margin fixed before examining the confirmatory data.",
            "Marginal terminal-reason counts cannot replace the paired binary success analysis.",
        ],
    }
    (args.output / "paired_inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (args.output / "paired_success_2x2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["active_status", "selector_success", "selector_failure", "total"])
        writer.writerow(["success", success["both_success"], success["active_only"],
                         success["both_success"] + success["active_only"]])
        writer.writerow(["failure", success["selector_only"], success["both_failure"],
                         success["selector_only"] + success["both_failure"]])

    with (args.output / "displacement_threshold_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(displacement_results[0]))
        writer.writeheader()
        writer.writerows(displacement_results)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
