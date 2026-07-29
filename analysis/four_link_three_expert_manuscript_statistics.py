"""Generate manuscript statistics for the four-link three-expert evaluation.

This script reads only the evaluator's matched trial-level CSV files.  It
produces controller summaries, paired binary inference, conditional Cmt
comparisons, displacement-threshold sensitivity, command-grid summaries,
expert-choice counts, terminal reasons, and full-domain penalty components.
Running the file directly uses the formal 2,160-case raw output directory,
which is supplied in Supplementary Archive S4. The public repository retains
the generated compact outputs under ``results/four_link_three_expert_statistics``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "results" / "four_link_three_expert_primary_2160"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "four_link_three_expert_statistics"
EXPECTED_CASES = 2160
Z95 = 1.959963984540054
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260728


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value {value!r}")


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else float("nan")


def median(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else float("nan")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def wilson(successes: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denominator = 1.0 + Z95**2 / n
    center = (p + Z95**2 / (2.0 * n)) / denominator
    half = (
        Z95
        * math.sqrt(p * (1.0 - p) / n + Z95**2 / (4.0 * n**2))
        / denominator
    )
    return [center - half, center + half]


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(
        math.comb(trials, index)
        for index in range(min(successes, trials - successes) + 1)
    )
    return min(1.0, 2.0 * tail / 2**trials)


def paired_binary(reference: list[bool], comparison: list[bool]) -> dict:
    if len(reference) != len(comparison):
        raise ValueError("Paired binary inputs have different lengths")
    n = len(reference)
    both = sum(a and b for a, b in zip(reference, comparison))
    reference_only = sum(a and not b for a, b in zip(reference, comparison))
    comparison_only = sum(not a and b for a, b in zip(reference, comparison))
    neither = n - both - reference_only - comparison_only
    difference = (comparison_only - reference_only) / n
    q = (reference_only + comparison_only) / n
    standard_error = math.sqrt(max(0.0, (q - difference**2) / n))
    return {
        "n": n,
        "both": both,
        "reference_only": reference_only,
        "comparison_only": comparison_only,
        "neither": neither,
        "reference_rate": (both + reference_only) / n,
        "comparison_rate": (both + comparison_only) / n,
        "comparison_minus_reference": difference,
        "paired_ci95": [
            difference - Z95 * standard_error,
            difference + Z95 * standard_error,
        ],
        "exact_mcnemar_p": exact_binomial_two_sided(
            min(reference_only, comparison_only),
            reference_only + comparison_only,
        ),
    }


def bootstrap_mean_ci(values: list[float], salt: int) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(BOOTSTRAP_SEED + salt)
    n = len(values)
    means = [
        statistics.fmean(rng.choices(values, k=n))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def pair_statistics(
    rows: list[dict[str, str]],
    *,
    pair_name: str,
    reference_name: str,
    comparison_name: str,
    prefix_reference: str,
    prefix_comparison: str,
    salt: int,
) -> dict:
    ref_success_key = f"{prefix_reference}_success"
    cmp_success_key = f"{prefix_comparison}_success"
    ref_valid_key = f"{prefix_reference}_valid"
    cmp_valid_key = f"{prefix_comparison}_valid"
    ref_cmt_key = f"{prefix_reference}_cmt"
    cmp_cmt_key = f"{prefix_comparison}_cmt"

    success = paired_binary(
        [as_bool(row[ref_success_key]) for row in rows],
        [as_bool(row[cmp_success_key]) for row in rows],
    )
    validity = paired_binary(
        [as_bool(row[ref_valid_key]) for row in rows],
        [as_bool(row[cmp_valid_key]) for row in rows],
    )
    both_valid = [
        row
        for row in rows
        if as_bool(row[ref_valid_key]) and as_bool(row[cmp_valid_key])
    ]
    reference_values = [as_float(row[ref_cmt_key]) for row in both_valid]
    comparison_values = [as_float(row[cmp_cmt_key]) for row in both_valid]
    differences = [
        comparison - reference
        for reference, comparison in zip(reference_values, comparison_values)
    ]
    reductions = [
        (reference - comparison) / reference
        for reference, comparison in zip(reference_values, comparison_values)
        if reference > 0.0
    ]
    comparison_lower = sum(value < 0.0 for value in differences)
    reference_lower = sum(value > 0.0 for value in differences)
    ties = sum(value == 0.0 for value in differences)
    return {
        "pair": pair_name,
        "reference": reference_name,
        "comparison": comparison_name,
        "success": success,
        "validity": validity,
        "both_valid_n": len(both_valid),
        "both_valid_rate": len(both_valid) / len(rows),
        "both_valid_rate_ci95": wilson(len(both_valid), len(rows)),
        "reference_mean_cmt": mean(reference_values),
        "comparison_mean_cmt": mean(comparison_values),
        "comparison_minus_reference_mean_cmt": mean(differences),
        "comparison_minus_reference_median_cmt": median(differences),
        "comparison_minus_reference_mean_cmt_ci95": bootstrap_mean_ci(
            differences, salt
        ),
        "positive_reference_n": len(reductions),
        "mean_pairwise_reduction": mean(reductions),
        "mean_pairwise_reduction_ci95": bootstrap_mean_ci(
            reductions, salt + 100
        ),
        "comparison_lower_count": comparison_lower,
        "reference_lower_count": reference_lower,
        "tie_count": ties,
        "comparison_lower_fraction": comparison_lower / max(1, len(differences)),
        "comparison_lower_fraction_ci95": wilson(
            comparison_lower, len(differences)
        ),
        "exact_sign_test_p": exact_binomial_two_sided(
            min(comparison_lower, reference_lower),
            comparison_lower + reference_lower,
        ),
    }


def controller_with_intervals(row: dict[str, str]) -> dict:
    n = int(row["n_all"])
    successes = int(row["n_success"])
    valid = int(row["n_valid_cmt"])
    return {
        **row,
        "success_ci95": wilson(successes, n),
        "valid_cmt_ci95": wilson(valid, n),
    }


def displacement_sensitivity(rows: list[dict[str, str]]) -> list[dict]:
    results = []
    for threshold in (0.001, 0.005, 0.010, 0.020):
        subset = [
            row
            for row in rows
            if as_bool(row["active_success"])
            and as_bool(row["passive_success"])
            and as_float(row["active_displacement_m"]) > threshold
            and as_float(row["passive_displacement_m"]) > threshold
            and math.isfinite(as_float(row["active_cmt"]))
            and math.isfinite(as_float(row["passive_cmt"]))
        ]
        active_values = [as_float(row["active_cmt"]) for row in subset]
        comparison_values = [as_float(row["passive_cmt"]) for row in subset]
        comparison_lower = sum(
            comparison < active
            for active, comparison in zip(active_values, comparison_values)
        )
        results.append(
            {
                "minimum_signed_displacement_m": threshold,
                "matched_pairs": len(subset),
                "active_mean_cmt": mean(active_values),
                "three_expert_mean_cmt": mean(comparison_values),
                "three_expert_lower_count": comparison_lower,
                "three_expert_lower_fraction": comparison_lower
                / max(1, len(subset)),
                "three_expert_lower_fraction_ci95": wilson(
                    comparison_lower, len(subset)
                ),
            }
        )
    return results


def grid_cell_summary(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                as_float(row["commanded_step_length_m"]),
                as_float(row["commanded_step_height_m"]),
            )
        ].append(row)
    results = []
    for (length, height), subset in sorted(groups.items()):
        pair = pair_statistics(
            subset,
            pair_name=f"{length:.2f}_{height:.2f}",
            reference_name="active",
            comparison_name="three_expert_selector",
            prefix_reference="active",
            prefix_comparison="passive",
            salt=int(round(10_000 * length + 100_000 * height)),
        )
        results.append(
            {
                "commanded_step_length_m": length,
                "commanded_step_height_m": height,
                "n": len(subset),
                "active_success_rate": pair["success"]["reference_rate"],
                "three_expert_success_rate": pair["success"]["comparison_rate"],
                "active_valid_rate": pair["validity"]["reference_rate"],
                "three_expert_valid_rate": pair["validity"]["comparison_rate"],
                "both_valid_n": pair["both_valid_n"],
                "both_valid_rate": pair["both_valid_rate"],
                "both_valid_rate_ci95": pair["both_valid_rate_ci95"],
                "mean_pairwise_reduction": pair["mean_pairwise_reduction"],
                "mean_pairwise_reduction_ci95": pair[
                    "mean_pairwise_reduction_ci95"
                ],
            }
        )
    return results


def expert_choice_summary(rows: list[dict[str, str]]) -> dict:
    choices = Counter()
    success_by_choice = Counter()
    valid_by_choice = Counter()
    for row in rows:
        selected = str(row["passive_selected_from"])
        if selected.startswith("three_expert_positive_"):
            choice = "positive"
        elif selected.startswith("three_expert_negative_"):
            choice = "negative"
        elif selected.startswith("three_expert_active_"):
            choice = "active"
        else:
            choice = "unknown"
        choices[choice] += 1
        success_by_choice[choice] += int(as_bool(row["passive_success"]))
        valid_by_choice[choice] += int(as_bool(row["passive_valid"]))
    return {
        choice: {
            "selected_n": choices[choice],
            "success_n": success_by_choice[choice],
            "valid_cmt_n": valid_by_choice[choice],
            "success_rate": success_by_choice[choice] / max(1, choices[choice]),
            "valid_cmt_rate": valid_by_choice[choice] / max(1, choices[choice]),
        }
        for choice in ("positive", "negative", "active", "unknown")
    }


def terminal_reason_rows(
    rollout_rows: list[dict[str, str]], controllers: tuple[str, ...]
) -> list[dict]:
    output = []
    for controller in controllers:
        subset = [
            row for row in rollout_rows if row["controller"] == controller
        ]
        counts = Counter(row["terminal_reason"] for row in subset)
        for reason, count in counts.most_common():
            output.append(
                {
                    "controller": controller,
                    "terminal_reason": reason,
                    "count": count,
                    "fraction": count / max(1, len(subset)),
                }
            )
    return output


def penalty_components(
    rollout_rows: list[dict[str, str]], controllers: tuple[str, ...]
) -> list[dict]:
    rows = []
    for controller in controllers:
        subset = [
            row for row in rollout_rows if row["controller"] == controller
        ]
        valid = [row for row in subset if as_bool(row["valid_for_cmt"])]
        success_invalid = [
            row
            for row in subset
            if as_bool(row["success"]) and not as_bool(row["valid_for_cmt"])
        ]
        failed = [row for row in subset if not as_bool(row["success"])]
        rows.append(
            {
                "controller": controller,
                "valid_cases": len(valid),
                "sum_valid_cmt": sum(as_float(row["cmt"]) for row in valid),
                "successful_but_invalid": len(success_invalid),
                "failed_cases": len(failed),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    paths = {
        "controller": args.data_dir / "controller_summary.csv",
        "paired": args.data_dir / "paired_trials.csv",
        "active_reference": args.data_dir
        / "active_reference_comparison_trials.csv",
        "mask_three": args.data_dir
        / "three_expert_vs_true_action_mask_trials.csv",
        "rollouts": args.data_dir / "rollouts_active_vs_passive.csv",
    }
    controller_rows = read_rows(paths["controller"])
    paired_rows = read_rows(paths["paired"])
    active_reference_rows = read_rows(paths["active_reference"])
    mask_three_rows = read_rows(paths["mask_three"])
    rollout_rows = read_rows(paths["rollouts"])

    controllers = {
        row["controller"]: controller_with_intervals(row)
        for row in controller_rows
    }
    required = {
        "active",
        "active_full",
        "true_action_mask_scratch",
        "three_expert_selector",
        "passive_oracle_envelope",
    }
    if not required.issubset(controllers):
        raise RuntimeError(
            f"Missing controller rows: {sorted(required - set(controllers))}"
        )
    if {int(controllers[key]["n_all"]) for key in required} != {
        EXPECTED_CASES
    }:
        raise RuntimeError("Formal controller counts are not all 2,160")

    three_pairs = [
        row
        for row in paired_rows
        if row["passive_controller"] == "three_expert_selector"
    ]
    oracle_pairs = [
        row
        for row in paired_rows
        if row["passive_controller"] == "passive_oracle_envelope"
    ]
    if len(three_pairs) != EXPECTED_CASES or len(oracle_pairs) != EXPECTED_CASES:
        raise RuntimeError("Expected 2,160 active/three-expert and active/oracle pairs")
    if len(mask_three_rows) != EXPECTED_CASES:
        raise RuntimeError("Expected 2,160 hard-mask/three-expert pairs")

    pair_results = {
        "active_vs_three_expert": pair_statistics(
            three_pairs,
            pair_name="active_vs_three_expert",
            reference_name="active",
            comparison_name="three_expert_selector",
            prefix_reference="active",
            prefix_comparison="passive",
            salt=1,
        ),
        "active_vs_oracle": pair_statistics(
            oracle_pairs,
            pair_name="active_vs_oracle",
            reference_name="active",
            comparison_name="passive_oracle_envelope",
            prefix_reference="active",
            prefix_comparison="passive",
            salt=2,
        ),
        "hard_mask_vs_three_expert": pair_statistics(
            mask_three_rows,
            pair_name="hard_mask_vs_three_expert",
            reference_name="true_action_mask_scratch",
            comparison_name="three_expert_selector",
            prefix_reference="reference",
            prefix_comparison="comparison",
            salt=3,
        ),
    }

    displacement = displacement_sensitivity(three_pairs)
    grid = grid_cell_summary(three_pairs)
    choices = expert_choice_summary(three_pairs)
    terminal = terminal_reason_rows(
        rollout_rows,
        ("active", "three_expert_selector", "true_action_mask_scratch"),
    )
    penalty = penalty_components(
        rollout_rows,
        ("active", "true_action_mask_scratch", "three_expert_selector"),
    )
    result = {
        "schema_version": 1,
        "case_count_per_controller": EXPECTED_CASES,
        "controllers": controllers,
        "pairs": pair_results,
        "displacement_sensitivity": displacement,
        "grid_cells": grid,
        "three_expert_choices": choices,
        "penalty_components": penalty,
        "input_sha256": {
            name: sha256(path) for name, path in paths.items()
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "inference_boundary": (
            "All paired intervals condition on one frozen checkpoint per "
            "controller and do not estimate independent-training-seed variance."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manuscript_statistics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_rows(args.output_dir / "displacement_sensitivity.csv", displacement)
    write_rows(args.output_dir / "grid_cell_summary.csv", grid)
    write_rows(args.output_dir / "terminal_reason_summary.csv", terminal)
    write_rows(args.output_dir / "penalty_components.csv", penalty)
    write_rows(
        args.output_dir / "pair_summary.csv",
        [
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            for row in pair_results.values()
        ],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
