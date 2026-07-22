"""Paired evaluation-case inference for the retained true action-mask run.

This analysis treats the 2,160 shared evaluation cases as paired observations.
It does not turn one scratch-trained checkpoint into independent training-seed
replications and therefore does not support algorithm-level variance claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "four_link" / "true_action_mask_scratch_c090_epoch1275"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "four_link_statistics"
    / "true_action_mask_scratch_c090_epoch1275"
)
EXPECTED_PAIR_COUNT = 2160
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPLICATES = 10000
Z95 = 1.959963984540054


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected a serialized Boolean, received {value!r}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(successes, trials - successes) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def paired_binary(reference: list[bool], comparison: list[bool]) -> dict[str, float | int | list[float]]:
    if len(reference) != len(comparison):
        raise ValueError("Paired binary inputs have different lengths")
    n = len(reference)
    neither = sum(not ref and not comp for ref, comp in zip(reference, comparison))
    reference_only = sum(ref and not comp for ref, comp in zip(reference, comparison))
    comparison_only = sum(not ref and comp for ref, comp in zip(reference, comparison))
    both = sum(ref and comp for ref, comp in zip(reference, comparison))
    difference = (comparison_only - reference_only) / n
    variance = ((reference_only + comparison_only) / n - difference**2) / n
    standard_error = math.sqrt(max(0.0, variance))
    return {
        "n_pairs": n,
        "neither": neither,
        "reference_only": reference_only,
        "comparison_only": comparison_only,
        "both": both,
        "reference_rate": (reference_only + both) / n,
        "comparison_rate": (comparison_only + both) / n,
        "comparison_minus_reference_risk_difference": difference,
        "paired_standard_error": standard_error,
        "paired_ci95": [
            difference - Z95 * standard_error,
            difference + Z95 * standard_error,
        ],
        "exact_mcnemar_p_two_sided": exact_two_sided_binomial(
            min(reference_only, comparison_only), reference_only + comparison_only
        ),
    }


def bootstrap_mean_ci(differences: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    means = [statistics.fmean(rng.choices(differences, k=n)) for _ in range(BOOTSTRAP_REPLICATES)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pair_path = args.data_dir / "selector_vs_true_action_mask_trials.csv"
    summary_path = args.data_dir / "selector_vs_true_action_mask_summary.csv"
    controller_path = args.data_dir / "controller_summary.csv"
    rows = read_rows(pair_path)
    if len(rows) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PAIR_COUNT} pairs, found {len(rows)}")
    if any("smoke" in row["eval_label"].lower() for row in rows):
        raise RuntimeError("Smoke-test rows cannot be analyzed as the paper result")
    if {row["reference_controller"] for row in rows} != {"true_action_mask_scratch"}:
        raise RuntimeError("Unexpected reference controller in paired table")
    if {row["comparison_controller"] for row in rows} != {"passive_sign_selector"}:
        raise RuntimeError("Unexpected comparison controller in paired table")

    key_fields = (
        "seed",
        "episode_index",
        "walk_direction",
        "reset_phase",
        "commanded_step_length_m",
        "commanded_step_height_m",
    )
    keys = {tuple(row[field] for field in key_fields) for row in rows}
    if len(keys) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PAIR_COUNT} unique case keys, found {len(keys)}")

    success = paired_binary(
        [as_bool(row["reference_success"]) for row in rows],
        [as_bool(row["comparison_success"]) for row in rows],
    )
    validity = paired_binary(
        [as_bool(row["reference_valid"]) for row in rows],
        [as_bool(row["comparison_valid"]) for row in rows],
    )

    both_valid = [
        row
        for row in rows
        if as_bool(row["reference_valid"]) and as_bool(row["comparison_valid"])
    ]
    differences = [
        float(row["comparison_cmt"]) - float(row["reference_cmt"])
        for row in both_valid
    ]
    ratios = [
        float(row["comparison_cmt"]) / float(row["reference_cmt"])
        for row in both_valid
        if float(row["reference_cmt"]) > 0.0
    ]
    comparison_lower = sum(value < 0.0 for value in differences)
    reference_lower = sum(value > 0.0 for value in differences)
    ties = sum(value == 0.0 for value in differences)
    cmt = {
        "n_both_valid": len(differences),
        "reference_mean_cmt": statistics.fmean(float(row["reference_cmt"]) for row in both_valid),
        "comparison_mean_cmt": statistics.fmean(float(row["comparison_cmt"]) for row in both_valid),
        "mean_comparison_minus_reference_cmt": statistics.fmean(differences),
        "median_comparison_minus_reference_cmt": statistics.median(differences),
        "bootstrap_mean_difference_ci95": bootstrap_mean_ci(differences),
        "n_positive_reference_cmt_for_ratio": len(ratios),
        "mean_comparison_over_reference_cmt": statistics.fmean(ratios),
        "comparison_lower_count": comparison_lower,
        "reference_lower_count": reference_lower,
        "tie_count": ties,
        "fraction_comparison_lower_cmt": comparison_lower / len(differences),
        "exact_sign_test_p_two_sided": exact_two_sided_binomial(
            min(comparison_lower, reference_lower), comparison_lower + reference_lower
        ),
    }

    released_summary = read_rows(summary_path)
    if len(released_summary) != 1:
        raise RuntimeError("Expected one direct paired-summary row")
    if int(released_summary[0]["both_valid_pairs"]) != len(differences):
        raise RuntimeError("Direct paired-summary row is inconsistent with trial-level data")

    controllers = {row["controller"]: row for row in read_rows(controller_path)}
    if int(controllers["true_action_mask_scratch"]["n_success"]) != sum(
        as_bool(row["reference_success"]) for row in rows
    ):
        raise RuntimeError("Action-mask success count differs between paired and controller summaries")
    if int(controllers["passive_sign_selector"]["n_success"]) != sum(
        as_bool(row["comparison_success"]) for row in rows
    ):
        raise RuntimeError("Selector success count differs between paired and controller summaries")

    result = {
        "schema_version": 1,
        "reference_controller": "true_action_mask_scratch",
        "comparison_controller": "passive_sign_selector",
        "success": success,
        "valid_cmt": validity,
        "both_valid_cmt": cmt,
        "input_sha256": {
            pair_path.name: sha256(pair_path),
            summary_path.name: sha256(summary_path),
            controller_path.name: sha256(controller_path),
        },
        "inference_boundary": (
            "Evaluation cases are paired. This single retained scratch-trained checkpoint "
            "does not estimate variability across independent training seeds."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_inference.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    binary_rows = []
    for endpoint, values in (("success", success), ("valid_cmt", validity)):
        binary_rows.append({"endpoint": endpoint, **values})
    write_csv(args.output_dir / "paired_binary_outcomes.csv", binary_rows)
    write_csv(args.output_dir / "paired_cmt_summary.csv", [cmt])

    output_files = sorted(
        path for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    with (args.output_dir / "CHECKSUMS.sha256").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for path in output_files:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
