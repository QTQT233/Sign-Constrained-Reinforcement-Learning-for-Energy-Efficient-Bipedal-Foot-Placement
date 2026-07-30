"""Regenerate the publication summary for the Active-default confirmation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAIR_FIELDS = (
    "seed",
    "episode_index",
    "walk_direction",
    "reset_phase",
    "commanded_step_length_m",
    "commanded_step_height_m",
)


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in PAIR_FIELDS)


def valid_cmt(row: dict[str, str]) -> float | None:
    if not parse_bool(row["valid_for_cmt"]):
        return None
    value = float(row["cmt"])
    return value if math.isfinite(value) else None


def wilson(count: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = count / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def paired_binary_interval(
    first: list[bool], second: list[bool], z: float = 1.959963984540054
) -> tuple[float, list[float], int, int]:
    second_only = sum((not a) and b for a, b in zip(first, second))
    first_only = sum(a and (not b) for a, b in zip(first, second))
    total = len(first)
    difference = (second_only - first_only) / total
    standard_error = math.sqrt(
        (second_only + first_only) / total**2 - difference**2 / total
    )
    return (
        difference,
        [
            difference - z * standard_error,
            difference + z * standard_error,
        ],
        second_only,
        first_only,
    )


def stratified_bootstrap(
    ordered_keys: list[tuple[str, ...]],
    active_success: list[bool],
    selector_success: list[bool],
    active_valid: list[bool],
    selector_valid: list[bool],
    active_cmt: list[float],
    selector_cmt: list[float],
    seed: int = 20260730,
    resamples: int = 20000,
) -> dict[str, list[float]]:
    generator = np.random.default_rng(seed)
    strata: dict[str, list[int]] = {}
    for index, pair_key in enumerate(ordered_keys):
        strata.setdefault(pair_key[0], []).append(index)
    stratum_indices = [
        np.asarray(indices, dtype=int) for indices in strata.values()
    ]
    success_difference = (
        np.asarray(selector_success, dtype=float)
        - np.asarray(active_success, dtype=float)
    )
    valid_difference = (
        np.asarray(selector_valid, dtype=float)
        - np.asarray(active_valid, dtype=float)
    )
    common = np.asarray(active_valid) & np.asarray(selector_valid)
    cmt_difference = (
        np.asarray(selector_cmt, dtype=float)
        - np.asarray(active_cmt, dtype=float)
    )
    success_samples = np.empty(resamples)
    valid_samples = np.empty(resamples)
    cmt_samples = np.empty(resamples)
    for repetition in range(resamples):
        sampled = np.concatenate(
            [
                generator.choice(indices, size=len(indices), replace=True)
                for indices in stratum_indices
            ]
        )
        success_samples[repetition] = success_difference[sampled].mean()
        valid_samples[repetition] = valid_difference[sampled].mean()
        sampled_common = sampled[common[sampled]]
        cmt_samples[repetition] = cmt_difference[sampled_common].mean()
    return {
        "success_difference_pp_95_ci": [
            100.0 * float(np.quantile(success_samples, 0.025)),
            100.0 * float(np.quantile(success_samples, 0.975)),
        ],
        "valid_difference_pp_95_ci": [
            100.0 * float(np.quantile(valid_samples, 0.025)),
            100.0 * float(np.quantile(valid_samples, 0.975)),
        ],
        "cmt_difference_95_ci": [
            float(np.quantile(cmt_samples, 0.025)),
            float(np.quantile(cmt_samples, 0.975)),
        ],
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=Path,
        default=(
            ROOT
            / "results/four_link_active_default_confirmation"
            / "matched_trials.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/four_link_active_default_confirmation"
            / "publication_summary.json"
        ),
    )
    args = parser.parse_args()

    rows = read_rows(args.trials)
    by_controller = {
        name: [row for row in rows if row["controller"] == name]
        for name in ("active", "active_default_confidence_selector")
    }
    if any(len(records) != 2160 for records in by_controller.values()):
        raise RuntimeError("Each controller must have exactly 2,160 trials")

    active = {key(row): row for row in by_controller["active"]}
    selector = {
        key(row): row
        for row in by_controller["active_default_confidence_selector"]
    }
    if len(active) != 2160 or active.keys() != selector.keys():
        raise RuntimeError("Controller records are not one-to-one matched")

    ordered_keys = sorted(active, key=lambda item: (int(item[0]), int(item[1])))
    active_success = [parse_bool(active[item]["success"]) for item in ordered_keys]
    selector_success = [
        parse_bool(selector[item]["success"]) for item in ordered_keys
    ]
    active_valid = [
        valid_cmt(active[item]) is not None for item in ordered_keys
    ]
    selector_valid = [
        valid_cmt(selector[item]) is not None for item in ordered_keys
    ]
    active_all_cmt = [
        float(active[item]["cmt"]) for item in ordered_keys
    ]
    selector_all_cmt = [
        float(selector[item]["cmt"]) for item in ordered_keys
    ]
    success_difference, success_ci, selector_only_success, active_only_success = (
        paired_binary_interval(active_success, selector_success)
    )
    valid_difference, valid_ci, selector_only_valid, active_only_valid = (
        paired_binary_interval(active_valid, selector_valid)
    )

    common = [
        item
        for item in ordered_keys
        if valid_cmt(active[item]) is not None
        and valid_cmt(selector[item]) is not None
    ]
    active_common = [float(valid_cmt(active[item])) for item in common]
    selector_common = [float(valid_cmt(selector[item])) for item in common]
    differences = [
        selector_value - active_value
        for active_value, selector_value in zip(active_common, selector_common)
    ]

    route_counts = Counter(
        row["passive_selected_from"].split("_p", maxsplit=1)[0]
        for row in by_controller["active_default_confidence_selector"]
    )
    bootstrap = stratified_bootstrap(
        ordered_keys,
        active_success,
        selector_success,
        active_valid,
        selector_valid,
        active_all_cmt,
        selector_all_cmt,
    )
    summary = {
        "schema": "four-link-active-default-confidence-publication-summary/v1",
        "n": 2160,
        "seeds": [20261301, 20261302, 20261303],
        "thresholds": {
            "nonnegative": 0.86,
            "nonpositive": 0.54,
            "default": "active",
            "selection_source": "development data",
        },
        "active_success_count": sum(active_success),
        "selector_success_count": sum(selector_success),
        "active_success_rate": statistics.fmean(active_success),
        "selector_success_rate": statistics.fmean(selector_success),
        "success_difference_pp": 100.0 * success_difference,
        "success_difference_pp_95_ci": bootstrap[
            "success_difference_pp_95_ci"
        ],
        "selector_only_success_count": selector_only_success,
        "active_only_success_count": active_only_success,
        "active_valid_count": sum(active_valid),
        "selector_valid_count": sum(selector_valid),
        "valid_difference_pp": 100.0 * valid_difference,
        "valid_difference_pp_95_ci": bootstrap["valid_difference_pp_95_ci"],
        "selector_only_valid_count": selector_only_valid,
        "active_only_valid_count": active_only_valid,
        "common_valid_count": len(common),
        "active_common_mean_cmt": statistics.fmean(active_common),
        "selector_common_mean_cmt": statistics.fmean(selector_common),
        "selector_minus_active_mean_cmt": statistics.fmean(differences),
        "cmt_difference_95_ci": bootstrap["cmt_difference_95_ci"],
        "ratio_of_means_reduction": (
            1.0
            - statistics.fmean(selector_common) / statistics.fmean(active_common)
        ),
        "selector_lower_count": sum(value < -1e-12 for value in differences),
        "selector_equal_count": sum(abs(value) <= 1e-12 for value in differences),
        "selector_higher_count": sum(value > 1e-12 for value in differences),
        "route_counts": dict(route_counts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
