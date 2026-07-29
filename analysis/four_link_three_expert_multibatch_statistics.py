"""Aggregate five independently seeded four-link three-expert evaluations.

The script combines ``batch01`` through ``batch05`` below
``results/four_link_three_expert_additional_batches``.  It reports:

* per-batch and pooled success/valid-Cmt rates;
* paired Active-PPO versus three-expert-selector inference;
* conditional paired Cmt contrasts and selector-lower shares;
* command-grid strata; and
* full-domain penalty components.

All paired comparisons preserve the evaluator's case matching.  Conditional
Cmt confidence intervals use a bootstrap stratified by evaluation batch (and,
for command-grid rows, by batch within grid cell).  The inference therefore
describes repeated randomized evaluation states for the same frozen
checkpoints; it does not estimate retraining-seed variation.

The full batch input directories are supplied in Supplementary Archive S4.
The public repository retains the generated compact pooled outputs under
``results/four_link_three_expert_additional_batch_statistics``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import four_link_three_expert_manuscript_statistics as single


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    ROOT / "results" / "four_link_three_expert_additional_batches"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "four_link_three_expert_additional_batch_statistics"
)
BATCH_NAMES = tuple(f"batch{index:02d}" for index in range(1, 6))
EXPECTED_CASES_PER_BATCH = 2160
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260729


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_binomial_two_sided_stable(successes: int, trials: int) -> float:
    """Exact two-sided sign/McNemar p value without giant-int conversion."""

    if trials == 0:
        return 1.0
    tail_index = min(successes, trials - successes)
    log_term = (
        math.lgamma(trials + 1)
        - math.lgamma(tail_index + 1)
        - math.lgamma(trials - tail_index + 1)
        - trials * math.log(2.0)
    )
    term = math.exp(log_term)
    tail = term
    for index in range(tail_index, 0, -1):
        term *= index / (trials - index + 1)
        tail += term
    return min(1.0, 2.0 * tail)


def stratified_bootstrap_mean_ci(
    values_by_stratum: dict[str, list[float]],
    *,
    salt: int,
) -> list[float]:
    """Bootstrap a pooled mean while retaining each stratum's sample size."""

    groups = [values for values in values_by_stratum.values() if values]
    total_n = sum(len(values) for values in groups)
    if total_n == 0:
        return [float("nan"), float("nan")]
    rng = random.Random(BOOTSTRAP_SEED + salt)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for values in groups:
            total += sum(rng.choices(values, k=len(values)))
        replicates.append(total / total_n)
    return [quantile(replicates, 0.025), quantile(replicates, 0.975)]


def paired_key(row: dict[str, str]) -> tuple:
    return (
        row["seed"],
        row["episode_index"],
        row["walk_direction"],
        row["reset_phase"],
        row["commanded_step_length_m"],
        row["commanded_step_height_m"],
    )


def validate_batch_pairs(rows: list[dict[str, str]], batch_name: str) -> None:
    if len(rows) != EXPECTED_CASES_PER_BATCH:
        raise RuntimeError(
            f"{batch_name}: expected {EXPECTED_CASES_PER_BATCH} matched "
            f"Active/three-expert rows, found {len(rows)}"
        )
    keys = [paired_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        duplicates = len(keys) - len(set(keys))
        raise RuntimeError(f"{batch_name}: {duplicates} duplicate paired keys")


def controller_summary(
    rollout_rows: list[dict[str, str]],
    *,
    controller: str,
) -> dict:
    subset = [row for row in rollout_rows if row["controller"] == controller]
    n = len(subset)
    successes = sum(single.as_bool(row["success"]) for row in subset)
    valid_rows = [row for row in subset if single.as_bool(row["valid_for_cmt"])]
    cmts = [single.as_float(row["cmt"]) for row in valid_rows]
    return {
        "controller": controller,
        "n_all": n,
        "n_success": successes,
        "success_rate": successes / max(1, n),
        "success_ci95": single.wilson(successes, n),
        "n_valid_cmt": len(valid_rows),
        "valid_cmt_rate": len(valid_rows) / max(1, n),
        "valid_cmt_ci95": single.wilson(len(valid_rows), n),
        "mean_cmt_valid": single.mean(cmts),
        "median_cmt_valid": single.median(cmts),
    }


def paired_cmt_vectors(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    both_valid = [
        row
        for row in rows
        if single.as_bool(row["active_valid"])
        and single.as_bool(row["passive_valid"])
    ]
    active = [single.as_float(row["active_cmt"]) for row in both_valid]
    selector = [single.as_float(row["passive_cmt"]) for row in both_valid]
    differences = [
        comparison - reference
        for reference, comparison in zip(active, selector)
    ]
    reductions = [
        (reference - comparison) / reference
        for reference, comparison in zip(active, selector)
        if reference > 0.0
    ]
    return both_valid, active, selector, differences, reductions


def pooled_pair_statistics(rows: list[dict[str, str]]) -> dict:
    result = single.pair_statistics(
        rows,
        pair_name="active_vs_three_expert",
        reference_name="active",
        comparison_name="three_expert_selector",
        prefix_reference="active",
        prefix_comparison="passive",
        salt=501,
    )
    both_valid, _active, _selector, differences, _reductions = (
        paired_cmt_vectors(rows)
    )
    differences_by_batch: dict[str, list[float]] = defaultdict(list)
    reductions_by_batch: dict[str, list[float]] = defaultdict(list)
    for row, difference in zip(both_valid, differences):
        batch = row["_batch"]
        differences_by_batch[batch].append(difference)
        reference = single.as_float(row["active_cmt"])
        comparison = single.as_float(row["passive_cmt"])
        if reference > 0.0:
            reductions_by_batch[batch].append(
                (reference - comparison) / reference
            )
    result["comparison_minus_reference_mean_cmt_ci95"] = (
        stratified_bootstrap_mean_ci(differences_by_batch, salt=1)
    )
    result["mean_pairwise_reduction_ci95"] = stratified_bootstrap_mean_ci(
        reductions_by_batch, salt=2
    )
    result["bootstrap_strata"] = "evaluation_batch"
    return result


def per_batch_rows(
    batch_pairs: dict[str, list[dict[str, str]]],
    batch_rollouts: dict[str, list[dict[str, str]]],
) -> list[dict]:
    output: list[dict] = []
    for index, batch_name in enumerate(BATCH_NAMES):
        pairs = batch_pairs[batch_name]
        pair = single.pair_statistics(
            pairs,
            pair_name=f"{batch_name}_active_vs_three_expert",
            reference_name="active",
            comparison_name="three_expert_selector",
            prefix_reference="active",
            prefix_comparison="passive",
            salt=600 + index,
        )
        active = controller_summary(
            batch_rollouts[batch_name], controller="active"
        )
        selector = controller_summary(
            batch_rollouts[batch_name], controller="three_expert_selector"
        )
        output.append(
            {
                "batch": batch_name,
                "n": len(pairs),
                "active_success_n": active["n_success"],
                "active_success_rate": active["success_rate"],
                "three_expert_success_n": selector["n_success"],
                "three_expert_success_rate": selector["success_rate"],
                "success_difference": pair["success"][
                    "comparison_minus_reference"
                ],
                "success_difference_ci95": pair["success"]["paired_ci95"],
                "success_mcnemar_p": pair["success"]["exact_mcnemar_p"],
                "active_valid_n": active["n_valid_cmt"],
                "active_valid_rate": active["valid_cmt_rate"],
                "three_expert_valid_n": selector["n_valid_cmt"],
                "three_expert_valid_rate": selector["valid_cmt_rate"],
                "valid_difference": pair["validity"][
                    "comparison_minus_reference"
                ],
                "valid_difference_ci95": pair["validity"]["paired_ci95"],
                "valid_mcnemar_p": pair["validity"]["exact_mcnemar_p"],
                "both_valid_n": pair["both_valid_n"],
                "both_valid_rate": pair["both_valid_rate"],
                "both_valid_rate_ci95": pair["both_valid_rate_ci95"],
                "active_paired_mean_cmt": pair["reference_mean_cmt"],
                "three_expert_paired_mean_cmt": pair[
                    "comparison_mean_cmt"
                ],
                "three_minus_active_mean_cmt": pair[
                    "comparison_minus_reference_mean_cmt"
                ],
                "three_minus_active_mean_cmt_ci95": pair[
                    "comparison_minus_reference_mean_cmt_ci95"
                ],
                "mean_pairwise_reduction": pair[
                    "mean_pairwise_reduction"
                ],
                "mean_pairwise_reduction_ci95": pair[
                    "mean_pairwise_reduction_ci95"
                ],
                "three_expert_lower_n": pair["comparison_lower_count"],
                "three_expert_lower_fraction": pair[
                    "comparison_lower_fraction"
                ],
                "three_expert_lower_fraction_ci95": pair[
                    "comparison_lower_fraction_ci95"
                ],
                "sign_test_p": pair["exact_sign_test_p"],
            }
        )
    return output


def grid_summary(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                single.as_float(row["commanded_step_length_m"]),
                single.as_float(row["commanded_step_height_m"]),
            )
        ].append(row)

    output: list[dict] = []
    for grid_index, ((length, height), subset) in enumerate(
        sorted(groups.items())
    ):
        pair = single.pair_statistics(
            subset,
            pair_name=f"{length:.2f}_{height:.2f}",
            reference_name="active",
            comparison_name="three_expert_selector",
            prefix_reference="active",
            prefix_comparison="passive",
            salt=700 + grid_index,
        )
        both_valid, _active, _selector, differences, _reductions = (
            paired_cmt_vectors(subset)
        )
        differences_by_batch: dict[str, list[float]] = defaultdict(list)
        reductions_by_batch: dict[str, list[float]] = defaultdict(list)
        for row, difference in zip(both_valid, differences):
            batch = row["_batch"]
            differences_by_batch[batch].append(difference)
            reference = single.as_float(row["active_cmt"])
            comparison = single.as_float(row["passive_cmt"])
            if reference > 0.0:
                reductions_by_batch[batch].append(
                    (reference - comparison) / reference
                )
        output.append(
            {
                "commanded_step_length_m": length,
                "commanded_step_height_m": height,
                "n": len(subset),
                "active_success_rate": pair["success"]["reference_rate"],
                "three_expert_success_rate": pair["success"][
                    "comparison_rate"
                ],
                "success_difference": pair["success"][
                    "comparison_minus_reference"
                ],
                "success_difference_ci95": pair["success"]["paired_ci95"],
                "success_mcnemar_p": pair["success"]["exact_mcnemar_p"],
                "active_valid_rate": pair["validity"]["reference_rate"],
                "three_expert_valid_rate": pair["validity"][
                    "comparison_rate"
                ],
                "valid_difference": pair["validity"][
                    "comparison_minus_reference"
                ],
                "valid_difference_ci95": pair["validity"]["paired_ci95"],
                "valid_mcnemar_p": pair["validity"]["exact_mcnemar_p"],
                "both_valid_n": pair["both_valid_n"],
                "both_valid_rate": pair["both_valid_rate"],
                "both_valid_rate_ci95": pair["both_valid_rate_ci95"],
                "active_paired_mean_cmt": pair["reference_mean_cmt"],
                "three_expert_paired_mean_cmt": pair[
                    "comparison_mean_cmt"
                ],
                "three_minus_active_mean_cmt": pair[
                    "comparison_minus_reference_mean_cmt"
                ],
                "three_minus_active_mean_cmt_ci95": (
                    stratified_bootstrap_mean_ci(
                        differences_by_batch, salt=800 + grid_index
                    )
                ),
                "mean_pairwise_reduction": pair[
                    "mean_pairwise_reduction"
                ],
                "mean_pairwise_reduction_ci95": (
                    stratified_bootstrap_mean_ci(
                        reductions_by_batch, salt=900 + grid_index
                    )
                ),
                "three_expert_lower_n": pair["comparison_lower_count"],
                "three_expert_lower_fraction": pair[
                    "comparison_lower_fraction"
                ],
                "three_expert_lower_fraction_ci95": pair[
                    "comparison_lower_fraction_ci95"
                ],
                "sign_test_p": pair["exact_sign_test_p"],
            }
        )
    return output


def penalty_rows(
    batch_rollouts: dict[str, list[dict[str, str]]],
    pooled_rollouts: list[dict[str, str]],
) -> list[dict]:
    output: list[dict] = []
    controllers = ("active", "three_expert_selector")
    for batch_name in BATCH_NAMES:
        for row in single.penalty_components(
            batch_rollouts[batch_name], controllers
        ):
            output.append({"scope": batch_name, **row})
    for row in single.penalty_components(pooled_rollouts, controllers):
        output.append({"scope": "pooled", **row})
    return output


def terminal_reason_rows(
    batch_rollouts: dict[str, list[dict[str, str]]],
    pooled_rollouts: list[dict[str, str]],
) -> list[dict]:
    output: list[dict] = []
    controllers = ("active", "three_expert_selector")
    for batch_name in BATCH_NAMES:
        for row in single.terminal_reason_rows(
            batch_rollouts[batch_name], controllers
        ):
            output.append({"scope": batch_name, **row})
    for row in single.terminal_reason_rows(pooled_rollouts, controllers):
        output.append({"scope": "pooled", **row})
    return output


def expert_choice_summary(rows: list[dict[str, str]]) -> dict:
    return single.expert_choice_summary(rows)


def json_ready_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            key: json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # The single-batch helper's direct ``2**n`` conversion overflows once the
    # pooled discordant count becomes large.  Preserve the exact test while
    # using a log-scaled binomial tail for the five-batch analysis.
    single.exact_binomial_two_sided = exact_binomial_two_sided_stable

    batch_pairs: dict[str, list[dict[str, str]]] = {}
    batch_rollouts: dict[str, list[dict[str, str]]] = {}
    input_hashes: dict[str, str] = {}
    seen_global: set[tuple] = set()

    for batch_name in BATCH_NAMES:
        batch_dir = args.input_root / batch_name
        paired_path = batch_dir / "paired_trials.csv"
        rollout_path = batch_dir / "rollouts_active_vs_passive.csv"
        pairs = [
            row
            for row in read_rows(paired_path)
            if row["passive_controller"] == "three_expert_selector"
        ]
        validate_batch_pairs(pairs, batch_name)
        for row in pairs:
            row["_batch"] = batch_name
            globally_unique = (batch_name, *paired_key(row))
            if globally_unique in seen_global:
                raise RuntimeError(
                    f"Duplicate pooled pair key in {batch_name}: "
                    f"{paired_key(row)}"
                )
            seen_global.add(globally_unique)
        rollouts = read_rows(rollout_path)
        expected_controllers = Counter(row["controller"] for row in rollouts)
        for controller in ("active", "three_expert_selector"):
            if expected_controllers[controller] != EXPECTED_CASES_PER_BATCH:
                raise RuntimeError(
                    f"{batch_name}: expected {EXPECTED_CASES_PER_BATCH} "
                    f"{controller} rollouts, found "
                    f"{expected_controllers[controller]}"
                )
        for row in rollouts:
            row["_batch"] = batch_name
        batch_pairs[batch_name] = pairs
        batch_rollouts[batch_name] = rollouts
        input_hashes[f"{batch_name}/paired_trials.csv"] = sha256(paired_path)
        input_hashes[f"{batch_name}/rollouts_active_vs_passive.csv"] = (
            sha256(rollout_path)
        )

    pooled_pairs = [
        row for batch_name in BATCH_NAMES for row in batch_pairs[batch_name]
    ]
    pooled_rollouts = [
        row
        for batch_name in BATCH_NAMES
        for row in batch_rollouts[batch_name]
    ]
    expected_pooled = EXPECTED_CASES_PER_BATCH * len(BATCH_NAMES)
    if len(pooled_pairs) != expected_pooled:
        raise RuntimeError(
            f"Expected {expected_pooled} pooled matched pairs, "
            f"found {len(pooled_pairs)}"
        )

    per_batch = per_batch_rows(batch_pairs, batch_rollouts)
    pooled_pair = pooled_pair_statistics(pooled_pairs)
    pooled_controllers = {
        controller: controller_summary(
            pooled_rollouts, controller=controller
        )
        for controller in ("active", "three_expert_selector")
    }
    grids = grid_summary(pooled_pairs)
    penalties = penalty_rows(batch_rollouts, pooled_rollouts)
    terminals = terminal_reason_rows(batch_rollouts, pooled_rollouts)
    choices = expert_choice_summary(pooled_pairs)

    result = {
        "schema_version": 1,
        "batch_names": list(BATCH_NAMES),
        "cases_per_batch": EXPECTED_CASES_PER_BATCH,
        "pooled_case_count": expected_pooled,
        "per_batch": per_batch,
        "pooled_controllers": pooled_controllers,
        "pooled_active_vs_three_expert": pooled_pair,
        "pooled_grid_cells": grids,
        "pooled_three_expert_choices": choices,
        "penalty_components": penalties,
        "input_sha256": input_hashes,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "pooled_strata": "evaluation_batch",
            "grid_strata": "evaluation_batch_within_command_grid_cell",
        },
        "inference_boundary": (
            "The five batches vary randomized evaluation states while holding "
            "each controller checkpoint fixed. Confidence intervals therefore "
            "do not estimate independent-training-seed variability."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multibatch_statistics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_rows(
        args.output_dir / "per_batch_summary.csv",
        json_ready_rows(per_batch),
    )
    write_rows(
        args.output_dir / "pooled_controller_summary.csv",
        json_ready_rows(list(pooled_controllers.values())),
    )
    write_rows(
        args.output_dir / "pooled_pair_summary.csv",
        json_ready_rows([pooled_pair]),
    )
    write_rows(
        args.output_dir / "pooled_grid_summary.csv",
        json_ready_rows(grids),
    )
    write_rows(args.output_dir / "penalty_components.csv", penalties)
    write_rows(args.output_dir / "terminal_reason_summary.csv", terminals)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
