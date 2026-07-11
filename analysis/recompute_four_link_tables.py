"""Recompute manuscript-facing four-link statistics from frozen trial CSVs.

This is an external, read-only analysis.  It does not import or modify the
training/evaluation programs.  All resampling is paired and deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


Z95 = 1.959963984540054
Z90 = 1.6448536269514722
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000


def wilson(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return center - half, center + half


def paired_binary(a: np.ndarray, b: np.ndarray) -> dict[str, float | int | list[float]]:
    """Return paired binary inference for difference b - a."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)
    both = int(np.sum(a & b))
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    d = (b_only - a_only) / n
    # Variance of the sample mean of paired Bernoulli differences.
    q = (a_only + b_only) / n
    se = math.sqrt(max(0.0, (q - d * d) / n))
    discordant = a_only + b_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    return {
        "n": n,
        "both_true": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "rate_a": float(np.mean(a)),
        "rate_b": float(np.mean(b)),
        "difference_b_minus_a": d,
        "standard_error": se,
        "ci95": [d - Z95 * se, d + Z95 * se],
        "ci90": [d - Z90 * se, d + Z90 * se],
        "one_sided_95_lower": d - Z90 * se,
        "exact_mcnemar_two_sided_p": mcnemar_p,
    }


def percentile_bootstrap(matrix: np.ndarray, seed: int = BOOTSTRAP_SEED) -> list[list[float]]:
    matrix = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, matrix.shape[0], size=(BOOTSTRAP_REPLICATES, matrix.shape[0]))
    means = matrix[index].mean(axis=1)
    q = np.percentile(means, [2.5, 97.5], axis=0)
    return [[float(q[0, i]), float(q[1, i])] for i in range(matrix.shape[1])]


def cmt_pair_stats(df: pd.DataFrame, left: str, right: str, left_valid: str, right_valid: str) -> dict:
    both = df[df[left_valid].astype(bool) & df[right_valid].astype(bool)].copy()
    a = both[left].to_numpy(float)
    b = both[right].to_numpy(float)
    active_minus_comparison = a - b
    pairwise_reduction = (a - b) / a
    lower = (b < a).astype(float)
    boot = percentile_bootstrap(np.column_stack([active_minus_comparison, pairwise_reduction]))
    low_ci = wilson(int(lower.sum()), len(lower))
    return {
        "both_valid_pairs": len(both),
        "both_valid_rate": len(both) / len(df),
        "left_mean": float(np.mean(a)),
        "right_mean": float(np.mean(b)),
        "left_median": float(np.median(a)),
        "right_median": float(np.median(b)),
        "mean_left_minus_right": float(np.mean(active_minus_comparison)),
        "mean_left_minus_right_bootstrap95": boot[0],
        "mean_pairwise_reduction": float(np.mean(pairwise_reduction)),
        "mean_pairwise_reduction_bootstrap95": boot[1],
        "fraction_right_lower": float(np.mean(lower)),
        "fraction_right_lower_wilson95": [float(low_ci[0]), float(low_ci[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    controller = pd.read_csv(args.input_dir / "controller_summary.csv")
    paired_all = pd.read_csv(args.input_dir / "paired_trials.csv")
    active_ref = pd.read_csv(args.input_dir / "active_reference_comparison_trials.csv")
    selector_oracle = pd.read_csv(args.input_dir / "selector_vs_oracle_trials.csv")
    terminal = pd.read_csv(args.input_dir / "terminal_reason_summary.csv")

    controller_rows = []
    for row in controller.to_dict("records"):
        n = int(row["n_all"])
        ns = int(row["n_success"])
        nv = int(row["n_valid_cmt"])
        s_ci = wilson(ns, n)
        v_ci = wilson(nv, n)
        controller_rows.append({
            "controller": row["controller"],
            "n": n,
            "n_success": ns,
            "success_rate": ns / n,
            "success_wilson95_low": s_ci[0],
            "success_wilson95_high": s_ci[1],
            "n_valid_cmt": nv,
            "valid_rate": nv / n,
            "valid_wilson95_low": v_ci[0],
            "valid_wilson95_high": v_ci[1],
            "mean_cmt_valid": row["mean_cmt_valid"],
            "median_cmt_valid": row["median_cmt_valid"],
            "mean_support_knee_cmt_valid": row["mean_support_knee_cmt_valid"],
            "mean_hip_cmt_valid": row["mean_hip_cmt_valid"],
            "mean_swing_knee_cmt_valid": row["mean_swing_knee_cmt_valid"],
        })
    pd.DataFrame(controller_rows).to_csv(args.output_dir / "table_vii_controller_summary.csv", index=False)

    selector = paired_all[paired_all["passive_controller"] == "passive_sign_selector"].copy()
    oracle = paired_all[paired_all["passive_controller"] == "passive_oracle_envelope"].copy()
    selector_success = paired_binary(selector["active_success"], selector["passive_success"])
    selector_validity = paired_binary(selector["active_valid"], selector["passive_valid"])
    selector_cmt = cmt_pair_stats(selector, "active_cmt", "passive_cmt", "active_valid", "passive_valid")
    oracle_cmt = cmt_pair_stats(oracle, "active_cmt", "passive_cmt", "active_valid", "passive_valid")

    pair_table = []
    for label, stats in (("active_vs_online_selector", selector_cmt), ("active_vs_oracle", oracle_cmt)):
        pair_table.append({"pair": label, **stats})
    pd.DataFrame(pair_table).to_csv(args.output_dir / "table_viii_paired_cmt.csv", index=False)

    reference_rows = []
    for name, group in active_ref.groupby("comparison_controller", sort=True):
        cmt = cmt_pair_stats(group, "reference_cmt", "comparison_cmt", "reference_valid", "comparison_valid")
        succ = paired_binary(group["reference_success"], group["comparison_success"])
        valid = paired_binary(group["reference_valid"], group["comparison_valid"])
        reference_rows.append({"comparison_controller": name, **cmt,
                               "success_difference_comparison_minus_reference": succ["difference_b_minus_a"],
                               "valid_difference_comparison_minus_reference": valid["difference_b_minus_a"]})

    so = selector_oracle.copy()
    match_n = int(so["selector_matches_oracle_sign"].astype(bool).sum())
    oracle_success_mask = so["oracle_success"].astype(bool)
    oracle_valid_mask = so["oracle_valid"].astype(bool)
    so_both_valid = so[so["selector_valid"].astype(bool) & so["oracle_valid"].astype(bool)].copy()
    regret = so_both_valid["selector_minus_oracle_cmt"].to_numpy(float)
    regret_ci = percentile_bootstrap(regret.reshape(-1, 1))[0]
    oracle_metrics = {
        "selector_sign_match_count": match_n,
        "selector_sign_match_rate": match_n / len(so),
        "selector_sign_match_wilson95": list(wilson(match_n, len(so))),
        "selector_success_given_oracle_success_count": int(so.loc[oracle_success_mask, "selector_success"].astype(bool).sum()),
        "oracle_success_count": int(oracle_success_mask.sum()),
        "selector_success_given_oracle_success_rate": float(so.loc[oracle_success_mask, "selector_success"].astype(bool).mean()),
        "selector_valid_given_oracle_valid_count": int(so.loc[oracle_valid_mask, "selector_valid"].astype(bool).sum()),
        "oracle_valid_count": int(oracle_valid_mask.sum()),
        "selector_valid_given_oracle_valid_rate": float(so.loc[oracle_valid_mask, "selector_valid"].astype(bool).mean()),
        "both_valid_selector_oracle_pairs": len(so_both_valid),
        "mean_selector_minus_oracle_cmt": float(np.mean(regret)),
        "mean_selector_minus_oracle_cmt_bootstrap95": regret_ci,
        "median_selector_minus_oracle_cmt": float(np.median(regret)),
    }
    pd.DataFrame(reference_rows).to_csv(args.output_dir / "table_ix_active_reference_diagnostics.csv", index=False)
    (args.output_dir / "table_ix_selector_oracle.json").write_text(json.dumps(oracle_metrics, indent=2), encoding="utf-8")

    grid_rows = []
    for (length, height), group in selector.groupby(["commanded_step_length_m", "commanded_step_height_m"], sort=True):
        cmt = cmt_pair_stats(group, "active_cmt", "passive_cmt", "active_valid", "passive_valid")
        active_success_n = int(group["active_success"].astype(bool).sum())
        selector_success_n = int(group["passive_success"].astype(bool).sum())
        active_valid_n = int(group["active_valid"].astype(bool).sum())
        selector_valid_n = int(group["passive_valid"].astype(bool).sum())
        grid_rows.append({
            "step_length_m": length,
            "step_height_m": height,
            "n": len(group),
            "active_success_rate": active_success_n / len(group),
            "selector_success_rate": selector_success_n / len(group),
            "active_valid_rate": active_valid_n / len(group),
            "selector_valid_rate": selector_valid_n / len(group),
            **cmt,
        })
    pd.DataFrame(grid_rows).to_csv(args.output_dir / "table_x_grid_summary.csv", index=False)

    terminal_exact = terminal[terminal["reason_group"].isin(["failure_exact_terminal_reason", "success_terminal_reason"])].copy()
    terminal_exact.to_csv(args.output_dir / "table_xi_terminal_reasons_long.csv", index=False)

    all_stats = {
        "dataset": str(args.input_dir),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "selector_success": selector_success,
        "selector_validity": selector_validity,
        "selector_cmt": selector_cmt,
        "oracle_cmt": oracle_cmt,
        "selector_oracle": oracle_metrics,
        "limitations": [
            "Inference is conditional on one fixed checkpoint per controller.",
            "The three RNG seeds generate evaluation cases and are not independent training seeds.",
            "Non-inferiority margins are sensitivity analyses unless prespecified on a locked test set.",
        ],
    }
    (args.output_dir / "four_link_statistics.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
