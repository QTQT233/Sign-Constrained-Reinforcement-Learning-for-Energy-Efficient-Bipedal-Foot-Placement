"""Build manuscript-ready tables and aggregate metrics from the frozen run.

This post-processing step does not rerun a controller or alter the frozen
acceptance analysis. It joins the immutable source-policy and hard-mask rows
to the route assignments in ``analysis_final/matched_trials_10800.csv.gz``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw"
ANALYSIS = ROOT / "results"
OUTPUT = ROOT / "reproduced_manuscript_metrics"
MATCHED = ANALYSIS / "matched_trials_10800.csv.gz"
REPORT = ANALYSIS / "final_confirmation_report.json"
THRESHOLDS_M = (0.001, 0.005, 0.010, 0.020)
CONTROLLERS = (
    "active",
    "hard_mask",
    "two_expert",
    "proposed_three_expert",
    "sign_oracle",
)
SOURCE_FOR_ROUTE = {
    "positive": "positive",
    "negative": "negative",
    "active": "active",
}


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def mean_sd(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "sample_sd": float(clean.std(ddof=1)),
        "median": float(clean.median()),
    }


def load_raw() -> pd.DataFrame:
    frames = []
    for batch_dir in sorted(RAW.glob("batch*")):
        for seed_dir in sorted(batch_dir.glob("seed_*")):
            active = pd.read_csv(
                seed_dir / "rollouts_active_vs_passive.csv"
            )
            active = active.loc[active["controller"] == "active"].copy()
            active["source"] = "active"
            signs = pd.read_csv(seed_dir / "passive_candidate_audit.csv")
            signs = signs.loc[
                signs["candidate_for_mode"].isin(["positive", "negative"])
            ].copy()
            signs["source"] = signs["candidate_for_mode"]
            hard = pd.read_csv(seed_dir / "hard_mask_replay.csv")
            hard["source"] = "hard_mask"
            for frame in (active, signs, hard):
                frame["batch"] = batch_dir.name
                frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["success"] = parse_bool(data["success"])
    data["valid_for_cmt"] = parse_bool(data["valid_for_cmt"])
    key = ["seed", "episode_index", "source"]
    if data.duplicated(key).any():
        raise RuntimeError("Duplicate raw controller rows")
    return data


def choose_oracle(group: pd.DataFrame) -> str:
    signs = group.loc[group["source"].isin(["positive", "negative"])]
    valid = signs.loc[signs["valid_for_cmt"]]
    if not valid.empty:
        return str(
            valid.sort_values(
                ["cmt", "source"], kind="stable"
            ).iloc[0]["source"]
        )
    successful = signs.loc[signs["success"]].copy()
    if not successful.empty:
        successful["energy_j"] = pd.to_numeric(
            successful["energy_j"], errors="coerce"
        )
        return str(
            successful.sort_values(
                ["energy_j", "source"], kind="stable", na_position="last"
            ).iloc[0]["source"]
        )
    return "positive"


def assemble_selected(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matched = pd.read_csv(MATCHED)
    keys = ["seed", "episode_index"]
    selected: dict[str, pd.DataFrame] = {}

    def select_by_source(name: str, source_series: pd.Series) -> None:
        request = matched[keys].copy()
        request["source"] = source_series.to_numpy()
        merged = request.merge(
            raw,
            on=["seed", "episode_index", "source"],
            how="left",
            validate="one_to_one",
        )
        if merged["cmt"].isna().all():
            raise RuntimeError(f"No selected rows for {name}")
        selected[name] = merged

    select_by_source(
        "active", pd.Series("active", index=matched.index)
    )
    select_by_source(
        "hard_mask", pd.Series("hard_mask", index=matched.index)
    )
    select_by_source(
        "two_expert",
        matched["released_two_expert_selector_route"],
    )
    select_by_source(
        "proposed_three_expert",
        matched["proposed_phase_aware_three_expert_route"],
    )
    oracle_sources = (
        raw.groupby(keys, sort=False, group_keys=False)
        .apply(choose_oracle, include_groups=False)
        .rename("source")
        .reset_index()
    )
    oracle = matched[keys].merge(
        oracle_sources, on=keys, how="left", validate="one_to_one"
    )
    oracle = oracle.merge(
        raw,
        on=["seed", "episode_index", "source"],
        how="left",
        validate="one_to_one",
    )
    selected["sign_oracle"] = oracle
    return selected


def summarize_controller(frame: pd.DataFrame) -> dict:
    valid = frame["valid_for_cmt"]
    n = len(frame)
    success_n = int(frame["success"].sum())
    valid_n = int(valid.sum())
    cmt = mean_sd(frame.loc[valid, "cmt"])
    result = {
        "n": n,
        "success_count": success_n,
        "success_rate": success_n / n,
        "success_wilson_95_ci": wilson(success_n, n),
        "valid_count": valid_n,
        "valid_rate": valid_n / n,
        "valid_wilson_95_ci": wilson(valid_n, n),
        "cmt": cmt,
        "component_mean_cmt": {
            "hip": float(frame.loc[valid, "hip_cmt"].mean()),
            "support_knee": float(
                frame.loc[valid, "support_knee_cmt"].mean()
            ),
            "swing_knee": float(
                frame.loc[valid, "swing_knee_cmt"].mean()
            ),
        },
        "terminal_reasons": dict(
            sorted(Counter(frame["terminal_reason"]).items())
        ),
        "route_counts": {
            source: int((frame["source"] == source).sum())
            for source in ("positive", "negative", "active", "hard_mask")
        },
        "route_outcomes": {},
    }
    for source in ("positive", "negative", "active", "hard_mask"):
        subset = frame["source"] == source
        result["route_outcomes"][source] = {
            "trials": int(subset.sum()),
            "success": int(frame.loc[subset, "success"].sum()),
            "valid": int(frame.loc[subset, "valid_for_cmt"].sum()),
        }
    return result


def summarize_pair(
    first_name: str,
    second_name: str,
    selected: dict[str, pd.DataFrame],
    frozen_pairs: dict[str, dict],
) -> dict:
    first = selected[first_name]
    second = selected[second_name]
    common = (
        first["valid_for_cmt"].to_numpy()
        & second["valid_for_cmt"].to_numpy()
    )
    first_values = first.loc[common, "cmt"].to_numpy(dtype=float)
    second_values = second.loc[common, "cmt"].to_numpy(dtype=float)
    delta = second_values - first_values
    first_success = first["success"].to_numpy(dtype=bool)
    second_success = second["success"].to_numpy(dtype=bool)
    result = {
        "first": first_name,
        "second": second_name,
        "success_difference_pp": float(
            100.0 * (second_success.mean() - first_success.mean())
        ),
        "first_only_success": int(
            np.sum(first_success & ~second_success)
        ),
        "second_only_success": int(
            np.sum(second_success & ~first_success)
        ),
        "common_valid_count": int(common.sum()),
        "common_valid_rate": float(common.mean()),
        "common_valid_wilson_95_ci": wilson(
            int(common.sum()), len(common)
        ),
        "common_fraction_of_first_valid": float(
            common.sum() / first["valid_for_cmt"].sum()
        ),
        "first_common_cmt": {
            **mean_sd(pd.Series(first_values)),
        },
        "second_common_cmt": {
            **mean_sd(pd.Series(second_values)),
        },
        "second_minus_first_mean_cmt": float(delta.mean()),
        "ratio_of_means_reduction": float(
            1.0 - second_values.mean() / first_values.mean()
        ),
        "second_lower_count": int(np.sum(delta < -1.0e-12)),
        "second_equal_count": int(np.sum(np.abs(delta) <= 1.0e-12)),
        "second_higher_count": int(np.sum(delta > 1.0e-12)),
        "second_lower_fraction": float(np.mean(delta < -1.0e-12)),
        "second_lower_wilson_95_ci": wilson(
            int(np.sum(delta < -1.0e-12)), int(common.sum())
        ),
    }
    frozen_key = {
        ("active", "proposed_three_expert"):
            "active_vs_proposed_phase_aware_three_expert",
        ("hard_mask", "proposed_three_expert"):
            "true_hard_mask_vs_proposed_phase_aware_three_expert",
        ("two_expert", "proposed_three_expert"):
            "released_two_expert_selector_vs_proposed_phase_aware_three_expert",
    }.get((first_name, second_name))
    if frozen_key:
        inference = frozen_pairs[frozen_key]
        for field in (
            "exact_two_sided_mcnemar_p",
            "success_difference_95_ci_pp",
            "success_difference_one_sided_95_lower_pp",
            "cmt_difference_95_ci",
            "ratio_of_means_reduction_95_ci",
        ):
            result[field] = inference[field]
    return result


def displacement_sensitivity(
    first: pd.DataFrame, second: pd.DataFrame
) -> list[dict]:
    output = []
    for threshold in THRESHOLDS_M:
        first_ok = (
            first["success"]
            & np.isfinite(first["cmt"])
            & np.isfinite(first["signed_com_displacement_m"])
            & (first["signed_com_displacement_m"] > threshold)
        )
        second_ok = (
            second["success"]
            & np.isfinite(second["cmt"])
            & np.isfinite(second["signed_com_displacement_m"])
            & (second["signed_com_displacement_m"] > threshold)
        )
        common = first_ok.to_numpy() & second_ok.to_numpy()
        first_values = first.loc[common, "cmt"].to_numpy(dtype=float)
        second_values = second.loc[common, "cmt"].to_numpy(dtype=float)
        lower = int(np.sum(second_values < first_values - 1.0e-12))
        output.append(
            {
                "minimum_displacement_m": threshold,
                "matched_pairs": int(common.sum()),
                "first_mean_cmt": float(first_values.mean()),
                "second_mean_cmt": float(second_values.mean()),
                "second_lower_count": lower,
                "second_lower_fraction": lower / int(common.sum()),
            }
        )
    return output


def three_way_common(
    selected: dict[str, pd.DataFrame]
) -> dict:
    names = ("active", "hard_mask", "proposed_three_expert")
    common = np.logical_and.reduce(
        [selected[name]["valid_for_cmt"].to_numpy() for name in names]
    )
    return {
        "controllers": list(names),
        "common_valid_count": int(common.sum()),
        "common_valid_rate": float(common.mean()),
        "cmt": {
            name: mean_sd(selected[name].loc[common, "cmt"])
            for name in names
        },
    }


def cell_table(
    selected: dict[str, pd.DataFrame]
) -> list[dict]:
    active = selected["active"]
    proposed = selected["proposed_three_expert"]
    rows = []
    for length in (0.30, 0.35, 0.40):
        for height in (0.00, 0.01, 0.02):
            mask = np.isclose(
                active["commanded_step_length_m"], length
            ) & np.isclose(active["commanded_step_height_m"], height)
            av = active.loc[mask, "valid_for_cmt"].to_numpy()
            pv = proposed.loc[mask, "valid_for_cmt"].to_numpy()
            common = av & pv
            rows.append(
                {
                    "length_m": length,
                    "height_m": height,
                    "n": int(mask.sum()),
                    "active_success_rate": float(
                        active.loc[mask, "success"].mean()
                    ),
                    "proposed_success_rate": float(
                        proposed.loc[mask, "success"].mean()
                    ),
                    "active_valid_rate": float(av.mean()),
                    "proposed_valid_rate": float(pv.mean()),
                    "common_valid_count": int(common.sum()),
                    "active_common_mean_cmt": float(
                        active.loc[mask].loc[common, "cmt"].mean()
                    ),
                    "proposed_common_mean_cmt": float(
                        proposed.loc[mask].loc[common, "cmt"].mean()
                    ),
                }
            )
    return rows


def batch_table(
    selected: dict[str, pd.DataFrame]
) -> list[dict]:
    active = selected["active"]
    proposed = selected["proposed_three_expert"]
    hard = selected["hard_mask"]
    rows = []
    for batch in sorted(active["batch"].unique()):
        mask = active["batch"] == batch
        av = active.loc[mask, "valid_for_cmt"].to_numpy()
        pv = proposed.loc[mask, "valid_for_cmt"].to_numpy()
        hv = hard.loc[mask, "valid_for_cmt"].to_numpy()
        common_active = av & pv
        common_hard = hv & pv
        rows.append(
            {
                "batch": batch,
                "n": int(mask.sum()),
                "active_success_rate": float(
                    active.loc[mask, "success"].mean()
                ),
                "proposed_success_rate": float(
                    proposed.loc[mask, "success"].mean()
                ),
                "hard_mask_success_rate": float(
                    hard.loc[mask, "success"].mean()
                ),
                "proposed_minus_active_success_pp": float(
                    100.0
                    * (
                        proposed.loc[mask, "success"].mean()
                        - active.loc[mask, "success"].mean()
                    )
                ),
                "common_valid_active_proposed": int(
                    common_active.sum()
                ),
                "active_common_mean_cmt": float(
                    active.loc[mask].loc[common_active, "cmt"].mean()
                ),
                "proposed_active_common_mean_cmt": float(
                    proposed.loc[mask]
                    .loc[common_active, "cmt"]
                    .mean()
                ),
                "common_valid_hard_proposed": int(common_hard.sum()),
                "hard_common_mean_cmt": float(
                    hard.loc[mask].loc[common_hard, "cmt"].mean()
                ),
                "proposed_hard_common_mean_cmt": float(
                    proposed.loc[mask].loc[common_hard, "cmt"].mean()
                ),
            }
        )
    return rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    selected = assemble_selected(raw)
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    controllers = {
        name: summarize_controller(frame)
        for name, frame in selected.items()
    }
    pair_specs = (
        ("active", "proposed_three_expert"),
        ("hard_mask", "proposed_three_expert"),
        ("two_expert", "proposed_three_expert"),
        ("active", "hard_mask"),
        ("active", "two_expert"),
        ("active", "sign_oracle"),
    )
    pairs = {
        f"{first}_vs_{second}": summarize_pair(
            first, second, selected, frozen["pairs"]
        )
        for first, second in pair_specs
    }
    report = {
        "schema":
            "phase-aware-three-expert-five-batch-manuscript-metrics/v1",
        "n_cases": 10800,
        "seeds": list(range(20263101, 20263116)),
        "proposed_rule": frozen["protocol"]["proposed_rule"],
        "controllers": controllers,
        "pairs": pairs,
        "active_vs_proposed_displacement_sensitivity":
            displacement_sensitivity(
                selected["active"],
                selected["proposed_three_expert"],
            ),
        "three_way_common_valid": three_way_common(selected),
        "cells": cell_table(selected),
        "batches": batch_table(selected),
        "frozen_acceptance": frozen["primary_acceptance"],
        "bootstrap": frozen["bootstrap"],
    }
    (OUTPUT / "five_batch_manuscript_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(report["cells"]).to_csv(
        OUTPUT / "table_grid_cells.csv", index=False
    )
    pd.DataFrame(report["batches"]).to_csv(
        OUTPUT / "table_five_batches.csv", index=False
    )
    rows = []
    for name, values in controllers.items():
        row = {"controller": name}
        row.update(
            {
                key: value
                for key, value in values.items()
                if key
                in {
                    "n",
                    "success_count",
                    "success_rate",
                    "valid_count",
                    "valid_rate",
                }
            }
        )
        row.update(
            {
                "mean_cmt": values["cmt"]["mean"],
                "sample_sd_cmt": values["cmt"]["sample_sd"],
                "median_cmt": values["cmt"]["median"],
                **{
                    f"mean_{key}": value
                    for key, value in values[
                        "component_mean_cmt"
                    ].items()
                },
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        OUTPUT / "table_controller_summary.csv", index=False
    )
    print(
        json.dumps(
            {
                "three_way_common_valid":
                    report["three_way_common_valid"],
                "controllers": {
                    name: {
                        "success_count": values["success_count"],
                        "valid_count": values["valid_count"],
                        "mean_cmt": values["cmt"]["mean"],
                    }
                    for name, values in controllers.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
