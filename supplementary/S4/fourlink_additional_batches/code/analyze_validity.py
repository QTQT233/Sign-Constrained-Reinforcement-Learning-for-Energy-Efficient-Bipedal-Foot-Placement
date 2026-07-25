"""Pool fresh frozen-policy batches under two Cmt-validity definitions.

The analysis always starts from controller-level rollout records.  Failure
trajectories retain their diagnostic energy/displacement fields but are never
assigned a Cmt value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ("active", "passive_sign_selector")
BOOTSTRAP_SEED = 2026072504
BOOTSTRAP_REPLICATES = 4000
MASS_TIMES_GRAVITY_N = 24.95276


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def load_batch(batch_dir: Path, mass_g: float) -> pd.DataFrame:
    path = batch_dir / "rollouts_active_vs_passive.csv"
    rows = pd.read_csv(path)
    rows = rows.loc[rows["controller"].isin(CONTROLLERS)].copy()
    rows["success"] = coerce_bool(rows["success"])
    rows["valid_for_cmt"] = coerce_bool(rows["valid_for_cmt"])
    rows["batch"] = batch_dir.name
    rows["finite_energy"] = np.isfinite(rows["energy_j"].to_numpy(float))
    rows["finite_displacement"] = np.isfinite(
        rows["signed_com_displacement_m"].to_numpy(float)
    )
    rows["computed_cmt"] = (
        rows["energy_j"].to_numpy(float)
        / (mass_g * rows["signed_com_displacement_m"].to_numpy(float))
    )
    rows["finite_computed_cmt"] = np.isfinite(rows["computed_cmt"])
    base = (
        rows["success"]
        & rows["finite_energy"]
        & rows["finite_displacement"]
        & rows["finite_computed_cmt"]
        & (rows["steps"] >= 1)
        & (rows["signed_com_displacement_m"] > 0.001)
    )
    rows["valid_B"] = base
    rows["valid_A"] = base & (rows["nonzero_torque_steps"] >= 1)
    rows["cmt_A"] = rows["computed_cmt"].where(rows["valid_A"])
    rows["cmt_B"] = rows["computed_cmt"].where(rows["valid_B"])
    return rows


def make_matched(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "batch",
        "seed",
        "episode_index",
        "walk_direction",
        "reset_phase",
        "commanded_step_length_m",
        "commanded_step_height_m",
    ]
    duplicates = rows.duplicated(keys + ["controller"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Non-unique controller records:\n"
            + rows.loc[duplicates, keys + ["controller"]].to_string(index=False)
        )
    value_columns = [
        "success",
        "valid_for_cmt",
        "valid_A",
        "valid_B",
        "cmt",
        "cmt_A",
        "cmt_B",
        "computed_cmt",
        "energy_j",
        "signed_com_displacement_m",
        "steps",
        "nonzero_torque_steps",
        "terminal_reason",
        "initial_q1_deg",
        "initial_q2_deg",
        "initial_q3_deg",
        "initial_q4_deg",
        "initial_dq1_rad_s",
        "initial_dq2_rad_s",
        "initial_dq3_rad_s",
        "initial_dq4_rad_s",
    ]
    pieces = []
    for controller in CONTROLLERS:
        piece = rows.loc[rows["controller"].eq(controller), keys + value_columns].copy()
        rename = {column: f"{controller}_{column}" for column in value_columns}
        pieces.append(piece.rename(columns=rename))
    matched = pieces[0].merge(pieces[1], on=keys, how="outer", validate="one_to_one")
    if len(matched) != len(pieces[0]) or matched.isna().all(axis=1).any():
        raise RuntimeError("Controller pairing failed.")
    return matched.sort_values(["batch", "seed", "episode_index"]).reset_index(drop=True)


def metric_values(records: pd.DataFrame, validity: str) -> dict[str, float]:
    active_success = records["active_success"].to_numpy(bool)
    selector_success = records["passive_sign_selector_success"].to_numpy(bool)
    active_valid = records[f"active_valid_{validity}"].to_numpy(bool)
    selector_valid = records[
        f"passive_sign_selector_valid_{validity}"
    ].to_numpy(bool)
    both = active_valid & selector_valid
    active_cmt = records[f"active_cmt_{validity}"].to_numpy(float)
    selector_cmt = records[
        f"passive_sign_selector_cmt_{validity}"
    ].to_numpy(float)
    deltas = selector_cmt[both] - active_cmt[both]
    return {
        "n_pairs": int(len(records)),
        "active_success_n": int(active_success.sum()),
        "selector_success_n": int(selector_success.sum()),
        "active_success_rate": float(active_success.mean()),
        "selector_success_rate": float(selector_success.mean()),
        "selector_minus_active_success_rate": float(
            selector_success.mean() - active_success.mean()
        ),
        "active_valid_n": int(active_valid.sum()),
        "selector_valid_n": int(selector_valid.sum()),
        "active_valid_rate": float(active_valid.mean()),
        "selector_valid_rate": float(selector_valid.mean()),
        "selector_minus_active_valid_rate": float(
            selector_valid.mean() - active_valid.mean()
        ),
        "both_valid_n": int(both.sum()),
        "both_valid_rate": float(both.mean()),
        "active_mean_cmt_both_valid": (
            float(np.mean(active_cmt[both])) if both.any() else float("nan")
        ),
        "selector_mean_cmt_both_valid": (
            float(np.mean(selector_cmt[both])) if both.any() else float("nan")
        ),
        "mean_selector_minus_active_cmt": (
            float(np.mean(deltas)) if len(deltas) else float("nan")
        ),
        "median_selector_minus_active_cmt": (
            float(np.median(deltas)) if len(deltas) else float("nan")
        ),
        "fraction_selector_lower_cmt": (
            float(np.mean(deltas < 0.0)) if len(deltas) else float("nan")
        ),
        "zero_torque_success_displacement_active_n": int(
            (
                active_success
                & (records["active_steps"].to_numpy(int) >= 1)
                & (
                    records["active_signed_com_displacement_m"].to_numpy(float)
                    > 0.001
                )
                & (records["active_nonzero_torque_steps"].to_numpy(int) < 1)
            ).sum()
        ),
        "zero_torque_success_displacement_selector_n": int(
            (
                selector_success
                & (
                    records[
                        "passive_sign_selector_steps"
                    ].to_numpy(int)
                    >= 1
                )
                & (
                    records[
                        "passive_sign_selector_signed_com_displacement_m"
                    ].to_numpy(float)
                    > 0.001
                )
                & (
                    records[
                        "passive_sign_selector_nonzero_torque_steps"
                    ].to_numpy(int)
                    < 1
                )
            ).sum()
        ),
    }


def stratified_bootstrap_ci(
    records: pd.DataFrame,
    validity: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    records = records.reset_index(drop=True)
    strata_columns = [
        "seed",
        "walk_direction",
        "commanded_step_length_m",
        "commanded_step_height_m",
    ]
    groups = [
        np.asarray(index, dtype=int)
        for index in records.groupby(strata_columns, sort=True).indices.values()
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if validity == "A" else 1))
    names = [
        "selector_minus_active_success_rate",
        "selector_minus_active_valid_rate",
        "both_valid_rate",
        "mean_selector_minus_active_cmt",
        "fraction_selector_lower_cmt",
    ]
    samples = {name: np.empty(replicates, dtype=float) for name in names}
    for replicate in range(replicates):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        values = metric_values(records.iloc[indices], validity)
        for name in names:
            samples[name][replicate] = values[name]
    output: dict[str, float] = {}
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        output[f"{name}_ci95_low"] = (
            float(np.quantile(finite, 0.025)) if len(finite) else float("nan")
        )
        output[f"{name}_ci95_high"] = (
            float(np.quantile(finite, 0.975)) if len(finite) else float("nan")
        )
    return output


def summarize(
    records: pd.DataFrame, scope: str, validity: str, include_ci: bool = True
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "scope": scope,
        "validity_definition": validity,
    }
    row.update(metric_values(records, validity))
    if include_ci:
        row.update(stratified_bootstrap_ci(records, validity))
    return row


def stream_rows(records: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (batch, seed), subset in records.groupby(["batch", "seed"], sort=True):
        for validity in ("A", "B"):
            row = {
                "batch": batch,
                "seed": int(seed),
                "validity_definition": validity,
            }
            row.update(metric_values(subset, validity))
            both = (
                subset[f"active_valid_{validity}"].to_numpy(bool)
                & subset[
                    f"passive_sign_selector_valid_{validity}"
                ].to_numpy(bool)
            )
            deltas = (
                subset.loc[both, f"passive_sign_selector_cmt_{validity}"].to_numpy(
                    float
                )
                - subset.loc[both, f"active_cmt_{validity}"].to_numpy(float)
            )
            row["paired_delta_variance"] = (
                float(np.var(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
            )
            row["paired_delta_standard_error"] = (
                float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
                if len(deltas) > 1
                else float("nan")
            )
            output.append(row)
    return pd.DataFrame(output)


def heterogeneity(streams: pd.DataFrame, validity: str) -> dict[str, float | int | str]:
    subset = streams.loc[streams["validity_definition"].eq(validity)].copy()
    means = subset["mean_selector_minus_active_cmt"].to_numpy(float)
    variances = (
        subset["paired_delta_variance"].to_numpy(float)
        / subset["both_valid_n"].to_numpy(float)
    )
    finite = np.isfinite(means) & np.isfinite(variances) & (variances > 0.0)
    means = means[finite]
    variances = variances[finite]
    weights = 1.0 / variances
    fixed_mean = float(np.sum(weights * means) / np.sum(weights))
    q_value = float(np.sum(weights * (means - fixed_mean) ** 2))
    df = max(0, len(means) - 1)
    i2 = (
        float(max(0.0, (q_value - df) / q_value) * 100.0)
        if q_value > 0 and df > 0
        else float("nan")
    )
    return {
        "validity_definition": validity,
        "n_seed_streams": int(len(means)),
        "unweighted_stream_mean_delta_cmt": float(np.mean(means)),
        "stream_mean_delta_cmt_sd": (
            float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
        ),
        "stream_mean_delta_cmt_min": float(np.min(means)),
        "stream_mean_delta_cmt_max": float(np.max(means)),
        "inverse_variance_fixed_mean_delta_cmt": fixed_mean,
        "cochran_Q": q_value,
        "cochran_Q_df": int(df),
        "cochran_Q_p": float(chi2.sf(q_value, df)) if df > 0 else float("nan"),
        "I2_percent": i2,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched-records",
        type=Path,
        default=ROOT / "results" / "matched_records.csv",
        help="Published 10,800-row matched record table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_generated",
        help="Directory for recomputed summaries; published results are read-only.",
    )
    args = parser.parse_args()

    if not args.matched_records.is_file():
        raise SystemExit(f"Missing matched records: {args.matched_records}")
    matched = pd.read_csv(args.matched_records)
    bool_columns = [
        column
        for column in matched.columns
        if column.endswith(("_success", "_valid_for_cmt", "_valid_A", "_valid_B"))
    ]
    for column in bool_columns:
        matched[column] = coerce_bool(matched[column])
    expected_batches = {"batch01", "batch02", "batch03", "batch04", "batch05"}
    observed_batches = set(matched["batch"].astype(str))
    if observed_batches != expected_batches:
        raise RuntimeError(
            f"Expected batches {sorted(expected_batches)}, found "
            f"{sorted(observed_batches)}"
        )
    if len(matched) != 10_800:
        raise RuntimeError(f"Expected 10,800 matched records, found {len(matched)}")

    summary_rows = []
    for batch, subset in matched.groupby("batch", sort=True):
        for validity in ("A", "B"):
            summary_rows.append(summarize(subset, batch, validity))
    for validity in ("A", "B"):
        summary_rows.append(summarize(matched, "pooled_raw_pairs", validity))
    summaries = pd.DataFrame(summary_rows)

    streams = stream_rows(matched)
    heterogeneity_rows = pd.DataFrame(
        [heterogeneity(streams, validity) for validity in ("A", "B")]
    )
    cell_rows = []
    cell_columns = [
        "walk_direction",
        "commanded_step_length_m",
        "commanded_step_height_m",
    ]
    for cell_key, subset in matched.groupby(cell_columns, sort=True):
        cell_scope = (
            f"direction={cell_key[0]:+.0f};"
            f"length={cell_key[1]:.2f};height={cell_key[2]:.2f}"
        )
        for validity in ("A", "B"):
            cell_rows.append(
                summarize(subset, cell_scope, validity, include_ci=False)
            )
    cell_summary = pd.DataFrame(cell_rows)
    pooling_rows = []
    for validity in ("A", "B"):
        batch_rows = summaries.loc[
            (summaries["validity_definition"] == validity)
            & (summaries["scope"] != "pooled_raw_pairs")
        ]
        pooled_row = summaries.loc[
            (summaries["validity_definition"] == validity)
            & (summaries["scope"] == "pooled_raw_pairs")
        ].iloc[0]
        weights = batch_rows["both_valid_n"].to_numpy(float)
        batch_means = batch_rows[
            "mean_selector_minus_active_cmt"
        ].to_numpy(float)
        pooling_rows.append(
            {
                "validity_definition": validity,
                "raw_record_pooled_mean_delta_cmt": pooled_row[
                    "mean_selector_minus_active_cmt"
                ],
                "both_valid_n_weighted_batch_mean_delta_cmt": float(
                    np.average(batch_means, weights=weights)
                ),
                "equal_batch_mean_delta_cmt_diagnostic_only": float(
                    np.mean(batch_means)
                ),
                "pooled_both_valid_n": int(weights.sum()),
            }
        )
    pooling_check = pd.DataFrame(pooling_rows)

    analysis_dir = args.output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(analysis_dir / "matched_records.csv", index=False)
    summaries.to_csv(analysis_dir / "batch_and_pooled_summary.csv", index=False)
    streams.to_csv(analysis_dir / "seed_stream_summary.csv", index=False)
    cell_summary.to_csv(analysis_dir / "command_cell_summary.csv", index=False)
    heterogeneity_rows.to_csv(
        analysis_dir / "seed_stream_heterogeneity.csv", index=False
    )
    pooling_check.to_csv(analysis_dir / "pooling_check.csv", index=False)

    checks = {
        "mass_times_gravity_N": MASS_TIMES_GRAVITY_N,
        "n_batches": int(matched["batch"].nunique()),
        "n_seed_streams": int(matched["seed"].nunique()),
        "n_matched_records": int(len(matched)),
        "duplicate_pair_keys": int(
            matched.duplicated(["batch", "seed", "episode_index"]).sum()
        ),
        "failures_with_assigned_cmt_A": int(
            (
                (~matched["active_success"] & matched["active_cmt_A"].notna())
                | (
                    ~matched["passive_sign_selector_success"]
                    & matched["passive_sign_selector_cmt_A"].notna()
                )
            ).sum()
        ),
        "failures_with_assigned_cmt_B": int(
            (
                (~matched["active_success"] & matched["active_cmt_B"].notna())
                | (
                    ~matched["passive_sign_selector_success"]
                    & matched["passive_sign_selector_cmt_B"].notna()
                )
            ).sum()
        ),
        "validity_A_B_changed_active_records": int(
            (
                matched["active_valid_A"] != matched["active_valid_B"]
            ).sum()
        ),
        "validity_A_B_changed_selector_records": int(
            (
                matched["passive_sign_selector_valid_A"]
                != matched["passive_sign_selector_valid_B"]
            ).sum()
        ),
        "source_validity_disagrees_with_recomputed_A_active": int(
            (
                matched["active_valid_for_cmt"]
                != matched["active_valid_A"]
            ).sum()
        ),
        "source_validity_disagrees_with_recomputed_A_selector": int(
            (
                matched["passive_sign_selector_valid_for_cmt"]
                != matched["passive_sign_selector_valid_A"]
            ).sum()
        ),
        "max_abs_source_vs_recomputed_cmt_A_active": float(
            np.nanmax(
                np.abs(
                    matched["active_cmt"].to_numpy(float)
                    - matched["active_cmt_A"].to_numpy(float)
                )
            )
        ),
        "max_abs_source_vs_recomputed_cmt_A_selector": float(
            np.nanmax(
                np.abs(
                    matched["passive_sign_selector_cmt"].to_numpy(float)
                    - matched[
                        "passive_sign_selector_cmt_A"
                    ].to_numpy(float)
                )
            )
        ),
        "pooling_method": (
            "Raw matched records are concatenated before analysis; conditional "
            "Cmt means are weighted by the actual both-valid record count, not "
            "by equally averaging batch means."
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "strata": [
                "seed",
                "walk_direction",
                "commanded_step_length_m",
                "commanded_step_height_m",
            ],
        },
    }
    (analysis_dir / "validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    outputs = sorted(
        path for path in analysis_dir.rglob("*") if path.is_file()
    )
    with (analysis_dir / "CHECKSUMS.sha256").open("w", encoding="utf-8") as stream:
        for path in outputs:
            if path.name == "CHECKSUMS.sha256":
                continue
            stream.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")

    print(summaries.to_string(index=False))
    print("\nSeed-stream heterogeneity")
    print(heterogeneity_rows.to_string(index=False))
    print("\nValidation")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
