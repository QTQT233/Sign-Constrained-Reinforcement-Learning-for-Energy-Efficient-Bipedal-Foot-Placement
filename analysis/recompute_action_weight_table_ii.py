"""Recompute the manuscript Table II action-weight comparison.

The script is intentionally read-only with respect to the evaluator tree.  It
loads the 12 fixed-seed HDF5 result cells produced by the canonical
``Whole_energy_comparison_low_dim.py`` programs, validates the expert-fusion
logic, and writes both an audit table and the manuscript-facing table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SEED = 20260716
MIN_COM_DISPLACEMENT_M = 0.01
ROBOT_WEIGHT_N = (1.4122 + 0.0839) * 9.81
WEIGHTS = ("0.02", "0.04", "0.06")
CONDITIONS = (
    {
        "directory": "60,30(0.33m)",
        "step_height_m": 0.00,
        "max_horizontal_advance_m": 0.25,
        "paper_column": "0.00 m step; 0.25 m advance",
        "paper_order": 0,
    },
    {
        "directory": "70,20(0.225m)",
        "step_height_m": 0.00,
        "max_horizontal_advance_m": 0.17,
        "paper_column": "0.00 m step; 0.17 m advance",
        "paper_order": 1,
    },
    {
        "directory": "60,30(0.33m),0.01m",
        "step_height_m": 0.01,
        "max_horizontal_advance_m": 0.25,
        "paper_column": "0.01 m step; 0.25 m advance",
        "paper_order": 2,
    },
    {
        "directory": "70,20,(0.225m),0.01m",
        "step_height_m": 0.01,
        "max_horizontal_advance_m": 0.17,
        "paper_column": "0.01 m step; 0.17 m advance",
        "paper_order": 3,
    },
)

ARRAY_FILES = {
    "negative_status": ("working_save(-1,0)-10-30", "working_save"),
    "negative_cmt": ("Cmt_save(-1,0)-10-30", "Cmt_save"),
    "negative_energy": ("Energy_save(-1,0)-10-30", "Energy_save"),
    "positive_status": ("working_save(0,1)-10-30", "working_save"),
    "positive_cmt": ("Cmt_save(0,1)-10-30", "Cmt_save"),
    "positive_energy": ("Energy_save(0,1)-10-30", "Energy_save"),
    "proposed_status": ("working_save_passive-10-30", "working_save_passive"),
    "proposed_cmt": ("Cmt_save_passive-10-30", "Cmt_save_passive"),
    "active_status": (
        "working_save_active_discrete-10-30",
        "working_save_active_discrete",
    ),
    "active_cmt": ("Cmt_save_active_discrete(0,1)-10-30", "Cmt_save"),
    "active_energy": (
        "Energy_save_active_discrete-10-30",
        "Energy_save_active_discrete",
    ),
    "continuous_status": (
        "working_save_active_continuous-10-30",
        "working_save_active_continuous",
    ),
    "continuous_cmt": ("Cmt_save_active_continuous-10-30", "Cmt_save"),
    "continuous_energy": (
        "Energy_save_active_continuous-10-30",
        "Energy_save_active_continuous",
    ),
}

METHODS = (
    ("continuous", "Continuous PPO"),
    ("active", "Active PPO"),
    ("proposed", "Proposed"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_array(directory: Path, file_name: str, dataset: str) -> np.ndarray:
    path = directory / file_name
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"{dataset!r} is missing from {path}")
        array = handle[dataset][...]
    if array.shape != (10, 30, 10, 30):
        raise ValueError(f"unexpected shape {array.shape} in {path}")
    return array


def expected_fused_cmt(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Implement the complete evaluator branch used for Cmt_save_passive."""

    fused_status = arrays["proposed_status"]
    negative_status = arrays["negative_status"]
    positive_status = arrays["positive_status"]
    negative_cmt = arrays["negative_cmt"]
    positive_cmt = arrays["positive_cmt"]

    expected = np.full(fused_status.shape, -2.0, dtype=float)
    negative_route = fused_status == -1
    positive_route = fused_status == 1
    zero_route = fused_status == 0
    both_valid = zero_route & (negative_status != -2) & (positive_status != -2)
    negative_failed = zero_route & (negative_status == -2)
    remaining_zero = zero_route & ~both_valid & ~negative_failed

    expected[negative_route] = negative_cmt[negative_route]
    expected[positive_route] = positive_cmt[positive_route]
    negative_finite = np.isfinite(negative_cmt)
    positive_finite = np.isfinite(positive_cmt)
    both_finite = both_valid & negative_finite & positive_finite
    negative_only = both_valid & negative_finite & ~positive_finite
    positive_only = both_valid & ~negative_finite & positive_finite
    neither_finite = both_valid & ~negative_finite & ~positive_finite
    expected[both_finite] = np.minimum(
        negative_cmt[both_finite], positive_cmt[both_finite]
    )
    expected[negative_only] = negative_cmt[negative_only]
    expected[positive_only] = positive_cmt[positive_only]
    expected[neither_finite] = np.nan
    expected[negative_failed] = positive_cmt[negative_failed]
    expected[remaining_zero] = negative_cmt[remaining_zero]
    return expected


def expected_fused_energy(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Select the expert energy paired with the fused Cmt value."""

    fused_status = arrays["proposed_status"]
    negative_status = arrays["negative_status"]
    positive_status = arrays["positive_status"]
    negative_cmt = arrays["negative_cmt"]
    positive_cmt = arrays["positive_cmt"]
    negative_energy = arrays["negative_energy"]
    positive_energy = arrays["positive_energy"]

    expected = np.full(fused_status.shape, np.nan, dtype=float)
    negative_route = fused_status == -1
    positive_route = fused_status == 1
    zero_route = fused_status == 0
    both_valid = zero_route & (negative_status != -2) & (positive_status != -2)
    negative_failed = zero_route & (negative_status == -2)
    remaining_zero = zero_route & ~both_valid & ~negative_failed

    expected[negative_route] = negative_energy[negative_route]
    expected[positive_route] = positive_energy[positive_route]
    negative_finite = np.isfinite(negative_cmt)
    positive_finite = np.isfinite(positive_cmt)
    choose_negative = both_valid & negative_finite & (
        ~positive_finite | (negative_cmt <= positive_cmt)
    )
    choose_positive = both_valid & positive_finite & (
        ~negative_finite | (positive_cmt < negative_cmt)
    )
    expected[choose_negative] = negative_energy[choose_negative]
    expected[choose_positive] = positive_energy[choose_positive]
    expected[negative_failed] = positive_energy[negative_failed]
    expected[remaining_zero] = negative_energy[remaining_zero]
    return expected


def displacement_eligible(
    cmt: np.ndarray, energy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover legacy positive-work displacement and apply the 0.01 m floor.

    The retained evaluator archive stores both energy and Cmt, so for Cmt > 0
    the original displacement is exactly recoverable as E/(W*Cmt).  Entries
    with zero energy and zero Cmt cannot be inverted, but they do not create a
    small-denominator tail and are retained as zero-actuation successes.
    Future evaluator runs write D_save arrays directly.
    """

    displacement = np.full(cmt.shape, np.nan, dtype=float)
    positive_cmt = np.isfinite(cmt) & (cmt > 0) & np.isfinite(energy) & (energy >= 0)
    displacement[positive_cmt] = energy[positive_cmt] / (
        ROBOT_WEIGHT_N * cmt[positive_cmt]
    )
    zero_energy = (
        np.isfinite(cmt)
        & np.isfinite(energy)
        & np.isclose(cmt, 0.0, rtol=0.0, atol=1e-15)
        & np.isclose(energy, 0.0, rtol=0.0, atol=1e-15)
    )
    eligible = (displacement > MIN_COM_DISPLACEMENT_M) | zero_energy
    return eligible, displacement, zero_energy


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Local Energy_Comparison directory containing action_weight=* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    audit_rows: list[dict[str, object]] = []
    file_records: list[dict[str, object]] = []

    for weight in WEIGHTS:
        for condition in CONDITIONS:
            directory = data_root / f"action_weight={weight}" / condition["directory"]
            arrays: dict[str, np.ndarray] = {}
            for key, (file_name, dataset) in ARRAY_FILES.items():
                path = directory / file_name
                arrays[key] = load_array(directory, file_name, dataset)
                file_records.append(
                    {
                        "action_weight": weight,
                        "condition": condition["directory"],
                        "logical_array": key,
                        "file_name": file_name,
                        "dataset": dataset,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

            expected = expected_fused_cmt(arrays)
            actual = arrays["proposed_cmt"]
            comparison = np.isclose(
                expected, actual, rtol=0.0, atol=1e-12, equal_nan=True
            )
            fusion_bad_n = int(np.size(comparison) - np.count_nonzero(comparison))
            if fusion_bad_n:
                max_error = float(np.max(np.abs(expected[~comparison] - actual[~comparison])))
                raise ValueError(
                    f"fusion validation failed in {directory}: "
                    f"{fusion_bad_n} cells, max error {max_error}"
                )

            if np.array_equal(arrays["negative_cmt"], arrays["positive_cmt"]):
                raise ValueError(f"negative and positive Cmt arrays are identical in {directory}")

            status_common = (
                (arrays["proposed_status"] != -2)
                & (arrays["active_status"] != -2)
                & (arrays["continuous_status"] != -6)
            )
            proposed_energy = expected_fused_energy(arrays)
            proposed_eligible, _proposed_d, proposed_zero_energy = displacement_eligible(
                arrays["proposed_cmt"], proposed_energy
            )
            active_eligible, _active_d, active_zero_energy = displacement_eligible(
                arrays["active_cmt"], arrays["active_energy"]
            )
            continuous_eligible, _continuous_d, continuous_zero_energy = (
                displacement_eligible(
                    arrays["continuous_cmt"], arrays["continuous_energy"]
                )
            )
            common = (
                status_common
                & proposed_eligible
                & active_eligible
                & continuous_eligible
            )
            common_n = int(np.count_nonzero(common))
            if common_n == 0:
                raise ValueError(f"empty common-feasible mask in {directory}")

            row: dict[str, object] = {
                "action_weight": weight,
                "random_seed": SEED,
                "archived_condition": condition["directory"],
                "step_height_m": condition["step_height_m"],
                "max_horizontal_advance_m": condition["max_horizontal_advance_m"],
                "paper_column": condition["paper_column"],
                "paper_order": condition["paper_order"],
                "minimum_com_displacement_m": MIN_COM_DISPLACEMENT_M,
                "status_common_n": int(np.count_nonzero(status_common)),
                "common_mask_n": common_n,
                "displacement_excluded_n": int(
                    np.count_nonzero(status_common & ~common)
                ),
                "continuous_displacement_excluded_n": int(
                    np.count_nonzero(status_common & ~continuous_eligible)
                ),
                "active_displacement_excluded_n": int(
                    np.count_nonzero(status_common & ~active_eligible)
                ),
                "proposed_displacement_excluded_n": int(
                    np.count_nonzero(status_common & ~proposed_eligible)
                ),
                "legacy_zero_energy_retained_n": int(
                    np.count_nonzero(
                        status_common
                        & (proposed_zero_energy | active_zero_energy | continuous_zero_energy)
                    )
                ),
                "fusion_bad_n": fusion_bad_n,
            }
            for prefix, _label in METHODS:
                values = np.asarray(arrays[f"{prefix}_cmt"][common], dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite {prefix} Cmt in {directory}")
                for statistic, value in summarize(values).items():
                    row[f"{prefix}_{statistic}"] = value
            audit_rows.append(row)

    audit_rows.sort(key=lambda item: (float(item["action_weight"]), int(item["paper_order"])))
    audit_fields = [
        "action_weight",
        "random_seed",
        "archived_condition",
        "step_height_m",
        "max_horizontal_advance_m",
        "paper_column",
        "paper_order",
        "minimum_com_displacement_m",
        "status_common_n",
        "common_mask_n",
        "displacement_excluded_n",
        "continuous_displacement_excluded_n",
        "active_displacement_excluded_n",
        "proposed_displacement_excluded_n",
        "legacy_zero_energy_retained_n",
        "fusion_bad_n",
    ]
    for prefix, _label in METHODS:
        audit_fields.extend(
            f"{prefix}_{statistic}"
            for statistic in ("mean", "median", "sample_sd", "p99", "max")
        )
    audit_path = output_dir / "action_weight_rerun_seed_20260716.csv"
    write_csv(audit_path, audit_rows, audit_fields)

    paper_columns = [str(condition["paper_column"]) for condition in CONDITIONS]
    manuscript_rows: list[dict[str, object]] = []
    for weight in WEIGHTS:
        weight_rows = [row for row in audit_rows if row["action_weight"] == weight]
        for prefix, label in METHODS:
            manuscript_row: dict[str, object] = {
                "action_weight": weight,
                "method": label,
                "statistic": "mean Cmt",
            }
            for row in weight_rows:
                manuscript_row[str(row["paper_column"])] = f"{float(row[f'{prefix}_mean']):.3f}"
            manuscript_rows.append(manuscript_row)
        count_row: dict[str, object] = {
            "action_weight": weight,
            "method": "Common-mask n",
            "statistic": "count",
        }
        for row in weight_rows:
            count_row[str(row["paper_column"])] = int(row["common_mask_n"])
        manuscript_rows.append(count_row)
    table_path = output_dir / "action_weight_table_ii_seed_20260716.csv"
    write_csv(
        table_path,
        manuscript_rows,
        ["action_weight", "method", "statistic", *paper_columns],
    )

    repository_root = Path(__file__).resolve().parents[1]
    scripts = []
    for weight in WEIGHTS:
        for condition in CONDITIONS:
            script = (
                repository_root
                / "src"
                / "two_link"
                / "action_weight"
                / f"action_weight={weight}"
                / str(condition["directory"])
                / "Whole_energy_comparison_low_dim.py"
            )
            scripts.append(
                {
                    "path": script.relative_to(repository_root).as_posix(),
                    "bytes": script.stat().st_size,
                    "sha256": sha256(script),
                }
            )
    manifest = {
        "protocol": (
            "Table II fixed-checkpoint action-weight evaluation with a strict "
            "positive-work COM-displacement floor"
        ),
        "random_seed": SEED,
        "minimum_com_displacement_m": MIN_COM_DISPLACEMENT_M,
        "common_mask": (
            "(proposed_status != -2) & (active_status != -2) & "
            "(continuous_status != -6) & proposed_displacement_eligible & "
            "active_displacement_eligible & continuous_displacement_eligible"
        ),
        "legacy_displacement_recovery": (
            "For positive Cmt, D is recovered exactly from retained Energy and Cmt as "
            "D=E/(W*Cmt), W=(1.4122+0.0839)*9.81 N. Zero-energy/zero-Cmt "
            "successes are retained because they do not create a small-denominator tail. "
            "Future canonical evaluators write D_save arrays directly."
        ),
        "fusion_validation": "complete evaluator branch; only -2 denotes expert failure",
        "pre_filter_manifest": (
            "action_weight_rerun_seed_20260716_pre_0p01m_filter_manifest.json"
        ),
        "audit_csv": audit_path.name,
        "manuscript_csv": table_path.name,
        "canonical_scripts": scripts,
        "input_hdf5": file_records,
    }
    manifest_path = output_dir / "action_weight_rerun_seed_20260716_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"validated {len(audit_rows)} action-weight cells")
    print(f"fusion mismatches: {sum(int(row['fusion_bad_n']) for row in audit_rows)}")
    print(f"wrote {audit_path}")
    print(f"wrote {table_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
