"""Recompute Table II from the 12-cell action-weight HDF5 archive.

Eligibility is evaluated from each controller's archived center-of-mass
displacement array. This analysis does not infer missing displacement from
energy/Cmt, and zero-work entries receive no exception to the strict
D > 0.01 m rule.
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
WEIGHTS = (
    ("0.02", "action_weight_0p02"),
    ("0.04", "action_weight_0p04"),
    ("0.06", "action_weight_0p06"),
)
CONDITIONS = (
    ("flat_0p25m", 0.00, 0.25, "0.00 m step; 0.25 m advance", 0),
    ("flat_0p17m", 0.00, 0.17, "0.00 m step; 0.17 m advance", 1),
    ("raised_0p01m_0p25m", 0.01, 0.25, "0.01 m step; 0.25 m advance", 2),
    ("raised_0p01m_0p17m", 0.01, 0.17, "0.01 m step; 0.17 m advance", 3),
)
METHODS = (
    ("continuous_active_ppo", "continuous", "Active Continuous PPO"),
    ("discrete_active_ppo", "active", "Active discrete PPO"),
    ("offline_route", "proposed", "Proposed"),
)
EXPECTED_SHAPE = (10, 30, 10, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset(group: h5py.Group, relative: str) -> np.ndarray:
    values = group[relative][...]
    if values.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {values.shape} for {group.name}/{relative}")
    return values


def expected_fused_cmt(group: h5py.Group) -> np.ndarray:
    route = dataset(group, "offline_route/outcome_status")
    negative_status = dataset(group, "negative_expert/outcome_status")
    positive_status = dataset(group, "positive_expert/outcome_status")
    negative_cmt = dataset(group, "negative_expert/cmt")
    positive_cmt = dataset(group, "positive_expert/cmt")

    expected = np.full(EXPECTED_SHAPE, -2.0, dtype=float)
    expected[route == -1] = negative_cmt[route == -1]
    expected[route == 1] = positive_cmt[route == 1]
    zero_route = route == 0
    both = zero_route & (negative_status != -2) & (positive_status != -2)
    negative_failed = zero_route & (negative_status == -2)
    remaining = zero_route & ~both & ~negative_failed
    expected[both] = np.minimum(negative_cmt[both], positive_cmt[both])
    expected[negative_failed] = positive_cmt[negative_failed]
    expected[remaining] = negative_cmt[remaining]
    return expected


def resolve_archive(value: Path) -> Path:
    value = value.resolve()
    if value.is_file():
        return value
    candidates = (
        value / "action_weight_12_cell_arrays.h5",
        value / "data/action_weight_arrays/action_weight_12_cell_arrays.h5",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"action_weight_12_cell_arrays.h5 not found under {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="S2 directory or the action_weight_12_cell_arrays.h5 file itself",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    args = parser.parse_args()
    archive = resolve_archive(args.data_root)
    output_dir = args.output_dir.resolve()
    rows: list[dict[str, object]] = []

    with h5py.File(archive, "r") as handle:
        for weight, weight_group in WEIGHTS:
            for condition, height, advance, paper_column, order in CONDITIONS:
                group = handle[f"{weight_group}/{condition}"]
                statuses = {
                    name: dataset(group, f"{name}/outcome_status")
                    for name, _prefix, _label in METHODS
                }
                status_common = (
                    (statuses["continuous_active_ppo"] != -6)
                    & (statuses["discrete_active_ppo"] != -2)
                    & (statuses["offline_route"] != -2)
                )
                archived_status = dataset(group, "status_common_mask").astype(bool)
                if not np.array_equal(status_common, archived_status):
                    raise ValueError(f"status common-mask mismatch in {group.name}")

                archived_displacements = {
                    name: dataset(group, f"{name}/com_displacement_m")
                    for name, _prefix, _label in METHODS
                }
                strict_common = status_common.copy()
                for displacement in archived_displacements.values():
                    strict_common &= np.isfinite(displacement) & (
                        displacement > MIN_COM_DISPLACEMENT_M
                    )
                common_n = int(np.count_nonzero(strict_common))
                if common_n == 0:
                    raise ValueError(f"empty strict common mask in {group.name}")

                primary_mask = dataset(
                    group, "primary_common_mask_D_gt_0p01m"
                ).astype(bool)
                if not np.array_equal(primary_mask, strict_common):
                    raise ValueError(f"primary strict-mask mismatch in {group.name}")
                expected = expected_fused_cmt(group)
                actual = dataset(group, "offline_route/cmt")
                comparison = np.isclose(expected, actual, rtol=0.0, atol=1e-12)
                fusion_bad_n = int(np.size(comparison) - np.count_nonzero(comparison))
                if fusion_bad_n:
                    raise ValueError(f"expert-fusion validation failed in {group.name}")

                row: dict[str, object] = {
                    "action_weight": weight,
                    "random_seed": SEED,
                    "archive_group": group.name.lstrip("/"),
                    "step_height_m": height,
                    "max_horizontal_advance_m": advance,
                    "paper_column": paper_column,
                    "paper_order": order,
                    "minimum_com_displacement_m": MIN_COM_DISPLACEMENT_M,
                    "status_common_n": int(np.count_nonzero(status_common)),
                    "strict_displacement_common_n": common_n,
                    "excluded_by_strict_displacement_n": int(
                        np.count_nonzero(status_common & ~strict_common)
                    ),
                    "archive_primary_mask_n": int(np.count_nonzero(primary_mask)),
                    "fusion_bad_n": fusion_bad_n,
                }
                for name, prefix, _label in METHODS:
                    values = dataset(group, f"{name}/cmt")[strict_common]
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"non-finite {name} Cmt in {group.name}")
                    for statistic, value in summarize(values).items():
                        row[f"{prefix}_{statistic}"] = value
                rows.append(row)

    rows.sort(key=lambda row: (float(row["action_weight"]), int(row["paper_order"])))
    fields = [
        "action_weight", "random_seed", "archive_group", "step_height_m",
        "max_horizontal_advance_m", "paper_column", "paper_order",
        "minimum_com_displacement_m", "status_common_n",
        "strict_displacement_common_n", "excluded_by_strict_displacement_n",
        "archive_primary_mask_n", "fusion_bad_n",
    ]
    for _name, prefix, _label in METHODS:
        fields.extend(
            f"{prefix}_{statistic}"
            for statistic in ("mean", "median", "sample_sd", "p99", "max")
        )
    audit_path = output_dir / "action_weight_rerun_seed_20260716.csv"
    write_csv(audit_path, rows, fields)

    paper_columns = [condition[3] for condition in CONDITIONS]
    manuscript_rows: list[dict[str, object]] = []
    for weight, _group in WEIGHTS:
        weight_rows = [row for row in rows if row["action_weight"] == weight]
        for _name, prefix, label in METHODS:
            item: dict[str, object] = {
                "action_weight": weight, "method": label, "statistic": "mean Cmt"
            }
            for row in weight_rows:
                item[str(row["paper_column"])] = f"{float(row[f'{prefix}_mean']):.3f}"
            manuscript_rows.append(item)
        count: dict[str, object] = {
            "action_weight": weight, "method": "Common-mask n", "statistic": "count"
        }
        for row in weight_rows:
            count[str(row["paper_column"])] = int(row["strict_displacement_common_n"])
        manuscript_rows.append(count)
    table_path = output_dir / "action_weight_table_ii_seed_20260716.csv"
    write_csv(
        table_path,
        manuscript_rows,
        ["action_weight", "method", "statistic", *paper_columns],
    )

    root = Path(__file__).resolve().parents[1]
    scripts = []
    condition_dirs = (
        "60,30(0.33m)", "70,20(0.225m)",
        "60,30(0.33m),0.01m", "70,20,(0.225m),0.01m",
    )
    for weight, _group in WEIGHTS:
        for directory in condition_dirs:
            script = root / "src/two_link/action_weight" / f"action_weight={weight}" / directory / "Whole_energy_comparison_low_dim.py"
            scripts.append({
                "path": script.relative_to(root).as_posix(),
                "bytes": script.stat().st_size,
                "sha256": sha256(script),
            })
    manifest = {
        "protocol": "Table II fixed-checkpoint action-weight evaluation",
        "random_seed": SEED,
        "minimum_com_displacement_m": MIN_COM_DISPLACEMENT_M,
        "eligibility": (
            "status common mask and finite archived com_displacement_m > 0.01 m "
            "for continuous active PPO, discrete active PPO, and the offline route"
        ),
        "displacement_source": (
            "Archived com_displacement_m arrays in Supplementary Data S2; this "
            "analysis does not infer missing displacement from energy/Cmt and applies "
            "no zero-work exception. The S2 schema records array provenance."
        ),
        "archive_masks": (
            "primary_common_mask_D_gt_0p01m must equal the strict recomputation; "
            "the earlier zero-work-inclusive mask is not distributed in the current S2 release."
        ),
        "fusion_validation": "complete evaluator branch; only -2 denotes expert failure",
        "source_archive": {
            "file_name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "audit_csv": audit_path.name,
        "manuscript_csv": table_path.name,
        "canonical_scripts": scripts,
    }
    manifest_path = output_dir / "action_weight_rerun_seed_20260716_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"validated {len(rows)} action-weight cells with archived displacement")
    print(f"wrote {audit_path}")
    print(f"wrote {table_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
