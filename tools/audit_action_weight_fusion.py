"""Read-only audit of the retained action-weight expert-fusion arrays.

This script never opens an HDF5 file in write mode.  It exposes the status
semantics that the legacy evaluator obscured: -2 is failure; -1, 0, and +1 are
successful rollouts labelled by their first action.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


WEIGHTS = ("0.02", "0.04", "0.06")
CONDITIONS = (
    "60,30(0.33m)",
    "60,30(0.33m),0.01m",
    "70,20(0.225m)",
    "70,20,(0.225m),0.01m",
)


def read_h5(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        return np.asarray(stream[dataset][()])


def audit_cell(folder: Path, weight: str, condition: str) -> dict[str, object]:
    negative_status = read_h5(folder / "working_save(-1,0)-10-30", "working_save")
    positive_status = read_h5(folder / "working_save(0,1)-10-30", "working_save")
    legacy_status = read_h5(
        folder / "working_save_passive-10-30", "working_save_passive"
    )
    active_status = read_h5(
        folder / "working_save_active_discrete-10-30",
        "working_save_active_discrete",
    )
    continuous_status = read_h5(
        folder / "working_save_active_continuous-10-30",
        "working_save_active_continuous",
    )
    legacy_cmt = read_h5(folder / "Cmt_save_passive-10-30", "Cmt_save_passive")

    arrays = (
        positive_status,
        legacy_status,
        active_status,
        continuous_status,
        legacy_cmt,
    )
    if any(array.shape != negative_status.shape for array in arrays):
        raise ValueError(f"shape mismatch in {folder}")

    negative_success = negative_status != -2
    positive_success = positive_status != -2
    dual_success = negative_success & positive_success
    negative_only = negative_success & ~positive_success
    positive_only = ~negative_success & positive_success
    dual_failure = ~negative_success & ~positive_success
    literal_minus1_plus1 = (negative_status == -1) & (positive_status == 1)
    dual_success_with_zero = dual_success & (
        (negative_status == 0) | (positive_status == 0)
    )
    single_success_first_action_zero = (
        negative_only & (negative_status == 0)
    ) | (positive_only & (positive_status == 0))

    common_mask = (
        (legacy_status != -2)
        & (active_status != -2)
        & (continuous_status != -6)
    )
    pseudozero_common = (
        common_mask & single_success_first_action_zero & (legacy_cmt == 0)
    )
    common_n = int(np.count_nonzero(common_mask))
    pseudozero_n = int(np.count_nonzero(pseudozero_common))

    return {
        "action_weight": weight,
        "archived_condition": condition,
        "total_grid_n": int(negative_status.size),
        "dual_success_n": int(np.count_nonzero(dual_success)),
        "dual_success_minus1_plus1_n": int(np.count_nonzero(literal_minus1_plus1)),
        "dual_success_with_first_action_zero_n": int(
            np.count_nonzero(dual_success_with_zero)
        ),
        "negative_only_success_n": int(np.count_nonzero(negative_only)),
        "positive_only_success_n": int(np.count_nonzero(positive_only)),
        "both_failed_n": int(np.count_nonzero(dual_failure)),
        "single_success_first_action_zero_n": int(np.count_nonzero(single_success_first_action_zero)),
        "three_method_common_mask_n": common_n,
        "single_success_first_action_zero_pseudozero_on_common_mask_n": pseudozero_n,
        "pseudozero_fraction_of_common_mask": pseudozero_n / common_n,
        "legacy_hdf5_rewritten": False,
        "legacy_proposed_valid_for_method_ranking": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for weight in WEIGHTS:
        for condition in CONDITIONS:
            folder = args.data_root / f"action_weight={weight}" / condition
            rows.append(audit_cell(folder, weight, condition))

    if len(rows) != 12:
        raise RuntimeError(f"expected 12 rows, found {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "dual_success_n",
            "dual_success_minus1_plus1_n",
            "dual_success_with_first_action_zero_n",
            "single_success_first_action_zero_n",
            "three_method_common_mask_n",
            "single_success_first_action_zero_pseudozero_on_common_mask_n",
        )
    }
    expected = {
        "dual_success_n": 125038,
        "dual_success_minus1_plus1_n": 0,
        "dual_success_with_first_action_zero_n": 125038,
        "single_success_first_action_zero_n": 227935,
        "three_method_common_mask_n": 443163,
        "single_success_first_action_zero_pseudozero_on_common_mask_n": 216946,
    }
    if totals != expected:
        raise RuntimeError(f"retained-archive truth counts changed: {totals} != {expected}")
    print(totals)


if __name__ == "__main__":
    main()
