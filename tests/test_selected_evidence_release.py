from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class SelectedEvidenceReleaseTests(unittest.TestCase):
    def test_source_aligned_table_i_partition_and_executable_route(self) -> None:
        table = {
            row["controller_or_set"]: row
            for row in rows(ROOT / "results/table1/table_i.csv")
        }
        expected = {
            "Unrestricted Active PPO": 647160,
            "One-sided expert union": 562003,
            "Active-default three-expert lookup": 672382,
            "Active and one-sided feasible": 536781,
            "Active only": 110379,
            "One-sided only": 25222,
            "Neither": 137618,
        }
        for label, value in expected.items():
            self.assertEqual(int(table[label]["feasible_states"]), value)

        route_path = (
            ROOT
            / "data/two_link/action_map_source_aligned"
            / "active_default_route_30x30x30x30.h5"
        )
        with h5py.File(route_path, "r") as handle:
            route = handle["route_code"][:]
            self.assertEqual(
                handle["route_code"].attrs["codes"],
                "-1=uncovered,0=non-positive,1=non-negative,2=active",
            )
        self.assertEqual(route.shape, (30, 30, 30, 30))
        self.assertEqual(int(np.count_nonzero(route >= 0)), 672382)
        self.assertEqual(set(np.unique(route).tolist()), {-1, 0, 1, 2})

    def test_fixed_12_case_active_fallback_record(self) -> None:
        path = (
            ROOT
            / "results/two_link_active_fallback_fixed12"
            / "fixed_proposed_push_off_results.csv"
        )
        records = rows(path)
        selected = [
            row for row in records if row["controller"] == "active_fallback"
        ]
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(row["success"] == "1" for row in selected))
        self.assertTrue(
            all(row["cmt_below_continuous_table_value"] == "1" for row in selected)
        )
        self.assertAlmostEqual(
            sum(float(row["cmt"]) for row in selected) / 12,
            0.12234630641701309,
        )
        self.assertTrue(
            all("active_fallback" not in row["branch_sequence"] for row in selected)
        )

    def test_four_link_active_default_confirmation(self) -> None:
        summary = json.loads(
            (
                ROOT
                / "results/four_link_active_default_confirmation"
                / "publication_summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["n"], 2160)
        self.assertEqual(summary["active_success_count"], 1061)
        self.assertEqual(summary["selector_success_count"], 1063)
        self.assertEqual(summary["common_valid_count"], 509)
        self.assertAlmostEqual(
            summary["active_common_mean_cmt"], 1.50424431451462
        )
        self.assertAlmostEqual(
            summary["selector_common_mean_cmt"], 0.6688318479590704
        )
        self.assertLess(summary["cmt_difference_95_ci"][1], 0.0)
        self.assertEqual(
            summary["route_counts"],
            {
                "confidence_active_default": 1373,
                "confidence_nonnegative": 249,
                "confidence_nonpositive": 538,
            },
        )

        trials = (
            ROOT
            / "results/four_link_active_default_confirmation"
            / "matched_trials.csv.gz"
        )
        with gzip.open(trials, "rt", newline="", encoding="utf-8") as handle:
            trial_rows = list(csv.DictReader(handle))
        self.assertEqual(len(trial_rows), 4320)
        self.assertNotIn(
            "three_expert_selector",
            {row["controller"] for row in trial_rows},
        )

    def test_active_default_route_unit_cases(self) -> None:
        source = (
            ROOT
            / "src/four_link/evaluation"
            / "run_active_default_confidence_confirmation.py"
        )
        spec = importlib.util.spec_from_file_location(
            "active_default_confidence_confirmation_test_module", source
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        route = module.route_from_probabilities
        self.assertEqual(route([0.90, 0.20, 0.10])[0], "positive")
        self.assertEqual(route([0.20, 0.60, 0.20])[0], "negative")
        self.assertEqual(route([0.90, 0.95, 0.01])[0], "negative")
        self.assertEqual(route([0.50, 0.50, 0.00])[0], "active")


if __name__ == "__main__":
    unittest.main()
