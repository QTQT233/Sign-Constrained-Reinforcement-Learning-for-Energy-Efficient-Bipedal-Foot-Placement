"""Regression tests for the canonical Table II evaluator release."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

WEIGHTS = ("0.02", "0.04", "0.06")
CONDITIONS = (
    "60,30(0.33m)",
    "60,30(0.33m),0.01m",
    "70,20(0.225m)",
    "70,20,(0.225m),0.01m",
)
EXPECTED_COMMON_N = {
    ("0.02", "60,30(0.33m)"): 52668,
    ("0.02", "70,20(0.225m)"): 39675,
    ("0.02", "60,30(0.33m),0.01m"): 50283,
    ("0.02", "70,20,(0.225m),0.01m"): 36607,
    ("0.04", "60,30(0.33m)"): 47241,
    ("0.04", "70,20(0.225m)"): 35426,
    ("0.04", "60,30(0.33m),0.01m"): 50175,
    ("0.04", "70,20,(0.225m),0.01m"): 27651,
    ("0.06", "60,30(0.33m)"): 52308,
    ("0.06", "70,20(0.225m)"): 30680,
    ("0.06", "60,30(0.33m),0.01m"): 48284,
    ("0.06", "70,20,(0.225m),0.01m"): 27006,
}

from cmt_metrics import positive_actuator_work_increment, select_successful_expert_by_cmt


def canonical_scripts() -> list[tuple[str, str, Path]]:
    return [
        (
            weight,
            condition,
            ROOT
            / f"action_weight={weight}"
            / condition
            / "Whole_energy_comparison_low_dim.py",
        )
        for weight in WEIGHTS
        for condition in CONDITIONS
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MetricUnitTests(unittest.TestCase):
    def test_positive_commanded_work_has_joule_units(self):
        torque_nm = np.array([2.0, -3.0, 1.5, -4.0])
        relative_speed_rad_s = np.array([4.0, -2.0, -5.0, 0.0])
        expected_j = np.array([0.8, 0.6, 0.0, 0.0])
        actual_j = positive_actuator_work_increment(
            torque_nm, relative_speed_rad_s, 0.1
        )
        np.testing.assert_allclose(actual_j, expected_j)

    def test_only_minus_two_is_expert_failure(self):
        negative_status = np.array([-2, 0, -2, -1, 0])
        positive_status = np.array([-2, -2, 0, 1, 1])
        negative_cmt = np.array([0.0, 0.31, 0.0, 0.24, 0.41])
        positive_cmt = np.array([0.0, 0.0, 0.29, 0.18, 0.35])
        selector, selected = select_successful_expert_by_cmt(
            negative_status,
            negative_cmt,
            positive_status,
            positive_cmt,
            tie_break="positive",
        )
        np.testing.assert_array_equal(selector, [-2, -1, 1, 1, 1])
        np.testing.assert_allclose(selected, [-2.0, 0.31, 0.29, 0.18, 0.35])


class CanonicalSourceTests(unittest.TestCase):
    def test_all_twelve_official_scripts_are_present_and_compile(self):
        scripts = canonical_scripts()
        self.assertEqual(len(scripts), 12)
        for _weight, _condition, script in scripts:
            self.assertTrue(script.is_file(), msg=f"missing {script}")
            compile(script.read_text(encoding="utf-8"), str(script), "exec")

    def test_fixed_seed_is_declared_in_every_script(self):
        required = (
            "RANDOM_SEED = 20260716",
            "random.seed(RANDOM_SEED)",
            "np.random.seed(RANDOM_SEED)",
            "torch.manual_seed(RANDOM_SEED)",
            "torch.cuda.manual_seed_all(RANDOM_SEED)",
        )
        for _weight, _condition, script in canonical_scripts():
            source = script.read_text(encoding="utf-8")
            for statement in required:
                self.assertIn(statement, source, msg=f"{statement} missing from {script}")

    def test_negative_expert_file_receives_negative_cmt_array(self):
        pattern = re.compile(
            r"Cmt_save\(-1,0\)-10-30.*?"
            r"create_dataset\(\s*['\"]Cmt_save['\"]\s*,\s*"
            r"data\s*=\s*(Cmt_save\w+)\s*\)",
            flags=re.DOTALL,
        )
        for _weight, _condition, script in canonical_scripts():
            source = script.read_text(encoding="utf-8")
            match = pattern.search(source)
            self.assertIsNotNone(match, msg=f"negative Cmt write missing in {script}")
            self.assertEqual(match.group(1), "Cmt_save0_1", msg=str(script))

    def test_complete_fusion_guard_is_preserved(self):
        required = (
            "working_save_passive[i, j, k, ll] == 0",
            "working_save0_1[i, j, k, ll] != -2 and working_save01[i, j, k, ll] != -2",
            "min(Cmt_save01[i, j, k, ll], Cmt_save0_1[i, j, k, ll])",
            "elif working_save0_1[i, j, k, ll] == -2",
            "Cmt_save_passive[i, j, k, ll] = Cmt_save01[i, j, k, ll]",
            "Cmt_save_passive[i, j, k, ll] = Cmt_save0_1[i, j, k, ll]",
        )
        for _weight, _condition, script in canonical_scripts():
            source = script.read_text(encoding="utf-8")
            for statement in required:
                self.assertIn(statement, source, msg=f"fusion branch differs in {script}")

    def test_continuous_work_uses_applied_torque_once(self):
        correct_increment = (
            "Energy_save_active_continuous[i_, j, k, ll] += "
            "abs(action_value) * abs(y[3]-y[1]) * dt"
        )
        for _weight, _condition, script in canonical_scripts():
            source = script.read_text(encoding="utf-8")
            self.assertIn(correct_increment, source, msg=str(script))
            self.assertNotRegex(source, r"action_value\s*\*\s*torque")
            self.assertNotRegex(source, r"Cmt_save_active_continuous.*?/\s*4")


class PublishedResultTests(unittest.TestCase):
    def test_audit_csv_contains_all_rerun_cells(self):
        path = REPOSITORY_ROOT / "results" / "action_weight_rerun_seed_20260716.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        for row in rows:
            key = (row["action_weight"], row["archived_condition"])
            self.assertEqual(int(row["random_seed"]), 20260716)
            self.assertEqual(int(row["fusion_bad_n"]), 0)
            self.assertEqual(int(row["common_mask_n"]), EXPECTED_COMMON_N[key])
            for field in ("continuous_mean", "active_mean", "proposed_mean"):
                self.assertTrue(np.isfinite(float(row[field])))

    def test_manifest_binds_current_canonical_scripts(self):
        path = (
            REPOSITORY_ROOT
            / "results"
            / "action_weight_rerun_seed_20260716_manifest.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["random_seed"], 20260716)
        self.assertEqual(len(manifest["canonical_scripts"]), 12)
        for record in manifest["canonical_scripts"]:
            script = REPOSITORY_ROOT / record["path"]
            self.assertEqual(record["sha256"], sha256(script), msg=str(script))

    def test_manuscript_csv_has_three_methods_and_counts_per_weight(self):
        path = (
            REPOSITORY_ROOT
            / "results"
            / "action_weight_table_ii_seed_20260716.csv"
        )
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        for weight in WEIGHTS:
            methods = {row["method"] for row in rows if row["action_weight"] == weight}
            self.assertEqual(
                methods,
                {"Continuous PPO", "Active PPO", "Proposed", "Common-mask n"},
            )


if __name__ == "__main__":
    unittest.main()
