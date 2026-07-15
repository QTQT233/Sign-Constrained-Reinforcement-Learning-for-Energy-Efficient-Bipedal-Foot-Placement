"""Regression tests for the continuous-controller Cmt work accumulator."""

from __future__ import annotations

import csv
import re
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHTS = ("0.02", "0.04", "0.06")
CONDITIONS = (
    "60,30(0.33m)",
    "60,30(0.33m),0.01m",
    "70,20(0.225m)",
    "70,20,(0.225m),0.01m",
)
LEGACY_DATA_ROOT = "D:/L&S/Mas/Project/Paper1/Energy_Comparison"

from cmt_metrics import (
    positive_actuator_work_increment,
    select_successful_expert_by_cmt,
)


class PositiveActuatorWorkTests(unittest.TestCase):
    def test_published_fusion_audit_truth_counts(self):
        evidence_name = "action_weight_fusion_audit_12_cells.csv"
        candidates = [
            parent / "results" / evidence_name
            for parent in (ROOT, *Path(__file__).resolve().parents)
        ]
        evidence_csv = next((path for path in candidates if path.is_file()), None)
        self.assertIsNotNone(evidence_csv, msg=f"missing fusion audit: {candidates}")
        with evidence_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)

        def total(field):
            return sum(int(row[field]) for row in rows)

        self.assertEqual(total("dual_success_n"), 125038)
        self.assertEqual(total("dual_success_minus1_plus1_n"), 0)
        self.assertEqual(total("dual_success_with_first_action_zero_n"), 125038)
        self.assertEqual(total("single_success_first_action_zero_n"), 227935)
        self.assertEqual(total("three_method_common_mask_n"), 443163)
        self.assertEqual(
            total("single_success_first_action_zero_pseudozero_on_common_mask_n"),
            216946,
        )

    def test_published_12_cell_correction_table_matches_raw_divided_by_four(self):
        evidence_name = "action_weight_cmt_correction_12_cells.csv"
        candidates = [
            parent / "results" / evidence_name
            for parent in (ROOT, *Path(__file__).resolve().parents)
        ]
        evidence_csv = next((path for path in candidates if path.is_file()), None)
        self.assertIsNotNone(
            evidence_csv,
            msg=f"missing published audit table; searched: {candidates}",
        )

        with evidence_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertIn("legacy_proposed_forensic_mean", row)
            self.assertNotIn("proposed_hindsight_envelope_mean", row)
            self.assertGreater(int(row["common_mask_n"]), 0)
            correction_factor = float(
                row["deterministic_legacy_correction_factor"]
            )
            self.assertEqual(correction_factor, 4.0)
            raw_mean = float(row["continuous_legacy_raw_mean"])
            corrected_mean = float(row["continuous_corrected_mean"])
            self.assertTrue(
                np.isclose(
                    raw_mean / correction_factor,
                    corrected_mean,
                    rtol=1e-12,
                    atol=1e-12,
                ),
                msg=(
                    "published correction mismatch for "
                    f"w={row['action_weight']}, {row['archived_condition']}"
                ),
            )

    def test_legacy_raw_divided_by_four_matches_corrected_fixed_scale(self):
        """The deterministic legacy factor-of-four correction is exact at T=4."""

        actions_nm = np.array([-4.0, -2.5, 0.0, 1.25, 4.0])
        relative_speed = np.array([-1.5, 0.4, 2.0, 3.0, -0.2])
        dt_s = 0.01
        legacy_scale = 4.0

        legacy_raw_j = np.where(
            actions_nm * relative_speed > 0,
            np.abs(actions_nm * legacy_scale)
            * np.abs(relative_speed)
            * dt_s,
            0.0,
        )
        corrected_j = positive_actuator_work_increment(
            actions_nm, relative_speed, dt_s
        )
        np.testing.assert_allclose(legacy_raw_j / legacy_scale, corrected_j)

    def test_general_physical_torque_has_joule_units_and_linear_scaling(self):
        torques_nm = np.array([2.0, -3.0, 1.5, -4.0])
        relative_speed = np.array([4.0, -2.0, -5.0, 0.0])
        expected_j = np.array([0.8, 0.6, 0.0, 0.0])

        actual_j = positive_actuator_work_increment(
            torques_nm, relative_speed, 0.1
        )
        np.testing.assert_allclose(actual_j, expected_j)
        np.testing.assert_allclose(
            positive_actuator_work_increment(
                2.0 * torques_nm, relative_speed, 0.1
            ),
            2.0 * expected_j,
        )

    def test_nonpositive_power_is_not_accumulated(self):
        self.assertEqual(positive_actuator_work_increment(2.0, -3.0, 0.1), 0.0)
        self.assertEqual(positive_actuator_work_increment(0.0, 3.0, 0.1), 0.0)

    def test_negative_timestep_is_rejected(self):
        with self.assertRaises(ValueError):
            positive_actuator_work_increment(1.0, 1.0, -0.01)

    def test_success_first_fusion_truth_table(self):
        """Status zero is success; only -2 is failure."""

        negative_status = np.array([-2, 0, -2, -1, 0, -1, 0])
        positive_status = np.array([-2, -2, 0, 1, 1, 0, 1])
        # Failure-side zeros must never win a single-success comparison.
        negative_cmt = np.array([0.0, 0.31, 0.0, 0.24, 0.41, 0.20, 0.17])
        positive_cmt = np.array([0.0, 0.0, 0.29, 0.18, 0.35, 0.20, 0.23])

        selector, selected = select_successful_expert_by_cmt(
            negative_status,
            negative_cmt,
            positive_status,
            positive_cmt,
            tie_break="positive",
        )
        np.testing.assert_array_equal(selector, [-2, -1, 1, 1, 1, 1, -1])
        np.testing.assert_allclose(selected, [-2.0, 0.31, 0.29, 0.18, 0.35, 0.20, 0.17])

    def test_fusion_tie_rule_is_explicit(self):
        self.assertEqual(
            select_successful_expert_by_cmt(0, 0.2, 1, 0.2),
            (1, 0.2),
        )
        self.assertEqual(
            select_successful_expert_by_cmt(
                0, 0.2, 1, 0.2, tie_break="negative"
            ),
            (-1, 0.2),
        )
        with self.assertRaises(ValueError):
            select_successful_expert_by_cmt(0, 0.2, 1, 0.2, tie_break="implicit")

    def test_successful_rollout_requires_valid_cmt(self):
        with self.assertRaises(ValueError):
            select_successful_expert_by_cmt(0, np.nan, -2, 0.0)
        with self.assertRaises(ValueError):
            select_successful_expert_by_cmt(-2, 0.0, 1, -0.1)

    def test_all_comparison_scripts_use_the_shared_accumulator(self):
        scripts = sorted(
            ROOT.glob("action_weight=*/**/Whole_energy_comparison_low_dim*.py")
        )
        # This public patch intentionally contains the 12 canonical evaluators
        # and the single optimized evaluator needed for the archived comparison.
        self.assertEqual(len(scripts), 13)
        optimized = [script for script in scripts if script.stem.endswith("_optimized")]
        self.assertEqual(len(optimized), 1)

        duplicate_scale_patterns = (
            re.compile(r"action_value\s*\*\s*torque"),
            re.compile(r"actions\[positive_work\]\s*\*\s*TORQUE"),
        )
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            self.assertIn(
                "positive_actuator_work_increment",
                source,
                msg=f"shared work metric missing from {script}",
            )
            self.assertIn(
                "select_successful_expert_by_cmt",
                source,
                msg=f"success-first expert fusion missing from {script}",
            )
            for pattern in duplicate_scale_patterns:
                self.assertIsNone(
                    pattern.search(source),
                    msg=f"duplicate torque scale remains in {script}",
                )
            self.assertIn(
                'ENERGY_COMPARISON_DATA_ROOT = os.environ.get(',
                source,
                msg=f"portable data-root configuration missing from {script}",
            )
            self.assertEqual(
                source.count(LEGACY_DATA_ROOT),
                1,
                msg=(
                    "the archived absolute path may appear only as the "
                    f"documented default in {script}"
                ),
            )

    def test_each_canonical_evaluator_writes_negative_cmt_and_outputs_correctly(self):
        canonical_scripts = []
        for weight in WEIGHTS:
            for condition in CONDITIONS:
                script = (
                    ROOT
                    / f"action_weight={weight}"
                    / condition
                    / "Whole_energy_comparison_low_dim.py"
                )
                self.assertTrue(script.is_file(), msg=f"missing evaluator: {script}")
                canonical_scripts.append((weight, condition, script))
        self.assertEqual(len(canonical_scripts), 12)

        negative_write = re.compile(
            r"Cmt_save\(-1,0\)-10-30.*?"
            r"create_dataset\(\s*['\"]Cmt_save['\"]\s*,\s*"
            r"data\s*=\s*(Cmt_save\w+)\s*\)",
            flags=re.DOTALL,
        )
        write_call = re.compile(
            r"h5py\.File\((.*?)\)\s+as\s+h5f", flags=re.DOTALL
        )

        for weight, condition, script in canonical_scripts:
            source = script.read_text(encoding="utf-8")
            active_source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith("#")
            )
            match = negative_write.search(active_source)
            self.assertIsNotNone(match, msg=f"negative Cmt write missing: {script}")
            self.assertEqual(
                match.group(1),
                "Cmt_save0_1",
                msg=f"negative Cmt file receives wrong array: {script}",
            )

            expected_condition_path = (
                f"/action_weight={weight}/{condition}/"
            )
            write_args = [
                match.group(1)
                for match in write_call.finditer(active_source)
                if re.search(r"['\"]w['\"]", match.group(1))
            ]
            self.assertGreater(len(write_args), 0, msg=f"no output writes: {script}")
            for arguments in write_args:
                self.assertIn(
                    "ENERGY_COMPARISON_DATA_ROOT",
                    arguments,
                    msg=f"output does not use the configured data root: {script}",
                )
                self.assertIn(
                    expected_condition_path,
                    arguments,
                    msg=f"output bypasses nominal action-weight folder: {script}",
                )

            envelope_start = active_source.find("select_successful_expert_by_cmt(")
            envelope_write = active_source.find("Cmt_save_passive-10-30")
            self.assertGreaterEqual(
                envelope_start,
                0,
                msg=f"success-first fusion missing: {script}",
            )
            self.assertGreater(
                envelope_write,
                envelope_start,
                msg=f"fused result is not saved after construction: {script}",
            )
            envelope_source = active_source[envelope_start:envelope_write]
            self.assertIn("Cmt_save01", envelope_source)
            self.assertIn("Cmt_save0_1", envelope_source)
            self.assertIn('tie_break="positive"', envelope_source)
            self.assertNotIn("working_save0_1[i, j, k, ll] == 0", envelope_source)


if __name__ == "__main__":
    unittest.main()
