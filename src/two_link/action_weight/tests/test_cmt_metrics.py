"""Regression tests for the continuous-controller Cmt work accumulator."""

from __future__ import annotations

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

from cmt_metrics import positive_actuator_work_increment


class PositiveActuatorWorkTests(unittest.TestCase):
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
            for pattern in duplicate_scale_patterns:
                self.assertIsNone(
                    pattern.search(source),
                    msg=f"duplicate torque scale remains in {script}",
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

            expected_prefix = (
                f"Energy_Comparison/action_weight={weight}/{condition}/"
            )
            write_args = [
                match.group(1)
                for match in write_call.finditer(active_source)
                if re.search(r"['\"]w['\"]", match.group(1))
            ]
            self.assertGreater(len(write_args), 0, msg=f"no output writes: {script}")
            for arguments in write_args:
                self.assertIn(
                    expected_prefix,
                    arguments,
                    msg=f"output bypasses nominal action-weight folder: {script}",
                )


if __name__ == "__main__":
    unittest.main()
