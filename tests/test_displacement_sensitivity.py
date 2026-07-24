import csv
import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "four_link_paired_inference.py"
PAIRED = (
    ROOT
    / "data"
    / "four_link"
    / "true_action_mask_scratch_c090_epoch1275"
    / "paired_trials.csv"
)


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("four_link_paired_inference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DisplacementSensitivityTest(unittest.TestCase):
    def test_released_active_selector_threshold_summary(self) -> None:
        module = load_analysis_module()
        with PAIRED.open(newline="", encoding="utf-8-sig") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["passive_controller"] == "passive_sign_selector"
            ]

        result = module.displacement_sensitivity(rows, [0.001, 0.005, 0.010, 0.020])
        self.assertEqual([row["n_both_valid"] for row in result], [419, 313, 192, 70])
        self.assertEqual(
            [row["selector_lower_count"] for row in result],
            [334, 255, 160, 64],
        )
        expected_means = [
            (1.4168741659, 0.3706641632),
            (0.9253951212, 0.2972669279),
            (0.9242324868, 0.3473463574),
            (1.0500682473, 0.3923736636),
        ]
        for row, (active, selector) in zip(result, expected_means):
            self.assertTrue(math.isclose(row["active_mean_cmt"], active, abs_tol=1e-10))
            self.assertTrue(math.isclose(row["selector_mean_cmt"], selector, abs_tol=1e-10))


if __name__ == "__main__":
    unittest.main()
