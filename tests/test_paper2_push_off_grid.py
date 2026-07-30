from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/paper2_push_off_grid_12case"


class Paper2PushOffGridTests(unittest.TestCase):
    def test_release_shape_and_summary(self) -> None:
        validation = json.loads(
            (RESULT_DIR / "validation_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(validation["all_checks_passed"])
        self.assertEqual(validation["candidate_count"], 518_400)
        self.assertEqual(validation["controller_case_count"], 36)
        self.assertEqual(validation["fixed_replay_count"], 36)
        self.assertEqual(len(validation["boundary_optima"]), 1)

        with (RESULT_DIR / "accepted_cases.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(int(row["completed_steps"]) == 3 for row in rows))
        self.assertTrue(all(row["cmt_reproduced"] == "True" for row in rows))

    def test_portable_validator(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src/paper2/push_off_grid/validate_release.py"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('"all_checks_passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
