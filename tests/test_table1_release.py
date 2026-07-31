from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TableIReleaseTest(unittest.TestCase):
    def test_complete_reachability_partition(self) -> None:
        path = ROOT / "results" / "table1" / "table_i.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = {row["controller_or_set"]: row for row in csv.DictReader(handle)}

        expected = {
            "Unrestricted discrete PPO": 630_328,
            "One-sided expert union": 582_861,
            "Proposed three-expert route": 679_147,
            "Unrestricted discrete PPO and one-sided-expert union": 534_042,
            "Unrestricted discrete PPO only": 96_286,
            "One-sided-expert union only": 48_819,
            "Neither": 130_853,
        }
        self.assertEqual(set(rows), set(expected))
        for name, count in expected.items():
            self.assertEqual(int(rows[name]["sampled_states"]), 810_000)
            self.assertEqual(int(rows[name]["feasible_states"]), count)

        self.assertEqual(
            expected["Unrestricted discrete PPO and one-sided-expert union"]
            + expected["Unrestricted discrete PPO only"]
            + expected["One-sided-expert union only"]
            + expected["Neither"],
            810_000,
        )
        self.assertEqual(
            expected["Proposed three-expert route"],
            expected["Unrestricted discrete PPO and one-sided-expert union"]
            + expected["Unrestricted discrete PPO only"]
            + expected["One-sided-expert union only"],
        )


if __name__ == "__main__":
    unittest.main()
