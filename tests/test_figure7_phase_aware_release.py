"""Integrity checks for the phase-aware 2,160-case Figure 7 record."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "four_link_phase_aware_primary_2160"
SCRIPT = ROOT / "analysis" / "reconstruct_phase_aware_primary_2160.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Figure7PhaseAwareReleaseTests(unittest.TestCase):
    def test_file_checksums(self) -> None:
        for line in (RESULTS / "CHECKSUMS.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(sha256(RESULTS / name), expected)

    def test_controller_summary(self) -> None:
        summary = json.loads(
            (RESULTS / "controller_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["n_all"], 2160)
        self.assertEqual(summary["n_success"], 987)
        self.assertEqual(summary["n_valid_cmt"], 492)
        self.assertAlmostEqual(summary["mean_cmt_valid"], 0.6809030264976478)

    def test_active_pair_summary(self) -> None:
        pair = json.loads(
            (RESULTS / "active_pair_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(pair["both_valid_n"], 445)
        self.assertEqual(pair["proposed_lower_count"], 231)
        self.assertAlmostEqual(pair["proposed_lower_fraction"], 231 / 445)

    def test_route_assignments(self) -> None:
        with (RESULTS / "route_assignments.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2160)
        self.assertEqual(
            Counter(row["route"] for row in rows),
            Counter({"active": 1398, "positive": 259, "negative": 503}),
        )

    def test_reconstruction_source_is_repository_relative(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("REPO_ROOT = Path(__file__).resolve().parents[1]", source)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\\\/]", source))


if __name__ == "__main__":
    unittest.main()
