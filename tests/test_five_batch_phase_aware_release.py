"""Integrity tests for the frozen five-batch phase-aware confirmation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "supplementary"
    / "S4"
    / "five_batch_phase_aware_confirmation"
)
RESULTS = BUNDLE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FiveBatchPhaseAwareReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (BUNDLE / "protocol.json").read_text(encoding="utf-8")
        )
        cls.report = json.loads(
            (RESULTS / "final_confirmation_report.json").read_text(
                encoding="utf-8"
            )
        )

    def test_protocol_inventory(self) -> None:
        seeds = [
            int(seed)
            for values in self.protocol["batches"].values()
            for seed in values
        ]
        self.assertEqual(len(seeds), 15)
        self.assertEqual(len(set(seeds)), 15)
        self.assertEqual(
            self.protocol["design"]["total_cases_per_controller"], 10800
        )
        self.assertEqual(
            self.protocol["analysis"]["bootstrap_replicates"], 20000
        )

    def test_frozen_artifact_hashes(self) -> None:
        frozen = self.protocol["frozen_sha256"]
        observed = {
            "evaluator": sha256(
                ROOT
                / "src"
                / "four_link"
                / "evaluation"
                / "paper_four_link_three_expert_selector_evaluation.py"
            ),
            "active_policy": sha256(
                ROOT
                / "models"
                / "four_link"
                / "v22_3_v9_bi_400_grid"
                / "V22_3V9Bi400Grid_c090_Policy_best.pth"
            ),
            "non_negative_policy": sha256(
                ROOT
                / "models"
                / "four_link"
                / "v22_3_v9_uni_pos_400_grid"
                / "V22_3V9UniPos400Grid_c097_Policy_best.pth"
            ),
            "non_positive_policy": sha256(
                ROOT
                / "models"
                / "four_link"
                / "v22_3_v9_uni_neg_400_grid"
                / "V22_3V9UniNeg400Grid_c100_Policy_best.pth"
            ),
            "released_two_expert_gate": sha256(
                ROOT
                / "models"
                / "four_link"
                / "paper_four_link_passive_sign_selector_v22_3_grid.pth"
            ),
            "hard_mask_policy": sha256(
                ROOT
                / "models"
                / "four_link"
                / "v22_3_v9_true_action_mask_scratch_400_grid"
                / "V22_3V9TrueActionMaskScratch400Grid_c090_epoch1275_success250_Policy.pth"
            ),
        }
        self.assertEqual(observed, frozen)

    def test_final_acceptance_and_counts(self) -> None:
        self.assertTrue(
            self.report["all_primary_acceptance_criteria_met"]
        )
        self.assertEqual(self.report["n_cases_per_controller"], 10800)
        controllers = self.report["controllers"]
        self.assertEqual(controllers["active"]["success_count"], 5229)
        self.assertEqual(
            controllers["proposed_phase_aware_three_expert"][
                "success_count"
            ],
            5247,
        )
        self.assertEqual(
            controllers["true_hard_mask"]["success_count"], 5030
        )

    def test_matched_record_design(self) -> None:
        path = RESULTS / "matched_trials_10800.csv.gz"
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10800)
        keys = {
            (int(row["seed"]), int(row["episode_index"])) for row in rows
        }
        self.assertEqual(len(keys), 10800)
        strata = Counter(
            (
                int(row["seed"]),
                float(row["walk_direction"]),
                float(row["commanded_step_length_m"]),
                float(row["commanded_step_height_m"]),
            )
            for row in rows
        )
        self.assertEqual(len(strata), 270)
        self.assertEqual(set(strata.values()), {40})

    def test_result_checksums(self) -> None:
        expected = {}
        for line in (RESULTS / "SHA256SUMS.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
        for relative, digest in expected.items():
            self.assertEqual(sha256(RESULTS / relative), digest)


if __name__ == "__main__":
    unittest.main()
