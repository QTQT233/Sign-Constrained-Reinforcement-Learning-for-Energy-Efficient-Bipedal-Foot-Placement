from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = (
    ROOT
    / "models/four_link/three_expert_selector_v22_3"
    / "V22_3_three_expert_selector.pth"
)
TRAINING = ROOT / "results/four_link_three_expert_training"
PRIMARY = ROOT / "results/four_link_three_expert_statistics"
POOLED = ROOT / "results/four_link_three_expert_additional_batch_statistics"
INCREMENTAL = ROOT / "results/four_link_three_expert_incremental_comparison"
ANALYSIS = ROOT / "analysis/four_link_three_vs_two_and_fallback.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        context = gzip.open(path, mode="rt", newline="", encoding="utf-8-sig")
    else:
        context = path.open(newline="", encoding="utf-8-sig")
    with context as handle:
        return list(csv.DictReader(handle))


class ThreeExpertReleaseTests(unittest.TestCase):
    def test_checkpoint_and_training_record(self) -> None:
        self.assertEqual(
            sha256(MODEL),
            "d7f63ce349fda4f670a343f8b579f1512420616471d3ec4650e3c6c07105b954",
        )
        training = json.loads(
            (TRAINING / "training_summary.json").read_text(encoding="utf-8")
        )
        reconstruction = json.loads(
            (TRAINING / "reconstruction_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(training["transition_locked"])
        self.assertTrue(training["experts_frozen"])
        self.assertEqual(training["best_epoch"], 119)
        self.assertAlmostEqual(training["best_validation_accuracy"], 0.8697222222)
        self.assertAlmostEqual(
            training["best_validation_balanced_accuracy"], 0.8502300890
        )
        self.assertEqual(reconstruction["candidate_count"], 75600)
        self.assertEqual(
            sum(reconstruction["split_class_counts"]["train"].values()), 64800
        )
        self.assertEqual(
            sum(reconstruction["split_class_counts"]["validation"].values()),
            10800,
        )

    def test_primary_and_pooled_statistics_exist(self) -> None:
        primary = json.loads(
            (PRIMARY / "manuscript_statistics.json").read_text(encoding="utf-8")
        )
        pooled = json.loads(
            (POOLED / "multibatch_statistics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(primary["case_count_per_controller"], 2160)
        self.assertEqual(
            int(primary["controllers"]["three_expert_selector"]["n_success"]),
            970,
        )
        self.assertEqual(
            pooled["pooled_controllers"]["three_expert_selector"]["n_success"],
            4993,
        )
        self.assertEqual(
            pooled["pooled_controllers"]["three_expert_selector"]["n_valid_cmt"],
            2681,
        )

    def test_incremental_summary_and_pairing_provenance(self) -> None:
        summary = json.loads(
            (INCREMENTAL / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["n_trials"], 10800)
        self.assertEqual(summary["two_expert"]["success_count"], 4996)
        self.assertEqual(summary["three_expert"]["success_count"], 4993)
        self.assertEqual(summary["two_expert"]["valid_cmt_count"], 2611)
        self.assertEqual(summary["three_expert"]["valid_cmt_count"], 2681)
        self.assertEqual(summary["common_valid_cmt"]["count"], 2554)
        self.assertEqual(
            summary["active_fallback"]["strict_opportunity_count"], 5253
        )
        self.assertEqual(
            summary["active_fallback"]["strict_success_rescue_count"], 122
        )
        self.assertEqual(
            summary["active_fallback"]["strict_valid_cmt_rescue_count"], 50
        )

        provenance = json.loads(
            (INCREMENTAL / "provenance_validation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(provenance["matched_key_count"], 10800)
        self.assertEqual(set(provenance["mismatch_counts"].values()), {0})
        self.assertLessEqual(
            provenance["max_active_cmt_abs_diff"],
            provenance["float_tolerance"],
        )
        self.assertEqual(len(provenance["three_expert_inputs"]), 5)

    def test_compact_trials_and_strict_rescues(self) -> None:
        rows = read_csv(INCREMENTAL / "paired_trials_compact.csv.gz")
        self.assertEqual(len(rows), 10800)
        self.assertEqual(
            len(
                {
                    (row["batch"], row["seed"], row["episode_index"])
                    for row in rows
                }
            ),
            10800,
        )
        rescues = read_csv(INCREMENTAL / "strict_success_rescue_trials.csv")
        self.assertEqual(len(rescues), 122)
        self.assertEqual({row["three_success"] for row in rescues}, {"True"})
        self.assertEqual({row["two_success"] for row in rescues}, {"False"})
        self.assertEqual({row["oracle_success"] for row in rescues}, {"False"})
        self.assertEqual({row["two_valid"] for row in rescues}, {"False"})
        self.assertEqual(
            sum(row["three_valid"] == "True" for row in rescues),
            50,
        )
        self.assertTrue(
            all(
                row["selected_branch"].startswith("three_expert_active")
                for row in rescues
            )
        )

    def test_compact_checker_and_release_cleanliness(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ANALYSIS),
                "--check-published",
                str(INCREMENTAL),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("PASS (10800 matched trials)", completed.stdout)
        public_text = [
            ANALYSIS,
            INCREMENTAL / "README.md",
            INCREMENTAL / "summary.json",
            INCREMENTAL / "provenance_validation.json",
        ]
        for path in public_text:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("C:\\Users\\", text, path)
            self.assertNotIn("D:\\L&S\\", text, path)
        self.assertFalse((ROOT / "scripts/analysis").exists())
        self.assertFalse((ROOT / "results/figures").exists())
        self.assertFalse(
            (ROOT / "results/four_link_three_expert_additional_batches").exists()
        )


if __name__ == "__main__":
    unittest.main()
