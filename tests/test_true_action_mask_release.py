from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/four_link_true_action_mask_scratch_c090_epoch1275.json"
DATA_DIR = ROOT / "data/four_link/true_action_mask_scratch_c090_epoch1275"
STATS_DIR = ROOT / "results/four_link_statistics/true_action_mask_scratch_c090_epoch1275"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class TrueActionMaskReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_source_model_and_log_hashes(self) -> None:
        for record in self.config["source_files"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(sha256(path), record["sha256"])
        for record in self.config["checkpoint_files"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(sha256(path), record["sha256"])
        for record in self.config["training_logs"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(sha256(path), record["sha256"])

    def test_training_snapshot_is_scratch_and_contiguous(self) -> None:
        csv_path = ROOT / self.config["training_logs"]["csv"]["path"]
        rows = read_rows(csv_path)
        self.assertEqual(len(rows), 1307)
        self.assertEqual([int(row["epoch"]) for row in rows], list(range(1, 1308)))
        self.assertEqual({row["initialization"] for row in rows}, {"scratch"})
        self.assertEqual(
            {row["checkpoint_source_kind"] for row in rows},
            {"random_initialization"},
        )
        self.assertEqual({int(row["action_dim_before_mask"]) for row in rows}, {27})
        self.assertEqual({int(row["valid_actions_after_mask"]) for row in rows}, {18})
        self.assertEqual({float(row["action_value_weight"]) for row in rows}, {0.026})
        self.assertEqual(rows[1274]["curriculum_bucket"], "90")
        self.assertEqual(rows[1274]["success_count"], "250")

    def test_evaluator_is_portable_and_preserves_v22_3_gate(self) -> None:
        evaluator = ROOT / self.config["source_files"]["formal_evaluator"]["path"]
        text = evaluator.read_text(encoding="utf-8")
        self.assertNotIn("D:\\L&S\\", text)
        self.assertNotIn("C:\\Users\\", text)
        self.assertIn("REPO_ROOT = Path(__file__).resolve().parents[3]", text)
        self.assertIn(
            "touchdown_safe_q_low = np.deg2rad(np.array([35.0, 8.0, -115.0, 5.0]",
            text,
        )
        self.assertIn('"true_action_mask_scratch_c090_epoch1275_rerun"', text)
        self.assertIn("--check-only", text)
        self.assertIn("--metadata-only", text)

    def test_formal_output_is_complete_non_smoke_and_paired(self) -> None:
        controllers = {row["controller"]: row for row in read_rows(DATA_DIR / "controller_summary.csv")}
        expected = {
            "active": (975, 458),
            "active_full": (984, 474),
            "per_step_sign": (983, 473),
            "true_action_mask_scratch": (946, 430),
            "passive_sign_selector": (964, 480),
            "passive_oracle_envelope": (989, 508),
        }
        self.assertEqual(set(controllers), set(expected))
        for controller, (success, valid) in expected.items():
            row = controllers[controller]
            self.assertEqual(int(row["n_all"]), 2160)
            self.assertEqual(int(row["n_success"]), success)
            self.assertEqual(int(row["n_valid_cmt"]), valid)
            self.assertNotIn("smoke", row["eval_label"].lower())

        pairs = read_rows(DATA_DIR / "selector_vs_true_action_mask_trials.csv")
        self.assertEqual(len(pairs), 2160)
        key_fields = (
            "seed",
            "episode_index",
            "walk_direction",
            "reset_phase",
            "commanded_step_length_m",
            "commanded_step_height_m",
        )
        self.assertEqual(len({tuple(row[field] for field in key_fields) for row in pairs}), 2160)
        self.assertEqual({row["reference_controller"] for row in pairs}, {"true_action_mask_scratch"})
        self.assertEqual({row["comparison_controller"] for row in pairs}, {"passive_sign_selector"})
        self.assertEqual(
            sum(row["reference_valid"] == "True" and row["comparison_valid"] == "True" for row in pairs),
            395,
        )

    def test_evaluation_manifest_and_local_checksums(self) -> None:
        manifest = json.loads((DATA_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["release_id"], "true_action_mask_scratch_c090_epoch1275")
        self.assertEqual(manifest["evaluation"]["case_count_per_controller"], 2160)
        source = manifest["source"]
        self.assertEqual(sha256(ROOT / source["path"]), source["sha256"])
        for relative, expected in manifest["frozen_policy_files"].items():
            self.assertEqual(sha256(ROOT / relative), expected)
        for name, expected in manifest["result_csv_sha256"].items():
            self.assertEqual(sha256(DATA_DIR / name), expected)

        checksum_lines = (DATA_DIR / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(checksum_lines), 23)
        for line in checksum_lines:
            expected, name = line.split("  ", 1)
            self.assertEqual(sha256(DATA_DIR / name), expected)

    def test_paired_statistical_outputs(self) -> None:
        result = json.loads((STATS_DIR / "paired_inference.json").read_text(encoding="utf-8"))
        self.assertEqual(result["success"]["reference_only"], 64)
        self.assertEqual(result["success"]["comparison_only"], 82)
        self.assertAlmostEqual(
            result["success"]["comparison_minus_reference_risk_difference"],
            18 / 2160,
        )
        self.assertEqual(result["valid_cmt"]["reference_only"], 35)
        self.assertEqual(result["valid_cmt"]["comparison_only"], 85)
        self.assertEqual(result["both_valid_cmt"]["n_both_valid"], 395)
        self.assertEqual(result["both_valid_cmt"]["comparison_lower_count"], 354)
        self.assertAlmostEqual(
            result["both_valid_cmt"]["fraction_comparison_lower_cmt"],
            354 / 395,
        )
        for line in (STATS_DIR / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(sha256(STATS_DIR / name), expected)

    def test_release_text_has_no_workstation_absolute_paths(self) -> None:
        text_paths = [
            CONFIG_PATH,
            ROOT / "docs/TRUE_ACTION_MASK_BASELINE.md",
            ROOT / "analysis/four_link_true_action_mask_paired_inference.py",
            ROOT / "src/four_link/training/v22_3_v9_true_action_mask_400_grid_common.py",
            ROOT / "src/four_link/training/v22_3_v9_true_action_mask_scratch_400_grid.py",
            ROOT / "src/four_link/evaluation/paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py",
        ]
        for path in text_paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("C:\\Users\\", text, path)
            self.assertNotIn("D:\\L&S\\", text, path)


if __name__ == "__main__":
    unittest.main()
