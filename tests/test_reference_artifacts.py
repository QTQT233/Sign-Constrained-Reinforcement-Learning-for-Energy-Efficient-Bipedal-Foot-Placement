from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOUR_LINK_DIR = ROOT / "data/four_link/true_action_mask_scratch_c090_epoch1275"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class ReferenceArtifactTests(unittest.TestCase):
    def test_table1_hashes_and_counts(self) -> None:
        config = json.loads((ROOT / "configs/table1.json").read_text(encoding="utf-8"))
        for item in config["files"].values():
            self.assertEqual(sha256(ROOT / item["path"]), item["sha256"])
        rows = {
            row["controller_or_set"]: row
            for row in read_rows(ROOT / "results/table1/table_i.csv")
        }
        self.assertEqual(int(rows["Active PPO"]["feasible_states"]), 630328)
        self.assertEqual(int(rows["Sign-constrained union"]["feasible_states"]), 582861)
        self.assertEqual(int(rows["Common feasible intersection"]["feasible_states"]), 534042)

    def test_table2_uses_strict_archived_displacement_mask(self) -> None:
        rows = read_rows(ROOT / "results/action_weight_table_ii_seed_20260716.csv")
        indexed = {(row["action_weight"], row["method"]): row for row in rows}
        counts = indexed[("0.02", "Common-mask n")]
        self.assertEqual(
            [int(counts[column]) for column in (
                "0.00 m step; 0.25 m advance",
                "0.00 m step; 0.17 m advance",
                "0.01 m step; 0.25 m advance",
                "0.01 m step; 0.17 m advance",
            )],
            [38917, 20502, 36702, 19776],
        )
        manifest = json.loads(
            (ROOT / "results/action_weight_rerun_seed_20260716_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("Archived com_displacement_m", manifest["displacement_source"])
        self.assertIn("no zero-work exception", manifest["displacement_source"])
        self.assertIn("strict recomputation", manifest["archive_masks"])

    def test_current_four_link_archive_and_scratch_sources(self) -> None:
        config = json.loads(
            (ROOT / "configs/four_link_true_action_mask_scratch_c090_epoch1275.json").read_text(
                encoding="utf-8"
            )
        )
        for record in config["source_files"].values():
            self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])
        for record in config["checkpoint_files"].values():
            self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])

        summary = {row["controller"]: row for row in read_rows(FOUR_LINK_DIR / "controller_summary.csv")}
        self.assertEqual(int(summary["active"]["n_success"]), 975)
        self.assertEqual(int(summary["true_action_mask_scratch"]["n_success"]), 946)
        self.assertEqual(int(summary["passive_sign_selector"]["n_success"]), 964)
        self.assertEqual(int(summary["passive_oracle_envelope"]["n_success"]), 989)

        pairs = read_rows(FOUR_LINK_DIR / "paired_trials.csv")
        selector = [row for row in pairs if row["passive_controller"] == "passive_sign_selector"]
        self.assertEqual(len(selector), 2160)
        self.assertEqual(
            sum(row["active_valid"] == "True" and row["passive_valid"] == "True" for row in selector),
            419,
        )

        for relative in (
            "src/four_link/training/v22_3_v9_active_full_400_grid.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("action_value_weight = 0\n", text)
            self.assertIn("def assert_fresh_scratch_output_dir():", text)
            self.assertNotIn("load_previous_model", text)
            self.assertNotIn("torch.load(", text)

    def test_paper2_current_tables_and_primary_csv_are_synchronized(self) -> None:
        current_rows = read_rows(ROOT / "results/paper2_current/paper2_combined_cases.csv")
        current = {(row["case_id"], row["method"]): row for row in current_rows}
        self.assertEqual(len(current_rows), 72)
        self.assertEqual(
            float(current[("raised_L1145_r1", "Unrestricted discrete PPO")]["cmt"]),
            0.25098054963443095,
        )
        self.assertEqual(
            float(current[("raised_L1145_r1", "Continuous-torque PPO")]["cmt"]),
            0.25817243332390905,
        )

        primary = {row["case_id"]: row for row in read_rows(ROOT / "results/two_link_primary_12_cases.csv")}
        self.assertEqual(set(primary), {row["case_id"] for row in current_rows})
        self.assertEqual(
            float(primary["raised_L1145_r1"]["proposed_selector_cmt"]),
            0.14785439168930947,
        )
        self.assertEqual(
            float(primary["raised_L1145_r1"]["discrete_active_ppo_cmt"]),
            0.25098054963443095,
        )
        self.assertEqual(
            float(primary["raised_L1145_r1"]["continuous_active_ppo_cmt"]),
            0.25817243332390905,
        )

        table_v = {row["method"]: row for row in read_rows(ROOT / "results/paper2_current/table_v_current.csv")}
        self.assertAlmostEqual(float(table_v["Transition-start lookup router"]["mean_cmt"]), 0.122346306417)
        self.assertAlmostEqual(float(table_v["Unrestricted discrete PPO"]["mean_cmt"]), 0.226620152405)
        self.assertAlmostEqual(float(table_v["Continuous-torque PPO"]["mean_cmt"]), 0.238107711255)
        self.assertAlmostEqual(float(table_v["Continuous-torque MPC"]["mean_cmt"]), 0.198945098769)
        expected_landing_mae = {
            "Unrestricted discrete PPO": 0.044260284304,
            "Continuous-torque PPO": 0.040952732815,
            "Continuous-torque MPC": 0.030284639682,
            "Transition-start lookup router": 0.041389074712,
        }
        for method, expected in expected_landing_mae.items():
            self.assertEqual(int(table_v[method]["n_landing_transitions"]), 36)
            self.assertAlmostEqual(
                float(table_v[method]["pooled_foot_placement_mae_m"]),
                expected,
                places=12,
            )

    def test_tvlqr_release_hashes_and_seeded_results(self) -> None:
        config = json.loads((ROOT / "configs/tvlqr_release.json").read_text(encoding="utf-8"))
        for record in config["models"].values():
            self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])
        rows = read_rows(ROOT / "results/tvlqr_seed0/tvlqr_cases.csv")
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertAlmostEqual(sum(float(row["cmt"]) for row in rows) / 12, 0.234644083333)


if __name__ == "__main__":
    unittest.main()
