from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceArtifactTests(unittest.TestCase):
    def test_table1_hashes_and_counts(self) -> None:
        config = json.loads((ROOT / "configs/table1.json").read_text(encoding="utf-8"))
        for item in config["files"].values():
            self.assertEqual(sha256(ROOT / item["path"]), item["sha256"])
        with (ROOT / "results/table1_figure4/table_i.csv").open(newline="", encoding="utf-8") as handle:
            rows = {row["controller_or_set"]: row for row in csv.DictReader(handle)}
        self.assertEqual(int(rows["Active PPO"]["feasible_states"]), 630328)
        self.assertEqual(int(rows["Sign-constrained union"]["feasible_states"]), 582861)
        self.assertEqual(int(rows["Common feasible intersection"]["feasible_states"]), 534042)

    def test_four_link_config_hashes(self) -> None:
        config = json.loads(
            (ROOT / "configs/four_link_legacy_manuscript.json").read_text(encoding="utf-8")
        )
        reconstruction = config["source_status"]["reconstruction"]
        self.assertEqual(sha256(ROOT / reconstruction["path"]), reconstruction["sha256"])
        for path, expected, _weight in config["training_sources"].values():
            if path.startswith("@V1.0.0:"):
                self.assertEqual(
                    config["source_status"]["legacy_training_source_revision"]["commit"],
                    "77f2f7d69df183344ab06ae55e924bd071646393",
                )
                self.assertEqual(
                    expected,
                    "6b86eb447b03aa836f5b41e9548cd6d0e7491139d91d7579a094cb4059014fde",
                )
                continue
            self.assertEqual(sha256(ROOT / path), expected)
        for path, expected in config["checkpoints"].values():
            self.assertEqual(sha256(ROOT / path), expected)

        active_full = (
            ROOT / "src/four_link/training/v22_3_v9_active_full_400_grid.py"
        ).read_text(encoding="utf-8")
        self.assertIn("action_value_weight = 0\n", active_full)
        self.assertIn("def assert_fresh_scratch_output_dir():", active_full)
        self.assertNotIn("load_previous_model", active_full)
        self.assertNotIn("torch.load(", active_full)
        root_compatibility_copy = (ROOT / "v22_3_v9_active_full_400_grid.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("action_value_weight = 0\n", root_compatibility_copy)
        self.assertIn("def assert_fresh_scratch_output_dir():", root_compatibility_copy)
        self.assertNotIn("load_previous_model", root_compatibility_copy)
        self.assertNotIn("torch.load(", root_compatibility_copy)
        self.assertFalse((ROOT / "configs/four_link_v23_success.json").exists())

    def test_four_link_reconstruction_validation_record(self) -> None:
        record = json.loads(
            (
                ROOT
                / "results/four_link_statistics/legacy_manuscript/reconstruction_validation.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(record["all_exact"])
        self.assertEqual(record["exact_sha256_matches"], 19)
        self.assertEqual(len(record["files"]), 19)
        for name, expected in record["files"].items():
            self.assertEqual(
                sha256(ROOT / "data/four_link/legacy_manuscript" / name), expected
            )

    def test_four_link_success_counts_and_pairs(self) -> None:
        with (ROOT / "data/four_link/legacy_manuscript/controller_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            summary = {row["controller"]: row for row in csv.DictReader(handle)}
        self.assertEqual(int(summary["active"]["n_success"]), 975)
        self.assertEqual(int(summary["passive_sign_selector"]["n_success"]), 964)
        self.assertEqual(int(summary["passive_oracle_envelope"]["n_success"]), 989)
        self.assertEqual(int(summary["active_full"]["n_success"]), 984)

        with (ROOT / "data/four_link/legacy_manuscript/paired_trials.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["passive_controller"] == "passive_sign_selector"]
        self.assertEqual(len(rows), 2160)
        both_valid = sum(row["active_valid"] == "True" and row["passive_valid"] == "True" for row in rows)
        self.assertEqual(both_valid, 419)
        pairing = json.loads(
            (ROOT / "results/four_link_statistics/legacy_manuscript/pairing_validation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(pairing["case_key_sets_equal"])
        self.assertEqual(pairing["initial_state_hash_mismatches"], 0)
        self.assertEqual(pairing["both_valid_for_cmt"], 419)

    def test_paper2_readonly_audit_and_current_tables(self) -> None:
        audit_csv = ROOT / "data/paper2/readonly_rerun/paper2_rerun_results.csv"
        with audit_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(row["status"] == "ok" for row in rows), 59)
        self.assertEqual(sum(row["status"] == "timeout" for row in rows), 1)
        self.assertTrue(all(row["source_tree_changed"] == "false" for row in rows))
        self.assertNotIn("C:\\Users\\", audit_csv.read_text(encoding="utf-8-sig"))
        self.assertNotIn("D:\\L&S\\", audit_csv.read_text(encoding="utf-8-sig"))

        folder_name = {
            "flat_L1145": "flat_1.145",
            "raised_L1145": "raised_1.145",
            "flat_L128": "flat_1.28",
            "raised_L128": "raised_1.28",
        }
        for row in rows:
            prefix = next(key for key in folder_name if row["case_id"].startswith(key))
            entry_dir = folder_name[prefix] + row["case_id"][len(prefix):]
            source = ROOT / "src/paper2/entrypoints" / entry_dir / row["script"]
            self.assertEqual(sha256(source), row["source_sha256"])

        with (ROOT / "results/paper2_current/table_v_current.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            table_v = {row["method"]: row for row in csv.DictReader(handle)}
        self.assertEqual(int(table_v["Proposed one-sided selector"]["n_cmt"]), 12)
        self.assertEqual(int(table_v["Proposed one-sided selector"]["n_landing_error"]), 11)
        self.assertAlmostEqual(float(table_v["Continuous active MPC"]["mean_cmt"]), 0.186645)
        self.assertAlmostEqual(float(table_v["Proposed one-sided selector"]["mean_cmt"]), 0.123472869728)
        self.assertEqual(int(table_v["TVLQR tracking"]["n_cmt"]), 12)
        self.assertAlmostEqual(float(table_v["TVLQR tracking"]["mean_cmt"]), 0.232348)

    def test_tvlqr_release_hashes_and_seeded_results(self) -> None:
        config = json.loads((ROOT / "configs/tvlqr_release.json").read_text(encoding="utf-8"))
        for record in config["models"].values():
            self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])
        with (ROOT / "results/tvlqr_seed0/tvlqr_cases.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertTrue(all(row["seed_override"] == "0" for row in rows))
        self.assertAlmostEqual(
            sum(float(row["cmt"]) for row in rows) / len(rows), 0.232348
        )
        with (ROOT / "data/paper2/readonly_rerun/paper2_rerun_results.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            audit = {
                row["case_id"]: row["source_sha256"]
                for row in csv.DictReader(handle)
                if row["method"] == "TVLQR tracking"
            }
        aliases = {
            "flat_1.145": "flat_L1145",
            "raised_1.145": "raised_L1145",
            "flat_1.28": "flat_L128",
            "raised_1.28": "raised_L128",
        }
        for row in rows:
            prefix = next(name for name in aliases if row["case_id"].startswith(name))
            audit_id = aliases[prefix] + row["case_id"][len(prefix):]
            source = ROOT / "src/paper2/entrypoints" / row["case_id"] / "LQR.py"
            self.assertEqual(sha256(source), audit[audit_id])


if __name__ == "__main__":
    unittest.main()
