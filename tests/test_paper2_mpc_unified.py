from __future__ import annotations

import ast
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper2_mpc_unified_12_cases.json"
RESULT_DIR = ROOT / "results/paper2_mpc_unified_12case"
SOURCE_ALIGNED_DIR = ROOT / "results/paper2_source_aligned_r1_rerun_20260721"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assigned_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1:
        raise AssertionError(f"Expected one literal {name} assignment in {path}, found {len(values)}")
    return values[0]


class UnifiedMPCReleaseTests(unittest.TestCase):
    def test_pinned_sources_and_case_config(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["schema"], "paper2-mpc-unified-12case-v1")
        self.assertEqual(len(config["cases"]), 12)
        self.assertEqual(len({item["case_id"] for item in config["cases"]}), 12)
        for path, expected in config["source_files"].items():
            self.assertEqual(sha256(ROOT / path), expected)

        for item in config["cases"]:
            for filename in ("Qi_multi_passive_sim.py", "Multi_continuous.py"):
                path = ROOT / "src/paper2/entrypoints" / item["entrypoint_dir"] / filename
                self.assertEqual(tuple(assigned_literal(path, "init_idx")), tuple(item["initial_idx"]))

    def test_r1_index_corrections_do_not_change_continuous_reset(self) -> None:
        flat_qi = ROOT / "src/paper2/entrypoints/flat_1.145_r1/Qi_multi_passive_sim.py"
        raised_dir = ROOT / "src/paper2/entrypoints/raised_1.145_r1"
        raised_qi = raised_dir / "Qi_multi_passive_sim.py"
        raised_continuous = raised_dir / "Multi_continuous.py"
        self.assertEqual(assigned_literal(flat_qi, "init_idx"), (19, 11, 15, 7))
        for filename in (
            "Qi_multi_passive_sim.py",
            "Qi_multi_ac_discrete_sim.py",
            "Multi_continuous.py",
            "LIPM.py",
            "LQR.py",
        ):
            self.assertEqual(assigned_literal(raised_dir / filename, "init_idx"), (19, 11, 15, 1))
        source = raised_continuous.read_text(encoding="utf-8-sig")
        self.assertIn("dtheta1_new -= 0.87", source)
        self.assertIn("dtheta1_new -= 0.89", source)

    def test_accepted_results_and_legacy_boundary(self) -> None:
        with (RESULT_DIR / "accepted_cases.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["method"] == "Continuous-torque MPC" for row in rows))
        self.assertTrue(all(row["success"] == "true" for row in rows))
        self.assertTrue(all(row["replay_1_exact"] == "true" for row in rows))
        self.assertTrue(all(row["replay_2_exact"] == "true" for row in rows))
        values = [float(row["cmt"]) for row in rows]
        self.assertAlmostEqual(statistics.fmean(values), 0.19894509876853902, places=14)
        self.assertAlmostEqual(statistics.stdev(values), 0.021963452200466357, places=14)
        absolute_errors = [float(row["absolute_foot_error_m"]) for row in rows]
        self.assertTrue(
            all(
                value == abs(float(row["foot_error_m"]))
                for value, row in zip(absolute_errors, rows)
            )
        )
        self.assertAlmostEqual(statistics.fmean(absolute_errors), 0.06382515890169439, places=14)
        summary = json.loads((RESULT_DIR / "accepted_summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            summary["overall_landing_error"]["mean_absolute_m"],
            0.06382515890169439,
            places=14,
        )

        formal = json.loads((RESULT_DIR / "formal_validation.json").read_text(encoding="utf-8"))
        self.assertTrue(formal["integrity_pass"])
        self.assertEqual(formal["error_count"], 0)
        legacy = ROOT / "results/legacy/paper2_mpc_pre_unified_20260720.csv"
        current = ROOT / "results/paper2_mpc_current_12_cases.csv"
        self.assertTrue(legacy.is_file())
        self.assertNotEqual(sha256(legacy), sha256(current))

    def test_public_release_has_no_machine_local_paths(self) -> None:
        files = [
            CONFIG,
            RESULT_DIR / "accepted_cases.csv",
            RESULT_DIR / "accepted_summary.json",
            RESULT_DIR / "formal_validation.json",
            ROOT / "results/paper2_mpc_current_12_cases.csv",
        ]
        forbidden = ("C:\\Users\\", "D:\\L&S\\", "D:\\L-Environment\\")
        for path in files:
            text = path.read_text(encoding="utf-8-sig")
            self.assertFalse(any(token in text for token in forbidden), path)

    def test_source_aligned_r1_records_and_boundary(self) -> None:
        with (SOURCE_ALIGNED_DIR / "rerun_records.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = {(row["case_id"], row["method"]): row for row in csv.DictReader(handle)}
        discrete = rows[("raised_L1145_r1", "Discrete active PPO")]
        continuous = rows[("raised_L1145_r1", "Continuous-torque PPO")]
        self.assertEqual(float(discrete["cmt"]), 0.267040040566968)
        self.assertEqual(float(continuous["cmt"]), 0.2780932427752225)
        self.assertEqual(discrete["repository_entrypoint_reproduces_record"], "true")
        self.assertEqual(continuous["repository_entrypoint_reproduces_record"], "false")
        self.assertEqual(continuous["recovery_setting"], "1.00/0.89")

        manifest = json.loads((SOURCE_ALIGNED_DIR / "rerun_manifest.json").read_text())
        self.assertEqual(manifest["initial_index"], [19, 11, 15, 1])
        self.assertNotIn("C:\\Users\\", json.dumps(manifest))
        self.assertNotIn("D:\\L&S\\", json.dumps(manifest))
        for record in manifest["records"]:
            snapshot = ROOT / record["source_snapshot"]
            self.assertTrue(snapshot.is_file())
            self.assertEqual(sha256(snapshot).upper(), record["source_sha256"])
        continuous_snapshot = (
            SOURCE_ALIGNED_DIR / "source_snapshot" / "Multi_continuous.py"
        ).read_text(encoding="utf-8-sig")
        self.assertIn("dtheta1_new -= 1", continuous_snapshot)
        self.assertIn("dtheta1_new -= 0.89", continuous_snapshot)
        for line in (SOURCE_ALIGNED_DIR / "CHECKSUMS.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            self.assertEqual(sha256(SOURCE_ALIGNED_DIR / relative), expected)

        with (ROOT / "results/paper2_current/paper2_combined_cases.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            combined_rows = list(csv.DictReader(handle))
            current = {(row["case_id"], row["method"]): row for row in combined_rows}
        expected_cases = {
            f"{terrain}_L{length}_r{replicate}"
            for terrain in ("flat", "raised")
            for length in ("1145", "1280")
            for replicate in (1, 2, 3)
        }
        self.assertEqual({row["case_id"] for row in combined_rows}, expected_cases)
        for case_id in expected_cases:
            self.assertEqual(sum(row["case_id"] == case_id for row in combined_rows), 6)
        self.assertEqual(float(current[("raised_L1145_r1", "Discrete active PPO")]["cmt"]), 0.267040040566968)
        self.assertEqual(float(current[("raised_L1145_r1", "Continuous-torque PPO")]["cmt"]), 0.2780932427752225)

        with (ROOT / "results/paper2_current/table_v_current.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            table_v = {row["method"]: row for row in csv.DictReader(handle)}
        self.assertNotIn("mean_landing_error_m", next(iter(table_v.values())))
        self.assertEqual(
            float(table_v["Continuous-torque MPC"]["mean_absolute_landing_error_m"]),
            0.063825158902,
        )

    def test_release_validator(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src/paper2/mpc/validate_release.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Unified MPC release validation: PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
