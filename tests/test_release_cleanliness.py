from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCleanlinessTests(unittest.TestCase):
    def test_no_superseded_or_figure_rendering_artifacts(self) -> None:
        forbidden_paths = (
            "results/legacy",
            "results/figures",
            "data/four_link/legacy_manuscript",
            "configs/four_link_legacy_manuscript.json",
            "docs/VERSION_AUDIT.md",
            "ms_Qi_Long_version.pdf",
            "data/LOCAL_ARTIFACT_SOURCES.csv",
            "l1_stand_sim_draw.py",
            "l2_stand_sim_draw.py",
            "src/two_link/offline_lookup/Energy_comparison.py",
            "src/two_link/offline_routing/Energy_comparison_draw_low_dim.py",
            "src/two_link/action_map/(-1,0,1)simulation_continuous.py",
            "src/two_link/action_map/(-1,0,1)simulation.py",
            "src/two_link/action_map/(-1,0)simulation.py",
            "src/two_link/action_map/(0,1)simulation.py",
        )
        for relative in forbidden_paths:
            self.assertFalse((ROOT / relative).exists(), relative)
        forbidden_name_tokens = (
            "plot_figure",
            "near_extended",
            "original_layout",
            "three_column",
            "manual_state",
            "mechanistic_ablation",
        )
        tracked_scope = list((ROOT / "analysis").rglob("*")) + list((ROOT / "results").rglob("*"))
        for path in tracked_scope:
            if path.is_file():
                lower = path.name.lower()
                self.assertFalse(any(token in lower for token in forbidden_name_tokens), path)
        self.assertTrue(
            (ROOT / "src/two_link/offline_routing/transition_locked_router.py").is_file()
        )
        self.assertTrue(
            (ROOT / "src/two_link/offline_lookup/atc50_event_lookup.py").is_file()
        )

    def test_primary_two_link_values_are_current(self) -> None:
        with (ROOT / "results/two_link_primary_12_cases.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = {row["case_id"]: row for row in csv.DictReader(handle)}
        row = rows["raised_L1145_r1"]
        self.assertEqual(
            float(row["discrete_active_ppo_cmt"]),
            0.25098054963443095,
        )
        self.assertEqual(
            float(row["continuous_active_ppo_cmt"]),
            0.25817243332390905,
        )

    def test_public_text_has_no_retired_release_pins(self) -> None:
        targets = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
        forbidden = (
            "21406793",
            "21406792",
            "77f2f7d69df183344ab06ae55e924bd071646393",
            "straight-knee-permissive",
            "unchanged author-owned",
        )
        for path in targets:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, path)

    def test_action_weight_sources_use_portable_artifact_paths(self) -> None:
        boundary = ROOT / "src/two_link/action_weight/README.md"
        text = boundary.read_text(encoding="utf-8")
        self.assertIn("ACTION_WEIGHT_MODEL_DIR", text)
        self.assertIn("ACTION_WEIGHT_OUTPUT_DIR", text)
        self.assertIn("Supplementary Data S2", text)

        evaluators = sorted(
            (ROOT / "src/two_link/action_weight").glob(
                "action_weight=*/*/Whole_energy_comparison_low_dim.py"
            )
        )
        self.assertEqual(len(evaluators), 12)
        manifest = json.loads(
            (ROOT / "configs/action_weight_checkpoint_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["cases"]), 12)
        manifest_by_evaluator = {
            record["evaluator"]: record for record in manifest["cases"]
        }
        table_ii_manifest = json.loads(
            (
                ROOT / "results/action_weight_rerun_seed_20260716_manifest.json"
            ).read_text(encoding="utf-8")
        )
        table_ii_scripts = {
            record["path"]: record
            for record in table_ii_manifest["canonical_scripts"]
        }
        for path in evaluators:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("D:/", source, path)
            self.assertNotIn("C:/Users/", source, path)
            self.assertIsNone(
                re.search(r"(?i)(?:[a-z]:[/\\]|\\\\[^\\])", source), path
            )
            self.assertIn("ACTION_WEIGHT_MODEL_DIR", source, path)
            self.assertIn("ACTION_WEIGHT_OUTPUT_DIR", source, path)
            tree = ast.parse(source)
            checkpoint_loads = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "load"
            ]
            output_files = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "h5py"
                and node.func.attr == "File"
            ]
            self.assertEqual(len(checkpoint_loads), 4, path)
            self.assertEqual(len(output_files), 18, path)

            relative = path.relative_to(ROOT).as_posix()
            record = manifest_by_evaluator[relative]
            self.assertEqual(path.stat().st_size, record["evaluator_bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["evaluator_sha256"],
            )
            self.assertEqual(
                table_ii_scripts[relative]["sha256"],
                record["evaluator_sha256"],
            )
            manifest_names = {
                checkpoint["filename"] for checkpoint in record["checkpoints"]
            }
            source_names = {
                node.args[0].right.value
                for node in checkpoint_loads
                if isinstance(node.args[0], ast.BinOp)
                and isinstance(node.args[0].right, ast.Constant)
            }
            self.assertEqual(source_names, manifest_names, path)

        duplicate_name = "Policy_Net_Pytorch(-1,0,1)_1398_continuous.pth"
        duplicate_hashes = {
            checkpoint["sha256"]
            for record in manifest["cases"]
            for checkpoint in record["checkpoints"]
            if checkpoint["filename"] == duplicate_name
        }
        self.assertEqual(len(duplicate_hashes), 2)

        action_map = ROOT / "data/two_link/action_map"
        self.assertFalse(any("10,30" in path.name for path in action_map.iterdir()))

    def test_paper2_entrypoints_use_portable_artifact_resolution(self) -> None:
        resolver = ROOT / "src/paper2/artifact_paths.py"
        self.assertTrue(resolver.is_file())
        entrypoints = ROOT / "src/paper2/entrypoints"
        for path in entrypoints.glob("*/*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("D:/", text, path)
            self.assertNotIn("D:\\", text, path)
            if "torch.load(" in text or "h5py.File(" in text:
                self.assertIn("_resolve_artifact", text, path)


if __name__ == "__main__":
    unittest.main()
