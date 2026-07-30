from __future__ import annotations

import csv
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

    def test_action_weight_path_bound_sources_are_labeled_provenance_only(self) -> None:
        boundary = ROOT / "src/two_link/action_weight/README.md"
        text = boundary.read_text(encoding="utf-8")
        self.assertIn("provenance snapshots, not portable entry points", text)
        self.assertIn("Supplementary Data S2", text)

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
