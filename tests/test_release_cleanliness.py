from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCleanlinessTests(unittest.TestCase):
    def test_current_landing_metric_uses_horizontal_foot_residual(self) -> None:
        executable_sources = [
            *sorted((ROOT / "src/paper2/entrypoints").rglob("Qi_multi_passive_sim.py")),
            *sorted((ROOT / "src/paper2/entrypoints").rglob("Qi_multi_ac_discrete_sim.py")),
            *sorted((ROOT / "src/paper2/entrypoints").rglob("Multi_continuous.py")),
            ROOT / "src/paper2/mpc/two_link_mpc_multistep_compare.py",
            ROOT / "src/paper2/mpc/mpc_ppo_aligned_common.py",
        ]
        self.assertEqual(len(executable_sources), 38)
        for path in executable_sources:
            text = path.read_text(encoding="utf-8-sig")
            self.assertNotIn("Foot_D", text, path)
            self.assertNotIn("Foot_error", text, path)
            self.assertNotIn("(3 * 0.521)", text, path)

        current_csvs = (
            ROOT / "results/paper2_push_off_grid_12case/accepted_cases.csv",
            ROOT / "results/paper2_push_off_grid_12case/controller_summary.csv",
            ROOT / "results/paper2_mpc_current_12_cases.csv",
            ROOT / "results/paper2_mpc_unified_12case/accepted_cases.csv",
            ROOT / "results/paper2_current/paper2_combined_cases.csv",
            ROOT / "results/paper2_current/table_v_current.csv",
        )
        for path in current_csvs:
            header = path.read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertNotIn("foot_error", header, path)
            self.assertNotIn("landing_error", header, path)

        provenance_readmes = (
            ROOT / "data/paper2/readonly_rerun/README.md",
            ROOT / "results/paper2_source_aligned_r1_rerun_20260721/README.md",
        )
        for path in provenance_readmes:
            text = path.read_text(encoding="utf-8")
            self.assertIn("deprecated historical", text, path)
            self.assertIn("supplementary/S4/landing_metric/", text, path)

        derived_historical_files = (
            ROOT / "data/paper2/readonly_rerun/paper2_rerun_results.csv",
            ROOT / "data/paper2/readonly_rerun/paper2_overall_summary.csv",
            ROOT / "results/paper2_source_aligned_r1_rerun_20260721/rerun_records.csv",
            ROOT / "results/paper2_source_aligned_r1_rerun_20260721/rerun_manifest.json",
        )
        for path in derived_historical_files:
            text = path.read_text(encoding="utf-8-sig")
            self.assertNotIn("foot_error_m", text, path)
            self.assertNotIn("landing_error_m", text, path)

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
        self.assertEqual(float(row["discrete_active_ppo_cmt"]), 0.25098054963443095)
        self.assertEqual(float(row["continuous_active_ppo_cmt"]), 0.25817243332390905)

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

    def test_archive_scope_and_four_link_torque_are_consistent(self) -> None:
        availability = (ROOT / "docs/DATA_AVAILABILITY.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "S2: submission-supplied strict-\\(D>0.01\\)", availability
        )
        self.assertIn("not included in Zenodo", availability)
        self.assertIn("S4:", availability)
        self.assertIn("not included in", availability)

        entrypoint_boundary = (
            ROOT / "src/paper2/entrypoints/README.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Supplementary Archive S1 is limited to", entrypoint_boundary
        )

        config = json.loads(
            (
                ROOT
                / "configs/four_link_true_action_mask_scratch_c090_epoch1275.json"
            ).read_text(encoding="utf-8")
        )
        matched = config["training_alignment"]["matched_to_fixed_sign_experts"]
        self.assertEqual(matched["hip_torque_limit_nm"], 4.0)
        self.assertEqual(matched["knee_torque_limit_nm"], 0.2)

        evaluator = (
            ROOT
            / "src/four_link/evaluation/"
            "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
        ).read_text(encoding="utf-8")
        self.assertIn("HIP_TORQUE_NM = 4.0", evaluator)
        self.assertIn("KNEE_TORQUE_NM = 0.20", evaluator)

    def test_repository_manifest_verifier(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/verify_manifest.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
