"""Integrity tests for the frozen five-batch phase-aware confirmation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import re
import sys
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


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

    def test_three_expert_gate_hash_amendment(self) -> None:
        amendment_path = BUNDLE / "GATE_HASH_AMENDMENT_02.json"
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        expected_gate_hash = (
            "d7f63ce349fda4f670a343f8b579f1512420616471d3ec4650e3c6c07105b954"
        )
        gate_path = (
            ROOT
            / "models"
            / "four_link"
            / "three_expert_selector_v22_3"
            / "V22_3_three_expert_selector.pth"
        )
        self.assertEqual(sha256(gate_path), expected_gate_hash)
        self.assertEqual(
            amendment["omitted_artifact"]["sha256"],
            expected_gate_hash,
        )
        self.assertEqual(
            self.report["artifact_hashes"]["three_expert_gate"],
            expected_gate_hash,
        )
        self.assertEqual(
            amendment["source_protocol_sha256"],
            sha256(BUNDLE / "protocol.json"),
        )
        self.assertEqual(
            amendment["source_final_report_sha256"],
            sha256(RESULTS / "final_confirmation_report.json"),
        )
        self.assertEqual(
            amendment["analysis_source_boundary"][
                "portable_release_sha256_after_device_portability_repair"
            ],
            sha256(BUNDLE / "code" / "analyze_confirmation.py"),
        )

    def test_no_manuscript_renderer_or_generated_figure(self) -> None:
        forbidden_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg", ".pdf"}
        for path in BUNDLE.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(BUNDLE).as_posix().lower()
            if "__pycache__/" in relative:
                continue
            self.assertNotIn("plot_figure", relative)
            self.assertNotIn("generated_figures/", relative)
            self.assertNotIn("__pycache__/", relative)
            self.assertNotEqual(path.suffix.lower(), ".pyc")
            self.assertNotIn(path.suffix.lower(), forbidden_extensions)

    def test_gate_tensor_uses_selector_device(self) -> None:
        analysis_source = (
            BUNDLE / "code" / "analyze_confirmation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "selector_device = next(selector.parameters()).device",
            analysis_source,
        )
        self.assertIn("device=selector_device", analysis_source)

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

    def test_public_entrypoints_resolve_repository_root(self) -> None:
        code = BUNDLE / "code"
        modules = {
            "analysis": load_module(
                code / "analyze_confirmation.py",
                "_five_batch_analysis_path_test",
            ),
            "route": load_module(
                code / "run_route_bank_seed.py",
                "_five_batch_route_path_test",
            ),
            "hard": load_module(
                code / "replay_hard_mask.py",
                "_five_batch_hard_path_test",
            ),
        }
        self.assertEqual(modules["analysis"].REPO_ROOT, ROOT)
        self.assertEqual(modules["route"].REPO_ROOT, ROOT)
        self.assertEqual(modules["hard"].REPO, ROOT)
        for module in modules.values():
            self.assertEqual(module.EVALUATOR_PATH.parents[3], ROOT)
            self.assertTrue(module.EVALUATOR_PATH.is_file())

    def test_published_readmes_have_no_local_drive_paths(self) -> None:
        local_drive = re.compile(r"(?i)\b[a-z]:[\\/]")
        for path in BUNDLE.rglob("README.md"):
            contents = path.read_text(encoding="utf-8")
            self.assertIsNone(
                local_drive.search(contents),
                f"Local absolute path in {path.relative_to(ROOT)}",
            )

    def test_five_batch_dependency_inventory(self) -> None:
        requirements = (
            ROOT / "environment" / "requirements-five-batch.txt"
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            {line.strip() for line in requirements if line.strip()},
            {
                "numpy==1.26.4",
                "pandas==2.3.0",
                "scipy==1.13.1",
                "torch==2.7.0",
                "tqdm==4.67.1",
            },
        )
        readme = (BUNDLE / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "environment/requirements-five-batch.txt",
            readme,
        )

    def test_hash_only_provenance_mapping_is_complete(self) -> None:
        provenance = (BUNDLE / "PROVENANCE.md").read_text(encoding="utf-8")
        mappings = {
            "FROZEN_PROTOCOL.json": "protocol.json",
            "code/run_new_seed_route_bank.py": "code/run_route_bank_seed.py",
            "code/replay_hard_mask_on_confirmation_states.py":
                "code/replay_hard_mask.py",
            "code/run_validated_route_bank.py": "code/run_route_bank_seed.py",
        }
        for ledger_name, public_name in mappings.items():
            self.assertIn(ledger_name, provenance)
            self.assertIn(public_name, provenance)

        self.assertEqual(
            sha256(BUNDLE / "protocol.json"),
            "3ee4fc567e0ae922fd9379ff0eb7cd3eb2bb5dd850cc26e199230ea45bda642f",
        )


if __name__ == "__main__":
    unittest.main()
