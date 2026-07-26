from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S4 = ROOT / "supplementary" / "S4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class SupplementaryS4Tests(unittest.TestCase):
    def test_compact_scope_and_portability(self) -> None:
        self.assertTrue((S4 / "README.md").is_file())
        forbidden_parts = {
            "models",
            "formal_evaluator_snapshot.py",
            "formal_rollouts_active_vs_passive.csv",
            "formal_paired_trials.csv",
        }
        for path in S4.rglob("*"):
            self.assertFalse(any(part in forbidden_parts for part in path.parts), path)
            self.assertNotIn("touchdown", path.name.lower(), path)
        for path in S4.rglob("*"):
            if path.suffix.lower() in {".md", ".py", ".json"} and path.is_file():
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("D:\\L&S", text, path)
                self.assertNotIn("local-only", text.lower(), path)
                self.assertNotIn("not synchronized to the manuscript", text.lower(), path)

    def test_canonical_four_link_inputs_are_hash_pinned(self) -> None:
        expected = {
            "src/four_link/evaluation/"
            "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py":
                "c13c0c45b092fc43aa75fc091e870a14e4d8b6730cda5b2b8659c21a2d2bb186",
            "models/four_link/paper_four_link_passive_sign_selector_v22_3_grid.pth":
                "99fdebbe1ecdccfdfd1d36902ff4dd6936712488065aa0e6cf5c8391b24979e1",
            "models/four_link/v22_3_v9_bi_400_grid/"
            "V22_3V9Bi400Grid_c090_Policy_best.pth":
                "0ca5581f39f5598ceb9c3283704e532a5efe441beb499a6c361402bf92e4720f",
            "models/four_link/v22_3_v9_uni_pos_400_grid/"
            "V22_3V9UniPos400Grid_c097_Policy_best.pth":
                "521962b8a3deea4411dc1b1c801cf9d8fcecfafa51d240f6752c07952ec1926a",
            "models/four_link/v22_3_v9_uni_neg_400_grid/"
            "V22_3V9UniNeg400Grid_c100_Policy_best.pth":
                "4752c0c4632d5958ee64c7f3004a76e9a7bd7f3204dd914fdb7003bce9a92d7c",
            "data/four_link/true_action_mask_scratch_c090_epoch1275/"
            "rollouts_active_vs_passive.csv":
                "39557858bdee443dcfbb837b15a2c337570eb8b8828721fd5e4dfa00d42f6ade",
            "data/four_link/true_action_mask_scratch_c090_epoch1275/"
            "paired_trials.csv":
                "cab0475943a31109c386910f815e1a361fc68594c3ff1b61c872a51356db1f4d",
        }
        for relative, digest in expected.items():
            self.assertEqual(sha256(ROOT / relative), digest, relative)

    def test_landing_metric_recomputes(self) -> None:
        script = S4 / "landing_metric" / "code" / "recompute_landing_metrics.py"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "validation.json"
            completed = subprocess.run(
                [sys.executable, str(script), "--output", str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["all_checks_passed"])
        self.assertEqual(result["counts"]["transitions"], 144)
        expected = {
            "continuous": 0.04095273281529577,
            "discrete": 0.04426028430392331,
            "mpc": 0.03028463968177649,
            "passive": 0.041389074711562,
        }
        for controller, value in expected.items():
            self.assertAlmostEqual(
                result["pooled_per_transition_mae_m"][controller],
                value,
                places=14,
            )

    def test_multibatch_published_statistics(self) -> None:
        result_root = S4 / "fourlink_additional_batches" / "results"
        with (result_root / "matched_records.csv").open(
            encoding="utf-8-sig"
        ) as handle:
            self.assertEqual(sum(1 for _ in handle) - 1, 10_800)
        rows = read_csv(result_root / "batch_and_pooled_summary.csv")
        primary = next(
            row
            for row in rows
            if row["scope"] == "pooled_raw_pairs"
            and row["validity_definition"] == "A"
        )
        self.assertEqual(int(float(primary["both_valid_n"])), 2_244)
        self.assertAlmostEqual(
            float(primary["mean_selector_minus_active_cmt"]),
            -1.080266,
            places=6,
        )
        self.assertAlmostEqual(
            float(primary["fraction_selector_lower_cmt"]),
            0.81818,
            places=5,
        )
        validation = json.loads(
            (result_root / "validation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validation["validity_A_B_changed_active_records"], 0)
        self.assertEqual(validation["validity_A_B_changed_selector_records"], 0)

    def test_hold_requery_published_statistics(self) -> None:
        result_root = S4 / "hold_vs_requery" / "results"
        trials = read_csv(result_root / "hold_vs_requery_trials.csv")
        self.assertEqual(len(trials), 2_160)
        inference = json.loads(
            (result_root / "paired_inference.json").read_text(encoding="utf-8")
        )
        self.assertEqual(inference["success"]["hold_rate"], 964 / 2160)
        self.assertEqual(inference["success"]["requery_rate"], 893 / 2160)
        self.assertEqual(inference["cmt_on_both_valid"]["n_both_valid"], 419)
        interval = inference["cmt_on_both_valid"][
            "bootstrap_ci95_mean_difference"
        ]
        self.assertLess(interval[0], 0.0)
        self.assertGreater(interval[1], 0.0)

    def test_scoped_manifest_matches_files(self) -> None:
        manifest = read_csv(S4 / "MANIFEST.csv")
        indexed = {row["path"]: row for row in manifest}
        actual = {
            path.relative_to(S4).as_posix(): path
            for path in S4.rglob("*")
            if path.is_file()
            and path.name not in {"MANIFEST.csv", "CHECKSUMS.sha256"}
            and "_generated" not in path.parts
            and "outputs" not in path.parts
            and "__pycache__" not in path.parts
        }
        self.assertEqual(set(indexed), set(actual))
        for relative, path in actual.items():
            self.assertEqual(int(indexed[relative]["size_bytes"]), path.stat().st_size)
            self.assertEqual(indexed[relative]["sha256"], sha256(path))

        checksums = {}
        for line in (S4 / "CHECKSUMS.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, relative = line.split("  ", 1)
            checksums[relative] = digest
        self.assertEqual(set(checksums), set(actual))
        for relative, path in actual.items():
            self.assertEqual(checksums[relative], sha256(path))


if __name__ == "__main__":
    unittest.main()
