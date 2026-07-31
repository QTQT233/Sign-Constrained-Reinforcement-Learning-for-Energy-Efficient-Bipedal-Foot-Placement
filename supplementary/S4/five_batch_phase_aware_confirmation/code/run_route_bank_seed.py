"""Run and validate one frozen Active/positive/negative route-bank seed.

The public evaluator's release validator expects its full six-controller
release layout.  This wrapper deliberately runs only the three source policies
needed by the frozen phase-aware and two-expert routes, replaces only that
release-layout callback, and writes a route-bank-specific manifest after
strictly validating the resulting matched records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[4]
EVALUATOR_PATH = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_three_expert_selector_evaluation.py"
)
EXPECTED_EVALUATOR_SHA256 = (
    "6f4d5717ade86c4906d7ea472c3a46c146d8041d1febd5028551ee4e57c0396c"
)
EXPECTED_CONTROLLERS = {
    "active",
    "passive_negative_only",
    "passive_positive_only",
}
MATCH_FIELDS = (
    "walk_direction",
    "reset_phase",
    "commanded_step_length_m",
    "commanded_step_height_m",
    "initial_q1_deg",
    "initial_q2_deg",
    "initial_q3_deg",
    "initial_q4_deg",
    "initial_dq1_rad_s",
    "initial_dq2_rad_s",
    "initial_dq3_rad_s",
    "initial_dq4_rad_s",
)
EXPECTED_MODEL_HASHES = {
    "active": "0ca5581f39f5598ceb9c3283704e532a5efe441beb499a6c361402bf92e4720f",
    "positive": "521962b8a3deea4411dc1b1c801cf9d8fcecfafa51d240f6752c07952ec1926a",
    "negative": "4752c0c4632d5958ee64c7f3004a76e9a7bd7f3204dd914fdb7003bce9a92d7c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_evaluator():
    if sha256(EVALUATOR_PATH) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("Frozen evaluator SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location(
        "frozen_five_batch_route_bank_evaluator", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_route_bank(output_dir: Path, seed: int) -> dict:
    rollouts_path = output_dir / "rollouts_active_vs_passive.csv"
    pairs_path = output_dir / "paired_trials.csv"
    audit_path = output_dir / "passive_candidate_audit.csv"
    if not all(path.exists() for path in (rollouts_path, pairs_path, audit_path)):
        raise RuntimeError("Missing one or more route-bank core CSV files")

    rollouts = read_csv(rollouts_path)
    pairs = read_csv(pairs_path)
    audits = read_csv(audit_path)
    if len(rollouts) != 2160:
        raise RuntimeError(f"Expected 2,160 rollouts, found {len(rollouts)}")
    if len(pairs) != 1440:
        raise RuntimeError(f"Expected 1,440 paired rows, found {len(pairs)}")
    if len(audits) != 1440:
        raise RuntimeError(f"Expected 1,440 candidate rows, found {len(audits)}")

    controller_counts = Counter(row["controller"] for row in rollouts)
    if set(controller_counts) != EXPECTED_CONTROLLERS:
        raise RuntimeError(f"Unexpected controllers: {sorted(controller_counts)}")
    if set(controller_counts.values()) != {720}:
        raise RuntimeError(f"Unexpected controller counts: {controller_counts}")
    if {int(row["seed"]) for row in rollouts} != {seed}:
        raise RuntimeError("Seed mismatch in route-bank rollouts")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rollouts:
        grouped[int(row["episode_index"])].append(row)
    if set(grouped) != set(range(1, 721)):
        raise RuntimeError("Episode indices are not exactly 1--720")
    for episode_index, rows in grouped.items():
        if len(rows) != 3:
            raise RuntimeError(
                f"Episode {episode_index} does not contain three source rollouts"
            )
        reference = tuple(rows[0][field] for field in MATCH_FIELDS)
        if any(tuple(row[field] for field in MATCH_FIELDS) != reference for row in rows[1:]):
            raise RuntimeError(f"Matched-case fields differ in episode {episode_index}")

    model_paths = {
        "active": REPO_ROOT
        / "models"
        / "four_link"
        / "v22_3_v9_bi_400_grid"
        / "V22_3V9Bi400Grid_c090_Policy_best.pth",
        "positive": REPO_ROOT
        / "models"
        / "four_link"
        / "v22_3_v9_uni_pos_400_grid"
        / "V22_3V9UniPos400Grid_c097_Policy_best.pth",
        "negative": REPO_ROOT
        / "models"
        / "four_link"
        / "v22_3_v9_uni_neg_400_grid"
        / "V22_3V9UniNeg400Grid_c100_Policy_best.pth",
    }
    observed_model_hashes = {name: sha256(path) for name, path in model_paths.items()}
    if observed_model_hashes != EXPECTED_MODEL_HASHES:
        raise RuntimeError("Frozen source-policy SHA-256 mismatch")

    csv_hashes = {}
    row_counts = {}
    for path in sorted(output_dir.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        csv_hashes[path.name] = sha256(path)
        row_counts[path.name] = len(read_csv(path))

    return {
        "schema": "phase-aware-three-expert-route-bank-seed/v1",
        "seed": seed,
        "n_matched_cases": 720,
        "n_source_rollouts": len(rollouts),
        "controller_counts": dict(sorted(controller_counts.items())),
        "paired_rows": len(pairs),
        "candidate_rows": len(audits),
        "matched_case_fields_verified": list(MATCH_FIELDS),
        "deterministic_policy_actions": True,
        "evaluator_path": str(EVALUATOR_PATH),
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "model_sha256": observed_model_hashes,
        "csv_rows": row_counts,
        "csv_sha256": csv_hashes,
    }


def write_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "route_bank_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    checksum_lines = [
        f"{digest}  {name}"
        for name, digest in sorted(manifest["csv_sha256"].items())
    ]
    checksum_lines.append(f"{sha256(manifest_path)}  {manifest_path.name}")
    (output_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate an already completed core route bank without rerunning it.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if not args.validate_existing:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite nonempty {output_dir}")
        evaluator = import_evaluator()
        evaluator.OUTPUT_DIR = output_dir
        evaluator.RANDOM_SEEDS = [int(args.seed)]
        evaluator.EPISODES_PER_GRID_CELL_DIRECTION = 40
        evaluator.WALK_DIRECTIONS = [1.0, -1.0]
        evaluator.COMMAND_STEP_LENGTH_OVERRIDE_M = None
        evaluator.COMMAND_STEP_HEIGHT_OVERRIDE_M = None
        evaluator.RESET_PHASE_MODE = "random"
        evaluator.PASSIVE_POLICY_MODES = ["positive", "negative"]
        evaluator.PASSIVE_POLICY_MODE = "positive"
        evaluator.INCLUDE_ACTIVE_FULL_POLICY = False
        evaluator.INCLUDE_PER_STEP_SIGN_POLICY = False
        evaluator.INCLUDE_TRUE_ACTION_MASK_POLICY = False
        evaluator.EVALUATION_LABEL = (
            "4DoF_V22_3_3x3_grid_4Nm_phase_aware_source_bank_"
            f"seed_{args.seed}"
        )
        evaluator.REPRESENTATIVE_STATE_COUNT = 0
        evaluator.SUCCESSFUL_STATE_EXPORT_COUNT = 0

        # Only the release-layout metadata callback is replaced. All simulation,
        # terminal, energy, displacement, and validity code remains unchanged.
        evaluator.write_release_metadata = lambda _output_dir: None
        evaluator.run_evaluation()

    manifest = validate_route_bank(output_dir, int(args.seed))
    write_manifest(output_dir, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
