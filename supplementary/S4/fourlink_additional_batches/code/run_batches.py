"""Regenerate one frozen-policy four-link robustness batch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parents[2]
CONFIG_PATH = COMPONENT_ROOT / "config" / "batches.json"
EVALUATOR_PATH = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_reachability_cmt_v22_3_grid_true_action_mask.py"
)
MODEL_PATHS = {
    "active": (
        REPO_ROOT
        / "models"
        / "four_link"
        / "v22_3_v9_bi_400_grid"
        / "V22_3V9Bi400Grid_c090_Policy_best.pth"
    ),
    "positive": (
        REPO_ROOT
        / "models"
        / "four_link"
        / "v22_3_v9_uni_pos_400_grid"
        / "V22_3V9UniPos400Grid_c097_Policy_best.pth"
    ),
    "negative": (
        REPO_ROOT
        / "models"
        / "four_link"
        / "v22_3_v9_uni_neg_400_grid"
        / "V22_3V9UniNeg400Grid_c100_Policy_best.pth"
    ),
    "selector": (
        REPO_ROOT
        / "models"
        / "four_link"
        / "paper_four_link_passive_sign_selector_v22_3_grid.pth"
    ),
}
EXPECTED_SHA256 = {
    EVALUATOR_PATH:
        "c13c0c45b092fc43aa75fc091e870a14e4d8b6730cda5b2b8659c21a2d2bb186",
    MODEL_PATHS["active"]:
        "0ca5581f39f5598ceb9c3283704e532a5efe441beb499a6c361402bf92e4720f",
    MODEL_PATHS["positive"]:
        "521962b8a3deea4411dc1b1c801cf9d8fcecfafa51d240f6752c07952ec1926a",
    MODEL_PATHS["negative"]:
        "4752c0c4632d5958ee64c7f3004a76e9a7bd7f3204dd914fdb7003bce9a92d7c",
    MODEL_PATHS["selector"]:
        "99fdebbe1ecdccfdfd1d36902ff4dd6936712488065aa0e6cf5c8391b24979e1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {path.relative_to(REPO_ROOT)}: "
                f"expected {expected}, observed {observed}"
            )


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "s4_frozen_fourlink_evaluator", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(module, output_dir: Path, seeds: list[int], episodes: int) -> None:
    module.MODEL_ROOT = REPO_ROOT / "models"
    module.ACTIVE_POLICY_PATH = MODEL_PATHS["active"]
    module.PASSIVE_POS_POLICY_PATH = MODEL_PATHS["positive"]
    module.PASSIVE_NEG_POLICY_PATH = MODEL_PATHS["negative"]
    module.PASSIVE_SIGN_SELECTOR_PATH = MODEL_PATHS["selector"]
    module.EXPECTED_POLICY_SHA256 = {
        MODEL_PATHS["active"]: EXPECTED_SHA256[MODEL_PATHS["active"]],
        MODEL_PATHS["positive"]: EXPECTED_SHA256[MODEL_PATHS["positive"]],
        MODEL_PATHS["negative"]: EXPECTED_SHA256[MODEL_PATHS["negative"]],
        MODEL_PATHS["selector"]: EXPECTED_SHA256[MODEL_PATHS["selector"]],
    }
    module.INCLUDE_ACTIVE_FULL_POLICY = False
    module.INCLUDE_PER_STEP_SIGN_POLICY = False
    module.INCLUDE_TRUE_ACTION_MASK_POLICY = False
    module.PASSIVE_POLICY_MODES = ["sign_selector"]
    module.PASSIVE_POLICY_MODE = "sign_selector"
    module.RANDOM_SEEDS = seeds
    module.EPISODES_PER_GRID_CELL_DIRECTION = episodes
    module.OUTPUT_DIR = output_dir
    module.EVALUATION_LABEL = f"S4_frozen_active_vs_selector_{output_dir.name}"
    module.REQUIRE_NONZERO_TORQUE_FOR_CMT = True
    module.MIN_NONZERO_TORQUE_STEPS = 1
    module.REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT = True
    module.MIN_COM_DISPLACEMENT_M = 0.001
    module.write_release_metadata = lambda _output_dir: None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--probe-episodes", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.batch == "probe":
        batch_name = "runtime_probe"
        seeds = [20269991]
        episodes = args.probe_episodes
    elif args.batch == "formal_reference":
        batch_name = args.batch
        seeds = config["formal_reference_batch"]["formal_reference"]
        episodes = config["episodes_per_grid_cell_direction"]
    else:
        try:
            seeds = config["fresh_batches"][args.batch]
        except KeyError as exc:
            raise SystemExit(f"Unknown batch: {args.batch}") from exc
        batch_name = args.batch
        episodes = config["episodes_per_grid_cell_direction"]

    output_dir = COMPONENT_ROOT / "_generated" / batch_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output_dir}")
    verify_inputs()
    module = load_evaluator()
    configure(module, output_dir, [int(seed) for seed in seeds], int(episodes))
    started = time.time()
    status = "started"
    error = None
    try:
        module.run_evaluation()
        status = "completed"
    except Exception as exc:
        status = "failed"
        error = "".join(traceback.format_exception(exc))
        raise
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "batch": batch_name,
            "seeds": seeds,
            "episodes_per_grid_cell_direction": episodes,
            "matched_states_expected": len(seeds) * 18 * int(episodes),
            "status": status,
            "elapsed_seconds": time.time() - started,
            "error": error,
            "artifacts_sha256": {
                str(path.relative_to(REPO_ROOT)).replace("\\", "/"): digest
                for path, digest in EXPECTED_SHA256.items()
            },
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
