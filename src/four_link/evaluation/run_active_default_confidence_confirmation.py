"""Evaluate the frozen Active-default confidence selector.

This release wrapper preserves the dynamics, endpoint-target termination,
mechanical-work metric, initial-state sampler, and low-level checkpoints in
``paper_four_link_three_expert_selector_evaluation.py``.  It changes only the
transition-start route:

* Active PPO is the default.
* The non-negative expert is released when its frozen-gate probability is at
  least 0.86.
* The non-positive expert is released when its frozen-gate probability is at
  least 0.54.
* If both thresholds are crossed, the higher-probability sign expert is used.

The selected expert is held until transition termination.  The thresholds were
selected on development data; the manuscript-facing confirmation uses the
previously unused seeds 20261301--20261303.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BASE_EVALUATOR = HERE / "paper_four_link_three_expert_selector_evaluation.py"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "_scratch" / "active_default_confidence_confirmation"
)
POSITIVE_THRESHOLD = 0.86
NEGATIVE_THRESHOLD = 0.54
PUBLIC_CONTROLLER_LABEL = "active_default_confidence_selector"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "four_link_three_expert_base_evaluator", BASE_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def route_from_probabilities(
    probabilities,
    positive_threshold: float = POSITIVE_THRESHOLD,
    negative_threshold: float = NEGATIVE_THRESHOLD,
) -> tuple[str, float]:
    """Return the frozen transition route and its gate probability."""
    positive_ok = float(probabilities[0]) >= positive_threshold
    negative_ok = float(probabilities[1]) >= negative_threshold
    if positive_ok and negative_ok:
        label = 0 if probabilities[0] >= probabilities[1] else 1
    elif positive_ok:
        label = 0
    elif negative_ok:
        label = 1
    else:
        label = 2
    return ("positive", "negative", "active")[label], float(probabilities[label])


def relabel_csv_outputs(output_dir: Path) -> None:
    """Replace the base evaluator's generic selector labels in every CSV."""
    replacements = {
        "three_expert_selector": PUBLIC_CONTROLLER_LABEL,
        "three_expert_positive": "confidence_nonnegative",
        "three_expert_negative": "confidence_nonpositive",
        "three_expert_active": "confidence_active_default",
    }
    for path in sorted(output_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            records = list(csv.reader(stream))
        if not records:
            path.unlink()
            continue
        changed = [
            [
                next(
                    (
                        value.replace(old, new)
                        for old, new in replacements.items()
                        if old in value
                    ),
                    value,
                )
                for value in row
            ]
            for row in records
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(changed)


def write_release_metadata(
    output_dir: Path,
    evaluator,
    seeds: list[int],
    episodes_per_cell_direction: int,
) -> None:
    csv_files = sorted(output_dir.glob("*.csv"), key=lambda path: path.name)
    manifest = {
        "schema": "four-link-active-default-confidence-evaluation/v1",
        "controller": PUBLIC_CONTROLLER_LABEL,
        "evidence_role": "held-out Active-reference confirmation",
        "threshold_selection": "selected on development data",
        "routing": {
            "default": "active",
            "nonnegative_gate_probability_minimum": POSITIVE_THRESHOLD,
            "nonpositive_gate_probability_minimum": NEGATIVE_THRESHOLD,
            "query_schedule": "once at transition start; selected expert held",
        },
        "evaluation": {
            "seeds": seeds,
            "episodes_per_grid_cell_direction": episodes_per_cell_direction,
            "walk_directions": evaluator.WALK_DIRECTIONS,
            "step_lengths_m": evaluator.command_grid_values()[0],
            "step_heights_m": evaluator.command_grid_values()[1],
            "deterministic_policy_actions": evaluator.DETERMINISTIC_POLICY,
            "hip_torque_limit_nm": evaluator.HIP_TORQUE_NM,
            "knee_torque_limit_nm": evaluator.KNEE_TORQUE_NM,
        },
        "source": {
            "wrapper": (
                "src/four_link/evaluation/"
                "run_active_default_confidence_confirmation.py"
            ),
            "wrapper_sha256": sha256_file(Path(__file__)),
            "base_evaluator": (
                "src/four_link/evaluation/"
                "paper_four_link_three_expert_selector_evaluation.py"
            ),
            "base_evaluator_sha256": sha256_file(BASE_EVALUATOR),
        },
        "frozen_policy_sha256": {
            evaluator.repository_path(path): expected
            for path, expected in evaluator.EXPECTED_POLICY_SHA256.items()
        },
        "result_csv_sha256": {
            path.name: sha256_file(path) for path in csv_files
        },
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_paths = csv_files + [manifest_path]
    (output_dir / "CHECKSUMS.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        default="20261301,20261302,20261303",
        help="Comma-separated evaluation seeds.",
    )
    parser.add_argument("--episodes-per-cell-direction", type=int, default=40)
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one evaluation seed is required")

    evaluator = load_evaluator()
    evaluator.OUTPUT_DIR = args.output_dir.resolve()
    evaluator.RANDOM_SEEDS = seeds
    evaluator.EPISODES_PER_GRID_CELL_DIRECTION = args.episodes_per_cell_direction
    evaluator.INCLUDE_ACTIVE_FULL_POLICY = False
    evaluator.INCLUDE_PER_STEP_SIGN_POLICY = False
    evaluator.INCLUDE_TRUE_ACTION_MASK_POLICY = False
    evaluator.PASSIVE_POLICY_MODES = ["three_expert_selector"]
    evaluator.PASSIVE_POLICY_MODE = "three_expert_selector"
    evaluator.EVALUATION_LABEL = (
        "4DoF_V22_3_3x3_grid_4Nm_active_default_confidence_p086_n054"
    )

    def choose_active_default(
        selector,
        initial_state,
        walk_direction,
        commanded_step_length,
        commanded_step_height,
    ):
        state_net = evaluator.normalize_state(
            initial_state,
            0.0,
            walk_direction,
            commanded_step_length,
            commanded_step_height,
        )
        with torch.no_grad():
            logits = selector(
                torch.as_tensor(
                    state_net,
                    dtype=torch.float32,
                    device=evaluator.DEVICE,
                ).reshape(1, evaluator.state_dim)
            )[0]
            probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return route_from_probabilities(probabilities)

    evaluator.choose_three_expert = choose_active_default
    # The base evaluator's completed-output validator targets its six-controller
    # campaign.  This focused two-controller run writes the dedicated manifest
    # below after removing disabled-controller empty files.
    evaluator.write_release_metadata = lambda _output_dir: None
    evaluator.run_evaluation()
    relabel_csv_outputs(evaluator.OUTPUT_DIR)
    write_release_metadata(
        evaluator.OUTPUT_DIR,
        evaluator,
        seeds,
        args.episodes_per_cell_direction,
    )
    print(f"Published confirmation outputs: {evaluator.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
