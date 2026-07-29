"""Run one independently seeded four-link three-expert evaluation batch.

This thin wrapper preserves the manuscript evaluator unchanged while exposing
the seed triplet and output directory as command-line arguments.  It is useful
for launching the five additional batches in parallel.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    REPO_ROOT
    / "src"
    / "four_link"
    / "evaluation"
    / "paper_four_link_three_expert_selector_evaluation.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("four_link_three_expert_eval", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.RANDOM_SEEDS = list(args.seeds)
    module.OUTPUT_DIR = args.output_dir.resolve()
    module.EVALUATION_LABEL = args.label
    module.validate_input_artifacts()
    module.run_evaluation()


if __name__ == "__main__":
    main()
