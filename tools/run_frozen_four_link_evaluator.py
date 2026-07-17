"""Run the reconstructed legacy V22_3 evaluator without editing its source.

The historical script hard-codes an author-workstation path. This launcher
verifies the reconstruction hash, substitutes only MODEL_ROOT and OUTPUT_DIR
in a temporary copy, runs that copy, and deletes it afterward. The reconstruction
restores the legacy touchdown posture bounds and output label documented in the
archived 5 July trial files; it is not claimed to be the missing original bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "f9597d0110e9d317b5c8b428c6850b3be3d2868bd61fe73d12c1c4188cc829c4"
OLD_ROOT = 'MODEL_ROOT = Path(r"D:\\L&S\\Mas\\Project\\Paper1\\Knee\\Four_link")'
OLD_OUTPUT = 'OUTPUT_DIR = MODEL_ROOT / "paper_four_link_reachability_cmt_v22_3_grid_4Nm_extended_legacy_reconstructed"'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=repo / "src/four_link/evaluation/paper_four_link_reachability_cmt_v22_3_grid_extended_legacy_reconstructed.py")
    parser.add_argument("--model-root", type=Path, default=repo / "models/four_link")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    actual_sha = digest(args.source)
    if actual_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"Frozen source hash mismatch: {actual_sha}")

    expected_models = [
        "v22_3_v9_bi_400_grid/V22_3V9Bi400Grid_c090_Policy_best.pth",
        "v22_3_v9_active_full_400_grid/V22_3V9ActiveFull400Grid_c090_Policy_best.pth",
        "v22_3_v9_per_step_sign_400_grid/V22_3V9PerStepSign400Grid_c091_Policy_258.pth",
        "v22_3_v9_uni_pos_400_grid/V22_3V9UniPos400Grid_c097_Policy_best.pth",
        "v22_3_v9_uni_neg_400_grid/V22_3V9UniNeg400Grid_c100_Policy_best.pth",
        "paper_four_link_passive_sign_selector_v22_3_grid.pth",
    ]
    missing = [name for name in expected_models if not (args.model_root / name).is_file()]
    if missing:
        raise SystemExit("Missing model artifacts:\n" + "\n".join(missing))

    text = args.source.read_text(encoding="utf-8")
    if text.count(OLD_ROOT) != 1 or text.count(OLD_OUTPUT) != 1:
        raise SystemExit("Expected path assignments were not found exactly once")
    args.output.mkdir(parents=True, exist_ok=True)
    patched = text.replace(OLD_ROOT, f'MODEL_ROOT = Path(r"{args.model_root.resolve()}")')
    patched = patched.replace(OLD_OUTPUT, f'OUTPUT_DIR = Path(r"{args.output.resolve()}")')

    print(f"source_sha256={actual_sha}")
    print("protocol=legacy_v22_3_reconstruction")
    print(f"model_root={args.model_root.resolve()}")
    print(f"output={args.output.resolve()}")
    print("expected_shared_cases=2160")
    if args.check_only:
        return

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False,
                                         dir=args.output) as handle:
            handle.write(patched)
            temporary_path = Path(handle.name)
        subprocess.run([sys.executable, str(temporary_path)], check=True, cwd=repo)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    main()
