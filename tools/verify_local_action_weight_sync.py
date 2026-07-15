"""Verify the 16-script local action-weight tree after mechanical sync."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path


WEIGHTS = ("0.02", "0.04", "0.06")
CONDITIONS = (
    "60,30(0.33m)",
    "60,30(0.33m),0.01m",
    "70,20(0.225m)",
    "70,20,(0.225m),0.01m",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    public_root = repo_root / "src" / "two_link" / "action_weight"
    local_root = args.local_root.resolve()

    expected_pairs: list[tuple[Path, Path]] = [
        (public_root / "cmt_metrics.py", local_root / "cmt_metrics.py")
    ]
    for weight in WEIGHTS:
        for condition in CONDITIONS:
            relative = (
                Path(f"action_weight={weight}")
                / condition
                / "Whole_energy_comparison_low_dim.py"
            )
            expected_pairs.append((public_root / relative, local_root / relative))

    optimized_relative = (
        Path("action_weight=0.02")
        / "60,30(0.33m)"
        / "Whole_energy_comparison_low_dim_optimized.py"
    )
    expected_pairs.append(
        (public_root / optimized_relative, local_root / optimized_relative)
    )
    archival_copies = []
    for weight in WEIGHTS:
        local_flat = (
            local_root
            / f"action_weight={weight}"
            / "60,30(0.33m)"
        )
        copies = sorted(local_flat.glob("Whole_energy_comparison_low_dim - *.py"))
        if len(copies) != 1:
            raise RuntimeError(
                f"expected one archival copy for action_weight={weight}; "
                f"found {len(copies)}"
            )
        archival_copies.extend(copies)

    for source, destination in expected_pairs:
        if not destination.is_file():
            raise FileNotFoundError(destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"sync hash mismatch: {destination}")

    canonical = list(
        local_root.glob("action_weight=*/**/Whole_energy_comparison_low_dim.py")
    )
    optimized = list(
        local_root.glob("action_weight=*/**/Whole_energy_comparison_low_dim_optimized.py")
    )
    if (len(canonical), len(optimized), len(archival_copies)) != (12, 1, 3):
        raise RuntimeError(
            "expected 12 canonical, 1 optimized, and 3 archival copies; found "
            f"{len(canonical)}, {len(optimized)}, and {len(archival_copies)}"
        )
    scripts = sorted(canonical + optimized + archival_copies)
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        for required in (
            "positive_actuator_work_increment",
            "select_successful_expert_by_cmt",
            'tie_break="positive"',
        ):
            if required not in source:
                raise RuntimeError(f"{required!r} missing from {script}")
        if "action_value * torque" in source:
            raise RuntimeError(f"duplicate continuous torque scaling remains: {script}")

    spec = importlib.util.spec_from_file_location(
        "local_cmt_metrics", local_root / "cmt_metrics.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import local cmt_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fuse = module.select_successful_expert_by_cmt
    assert fuse(0, 0.31, -2, 0.0) == (-1, 0.31)
    assert fuse(-2, 0.0, 0, 0.29) == (1, 0.29)
    assert fuse(-1, 0.24, 1, 0.18) == (1, 0.18)
    assert fuse(-2, 0.0, -2, 0.0) == (-2, -2.0)
    print(f"PASS: {len(scripts)}/16 local evaluators and fusion truth table")


if __name__ == "__main__":
    main()
