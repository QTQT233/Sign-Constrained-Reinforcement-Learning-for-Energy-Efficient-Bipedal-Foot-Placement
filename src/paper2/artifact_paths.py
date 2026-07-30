"""Portable artifact resolution for the Paper2 fixed-case entry points.

The complete learned-controller case bundle is distributed with the
supplementary material rather than duplicated in this repository.  Set
``PAPER2_ARTIFACT_ROOT`` to the extracted ``paper2_artifacts`` directory,
which must contain separate ``flat`` and ``raised`` subdirectories.  Keeping
the terrain directories separate is required because several artifact
filenames occur in both directories with different contents.  Frozen TVLQR
checkpoints bundled in this repository are discovered automatically.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve_artifact(entrypoint_file: str, filename: str) -> str:
    """Return a concrete artifact path or raise an actionable error."""

    case_name = Path(entrypoint_file).resolve().parent.name
    terrain = "raised" if case_name.startswith("raised_") else "flat"
    candidates: list[Path] = []

    configured_root = os.environ.get("PAPER2_ARTIFACT_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        candidates.append(root / terrain / filename)

    candidates.extend(
        (
            REPOSITORY_ROOT
            / "models"
            / "paper2"
            / "case_bundle"
            / terrain
            / filename,
            REPOSITORY_ROOT / "models" / "paper2" / "tvlqr" / filename,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Required Paper2 artifact was not found. Extract Supplementary "
        "Archive S1, set PAPER2_ARTIFACT_ROOT to its artifact directory, "
        f"and rerun. Searched:\n  - {searched}"
    )
