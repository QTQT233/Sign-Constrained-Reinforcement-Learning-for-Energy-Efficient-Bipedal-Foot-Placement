"""Portable artifact resolution for the source-aligned Paper2 entry points.

The complete learned-controller case bundle is distributed with the
supplementary material rather than duplicated in this repository.  Set
``PAPER2_ARTIFACT_ROOT`` to the extracted bundle root.  The root may either
contain ``flat`` and ``raised`` subdirectories or contain the files directly.
Frozen TVLQR checkpoints bundled in this repository are discovered
automatically.
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
        candidates.extend((root / terrain / filename, root / filename))

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
