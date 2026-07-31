"""Write a SHA-256/size manifest for every non-Git file in the repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


TRANSIENT_DIRECTORIES = {
    "__pycache__",
    "generated_figures",
    "reproduced_results",
    "reproduced_manuscript_metrics",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    excluded = {root / "MANIFEST.csv", root / "CHECKSUMS.sha256"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and not TRANSIENT_DIRECTORIES.intersection(path.parts)
        and path.suffix != ".pyc"
        and path not in excluded
    ]
    rows = [(path.relative_to(root).as_posix(), path.stat().st_size, digest(path)) for path in sorted(files)]
    with (root / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(rows)
    with (root / "CHECKSUMS.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for path, _size, sha in rows:
            handle.write(f"{sha}  {path}\n")


if __name__ == "__main__":
    main()
