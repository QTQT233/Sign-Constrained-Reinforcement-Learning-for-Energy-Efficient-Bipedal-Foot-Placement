"""Write a SHA-256/size manifest for every non-Git file in the repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def is_generated_or_cache(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        ".git" in relative.parts
        or "__pycache__" in relative.parts
        or "_generated" in relative.parts
        or relative.parts[:4]
        == ("supplementary", "S4", "hold_vs_requery", "outputs")
        or path.suffix == ".pyc"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    excluded = {root / "MANIFEST.csv", root / "CHECKSUMS.sha256"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not is_generated_or_cache(path, root)
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
