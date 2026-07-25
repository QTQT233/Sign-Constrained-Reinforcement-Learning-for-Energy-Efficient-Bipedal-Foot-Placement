"""Build the scoped S4 manifest and SHA-256 inventory."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.csv", "CHECKSUMS.sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and "_generated" not in path.parts
        and "outputs" not in path.parts
        and "__pycache__" not in path.parts
    )
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    with (ROOT / "MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "CHECKSUMS.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Indexed {len(rows)} S4 files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
