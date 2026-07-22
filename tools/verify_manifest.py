"""Verify complete repository file coverage, byte sizes, and SHA-256 hashes."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def current_files(root: Path) -> dict[str, Path]:
    excluded = {root / "MANIFEST.csv", root / "CHECKSUMS.sha256"}
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path not in excluded
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = current_files(root)
    with (root / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        manifest_rows = {row["path"]: row for row in csv.DictReader(handle)}

    checksum_rows: dict[str, str] = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        checksum_rows[relative] = expected

    errors: list[str] = []
    actual_names = set(files)
    if set(manifest_rows) != actual_names:
        errors.append(
            f"manifest file-set mismatch: missing={sorted(actual_names - set(manifest_rows))}, "
            f"extra={sorted(set(manifest_rows) - actual_names)}"
        )
    if set(checksum_rows) != actual_names:
        errors.append(
            f"checksum file-set mismatch: missing={sorted(actual_names - set(checksum_rows))}, "
            f"extra={sorted(set(checksum_rows) - actual_names)}"
        )

    for relative, path in files.items():
        actual_hash = digest(path)
        actual_size = path.stat().st_size
        row = manifest_rows.get(relative)
        if row is not None:
            if int(row["bytes"]) != actual_size:
                errors.append(f"size mismatch: {relative}")
            if row["sha256"] != actual_hash:
                errors.append(f"manifest hash mismatch: {relative}")
        if checksum_rows.get(relative) != actual_hash:
            errors.append(f"checksum mismatch: {relative}")

    if errors:
        print("Repository manifest verification: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Repository manifest verification: PASS ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
