"""Build the scoped S4 manifest and SHA-256 inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROOT_INVENTORY_FILES = {"MANIFEST.csv", "CHECKSUMS.sha256"}
TRANSIENT_DIRECTORIES = {
    "outputs",
    "__pycache__",
    "generated_figures",
    "reproduced_results",
    "reproduced_manuscript_metrics",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_rows() -> list[dict[str, str | int]]:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not TRANSIENT_DIRECTORIES.intersection(path.parts)
        and not (
            path.parent == ROOT
            and path.name in ROOT_INVENTORY_FILES
        )
    )
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]


def render_checksums(rows: list[dict[str, str | int]]) -> str:
    return "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)


def write_inventory(rows: list[dict[str, str | int]]) -> None:
    with (ROOT / "MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "CHECKSUMS.sha256").write_text(
        render_checksums(rows),
        encoding="utf-8",
    )


def check_inventory(rows: list[dict[str, str | int]]) -> None:
    manifest_path = ROOT / "MANIFEST.csv"
    checksum_path = ROOT / "CHECKSUMS.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise RuntimeError("MANIFEST.csv or CHECKSUMS.sha256 is missing")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        existing_rows = list(csv.DictReader(handle))
    normalized_rows = [
        {
            "path": str(row["path"]),
            "size_bytes": str(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in rows
    ]
    if existing_rows != normalized_rows:
        raise RuntimeError("MANIFEST.csv does not match the current S4 payload")
    if checksum_path.read_text(encoding="utf-8") != render_checksums(rows):
        raise RuntimeError("CHECKSUMS.sha256 does not match the current S4 payload")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the current inventories without modifying any file",
    )
    args = parser.parse_args()
    rows = collect_rows()
    if args.check:
        check_inventory(rows)
        print(f"S4 inventory check: PASS ({len(rows)} payload files)")
        return 0
    write_inventory(rows)
    print(f"Indexed {len(rows)} S4 files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
