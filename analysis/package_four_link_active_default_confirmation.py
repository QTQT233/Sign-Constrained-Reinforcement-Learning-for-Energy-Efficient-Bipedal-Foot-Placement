"""Combine per-seed confidence-selector evaluations into the release table."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


REPLACEMENTS = {
    "three_expert_selector": "active_default_confidence_selector",
    "three_expert_positive": "confidence_nonnegative",
    "three_expert_negative": "confidence_nonpositive",
    "three_expert_active": "confidence_active_default",
}


def relabel(value: str) -> str:
    for old, new in REPLACEMENTS.items():
        value = value.replace(old, new)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) != 3:
        raise ValueError("Exactly three per-seed CSV files are required")

    records: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            current = list(reader)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing CSV header: {path}")
        if fieldnames is None:
            fieldnames = list(reader.fieldnames)
        elif list(reader.fieldnames) != fieldnames:
            raise RuntimeError(f"CSV schema mismatch: {path}")
        if len(current) != 1440:
            raise RuntimeError(f"{path} contains {len(current)} rows; expected 1,440")
        records.extend(
            {field: relabel(value) for field, value in row.items()}
            for row in current
        )

    assert fieldnames is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
