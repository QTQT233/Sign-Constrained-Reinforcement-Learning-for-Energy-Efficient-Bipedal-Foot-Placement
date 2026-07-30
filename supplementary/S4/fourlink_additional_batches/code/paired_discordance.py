"""Compute exact paired discordance tests from the fresh matched records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]


def paired_row(
    records: pd.DataFrame,
    endpoint: str,
    active_column: str,
    selector_column: str,
) -> dict[str, float | int | str]:
    active = records[active_column].astype(str).str.lower().eq("true")
    selector = records[selector_column].astype(str).str.lower().eq("true")
    active_only = int((active & ~selector).sum())
    selector_only = int((~active & selector).sum())
    discordant = active_only + selector_only
    test = binomtest(
        selector_only,
        n=discordant,
        p=0.5,
        alternative="two-sided",
    )
    return {
        "endpoint": endpoint,
        "n_pairs": int(len(records)),
        "both_positive": int((active & selector).sum()),
        "active_only": active_only,
        "selector_only": selector_only,
        "neither_positive": int((~active & ~selector).sum()),
        "discordant_n": discordant,
        "selector_only_over_active_only": (
            selector_only / active_only if active_only else float("inf")
        ),
        "mcnemar_exact_two_sided_p": float(test.pvalue),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched-records",
        type=Path,
        default=ROOT / "results" / "matched_records.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_generated",
    )
    args = parser.parse_args()
    records = pd.read_csv(args.matched_records)
    rows = [
        paired_row(
            records,
            "success",
            "active_success",
            "passive_sign_selector_success",
        ),
        paired_row(
            records,
            "valid_Cmt_A",
            "active_valid_A",
            "passive_sign_selector_valid_A",
        ),
        paired_row(
            records,
            "valid_Cmt_B",
            "active_valid_B",
            "passive_sign_selector_valid_B",
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "paired_discordance.csv", index=False
    )
    (args.output_dir / "paired_discordance.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
