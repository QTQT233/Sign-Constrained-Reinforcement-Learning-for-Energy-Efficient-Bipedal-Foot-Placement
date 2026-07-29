#!/usr/bin/env python3
"""Reproduce the pooled four-link two-/three-expert incremental comparison.

The script joins the frozen two-class selector records and the subsequently
evaluated three-class selector records by batch, seed, and episode index.  It
also identifies strict active-fallback rescues: cases in which the learned gate
selected the unrestricted active expert, that expert succeeded, and the
offline envelope confirms that both one-sided experts failed.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean


KEY_FIELDS = ("batch", "seed", "episode_index")
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 10_000
FLOAT_MATCH_TOLERANCE = 1.0e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Unsupported Boolean value: {value!r}")


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Return the two-sided exact McNemar/binomial p-value."""

    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    tail = sum(math.comb(total, k) for k in range(lower + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_binary_normal_ci(
    rows: list[dict[str, object]], field: str
) -> tuple[float, float]:
    differences = [
        float(bool(row[f"three_{field}"])) - float(bool(row[f"two_{field}"]))
        for row in rows
    ]
    mean = fmean(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    half_width = 1.96 * math.sqrt(variance / len(differences))
    return mean - half_width, mean + half_width


def stratified_bootstrap_mean_ci(
    differences_by_batch: dict[str, list[float]],
) -> tuple[float, float]:
    groups = [values for _, values in sorted(differences_by_batch.items()) if values]
    total_n = sum(len(values) for values in groups)
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for values in groups:
            total += sum(rng.choices(values, k=len(values)))
        replicates.append(total / total_n)
    return quantile(replicates, 0.025), quantile(replicates, 0.975)


def load_two_expert(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    records: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in records:
                raise ValueError(f"Duplicate two-expert key: {key}")
            records[key] = row
    return records


def load_three_expert(root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    records: dict[tuple[str, str, str], dict[str, str]] = {}
    batch_dirs = sorted(path for path in root.glob("batch*") if path.is_dir())
    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found under {root}")
    for batch_dir in batch_dirs:
        path = batch_dir / "three_expert_vs_oracle_trials.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (batch_dir.name, row["seed"], row["episode_index"])
                if key in records:
                    raise ValueError(f"Duplicate three-expert key: {key}")
                row["batch"] = batch_dir.name
                records[key] = row
    return records


def floats_match(left: str, right: str, tolerance: float = FLOAT_MATCH_TOLERANCE) -> bool:
    left_value = finite_float(left)
    right_value = finite_float(right)
    if left_value is None or right_value is None:
        return left_value is None and right_value is None
    return abs(left_value - right_value) <= tolerance


def validate_full_pairing(
    two_records: dict[tuple[str, str, str], dict[str, str]],
    three_records: dict[tuple[str, str, str], dict[str, str]],
    two_path: Path,
    three_root: Path,
) -> dict[str, object]:
    if set(two_records) != set(three_records):
        raise ValueError("Full-record key sets differ before provenance validation")

    mismatch_counts = {
        "walk_direction": 0,
        "reset_phase": 0,
        "commanded_step_length_m": 0,
        "commanded_step_height_m": 0,
        "initial_state": 0,
        "active_success": 0,
        "active_valid": 0,
        "active_terminal_reason": 0,
        "active_displacement_m": 0,
        "active_cmt": 0,
    }
    max_active_cmt_abs_diff = 0.0
    old_initial_fields = [
        *(f"active_initial_q{index}_deg" for index in range(1, 5)),
        *(f"active_initial_dq{index}_rad_s" for index in range(1, 5)),
    ]
    new_initial_fields = [
        *(f"initial_q{index}_deg" for index in range(1, 5)),
        *(f"initial_dq{index}_rad_s" for index in range(1, 5)),
    ]

    for key in sorted(two_records):
        two = two_records[key]
        three = three_records[key]
        for field in ("walk_direction", "reset_phase"):
            if two[field] != three[field]:
                mismatch_counts[field] += 1
        for field in ("commanded_step_length_m", "commanded_step_height_m"):
            if not floats_match(two[field], three[field]):
                mismatch_counts[field] += 1
        if any(
            not floats_match(two[old_field], three[new_field])
            for old_field, new_field in zip(old_initial_fields, new_initial_fields)
        ):
            mismatch_counts["initial_state"] += 1
        if parse_bool(two["active_success"]) != parse_bool(three["active_success"]):
            mismatch_counts["active_success"] += 1
        if parse_bool(two["active_valid_for_cmt"]) != parse_bool(three["active_valid"]):
            mismatch_counts["active_valid"] += 1
        if two["active_terminal_reason"] != three["active_terminal_reason"]:
            mismatch_counts["active_terminal_reason"] += 1
        if not floats_match(
            two["active_signed_com_displacement_m"],
            three["active_displacement_m"],
        ):
            mismatch_counts["active_displacement_m"] += 1
        if not floats_match(two["active_cmt"], three["active_cmt"]):
            mismatch_counts["active_cmt"] += 1
        old_cmt = finite_float(two["active_cmt"])
        new_cmt = finite_float(three["active_cmt"])
        if old_cmt is not None and new_cmt is not None:
            max_active_cmt_abs_diff = max(
                max_active_cmt_abs_diff,
                abs(old_cmt - new_cmt),
            )

    if any(mismatch_counts.values()):
        raise ValueError(f"Provenance mismatch counts: {mismatch_counts}")

    three_inputs = sorted(three_root.glob("batch*/three_expert_vs_oracle_trials.csv"))
    return {
        "matched_key_count": len(two_records),
        "key_fields": list(KEY_FIELDS),
        "mismatch_counts": mismatch_counts,
        "float_tolerance": FLOAT_MATCH_TOLERANCE,
        "max_active_cmt_abs_diff": max_active_cmt_abs_diff,
        "two_expert_input": {
            "archive_path": (
                "supplementary/S4/fourlink_additional_batches/results/"
                "matched_records.csv"
            ),
            "sha256": sha256(two_path),
        },
        "three_expert_inputs": [
            {
                "archive_path": (
                    "supplementary/S4/three_expert/repository_overlay/results/"
                    "four_link_three_expert_additional_batches/"
                    f"{path.parent.name}/{path.name}"
                ),
                "sha256": sha256(path),
            }
            for path in three_inputs
        ],
    }


def paired_binary_counts(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    three_only = sum(bool(row[f"three_{field}"]) and not bool(row[f"two_{field}"]) for row in rows)
    two_only = sum(bool(row[f"two_{field}"]) and not bool(row[f"three_{field}"]) for row in rows)
    both = sum(bool(row[f"two_{field}"]) and bool(row[f"three_{field}"]) for row in rows)
    neither = len(rows) - three_only - two_only - both
    return {
        "both": both,
        "three_only": three_only,
        "two_only": two_only,
        "neither": neither,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    success = paired_binary_counts(rows, "success")
    valid = paired_binary_counts(rows, "valid")
    common_valid = [row for row in rows if row["two_valid"] and row["three_valid"]]
    two_cmt = [float(row["two_cmt"]) for row in common_valid]
    three_cmt = [float(row["three_cmt"]) for row in common_valid]
    differences = [three - two for two, three in zip(two_cmt, three_cmt)]
    differences_by_batch = {
        batch: [
            float(row["three_cmt"]) - float(row["two_cmt"])
            for row in common_valid
            if row["batch"] == batch
        ]
        for batch in sorted({str(row["batch"]) for row in rows})
    }
    identical = sum(abs(diff) <= 1.0e-12 for diff in differences)

    active_rows = [row for row in rows if str(row["selected_branch"]).startswith("three_expert_active")]
    strict_opportunities = [
        row for row in active_rows if not row["oracle_success"]
    ]
    strict_rescues = [
        row
        for row in active_rows
        if row["three_success"]
        and not row["oracle_success"]
        and not row["two_success"]
    ]
    strict_valid_rescues = [
        row
        for row in active_rows
        if row["three_valid"]
        and not row["oracle_valid"]
        and not row["two_valid"]
    ]

    result = {
        "n_trials": n,
        "two_expert": {
            "success_count": sum(bool(row["two_success"]) for row in rows),
            "success_rate": sum(bool(row["two_success"]) for row in rows) / n,
            "valid_cmt_count": sum(bool(row["two_valid"]) for row in rows),
            "valid_cmt_rate": sum(bool(row["two_valid"]) for row in rows) / n,
        },
        "three_expert": {
            "success_count": sum(bool(row["three_success"]) for row in rows),
            "success_rate": sum(bool(row["three_success"]) for row in rows) / n,
            "valid_cmt_count": sum(bool(row["three_valid"]) for row in rows),
            "valid_cmt_rate": sum(bool(row["three_valid"]) for row in rows) / n,
        },
        "paired_success": {
            **success,
            "three_minus_two_rate": (success["three_only"] - success["two_only"]) / n,
            "three_minus_two_rate_ci95": paired_binary_normal_ci(rows, "success"),
            "exact_mcnemar_p": exact_mcnemar_p(success["three_only"], success["two_only"]),
        },
        "paired_valid_cmt": {
            **valid,
            "three_minus_two_rate": (valid["three_only"] - valid["two_only"]) / n,
            "three_minus_two_rate_ci95": paired_binary_normal_ci(rows, "valid"),
            "exact_mcnemar_p": exact_mcnemar_p(valid["three_only"], valid["two_only"]),
        },
        "common_valid_cmt": {
            "count": len(common_valid),
            "two_expert_mean": fmean(two_cmt),
            "three_expert_mean": fmean(three_cmt),
            "three_minus_two_mean": fmean(differences),
            "three_minus_two_mean_ci95": stratified_bootstrap_mean_ci(
                differences_by_batch
            ),
            "identical_within_1e-12": identical,
        },
        "active_fallback": {
            "selected_count": len(active_rows),
            "success_count": sum(bool(row["three_success"]) for row in active_rows),
            "valid_cmt_count": sum(bool(row["three_valid"]) for row in active_rows),
            "strict_opportunity_count": len(strict_opportunities),
            "strict_success_rescue_count": len(strict_rescues),
            "strict_success_rescue_rate_all": len(strict_rescues) / n,
            "strict_success_rescue_rate_opportunities": (
                len(strict_rescues) / len(strict_opportunities)
            ),
            "strict_valid_cmt_rescue_count": len(strict_valid_rescues),
            "strict_valid_cmt_rescue_rate_all": len(strict_valid_rescues) / n,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "strata": "evaluation_batch",
        },
    }
    return result


def make_joined_rows(
    two_records: dict[tuple[str, str, str], dict[str, str]],
    three_records: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    if set(two_records) != set(three_records):
        missing_three = sorted(set(two_records) - set(three_records))
        missing_two = sorted(set(three_records) - set(two_records))
        raise ValueError(
            f"Unmatched records: missing_three={len(missing_three)}, "
            f"missing_two={len(missing_two)}"
        )

    rows: list[dict[str, object]] = []
    for key in sorted(two_records):
        two = two_records[key]
        three = three_records[key]
        two_valid = parse_bool(two["passive_sign_selector_valid_for_cmt"])
        three_valid = parse_bool(three["selector_valid"])
        row = {
            "batch": key[0],
            "seed": key[1],
            "episode_index": key[2],
            "walk_direction": three["walk_direction"],
            "reset_phase": three["reset_phase"],
            "commanded_step_length_m": finite_float(
                three["commanded_step_length_m"]
            ),
            "commanded_step_height_m": finite_float(
                three["commanded_step_height_m"]
            ),
            **{
                f"initial_q{index}_deg": finite_float(
                    three[f"initial_q{index}_deg"]
                )
                for index in range(1, 5)
            },
            **{
                f"initial_dq{index}_rad_s": finite_float(
                    three[f"initial_dq{index}_rad_s"]
                )
                for index in range(1, 5)
            },
            "active_reference_success": parse_bool(three["active_success"]),
            "active_reference_valid": parse_bool(three["active_valid"]),
            "active_reference_cmt": finite_float(three["active_cmt"]),
            "active_reference_displacement_m": finite_float(
                three["active_displacement_m"]
            ),
            "active_reference_terminal_reason": three["active_terminal_reason"],
            "two_success": parse_bool(two["passive_sign_selector_success"]),
            "three_success": parse_bool(three["selector_success"]),
            "two_valid": two_valid,
            "three_valid": three_valid,
            "two_cmt": finite_float(two["passive_sign_selector_cmt"]) if two_valid else None,
            "three_cmt": finite_float(three["selector_cmt"]) if three_valid else None,
            "selected_branch": three["selector_selected_from"],
            "oracle_success": parse_bool(three["oracle_success"]),
            "oracle_valid": parse_bool(three["oracle_valid"]),
        }
        if two_valid and row["two_cmt"] is None:
            raise ValueError(f"Missing finite two-expert Cmt for {key}")
        if three_valid and row["three_cmt"] is None:
            raise ValueError(f"Missing finite three-expert Cmt for {key}")
        rows.append(row)
    return rows


def load_compact(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    boolean_fields = (
        "two_success",
        "three_success",
        "two_valid",
        "three_valid",
        "oracle_success",
        "oracle_valid",
    )
    nullable_float_fields = (
        "two_cmt",
        "three_cmt",
    )
    if path.suffix == ".gz":
        handle_context = gzip.open(
            path,
            mode="rt",
            newline="",
            encoding="utf-8-sig",
        )
    else:
        handle_context = path.open(
            mode="r",
            newline="",
            encoding="utf-8-sig",
        )
    with handle_context as handle:
        for source in csv.DictReader(handle):
            row: dict[str, object] = dict(source)
            for field in boolean_fields:
                row[field] = parse_bool(source[field])
            for field in nullable_float_fields:
                row[field] = finite_float(source[field])
            rows.append(row)
    return rows


def assert_manuscript_values(summary: dict[str, object]) -> None:
    assert summary["n_trials"] == 10800
    assert summary["two_expert"]["success_count"] == 4996
    assert summary["three_expert"]["success_count"] == 4993
    assert summary["two_expert"]["valid_cmt_count"] == 2611
    assert summary["three_expert"]["valid_cmt_count"] == 2681
    assert summary["common_valid_cmt"]["count"] == 2554
    assert summary["active_fallback"]["selected_count"] == 5345
    assert summary["active_fallback"]["success_count"] == 177
    assert summary["active_fallback"]["valid_cmt_count"] == 72
    assert summary["active_fallback"]["strict_opportunity_count"] == 5253
    assert summary["active_fallback"]["strict_success_rescue_count"] == 122
    assert summary["active_fallback"]["strict_valid_cmt_rescue_count"] == 50


def structures_match(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            structures_match(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            structures_match(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    return left == right


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as binary_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary_handle,
            mtime=0,
        ) as compressed:
            with compressed_text_writer(compressed) as text_handle:
                writer = csv.DictWriter(
                    text_handle,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)


def compressed_text_writer(compressed: gzip.GzipFile):
    import io

    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def flatten_summary(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for section, values in summary.items():
        if isinstance(values, dict):
            for metric, value in values.items():
                rows.append({"section": section, "metric": metric, "value": value})
        else:
            rows.append({"section": "overall", "metric": section, "value": values})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-expert-records", type=Path)
    parser.add_argument("--three-expert-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check-published",
        type=Path,
        help="verify a published compact output directory without the full S4 inputs",
    )
    parser.add_argument("--assert-manuscript-values", action="store_true")
    args = parser.parse_args()

    if args.check_published is not None:
        compact_path = args.check_published / "paired_trials_compact.csv.gz"
        saved_path = args.check_published / "summary.json"
        compact_summary = summarize(load_compact(compact_path))
        assert_manuscript_values(compact_summary)
        saved_summary = json.loads(saved_path.read_text(encoding="utf-8"))
        for section in (
            "two_expert",
            "three_expert",
            "paired_success",
            "paired_valid_cmt",
            "common_valid_cmt",
            "active_fallback",
        ):
            if not structures_match(
                compact_summary[section],
                saved_summary[section],
            ):
                raise RuntimeError(f"Published compact summary differs in {section}")
        print(
            "Published compact comparison check: PASS "
            f"({compact_summary['n_trials']} matched trials)"
        )
        return

    if (
        args.two_expert_records is None
        or args.three_expert_root is None
        or args.output_dir is None
    ):
        parser.error(
            "full regeneration requires --two-expert-records, "
            "--three-expert-root, and --output-dir"
        )

    two_records = load_two_expert(args.two_expert_records)
    three_records = load_three_expert(args.three_expert_root)
    provenance = validate_full_pairing(
        two_records,
        three_records,
        args.two_expert_records,
        args.three_expert_root,
    )
    joined = make_joined_rows(two_records, three_records)
    summary = summarize(joined)

    if args.assert_manuscript_values:
        assert_manuscript_values(summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "provenance_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    flat = flatten_summary(summary)
    write_csv(output_dir / "summary.csv", flat, ["section", "metric", "value"])

    compact_fields = [
        "batch",
        "seed",
        "episode_index",
        "two_success",
        "three_success",
        "two_valid",
        "three_valid",
        "two_cmt",
        "three_cmt",
        "selected_branch",
        "oracle_success",
        "oracle_valid",
    ]
    strict_context_fields = [
        "walk_direction",
        "reset_phase",
        "commanded_step_length_m",
        "commanded_step_height_m",
        "initial_q1_deg",
        "initial_q2_deg",
        "initial_q3_deg",
        "initial_q4_deg",
        "initial_dq1_rad_s",
        "initial_dq2_rad_s",
        "initial_dq3_rad_s",
        "initial_dq4_rad_s",
        "active_reference_success",
        "active_reference_valid",
        "active_reference_cmt",
        "active_reference_displacement_m",
        "active_reference_terminal_reason",
    ]
    write_gzip_csv(
        output_dir / "paired_trials_compact.csv.gz",
        joined,
        compact_fields,
    )
    strict_rescues = [
        row
        for row in joined
        if str(row["selected_branch"]).startswith("three_expert_active")
        and row["three_success"]
        and not row["oracle_success"]
        and not row["two_success"]
    ]
    write_csv(
        output_dir / "strict_success_rescue_trials.csv",
        strict_rescues,
        compact_fields + strict_context_fields,
    )

    batch_rows = []
    for batch in sorted({str(row["batch"]) for row in joined}):
        batch_summary = summarize([row for row in joined if row["batch"] == batch])
        batch_rows.append(
            {
                "batch": batch,
                "n": batch_summary["n_trials"],
                "two_success": batch_summary["two_expert"]["success_count"],
                "three_success": batch_summary["three_expert"]["success_count"],
                "two_valid": batch_summary["two_expert"]["valid_cmt_count"],
                "three_valid": batch_summary["three_expert"]["valid_cmt_count"],
                "common_valid": batch_summary["common_valid_cmt"]["count"],
                "two_common_valid_mean_cmt": batch_summary["common_valid_cmt"]["two_expert_mean"],
                "three_common_valid_mean_cmt": batch_summary["common_valid_cmt"]["three_expert_mean"],
                "strict_success_rescues": batch_summary["active_fallback"][
                    "strict_success_rescue_count"
                ],
            }
        )
    write_csv(
        output_dir / "per_batch_summary.csv",
        batch_rows,
        list(batch_rows[0]),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
