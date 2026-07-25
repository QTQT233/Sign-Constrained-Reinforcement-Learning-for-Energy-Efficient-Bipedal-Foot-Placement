"""Post-process the completed paired hold-versus-requery trial table."""

import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"
TRIALS = DATA / "hold_vs_requery_trials.csv"
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPLICATES = 10000
Z95 = 1.959963984540054


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value):
    return str(value).lower() == "true"


def exact_mcnemar(reference_only, comparison_only):
    discordant = reference_only + comparison_only
    if discordant == 0:
        return 1.0
    smaller = min(reference_only, comparison_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def paired_binary(rows, reference_field, comparison_field):
    reference = [as_bool(row[reference_field]) for row in rows]
    comparison = [as_bool(row[comparison_field]) for row in rows]
    both = sum(ref and comp for ref, comp in zip(reference, comparison))
    reference_only = sum(ref and not comp for ref, comp in zip(reference, comparison))
    comparison_only = sum(not ref and comp for ref, comp in zip(reference, comparison))
    neither = len(rows) - both - reference_only - comparison_only
    difference = (comparison_only - reference_only) / len(rows)
    q_value = (reference_only + comparison_only) / len(rows)
    standard_error = math.sqrt(
        max(0.0, (q_value - difference * difference) / len(rows))
    )
    paired_differences = [
        float(comp) - float(ref) for ref, comp in zip(reference, comparison)
    ]
    bootstrap_interval = bootstrap_ci(paired_differences)
    return {
        "n": len(rows),
        "both": both,
        "hold_only": reference_only,
        "requery_only": comparison_only,
        "neither": neither,
        "hold_rate": (both + reference_only) / len(rows),
        "requery_rate": (both + comparison_only) / len(rows),
        "requery_minus_hold_risk_difference": difference,
        "requery_minus_hold_percentage_points": 100.0 * difference,
        "paired_ci95": [
            difference - Z95 * standard_error,
            difference + Z95 * standard_error,
        ],
        "paired_ci95_percentage_points": [
            100.0 * (difference - Z95 * standard_error),
            100.0 * (difference + Z95 * standard_error),
        ],
        "paired_bootstrap_ci95": bootstrap_interval,
        "paired_bootstrap_ci95_percentage_points": [
            100.0 * bootstrap_interval[0],
            100.0 * bootstrap_interval[1],
        ],
        "exact_mcnemar_p_two_sided": exact_mcnemar(
            reference_only, comparison_only
        ),
    }


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(differences):
    if not differences:
        return [float("nan"), float("nan")]
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(differences)
    values = [
        statistics.fmean(rng.choices(differences, k=count))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return [quantile(values, 0.025), quantile(values, 0.975)]


def cmt_summary(rows):
    both_valid = [row for row in rows if as_bool(row["both_valid_for_cmt"])]
    differences = [float(row["requery_minus_hold_cmt"]) for row in both_valid]
    requery_lower = sum(as_bool(row["requery_lower_cmt"]) for row in both_valid)
    hold_lower = sum(value > 0.0 for value in differences)
    ties = len(differences) - requery_lower - hold_lower
    return {
        "n_both_valid": len(both_valid),
        "hold_mean_cmt": statistics.fmean(
            float(row["hold_cmt"]) for row in both_valid
        ) if both_valid else float("nan"),
        "requery_mean_cmt": statistics.fmean(
            float(row["requery_cmt"]) for row in both_valid
        ) if both_valid else float("nan"),
        "mean_requery_minus_hold_cmt": statistics.fmean(differences)
        if differences else float("nan"),
        "bootstrap_ci95_mean_difference": bootstrap_ci(differences),
        "requery_lower_count": requery_lower,
        "hold_lower_count": hold_lower,
        "tie_count": ties,
        "requery_lower_fraction": requery_lower / len(both_valid)
        if both_valid else float("nan"),
    }


rows = read_csv(TRIALS)
switched = [row for row in rows if int(row["requery_sign_switch_count"]) > 0]
not_switched = [row for row in rows if int(row["requery_sign_switch_count"]) == 0]

result = {
    "scope": "one frozen selector and one frozen positive/negative expert pair",
    "case_count": len(rows),
    "success": paired_binary(rows, "hold_success", "requery_success"),
    "valid_cmt": paired_binary(
        rows, "hold_valid_for_cmt", "requery_valid_for_cmt"
    ),
    "cmt_on_both_valid": cmt_summary(rows),
    "switch_strata": {
        "no_requery_switch": {
            "case_count": len(not_switched),
            "success": paired_binary(
                not_switched, "hold_success", "requery_success"
            ),
            "valid_cmt": paired_binary(
                not_switched, "hold_valid_for_cmt", "requery_valid_for_cmt"
            ),
            "cmt_on_both_valid": cmt_summary(not_switched),
        },
        "at_least_one_requery_switch": {
            "case_count": len(switched),
            "success": paired_binary(
                switched, "hold_success", "requery_success"
            ),
            "valid_cmt": paired_binary(
                switched, "hold_valid_for_cmt", "requery_valid_for_cmt"
            ),
            "cmt_on_both_valid": cmt_summary(switched),
        },
    },
    "requery_switch_count_distribution": dict(
        sorted(
            Counter(int(row["requery_sign_switch_count"]) for row in rows).items()
        )
    ),
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "interpretation_boundary": (
        "This is a paired inference-time query-schedule comparison conditional "
        "on the specified frozen selector and experts; it is not a comparison "
        "across independent training runs."
    ),
    "cmt_analysis_rule": (
        "The all-valid hold and requery Cmt means use different validity sets "
        "and are descriptive only. The primary paired Cmt comparison is the "
        "within-case requery-minus-hold difference on the 419 both-valid cases."
    ),
}
(DATA / "paired_inference.json").write_text(
    json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
)

flat_rows = []
for family in ("success", "valid_cmt", "cmt_on_both_valid"):
    for metric, value in result[family].items():
        flat_rows.append(
            {
                "analysis_set": "all_2160",
                "metric_family": family,
                "metric": metric,
                "value": json.dumps(value) if isinstance(value, list) else value,
            }
        )
for stratum, payload in result["switch_strata"].items():
    flat_rows.append(
        {
            "analysis_set": stratum,
            "metric_family": "case_count",
            "metric": "n",
            "value": payload["case_count"],
        }
    )
    for family in ("success", "valid_cmt", "cmt_on_both_valid"):
        for metric, value in payload[family].items():
            flat_rows.append(
                {
                    "analysis_set": stratum,
                    "metric_family": family,
                    "metric": metric,
                    "value": json.dumps(value)
                    if isinstance(value, list) else value,
                }
            )
with (DATA / "paired_inference.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
    writer.writeheader()
    writer.writerows(flat_rows)

transitions = Counter(
    (row["hold_terminal_reason"], row["requery_terminal_reason"])
    for row in rows
)
with (DATA / "terminal_reason_transitions.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.writer(handle)
    writer.writerow(["hold_terminal_reason", "requery_terminal_reason", "count"])
    for (hold_reason, requery_reason), count in sorted(transitions.items()):
        writer.writerow([hold_reason, requery_reason, count])

with (DATA / "switch_count_distribution.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.writer(handle)
    writer.writerow(["requery_sign_switch_count", "case_count"])
    for switch_count, count in sorted(
        Counter(int(row["requery_sign_switch_count"]) for row in rows).items()
    ):
        writer.writerow([switch_count, count])

print(json.dumps(result, indent=2, allow_nan=True))
