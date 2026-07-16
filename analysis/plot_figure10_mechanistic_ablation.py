"""Reproduce the manuscript four-link diagnostic figure from released summaries.

The final table/figure reorganization numbers this panel set as Figure 8.  The
historical filename is retained so links from the intermediate manuscript do
not break; ``plot_figure08_four_link_diagnostics.py`` is the final-number alias.

The figure separates the controller timing mechanism (panel a), full-domain
success/valid-Cmt rates (panel b), controller-level valid-transition Cmt
components (panel c), and paired lower-Cmt fractions (panel d).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "four_link" / "legacy_manuscript"
CONTROLLER_SUMMARY = DATA / "controller_summary.csv"
PAIRED_SUMMARY = DATA / "paired_summary.csv"
REFERENCE_SUMMARY = DATA / "active_reference_summary.csv"

ORDER = (
    "active",
    "active_full",
    "per_step_sign",
    "passive_sign_selector",
    "passive_oracle_envelope",
)
LABELS = {
    "active": "Active\nPPO",
    "active_full": "Active-full\nreference",
    "per_step_sign": "Per-step\nsign",
    "passive_sign_selector": "Online\nselector",
    "passive_oracle_envelope": "Oracle\nenvelope",
}
COLORS = {
    "active": "#4C78A8",
    "active_full": "#9D9D9D",
    "per_step_sign": "#7A6BB7",
    "passive_sign_selector": "#D65F5F",
    "passive_oracle_envelope": "#6C9E5B",
}
Z95 = 1.959963984540054


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def wilson(successes: float, n: int) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denominator
    return center - half, center + half


def draw_sequence(ax, y: float, values: list[int], color: str, label: str, note: str) -> None:
    ax.text(-0.3, y + 0.18, label, ha="right", va="center", fontsize=8.5, weight="bold")
    ax.text(-0.3, y - 0.18, note, ha="right", va="center", fontsize=6.8, color="#555555")
    for i, value in enumerate(values):
        fill = "#F4B6B6" if value > 0 else "#B9D0EC" if value < 0 else "#EEEEEE"
        ax.add_patch(Rectangle((i, y - 0.22), 0.84, 0.44, facecolor=fill,
                               edgecolor=color, linewidth=0.8))
        ax.text(i + 0.42, y, "+" if value > 0 else "-" if value < 0 else "0",
                ha="center", va="center", fontsize=8)


def main(default_basename: str = "figure10_mechanistic_ablation") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--basename", default=default_basename)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    controllers = {row["controller"]: row for row in read_rows(CONTROLLER_SUMMARY)}
    paired = {row["passive_controller"]: row for row in read_rows(PAIRED_SUMMARY)}
    references = {row["comparison_controller"]: row for row in read_rows(REFERENCE_SUMMARY)}

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.08, wspace=0.08, hspace=0.16)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a: timing mechanism schematic
    ax_a.set_title("a   Where the sign decision is made", loc="left", weight="bold", fontsize=10)
    draw_sequence(ax_a, 3.2, [1, -1, 1, 0, -1, 1, -1, 1], COLORS["active"],
                  "Active PPO", "unrestricted signs")
    draw_sequence(ax_a, 2.2, [1, 1, -1, -1, 1, 0, -1, 1], COLORS["per_step_sign"],
                  "Per-step sign", "mask can change each step")
    draw_sequence(ax_a, 1.2, [1, 1, 1, 1, 1, 1, 1, 1], COLORS["passive_sign_selector"],
                  "Online selector", "one sign committed")
    draw_sequence(ax_a, 0.2, [-1, -1, -1, -1, -1, -1, -1, -1], COLORS["passive_oracle_envelope"],
                  "Oracle envelope", "offline lower-Cmt sign")
    ax_a.text(3.4, 3.75, "control steps within one transition", ha="center", fontsize=7, color="#555555")
    ax_a.set_xlim(-2.7, 8.0)
    ax_a.set_ylim(-0.25, 4.0)
    ax_a.axis("off")

    # b: rates on all 2160 trials
    ax_b.set_title("b   Task completion and valid-Cmt rates", loc="left", weight="bold", fontsize=10)
    x = np.arange(len(ORDER))
    success = np.array([float(controllers[key]["success_rate_all"]) for key in ORDER]) * 100
    valid = np.array([float(controllers[key]["valid_cmt_rate_all"]) for key in ORDER]) * 100
    width = 0.36
    ax_b.bar(x - width / 2, success, width, color="#AABBE0", label="Success")
    ax_b.bar(x + width / 2, valid, width, color="#4D9AA6", label="Valid Cmt")
    ax_b.set_xticks(x, [LABELS[key] for key in ORDER], fontsize=7.5)
    ax_b.set_ylabel("Rate (%)")
    ax_b.set_ylim(0, 55)
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_b.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax_b.text(0.02, 0.88, "n = 2160/controller", transform=ax_b.transAxes, fontsize=7, color="#555555")

    # c: valid-transition Cmt components
    ax_c.set_title("c   Selector lowers actuator Cmt", loc="left", weight="bold", fontsize=10)
    hip = np.array([float(controllers[key]["mean_hip_cmt_valid"]) for key in ORDER])
    support = np.array([float(controllers[key]["mean_support_knee_cmt_valid"]) for key in ORDER])
    swing = np.array([float(controllers[key]["mean_swing_knee_cmt_valid"]) for key in ORDER])
    ax_c.bar(x, hip, color="#205493", label="Hip")
    ax_c.bar(x, support, bottom=hip, color="#9AC28F", label="Support knee")
    ax_c.bar(x, swing, bottom=hip + support, color="#985A8C", label="Swing knee")
    ax_c.set_xticks(x, [LABELS[key] for key in ORDER], fontsize=7.5)
    ax_c.set_ylabel("Mean Cmt (valid transitions)")
    ax_c.set_ylim(0, 1.65)
    ax_c.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_c.legend(frameon=False, fontsize=7.2, loc="upper right")
    totals = hip + support + swing
    for xi, value in zip(x, totals):
        ax_c.text(xi, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8)

    # d: paired lower-Cmt fractions with Wilson intervals
    ax_d.set_title("d   Paired Cmt dominance", loc="left", weight="bold", fontsize=10)
    comparisons = [
        ("Active-full reference", references["active_full"], "both_valid_pairs",
         "fraction_comparison_lower_cmt", COLORS["active_full"]),
        ("Per-step sign", references["per_step_sign"], "both_valid_pairs",
         "fraction_comparison_lower_cmt", COLORS["per_step_sign"]),
        ("Online selector", paired["passive_sign_selector"], "both_valid_pairs",
         "fraction_passive_lower_cmt", COLORS["passive_sign_selector"]),
        ("Oracle envelope", paired["passive_oracle_envelope"], "both_valid_pairs",
         "fraction_passive_lower_cmt", COLORS["passive_oracle_envelope"]),
    ]
    y = np.arange(len(comparisons))[::-1]
    for yi, (label, row, n_field, fraction_field, color) in zip(y, comparisons):
        n = int(row[n_field])
        fraction = float(row[fraction_field])
        lower, upper = wilson(round(fraction * n), n)
        ax_d.errorbar(fraction * 100, yi,
                      xerr=[[fraction * 100 - lower * 100], [upper * 100 - fraction * 100]],
                      fmt="o", color=color, ecolor=color, capsize=3, markersize=5)
        ax_d.text(fraction * 100 + 1.5, yi, f"{fraction * 100:.1f}%\n(n={n})",
                  va="center", fontsize=7)
    ax_d.axvline(50, color="#777777", linestyle="--", linewidth=0.8)
    ax_d.set_yticks(y, [item[0] for item in comparisons], fontsize=8)
    ax_d.set_xlim(30, 94)
    ax_d.set_xlabel("Fraction with lower Cmt than Active PPO (%)")
    ax_d.grid(axis="x", alpha=0.25, linewidth=0.6)

    fig.suptitle(
        "Four-link mechanistic ablation of transition-level sign commitment",
        fontsize=11,
        weight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        path = args.output_dir / f"{args.basename}.{suffix}"
        fig.savefig(path, dpi=350 if suffix == "png" else None, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
