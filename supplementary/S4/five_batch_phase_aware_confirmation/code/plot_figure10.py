"""Draw original-layout Figure 10 from the frozen five-batch confirmation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_DLL_HANDLES = []
if os.name == "nt":
    for relative in (Path("Library") / "bin", Path("DLLs")):
        directory = Path(sys.prefix) / relative
        if directory.is_dir():
            os.environ["PATH"] = (
                str(directory)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_METRICS = (
    ROOT / "manuscript_metrics" / "five_batch_manuscript_metrics.json"
)
DEFAULT_OUTPUT = ROOT / "generated_figures"
CONTROLLER_ORDER = (
    "active",
    "hard_mask",
    "two_expert",
    "proposed_three_expert",
    "sign_oracle",
)
LABELS = {
    "active": "Active\nPPO",
    "hard_mask": "Hard\nmask",
    "two_expert": "2-expert\nselector",
    "proposed_three_expert": "Proposed\n3-expert",
    "sign_oracle": "Oracle\nenvelope",
}
SUCCESS_COLOR = "#B4C0E4"
VALID_COLOR = "#42949E"
HIP_COLOR = "#0F4D92"
SUPPORT_COLOR = "#8BCF8B"
SWING_COLOR = "#9A4D8E"
PAIR_COLORS = {
    "hard_mask": "#7884B4",
    "two_expert": "#9D9D9D",
    "proposed_three_expert": "#C65D6A",
    "sign_oracle": "#6C8F5A",
}
Z95 = 1.959963984540054


def wilson(successes: int, total: int):
    p = successes / total
    denominator = 1.0 + Z95 * Z95 / total
    center = (p + Z95 * Z95 / (2.0 * total)) / denominator
    half = (
        Z95
        * math.sqrt(
            p * (1.0 - p) / total
            + Z95 * Z95 / (4.0 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def percent_error(successes: int, total: int):
    rate = 100.0 * successes / total
    lower, upper = wilson(successes, total)
    return rate, rate - 100.0 * lower, 100.0 * upper - rate


def style_axis(axis):
    axis.tick_params(width=0.7, length=3, labelsize=7.2)
    axis.grid(axis="y", alpha=0.24, linewidth=0.6, zorder=0)


def panel_letter(axis, letter):
    axis.text(
        -0.04,
        1.075,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        weight="bold",
    )


def draw_sequence(axis, y, label, note, values, color):
    axis.text(
        -2.82,
        y + 0.12,
        label,
        ha="left",
        va="center",
        fontsize=7.1,
        weight="bold",
    )
    axis.text(
        -2.82,
        y - 0.26,
        note,
        ha="left",
        va="center",
        fontsize=5.5,
        color="#666666",
    )
    left, width, gap = 0.95, 0.82, 0.08
    total_width = len(values) * (width + gap) - gap
    axis.add_patch(
        Rectangle(
            (left - 0.08, y - 0.38),
            total_width + 0.16,
            0.76,
            facecolor="none",
            edgecolor=color,
            linewidth=1.0,
        )
    )
    for index, value in enumerate(values):
        x = left + index * (width + gap)
        if value > 0:
            face, symbol = "#E9A6A1", "+"
        elif value < 0:
            face, symbol = "#AFC7E8", "-"
        else:
            face, symbol = "#D8D8D8", "0"
        axis.add_patch(
            Rectangle(
                (x, y - 0.31),
                width,
                0.62,
                facecolor=face,
                edgecolor="white",
                linewidth=0.6,
            )
        )
        axis.text(
            x + width / 2.0,
            y,
            symbol,
            ha="center",
            va="center",
            fontsize=6.8,
        )


def draw_panel_a(axis):
    panel_letter(axis, "a")
    axis.set_title(
        "Where the sign decision is made",
        loc="left",
        fontsize=8.2,
        weight="bold",
        pad=8,
    )
    rows = (
        (
            "Active PPO",
            "unrestricted signs",
            [1, -1, 1, 0, -1, 1, -1, 1],
            "#555555",
        ),
        (
            "Hard mask",
            "stepwise hard mask",
            [1, 1, -1, -1, 1, 0, -1, 1],
            "#7884B4",
        ),
        (
            "3-expert selector",
            "one expert committed",
            [1, 1, 1, 1, 1, 1, 1, 1],
            "#C65D6A",
        ),
        (
            "Oracle envelope",
            "offline lower-Cmt sign",
            [-1, -1, -1, -1, -1, -1, -1, -1],
            "#6C8F5A",
        ),
    )
    for y, row in zip((3.15, 2.13, 1.11, 0.09), rows):
        draw_sequence(axis, y, *row)
    axis.text(
        4.50,
        3.67,
        "control steps within one transition",
        ha="center",
        fontsize=5.8,
        color="#666666",
    )
    axis.text(
        4.50,
        -0.38,
        "colored strips are schematic",
        ha="center",
        fontsize=5.5,
        color="#888888",
    )
    axis.set_xlim(-3.0, 8.25)
    axis.set_ylim(-0.55, 3.9)
    axis.axis("off")


def draw_panel_b(axis, controllers):
    panel_letter(axis, "b")
    x = np.arange(len(CONTROLLER_ORDER), dtype=float)
    width = 0.32
    success, success_error, valid, valid_error = [], [], [], []
    for name in CONTROLLER_ORDER:
        row = controllers[name]
        rate, low, high = percent_error(row["success_count"], row["n"])
        success.append(rate)
        success_error.append((low, high))
        rate, low, high = percent_error(row["valid_count"], row["n"])
        valid.append(rate)
        valid_error.append((low, high))
    axis.bar(
        x - width / 2,
        success,
        width,
        color=SUCCESS_COLOR,
        label="Success",
        yerr=np.asarray(success_error).T,
        error_kw={
            "ecolor": "#444444",
            "elinewidth": 0.8,
            "capsize": 2.5,
            "capthick": 0.8,
        },
        zorder=3,
    )
    axis.bar(
        x + width / 2,
        valid,
        width,
        color=VALID_COLOR,
        label="Valid Cmt",
        yerr=np.asarray(valid_error).T,
        error_kw={
            "ecolor": "#444444",
            "elinewidth": 0.8,
            "capsize": 2.5,
            "capthick": 0.8,
        },
        zorder=3,
    )
    axis.set_xticks(x, [LABELS[name] for name in CONTROLLER_ORDER])
    axis.set_ylabel("Rate (%)", fontsize=7.2)
    axis.set_ylim(0, 55)
    axis.legend(fontsize=6.4, loc="upper right", handlelength=1.1)
    axis.text(
        0.02,
        0.93,
        f"n = {controllers['active']['n']:,}/controller",
        transform=axis.transAxes,
        fontsize=5.8,
        color="#666666",
    )
    style_axis(axis)


def draw_panel_c(axis, controllers):
    panel_letter(axis, "c")
    axis.set_title(
        "Mean actuator Cmt on valid transitions", fontsize=8.2, pad=8
    )
    x = np.arange(len(CONTROLLER_ORDER), dtype=float)
    hip = np.asarray(
        [
            controllers[name]["component_mean_cmt"]["hip"]
            for name in CONTROLLER_ORDER
        ]
    )
    support = np.asarray(
        [
            controllers[name]["component_mean_cmt"]["support_knee"]
            for name in CONTROLLER_ORDER
        ]
    )
    swing = np.asarray(
        [
            controllers[name]["component_mean_cmt"]["swing_knee"]
            for name in CONTROLLER_ORDER
        ]
    )
    totals = hip + support + swing
    axis.bar(x, hip, color=HIP_COLOR, width=0.78, label="Hip", zorder=3)
    axis.bar(
        x,
        support,
        bottom=hip,
        color=SUPPORT_COLOR,
        width=0.78,
        label="Support knee",
        zorder=3,
    )
    axis.bar(
        x,
        swing,
        bottom=hip + support,
        color=SWING_COLOR,
        width=0.78,
        label="Swing knee",
        zorder=3,
    )
    axis.set_xticks(x, [LABELS[name] for name in CONTROLLER_ORDER])
    axis.set_ylabel("Mean Cmt on valid transitions", fontsize=7.0)
    axis.set_ylim(0, 1.85)
    axis.legend(fontsize=5.8, loc="upper right", handlelength=1.0)
    for xi, total in zip(x, totals):
        axis.text(
            xi,
            total + 0.035,
            f"{total:.3g}",
            ha="center",
            va="bottom",
            fontsize=6.1,
        )
    style_axis(axis)


def draw_panel_d(axis, pairs):
    panel_letter(axis, "d")
    axis.set_title(
        "Fraction with lower Cmt than Active PPO", fontsize=8.2, pad=8
    )
    comparisons = (
        ("Hard mask", "hard_mask", "active_vs_hard_mask"),
        ("2-expert selector", "two_expert", "active_vs_two_expert"),
        (
            "Proposed 3-expert",
            "proposed_three_expert",
            "active_vs_proposed_three_expert",
        ),
        ("Oracle envelope", "sign_oracle", "active_vs_sign_oracle"),
    )
    y = np.arange(len(comparisons), dtype=float)[::-1]
    for yi, (label, name, pair_name) in zip(y, comparisons):
        row = pairs[pair_name]
        total = row["common_valid_count"]
        lower_count = row["second_lower_count"]
        fraction = lower_count / total
        lower, upper = wilson(lower_count, total)
        value = 100.0 * fraction
        error = np.asarray(
            [[value - 100.0 * lower], [100.0 * upper - value]]
        )
        axis.errorbar(
            value,
            yi,
            xerr=error,
            fmt="o",
            color=PAIR_COLORS[name],
            ecolor=PAIR_COLORS[name],
            markersize=4.3,
            elinewidth=1.3,
            capsize=0,
            zorder=3,
        )
        axis.text(
            value + 2.8,
            yi,
            f"{value:.1f}%\n(n={total})",
            ha="left",
            va="center",
            fontsize=5.8,
        )
    axis.axvline(
        50.0, color="#777777", linestyle="--", linewidth=0.8, zorder=1
    )
    axis.set_yticks(y, [item[0] for item in comparisons])
    axis.set_xlim(30, 96)
    axis.set_ylim(-0.25, 3.25)
    axis.set_xlabel(
        "Fraction with lower Cmt than Active PPO (%)",
        fontsize=6.8,
        labelpad=12,
    )
    axis.grid(axis="x", alpha=0.24, linewidth=0.6, zorder=0)
    axis.tick_params(width=0.7, length=3, labelsize=7.2)


def build_figure(report):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(6.515, 5.193),
        gridspec_kw={
            "width_ratios": (1.05, 1.0),
            "height_ratios": (1.0, 1.03),
        },
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.095,
        top=0.875,
        wspace=0.36,
        hspace=0.58,
    )
    draw_panel_a(axes[0, 0])
    draw_panel_b(axes[0, 1], report["controllers"])
    draw_panel_c(axes[1, 0], report["controllers"])
    draw_panel_d(axes[1, 1], report["pairs"])
    figure.suptitle(
        "Five-batch four-link comparison with phase-aware three-expert routing",
        y=0.972,
        fontsize=9.25,
        weight="bold",
    )
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--basename", default="Figure10_phase_aware_five_batch"
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    figure = build_figure(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "tiff", "pdf", "svg"):
        output = args.output_dir / f"{args.basename}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = args.dpi
        figure.savefig(output, **kwargs)
        print(f"Saved: {output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
