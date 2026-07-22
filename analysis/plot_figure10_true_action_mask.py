"""Create Figure 10 for the frozen scratch true-action-mask comparison.

Figure contract
---------------
Core conclusion: Under the shared V22_3/V9 fixed-checkpoint evaluator, the
transition-persistent selector's low-Cmt pattern is not reproduced by the
scratch true per-step hard mask; this comparison is not a strict one-factor
causal demonstration.
Archetype: 2 x 2 quantitative grid with a mechanism schematic.
Backend: Python/matplotlib only.
Final size: 5.764 x 8.2 inches; editable SVG/PDF text; 600-dpi PNG.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    REPO_ROOT / "data" / "four_link" / "true_action_mask_scratch_c090_epoch1275"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "true_action_mask"
Z95 = 1.959963984540054

ORDER = (
    "active",
    "active_full",
    "true_action_mask_scratch",
    "passive_sign_selector",
    "passive_oracle_envelope",
)
LABELS = {
    "active": "Active",
    "active_full": "Full",
    "true_action_mask_scratch": "Mask",
    "passive_sign_selector": "Selector",
    "passive_oracle_envelope": "Oracle",
}
COLORS = {
    "active": "#4C78A8",
    "active_full": "#8C8C8C",
    "true_action_mask_scratch": "#7566B1",
    "passive_sign_selector": "#C85454",
    "passive_oracle_envelope": "#629456",
}
JOINT_COLORS = {
    "Hip": "#376FA3",
    "Support knee": "#8DBA83",
    "Swing knee": "#96628D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="figure10_true_action_mask")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + Z95 * Z95 / total
    center = (p + Z95 * Z95 / (2.0 * total)) / denominator
    half = Z95 * math.sqrt(
        p * (1.0 - p) / total + Z95 * Z95 / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _ = to_rgba(color)
    return red, green, blue, alpha


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 7.5,
    linewidth: float = 0.7,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )


def arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
    )


def draw_panel_a(ax: plt.Axes, source_rows: list[dict]) -> None:
    ax.set_title("a  Decision persistence", loc="left", fontweight="bold")
    ax.set_axis_off()
    ax.text(
        0.0,
        0.96,
        "Mechanism schematic; not rollout data",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
        va="top",
    )

    cards = (
        (
            0.53,
            COLORS["true_action_mask_scratch"],
            "#F4F1FA",
            "True per-step hard mask",
            "Every step:  s[t] -> z[t] in {-1, +1}\n"
            "A(z[t]): hip sign in {0, z[t]}; knees 3 x 3\n"
            "18/27 actions; z[t] may change at t + 1",
        ),
        (
            0.08,
            COLORS["passive_sign_selector"],
            "#FBEFEF",
            "Transition-persistent selector",
            "Transition start:  s0 -> z = pi_sel(s0)\n"
            "A(z): hip sign in {0, z}; knees 3 x 3\n"
            "18 actions; the same z until termination",
        ),
    )
    for y, edge, face, title, body in cards:
        card = FancyBboxPatch(
            (0.02, y),
            0.96,
            0.34,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(card)
        ax.text(
            0.06,
            y + 0.285,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=edge,
        )
        ax.text(
            0.06,
            y + 0.205,
            body,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            linespacing=1.35,
            color="#222222",
        )

    source_rows.extend(
        [
            {
                "panel": "a",
                "record": "mechanism",
                "method": "true_action_mask_scratch",
                "metric": "actions_before_mask",
                "value": 27,
                "description": "Latent sign is refreshed every control step.",
            },
            {
                "panel": "a",
                "record": "mechanism",
                "method": "true_action_mask_scratch",
                "metric": "valid_actions_after_mask",
                "value": 18,
                "description": "Nine zero-hip actions plus nine actions with the selected sign.",
            },
            {
                "panel": "a",
                "record": "mechanism",
                "method": "passive_sign_selector",
                "metric": "selector_refresh",
                "value": "once_per_transition",
                "description": "Selected one-sided expert is retained until termination.",
            },
        ]
    )


def controller_rates(
    rollout_rows: list[dict[str, str]], source_rows: list[dict]
) -> dict[str, dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rollout_rows:
        grouped[row["controller"]].append(row)
    missing = [controller for controller in ORDER if controller not in grouped]
    if missing:
        raise ValueError(f"Missing controllers: {missing}")

    result: dict[str, dict] = {}
    for controller in ORDER:
        rows = grouped[controller]
        if len(rows) != 2160:
            raise ValueError(f"{controller} has {len(rows)} rows, expected 2160")
        success = sum(as_bool(row["success"]) for row in rows)
        valid = sum(as_bool(row["valid_for_cmt"]) for row in rows)
        success_ci = wilson(success, len(rows))
        valid_ci = wilson(valid, len(rows))
        result[controller] = {
            "n": len(rows),
            "success_count": success,
            "success_rate": success / len(rows),
            "success_ci": success_ci,
            "valid_count": valid,
            "valid_rate": valid / len(rows),
            "valid_ci": valid_ci,
        }
        for metric, count, rate, interval in (
            ("success_rate", success, success / len(rows), success_ci),
            ("valid_cmt_rate", valid, valid / len(rows), valid_ci),
        ):
            source_rows.append(
                {
                    "panel": "b",
                    "record": "controller_rate",
                    "method": controller,
                    "metric": metric,
                    "value": rate,
                    "ci_low": interval[0],
                    "ci_high": interval[1],
                    "count": count,
                    "n": len(rows),
                    "interval": "Wilson 95% CI",
                }
            )
    return result


def draw_panel_b(ax: plt.Axes, rates: dict[str, dict]) -> None:
    ax.set_title(
        "b  Success and valid-Cₘₜ\nn = 2160 each; Wilson 95% CI",
        loc="left",
        fontweight="bold",
    )
    x = np.arange(len(ORDER), dtype=float)
    width = 0.34
    success = np.array([rates[key]["success_rate"] * 100.0 for key in ORDER])
    valid = np.array([rates[key]["valid_rate"] * 100.0 for key in ORDER])
    success_low = np.array([rates[key]["success_ci"][0] * 100.0 for key in ORDER])
    success_high = np.array([rates[key]["success_ci"][1] * 100.0 for key in ORDER])
    valid_low = np.array([rates[key]["valid_ci"][0] * 100.0 for key in ORDER])
    valid_high = np.array([rates[key]["valid_ci"][1] * 100.0 for key in ORDER])
    method_colors = [COLORS[key] for key in ORDER]
    ax.bar(
        x - width / 2,
        success,
        width,
        color=method_colors,
        edgecolor="white",
        linewidth=0.4,
        yerr=np.vstack([success - success_low, success_high - success]),
        capsize=1.8,
        error_kw=dict(ecolor="#333333", lw=0.6, capthick=0.6),
    )
    ax.bar(
        x + width / 2,
        valid,
        width,
        color=[rgba(color, 0.38) for color in method_colors],
        edgecolor=method_colors,
        linewidth=0.55,
        hatch="///",
        yerr=np.vstack([valid - valid_low, valid_high - valid]),
        capsize=1.8,
        error_kw=dict(ecolor="#333333", lw=0.6, capthick=0.6),
    )
    ax.set_xticks(
        x,
        [LABELS[key] for key in ORDER],
        fontsize=7.5,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 56)
    ax.set_yticks(np.arange(0, 51, 10))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="none", label="Success"),
            Patch(
                facecolor=rgba("#777777", 0.38),
                edgecolor="#777777",
                hatch="///",
                label="Valid Cₘₜ",
            ),
        ],
        loc="upper right",
        ncol=2,
        fontsize=7.5,
        handlelength=1.2,
        columnspacing=0.8,
    )


def draw_panel_c(
    ax: plt.Axes,
    controller_summary: dict[str, dict[str, str]],
    source_rows: list[dict],
) -> None:
    ax.set_title(
        "c  Valid-transition Cₘₜ by joint\ncontroller-specific valid sets",
        loc="left",
        fontweight="bold",
    )
    x = np.arange(len(ORDER), dtype=float)
    hip = np.array([float(controller_summary[key]["mean_hip_cmt_valid"]) for key in ORDER])
    support = np.array(
        [float(controller_summary[key]["mean_support_knee_cmt_valid"]) for key in ORDER]
    )
    swing = np.array(
        [float(controller_summary[key]["mean_swing_knee_cmt_valid"]) for key in ORDER]
    )
    ax.bar(x, hip, width=0.66, color=JOINT_COLORS["Hip"], label="Hip")
    ax.bar(
        x,
        support,
        width=0.66,
        bottom=hip,
        color=JOINT_COLORS["Support knee"],
        label="Support knee",
    )
    ax.bar(
        x,
        swing,
        width=0.66,
        bottom=hip + support,
        color=JOINT_COLORS["Swing knee"],
        label="Swing knee",
    )
    totals = hip + support + swing
    for position, total, controller in zip(x, totals, ORDER):
        ax.text(position, total + 0.035, f"{total:.3f}", ha="center", va="bottom", fontsize=7.5)
        for component, value in (
            ("Hip", hip[int(position)]),
            ("Support knee", support[int(position)]),
            ("Swing knee", swing[int(position)]),
        ):
            source_rows.append(
                {
                    "panel": "c",
                    "record": "valid_cmt_component",
                    "method": controller,
                    "metric": "mean_cmt_valid",
                    "component": component,
                    "value": value,
                    "n": int(controller_summary[controller]["n_valid_cmt"]),
                    "description": "Controller-specific valid-transition subset.",
                }
            )
    ax.set_xticks(
        x,
        [LABELS[key] for key in ORDER],
        fontsize=7.5,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Mean Cₘₜ")
    ax.set_ylim(0, max(totals) * 1.30)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper right",
        ncol=3,
        fontsize=7.5,
        handlelength=1.0,
        columnspacing=0.7,
    )


def pair_lower_stat(
    rows: list[dict[str, str]],
    reference_prefix: str,
    comparison_prefix: str,
) -> dict:
    both_valid = [
        row
        for row in rows
        if as_bool(row[f"{reference_prefix}_valid"])
        and as_bool(row[f"{comparison_prefix}_valid"])
    ]
    lower = sum(
        float(row[f"{comparison_prefix}_cmt"])
        < float(row[f"{reference_prefix}_cmt"])
        for row in both_valid
    )
    ties = sum(
        math.isclose(
            float(row[f"{comparison_prefix}_cmt"]),
            float(row[f"{reference_prefix}_cmt"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in both_valid
    )
    interval = wilson(lower, len(both_valid))
    return {
        "both_valid": len(both_valid),
        "comparison_lower": lower,
        "ties": ties,
        "fraction": lower / len(both_valid),
        "ci": interval,
    }


def build_panel_d_stats(data_dir: Path, source_rows: list[dict]) -> list[dict]:
    active_reference = read_rows(data_dir / "active_reference_comparison_trials.csv")
    active_passive = read_rows(data_dir / "paired_trials.csv")
    mask_selector = read_rows(data_dir / "selector_vs_true_action_mask_trials.csv")
    specifications = [
        (
            "Active PPO → true mask",
            "active",
            "true_action_mask_scratch",
            [
                row
                for row in active_reference
                if row["comparison_controller"] == "true_action_mask_scratch"
            ],
            "reference",
            "comparison",
            COLORS["true_action_mask_scratch"],
        ),
        (
            "Active PPO → selector",
            "active",
            "passive_sign_selector",
            [
                row
                for row in active_passive
                if row["passive_controller"] == "passive_sign_selector"
            ],
            "active",
            "passive",
            COLORS["passive_sign_selector"],
        ),
        (
            "True mask → selector",
            "true_action_mask_scratch",
            "passive_sign_selector",
            mask_selector,
            "reference",
            "comparison",
            "#A44D78",
        ),
        (
            "Active PPO → oracle",
            "active",
            "passive_oracle_envelope",
            [
                row
                for row in active_passive
                if row["passive_controller"] == "passive_oracle_envelope"
            ],
            "active",
            "passive",
            COLORS["passive_oracle_envelope"],
        ),
    ]
    output: list[dict] = []
    for label, reference, comparison, rows, left_prefix, right_prefix, color in specifications:
        if len(rows) != 2160:
            raise ValueError(f"{label} has {len(rows)} rows, expected 2160")
        stats = pair_lower_stat(rows, left_prefix, right_prefix)
        entry = {
            "label": label,
            "reference": reference,
            "comparison": comparison,
            "color": color,
            **stats,
        }
        output.append(entry)
        source_rows.append(
            {
                "panel": "d",
                "record": "paired_lower_cmt_fraction",
                "reference": reference,
                "comparison": comparison,
                "metric": "fraction_comparison_lower_cmt",
                "value": stats["fraction"],
                "ci_low": stats["ci"][0],
                "ci_high": stats["ci"][1],
                "count": stats["comparison_lower"],
                "n": stats["both_valid"],
                "ties": stats["ties"],
                "interval": "Wilson 95% CI",
                "description": "Restricted to pairs with valid Cmt for both controllers.",
            }
        )
    return output


def draw_panel_d(ax: plt.Axes, comparisons: list[dict]) -> None:
    ax.set_title("d  Paired lower-Cₘₜ fraction", loc="left", fontweight="bold")
    y = np.arange(len(comparisons))[::-1]
    for position, item in zip(y, comparisons):
        value = 100.0 * item["fraction"]
        low = 100.0 * item["ci"][0]
        high = 100.0 * item["ci"][1]
        ax.errorbar(
            value,
            position,
            xerr=[[value - low], [high - value]],
            fmt="o",
            color=item["color"],
            ecolor=item["color"],
            markersize=4.0,
            capsize=2.5,
            elinewidth=1.0,
            capthick=0.8,
            zorder=3,
        )
        if value > 72:
            x_text, alignment = value - 1.8, "right"
        else:
            x_text, alignment = value + 1.8, "left"
        ax.text(
            x_text,
            position,
            f"{value:.1f}% ({item['comparison_lower']}/{item['both_valid']})",
            va="center",
            ha=alignment,
            fontsize=7.5,
        )
    ax.axvline(50.0, color="#777777", linestyle="--", linewidth=0.7, zorder=1)
    short_labels = [
        "Active → mask",
        "Active → selector",
        "Mask → selector",
        "Active → oracle",
    ]
    ax.set_yticks(y, short_labels, fontsize=7.5)
    ax.set_xlim(25, 100)
    ax.set_ylim(-0.72, 3.58)
    ax.set_xticks(np.arange(25, 101, 15))
    ax.set_xlabel("Pairs with lower Cₘₜ for comparison (%)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.99,
        0.025,
        "reference → comparison; Wilson 95% CI\nboth-valid pairs only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )


def write_source_data(path: Path, rows: list[dict]) -> None:
    fields = [
        "panel",
        "record",
        "method",
        "reference",
        "comparison",
        "metric",
        "component",
        "value",
        "ci_low",
        "ci_high",
        "count",
        "n",
        "ties",
        "interval",
        "description",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titlepad": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    source_rows: list[dict] = []
    rollout_rows = read_rows(data_dir / "rollouts_active_vs_passive.csv")
    summary_rows = read_rows(data_dir / "controller_summary.csv")
    controller_summary = {row["controller"]: row for row in summary_rows}
    rates = controller_rates(rollout_rows, source_rows)
    comparisons = build_panel_d_stats(data_dir, source_rows)

    fig, axes = plt.subplots(2, 2, figsize=(5.764, 8.2))
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.095, top=0.91, wspace=0.48, hspace=0.48)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    draw_panel_a(ax_a, source_rows)
    draw_panel_b(ax_b, rates)
    draw_panel_c(ax_c, controller_summary, source_rows)
    draw_panel_d(ax_d, comparisons)

    fig.suptitle(
        "Four-link comparison under the shared V22_3/V9 evaluator",
        fontsize=10.0,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.018,
        "Fixed-checkpoint result: the selector's low-Cₘₜ pattern was not reproduced\n"
        "by the scratch hard mask. This is not a strict one-factor causal test.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )

    source_path = output_dir / f"{args.basename}_source_data.csv"
    write_source_data(source_path, source_rows)
    output_paths = []
    for suffix in ("svg", "pdf", "png"):
        path = output_dir / f"{args.basename}.{suffix}"
        # Preserve the declared 5.764 x 8.2-inch publication size exactly.
        fig.savefig(path, dpi=600)
        if suffix == "svg":
            # Matplotlib emits trailing spaces in multi-line SVG path data.
            # Normalize them so the versioned, editable figure passes git's
            # whitespace audit while retaining identical vector geometry.
            svg_text = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                encoding="utf-8",
            )
        output_paths.append(path)
    plt.close(fig)

    for path in output_paths:
        print(path)
    print(source_path)


if __name__ == "__main__":
    main()
