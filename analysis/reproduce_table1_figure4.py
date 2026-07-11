"""Recompute Table I and Figure 4 data from the archived 30^4 maps.

The unsupported continuous-PPO row is deliberately not synthesized: no
matching 30^4 continuous map was present in the audited artifact set.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import h5py
import numpy as np


ACTIVE = "working_save(-1,0,1)-30,30_from_2-2"
NEGATIVE = "working_save(-1,0)-30,30"
POSITIVE = "working_save(0,1)-30,30"


def load(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        array = handle["working_save"][:]
    if array.shape != (30, 30, 30, 30):
        raise ValueError(f"Unexpected shape for {path}: {array.shape}")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    active = load(args.data / ACTIVE)
    negative = load(args.data / NEGATIVE)
    positive = load(args.data / POSITIVE)

    # Preserve the ordered if/elif semantics of Action_ratio.py.
    constrained = np.select(
        [
            (negative == -1) & (positive != 0),
            (negative != 0) & (positive == 1),
            (negative == 0) | (positive == 0),
            (negative == -2) | (positive == -2),
        ],
        [-1, 1, 0, -2],
        default=np.nan,
    )
    if np.isnan(constrained).any():
        raise ValueError("Unclassified map entries")

    total = active.size
    active_feasible = int(np.sum(active != -2))
    constrained_feasible = int(np.sum(constrained != -2))
    common = int(np.sum((active != -2) & (constrained != -2)))
    table_rows = [
        ["Active PPO", total, active_feasible, active_feasible / total,
         int(np.sum(active == -2)), np.mean(active == -2)],
        ["Sign-constrained union", total, constrained_feasible, constrained_feasible / total,
         int(np.sum(constrained == -2)), np.mean(constrained == -2)],
        ["Common feasible intersection", total, common, common / total,
         total - common, 1.0 - common / total],
    ]
    with (args.output / "table_i.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["controller_or_set", "sampled_states", "feasible_states", "feasible_ratio",
                         "outside_set_states", "outside_set_ratio"])
        writer.writerows(table_rows)

    labels = ["non-completed", "negative", "zero", "positive"]
    codes = [-2, -1, 0, 1]
    active_pct = np.array([np.mean(active == code) for code in codes]) * 100
    constrained_pct = np.array([np.mean(constrained == code) for code in codes]) * 100
    with (args.output / "figure4_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "active_percent", "sign_constrained_percent"])
        writer.writerows(zip(labels, active_pct, constrained_pct))

    # Dependency-light SVG rendering keeps the result reproducible even on
    # machines whose binary Matplotlib stack differs from the training host.
    width, height = 760, 410
    left, right, top, bottom = 76, 24, 44, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = 70.0
    group_w = plot_w / len(labels)
    bar_w = 48
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#222">',
    ]
    for tick in range(0, 71, 10):
        y = top + plot_h * (1 - tick / ymax)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#222"/>')
    for index, label in enumerate(labels):
        center = left + group_w * (index + 0.5)
        for offset, value, color in ((-bar_w/2, active_pct[index], "#3B6FB6"),
                                     (bar_w/2, constrained_pct[index], "#D97832")):
            x = center + offset - bar_w / 2
            h = plot_h * value / ymax
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w/2:.2f}" y="{y-6:.2f}" text-anchor="middle" font-size="11">{value:.1f}</text>')
        parts.append(f'<text x="{center:.2f}" y="{top+plot_h+25}" text-anchor="middle">{html.escape(label)}</text>')
    parts.extend([
        f'<text x="20" y="{top+plot_h/2:.2f}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2:.2f})">Share of sampled states (%)</text>',
        '<rect x="430" y="12" width="14" height="14" fill="#3B6FB6"/><text x="450" y="24">Active PPO</text>',
        '<rect x="555" y="12" width="14" height="14" fill="#D97832"/><text x="575" y="24">Sign-constrained union</text>',
        '</g></svg>',
    ])
    (args.output / "figure4.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
