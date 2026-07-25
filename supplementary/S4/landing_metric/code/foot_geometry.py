"""Shared two-link foot geometry and landing-error definitions.

This local module corrects the legacy implementation that multiplied both
horizontal terms by ``l1``.  The model used by the manuscript has unequal link
lengths, so the swing-foot position is

    x = l1*cos(q1) + l2*sin(q2)
    y = l1*sin(q1) - l2*cos(q2)

The functions are deliberately independent of the controller implementation so
the reporting correction cannot alter actions, termination, energy, COM
displacement, or Cmt.
"""

from __future__ import annotations

import math


def foot_x(q1_rad: float, q2_rad: float, l1_m: float, l2_m: float) -> float:
    """Return the horizontal swing-foot coordinate in metres."""

    return float(l1_m * math.cos(q1_rad) + l2_m * math.sin(q2_rad))


def foot_y(q1_rad: float, q2_rad: float, l1_m: float, l2_m: float) -> float:
    """Return the vertical swing-foot coordinate in metres."""

    return float(l1_m * math.sin(q1_rad) - l2_m * math.cos(q2_rad))


def signed_landing_residual(
    actual_q1_rad: float,
    actual_q2_rad: float,
    target_q1_rad: float,
    target_q2_rad: float,
    l1_m: float,
    l2_m: float,
    direction_sign: float = 1.0,
) -> float:
    """Return signed actual-minus-target horizontal placement residual."""

    actual = foot_x(actual_q1_rad, actual_q2_rad, l1_m, l2_m)
    target = foot_x(target_q1_rad, target_q2_rad, l1_m, l2_m)
    return float(direction_sign * (actual - target))


__all__ = ["foot_x", "foot_y", "signed_landing_residual"]
