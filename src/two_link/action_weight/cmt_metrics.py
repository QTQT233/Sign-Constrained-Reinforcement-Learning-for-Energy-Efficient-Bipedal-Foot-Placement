"""Dimensionally explicit work metrics shared by the comparison scripts.

The continuous controller returns the physical actuator torque in N m.  The
torque limit is used only to clip that command; it must not be multiplied into
the command again when actuator power is evaluated.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


ArrayLike = Union[float, np.ndarray]


def positive_actuator_work_increment(
    applied_torque_nm: ArrayLike,
    relative_speed_rad_s: ArrayLike,
    dt_s: float,
) -> ArrayLike:
    """Return the positive actuator-work increment in joules.

    ``applied_torque_nm`` is the physical torque sent to the dynamics, not a
    normalized action.  Since radians are dimensionless, torque [N m] times
    relative angular speed [rad/s] is power [W], and multiplication by ``dt_s``
    gives work [J].  Negative or zero power contributes zero.
    """

    if dt_s < 0:
        raise ValueError("dt_s must be non-negative")

    power_w = np.asarray(applied_torque_nm) * np.asarray(relative_speed_rad_s)
    work_j = np.where(power_w > 0, power_w * dt_s, 0.0)
    if work_j.ndim == 0:
        return float(work_j)
    return work_j


def select_successful_expert_by_cmt(
    negative_status: ArrayLike,
    negative_cmt: ArrayLike,
    positive_status: ArrayLike,
    positive_cmt: ArrayLike,
    *,
    tie_break: str = "positive",
) -> Tuple[ArrayLike, ArrayLike]:
    """Fuse two completed expert rollouts without confusing action 0 with failure.

    A rollout is successful exactly when its stored status is not ``-2``.  The
    successful status may be ``-1``, ``0``, or ``+1`` because the legacy
    evaluator stored the *first action*, rather than a Boolean success flag.
    The returned selector is an unambiguous expert ID: ``-1`` for the
    negative-action expert, ``+1`` for the positive-action expert, and ``-2``
    only when both experts failed.  If both succeeded, the lower valid Cmt is
    selected.  ``tie_break`` makes the otherwise arbitrary exact-tie rule
    explicit and must be either ``"negative"`` or ``"positive"``.

    Failed-expert Cmt cells are deliberately ignored.  In the retained archive
    they were initialized to zero, so taking a numerical minimum before
    applying the success gate creates spuriously zero fused Cmt values.
    """

    if tie_break not in {"negative", "positive"}:
        raise ValueError("tie_break must be 'negative' or 'positive'")

    neg_status, neg_cmt, pos_status, pos_cmt = np.broadcast_arrays(
        np.asarray(negative_status),
        np.asarray(negative_cmt, dtype=float),
        np.asarray(positive_status),
        np.asarray(positive_cmt, dtype=float),
    )
    neg_success = neg_status != -2
    pos_success = pos_status != -2

    invalid_neg = neg_success & (~np.isfinite(neg_cmt) | (neg_cmt < 0))
    invalid_pos = pos_success & (~np.isfinite(pos_cmt) | (pos_cmt < 0))
    if np.any(invalid_neg) or np.any(invalid_pos):
        raise ValueError("every successful rollout must have a finite, non-negative Cmt")

    selector = np.full(neg_status.shape, -2, dtype=np.int8)
    selected_cmt = np.full(neg_status.shape, -2.0, dtype=float)

    neg_only = neg_success & ~pos_success
    pos_only = ~neg_success & pos_success
    both = neg_success & pos_success

    selector[neg_only] = -1
    selected_cmt[neg_only] = neg_cmt[neg_only]
    selector[pos_only] = 1
    selected_cmt[pos_only] = pos_cmt[pos_only]

    choose_neg = both & (neg_cmt < pos_cmt)
    if tie_break == "negative":
        choose_neg |= both & (neg_cmt == pos_cmt)
    choose_pos = both & ~choose_neg
    selector[choose_neg] = -1
    selected_cmt[choose_neg] = neg_cmt[choose_neg]
    selector[choose_pos] = 1
    selected_cmt[choose_pos] = pos_cmt[choose_pos]

    if selector.ndim == 0:
        return int(selector), float(selected_cmt)
    return selector, selected_cmt
