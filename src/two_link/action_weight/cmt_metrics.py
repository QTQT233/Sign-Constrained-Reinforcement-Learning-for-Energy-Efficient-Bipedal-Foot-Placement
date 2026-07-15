"""Dimensionally explicit work metrics shared by the comparison scripts.

The continuous controller returns the physical actuator torque in N m.  The
torque limit is used only to clip that command; it must not be multiplied into
the command again when actuator power is evaluated.
"""

from __future__ import annotations

from typing import Union

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
