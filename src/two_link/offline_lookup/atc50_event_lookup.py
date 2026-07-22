"""Portable core of the hardware ATC-50 event-updated action lookup.

The generated 50^4 table is supplied separately from the public repository.
This module executes the state discretization, 1.8-degree event rule,
inter-event action hold, and uncovered-cell fallback used by deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ATC50EventLookup:
    action_table: np.ndarray
    event_angle_deg: float = 1.8
    stance_direction: int = -1
    uncovered_fallback: int = 0

    def __post_init__(self) -> None:
        self.action_table = np.asarray(self.action_table)
        if self.action_table.shape != (50, 50, 50, 50):
            raise ValueError(f"Expected a 50^4 ATC table, found {self.action_table.shape}")
        if not np.isin(self.action_table, (-2, -1, 0, 1)).all():
            raise ValueError("ATC table contains an unsupported action code")
        if self.uncovered_fallback not in (-1, 0, 1):
            raise ValueError("uncovered_fallback must be -1, 0, or 1")
        if self.stance_direction not in (-1, 1):
            raise ValueError("stance_direction must be -1 or +1")
        self._last_theta1: float | None = None
        self._last_theta2: float | None = None
        self._last_time_s: float | None = None
        self._held_action = int(self.uncovered_fallback)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        dataset: str = "working_save",
        **kwargs,
    ) -> "ATC50EventLookup":
        import h5py

        with h5py.File(Path(path), "r") as handle:
            table = handle[dataset][:]
        return cls(table, **kwargs)

    @staticmethod
    def axes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.linspace(np.deg2rad(60.0), np.deg2rad(120.0), 50),
            np.linspace(-3.5, 3.5, 50),
            np.linspace(np.deg2rad(-60.0), np.deg2rad(60.0), 50),
            np.linspace(-3.5, 3.5, 50),
        )

    def _query(self, theta1: float, dtheta1: float, theta2: float, dtheta2: float) -> int:
        state = (theta1, dtheta1, theta2, dtheta2)
        indices = tuple(int(np.abs(axis - value).argmin()) for axis, value in zip(self.axes(), state))
        code = int(self.action_table[indices])
        return int(self.uncovered_fallback if code == -2 else code)

    def reset(self, state: np.ndarray, time_s: float = 0.0) -> int:
        state = np.asarray(state, dtype=float)
        if state.shape != (4,):
            raise ValueError("state must contain [theta1, dtheta1, theta2, dtheta2]")
        theta1, dtheta1, theta2, dtheta2 = map(float, state)
        self._held_action = self._query(theta1, dtheta1, theta2, dtheta2)
        self._last_theta1 = theta1
        self._last_theta2 = theta2
        self._last_time_s = float(time_s)
        return self._held_action

    def update(self, state: np.ndarray, time_s: float) -> tuple[int, bool]:
        state = np.asarray(state, dtype=float)
        if state.shape != (4,):
            raise ValueError("state must contain [theta1, dtheta1, theta2, dtheta2]")
        theta1, dtheta1, theta2, dtheta2 = map(float, state)
        if self._last_theta1 is None:
            return self.reset(state, time_s), True

        threshold = np.deg2rad(self.event_angle_deg)
        directed_change = self.stance_direction * (theta1 - self._last_theta1)
        if directed_change < threshold:
            return self._held_action, False

        elapsed = float(time_s) - float(self._last_time_s)
        if elapsed > 0:
            dtheta2 = (theta2 - float(self._last_theta2)) / elapsed
        self._held_action = self._query(theta1, dtheta1, theta2, dtheta2)
        self._last_theta1 = theta1
        self._last_theta2 = theta2
        self._last_time_s = float(time_s)
        return self._held_action, True
