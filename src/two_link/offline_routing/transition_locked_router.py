"""Portable transition-start router for the released two-link coarse map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TransitionLockedRouter:
    """Select one expert for a covered initial state and hold it throughout."""

    route_table: np.ndarray

    def __post_init__(self) -> None:
        table = np.asarray(self.route_table)
        if table.shape != (10, 30, 10, 30):
            raise ValueError(f"Expected a 10x30x10x30 routing table, found {table.shape}")
        if not np.isin(table, (-2, -1, 0, 1)).all():
            raise ValueError("Routing table contains an unsupported action code")
        object.__setattr__(self, "route_table", table)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        dataset: str = "working_save_passive",
    ) -> "TransitionLockedRouter":
        import h5py

        with h5py.File(Path(path), "r") as handle:
            table = handle[dataset][:]
        return cls(table)

    @staticmethod
    def axes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.linspace(np.deg2rad(60.0), np.deg2rad(120.0), 10),
            np.linspace(-3.5, 3.5, 30),
            np.linspace(np.deg2rad(-60.0), np.deg2rad(60.0), 10),
            np.linspace(-3.5, 3.5, 30),
        )

    def select(self, initial_state: np.ndarray) -> int:
        state = np.asarray(initial_state, dtype=float)
        if state.shape != (4,):
            raise ValueError("initial_state must contain [theta1, dtheta1, theta2, dtheta2]")
        indices = tuple(int(np.abs(axis - value).argmin()) for axis, value in zip(self.axes(), state))
        code = int(self.route_table[indices])
        if code == -2:
            raise LookupError("Initial state is outside the released routing coverage")
        # The released evaluator assigns a zero/tie route to the positive
        # expert; only -1 selects the negative expert.
        return -1 if code == -1 else 1


def repository_default() -> TransitionLockedRouter:
    root = Path(__file__).resolve().parents[3]
    path = root / "data/two_link/offline_routing/working_save_passive-10-30"
    return TransitionLockedRouter.from_hdf5(path)
