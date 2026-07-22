from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoLinkRoutingLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.routing = load_module(
            "released_transition_locked_router",
            "src/two_link/offline_routing/transition_locked_router.py",
        )
        cls.lookup = load_module(
            "released_atc50_event_lookup",
            "src/two_link/offline_lookup/atc50_event_lookup.py",
        )

    def test_transition_router_selects_once_from_nearest_grid_cell(self) -> None:
        table = np.zeros((10, 30, 10, 30), dtype=np.int8)
        axes = self.routing.TransitionLockedRouter.axes()
        table[2, 3, 4, 5] = -1
        state = np.array([axes[0][2], axes[1][3], axes[2][4], axes[3][5]])
        router = self.routing.TransitionLockedRouter(table)
        self.assertEqual(router.select(state), -1)
        table[2, 3, 4, 5] = 0
        self.assertEqual(self.routing.TransitionLockedRouter(table).select(state), 1)
        table[2, 3, 4, 5] = 1
        self.assertEqual(self.routing.TransitionLockedRouter(table).select(state), 1)
        table[2, 3, 4, 5] = -2
        with self.assertRaises(LookupError):
            self.routing.TransitionLockedRouter(table).select(state)

    def test_atc50_holds_action_until_directed_1p8_degree_event(self) -> None:
        table = np.zeros((50, 50, 50, 50), dtype=np.int8)
        axes = self.lookup.ATC50EventLookup.axes()
        start = np.array([axes[0][25], axes[1][25], axes[2][25], axes[3][25]])
        table[25, 25, 25, 25] = -1
        table[23, 25, 25, :] = 1
        lookup = self.lookup.ATC50EventLookup(table, stance_direction=-1)
        self.assertEqual(lookup.reset(start, 0.0), -1)

        below_threshold = start.copy()
        below_threshold[0] -= np.deg2rad(1.0)
        self.assertEqual(lookup.update(below_threshold, 0.1), (-1, False))

        event_state = start.copy()
        event_state[0] -= np.deg2rad(2.0)
        self.assertEqual(lookup.update(event_state, 0.2), (1, True))

    def test_atc50_uncovered_cell_uses_declared_fallback(self) -> None:
        table = np.full((50, 50, 50, 50), -2, dtype=np.int8)
        state = np.array([np.deg2rad(90.0), 0.0, 0.0, 0.0])
        lookup = self.lookup.ATC50EventLookup(table, uncovered_fallback=0)
        self.assertEqual(lookup.reset(state), 0)


if __name__ == "__main__":
    unittest.main()
