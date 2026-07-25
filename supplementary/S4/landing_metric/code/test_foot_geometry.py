from __future__ import annotations

import math
import unittest

from foot_geometry import foot_x, foot_y, signed_landing_residual


class FootGeometryTests(unittest.TestCase):
    def test_unequal_link_lengths_are_used(self) -> None:
        q1 = math.radians(60.0)
        q2 = math.radians(30.0)
        value = foot_x(q1, q2, 0.521, 0.481)
        self.assertAlmostEqual(value, 0.501, places=12)
        legacy = 0.521 * (math.cos(q1) + math.sin(q2))
        self.assertNotAlmostEqual(value, legacy, places=6)

    def test_target_state_has_zero_residual(self) -> None:
        q1 = math.radians(61.8)
        q2 = math.radians(31.7)
        self.assertAlmostEqual(
            signed_landing_residual(q1, q2, q1, q2, 0.521, 0.481),
            0.0,
            places=14,
        )

    def test_vertical_geometry_matches_manuscript(self) -> None:
        value = foot_y(
            math.radians(60.0),
            math.radians(30.0),
            0.521,
            0.481,
        )
        self.assertAlmostEqual(value, 0.0346410161513775, places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
