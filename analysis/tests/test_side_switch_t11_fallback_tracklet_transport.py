from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyObservation
from analysis.side_switch_t11_fallback_tracklet_transport import (
    FallbackAppearanceUnit,
    side_units,
)


def _observation(index: int) -> JerseyObservation:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyObservation(descriptor, 0.5, 0.7, 0.1, 0.8, 0.64, False)


class SideSwitchT11FallbackTrackletTransportTest(unittest.TestCase):
    def test_stable_tracklet_suppresses_pooled_fallback(self) -> None:
        units, stable, pool = side_units(
            ((_observation(1),), (_observation(1),), (), (), ())
        )
        self.assertTrue(pool.team.available)
        self.assertEqual(len(stable), 1)
        self.assertEqual(units, stable)
        self.assertNotIsInstance(units[0], FallbackAppearanceUnit)

    def test_unlinked_multiframe_observations_create_one_typed_fallback(self) -> None:
        units, stable, pool = side_units(
            ((_observation(1),), (_observation(9),), (), (), ())
        )
        self.assertTrue(pool.team.available)
        self.assertFalse(stable)
        self.assertEqual(len(units), 1)
        self.assertIsInstance(units[0], FallbackAppearanceUnit)
        self.assertEqual(units[0].kind, "pooled-fallback")

    def test_one_observed_frame_remains_unavailable(self) -> None:
        units, stable, pool = side_units(((_observation(1),), (), (), (), ()))
        self.assertFalse(pool.team.available)
        self.assertFalse(stable)
        self.assertFalse(units)


if __name__ == "__main__":
    unittest.main()
