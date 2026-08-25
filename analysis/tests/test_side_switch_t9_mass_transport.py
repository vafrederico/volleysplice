from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyObservation
from analysis.side_switch_t8_two_mode_transport import mode_set_cost, track_two_modes
from analysis.side_switch_t9_mass_transport import mass_transport_cost


def _observation(index: int, confidence: float = 0.8) -> JerseyObservation:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyObservation(descriptor, 0.5, 0.7, 0.1, confidence, 0.64, False)


class SideSwitchT9MassTransportTest(unittest.TestCase):
    def test_identical_mode_sets_have_zero_cost(self) -> None:
        left = track_two_modes(((_observation(1),), (_observation(9),), (), (), ()))
        right = track_two_modes(((_observation(9),), (_observation(1),), (), (), ()))
        self.assertAlmostEqual(mass_transport_cost(left, right), 0.0)

    def test_mass_preservation_penalizes_unmatched_support(self) -> None:
        left = track_two_modes(
            ((_observation(1, 0.9),), (_observation(9, 0.1),), (), (), ())
        )
        right = track_two_modes(
            ((_observation(1, 0.1),), (_observation(9, 0.9),), (), (), ())
        )
        self.assertGreater(mass_transport_cost(left, right), mode_set_cost(left, right))

    def test_unavailable_cost_remains_one(self) -> None:
        left = track_two_modes(((_observation(1),), (), (), (), ()))
        right = track_two_modes(((_observation(1),), (_observation(1),), (), (), ()))
        self.assertEqual(mass_transport_cost(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
