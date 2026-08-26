from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyObservation
from analysis.side_switch_t8_two_mode_transport import mode_set_cost, track_two_modes


def _observation(index: int, confidence: float = 0.8) -> JerseyObservation:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyObservation(descriptor, 0.5, 0.7, 0.1, confidence, 0.64, False)


class SideSwitchT8TwoModeTransportTest(unittest.TestCase):
    def test_two_distinct_modes_retain_every_observation(self) -> None:
        track = track_two_modes(
            ((_observation(1), _observation(9)), (_observation(1), _observation(9)), (), (), ())
        )
        self.assertTrue(track.available)
        self.assertEqual(len(track.modes), 2)
        self.assertEqual(sum(value.observations for value in track.modes), 4)
        self.assertAlmostEqual(sum(value.support for value in track.modes), 1.0)
        self.assertEqual(sorted(value.observed_frames for value in track.modes), [2, 2])

    def test_mode_set_cost_is_symmetric_and_identity_is_zero(self) -> None:
        left = track_two_modes(((_observation(1),), (_observation(9),), (), (), ()))
        right = track_two_modes(((_observation(9),), (_observation(1),), (), (), ()))
        self.assertAlmostEqual(mode_set_cost(left, right), 0.0)
        self.assertAlmostEqual(mode_set_cost(left, right), mode_set_cost(right, left))

    def test_availability_matches_unfiltered_two_frame_rule(self) -> None:
        available = track_two_modes(((_observation(1),), (_observation(9),), (), (), ()))
        self.assertTrue(available.available)
        one_frame = track_two_modes(((_observation(1), _observation(9)), (), (), (), ()))
        self.assertFalse(one_frame.available)


if __name__ == "__main__":
    unittest.main()
