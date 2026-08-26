from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyObservation
from analysis.side_switch_t7_soft_consensus import robust_weight, track_soft_consensus


def _observation(index: int) -> JerseyObservation:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyObservation(descriptor, 0.5, 0.7, 0.1, 0.8, 0.64, False)


class SideSwitchT7SoftConsensusTest(unittest.TestCase):
    def test_weight_is_continuous_and_frozen_at_half_radius(self) -> None:
        self.assertAlmostEqual(robust_weight(0.0), 1.0)
        self.assertAlmostEqual(robust_weight(0.38), 0.5)
        self.assertGreater(robust_weight(1.0), 0.0)

    def test_outlier_is_downweighted_but_not_discarded(self) -> None:
        track = track_soft_consensus(
            ((_observation(1), _observation(9)), (_observation(1),), (), (), ())
        )
        self.assertTrue(track.team.available)
        self.assertEqual(track.total_observations, 3)
        self.assertEqual(len(track.robust_weights), 3)
        self.assertTrue(all(value > 0.0 for value in track.robust_weights))
        self.assertGreater(track.team.descriptor[9], 0.0)  # type: ignore[index]
        self.assertLess(track.team.descriptor[9], track.team.descriptor[1])  # type: ignore[index]

    def test_availability_matches_unfiltered_two_frame_rule(self) -> None:
        track = track_soft_consensus(
            ((_observation(1),), (_observation(9),), (), (), ())
        )
        self.assertTrue(track.team.available)
        one_frame = track_soft_consensus(((_observation(1),), (), (), (), ()))
        self.assertFalse(one_frame.team.available)


if __name__ == "__main__":
    unittest.main()
