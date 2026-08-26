from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyObservation
from analysis.side_switch_t6_dominant_jersey import (
    CONSENSUS_RADIUS,
    T6_FEATURE_NAMES,
    dominant_jersey_transport_features,
    track_dominant_jersey,
)


def _observation(index: int, confidence: float = 0.8) -> JerseyObservation:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyObservation(descriptor, 0.5, 0.7, 0.1, confidence, 0.64, False)


class SideSwitchT6DominantJerseyTest(unittest.TestCase):
    def test_radius_is_frozen(self) -> None:
        self.assertEqual(CONSENSUS_RADIUS, 0.38)

    def test_dominant_mode_rejects_single_frame_outlier(self) -> None:
        track = track_dominant_jersey(
            (
                (_observation(1), _observation(9)),
                (_observation(1),),
                (_observation(1),),
                (),
                (),
            )
        )
        self.assertTrue(track.team.available)
        self.assertEqual(track.mode_observations, 3)
        self.assertEqual(track.rejected_observations, 1)
        self.assertGreater(track.team.descriptor[1], 0.99)  # type: ignore[index]
        self.assertLess(track.team.descriptor[9], 0.01)  # type: ignore[index]

    def test_mode_still_requires_two_frames(self) -> None:
        track = track_dominant_jersey(
            ((_observation(1), _observation(1)), (), (), (), ())
        )
        self.assertFalse(track.team.available)

    def test_feature_names_and_swap_semantics(self) -> None:
        a = track_dominant_jersey(tuple((_observation(1),) for _ in range(5)))
        b = track_dominant_jersey(tuple((_observation(2),) for _ in range(5)))

        class _Endpoint:
            def __init__(self, near, far):
                self.near = near.team
                self.far = far.team

        values, _ = dominant_jersey_transport_features(
            _Endpoint(a, b),  # type: ignore[arg-type]
            _Endpoint(b, a),  # type: ignore[arg-type]
        )
        self.assertEqual(tuple(values), T6_FEATURE_NAMES)
        self.assertGreater(values["dominantJerseyTeamTransportSwapMargin"], 0.5)
        self.assertGreater(values["dominantJerseyReliableSwapEvidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
