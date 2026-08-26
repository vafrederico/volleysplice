from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_player_detector import PersonDetectionResult, PlayerDetection
from analysis.side_switch_t3_jersey_transport import JerseyObservation
from analysis.side_switch_t5_court_tracking import (
    T5_FEATURE_NAMES,
    court_tracked_transport_features,
    summarize_court_tracked_endpoint,
    track_court_team,
)


def _observation(index: int, *, confidence: float = 0.8) -> JerseyObservation:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyObservation(descriptor, 0.5, 0.7, 0.1, confidence, 0.64, False)


def _detection(*, hip_y: float, shoulder_y: float, height: float) -> PlayerDetection:
    return PlayerDetection(
        x=90.0,
        y=max(0.0, shoulder_y - 10.0),
        width=50.0,
        height=height,
        hip_x=115.0,
        hip_y=hip_y,
        shoulder_x=115.0,
        shoulder_y=shoulder_y,
        score=0.9,
    )


class _Detector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> PersonDetectionResult:
        self.calls += 1
        detection = (
            _detection(hip_y=160.0, shoulder_y=115.0, height=80.0)
            if frame.shape[0] == 200
            else _detection(hip_y=30.0, shoulder_y=12.0, height=42.0)
        )
        return PersonDetectionResult((detection,), 1, 1.0)


class SideSwitchT5CourtTrackingTest(unittest.TestCase):
    def test_requires_two_distinct_team_frames(self) -> None:
        track = track_court_team(((_observation(1),), (), (), (), ()))
        self.assertFalse(track.team.available)
        self.assertEqual(track.observed_frames, 1)

    def test_aggregates_different_players_as_one_court_team(self) -> None:
        track = track_court_team(
            (
                (_observation(1),),
                (_observation(2),),
                (),
                (_observation(1), _observation(2)),
                (),
            )
        )
        self.assertTrue(track.team.available)
        self.assertEqual(track.observed_frames, 3)
        self.assertEqual(track.total_observations, 4)
        self.assertGreater(track.team.reliability, 0.0)
        self.assertGreater(track.team.descriptor[1], 0.0)  # type: ignore[index]
        self.assertGreater(track.team.descriptor[2], 0.0)  # type: ignore[index]

    def test_endpoint_preserves_t4_detector_ownership(self) -> None:
        frames = [np.full((200, 300, 3), (30, 120, 30), dtype=np.uint8)] * 5
        detector = _Detector()
        summary = summarize_court_tracked_endpoint(frames, 0.40, detector)  # type: ignore[arg-type]
        self.assertEqual(detector.calls, 10)
        self.assertTrue(summary.near.available)
        self.assertTrue(summary.far.available)
        self.assertEqual(summary.near_track.observed_frames, 5)
        self.assertEqual(summary.far_track.observed_frames, 5)

    def test_feature_names_are_versioned_and_perfect_swap_is_positive(self) -> None:
        frames_a = tuple((_observation(1),) for _ in range(5))
        frames_b = tuple((_observation(2),) for _ in range(5))
        near_a = track_court_team(frames_a)
        far_a = track_court_team(frames_b)

        class _Endpoint:
            def __init__(self, near, far):
                self.near = near.team
                self.far = far.team

        values, _ = court_tracked_transport_features(
            _Endpoint(near_a, far_a),  # type: ignore[arg-type]
            _Endpoint(far_a, near_a),  # type: ignore[arg-type]
        )
        self.assertEqual(tuple(values), T5_FEATURE_NAMES)
        self.assertGreater(
            values["courtTrackedFarJerseyTeamTransportSwapMargin"], 0.5
        )
        self.assertGreater(
            values["courtTrackedFarJerseyReliableSwapEvidence"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
