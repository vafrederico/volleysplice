from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_player_detector import PlayerDetection
from analysis.side_switch_t3_jersey_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    EndpointSummary,
    JerseyObservation,
    T3_FEATURE_NAMES,
    TeamSummary,
    build_tracklets,
    endpoint_sample_times,
    jersey_descriptor,
    summarize_team,
    team_transport_features,
)


def _descriptor(index: int) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[index] = 1.0
    return value


def _observation(index: int, x: float = 0.5) -> JerseyObservation:
    return JerseyObservation(
        descriptor=_descriptor(index),
        x=x,
        y=0.5,
        scale=0.25,
        confidence=0.9,
        support=0.8,
        background_fallback=False,
    )


def _team(index: int | None) -> TeamSummary:
    return TeamSummary(
        descriptor=None if index is None else _descriptor(index),
        reliability=0.0 if index is None else 0.8,
        cohesion=0.0 if index is None else 0.9,
        qualifying_tracklets=0 if index is None else 2,
    )


def _endpoint(near: int | None, far: int | None) -> EndpointSummary:
    return EndpointSummary(
        near=_team(near),
        far=_team(far),
        selected_counts=(2, 2, 2, 2, 2),
        raw_candidate_counts=(2, 2, 2, 2, 2),
        descriptor_counts=(2, 2, 2, 2, 2),
        background_fallbacks=0,
        detector_inference_milliseconds=1.0,
    )


class SideSwitchT3JerseyTransportTest(unittest.TestCase):
    def test_endpoint_sampling_uses_five_frozen_fractions(self) -> None:
        self.assertEqual(
            endpoint_sample_times(10.0, 20.0),
            tuple(10.0 + 10.0 * value for value in ENDPOINT_FRAME_FRACTIONS),
        )

    def test_jersey_descriptor_is_finite_normalized_and_64_values(self) -> None:
        frame = np.full((160, 240, 3), (30, 120, 30), dtype=np.uint8)
        frame[40:105, 85:155] = (210, 40, 40)
        detection = PlayerDetection(
            x=70.0,
            y=20.0,
            width=100.0,
            height=125.0,
            hip_x=120.0,
            hip_y=108.0,
            shoulder_x=120.0,
            shoulder_y=43.0,
            score=0.9,
        )
        descriptor, support, _ = jersey_descriptor(frame, detection)
        self.assertEqual(descriptor.shape, (64,))
        self.assertTrue(np.isfinite(descriptor).all())
        self.assertAlmostEqual(float(np.sum(descriptor)), 1.0)
        self.assertGreater(support, 0.1)

    def test_tracklets_require_two_frames(self) -> None:
        frames = [[_observation(2)], [], [], [], []]
        self.assertEqual(build_tracklets(frames), ())
        frames[1] = [_observation(2, 0.51)]
        tracks = build_tracklets(frames)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].observed_frames, 2)

    def test_team_summary_pools_qualified_tracklets(self) -> None:
        frames = [[_observation(3)] for _ in ENDPOINT_FRAME_FRACTIONS]
        summary = summarize_team(build_tracklets(frames))
        self.assertTrue(summary.available)
        self.assertEqual(summary.qualifying_tracklets, 1)
        self.assertGreater(summary.reliability, 0.0)
        self.assertAlmostEqual(summary.cohesion, 1.0)

    def test_perfect_swap_activates_reliable_swap_evidence(self) -> None:
        values, diagnostics = team_transport_features(
            _endpoint(1, 2), _endpoint(2, 1)
        )
        self.assertEqual(tuple(values), T3_FEATURE_NAMES)
        self.assertGreater(values["jerseyTeamTransportSwapMargin"], 0.5)
        self.assertGreater(values["jerseyReliableSwapEvidence"], 0.0)
        self.assertEqual(values["jerseyReliableContinuityEvidence"], 0.0)
        self.assertLess(
            diagnostics["swappedAssignmentCost"],
            diagnostics["sameAssignmentCost"],
        )

    def test_same_side_retention_activates_continuity_evidence(self) -> None:
        values, _ = team_transport_features(_endpoint(1, 2), _endpoint(1, 2))
        self.assertLess(values["jerseyTeamTransportSwapMargin"], -0.5)
        self.assertEqual(values["jerseyReliableSwapEvidence"], 0.0)
        self.assertGreater(values["jerseyReliableContinuityEvidence"], 0.0)

    def test_missing_team_has_zero_reliability_gate(self) -> None:
        values, _ = team_transport_features(_endpoint(1, None), _endpoint(None, 1))
        self.assertEqual(values["jerseyBaseReliabilityGate"], 0.0)
        self.assertEqual(values["jerseySwapReliabilityGate"], 0.0)
        self.assertEqual(values["jerseyContinuityReliabilityGate"], 0.0)
        self.assertEqual(values["jerseyReliableSwapEvidence"], 0.0)
        self.assertEqual(values["jerseyReliableContinuityEvidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
