from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_player_detector import PlayerDetection
from analysis.side_switch_t1_transport import (
    AppearanceObservation,
    AppearanceTracklet,
    EndpointSummary,
    T1_FEATURE_NAMES,
    appearance_descriptor,
    build_tracklets,
    endpoint_sample_times,
    transport_features,
    transport_reduction,
)


def _descriptor(index: int) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[index] = 1.0
    return value


def _track(index: int, reliability: float = 1.0) -> AppearanceTracklet:
    return AppearanceTracklet(
        descriptor=_descriptor(index),
        x=0.5,
        y=0.5,
        reliability=reliability,
        observed_frames=3,
    )


def _endpoint(near: int | None, far: int | None) -> EndpointSummary:
    return EndpointSummary(
        near=() if near is None else (_track(near),),
        far=() if far is None else (_track(far),),
        selected_counts=(2, 2, 2),
        raw_candidate_counts=(2, 2, 2),
        detector_inference_milliseconds=1.0,
    )


class SideSwitchT1TransportTest(unittest.TestCase):
    def test_endpoint_sampling_uses_frozen_fractions(self) -> None:
        self.assertEqual(endpoint_sample_times(10.0, 20.0), (11.5, 15.0, 18.5))

    def test_descriptor_is_finite_normalized_and_64_values(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame[20:80, 50:100] = (20, 90, 220)
        detection = PlayerDetection(
            x=50.0,
            y=20.0,
            width=50.0,
            height=60.0,
            hip_x=75.0,
            hip_y=75.0,
            shoulder_x=75.0,
            shoulder_y=30.0,
            score=0.9,
        )
        value = appearance_descriptor(frame, detection)
        self.assertEqual(value.shape, (64,))
        self.assertTrue(np.isfinite(value).all())
        self.assertAlmostEqual(float(np.sum(value)), 1.0)

    def test_tracklets_link_consistent_observations(self) -> None:
        frames = [
            [
                AppearanceObservation(
                    descriptor=_descriptor(3),
                    x=0.2 + index * 0.01,
                    y=0.7,
                    confidence=0.9,
                )
            ]
            for index in range(3)
        ]
        tracks = build_tracklets(frames)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].observed_frames, 3)
        self.assertAlmostEqual(tracks[0].reliability, 0.9)

    def test_perfect_swap_has_positive_margin_and_complete_transport(self) -> None:
        values, diagnostics = transport_features(_endpoint(1, 2), _endpoint(2, 1))
        self.assertEqual(tuple(values), T1_FEATURE_NAMES)
        self.assertGreater(values["appearanceTransportSwapMargin"], 0.5)
        self.assertEqual(values["bidirectionalMatchedIdentityMinimum"], 1.0)
        self.assertEqual(values["transportCoverageMinimum"], 1.0)
        self.assertLess(
            diagnostics["swappedAssignmentCost"], diagnostics["sameAssignmentCost"]
        )

    def test_same_side_retention_has_negative_swap_margin(self) -> None:
        values, _ = transport_features(_endpoint(1, 2), _endpoint(1, 2))
        self.assertLess(values["appearanceTransportSwapMargin"], -0.5)
        self.assertEqual(values["sameSideIdentityRetentionPenalty"], 1.0)

    def test_missing_side_has_zero_transport_coverage(self) -> None:
        reduction = transport_reduction((_track(1),), ())
        self.assertEqual(reduction.cost, 1.0)
        self.assertEqual(reduction.matched_identity_mass, 0.0)
        self.assertEqual(reduction.coverage, 0.0)
        values, _ = transport_features(_endpoint(1, None), _endpoint(None, 1))
        self.assertEqual(values["transportCoverageMinimum"], 0.0)


if __name__ == "__main__":
    unittest.main()
