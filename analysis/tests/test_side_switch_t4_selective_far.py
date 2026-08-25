from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_player_detector import PersonDetectionResult, PlayerDetection
from analysis.side_switch_t3_jersey_transport import EndpointSummary, TeamSummary
from analysis.side_switch_t4_selective_far import (
    T4_FEATURE_NAMES,
    far_crop_bounds,
    map_far_detection,
    selective_far_transport_features,
    summarize_selective_far_endpoint,
)


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
        if frame.shape[0] == 200:
            detection = _detection(hip_y=160.0, shoulder_y=115.0, height=80.0)
        else:
            detection = _detection(hip_y=30.0, shoulder_y=12.0, height=42.0)
        return PersonDetectionResult((detection,), 1, 1.0)


def _team(index: int) -> TeamSummary:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return TeamSummary(descriptor, 0.8, 0.9, 2)


def _endpoint(near: int, far: int) -> EndpointSummary:
    return EndpointSummary(
        near=_team(near),
        far=_team(far),
        selected_counts=(2, 2, 2, 2, 2),
        raw_candidate_counts=(2, 2, 2, 2, 2),
        descriptor_counts=(2, 2, 2, 2, 2),
        background_fallbacks=0,
        detector_inference_milliseconds=1.0,
    )


class SideSwitchT4SelectiveFarTest(unittest.TestCase):
    def test_far_crop_uses_frozen_canonical_band(self) -> None:
        self.assertEqual(far_crop_bounds(1000, 0.40), (80, 544))

    def test_far_detection_maps_vertical_geometry(self) -> None:
        source = _detection(hip_y=30.0, shoulder_y=12.0, height=42.0)
        mapped = map_far_detection(source, 16)
        self.assertEqual(mapped.y, source.y + 16)
        self.assertEqual(mapped.hip_y, 46.0)
        self.assertEqual(mapped.shoulder_y, 28.0)
        self.assertEqual(mapped.x, source.x)

    def test_endpoint_uses_full_near_and_selective_far_passes(self) -> None:
        frames = [np.full((200, 300, 3), (30, 120, 30), dtype=np.uint8)] * 5
        detector = _Detector()
        summary = summarize_selective_far_endpoint(frames, 0.40, detector)  # type: ignore[arg-type]
        self.assertEqual(detector.calls, 10)
        self.assertTrue(summary.near.available)
        self.assertTrue(summary.far.available)
        self.assertGreater(summary.near.reliability, 0.0)
        self.assertGreater(summary.far.reliability, 0.0)
        self.assertEqual(summary.far_crop_top, 16)
        self.assertEqual(summary.far_crop_bottom, 109)

    def test_feature_names_are_versioned_and_perfect_swap_is_positive(self) -> None:
        values, _ = selective_far_transport_features(
            _endpoint(1, 2),  # type: ignore[arg-type]
            _endpoint(2, 1),  # type: ignore[arg-type]
        )
        self.assertEqual(tuple(values), T4_FEATURE_NAMES)
        self.assertGreater(
            values["selectiveFarJerseyTeamTransportSwapMargin"], 0.5
        )
        self.assertGreater(values["selectiveFarJerseyReliableSwapEvidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
