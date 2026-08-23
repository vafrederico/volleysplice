from __future__ import annotations

import unittest

import cv2
import numpy as np

from analysis.side_switch_player_detector import (
    PersonDetectionResult,
    PlayerDetection,
)
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v6 import (
    VISUAL_FEATURE_NAMES,
    DetectedSequenceSummary,
    V6Model,
    build_adaptive_orientation_profile,
    build_frozen_orientation_profile,
    detected_features,
    fit_model,
    matrix_for,
    orientation_context,
    summarize_detected_sequence,
)


def _palette(index: int) -> np.ndarray:
    value = np.full(52, 1e-8, dtype=np.float64)
    value[index] = 1.0
    return value / np.sum(value)


def _summary(near: int, far: int) -> DetectedSequenceSummary:
    return DetectedSequenceSummary(
        near_palette=_palette(near),
        far_palette=_palette(far),
        global_palette=0.5 * (_palette(near) + _palette(far)),
        palette_instability=0.05,
        detection_coverage=0.02,
        detection_count=4.0,
        mean_confidence=0.9,
        near_support=0.5,
        far_support=0.5,
        temporal_consistency=1.0,
        raw_candidate_count=8.0,
        inference_milliseconds=10.0,
    )


class _FakeDetector:
    def detect(self, frame: np.ndarray) -> PersonDetectionResult:
        height, width = frame.shape[:2]
        detections = (
            PlayerDetection(
                x=42,
                y=35,
                width=30,
                height=35,
                hip_x=57,
                hip_y=68,
                shoulder_x=57,
                shoulder_y=40,
                score=0.9,
            ),
            PlayerDetection(
                x=132,
                y=118,
                width=38,
                height=45,
                hip_x=151,
                hip_y=160,
                shoulder_x=151,
                shoulder_y=124,
                score=0.95,
            ),
        )
        self.last_shape = (height, width)
        return PersonDetectionResult(detections, 5, 8.0)


def _colored_sequence(
    far_color: tuple[int, int, int], near_color: tuple[int, int, int]
) -> list[np.ndarray]:
    frames = []
    for _ in range(3):
        frame = np.full((200, 220, 3), 50, dtype=np.uint8)
        cv2.rectangle(frame, (42, 35), (72, 70), far_color, -1)
        cv2.rectangle(frame, (132, 118), (170, 163), near_color, -1)
        frames.append(frame)
    return frames


def _event(recording_id: str, gap: int, label: int, value: float) -> V3Event:
    row = {
        "eventId": f"{recording_id}:gap:{gap}",
        "recordingId": recording_id,
        "gapOrder": gap,
        "features": {name: value for name in VISUAL_FEATURE_NAMES},
    }
    return V3Event(
        event_id=row["eventId"],
        recording_id=recording_id,
        role="train",
        gap_order=gap,
        label=label,
        row=row,
    )


class SideSwitchV6Tests(unittest.TestCase):
    def test_detector_torso_palettes_preserve_side_swap(self) -> None:
        detector = _FakeDetector()
        blue = (255, 40, 20)
        red = (20, 40, 255)
        before = summarize_detected_sequence(
            _colored_sequence(red, blue), 0.45, detector
        )
        after = summarize_detected_sequence(
            _colored_sequence(blue, red), 0.45, detector
        )
        adaptive_context = {"before": 0.8, "after": -0.8, "quality": 0.8}
        v4 = {
            "broadSameAssignmentCost": 0.0,
            "tightSameAssignmentCost": 0.0,
            "meanSwapMargin": 0.0,
            "globalAppearanceChange": 0.0,
            "maximumCameraShift": 0.0,
            "minimumAlignmentResponse": 1.0,
        }

        features = detected_features(before, after, v4, adaptive_context)

        self.assertEqual(tuple(features), VISUAL_FEATURE_NAMES)
        self.assertEqual(before.detection_count, 2.0)
        self.assertEqual(before.temporal_consistency, 1.0)
        self.assertGreater(features["detectedSwapMargin"], 0.0)
        self.assertGreater(features["adaptiveOrientationFlipAgreement"], 0.0)

    def test_adaptive_profile_tracks_alternation_and_updates(self) -> None:
        summaries = {
            1: _summary(0, 1),
            2: _summary(0, 1),
            3: _summary(0, 1),
            4: _summary(1, 0),
            5: _summary(1, 0),
            6: _summary(0, 1),
            7: _summary(0, 1),
        }

        adaptive = build_adaptive_orientation_profile(summaries)
        frozen = build_frozen_orientation_profile(summaries)
        context = orientation_context(adaptive, 3)

        self.assertGreater(adaptive.coordinates[1], 0.0)
        self.assertLess(adaptive.coordinates[4], 0.0)
        self.assertGreater(adaptive.coordinates[7], 0.0)
        self.assertGreater(adaptive.update_count, 0)
        self.assertEqual(frozen.update_count, 0)
        self.assertGreater(context["before"], 0.0)
        self.assertLess(context["after"], 0.0)

    def test_compact_model_fits_and_round_trips(self) -> None:
        events = []
        for recording_id in ("a", "b", "c"):
            events.extend(
                [
                    _event(recording_id, 6, 0, 0.05),
                    _event(recording_id, 7, 1, 0.95),
                ]
            )

        model = fit_model(events, 0.1)
        restored = V6Model.from_dict(model.to_dict())

        np.testing.assert_allclose(
            model.predict_proba(matrix_for(events)),
            restored.predict_proba(matrix_for(events)),
        )


if __name__ == "__main__":
    unittest.main()
