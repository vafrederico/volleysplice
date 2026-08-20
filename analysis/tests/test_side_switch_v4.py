from __future__ import annotations

import unittest

import cv2
import numpy as np

from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v4 import (
    CANONICAL_NET_Y,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VISUAL_FEATURE_NAMES,
    CourtGeometry,
    V4Model,
    _align_frames,
    estimate_court_geometry,
    fit_model,
    matrix_for,
    normalize_court_frame,
    summarize_sequence,
    visual_features,
)


def _event(recording_id: str, gap_order: int, label: int, value: float) -> V3Event:
    row = {
        "eventId": f"{recording_id}:gap:{gap_order}",
        "recordingId": recording_id,
        "gapOrder": gap_order,
        "features": {name: value for name in VISUAL_FEATURE_NAMES},
    }
    return V3Event(
        event_id=str(row["eventId"]),
        recording_id=recording_id,
        role="train",
        gap_order=gap_order,
        label=label,
        row=row,
    )


def _side_sequence(
    net_y_ratio: float,
    near_color: tuple[int, int, int],
    far_color: tuple[int, int, int],
    scale: float,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    net_y = round(FRAME_HEIGHT * net_y_ratio)
    for index in range(7):
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 100, 45), np.uint8)
        cv2.line(frame, (8, net_y), (FRAME_WIDTH - 8, net_y), (230, 230, 230), 2)
        far_width = max(6, round(12 * scale))
        far_height = max(8, round(18 * scale))
        far_x = 80 + index * 3
        far_bottom = net_y + round(7 * scale)
        cv2.rectangle(
            frame,
            (far_x, far_bottom - far_height),
            (far_x + far_width, far_bottom),
            far_color,
            -1,
        )
        near_width = max(8, round(16 * scale))
        near_height = max(12, round(28 * scale))
        near_x = 145 - index * 3
        near_bottom = min(FRAME_HEIGHT - 5, net_y + round(48 * scale))
        cv2.rectangle(
            frame,
            (near_x, near_bottom - near_height),
            (near_x + near_width, near_bottom),
            near_color,
            -1,
        )
        frames.append(frame)
    return frames


class SideSwitchV4Tests(unittest.TestCase):
    def test_court_geometry_finds_long_horizontal_net(self) -> None:
        frames = _side_sequence(0.41, (255, 0, 0), (0, 0, 255), 0.8)

        geometry = estimate_court_geometry(frames)

        self.assertGreater(geometry.detected_frames, 0)
        self.assertAlmostEqual(geometry.net_y_ratio, 0.41, delta=0.04)

    def test_normalization_places_net_at_canonical_height(self) -> None:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        source_y = round(FRAME_HEIGHT * 0.37)
        cv2.line(frame, (0, source_y), (FRAME_WIDTH - 1, source_y), (255, 255, 255), 2)

        normalized = normalize_court_frame(frame, CourtGeometry(0.37, 1.0, 7, 7))
        row_energy = np.mean(normalized, axis=(1, 2))

        self.assertAlmostEqual(
            int(np.argmax(row_energy)), round((FRAME_HEIGHT - 1) * CANONICAL_NET_Y), delta=2
        )

    def test_camera_translation_is_compensated(self) -> None:
        generator = np.random.default_rng(7)
        reference = generator.integers(
            0, 255, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
        )
        frames = []
        for dx, dy in ((-4, 2), (-2, 1), (0, 0), (2, -1), (4, -2)):
            frames.append(
                cv2.warpAffine(
                    reference,
                    np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]]),
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    borderMode=cv2.BORDER_REFLECT,
                )
            )

        aligned, maximum_shift, minimum_response = _align_frames(frames)
        residual = np.mean(
            [cv2.absdiff(aligned[2], frame) for frame in aligned], axis=(0, 1, 2, 3)
        )

        self.assertGreater(maximum_shift, 0.01)
        self.assertGreater(minimum_response, 0.1)
        self.assertLess(float(residual), 12.0)

    def test_swap_evidence_survives_net_height_and_scale_change(self) -> None:
        blue = (255, 40, 20)
        red = (20, 40, 255)
        before_frames = _side_sequence(0.40, blue, red, 0.75)
        after_frames = _side_sequence(0.58, red, blue, 1.15)
        before = summarize_sequence(
            before_frames, CourtGeometry(0.40, 1.0, 7, 7)
        )
        after = summarize_sequence(after_frames, CourtGeometry(0.58, 1.0, 7, 7))

        features = visual_features(before, after)

        self.assertEqual(tuple(features), VISUAL_FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertGreater(features["meanSwapMargin"], 0.0)

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
        restored = V4Model.from_dict(model.to_dict())

        np.testing.assert_allclose(
            model.predict_proba(matrix_for(events)),
            restored.predict_proba(matrix_for(events)),
        )


if __name__ == "__main__":
    unittest.main()
