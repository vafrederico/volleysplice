from __future__ import annotations

import unittest

import cv2
import numpy as np

from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v4 import FRAME_HEIGHT, FRAME_WIDTH, CourtGeometry
from analysis.side_switch_v5 import (
    VISUAL_FEATURE_NAMES,
    OrientationDecoderSettings,
    PlayerSequenceSummary,
    V5Model,
    build_orientation_profile,
    decode_orientation_opportunities,
    fit_model,
    matrix_for,
    player_features,
    summarize_player_sequence,
)


def _palette(index: int) -> np.ndarray:
    value = np.full(52, 1e-8, dtype=np.float64)
    value[index] = 1.0
    return value / np.sum(value)


def _summary(near: int, far: int) -> PlayerSequenceSummary:
    return PlayerSequenceSummary(
        near_palette=_palette(near),
        far_palette=_palette(far),
        global_palette=0.5 * (_palette(near) + _palette(far)),
        palette_instability=0.1,
        proposal_coverage=0.1,
        proposal_count=4.0,
        near_support=0.5,
        far_support=0.5,
    )


def _event(
    recording_id: str,
    gap_order: int,
    label: int,
    value: float,
    before_orientation: float = 0.0,
    after_orientation: float = 0.0,
) -> V3Event:
    row = {
        "eventId": f"{recording_id}:gap:{gap_order}",
        "recordingId": recording_id,
        "gapOrder": gap_order,
        "features": {name: value for name in VISUAL_FEATURE_NAMES},
        "orientationContext": {
            "before": before_orientation,
            "after": after_orientation,
            "quality": 1.0,
        },
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
    generator = np.random.default_rng(11)
    background = generator.integers(
        35, 65, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )
    for index in range(7):
        frame = background.copy()
        cv2.line(frame, (8, net_y), (FRAME_WIDTH - 8, net_y), (220, 220, 220), 2)
        far_width = max(6, round(12 * scale))
        far_height = max(8, round(18 * scale))
        far_x = 75 + index * 3
        far_bottom = net_y + round(8 * scale)
        cv2.rectangle(
            frame,
            (far_x, far_bottom - far_height),
            (far_x + far_width, far_bottom),
            far_color,
            -1,
        )
        near_width = max(8, round(16 * scale))
        near_height = max(12, round(28 * scale))
        near_x = 150 - index * 3
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


class SideSwitchV5Tests(unittest.TestCase):
    def test_player_proposals_preserve_swap_across_scale_change(self) -> None:
        blue = (255, 40, 20)
        red = (20, 40, 255)
        before = summarize_player_sequence(
            _side_sequence(0.40, blue, red, 0.75),
            CourtGeometry(0.40, 1.0, 7, 7),
        )
        after = summarize_player_sequence(
            _side_sequence(0.58, red, blue, 1.15),
            CourtGeometry(0.58, 1.0, 7, 7),
        )

        features = player_features(
            before,
            after,
            {source_name: 0.0 for source_name in (
                "broadSameAssignmentCost",
                "tightSameAssignmentCost",
                "meanSwapMargin",
                "globalAppearanceChange",
                "maximumCameraShift",
                "minimumAlignmentResponse",
            )},
        )

        self.assertGreater(before.proposal_count, 1.0)
        self.assertGreater(after.proposal_count, 1.0)
        self.assertGreater(features["playerSwapMargin"], 0.0)

    def test_whole_set_orientation_axis_tracks_alternating_state(self) -> None:
        summaries = {
            1: _summary(0, 1),
            2: _summary(0, 1),
            3: _summary(0, 1),
            4: _summary(1, 0),
            5: _summary(1, 0),
            6: _summary(0, 1),
            7: _summary(0, 1),
        }

        profile = build_orientation_profile(summaries)

        self.assertGreater(profile.coordinates[1], 0.0)
        self.assertLess(profile.coordinates[4], 0.0)
        self.assertGreater(profile.coordinates[7], 0.0)
        self.assertGreater(profile.anchor_separation, 0.9)

    def test_orientation_decoder_requires_alternating_flip_direction(self) -> None:
        settings = OrientationDecoderSettings(
            candidate_margin=1,
            distance_penalty=0.0,
            orientation_weight=4.0,
            maximum_opportunities=2,
        )
        alternating = [
            _event("r", 7, 1, 0.0, 0.9, -0.9),
            _event("r", 14, 1, 0.0, -0.9, 0.9),
        ]
        repeated_direction = [
            _event("r", 7, 1, 0.0, 0.9, -0.9),
            _event("r", 14, 0, 0.0, 0.9, -0.9),
        ]
        probabilities = np.asarray([0.90, 0.60])

        alternating_predictions = decode_orientation_opportunities(
            alternating, probabilities, 0.5, settings
        )
        repeated_predictions = decode_orientation_opportunities(
            repeated_direction, probabilities, 0.5, settings
        )

        self.assertEqual(alternating_predictions.tolist(), [True, True])
        self.assertEqual(repeated_predictions.tolist(), [True, False])

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
        restored = V5Model.from_dict(model.to_dict())

        np.testing.assert_allclose(
            model.predict_proba(matrix_for(events)),
            restored.predict_proba(matrix_for(events)),
        )


if __name__ == "__main__":
    unittest.main()
