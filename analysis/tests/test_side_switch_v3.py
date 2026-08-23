from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_v3 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VISUAL_FEATURE_NAMES,
    DecoderSettings,
    SequenceSummary,
    V3Event,
    V3Model,
    decode_opportunities,
    event_metrics,
    fit_model,
    matrix_for,
    summarize_sequence,
    visual_features,
)


def _palette(index: int) -> np.ndarray:
    value = np.full(16, 1e-6, dtype=np.float64)
    value[index] = 1.0
    return value / np.sum(value)


def _summary(near: int, far: int) -> SequenceSummary:
    return SequenceSummary(
        near_palette=_palette(near),
        far_palette=_palette(far),
        global_palette=0.5 * (_palette(near) + _palette(far)),
        motion_coverage=0.2,
        within_frame_change=0.1,
        blur_log=4.0,
        luma=0.5,
        edge_density=0.1,
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


class SideSwitchV3Tests(unittest.TestCase):
    def test_low_resolution_sequence_summary_is_finite(self) -> None:
        frames = []
        for offset in (0, 20, 40):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 60, dtype=np.uint8)
            frame[40:60, 30 + offset : 50 + offset] = (255, 255, 255)
            frames.append(frame)

        summary = summarize_sequence(frames)

        self.assertEqual(summary.near_palette.shape, (16,))
        self.assertAlmostEqual(float(np.sum(summary.global_palette)), 1.0)
        self.assertGreater(summary.motion_coverage, 0.0)

    def test_visual_assignment_prefers_swapped_sides(self) -> None:
        features = visual_features(_summary(0, 1), _summary(1, 0))

        self.assertGreater(features["swapMargin"], 0.9)
        self.assertGreater(features["orientationFlipEvidence"], 0.9)

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
        restored = V3Model.from_dict(model.to_dict())

        np.testing.assert_allclose(
            model.predict_proba(matrix_for(events)),
            restored.predict_proba(matrix_for(events)),
        )

    def test_plus_minus_four_decoder_does_not_reuse_overlap_gap(self) -> None:
        events = [
            _event("r", 10, 1, 0.0),
            _event("r", 11, 0, 0.0),
        ]
        settings = DecoderSettings(candidate_margin=4, distance_penalty=0.0)

        predictions = decode_opportunities(
            events,
            np.asarray([0.95, 0.10]),
            0.5,
            settings,
        )

        self.assertEqual(predictions.tolist(), [True, False])

    def test_decoder_reanchors_next_window_after_selected_switch(self) -> None:
        events = [
            _event("r", 10, 1, 0.0),
            _event("r", 21, 1, 0.0),
        ]
        settings = DecoderSettings(
            candidate_margin=4,
            distance_penalty=0.0,
            maximum_opportunities=2,
            reanchor_on_selection=True,
        )

        predictions = decode_opportunities(
            events,
            np.asarray([0.95, 0.95]),
            0.5,
            settings,
            force_each_opportunity=True,
        )

        self.assertEqual(predictions.tolist(), [True, True])

    def test_tolerance_metrics_match_neighboring_gap_once(self) -> None:
        events = [
            _event("r", 7, 1, 0.0),
            _event("r", 8, 0, 0.0),
            _event("r", 14, 1, 0.0),
        ]
        predictions = np.asarray([False, True, True])

        exact = event_metrics(events, predictions, tolerance=0)
        tolerant = event_metrics(events, predictions, tolerance=1)

        self.assertEqual(exact["truePositives"], 1)
        self.assertEqual(tolerant["truePositives"], 2)


if __name__ == "__main__":
    unittest.main()
