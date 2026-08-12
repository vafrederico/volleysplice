from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from analysis.dead_ball import (
    DeadBallDecoderConfig,
    DeadBallDetection,
    dead_ball_labels_for_times,
    decode_dead_ball_probabilities,
    select_dead_ball_after_serve,
)


@dataclass(frozen=True)
class IntervalValue:
    start: float
    end: float


class DeadBallTargetTests(unittest.TestCase):
    def test_labels_end_radius_and_nearest_between_sample_boundary(self) -> None:
        times = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        pulse = dead_ball_labels_for_times(
            times, [IntervalValue(0.1, 0.37)], radius=0.0
        )
        window = dead_ball_labels_for_times(
            times, [IntervalValue(0.1, 0.5)], radius=0.25
        )

        np.testing.assert_array_equal(
            pulse, np.asarray([0, 1, 0, 0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            window, np.asarray([0, 1, 1, 1], dtype=np.float32)
        )

    def test_empty_grid_and_unordered_grid(self) -> None:
        result = dead_ball_labels_for_times(
            np.asarray([], dtype=np.float64),
            [IntervalValue(0.0, 1.0)],
            radius=0.25,
        )
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (0,))

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            dead_ball_labels_for_times(
                np.asarray([0.0, 0.5, 0.25]),
                [IntervalValue(0.0, 0.2)],
                radius=0.5,
            )


class DeadBallDecoderTests(unittest.TestCase):
    def test_emits_strongest_peak_per_threshold_island_in_time_order(self) -> None:
        detections = decode_dead_ball_probabilities(
            np.arange(8, dtype=np.float64),
            np.asarray([0.1, 0.8, 0.9, 0.1, 0.85, 0.1, 0.95, 0.95]),
            DeadBallDecoderConfig(threshold=0.8),
        )

        self.assertEqual([item.time for item in detections], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(
            [item.confidence for item in detections], [0.9, 0.85, 0.95]
        )

    def test_peak_tie_uses_earliest_time_and_offset_is_clipped(self) -> None:
        detections = decode_dead_ball_probabilities(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.9, 0.9, 0.1]),
            DeadBallDecoderConfig(threshold=0.8, time_offset_seconds=-0.5),
            duration=1.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].time, 0.0)
        self.assertAlmostEqual(detections[0].confidence, 0.9)

    def test_config_round_trip_and_invalid_probabilities(self) -> None:
        config = DeadBallDecoderConfig.from_dict(
            {"threshold": 0.7, "timeOffsetSeconds": 0.25}
        )
        self.assertEqual(
            config.to_dict(),
            {"threshold": 0.7, "timeOffsetSeconds": 0.25},
        )

        with self.assertRaisesRegex(ValueError, "between zero and one"):
            DeadBallDecoderConfig(threshold=1.0).validate()
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            decode_dead_ball_probabilities(
                np.asarray([0.0]),
                np.asarray([1.1]),
                DeadBallDecoderConfig(),
            )


class DeadBallSelectionTests(unittest.TestCase):
    def test_selects_strongest_then_earliest_inside_serve_window(self) -> None:
        selected = select_dead_ball_after_serve(
            [
                DeadBallDetection(11.0, 0.99),
                DeadBallDetection(12.5, 0.8),
                DeadBallDetection(12.0, 0.8),
                DeadBallDetection(15.5, 0.95),
            ],
            serve_time=10.0,
            min_after=2.0,
            max_after=5.0,
        )

        self.assertEqual(selected, DeadBallDetection(12.0, 0.8))

    def test_next_serve_is_an_exclusive_half_open_upper_bound(self) -> None:
        selected = select_dead_ball_after_serve(
            [
                DeadBallDetection(14.75, 0.7),
                DeadBallDetection(15.0, 0.99),
            ],
            serve_time=10.0,
            min_after=0.5,
            max_after=8.0,
            before_time=15.0,
        )

        self.assertEqual(selected, DeadBallDetection(14.75, 0.7))

    def test_duration_window_boundaries_are_inclusive(self) -> None:
        lower = DeadBallDetection(10.5, 0.6)
        upper = DeadBallDetection(15.0, 0.7)

        self.assertEqual(
            select_dead_ball_after_serve(
                [lower], 10.0, min_after=0.5, max_after=5.0
            ),
            lower,
        )
        self.assertEqual(
            select_dead_ball_after_serve(
                [upper], 10.0, min_after=0.5, max_after=5.0
            ),
            upper,
        )

    def test_returns_none_without_an_eligible_detection(self) -> None:
        selected = select_dead_ball_after_serve(
            [DeadBallDetection(11.0, 0.9)],
            serve_time=10.0,
            min_after=2.0,
            max_after=5.0,
        )

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
