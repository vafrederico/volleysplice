from __future__ import annotations

import unittest

import numpy as np

from analysis.config import DecoderConfig
from analysis.decoder import clean_mask, decode_probabilities, smooth_probabilities


class DecoderTests(unittest.TestCase):
    def test_even_smoothing_window_keeps_requested_length_and_alignment(self) -> None:
        probabilities = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        smoothed = smooth_probabilities(probabilities, window_samples=2)

        np.testing.assert_allclose(
            smoothed,
            np.asarray([1.0, 1.5, 2.5, 3.5], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(smoothed.shape, probabilities.shape)

    def test_even_four_sample_window_is_not_silently_reduced_to_three(self) -> None:
        impulse = np.asarray([0.0, 0.0, 4.0, 0.0, 0.0], dtype=np.float32)

        smoothed = smooth_probabilities(impulse, window_samples=4)

        np.testing.assert_array_equal(
            smoothed,
            np.asarray([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_bridges_only_internal_gaps_at_or_below_limit(self) -> None:
        mask = np.asarray(
            [True, True, False, False, True, True, False, False, False, True, True]
        )

        cleaned = clean_mask(mask, min_live_samples=1, bridge_gap_samples=2)

        np.testing.assert_array_equal(
            cleaned,
            np.asarray([True, True, True, True, True, True, False, False, False, True, True]),
        )

    def test_removes_runs_shorter_than_minimum_but_keeps_exact_minimum(self) -> None:
        mask = np.asarray([False, True, False, True, True, False])

        cleaned = clean_mask(mask, min_live_samples=2, bridge_gap_samples=0)

        np.testing.assert_array_equal(cleaned, np.asarray([False, False, False, True, True, False]))

    def test_decode_applies_minimum_duration_and_sample_boundaries(self) -> None:
        fps = 2.0
        times = np.arange(6, dtype=np.float64) / fps
        probabilities = np.asarray([0.1, 0.9, 0.1, 0.9, 0.9, 0.1], dtype=np.float32)
        config = DecoderConfig(
            smoothing_seconds=0.0,
            enter_threshold=0.5,
            exit_threshold=0.4,
            min_live_seconds=1.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=1.0,
            short_event_threshold=1.0,
        )

        intervals, smoothed = decode_probabilities(times, probabilities, 3.0, config, fps)

        np.testing.assert_array_equal(smoothed, probabilities)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].start, 1.25)
        self.assertAlmostEqual(intervals[0].end, 2.25)
        self.assertAlmostEqual(intervals[0].confidence, 0.9, places=6)

    def test_keeps_a_short_high_confidence_event_but_rejects_a_weak_one(self) -> None:
        fps = 4.0
        times = np.arange(8, dtype=np.float64) / fps
        probabilities = np.asarray(
            [0.1, 0.92, 0.91, 0.1, 0.1, 0.65, 0.64, 0.1], dtype=np.float32
        )
        config = DecoderConfig(
            smoothing_seconds=0.0,
            enter_threshold=0.6,
            exit_threshold=0.5,
            min_live_seconds=1.0,
            bridge_gap_seconds=0.0,
            short_event_min_seconds=0.5,
            short_event_threshold=0.85,
        )

        intervals, _ = decode_probabilities(times, probabilities, 2.0, config, fps)

        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].start, 0.125)
        self.assertAlmostEqual(intervals[0].end, 0.625)


if __name__ == "__main__":
    unittest.main()
