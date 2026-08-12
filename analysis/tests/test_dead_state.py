from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from analysis.dead_state import (
    DeadStateDecoderConfig,
    DeadStateDetection,
    dead_state_labels_for_times,
    decode_dead_state_after_serve,
    end_transition_labels_for_times,
)
from analysis.model import DEAD_STATE_TASK, PREDICTION_TASKS


@dataclass(frozen=True)
class IntervalValue:
    start: float
    end: float


class DeadStateTargetTests(unittest.TestCase):
    def test_serve_window_is_half_open_and_counts_live_and_dead_samples(self) -> None:
        times = np.arange(0.0, 3.25, 0.25)

        labels, mask = dead_state_labels_for_times(
            times,
            [IntervalValue(0.5, 1.25)],
            horizon=2.0,
        )

        np.testing.assert_array_equal(
            times[mask], np.arange(0.5, 2.5, 0.25)
        )
        self.assertEqual(int(np.sum(mask)), 8)
        self.assertEqual(int(np.sum(labels[mask] == 0)), 3)
        self.assertEqual(int(np.sum(labels[mask] == 1)), 5)
        self.assertEqual(labels[np.where(times == 1.25)[0][0]], 1.0)
        self.assertFalse(mask[np.where(times == 2.5)[0][0]])

    def test_long_rally_has_only_negative_samples_inside_horizon(self) -> None:
        labels, mask = dead_state_labels_for_times(
            np.arange(0.0, 5.0, 0.5),
            [IntervalValue(1.0, 4.5)],
            horizon=2.0,
        )

        self.assertEqual(int(np.sum(mask)), 4)
        np.testing.assert_array_equal(labels[mask], np.zeros(4, dtype=np.float32))

    def test_live_precedence_makes_overlaps_order_independent(self) -> None:
        times = np.arange(0.0, 5.0, 0.5)
        rallies = [IntervalValue(0.0, 1.0), IntervalValue(2.0, 3.0)]

        forward = dead_state_labels_for_times(times, rallies, horizon=4.0)
        reverse = dead_state_labels_for_times(times, reversed(rallies), horizon=4.0)

        np.testing.assert_array_equal(forward[0], reverse[0])
        np.testing.assert_array_equal(forward[1], reverse[1])
        # At 2.5, rally one says dead while rally two says live; live wins.
        self.assertEqual(forward[0][np.where(times == 2.5)[0][0]], 0.0)
        # Once both overlapping windows say dead, the result becomes positive.
        self.assertEqual(forward[0][np.where(times == 3.5)[0][0]], 1.0)

    def test_end_transition_window_is_clamped_to_rally_and_next_start(self) -> None:
        times = np.arange(0.0, 7.0, 0.5)
        labels, mask = end_transition_labels_for_times(
            times,
            [IntervalValue(1.0, 2.0), IntervalValue(3.0, 5.5)],
            before_seconds=2.0,
            after_seconds=2.0,
        )

        # First window is [1, 3), not [0, 4), and the second is [3.5, 7.5).
        expected_mask = ((times >= 1.0) & (times < 3.0)) | (times >= 3.5)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertEqual(labels[np.where(times == 1.5)[0][0]], 0.0)
        self.assertEqual(labels[np.where(times == 2.0)[0][0]], 1.0)
        self.assertFalse(mask[np.where(times == 3.0)[0][0]])
        self.assertEqual(labels[np.where(times == 5.0)[0][0]], 0.0)
        self.assertEqual(labels[np.where(times == 5.5)[0][0]], 1.0)

    def test_end_transition_overlap_uses_live_precedence(self) -> None:
        times = np.arange(0.0, 5.0, 0.5)
        labels, mask = end_transition_labels_for_times(
            times,
            [IntervalValue(0.0, 3.0), IntervalValue(2.0, 4.0)],
            before_seconds=2.0,
            after_seconds=2.0,
        )

        self.assertTrue(mask[np.where(times == 2.5)[0][0]])
        self.assertEqual(labels[np.where(times == 2.5)[0][0]], 0.0)
        self.assertEqual(labels[np.where(times == 4.0)[0][0]], 1.0)

    def test_target_validation_and_registered_prediction_task(self) -> None:
        self.assertIn(DEAD_STATE_TASK, PREDICTION_TASKS)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            dead_state_labels_for_times(
                np.asarray([0.0, 1.0, 0.5]),
                [IntervalValue(0.0, 1.0)],
                horizon=2.0,
            )
        with self.assertRaisesRegex(ValueError, "horizon must be positive"):
            dead_state_labels_for_times(
                np.asarray([0.0]),
                [IntervalValue(0.0, 1.0)],
                horizon=0.0,
            )
        with self.assertRaisesRegex(ValueError, "after their starts"):
            end_transition_labels_for_times(
                np.asarray([0.0]),
                [IntervalValue(1.0, 1.0)],
            )


class DeadStateDecoderTests(unittest.TestCase):
    @staticmethod
    def config(**overrides: float | int) -> DeadStateDecoderConfig:
        values: dict[str, float | int] = {
            "dead_threshold": 0.8,
            "live_reset_threshold": 0.3,
            "minimum_live_samples": 2,
            "minimum_dead_samples": 2,
            "min_after_serve_seconds": 0.5,
            "max_after_serve_seconds": 3.0,
        }
        values.update(overrides)
        return DeadStateDecoderConfig(**values)

    def test_requires_live_evidence_then_returns_first_stable_dead_run(self) -> None:
        detection = decode_dead_state_after_serve(
            np.arange(0.0, 4.0, 0.25),
            np.asarray(
                [
                    0.9,
                    0.2,
                    0.1,
                    0.5,
                    0.85,
                    0.4,
                    0.82,
                    0.81,
                    0.95,
                    0.95,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ]
            ),
            serve_time=0.0,
            config=self.config(),
        )

        self.assertEqual(detection, DeadStateDetection(time=1.5, confidence=0.81))

    def test_already_dead_anchor_is_rejected(self) -> None:
        detection = decode_dead_state_after_serve(
            np.arange(10.0, 14.0, 0.25),
            np.full(16, 0.95),
            serve_time=10.0,
            config=self.config(),
        )

        self.assertIsNone(detection)

    def test_live_evidence_before_minimum_lag_arms_decoder(self) -> None:
        detection = decode_dead_state_after_serve(
            np.asarray([10.0, 10.25, 10.5, 10.75]),
            np.asarray([0.1, 0.2, 0.9, 0.9]),
            serve_time=10.0,
            config=self.config(),
        )

        self.assertEqual(detection, DeadStateDetection(10.5, 0.9))

    def test_first_qualifying_run_wins_even_when_a_later_run_is_stronger(self) -> None:
        detection = decode_dead_state_after_serve(
            np.arange(0.0, 2.0, 0.25),
            np.asarray([0.1, 0.1, 0.8, 0.8, 0.1, 0.1, 0.99, 0.99]),
            serve_time=0.0,
            config=self.config(),
        )

        self.assertEqual(detection, DeadStateDetection(0.5, 0.8))

    def test_maximum_lag_is_inclusive_and_next_serve_is_exclusive(self) -> None:
        times = np.asarray([0.0, 0.25, 2.75, 3.0, 3.25])
        probabilities = np.asarray([0.1, 0.1, 0.2, 0.9, 0.9])
        one_sample = self.config(minimum_dead_samples=1)

        at_maximum = decode_dead_state_after_serve(
            times,
            probabilities,
            serve_time=0.0,
            config=one_sample,
        )
        before_next = decode_dead_state_after_serve(
            times,
            probabilities,
            serve_time=0.0,
            config=one_sample,
            next_serve_time=3.0,
        )

        self.assertEqual(at_maximum, DeadStateDetection(3.0, 0.9))
        self.assertIsNone(before_next)

    def test_offset_is_clipped_and_config_round_trips(self) -> None:
        config = self.config(
            minimum_live_samples=1,
            minimum_dead_samples=1,
            min_after_serve_seconds=0.0,
            time_offset_seconds=-2.0,
        )
        detection = decode_dead_state_after_serve(
            np.asarray([0.0, 0.25]),
            np.asarray([0.1, 0.9]),
            serve_time=0.0,
            config=config,
            duration=1.0,
        )

        self.assertEqual(detection, DeadStateDetection(0.0, 0.9))
        self.assertEqual(DeadStateDecoderConfig.from_dict(config.to_dict()), config)

    def test_decoder_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "live reset threshold"):
            self.config(live_reset_threshold=0.8).validate()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.config(minimum_dead_samples=0).validate()
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            decode_dead_state_after_serve(
                np.asarray([0.0]),
                np.asarray([1.1]),
                0.0,
                self.config(),
            )


if __name__ == "__main__":
    unittest.main()
