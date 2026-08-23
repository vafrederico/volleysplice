from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_peak_cleanup import (
    PeakCleanupSettings,
    SideSwitchPeakCleanupError,
    combine_soft_context,
    decode_peak_cleanup,
    selected_time_structure,
)
from analysis.side_switch_v3 import V3Event


def _event(gap_order: int, transition_time: float) -> V3Event:
    return V3Event(
        event_id=f"r:gap:{gap_order}",
        recording_id="r",
        role="validation",
        gap_order=gap_order,
        label=0,
        row={"transitionTime": transition_time},
    )


class SideSwitchPeakCleanupTests(unittest.TestCase):
    def test_zero_context_weight_preserves_primary_probabilities(self) -> None:
        primary = np.asarray([0.2, 0.8])
        context = np.asarray([0.99, 0.01])

        combined = combine_soft_context(primary, context, 0.0)

        np.testing.assert_array_equal(combined, primary)

    def test_local_peak_suppresses_adjacent_and_time_near_candidates(self) -> None:
        events = [_event(1, 10.0), _event(2, 50.0), _event(4, 75.0)]
        settings = PeakCleanupSettings(
            minimum_gap_separation=2,
            minimum_time_separation_seconds=30.0,
        )

        predictions, _ = decode_peak_cleanup(
            events,
            np.asarray([0.8, 0.9, 0.85]),
            np.full(3, 0.5),
            0.5,
            settings,
        )

        # Gap 2 is the strongest peak. It suppresses adjacent gap 1 and the
        # time-near gap 4, even though all three clear the score threshold.
        self.assertEqual(predictions.tolist(), [False, True, False])

    def test_count_prior_is_soft_and_not_a_hard_six_event_cap(self) -> None:
        events = [_event(index + 1, index * 100.0) for index in range(8)]
        probabilities = np.full(8, 0.7)
        settings = PeakCleanupSettings(
            free_predictions_per_recording=6,
            count_penalty_logit=0.5,
        )

        predictions, _ = decode_peak_cleanup(
            events,
            probabilities,
            np.full(8, 0.5),
            0.5,
            settings,
        )

        self.assertEqual(int(np.sum(predictions)), 7)
        self.assertTrue(predictions[6])
        self.assertFalse(predictions[7])

    def test_production_context_is_a_soft_log_odds_term(self) -> None:
        primary = np.asarray([0.7, 0.7])
        context = np.asarray([0.9, 0.1])
        combined = combine_soft_context(primary, context, 0.5)

        self.assertGreater(combined[0], primary[0])
        self.assertLess(combined[1], primary[1])
        # Low context does not force a zero probability or categorical veto.
        self.assertGreater(combined[1], 0.0)

    def test_settings_reject_cadence_reanchoring_and_hard_gates(self) -> None:
        payload = PeakCleanupSettings().to_dict()
        payload["usesCadence"] = True
        with self.assertRaises(SideSwitchPeakCleanupError):
            PeakCleanupSettings.from_dict(payload)

        payload = PeakCleanupSettings().to_dict()
        payload["productionContextIsHardGate"] = True
        with self.assertRaises(SideSwitchPeakCleanupError):
            PeakCleanupSettings.from_dict(payload)

    def test_time_structure_is_label_independent(self) -> None:
        events = [_event(1, 10.0), _event(2, 40.0), _event(3, 100.0)]
        structure = selected_time_structure(
            events, np.asarray([True, False, True])
        )

        self.assertEqual(structure["selectedCount"], 2)
        self.assertEqual(structure["minimumSelectedTimeSpacingSeconds"], 90.0)


if __name__ == "__main__":
    unittest.main()
