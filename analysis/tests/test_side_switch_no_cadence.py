from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_no_cadence import (
    IndependentGapDecoderSettings,
    SideSwitchNoCadenceError,
    independent_gap_predictions,
    prediction_structure,
    select_independent_threshold,
)
from analysis.side_switch_v3 import V3Event


def _event(recording_id: str, gap_order: int, label: int) -> V3Event:
    return V3Event(
        event_id=f"{recording_id}:gap:{gap_order}",
        recording_id=recording_id,
        role="validation",
        gap_order=gap_order,
        label=label,
        row={},
    )


class SideSwitchNoCadenceTests(unittest.TestCase):
    def test_independent_decoder_has_no_spacing_or_count_constraint(self) -> None:
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])

        predictions = independent_gap_predictions(probabilities, 0.7)

        self.assertEqual(predictions.tolist(), [True, True, True, False])
        settings = IndependentGapDecoderSettings()
        self.assertEqual(
            IndependentGapDecoderSettings.from_dict(settings.to_dict()), settings
        )
        self.assertFalse(settings.to_dict()["usesCadence"])

    def test_threshold_selection_uses_exact_validation_f1(self) -> None:
        events = [
            _event("r", 1, 1),
            _event("r", 2, 0),
            _event("r", 8, 1),
            _event("r", 14, 0),
        ]
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])

        selection, predictions = select_independent_threshold(
            events, probabilities
        )

        self.assertEqual(selection["threshold"], 0.7)
        self.assertAlmostEqual(selection["metrics"]["f1"], 0.8)
        self.assertEqual(predictions.tolist(), [True, True, True, False])

    def test_prediction_structure_reports_adjacent_runs(self) -> None:
        events = [
            _event("a", 2, 0),
            _event("a", 3, 0),
            _event("a", 7, 0),
            _event("b", 5, 0),
        ]

        structure = prediction_structure(
            events, np.asarray([True, True, True, True])
        )

        self.assertEqual(structure["selectedCount"], 4)
        self.assertEqual(structure["consecutiveClusterCount"], 3)
        self.assertEqual(structure["adjacentSelectedPairs"], 1)
        self.assertEqual(structure["maximumConsecutiveRun"], 2)
        self.assertEqual(
            structure["byRecording"]["a"]["consecutiveClusters"],
            [[2, 3], [7]],
        )

    def test_malformed_decoder_contract_is_rejected(self) -> None:
        payload = IndependentGapDecoderSettings().to_dict()
        payload["minimumGapSeparation"] = 1

        with self.assertRaises(SideSwitchNoCadenceError):
            IndependentGapDecoderSettings.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
