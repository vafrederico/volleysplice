from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_continuity import (
    QUALITY_FEATURES,
    VerifierRow,
    apply_veto,
    build_verifier_rows,
    robust_standardize,
    select_positive_threshold,
    select_veto_threshold,
)


def _source(event_id: str, player: float, v4: float, quality: float = 1.0):
    features = {
        "playerSwapMargin": player,
        "v4MeanSwapMargin": v4,
        **{name: quality for name in QUALITY_FEATURES},
    }
    return {"eventId": event_id, "recordingId": "r", "features": features}


def _row(event_id: str, score: float, label: int, selected: bool = True):
    return VerifierRow(
        event_id=event_id,
        recording_id="r",
        player_switch_evidence=score,
        v4_switch_evidence=score,
        agreement_switch_evidence=score,
        quality=1.0,
        label=label,
        selected_by_control=selected,
        source={},
    )


class SideSwitchContinuityTests(unittest.TestCase):
    def test_recording_normalization_is_label_free_and_centered(self) -> None:
        values = robust_standardize([-3.0, -1.0, 1.0, 3.0])
        self.assertAlmostEqual(float(np.median(values)), 0.0)
        self.assertLess(values[0], values[-1])

        rows = build_verifier_rows(
            [_source("a", -2.0, -4.0), _source("b", 2.0, 4.0)],
            {"a": 1, "b": 0},
            {"a"},
        )
        self.assertLess(rows[0].player_switch_evidence, 0.0)
        self.assertGreater(rows[1].agreement_switch_evidence, 0.0)
        self.assertTrue(rows[0].selected_by_control)

    def test_veto_threshold_obeys_true_positive_retention(self) -> None:
        rows = [
            _row("fp-strong", -3.0, 0),
            _row("fp", -2.0, 0),
            _row("tp-low", -1.0, 1),
            _row("tp-high", 1.0, 1),
        ]
        selected = select_veto_threshold(
            rows,
            signal="player",
            minimum_quality=0.0,
            marker_count=2,
            minimum_true_positive_retention=1.0,
        )
        retained = apply_veto(
            rows,
            signal="player",
            threshold=selected["threshold"],
            minimum_quality=0.0,
        )
        self.assertEqual(retained, {"tp-low", "tp-high"})
        self.assertEqual(selected["trainingCounts"]["truePositives"], 2)

    def test_positive_threshold_can_abstain_from_low_scores(self) -> None:
        rows = [
            _row("negative", -1.0, 0, selected=False),
            _row("positive", 2.0, 1, selected=False),
        ]
        selected = select_positive_threshold(
            rows, signal="agreement", minimum_quality=0.0, marker_count=1
        )
        self.assertEqual(selected["selectedEventIds"], ["positive"])


if __name__ == "__main__":
    unittest.main()
