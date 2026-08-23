import unittest

import numpy as np

from analysis.side_switch_full_union_ranker import (
    UnionDecoderSettings,
    add_derived_features,
    calibrate_recording_scores,
    decode_ranked_candidates,
    fit_weighted_logistic,
    penalize_internal_candidates,
)
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v6 import matrix_for


def _row(event_id: str, time: float) -> dict[str, object]:
    return {
        "eventId": event_id,
        "recordingId": "video",
        "kind": "adjacent-rally-boundary",
        "transitionTime": time,
        "features": {},
        "score": None,
    }


class SideSwitchFullUnionRankerTests(unittest.TestCase):
    def test_derived_candidate_features_distinguish_internal_peaks(self) -> None:
        row = add_derived_features(
            {
                "kind": "internal-dead-state-peak",
                "score": 0.99,
                "features": {"existing": 1.0},
            }
        )
        self.assertEqual(row["features"]["candidateIsInternalDeadStatePeak"], 1.0)
        self.assertEqual(row["features"]["candidateGeneratorScore"], 0.99)

    def test_decoder_uses_score_ranked_local_suppression(self) -> None:
        rows = [_row("a", 10.0), _row("b", 11.0), _row("c", 40.0)]
        selected = decode_ranked_candidates(
            rows,
            np.asarray([0.8, 0.9, 0.7]),
            0.5,
            UnionDecoderSettings(minimum_time_separation_seconds=5.0),
        )
        self.assertEqual(selected.tolist(), [False, True, True])

    def test_decoder_soft_count_penalizes_only_after_free_predictions(self) -> None:
        rows = [_row("a", 10.0), _row("b", 20.0)]
        selected = decode_ranked_candidates(
            rows,
            np.asarray([0.8, 0.7]),
            0.5,
            UnionDecoderSettings(
                free_predictions_per_recording=1,
                count_penalty_logit=1.0,
            ),
        )
        self.assertEqual(selected.tolist(), [True, False])

    def test_recording_calibration_preserves_within_recording_order(self) -> None:
        rows = [_row("a", 10.0), _row("b", 20.0), _row("c", 30.0)]
        scores = np.asarray([0.1, 0.8, 0.6])
        for method in ("robust-logit", "percentile"):
            calibrated = calibrate_recording_scores(rows, scores, method)
            self.assertEqual(np.argsort(calibrated).tolist(), np.argsort(scores).tolist())

    def test_balanced_opposite_head_is_probability_complement(self) -> None:
        rows = []
        for index, (value, label) in enumerate(
            ((-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1))
        ):
            row = _row(str(index), float(index))
            row["features"] = {"x": value}
            rows.append(
                V3Event(str(index), "video", "research", index + 1, label, row)
            )
        positive = fit_weighted_logistic(rows, 0.1, ("x",), 1.0)
        opposite_rows = [
            V3Event(
                event.event_id,
                event.recording_id,
                event.role,
                event.gap_order,
                1 - event.label,
                event.row,
            )
            for event in rows
        ]
        opposite = fit_weighted_logistic(opposite_rows, 0.1, ("x",), 1.0)
        values = matrix_for(rows, ("x",))
        np.testing.assert_allclose(
            positive.predict_proba(values),
            1.0 - opposite.predict_proba(values),
            atol=1e-12,
        )

    def test_internal_penalty_is_soft_and_kind_specific(self) -> None:
        boundary = _row("boundary", 10.0)
        internal = _row("internal", 20.0)
        internal["kind"] = "internal-dead-state-peak"
        adjusted = penalize_internal_candidates(
            [boundary, internal], np.asarray([0.8, 0.8]), 1.0
        )
        self.assertAlmostEqual(adjusted[0], 0.8)
        self.assertGreater(adjusted[1], 0.0)
        self.assertLess(adjusted[1], 0.8)

    def test_extra_sample_weights_must_be_positive_and_aligned(self) -> None:
        rows = [_row("a", 0.0), _row("b", 1.0)]
        rows[0]["features"] = {"x": 0.0}
        rows[1]["features"] = {"x": 1.0}
        events = [
            V3Event("a", "video", "research", 1, 0, rows[0]),
            V3Event("b", "video", "research", 2, 1, rows[1]),
        ]
        with self.assertRaises(ValueError):
            fit_weighted_logistic(
                events,
                0.1,
                ("x",),
                0.5,
                np.asarray([1.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
