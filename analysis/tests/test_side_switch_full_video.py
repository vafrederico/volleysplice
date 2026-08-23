from __future__ import annotations

import unittest

from analysis.side_switch_full_video import (
    SideSwitchFullVideoEvaluationError,
    event_metric_counts,
    monotonic_interval_match,
)


def _proposal(start: float, end: float, anchor: float) -> dict[str, float]:
    return {"gapStart": start, "gapEnd": end, "transitionTime": anchor}


def _marker(time: float) -> dict[str, float]:
    return {"time": time}


class SideSwitchFullVideoEvaluationTests(unittest.TestCase):
    def test_interval_padding_controls_boundary_match(self) -> None:
        proposals = [_proposal(10.0, 20.0, 15.0)]
        markers = [_marker(23.0)]

        strict = monotonic_interval_match(proposals, markers, 0.0)
        padded = monotonic_interval_match(proposals, markers, 4.0)

        self.assertEqual(len(strict.pairs), 0)
        self.assertEqual(len(padded.pairs), 1)

    def test_matching_is_one_to_one_when_padded_intervals_overlap(self) -> None:
        proposals = [
            _proposal(10.0, 20.0, 15.0),
            _proposal(21.0, 30.0, 25.0),
        ]
        markers = [_marker(19.0)]

        result = monotonic_interval_match(proposals, markers, 4.0)

        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.pairs[0].proposal_index, 0)
        self.assertEqual(result.unmatched_proposal_indices, (1,))

    def test_matching_maximizes_cardinality_before_anchor_distance(self) -> None:
        proposals = [
            _proposal(0.0, 11.0, 10.0),
            _proposal(9.0, 20.0, 11.0),
        ]
        markers = [_marker(1.0), _marker(10.0)]

        result = monotonic_interval_match(proposals, markers, 0.0)

        self.assertEqual(len(result.pairs), 2)
        self.assertEqual(
            [(pair.proposal_index, pair.marker_index) for pair in result.pairs],
            [(0, 0), (1, 1)],
        )

    def test_metrics_keep_zero_proposal_precision_undefined(self) -> None:
        result = monotonic_interval_match([], [_marker(10.0)], 0.0)

        metrics = event_metric_counts(result)

        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["falseNegatives"], 1)

    def test_metrics_report_zero_f1_for_complete_miss(self) -> None:
        result = monotonic_interval_match(
            [_proposal(0.0, 5.0, 2.5)], [_marker(10.0)], 0.0
        )

        metrics = event_metric_counts(result)

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_negative_padding_is_rejected(self) -> None:
        with self.assertRaises(SideSwitchFullVideoEvaluationError):
            monotonic_interval_match([], [], -1.0)


if __name__ == "__main__":
    unittest.main()
