from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from analysis.metrics import evaluate_intervals, interval_iou, ordered_interval_matches


@dataclass(frozen=True)
class IntervalValue:
    start: float
    end: float


@dataclass(frozen=True)
class IndexedInterval:
    start: float
    end: float
    index: int
    kind: str


class IntervalMetricTests(unittest.TestCase):
    def test_interval_iou_uses_continuous_half_open_duration(self) -> None:
        self.assertAlmostEqual(interval_iou(IntervalValue(0.0, 4.0), IntervalValue(1.0, 5.0)), 3 / 5)
        self.assertEqual(interval_iou(IntervalValue(0.0, 1.0), IntervalValue(1.0, 2.0)), 0.0)

    def test_ordered_matching_is_one_to_one_and_maximizes_iou(self) -> None:
        truth = [IntervalValue(0.0, 10.0), IntervalValue(20.0, 30.0)]
        predictions = [
            IntervalValue(0.0, 4.0),
            IntervalValue(4.0, 10.0),
            IntervalValue(20.0, 30.0),
        ]

        matches = ordered_interval_matches(truth, predictions, minimum_iou=0.3)

        self.assertEqual([(actual, predicted) for actual, predicted, _ in matches], [(0, 1), (1, 2)])
        self.assertEqual([score for _, _, score in matches], [0.6, 1.0])

    def test_ordered_matching_prioritizes_match_count_over_total_iou(self) -> None:
        truth = [IndexedInterval(float(index), float(index + 1), index, "truth") for index in range(5)]
        predictions = [
            IndexedInterval(float(index), float(index + 1), index, "prediction")
            for index in range(5)
        ]

        def synthetic_iou(actual: IndexedInterval, predicted: IndexedInterval) -> float:
            if actual.index == predicted.index:
                return 0.5
            if predicted.index == actual.index + 1:
                return 1.0
            return 0.0

        with patch("analysis.metrics.interval_iou", side_effect=synthetic_iou):
            matches = ordered_interval_matches(truth, predictions, minimum_iou=0.5)

        self.assertEqual(
            [(actual, predicted) for actual, predicted, _ in matches],
            [(index, index) for index in range(5)],
        )

    def test_evaluation_reports_event_and_time_metrics_for_ordered_intervals(self) -> None:
        truth = [IntervalValue(0.0, 4.0), IntervalValue(10.0, 14.0)]
        predictions = [
            IntervalValue(1.0, 5.0),
            IntervalValue(10.0, 13.0),
            IntervalValue(20.0, 22.0),
        ]

        metrics = evaluate_intervals(truth, predictions, minimum_iou=0.5)

        self.assertEqual(metrics["matchedRallies"], 2)
        self.assertAlmostEqual(metrics["eventPrecision"], 2 / 3)
        self.assertEqual(metrics["eventRecall"], 1.0)
        self.assertAlmostEqual(metrics["eventF1"], 0.8)
        self.assertAlmostEqual(metrics["timeIoU"], 6 / 11)
        self.assertAlmostEqual(metrics["liveTimeRecall"], 6 / 8)
        self.assertAlmostEqual(metrics["liveTimePrecision"], 6 / 9)
        self.assertEqual(metrics["missedLiveSeconds"], 2.0)
        self.assertEqual(metrics["deadSecondsRetained"], 3.0)
        self.assertEqual(metrics["startErrorsSeconds"], [1.0, 0.0])
        self.assertEqual(metrics["endErrorsSeconds"], [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
