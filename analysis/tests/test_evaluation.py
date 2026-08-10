from __future__ import annotations

import unittest

from analysis.evaluation import (
    Interval,
    aggregate_evaluations,
    evaluate_intervals,
    merge_intervals,
    ordered_interval_matches,
)


class EvaluationTests(unittest.TestCase):
    def test_exact_intervals_are_perfect(self):
        intervals = [Interval(2, 5), Interval(8, 12)]
        result = evaluate_intervals(intervals, intervals)
        self.assertEqual(result["eventF1"], 1)
        self.assertEqual(result["timeIoU"], 1)
        self.assertEqual(result["liveTimeRecall"], 1)
        self.assertEqual(result["liveTimePrecision"], 1)

    def test_overlapping_predictions_are_merged_before_time_scoring(self):
        truth = [Interval(2, 8)]
        predictions = [Interval(1, 5), Interval(4, 9)]
        self.assertEqual(merge_intervals(predictions), [Interval(1, 9)])
        result = evaluate_intervals(truth, predictions)
        self.assertEqual(result["predictedRallies"], 1)
        self.assertAlmostEqual(result["liveTimeRecall"], 1)
        self.assertAlmostEqual(result["liveTimePrecision"], 0.75)

    def test_matching_maximizes_count_before_overlap(self):
        truth = [Interval(0, 10), Interval(10, 20)]
        predictions = [Interval(0, 9), Interval(9, 20)]
        matches = ordered_interval_matches(truth, predictions, minimum_iou=0.5)
        self.assertEqual([(actual, predicted) for actual, predicted, _ in matches], [(0, 0), (1, 1)])

    def test_aggregate_uses_micro_counts_and_time(self):
        first = evaluate_intervals([Interval(0, 4)], [Interval(0, 4)])
        second = evaluate_intervals([Interval(0, 4)], [Interval(8, 12)])
        aggregate = aggregate_evaluations([first, second])
        self.assertEqual(aggregate["eventF1"], 0.5)
        self.assertEqual(aggregate["timeIoU"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
