from __future__ import annotations

import unittest

from analysis.crop_evaluation import (
    RecordingIntervals,
    evaluate_crop_padding,
    pad_and_merge_intervals,
)
from analysis.schema import Interval


class CropEvaluationTests(unittest.TestCase):
    def test_padding_clamps_and_merges_export_intervals(self) -> None:
        result = pad_and_merge_intervals(
            (Interval(1, 3), Interval(4, 6), Interval(9, 10)),
            duration=10,
            padding_seconds=1,
        )
        self.assertEqual(result, (Interval(0, 7), Interval(8, 10)))

    def test_sweep_reports_recall_and_retained_time_tradeoff(self) -> None:
        recordings = (
            RecordingIntervals(
                id="sample",
                split="test",
                duration=20,
                truth=(Interval(5, 10),),
                predictions=(Interval(6, 9),),
            ),
        )
        rows = evaluate_crop_padding(recordings, (0, 1))
        self.assertAlmostEqual(rows[0]["aggregate"]["liveTimeRecall"], 0.6)
        self.assertAlmostEqual(rows[1]["aggregate"]["liveTimeRecall"], 1.0)
        self.assertEqual(rows[1]["retainedVideoSeconds"], 5)
        self.assertAlmostEqual(
            rows[1]["deltaFromFirstSetting"]["liveTimeRecallPoints"], 40
        )

    def test_rejects_negative_padding(self) -> None:
        with self.assertRaises(ValueError):
            pad_and_merge_intervals((Interval(1, 2),), 10, -1)


if __name__ == "__main__":
    unittest.main()
