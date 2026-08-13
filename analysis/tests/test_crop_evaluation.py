from __future__ import annotations

import unittest

from analysis.crop_evaluation import (
    RecordingIntervals,
    evaluate_crop_padding,
    evaluate_f1_pad_p_core_r,
    pad_and_merge_intervals,
    subtract_intervals,
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

    def test_subtract_intervals_splits_and_merges_inputs(self) -> None:
        result = subtract_intervals(
            (
                Interval(0, 5, ("ace",)),
                Interval(4, 10, ("service-fault",)),
                Interval(12, 15),
            ),
            (Interval(2, 3), Interval(8, 13)),
        )
        self.assertEqual(
            result,
            (
                Interval(0, 2, ("ace", "service-fault")),
                Interval(3, 8, ("ace", "service-fault")),
                Interval(13, 15),
            ),
        )

    def test_new_ranking_metric_pools_components_and_excludes_ignored_time(self) -> None:
        recordings = (
            RecordingIntervals(
                id="one",
                split="validation",
                duration=20,
                truth=(Interval(5, 10),),
                predictions=(Interval(6, 12),),
                ignored_intervals=(Interval(11, 12),),
            ),
            RecordingIntervals(
                id="two",
                split="validation",
                duration=20,
                truth=(Interval(0, 10),),
                predictions=(Interval(0, 5),),
            ),
        )
        row = evaluate_f1_pad_p_core_r(recordings, (0,))[0]
        # P_pad = (4 + 5) / (5 + 5), R_core = (4 + 5) / (5 + 10).
        self.assertAlmostEqual(row["P_pad"], 0.9)
        self.assertAlmostEqual(row["R_core"], 0.6)
        self.assertAlmostEqual(row["F1_padP_coreR"], 0.72)
        self.assertEqual(row["paddedModelExportSeconds"], 10)
        self.assertEqual(row["paddedHumanExportSeconds"], 15)
        self.assertEqual(row["exportDurationDifferenceSeconds"], -5)

    def test_new_ranking_metric_pads_both_model_and_human(self) -> None:
        recordings = (
            RecordingIntervals(
                id="sample",
                split="validation",
                duration=20,
                truth=(Interval(5, 10),),
                predictions=(Interval(6, 9),),
            ),
        )
        row = evaluate_f1_pad_p_core_r(recordings, (1,))[0]
        self.assertAlmostEqual(row["P_pad"], 1.0)
        self.assertAlmostEqual(row["R_core"], 1.0)
        self.assertAlmostEqual(row["F1_padP_coreR"], 1.0)
        self.assertEqual(row["paddedModelExportSeconds"], 5)
        self.assertEqual(row["paddedHumanExportSeconds"], 7)


if __name__ == "__main__":
    unittest.main()
