from __future__ import annotations

import json
import math
import unittest

from analysis.crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
from analysis.neural_evaluation import evaluate_predictions, score_predictions
from analysis.schema import Interval


def recording(
    predictions: list,
    *,
    rallies: list | None = None,
    ignored: list | None = None,
    identifier: str = "one",
    group: str = "group-a",
    duration: float = 20,
) -> dict:
    return {
        "id": identifier,
        "sourceGroup": group,
        "durationSeconds": duration,
        "rallies": [[5, 10]] if rallies is None else rallies,
        "predictions": predictions,
        "ignoredIntervals": [] if ignored is None else ignored,
    }


class NeuralEvaluationTests(unittest.TestCase):
    def test_reports_four_cases_and_declared_primary_not_best_padding(self) -> None:
        report = evaluate_predictions([recording([[6, 9]])], primary_padding_seconds=0)
        self.assertEqual(
            [row["paddingSecondsBeforeAndAfter"] for row in report["padding"]],
            [0, 1, 2, 3],
        )
        self.assertAlmostEqual(report["objective"], 0.75)
        self.assertEqual(report["objective"], report["primary"]["F1_padP_coreR"])
        self.assertEqual(report["padding"][1]["F1_padP_coreR"], 1)
        self.assertEqual(report["metricContract"]["joinGapSeconds"], 3)

    def test_exact_three_second_gap_remains_a_cut(self) -> None:
        report = evaluate_predictions([recording([[0, 1], [4, 5]], rallies=[[0, 1]])])
        zero = report["padding"][0]
        self.assertEqual(zero["outputCropCount"], 2)
        self.assertEqual(zero["paddedModelExportSeconds"], 2)
        self.assertAlmostEqual(zero["F1_padP_coreR"], 2 / 3)
        joined = evaluate_predictions([recording([[0, 1], [3.999, 5]], rallies=[[0, 1]])])
        self.assertEqual(joined["padding"][0]["outputCropCount"], 1)
        self.assertEqual(joined["padding"][0]["paddedModelExportSeconds"], 5)

    def test_ignored_spans_removed_after_join_and_never_rejoined(self) -> None:
        report = evaluate_predictions([
            recording([[0, 1], [3, 4]], rallies=[[0, 1], [3, 4]], ignored=[[1.5, 2.5]])
        ])
        zero = report["padding"][0]
        self.assertEqual(zero["paddedModelExportSeconds"], 3)
        self.assertEqual(zero["paddedHumanExportSeconds"], 3)
        self.assertEqual(zero["outputCropCount"], 2)
        self.assertEqual(zero["F1_padP_coreR"], 1)

    def test_ignored_predictions_are_neither_true_nor_false_positive(self) -> None:
        report = evaluate_predictions([
            recording([[5, 10], [15, 18]], ignored=[[15, 18]])
        ], primary_padding_seconds=0)
        self.assertEqual(report["objective"], 1)
        self.assertEqual(report["primary"]["paddedModelExportSeconds"], 5)
        self.assertEqual(report["guardrails"]["predictedRallies"], 1)
        self.assertEqual(report["guardrails"]["eventF1"], 1)

    def test_pools_durations_instead_of_recording_or_source_f1(self) -> None:
        report = evaluate_predictions([
            recording([[0, 9]], rallies=[[0, 9]], identifier="long", group="large"),
            recording([], rallies=[[0, 1]], identifier="short", group="small"),
        ], primary_padding_seconds=0)
        self.assertEqual(report["primary"]["P_pad"], 1)
        self.assertAlmostEqual(report["primary"]["R_core"], 0.9)
        self.assertAlmostEqual(report["objective"], 1.8 / 1.9)
        self.assertNotAlmostEqual(report["objective"], 0.5)
        self.assertEqual(report["sourceGroups"]["large"]["objective"], 1)
        self.assertEqual(report["sourceGroups"]["small"]["objective"], 0)
        self.assertEqual(report["sourceGroupCount"], 2)

    def test_same_timestamps_in_other_recordings_do_not_intersect(self) -> None:
        report = evaluate_predictions([
            recording([], rallies=[[0, 5]], identifier="first"),
            recording([[0, 5]], rallies=[[10, 15]], identifier="second"),
        ], primary_padding_seconds=0)
        self.assertEqual(report["objective"], 0)
        self.assertEqual(report["primary"]["paddedPrecisionIntersectionSeconds"], 0)
        self.assertEqual(report["primary"]["coreRecallIntersectionSeconds"], 0)

    def test_empty_predictions_have_zero_objective_and_complete_losses(self) -> None:
        report = evaluate_predictions([recording([])])
        for row in report["padding"]:
            self.assertEqual(row["P_pad"], 0)
            self.assertEqual(row["R_core"], 0)
            self.assertEqual(row["F1_padP_coreR"], 0)
            self.assertEqual(row["paddedModelExportSeconds"], 0)
        self.assertEqual(report["guardrails"]["primaryExportCoverage"]["completeRallyLosses"], 1)
        self.assertEqual(report["guardrails"]["endBoundaryMaeSeconds"], None)
        json.dumps(report, allow_nan=False)

    def test_rejects_zero_core_denominator_in_any_recording(self) -> None:
        cases = [recording([], rallies=[]), recording([[5, 10]], ignored=[[0, 20]])]
        for row in cases:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "no evaluable core"):
                evaluate_predictions([recording([], identifier="valid"), row])
        with self.assertRaisesRegex(ValueError, "empty recording"):
            evaluate_predictions([])

    def test_clips_bounds_and_accepts_supported_interval_forms(self) -> None:
        report = evaluate_predictions([
            recording(
                [Interval(-2, 10), {"start": 9, "end": 25}],
                rallies=[{"start": -5, "end": 25, "tags": ["ace"]}],
            )
        ])
        self.assertEqual(report["primary"]["paddedModelExportSeconds"], 20)
        self.assertEqual(report["primary"]["coreHumanSeconds"], 20)
        self.assertEqual(report["objective"], 1)
        self.assertEqual(report["guardrails"]["outcomeSlices"]["ace"]["rallies"], 1)

    def test_guardrails_preserve_outcome_tags_and_original_rally_coverage(self) -> None:
        report = evaluate_predictions([
            recording(
                [[2, 4], [12, 15]],
                rallies=[Interval(2, 4, ("service-fault",)), Interval(10, 15)],
            )
        ])
        guards = report["guardrails"]
        self.assertEqual(guards["outcomeSlices"]["serviceFault"]["strictMatchRecall"], 1)
        self.assertEqual(guards["outcomeSlices"]["shortAtMost3Seconds"]["rallies"], 1)
        self.assertEqual(guards["coreCoverage"]["partialRallyLosses"], 1)
        self.assertEqual(guards["primaryExportCoverage"]["fullyCoveredRallies"], 2)
        self.assertAlmostEqual(guards["startBoundaryMaeSeconds"], 1)
        self.assertEqual(guards["endBoundaryMaeSeconds"], 0)

    def test_ignored_cut_does_not_invent_short_events_or_boundary_errors(self) -> None:
        report = evaluate_predictions([
            recording([[0, 10], [15, 17]], rallies=[[0, 10], [15, 17]], ignored=[[2, 8]])
        ], primary_padding_seconds=0)
        guards = report["guardrails"]
        self.assertEqual(guards["ignoredTouchedRalliesExcludedFromEvents"], 1)
        self.assertEqual(guards["trueRallies"], 1)
        self.assertEqual(guards["outcomeSlices"]["shortAtMost3Seconds"]["rallies"], 1)
        self.assertEqual(guards["coreCoverage"]["originalRallies"], 2)
        self.assertEqual(guards["coreCoverage"]["evaluableRallies"], 2)
        self.assertEqual(guards["timeCoverage"]["trueLiveSeconds"], 6)
        self.assertEqual(report["primary"]["coreHumanSeconds"], 6)

    def test_matches_canonical_crop_helper_components(self) -> None:
        rows = [recording([[6, 12]], ignored=[[11, 12]])]
        canonical = evaluate_f1_pad_p_core_r([
            RecordingIntervals("one", "development", 20, (Interval(5, 10),),
                               (Interval(6, 12),), (Interval(11, 12),))
        ], (0, 1, 2, 3))
        self.assertEqual(evaluate_predictions(rows)["padding"], canonical)

    def test_fast_selection_score_matches_full_report_for_each_padding(self) -> None:
        rows = [recording([[6, 12]], ignored=[[11, 12]]),
                recording([], rallies=[[0, 2]], identifier="two", group="group-b")]
        for padding in (0, 1, 2, 3):
            with self.subTest(padding=padding):
                self.assertEqual(
                    score_predictions(rows, primary_padding_seconds=padding),
                    evaluate_predictions(rows, primary_padding_seconds=padding)["objective"],
                )

    def test_no_uncensored_events_reports_unavailable_event_guardrails(self) -> None:
        report = evaluate_predictions([
            recording([[0, 10]], rallies=[[0, 10]], ignored=[[3, 6]])
        ])
        self.assertEqual(report["objective"], 1)
        self.assertFalse(report["guardrails"]["eventMetricsAvailable"])
        self.assertIsNone(report["guardrails"]["eventF1"])
        self.assertEqual(report["guardrails"]["liveTimeRecall"], 1)
        self.assertEqual(report["guardrails"]["trueLiveSeconds"], 7)
        json.dumps(report, allow_nan=False)

    def test_rejects_duplicate_ids_and_invalid_numeric_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_predictions([recording([]), recording([])])
        for duration in (0, -1, math.nan, math.inf, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                evaluate_predictions([recording([], duration=duration)])
        for predictions in ([[1, math.nan]], [[3, 2]], [[True, 3]]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_predictions([recording(predictions)])
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            evaluate_predictions([recording([], rallies=[[1, 3], [2, 4]])])
        with self.assertRaises(ValueError):
            evaluate_predictions([recording([])], primary_padding_seconds=4)
        with self.assertRaises(ValueError):
            evaluate_predictions([recording([])], join_gap_seconds=-1)


if __name__ == "__main__":
    unittest.main()
