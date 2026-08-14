from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from analysis.crop_evaluation import RecordingIntervals
from analysis.schema import Interval


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate-browser-parity-padding.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_browser_parity_padding", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class BrowserParityPaddingEvaluatorTests(unittest.TestCase):
    def test_scope_reports_canonical_metrics_and_actual_evaluable_counts(self) -> None:
        recording = RecordingIntervals(
            id="sample",
            split="validation",
            duration=20,
            truth=(Interval(5, 10),),
            predictions=(Interval(6, 9),),
            ignored_intervals=(Interval(7, 8),),
        )

        report = EVALUATOR.evaluate_scope(
            [recording], list(EVALUATOR.REQUIRED_PADDING_SECONDS)
        )

        self.assertEqual(
            set(report), {"pad-0s", "pad-1s", "pad-2s", "pad-3s"}
        )
        padded = report["pad-1s"]
        self.assertEqual(padded["fullyContainedCoreRallies"], 2)
        self.assertEqual(padded["expectedCoreRallies"], 2)
        self.assertEqual(padded["inputPredictionCount"], 1)
        self.assertEqual(padded["actualMergedExportSections"], 1)
        self.assertEqual(padded["evaluableExportFragments"], 2)
        self.assertEqual(
            padded["cropCounts"],
            {
                "inputPredictionRanges": 1,
                "actualMergedExportSections": 1,
                "evaluableExportFragments": 2,
            },
        )
        self.assertAlmostEqual(padded["adjustedMetrics"]["P_pad"], 1.0)
        self.assertAlmostEqual(padded["adjustedMetrics"]["R_core"], 1.0)
        self.assertAlmostEqual(
            padded["adjustedMetrics"]["F1_padP_coreR"], 1.0
        )
        self.assertEqual(
            padded["adjustedMetrics"]["paddedModelExportSeconds"], 4
        )
        self.assertEqual(
            padded["adjustedMetrics"]["paddedHumanExportSeconds"], 6
        )
        self.assertEqual(
            padded["adjustedMetrics"]["exportDurationDifferenceSeconds"], -2
        )
        per_recording = padded["perRecording"][0]
        self.assertEqual(per_recording["id"], "sample")
        self.assertEqual(per_recording["eventMetricsAtIou05"]["truePositives"], 2)
        self.assertEqual(per_recording["evaluableExportFragments"], 2)

    def test_lift_summary_compares_swr_to_linear_and_offline_at_two_and_three(
        self,
    ) -> None:
        def evaluate(predictions: tuple[Interval, ...]):
            recording = RecordingIntervals(
                id="sample",
                split="validation",
                duration=20,
                truth=(Interval(5, 10),),
                predictions=predictions,
            )
            return {
                "validation2": EVALUATOR.evaluate_scope(
                    [recording], list(EVALUATOR.REQUIRED_PADDING_SECONDS)
                )
            }

        reports = {
            "offline": evaluate((Interval(9, 10),)),
            "browserOnDevice": evaluate((Interval(8, 9),)),
            EVALUATOR.LIBSWRESAMPLE_WASM_VARIANT: evaluate((Interval(5, 10),)),
        }

        lifts = EVALUATOR.build_lift_summaries(
            reports,
            candidate_variant=EVALUATOR.LIBSWRESAMPLE_WASM_VARIANT,
            reference_variants=("browserOnDevice", "offline"),
        )

        pad_two = lifts["byScope"]["validation2"]["pad-2s"]
        self.assertEqual(
            pad_two["candidate"]["fullyContainedCoreRallies"], 1
        )
        self.assertEqual(
            pad_two["versus"]["browserOnDevice"][
                "fullyContainedCoreRallies"
            ],
            1,
        )
        self.assertEqual(
            pad_two["versus"]["offline"]["fullyContainedCoreRallies"], 1
        )
        self.assertGreater(
            pad_two["versus"]["browserOnDevice"]["F1_padP_coreRPoints"],
            0,
        )
        self.assertEqual(
            pad_two["perRecording"][0]["versus"]["browserOnDevice"][
                "fullyContainedCoreRallies"
            ],
            1,
        )
        pad_three = lifts["byScope"]["validation2"]["pad-3s"]
        self.assertEqual(
            pad_three["versus"]["browserOnDevice"][
                "fullyContainedCoreRallies"
            ],
            0,
        )
        self.assertEqual(
            pad_three["versus"]["offline"]["fullyContainedCoreRallies"], 1
        )


if __name__ == "__main__":
    unittest.main()
