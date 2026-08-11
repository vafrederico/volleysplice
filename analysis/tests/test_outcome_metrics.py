from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.crop_evaluation import RecordingIntervals, evaluate_crop_padding
from analysis.metrics import (
    aggregate_evaluations,
    evaluate_intervals,
    outcome_slice_metrics,
    truth_slice_metrics,
)
from analysis.schema import Interval, load_manifest


class ManifestOutcomeTagTests(unittest.TestCase):
    def test_manifest_preserves_rally_outcome_and_provenance_tags(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-outcome-manifest-") as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "name": "outcome-tag-test",
                        "annotationPolicy": {
                            "id": "serve-contact-to-dead-ball-v1"
                        },
                        "recordings": [
                            {
                                "id": "match-1",
                                "video": "match-1.mp4",
                                "split": "train",
                                "sourceGroup": "source-1",
                                "environment": "beach",
                                "consent": {"analyze": True, "train": True},
                                "rallies": [
                                    {
                                        "start": 1.0,
                                        "end": 3.0,
                                        "tags": [
                                            "ace",
                                            "ai-prelabel",
                                            "serve-confidence:high",
                                            "ace",
                                        ],
                                    },
                                    {
                                        "start": 5.0,
                                        "end": 7.0,
                                        "tags": [
                                            "service-fault",
                                            "end-confidence:medium",
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_manifest(manifest_path, require_videos=False)

        rallies = manifest.recordings[0].rallies
        self.assertEqual(
            rallies[0].tags,
            ("ace", "ai-prelabel", "serve-confidence:high"),
        )
        self.assertEqual(
            rallies[0].to_dict()["tags"],
            ["ace", "ai-prelabel", "serve-confidence:high"],
        )
        self.assertEqual(
            rallies[1].tags,
            ("service-fault", "end-confidence:medium"),
        )


class OutcomeSliceTests(unittest.TestCase):
    def test_truth_coverage_unions_fragments_from_multiple_predictions(self) -> None:
        result = truth_slice_metrics(
            (Interval(0.0, 10.0),),
            (Interval(0.0, 4.0), Interval(5.0, 10.0)),
            (0,),
        )

        self.assertAlmostEqual(result["meanCoverage"], 0.9)

    def test_outcome_slices_ignore_provenance_tags_and_include_three_second_events(self) -> None:
        truth = (
            Interval(
                0.0,
                2.0,
                tags=("ace", "ai-prelabel", "serve-confidence:high"),
            ),
            Interval(
                5.0,
                7.0,
                tags=("service-fault", "ai-prelabel"),
            ),
            Interval(
                10.0,
                13.0,
                tags=("ai-prelabel", "end-confidence:high"),
            ),
            Interval(
                20.0,
                23.001,
                tags=("ai-prelabel", "serve-confidence:medium"),
            ),
        )
        predictions = (
            Interval(0.0, 2.0),
            Interval(10.0, 13.0),
            Interval(20.0, 23.001),
        )

        slices = outcome_slice_metrics(truth, predictions)

        self.assertEqual(slices["all"]["rallies"], 4)
        self.assertEqual(slices["ace"]["rallies"], 1)
        self.assertEqual(slices["ace"]["strictMatchRecall"], 1.0)
        self.assertEqual(slices["serviceFault"]["rallies"], 1)
        self.assertEqual(slices["serviceFault"]["anyOverlapRecall"], 0.0)
        self.assertEqual(slices["shortAtMost3Seconds"]["rallies"], 3)
        self.assertAlmostEqual(
            slices["shortAtMost3Seconds"]["strictMatchRecall"],
            2 / 3,
        )
        self.assertEqual(slices["ordinaryLong"]["rallies"], 1)
        self.assertEqual(slices["ordinaryLong"]["strictMatchRecall"], 1.0)

    def test_padding_sweep_reports_outcome_coverage_containment_and_crop_merges(self) -> None:
        recording = RecordingIntervals(
            id="padding-outcomes",
            split="test",
            duration=30.0,
            truth=(
                Interval(5.0, 7.0, tags=("ace", "ai-prelabel")),
                Interval(11.0, 14.0, tags=("service-fault",)),
                Interval(20.0, 25.0, tags=("ai-prelabel",)),
            ),
            predictions=(
                Interval(6.0, 6.5),
                Interval(12.5, 13.0),
                Interval(22.5, 23.0),
            ),
        )

        rows = evaluate_crop_padding((recording,), (0, 1, 2, 3))

        self.assertEqual(
            [row["paddingSecondsBeforeAndAfter"] for row in rows],
            [0.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual([row["inputCropCount"] for row in rows], [3, 3, 3, 3])
        self.assertEqual([row["outputCropCount"] for row in rows], [3, 3, 3, 2])
        self.assertEqual([row["cropMergeRate"] for row in rows[:3]], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(rows[3]["cropMergeRate"], 1 / 3)

        outcomes = [row["aggregate"]["outcomeSlices"] for row in rows]
        self.assertEqual(
            [row["ace"]["fullyContainedRate"] for row in outcomes],
            [0.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(
            [row["serviceFault"]["fullyContainedRate"] for row in outcomes],
            [0.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(
            [row["ordinaryLong"]["fullyContainedRate"] for row in outcomes],
            [0.0, 0.0, 0.0, 1.0],
        )
        for actual, expected in zip(
            [row["ace"]["meanCoverage"] for row in outcomes],
            (0.25, 1.0, 1.0, 1.0),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            [row["serviceFault"]["meanCoverage"] for row in outcomes],
            (1 / 6, 5 / 6, 1.0, 1.0),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            [row["ordinaryLong"]["meanCoverage"] for row in outcomes],
            (0.1, 0.5, 0.9, 1.0),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)


class AggregateBoundaryMetricTests(unittest.TestCase):
    def test_aggregate_reports_iou_threshold_f1_and_subsecond_boundary_rates(self) -> None:
        first = evaluate_intervals(
            (Interval(0.0, 4.0), Interval(10.0, 14.0)),
            (Interval(0.25, 4.25), Interval(10.5, 14.5)),
        )
        second = evaluate_intervals(
            (Interval(20.0, 24.0), Interval(30.0, 34.0)),
            (
                Interval(21.0, 25.0),
                Interval(32.0, 36.0),
                Interval(40.0, 42.0),
            ),
        )

        aggregate = aggregate_evaluations((first, second))

        self.assertEqual(aggregate["trueRallies"], 4)
        self.assertEqual(aggregate["predictedRallies"], 5)
        self.assertEqual(aggregate["matchedRalliesAtIou03"], 4)
        self.assertEqual(aggregate["matchedRallies"], 3)
        self.assertEqual(aggregate["matchedRalliesAtIou07"], 2)
        self.assertAlmostEqual(aggregate["eventF1AtIou03"], 8 / 9)
        self.assertAlmostEqual(aggregate["eventF1"], 6 / 9)
        self.assertAlmostEqual(aggregate["eventF1AtIou07"], 4 / 9)
        self.assertAlmostEqual(aggregate["startBoundaryMaeSeconds"], 7 / 12)
        self.assertAlmostEqual(aggregate["endBoundaryMaeSeconds"], 7 / 12)
        self.assertAlmostEqual(aggregate["startBoundaryP50Seconds"], 0.5)
        self.assertAlmostEqual(aggregate["endBoundaryP90Seconds"], 0.9)
        self.assertAlmostEqual(aggregate["boundariesWithin025SecondRate"], 1 / 3)
        self.assertAlmostEqual(aggregate["boundariesWithin05SecondRate"], 2 / 3)
        self.assertEqual(aggregate["boundariesWithin1SecondRate"], 1.0)
        self.assertEqual(aggregate["boundariesWithin2SecondsRate"], 1.0)


if __name__ == "__main__":
    unittest.main()
