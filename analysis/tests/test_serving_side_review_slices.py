from __future__ import annotations

import unittest

from analysis.serving_side_review_slices import (
    build_review_slice_report,
    rescore_reviewed_predictions,
)


def annotation(visibility: str) -> dict[str, object]:
    return {
        "serverVisibility": visibility,
        "contactTiming": "on-anchor",
        "correctedServeAnchorSeconds": None,
        "ballFlightVisibility": "visible",
        "motionDirection": "matches-human-side",
        "notes": "",
        "reviewedAt": "2026-08-21T00:00:00Z",
    }


class ServingSideReviewSlicesTest(unittest.TestCase):
    def test_error_census_and_controls_recover_population(self) -> None:
        predictions = [
            {
                "rallyId": f"video:rally:{index}",
                "recordingId": "video",
                "environment": "indoor",
                "sourceGroup": "source",
                "decision": "near",
                "prediction": "far" if index == 0 else "near",
                "probabilityNear": 0.2 if index == 0 else 0.95,
                "correct": index != 0,
            }
            for index in range(5)
        ]
        annotations = {
            "video:rally:0": annotation("offscreen"),
            "video:rally:1": annotation("visible"),
            "video:rally:2": annotation("partial"),
        }
        cohort = [
            {
                "rallyId": "video:rally:1",
                "stratumKey": "indoor|source|near|high",
            },
            {
                "rallyId": "video:rally:2",
                "stratumKey": "indoor|source|near|high",
            },
        ]
        report = build_review_slice_report(predictions, annotations, cohort, {})
        self.assertEqual(report["counts"]["evaluationRows"], 5)
        self.assertEqual(report["counts"]["reviewedRows"], 3)
        self.assertAlmostEqual(report["overall"]["estimatedPopulationRows"], 5)
        self.assertAlmostEqual(report["overall"]["accuracy"], 0.8)
        self.assertAlmostEqual(
            report["slices"]["serverVisibility"]["offscreen"]["accuracy"], 0
        )

    def test_not_serve_correction_removes_a_control_and_reweights_stratum(self) -> None:
        predictions = [
            {
                "rallyId": f"video:rally:{index}",
                "recordingId": "video",
                "environment": "indoor",
                "sourceGroup": "source",
                "decision": "far",
                "prediction": "far",
                "probabilityNear": 0.05,
                "correct": True,
            }
            for index in range(3)
        ]
        annotations = {"video:rally:1": annotation("visible")}
        cohort = [
            {"rallyId": "video:rally:0", "stratumKey": "indoor|source|far|high"},
            {"rallyId": "video:rally:1", "stratumKey": "indoor|source|far|high"},
        ]
        report = build_review_slice_report(
            predictions,
            annotations,
            cohort,
            {"video:rally:0": "not-serve"},
        )
        self.assertEqual(report["counts"]["evaluationRows"], 2)
        self.assertEqual(report["counts"]["reviewedControls"], 1)
        self.assertAlmostEqual(report["reviewedPredictions"][0]["weight"], 2)

    def test_rescore_preserves_frozen_review_weights(self) -> None:
        reviewed = [
            {
                "rallyId": "video:rally:1",
                "human": "near",
                "prediction": "near",
                "weight": 3.0,
                "annotation": annotation("visible"),
            },
            {
                "rallyId": "video:rally:2",
                "human": "far",
                "prediction": "near",
                "weight": 1.0,
                "annotation": annotation("offscreen"),
            },
        ]
        rescored = rescore_reviewed_predictions(
            reviewed,
            [
                {"rallyId": "video:rally:1", "prediction": "far"},
                {"rallyId": "video:rally:2", "prediction": "far"},
            ],
        )
        self.assertEqual(rescored["overall"]["estimatedPopulationRows"], 4)
        self.assertEqual(rescored["overall"]["estimatedCorrectRows"], 1)
        self.assertEqual(
            rescored["slices"]["serverVisibility"]["offscreen"]["accuracy"], 1
        )


if __name__ == "__main__":
    unittest.main()
