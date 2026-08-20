from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_specialist import (
    ReviewedRally,
    ServingSideModel,
    ServingSideSpecialistError,
    binary_metrics,
    fit_specialist,
    grouped_cross_fit,
    matrix_for,
    rally_vector,
    reviewed_rallies,
    select_threshold,
)


def _rally(
    recording_id: str, index: int, label: int, value: float
) -> ReviewedRally:
    rally_id = f"{recording_id}:rally:{index}"
    payload = {
        "rallyId": rally_id,
        "recordingId": recording_id,
        "features": {
            "pixelMotionMargin": value,
            "paletteChangeMargin": value,
        },
    }
    return ReviewedRally(
        rally_id=rally_id,
        recording_id=recording_id,
        environment="grass",
        split="train",
        source_group=f"source-{recording_id}",
        source_type="test",
        target_status="gold",
        decision="near" if label else "far",
        label=label,
        rally=payload,
    )


class ServingSideSpecialistTests(unittest.TestCase):
    def test_rally_vector_includes_explicit_missing_indicators(self) -> None:
        vector = rally_vector({"features": {}}, "motion-palette")

        self.assertTrue(np.isnan(vector[0]))
        self.assertEqual(vector[1], 1.0)
        self.assertTrue(np.isnan(vector[2]))
        self.assertEqual(vector[3], 1.0)

    def test_fit_separates_simple_side_signal_and_round_trips(self) -> None:
        rows = [
            _rally("a", 0, 0, -0.95),
            _rally("a", 1, 0, -0.75),
            _rally("b", 0, 1, 0.75),
            _rally("b", 1, 1, 0.95),
        ]
        model = fit_specialist(rows, "motion-palette", 0.1)
        scores = model.predict_proba(matrix_for(rows, "motion-palette"))
        restored = ServingSideModel.from_dict(model.to_dict())

        self.assertLess(max(scores[:2]), min(scores[2:]))
        np.testing.assert_allclose(
            restored.predict_proba(matrix_for(rows, "motion-palette")), scores
        )

    def test_model_rejects_a_changed_decision_mapping(self) -> None:
        rows = [_rally("a", 0, 0, -0.5), _rally("a", 1, 1, 0.5)]
        payload = fit_specialist(rows, "motion-palette", 0.1).to_dict()
        payload["positiveDecision"] = "far"

        with self.assertRaisesRegex(ServingSideSpecialistError, "mapping"):
            ServingSideModel.from_dict(payload)

    def test_grouped_cross_fit_scores_every_held_recording(self) -> None:
        rows = []
        for recording_id in ("a", "b", "c"):
            rows.extend(
                [
                    _rally(recording_id, 0, 0, -0.9),
                    _rally(recording_id, 1, 0, -0.7),
                    _rally(recording_id, 2, 1, 0.7),
                    _rally(recording_id, 3, 1, 0.9),
                ]
            )

        scores, audit = grouped_cross_fit(rows, "motion-palette", 0.1)

        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(
            {fold["heldRecordingId"] for fold in audit["folds"]},
            {"a", "b", "c"},
        )
        self.assertEqual(
            audit["selectedThresholdMetrics"]["balancedAccuracy"], 1.0
        )

    def test_threshold_selection_treats_near_and_far_symmetrically(self) -> None:
        selected = select_threshold(
            np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.8, 0.7, 0.1])
        )

        self.assertAlmostEqual(selected["threshold"], 0.7)
        self.assertEqual(selected["balancedAccuracy"], 0.75)
        self.assertAlmostEqual(selected["macroF1"], 11 / 15)

    def test_binary_metrics_exposes_both_side_confusions(self) -> None:
        metrics = binary_metrics(
            np.asarray([1, 1, 0, 0]), np.asarray([True, False, True, False])
        )

        self.assertEqual(
            metrics["confusion"],
            {"near": {"near": 1, "far": 1}, "far": {"near": 1, "far": 1}},
        )
        self.assertEqual(metrics["balancedAccuracy"], 0.5)
        self.assertEqual(metrics["macroF1"], 0.5)

    def test_reviewed_rallies_validates_identity_and_excludes_unclear(self) -> None:
        report = {
            "kind": "serving-side",
            "createdAt": "today",
            "rallies": [
                {"rallyId": "one", "recordingId": "r1"},
                {"rallyId": "two", "recordingId": "r1"},
            ],
        }
        decisions = {
            "reportKind": "serving-side",
            "reportCreatedAt": "today",
            "decisions": {"one": "near", "two": "unclear"},
        }

        rows, counts = reviewed_rallies(report, decisions)

        self.assertEqual([row.rally_id for row in rows], ["one"])
        self.assertEqual(counts["near"], 1)
        self.assertEqual(counts["unclear"], 1)
        with self.assertRaisesRegex(ServingSideSpecialistError, "different"):
            reviewed_rallies(
                report, {**decisions, "reportCreatedAt": "yesterday"}
            )


if __name__ == "__main__":
    unittest.main()
