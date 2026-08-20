from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_specialist import (
    ReviewedEvent,
    SideSwitchSpecialistError,
    SpecialistModel,
    event_vector,
    fit_specialist,
    grouped_cross_fit,
    matrix_for,
    reviewed_events,
    select_threshold,
)


def _event(recording_id: str, index: int, label: int, value: float) -> ReviewedEvent:
    event_id = f"{recording_id}:{index}"
    payload = {
        "eventId": event_id,
        "recordingId": recording_id,
        "features": {"playerPaletteArea": value},
    }
    return ReviewedEvent(
        event_id=event_id,
        recording_id=recording_id,
        environment="grass",
        split="train",
        source_group=f"source-{recording_id}",
        source_type="test",
        target_status="gold",
        decision="switch" if label else "no-switch",
        label=label,
        event=payload,
    )


class SideSwitchSpecialistTests(unittest.TestCase):
    def test_event_vector_includes_explicit_missing_indicators(self) -> None:
        vector = event_vector({"features": {}}, "palette-area")

        self.assertTrue(np.isnan(vector[0]))
        self.assertEqual(vector[1], 1.0)

    def test_fit_separates_simple_switch_signal_and_round_trips(self) -> None:
        rows = [
            _event("a", 0, 0, 0.05),
            _event("a", 1, 0, 0.10),
            _event("b", 0, 1, 0.90),
            _event("b", 1, 1, 0.95),
        ]
        model = fit_specialist(rows, "palette-area", 0.1)
        scores = model.predict_proba(matrix_for(rows, "palette-area"))
        restored = SpecialistModel.from_dict(model.to_dict())

        self.assertLess(max(scores[:2]), min(scores[2:]))
        np.testing.assert_allclose(
            restored.predict_proba(matrix_for(rows, "palette-area")), scores
        )

    def test_model_rejects_feature_names_that_do_not_match_feature_set(self) -> None:
        rows = [_event("a", 0, 0, 0.1), _event("a", 1, 1, 0.9)]
        payload = fit_specialist(rows, "palette-area", 0.1).to_dict()
        payload["featureNames"] = ["invented", "invented:missing"]

        with self.assertRaisesRegex(SideSwitchSpecialistError, "feature names"):
            SpecialistModel.from_dict(payload)

    def test_grouped_cross_fit_scores_every_held_recording(self) -> None:
        rows = []
        for recording_id in ("a", "b", "c"):
            rows.extend(
                [
                    _event(recording_id, 0, 0, 0.05),
                    _event(recording_id, 1, 0, 0.15),
                    _event(recording_id, 2, 1, 0.85),
                    _event(recording_id, 3, 1, 0.95),
                ]
            )

        scores, audit = grouped_cross_fit(rows, "palette-area", 0.1)

        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(
            {fold["heldRecordingId"] for fold in audit["folds"]}, {"a", "b", "c"}
        )
        self.assertEqual(audit["selectedThresholdMetrics"]["f1"], 1.0)

    def test_threshold_selection_uses_precision_as_f1_tie_break(self) -> None:
        selected = select_threshold(
            np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.8, 0.7, 0.1])
        )

        self.assertAlmostEqual(selected["threshold"], 0.7)
        self.assertEqual(selected["truePositives"], 2)
        self.assertEqual(selected["falsePositives"], 1)

    def test_reviewed_events_validates_identity_and_excludes_unclear(self) -> None:
        report = {
            "kind": "appearance",
            "createdAt": "today",
            "events": [
                {"eventId": "one", "recordingId": "r1"},
                {"eventId": "two", "recordingId": "r1"},
            ],
        }
        decisions = {
            "reportKind": "appearance",
            "reportCreatedAt": "today",
            "decisions": {"one": "switch", "two": "unclear"},
        }

        rows, counts = reviewed_events(report, decisions)

        self.assertEqual([row.event_id for row in rows], ["one"])
        self.assertEqual(counts["switch"], 1)
        self.assertEqual(counts["unclear"], 1)
        with self.assertRaisesRegex(SideSwitchSpecialistError, "different"):
            reviewed_events(report, {**decisions, "reportCreatedAt": "yesterday"})


if __name__ == "__main__":
    unittest.main()
