from __future__ import annotations

import unittest

from analysis.serving_side_control_cohort import build_correct_control_cohort


class ServingSideControlCohortTest(unittest.TestCase):
    def test_is_deterministic_stratified_and_excludes_errors(self) -> None:
        predictions = []
        for index in range(18):
            human = "near" if index % 2 else "far"
            prediction = human if index != 17 else "far"
            predictions.append(
                {
                    "rallyId": f"video:rally:{index}",
                    "recordingId": "video",
                    "environment": "indoor" if index < 9 else "grass",
                    "sourceGroup": "source-a" if index < 9 else "source-b",
                    "decision": human,
                    "prediction": prediction,
                    "probabilityNear": 0.95 if prediction == "near" else 0.05,
                    "correct": prediction == human,
                }
            )
        first = build_correct_control_cohort(
            predictions, experiment_sha256="a" * 64, target_rows=8
        )
        second = build_correct_control_cohort(
            list(reversed(predictions)), experiment_sha256="a" * 64, target_rows=8
        )
        self.assertEqual(first, second)
        self.assertEqual(first["sampling"]["populationRows"], 17)
        self.assertEqual(first["sampling"]["sampledRows"], 8)
        self.assertEqual(len(first["sampling"]["strata"]), 4)
        self.assertNotIn("video:rally:17", {row["rallyId"] for row in first["rows"]})
        self.assertTrue(all(row["sampledRows"] >= 1 for row in first["sampling"]["strata"]))

    def test_rejects_a_target_too_small_to_cover_strata(self) -> None:
        predictions = [
            {
                "rallyId": f"video:rally:{index}",
                "environment": "indoor",
                "sourceGroup": f"source-{index}",
                "decision": "near",
                "prediction": "near",
                "probabilityNear": 0.95,
                "correct": True,
            }
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            build_correct_control_cohort(
                predictions, experiment_sha256="b" * 64, target_rows=2
            )


if __name__ == "__main__":
    unittest.main()

