from __future__ import annotations

import unittest

from analysis.side_switch_feature_development import (
    BASELINE_PROFILE,
    BASE_FEATURE_NAMES,
    apply_profile,
    metric_delta,
)


class SideSwitchFeatureDevelopmentTest(unittest.TestCase):
    def _row(self):
        return {
            "eventId": "video:boundary:R001:R002",
            "recordingId": "video",
            "kind": "adjacent-rally-boundary",
            "gapStart": 10.0,
            "gapEnd": 12.0,
            "transitionTime": 11.0,
            "score": 0.0,
            "features": {name: float(index) for index, name in enumerate(BASE_FEATURE_NAMES[:-2])},
        }

    def test_baseline_profile_materializes_candidate_metadata(self):
        row = apply_profile([self._row()], BASELINE_PROFILE)[0]
        self.assertEqual(row["features"]["candidateIsInternalDeadStatePeak"], 0.0)
        self.assertEqual(row["features"]["candidateGeneratorScore"], 0.0)
        self.assertEqual(tuple(BASELINE_PROFILE.feature_names), BASE_FEATURE_NAMES)

    def test_apply_profile_does_not_mutate_source(self):
        source = self._row()
        apply_profile([source], BASELINE_PROFILE)
        self.assertNotIn("candidateIsInternalDeadStatePeak", source["features"])

    def test_metric_delta_uses_candidate_minus_baseline(self):
        baseline = {
            "rowAveragePrecision": 0.4,
            "strict": {"f1": 0.3},
            "primary": {
                "precision": 0.5,
                "recall": 0.6,
                "f1": 0.55,
                "proposals": 10,
                "truePositives": 6,
                "falsePositives": 4,
                "falseNegatives": 4,
            },
        }
        candidate = {
            "rowAveragePrecision": 0.45,
            "strict": {"f1": 0.32},
            "primary": {
                "precision": 0.6,
                "recall": 0.7,
                "f1": 0.65,
                "proposals": 9,
                "truePositives": 7,
                "falsePositives": 2,
                "falseNegatives": 3,
            },
        }
        delta = metric_delta(baseline, candidate)
        self.assertAlmostEqual(delta["primaryF1"], 0.10)
        self.assertEqual(delta["proposals"], -1)
        self.assertEqual(delta["truePositives"], 1)
        self.assertEqual(delta["falsePositives"], -2)


if __name__ == "__main__":
    unittest.main()
