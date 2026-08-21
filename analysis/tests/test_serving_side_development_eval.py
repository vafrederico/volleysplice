from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_development_eval import (
    candidate_rank,
    tied_recording_ranks,
)


class ServingSideDevelopmentEvalTests(unittest.TestCase):
    def test_tied_ranks_are_local_to_each_recording(self) -> None:
        rows = [
            {"recordingId": "a"},
            {"recordingId": "a"},
            {"recordingId": "a"},
            {"recordingId": "b"},
        ]
        values = np.asarray([[1.0], [2.0], [2.0], [99.0]])
        ranked = tied_recording_ranks(rows, values)
        np.testing.assert_allclose(ranked[:, 0], [0.0, 0.75, 0.75, 0.5])

    def test_candidate_rank_uses_the_declared_guardrail_order(self) -> None:
        candidate = {
            "featureCount": 12,
            "evaluation": {
                "sourceGroupMacroBalancedAccuracy": 0.9,
                "pooledMetrics": {"balancedAccuracy": 0.8, "macroF1": 0.7},
                "worstSourceGroupBalancedAccuracy": 0.6,
            },
        }
        self.assertEqual(candidate_rank(candidate), (0.9, 0.8, 0.7, 0.6, -12))


if __name__ == "__main__":
    unittest.main()
