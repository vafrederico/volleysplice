from __future__ import annotations

import unittest

from analysis.side_switch_t18_representativeness import representativeness_features


class SideSwitchT18RepresentativenessTest(unittest.TestCase):
    def test_uses_worst_stable_medoid_similarity_and_exact_fallback(self) -> None:
        row = {
            "kind": "adjacent-rally-boundary",
            "features": {"dominantTrackletJerseyTeamTransportSwapMargin": 0.4},
            "t14DominantTrackletMedoid": {
                "before": {
                    "nearDominantKind": "stable-medoid",
                    "nearMedoidToPooledDistance": 0.2,
                    "farDominantKind": "pooled-fallback",
                    "farMedoidToPooledDistance": None,
                },
                "after": {
                    "nearDominantKind": "stable-medoid",
                    "nearMedoidToPooledDistance": 0.5,
                    "farDominantKind": "stable-medoid",
                    "farMedoidToPooledDistance": 0.1,
                },
            },
        }
        values = representativeness_features(row)
        self.assertAlmostEqual(values["representativeMedoidJerseyGateMinimum"], 0.5)
        self.assertAlmostEqual(values["representativeMedoidJerseyDistanceMaximum"], 0.5)
        self.assertAlmostEqual(values["representativeMedoidJerseySwapMargin"], 0.2)


if __name__ == "__main__":
    unittest.main()
