from __future__ import annotations

import unittest

from analysis.side_switch_t19_cross_representation import cross_representation_features


class SideSwitchT19CrossRepresentationTest(unittest.TestCase):
    def test_equal_weight_consensus_and_absolute_disagreement(self) -> None:
        row = {
            "kind": "adjacent-rally-boundary",
            "features": {
                "selectiveFarJerseyTeamTransportSwapMargin": 0.3,
                "dominantTrackletJerseyTeamTransportSwapMargin": -0.1,
            },
        }
        values = cross_representation_features(row)
        self.assertAlmostEqual(
            values["crossRepresentationJerseySwapConsensus"], 0.1
        )
        self.assertAlmostEqual(
            values["crossRepresentationJerseySwapDisagreement"], 0.4
        )


if __name__ == "__main__":
    unittest.main()
