from __future__ import annotations

import unittest

from analysis.side_switch_t20_compact_candidate import compact_candidate_features


class SideSwitchT20CompactCandidateTest(unittest.TestCase):
    def test_copies_exact_t14_and_t19_sources(self) -> None:
        row = {
            "kind": "adjacent-rally-boundary",
            "features": {
                "dominantTrackletJerseyTeamTransportSwapMargin": -0.125,
                "crossRepresentationJerseySwapDisagreement": 0.375,
            },
        }
        self.assertEqual(
            compact_candidate_features(row),
            {
                "compactMedoidJerseySwapMargin": -0.125,
                "compactCrossRepresentationJerseyDisagreement": 0.375,
            },
        )

    def test_rejects_internal_candidate(self) -> None:
        with self.assertRaises(ValueError):
            compact_candidate_features({"kind": "internal-dead-state-peak"})


if __name__ == "__main__":
    unittest.main()
