from __future__ import annotations

import unittest

from analysis.side_switch_t16_source_resolved import source_resolved_features


class SideSwitchT16SourceResolvedTest(unittest.TestCase):
    def test_preserves_signed_source_advantages_exactly(self) -> None:
        row = {
            "kind": "adjacent-rally-boundary",
            "t15BilateralMedoidConsensus": {
                "sourceAdvantages": {"near": -0.25, "far": 0.125}
            },
        }
        self.assertEqual(
            source_resolved_features(row),
            {
                "sourceResolvedMedoidJerseyNearSwapAdvantage": -0.25,
                "sourceResolvedMedoidJerseyFarSwapAdvantage": 0.125,
            },
        )

    def test_rejects_internal_candidate(self) -> None:
        with self.assertRaises(ValueError):
            source_resolved_features({"kind": "internal-dead-state-peak"})


if __name__ == "__main__":
    unittest.main()
