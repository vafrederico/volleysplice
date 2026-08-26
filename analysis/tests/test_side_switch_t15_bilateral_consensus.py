from __future__ import annotations

import unittest

from analysis.side_switch_t15_bilateral_consensus import bilateral_consensus_features


def _row(near_same: float, near_swap: float, far_same: float, far_swap: float):
    return {
        "kind": "adjacent-rally-boundary",
        "t14DominantTrackletMedoid": {
            "teamReductions": {
                "nearNear": {"cost": near_same},
                "nearFar": {"cost": near_swap},
                "farFar": {"cost": far_same},
                "farNear": {"cost": far_swap},
            }
        },
    }


class SideSwitchT15BilateralConsensusTest(unittest.TestCase):
    def test_both_sources_must_prefer_swap(self) -> None:
        values, diagnostic = bilateral_consensus_features(_row(0.7, 0.2, 0.6, 0.4))
        self.assertAlmostEqual(values["bilateralMedoidJerseySwapEvidence"], 0.2)
        self.assertEqual(values["bilateralMedoidJerseyContinuityEvidence"], 0.0)
        self.assertEqual(diagnostic["agreement"], "swap")

    def test_both_sources_must_prefer_continuity(self) -> None:
        values, diagnostic = bilateral_consensus_features(_row(0.2, 0.7, 0.4, 0.6))
        self.assertEqual(values["bilateralMedoidJerseySwapEvidence"], 0.0)
        self.assertAlmostEqual(values["bilateralMedoidJerseyContinuityEvidence"], 0.2)
        self.assertEqual(diagnostic["agreement"], "continuity")

    def test_disagreement_produces_no_directional_evidence(self) -> None:
        values, diagnostic = bilateral_consensus_features(_row(0.7, 0.2, 0.2, 0.6))
        self.assertEqual(values["bilateralMedoidJerseySwapEvidence"], 0.0)
        self.assertEqual(values["bilateralMedoidJerseyContinuityEvidence"], 0.0)
        self.assertEqual(diagnostic["agreement"], "disagree-or-tie")


if __name__ == "__main__":
    unittest.main()
