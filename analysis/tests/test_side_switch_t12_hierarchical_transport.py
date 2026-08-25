from __future__ import annotations

import unittest

from analysis.side_switch_t12_hierarchical_transport import select_reduction


class SideSwitchT12HierarchicalTransportTest(unittest.TestCase):
    def test_stable_pair_uses_exact_stable_reduction(self) -> None:
        reduction = select_reduction(
            left_stable=True,
            right_stable=True,
            stable_reduction={"cost": 0.2, "coverage": 0.5, "conditionalSimilarity": 0.8},
            pooled_cost=0.7,
            left_pooled_available=True,
            right_pooled_available=True,
        )
        self.assertEqual(reduction.route, "stable-player-units")
        self.assertEqual(reduction.cost, 0.2)
        self.assertEqual(reduction.coverage, 0.5)

    def test_mixed_pair_uses_pooled_team_reduction(self) -> None:
        reduction = select_reduction(
            left_stable=True,
            right_stable=False,
            stable_reduction={"cost": 0.2, "coverage": 1.0, "conditionalSimilarity": 0.8},
            pooled_cost=0.35,
            left_pooled_available=True,
            right_pooled_available=True,
        )
        self.assertEqual(reduction.route, "pooled-team")
        self.assertAlmostEqual(reduction.cost, 0.35)
        self.assertAlmostEqual(reduction.coverage, 1.0)
        self.assertAlmostEqual(reduction.conditional_similarity, 0.65)

    def test_unavailable_pooled_pair_has_zero_support(self) -> None:
        reduction = select_reduction(
            left_stable=False,
            right_stable=False,
            stable_reduction={"cost": 0.2, "coverage": 1.0, "conditionalSimilarity": 0.8},
            pooled_cost=1.0,
            left_pooled_available=True,
            right_pooled_available=False,
        )
        self.assertEqual(reduction.route, "pooled-team")
        self.assertEqual(reduction.cost, 1.0)
        self.assertEqual(reduction.coverage, 0.0)
        self.assertEqual(reduction.conditional_similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
