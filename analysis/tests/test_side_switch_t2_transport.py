from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t2_transport import (
    CONDITIONAL_FEATURE_NAME,
    T2_CORE_FEATURE_NAMES,
    directional_conditional_similarity,
    spearman,
    t2_features,
)


def _reduction(coverage: float, mass: float) -> dict[str, float]:
    return {"coverage": coverage, "matchedIdentityMass": mass}


class SideSwitchT2TransportTest(unittest.TestCase):
    def test_zero_coverage_maps_to_exact_zero(self) -> None:
        self.assertEqual(directional_conditional_similarity(_reduction(0.0, 0.0)), 0.0)

    def test_conditional_similarity_removes_coverage_scale(self) -> None:
        self.assertAlmostEqual(
            directional_conditional_similarity(_reduction(0.5, 0.4)), 0.8
        )
        self.assertAlmostEqual(
            directional_conditional_similarity(_reduction(0.25, 0.2)), 0.8
        )

    def test_inconsistent_zero_coverage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            directional_conditional_similarity(_reduction(0.0, 0.1))

    def test_t2_uses_minimum_directional_similarity(self) -> None:
        row = {
            "kind": "adjacent-rally-boundary",
            "features": {"appearanceTransportSwapMargin": 0.15, "transportCoverageMinimum": 0.25},
            "t1Transport": {
                "status": "ok",
                "reductions": {
                    "nearFar": _reduction(0.5, 0.4),
                    "farNear": _reduction(0.25, 0.1),
                },
            },
        }
        values, diagnostics = t2_features(row)
        self.assertEqual(tuple(values), T2_CORE_FEATURE_NAMES)
        self.assertEqual(values["appearanceTransportSwapMargin"], 0.15)
        self.assertAlmostEqual(values[CONDITIONAL_FEATURE_NAME], 0.4)
        self.assertAlmostEqual(
            diagnostics["nearBeforeToFarAfterConditionalSimilarity"], 0.8
        )

    def test_spearman_handles_ties_and_constant_vectors(self) -> None:
        self.assertAlmostEqual(
            spearman(np.asarray([1.0, 1.0, 2.0]), np.asarray([2.0, 2.0, 4.0])),
            1.0,
        )
        self.assertIsNone(
            spearman(np.ones(3), np.asarray([1.0, 2.0, 3.0]))
        )


if __name__ == "__main__":
    unittest.main()
