from __future__ import annotations

import unittest

from analysis.side_switch_t17_relative_contrast import (
    relative_contrast,
    relative_source_features,
)


class SideSwitchT17RelativeContrastTest(unittest.TestCase):
    def test_relative_contrast_is_signed_and_scale_free(self) -> None:
        self.assertAlmostEqual(relative_contrast(0.6, 0.2), 0.5)
        self.assertAlmostEqual(relative_contrast(0.2, 0.6), -0.5)
        self.assertAlmostEqual(relative_contrast(0.0, 0.0), 0.0)

    def test_uses_each_source_cost_pair(self) -> None:
        row = {
            "kind": "adjacent-rally-boundary",
            "t14DominantTrackletMedoid": {
                "teamReductions": {
                    "nearNear": {"cost": 0.6},
                    "nearFar": {"cost": 0.2},
                    "farFar": {"cost": 0.1},
                    "farNear": {"cost": 0.3},
                }
            },
        }
        values = relative_source_features(row)
        self.assertAlmostEqual(values["relativeMedoidJerseyNearSwapContrast"], 0.5)
        self.assertAlmostEqual(values["relativeMedoidJerseyFarSwapContrast"], -0.5)


if __name__ == "__main__":
    unittest.main()
