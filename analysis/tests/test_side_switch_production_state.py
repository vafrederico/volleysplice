from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_production_state import (
    PRODUCTION_STATE_FEATURE_NAMES,
    ProductionTrace,
    ScoredTime,
    TimeRange,
    gap_state_features,
    merge_production_components,
    rally_evidence,
)


class SideSwitchProductionStateTests(unittest.TestCase):
    def trace(self) -> ProductionTrace:
        times = np.arange(0.0, 30.0, 0.25)
        first_rally = ((times >= 2.0) & (times < 7.0)).astype(np.float64)
        second_rally = ((times >= 12.0) & (times < 18.0)).astype(np.float64)
        rally = np.maximum(first_rally, second_rally) * 0.8 + 0.1
        dead = np.where((times >= 7.0) & (times < 12.0), 0.9, 0.2)
        return ProductionTrace(
            times=times,
            duration=30.0,
            rally_scores={
                "all-labels-v2": rally,
                "previous-production": rally * 0.95,
            },
            serve_scores={
                "all-labels-v2": np.full_like(times, 0.1),
                "previous-production": np.full_like(times, 0.1),
            },
            dead_state_scores={
                "all-labels-v2": dead,
                "previous-production": dead * 0.95,
            },
            ranges={
                "all-labels-v2": (TimeRange(2.0, 7.0), TimeRange(12.0, 18.0)),
                "previous-production": (
                    TimeRange(2.25, 6.75),
                    TimeRange(12.25, 17.5),
                ),
            },
            serves={
                "all-labels-v2": (ScoredTime(2.0, 0.95), ScoredTime(12.0, 0.9)),
                "previous-production": (
                    ScoredTime(2.25, 0.85),
                    ScoredTime(12.25, 0.8),
                ),
            },
            suppression_scores=np.where(
                (times >= 7.0) & (times < 12.0), 0.8, 0.1
            ),
        )

    def test_overlap_components_preserve_support_and_touching_cut(self) -> None:
        components = merge_production_components(
            {
                "all-labels-v2": (TimeRange(1.0, 3.0), TimeRange(5.0, 6.0)),
                "previous-production": (
                    TimeRange(2.0, 4.0),
                    TimeRange(6.0, 7.0),
                ),
            }
        )
        self.assertEqual(
            [(item.start, item.end, item.support_count) for item in components],
            [(1.0, 4.0, 2), (5.0, 6.0, 1), (6.0, 7.0, 1)],
        )

    def test_serve_consensus_grounds_pre_contact_window(self) -> None:
        evidence = rally_evidence(TimeRange(1.9, 7.1), self.trace())
        self.assertEqual(evidence.support_count, 2)
        self.assertEqual(evidence.serve_support_count, 2)
        self.assertEqual(evidence.anchor_source, "serve-consensus")
        self.assertGreater(evidence.anchor_time, 2.0)
        self.assertLess(evidence.anchor_time, 2.25)
        self.assertAlmostEqual(evidence.comparison_end - evidence.comparison_start, 2.0)
        self.assertLess(evidence.comparison_start, evidence.anchor_time)

    def test_missing_component_uses_explicit_source_fallback(self) -> None:
        evidence = rally_evidence(TimeRange(22.0, 24.0), self.trace())
        self.assertEqual(evidence.support_count, 0)
        self.assertEqual(evidence.anchor_source, "source-start-fallback")
        self.assertEqual(evidence.anchor_time, 22.0)

    def test_gap_features_have_frozen_order_and_quarantine_suppression(self) -> None:
        trace = self.trace()
        before = rally_evidence(TimeRange(1.9, 7.1), trace)
        after = rally_evidence(TimeRange(11.9, 18.1), trace)
        features, suppression = gap_state_features(before, after, 7.0, 12.0, trace)
        self.assertEqual(tuple(features), PRODUCTION_STATE_FEATURE_NAMES)
        self.assertNotIn("suppressionGapMeanScore", features)
        self.assertEqual(set(suppression), {
            "suppressionGapMeanScore",
            "suppressionGapPeakScore",
        })
        self.assertAlmostEqual(features["productionMinimumAdjacentSupportCount"], 2.0)
        self.assertAlmostEqual(features["productionGapLiveFraction"], 0.0)
        self.assertGreater(suppression["suppressionGapMeanScore"], 0.7)


if __name__ == "__main__":
    unittest.main()
