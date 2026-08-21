from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from analysis.production_serve_gate import (
    ProductionRallyInterval,
    anchor_evidence,
    dual_head_prediction,
    hybrid_gate_prediction,
    load_production_serve_head,
    merge_production_rally_intervals,
    production_rally_anchor_evidence,
)
from analysis.serve import ServeDetection


ROOT = Path(__file__).resolve().parents[2]


class ProductionServeGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v2 = load_production_serve_head(
            ROOT / "prod/public/runtime/model-1ca43e38eefc.json"
        )
        self.previous = load_production_serve_head(
            ROOT / "prod/public/runtime/model-9c92b8e9333f.json"
        )

    def test_anchor_evidence_uses_peak_inside_source_aligned_window(self) -> None:
        evidence = anchor_evidence(
            self.v2,
            np.asarray([8.75, 9.5, 10.0, 10.75, 11.25]),
            np.asarray([0.99, 0.4, 0.8, 0.9, 0.95], dtype=np.float32),
            [ServeDetection(time=10.25, confidence=0.9)],
            10.0,
        )
        self.assertAlmostEqual(evidence.peak_probability, 0.9, places=6)
        self.assertEqual(evidence.peak_time, 10.75)
        self.assertEqual(evidence.nearest_detection, ServeDetection(10.25, 0.9))

    def test_either_production_head_can_mark_a_candidate_as_serve(self) -> None:
        times = np.asarray([5.0])
        v2 = anchor_evidence(
            self.v2,
            times,
            np.asarray([self.v2.decoder.threshold - 0.01], dtype=np.float32),
            [],
            5.0,
        )
        previous = anchor_evidence(
            self.previous,
            times,
            np.asarray(
                [self.previous.decoder.threshold + 0.01], dtype=np.float32
            ),
            [],
            5.0,
        )
        self.assertEqual(dual_head_prediction([v2, previous]), "serve")
        previous_below = anchor_evidence(
            self.previous,
            times,
            np.asarray(
                [self.previous.decoder.threshold - 0.01], dtype=np.float32
            ),
            [],
            5.0,
        )
        self.assertEqual(dual_head_prediction([v2, previous_below]), "not-serve")

    def test_production_rally_merge_preserves_agreement_and_touching_cuts(self) -> None:
        merged = merge_production_rally_intervals(
            [
                {"start": 10.0, "end": 14.0, "included": True},
                {"start": 20.0, "end": 22.0, "included": True},
            ],
            [
                {"start": 12.0, "end": 16.0, "included": True},
                {"start": 16.0, "end": 18.0, "included": True},
                {"start": 22.0, "end": 24.0, "included": True},
            ],
        )
        self.assertEqual(
            merged,
            (
                ProductionRallyInterval(10.0, 16.0, "both-models"),
                ProductionRallyInterval(16.0, 18.0, "previous-production-only"),
                ProductionRallyInterval(20.0, 22.0, "all-labels-v2-only"),
                ProductionRallyInterval(22.0, 24.0, "previous-production-only"),
            ),
        )

    def test_hybrid_gate_recovers_only_both_model_anchor_containment(self) -> None:
        times = np.asarray([5.0])
        below = [
            anchor_evidence(
                head,
                times,
                np.asarray([head.decoder.threshold - 0.01], dtype=np.float32),
                [],
                5.0,
            )
            for head in (self.v2, self.previous)
        ]
        both = production_rally_anchor_evidence(
            [ProductionRallyInterval(4.0, 8.0, "both-models")], 5.0
        )
        self.assertEqual(
            hybrid_gate_prediction(below, both),
            ("serve", "production-rally-recovery", True),
        )
        one_model = production_rally_anchor_evidence(
            [ProductionRallyInterval(4.0, 8.0, "all-labels-v2-only")], 5.0
        )
        self.assertEqual(
            hybrid_gate_prediction(below, one_model),
            ("not-serve", "none", False),
        )
        outside = production_rally_anchor_evidence(
            [ProductionRallyInterval(6.0, 8.0, "both-models")], 5.0
        )
        self.assertEqual(
            hybrid_gate_prediction(below, outside),
            ("not-serve", "none", False),
        )


if __name__ == "__main__":
    unittest.main()
