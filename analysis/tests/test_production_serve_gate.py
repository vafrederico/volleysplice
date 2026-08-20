from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from analysis.production_serve_gate import (
    anchor_evidence,
    dual_head_prediction,
    load_production_serve_head,
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


if __name__ == "__main__":
    unittest.main()
