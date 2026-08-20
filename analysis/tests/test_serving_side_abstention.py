from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_abstention import (
    select_precision_operating_point,
    selective_metrics,
)


class ServingSideAbstentionTests(unittest.TestCase):
    def test_selective_metrics_counts_abstentions_outside_recall(self) -> None:
        metrics = selective_metrics(
            [0, 1, 0, 1],
            [0.1, 0.4, 0.6, 0.9],
            far_threshold=0.3,
            near_threshold=0.8,
        )

        self.assertEqual(metrics["predictedFar"], 1)
        self.assertEqual(metrics["predictedNear"], 1)
        self.assertEqual(metrics["abstained"], 2)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["nearPrecision"], 1.0)
        self.assertEqual(metrics["farPrecision"], 1.0)
        self.assertEqual(metrics["nearRecall"], 0.5)
        self.assertEqual(metrics["farRecall"], 0.5)

    def test_selector_abstains_without_flipping_frozen_decisions(self) -> None:
        truth = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])

        selected = select_precision_operating_point(
            truth,
            scores,
            center_threshold=0.5,
            precision_floor=1.0,
            minimum_predictions_per_side=1,
        )

        assert selected is not None
        self.assertLessEqual(selected["farThreshold"], 0.5)
        self.assertGreaterEqual(selected["nearThreshold"], 0.5)
        self.assertEqual(selected["coverage"], 0.5)
        self.assertEqual(selected["nearPrecision"], 1.0)
        self.assertEqual(selected["farPrecision"], 1.0)

    def test_selector_returns_none_when_support_requirement_is_impossible(self) -> None:
        self.assertIsNone(
            select_precision_operating_point(
                [0, 1],
                [0.1, 0.9],
                center_threshold=0.5,
                precision_floor=0.9,
                minimum_predictions_per_side=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
