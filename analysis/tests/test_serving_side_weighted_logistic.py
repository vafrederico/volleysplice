from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_weighted_logistic import (
    fit_weighted_logistic,
    select_weighted_threshold,
    weighted_binary_metrics,
    weighted_brier_score,
)


class ServingSideWeightedLogisticTests(unittest.TestCase):
    def test_weighted_fit_separates_a_simple_problem(self) -> None:
        values = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        weights = np.asarray([1.0, 5.0, 2.0, 1.0])
        model = fit_weighted_logistic(values, labels, weights, l2=0.1)
        probabilities = model.predict_proba(values)
        self.assertTrue(np.all(probabilities[:2] < 0.5))
        self.assertTrue(np.all(probabilities[2:] > 0.5))

    def test_metrics_use_population_weights(self) -> None:
        metrics = weighted_binary_metrics(
            np.asarray([1, 1, 0, 0]),
            np.asarray([True, False, True, False]),
            np.asarray([4.0, 1.0, 2.0, 3.0]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.7)
        self.assertAlmostEqual(metrics["positiveRecall"], 0.8)
        self.assertAlmostEqual(metrics["negativeRecall"], 0.6)
        self.assertAlmostEqual(metrics["balancedAccuracy"], 0.7)

    def test_threshold_and_brier_are_finite(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        weights = np.ones(4)
        selected = select_weighted_threshold(labels, scores, weights)
        self.assertEqual(selected["balancedAccuracy"], 1.0)
        self.assertAlmostEqual(weighted_brier_score(labels, scores, weights), 0.085)


if __name__ == "__main__":
    unittest.main()
