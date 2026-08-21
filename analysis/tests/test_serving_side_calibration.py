import unittest

import numpy as np

from analysis.serving_side_calibration import (
    calibration_metrics,
    fit_platt,
    logit,
)


class ServingSideCalibrationTests(unittest.TestCase):
    def test_platt_reduces_overconfidence_without_changing_order(self) -> None:
        probabilities = np.asarray([0.01, 0.05, 0.20, 0.80, 0.95, 0.99])
        truth = np.asarray([0, 1, 0, 1, 0, 1])
        model = fit_platt(probabilities, truth, l2=0.1)
        calibrated = model.predict(probabilities)

        self.assertTrue(np.all(np.diff(calibrated) > 0))
        self.assertLess(
            calibration_metrics(truth, calibrated)["brier"],
            calibration_metrics(truth, probabilities)["brier"],
        )
        self.assertEqual(model.to_dict()["l2"], 0.1)

    def test_calibration_metrics_report_reliability_bins(self) -> None:
        metrics = calibration_metrics(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], bins=2
        )
        self.assertAlmostEqual(metrics["brier"], 0.025)
        self.assertAlmostEqual(metrics["expectedCalibrationError"], 0.15)
        self.assertEqual([item["rows"] for item in metrics["bins"]], [2, 2])

    def test_invalid_calibration_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            logit([])
        with self.assertRaises(ValueError):
            fit_platt([0.2, 0.8], [0, 2], l2=0.1)
        with self.assertRaises(ValueError):
            calibration_metrics([0, 1], [0.2, 1.2])


if __name__ == "__main__":
    unittest.main()
