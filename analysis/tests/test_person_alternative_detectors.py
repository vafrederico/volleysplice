import unittest

import numpy as np

from analysis.person_alternative_detectors import sanitize_boxes, box_geometry_diagnostics


class AlternativeDetectorContracts(unittest.TestCase):
    def test_clip_reject_degenerate_preserve_score_order(self):
        boxes, uncapped = sanitize_boxes(
            [[-1, -4, 30, 60], [1, 1, 2, 2], [11, 12, 50, 60], [70, 0, 80, 10]],
            [.5, .249, .8, .9], 40, 40)
        self.assertEqual(uncapped, 2)
        self.assertEqual(boxes[0]["box"], [11., 12., 40., 40.])
        self.assertEqual(boxes[1]["box"], [0., 0., 30., 40.])

    def test_cap_reports_uncapped_observations(self):
        boxes, count = sanitize_boxes([[i, 0, i + 1, 2] for i in range(30)], [.5] * 30, 50, 50)
        self.assertEqual(len(boxes), 24)
        self.assertEqual(count, 30)
        self.assertEqual(boxes[0]["box"], [0., 0., 1., 2.])

    def test_nonfinite_and_mismatch_are_errors(self):
        with self.assertRaises(ValueError):
            sanitize_boxes([[0, 0, np.nan, 4]], [.5], 10, 10)
        with self.assertRaises(ValueError):
            sanitize_boxes([[0, 0, 4, 4]], [], 10, 10)

    def test_containment_is_flagged_without_calling_it_identity(self):
        rows = [{"box": [0, 0, 10, 10]}, {"box": [2, 2, 8, 8]}]
        result = box_geometry_diagnostics(rows, 20, 20)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["highContainmentPairs"][0]["containment"], 1)
        self.assertAlmostEqual(result["highContainmentPairs"][0]["iou"], .36)
        self.assertFalse(result["imageHalfIsCourtGeometry"])

    def test_empty_detector_output_is_valid(self):
        boxes, count = sanitize_boxes([], [], 100, 100)
        self.assertEqual((boxes, count), ([], 0))
        result = box_geometry_diagnostics(boxes, 100, 100)
        self.assertEqual(result["highContainmentPairs"], [])
        self.assertIsNone(result["minimumBoxHeightPixels"])


if __name__ == "__main__":
    unittest.main()
