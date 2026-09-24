import unittest
import numpy as np

from analysis.neural_generalization_gap_stage import pts_selection
from analysis.mobile_visual_features import sample_selection


class SourceGapTests(unittest.TestCase):
    def test_normal_nearest_sampling_is_unchanged(self):
        pts = np.arange(600, dtype=np.float64)/60.
        for hz in (2., 4.):
            actual, indexes = pts_selection(pts, 9.9, hz)
            old, old_indexes = sample_selection(pts, 9.9, hz)
            np.testing.assert_array_equal(actual, old)
            np.testing.assert_array_equal(indexes, old_indexes)

    def test_bounded_gap_records_repeat_and_even_mobile_subset(self):
        pts = np.array([0., .25, .49, .99, 1.25, 1.49])
        times, indexes = pts_selection(pts, 1.5, 4.)
        np.testing.assert_array_equal(indexes, [0, 1, 2, 3, 3, 4])
        mobile_times, mobile_indexes = pts_selection(pts, 1.5, 2.)
        np.testing.assert_array_equal(mobile_times, times[::2])
        np.testing.assert_array_equal(mobile_indexes, indexes[::2])

    def test_larger_gap_fails_closed(self):
        with self.assertRaisesRegex(ValueError, '250ms'):
            pts_selection(np.array([0., .25, 1.49]), 1.5, 4.)


if __name__ == '__main__':
    unittest.main()
