import unittest
from unittest.mock import patch

import numpy as np

from analysis.neural_development import Example
from analysis import neural_recall_sweep_precision as precision
from analysis.schema import Interval


class Cache(dict):
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False


class HistoricalPrecisionTests(unittest.TestCase):
    def example(self):
        return Example('record', 'source', 1., np.asarray([.125, .375]), np.ones((2, 104), np.float32),
                       np.zeros((2, 4), np.float32), np.asarray([False, True]), (Interval(.2, .5),),
                       (Interval(0., .2),), 'indoor')

    def test_historical_nearest_tie_uses_left_and_preserves_labels_av_validity(self):
        example = self.example()
        cache = Cache(timestamps=np.asarray([0., .25, .5]),
                      tokens=np.broadcast_to(np.arange(3, dtype=np.float16)[:, None, None], (3, 10, 384)))
        with patch.object(precision, 'bind', return_value='bound-cache'), patch.object(precision.np, 'load', return_value=cache):
            result = precision.attach(example, {'path': 'unused', 'sha256': 'unused'})
        self.assertEqual(result.values.shape, (2, 3944))
        np.testing.assert_array_equal(result.values[:, :104], example.values)
        np.testing.assert_array_equal(result.values[:, 104:], np.broadcast_to(np.arange(2)[:, None], (2, 3840)))
        self.assertIs(result.valid, example.valid)
        self.assertEqual(result.truth, example.truth)
        self.assertEqual(result.ignored, example.ignored)

    def test_alignment_gaps_fail_before_precision_comparison(self):
        cache = Cache(timestamps=np.asarray([1., 1.25]), tokens=np.zeros((2, 10, 384), np.float16))
        with patch.object(precision, 'bind', return_value='bound-cache'), patch.object(precision.np, 'load', return_value=cache):
            with self.assertRaises(ValueError):
                precision.attach(self.example(), {'path': 'unused', 'sha256': 'unused'})


if __name__ == '__main__':
    unittest.main()
