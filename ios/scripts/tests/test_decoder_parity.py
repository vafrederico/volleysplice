import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('parity', Path(__file__).parents[1] / 'compare-decoder-parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


class DecoderParityTests(unittest.TestCase):
    def test_same_counts_different_inference_does_not_pass(self):
        result = parity.compare({'p': [0.3, 0.7], 'rallies': [[1, 4]]},
                                {'p': [0.31, 0.69], 'rallies': [[1, 4]]})
        self.assertFalse(result['equal'])
        self.assertEqual(result['numericDifferences'], 2)

    def test_same_probabilities_different_boundary_does_not_pass(self):
        self.assertFalse(parity.compare({'rallies': [[1, 4]]}, {'rallies': [[1, 4.25]]})['equal'])

    def test_nan_is_not_parity(self):
        self.assertFalse(parity.compare([float('nan')], [float('nan')])['equal'])

    def test_missing_field_is_not_null(self):
        self.assertFalse(parity.compare(parity.MISSING, None)['equal'])

    def test_dictionary_order_and_float_representation(self):
        self.assertTrue(parity.compare({'p': [1, 0.5], 'label': 'rally'},
                                       {'label': 'rally', 'p': [1.0, 0.5]})['equal'])


if __name__ == '__main__': unittest.main()
