import base64
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from analysis.features import feature_names

PATH = Path(__file__).resolve().parents[2]/'scripts/prepare-labeling-neural-review.py'
SPEC = importlib.util.spec_from_file_location('labeling_neural_review', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def encoded(values, dtype):
    values = np.asarray(values, dtype='<f8' if dtype == 'float64' else '<f4')
    return {'encoding': 'base64', 'byteOrder': 'little-endian', 'dataType': dtype,
            'shape': list(values.shape), 'data': base64.b64encode(values.tobytes()).decode()}


def feedback():
    return {'source': {'timelineCoordinates': 'seconds-from-start-of-source',
        'media': {'duration': 1., 'width': 1920, 'height': 1080, 'hasAudio': True}},
        'features': {'rows': 4, 'columns': 104, 'analysisFps': 4,
         'names': list(feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))),
         'timestamps': encoded([0, .24996, .50004, .74998], 'float64'),
         'values': encoded(np.arange(416).reshape(4, 104), 'float32')}}


class ReviewInputTests(unittest.TestCase):
    def test_native_timestamps_and_raw_features_preserved(self):
        result = MODULE.feature_input(feedback())
        self.assertEqual(result.times.dtype, np.float64)
        self.assertEqual(result.values.dtype, np.float32)
        self.assertEqual(result.times.tolist(), [0, .24996, .50004, .74998])
        self.assertEqual(result.values[3, 103], 415.)

    def test_human_changes_cannot_affect_features(self):
        raw = feedback()
        before = MODULE.feature_input(raw)
        raw.update(corrections={'rallies': object(), 'ignoredIntervals': object()},
                   finalExportIntervals=object(), finalExportProvenance=object(),
                   initialInference={'ranges': object(), 'servingSide': object()})
        raw['source']['gameWindow'] = {'start': .9, 'end': .91}
        after = MODULE.feature_input(raw)
        np.testing.assert_array_equal(before.times, after.times)
        np.testing.assert_array_equal(before.values, after.values)

    def test_feature_names_must_match_training_schema(self):
        raw = feedback()
        raw['features']['names'][0] = 'wrong-channel'
        with self.assertRaisesRegex(ValueError, 'signature'):
            MODULE.feature_input(raw)

    def test_nonfinite_values_rejected(self):
        raw = feedback()
        values = np.zeros((4, 104), np.float32)
        values[1, 4] = np.nan
        raw['features']['values'] = encoded(values, 'float32')
        with self.assertRaisesRegex(ValueError, 'numeric data'):
            MODULE.feature_input(raw)

    def test_duplicate_timestamps_rejected(self):
        raw = feedback()
        raw['features']['timestamps'] = encoded([0, .25, .25, .75], 'float64')
        with self.assertRaisesRegex(ValueError, 'timeline'):
            MODULE.feature_input(raw)

    def test_operational_ignored_only_explicit_metadata(self):
        self.assertEqual(MODULE.validate_ignored([{'start': 0, 'end': .5, 'reason': 'opening'}], 1),
                         [{'start': 0., 'end': .5}])
        with self.assertRaisesRegex(ValueError, 'ignored'):
            MODULE.validate_ignored([{'start': 0, 'end': 2}], 1)


if __name__ == '__main__':
    unittest.main()
