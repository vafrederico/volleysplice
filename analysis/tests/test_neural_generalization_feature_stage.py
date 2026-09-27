from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_generalization_feature_stage as stage


class GeneralizationStagingTests(unittest.TestCase):
    def test_mobile_grid_selects_exact_even_dino_frames_on_vfr_media(self):
        pts = np.cumsum(np.r_[0., np.tile([.031, .036, .032, .034], 30)])
        times4, frames4 = stage.pts_selection(pts, float(pts[-1]), 4.)
        times2, frames2 = stage.pts_selection(pts, float(pts[-1]), 2.)
        self.assertTrue(np.array_equal(times4[::2], times2))
        self.assertTrue(np.array_equal(frames4[::2], frames2))
        self.assertLessEqual(np.max(abs(pts[frames4]-times4)), .125)

    def test_audio_origin_aligns_same_pcm_before_existing_feature_formula(self):
        pcm = np.asarray([.5, -.5, .25], np.float32)
        config = SimpleNamespace(audio_sample_rate=16000)
        probes = [SimpleNamespace(stdout='{"streams":[{"start_time":"0.000125","sample_rate":"16000"}]}'),
                  SimpleNamespace(stdout='{"frames":[{"best_effort_timestamp_time":"0.000125"}]}')]
        observed = {}
        def formula(values, times, cfg, *, available):
            observed.update(values=values.copy(), available=available)
            return np.zeros((len(times), 3), np.float32)
        with patch.object(stage.av, '_decode_audio_samples', return_value=(pcm, True)), \
                patch.object(stage.subprocess, 'run', side_effect=probes), \
                patch.object(stage.av, '_audio_features_from_samples', side_effect=formula):
            _, metadata = stage.audio_on_grid('video', None, np.asarray([0., .25]), config)
        self.assertTrue(np.array_equal(observed['values'], np.r_[np.zeros(2, np.float32), pcm]))
        self.assertEqual(metadata['alignmentSamples'], 2)

    def test_audio_missing_does_not_invent_a_stream_origin(self):
        config = SimpleNamespace(audio_sample_rate=16000)
        with patch.object(stage.av, '_decode_audio_samples', return_value=(np.zeros(0, np.float32), False)), \
                patch.object(stage.subprocess, 'run', side_effect=AssertionError('No audio stream should be probed')), \
                patch.object(stage.av, '_audio_features_from_samples', return_value=np.zeros((1, 3), np.float32)):
            _, metadata = stage.audio_on_grid('video', None, np.asarray([0.]), config)
        self.assertFalse(metadata['available'])
        self.assertEqual(metadata['alignmentSamples'], 0)


if __name__ == '__main__':
    unittest.main()
