from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/audit-neural-repair-content.py'
SPEC = importlib.util.spec_from_file_location('independent_repair_content_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Capture:
    def __init__(self, pts):
        self.pts, self.index, self.released = pts, -1, False

    def read(self):
        self.index += 1
        return (True, np.full((2, 2, 3), self.index, dtype=np.uint8)) if self.index < len(self.pts) else (False, None)

    def get(self, _):
        return self.pts[self.index]*1000

    def release(self):
        self.released = True


class RepairContentTests(unittest.TestCase):
    def test_sequential_selection_uses_true_pts_and_earlier_tie_stops_at_prefix(self):
        cap = Capture([0, .125, .375, .49, .76, 1, 1.25, 1.5, 1.75, 2])
        rows, decoded = audit.nearest_from_sequential(cap, SimpleNamespace(CAP_PROP_POS_MSEC=0))
        self.assertEqual([row[0] for row in rows], [0, 1, 3, 4, 5, 6, 7, 8])
        self.assertEqual(decoded, 9)
        self.assertTrue(cap.released)
        self.assertTrue(all(np.all(row[2] == row[0]) for row in rows))

    def test_bad_timeline_or_incomplete_prefix_is_rejected_and_closed(self):
        for pts, pattern in (([0, .1, .1], 'strictly increasing'), ([0, .3], 'ended'), ([.01], 'origin'), ([0, .6], 'half-tick')):
            cap = Capture(pts)
            with self.assertRaisesRegex(ValueError, pattern):
                audit.nearest_from_sequential(cap, SimpleNamespace(CAP_PROP_POS_MSEC=0))
            self.assertTrue(cap.released)

    def test_bit_exact_comparison_rejects_dtype_sign_bit_and_values(self):
        values = np.zeros((2, 3), np.float32)
        self.assertTrue(audit.compare_arrays(values, values.copy(), 'same')['bitExact'])
        for changed in (values.astype(np.float64), -values, values+np.float32(1e-7)):
            with self.assertRaisesRegex(ValueError, 'not bit exact'):
                audit.compare_arrays(values, changed, 'changed')

    def test_prefix_audio_uses_decoded_origin_and_declared_sample_shift(self):
        config = SimpleNamespace(audio_sample_rate=16000)
        selection = {'audio': {'stream': {'sample_rate': '48000'}, 'startOffsetSeconds': .02, 'alignmentSamples': 320}}
        pcm = np.arange(36000, dtype=np.int16)
        probe = SimpleNamespace(stdout=json.dumps({'frames': [{'best_effort_timestamp_time': '0.020000'}]}))
        ffmpeg = SimpleNamespace(stdout=pcm.tobytes())
        with patch.object(audit.subprocess, 'run', side_effect=[probe, ffmpeg]) as run:
            samples, metadata = audit.prefix_audio(Path('/video'), selection, config)
        np.testing.assert_array_equal(samples[:320], 0)
        np.testing.assert_array_equal(samples[320:], pcm.astype(np.float32)/np.float32(32768))
        self.assertEqual(metadata['shiftSamples'], 320)
        self.assertIn('2.25', run.call_args_list[1].args[0])
        selection['audio']['alignmentSamples'] = 319
        with patch.object(audit.subprocess, 'run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'alignment differs'):
                audit.prefix_audio(Path('/video'), selection, config)

    def test_causal_audio_prefix_matches_whole_audio_independently_of_later_content(self):
        from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
        from analysis import features
        config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
        rng = np.random.default_rng(101)
        samples = (rng.normal(size=80000)*.1).astype(np.float32)
        prefix = features._audio_features_from_samples(samples[:36000], audit.TICKS, config, available=True)
        whole = features._audio_features_from_samples(samples, audit.TICKS, config, available=True)
        names = features._audio_feature_names(config)
        columns = [i for i, name in enumerate(names) if name not in audit.GLOBAL_AUDIO]
        audit.compare_arrays(prefix[:, columns], whole[:, columns], 'prefix-vs-whole-causal-audio')
        excluded = [names.index(name) for name in audit.GLOBAL_AUDIO]
        self.assertFalse(np.array_equal(prefix[:, excluded], whole[:, excluded]))

    def test_prepare_pins_source_and_scope_without_reading_cache_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root/'frozen.py'
            frozen.write_text('frozen bytes', encoding='utf-8')
            repair = {'recordingIds': [f'video-{n}' for n in range(7)], 'sourceCode': {'frozen.py': audit.identity(frozen)}}
            (root/'pts-repair-plan-v1.json').write_text(json.dumps(repair), encoding='utf-8')
            (root/'extraction-plan.json').write_text('{}', encoding='utf-8')
            output = root/'audit'
            with patch.object(audit.np, 'load', side_effect=AssertionError('must not read cache values')):
                audit.prepare(root, output, root)
            plan = audit.read(output/'plan.json')
            self.assertEqual(plan['ticksPerRecord'], 8)
            self.assertEqual(plan['gridTicksSeconds'], [0, .25, .5, .75, 1, 1.25, 1.5, 1.75])
            self.assertEqual(audit.identity(SCRIPT)['sha256'], plan['sourceSnapshot']['sha256'])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                audit.prepare(root, output, root)


if __name__ == '__main__':
    unittest.main()
