from analysis.private_ledger import private_value
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_generalization_inputs as inputs


def row(tier='exact'):
    return {'id': 'video', 'sourceGroup': 'group', 'durationSeconds': 2., 'environment': 'grass',
            'contentSha256': 'a'*64, 'featureOrigin': 'opencv-av104', 'labelTier': tier,
            'scoringPolicy': inputs.POLICIES[tier], 'eligibleRoles': ['fit', 'evaluate', 'infer'],
            'protected': False, 'consent': {'train': True}, 'featureCaches': {},
            'rallies': [{'start': .5, 'end': 1.5}], 'ignoredIntervals': [{'start': 0., 'end': .25}],
            'annotation': {'continuousVideoReviewed': True}, 'gameWindow': {'start': .25, 'end': 1.75},
            'keepTargets': [{'start': .5, 'end': 1.5}],
            'targetContract': {'negativesOutsideKeepAuthorizedByFullManualReview': True}}


class GeneralizedInputTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(8, dtype=np.float64)/4
        self.values = np.arange(8*104, dtype=np.float32).reshape(8, 104)

    def test_roles_reject_eval_only_and_protected_fitting(self):
        source = row()
        source['eligibleRoles'] = ['infer', 'evaluate']
        with self.assertRaisesRegex(ValueError, 'eligible'):
            inputs.select_rows([source], 'fit', ['video'])
        source = row()
        source['protected'] = True
        with self.assertRaisesRegex(ValueError, 'Protected'):
            inputs.select_rows([source], 'fit', ['video'])
        self.assertEqual(inputs.select_rows([source], 'infer')[0]['id'], 'video')

    def test_inference_never_uses_labels_ignored_or_game_window(self):
        source = row('coverage')
        source.update(rallies='poison', ignoredIntervals='poison', gameWindow='poison', keepTargets='poison')
        document = {'kind': 'neural-generalization-inputs-v1', 'records': [source]}
        with patch.object(inputs, 'read', return_value=document), patch.object(inputs, 'load_av',
                return_value=(self.times, self.values, 2.)):
            e = inputs.load_inference_examples('unused')[0]
        self.assertTrue(e.valid.all())
        self.assertFalse(e.targets.any())
        self.assertEqual(e.truth, ())
        self.assertEqual(e.ignored, ())

    def test_gold_attachment_cannot_change_inference_mask_or_values(self):
        source = row()
        document = {'kind': 'neural-generalization-inputs-v1', 'records': [source]}
        with patch.object(inputs, 'read', return_value=document), patch.object(inputs, 'load_av',
                return_value=(self.times, self.values, 2.)):
            e = inputs.load_inference_examples('unused')[0]
            scored = inputs.make_examples_for_evaluation([e], 'unused')[0]
        self.assertIs(scored.values, e.values)
        self.assertIs(scored.valid, e.valid)
        self.assertTrue(scored.valid.all())
        self.assertEqual(len(scored.truth), 1)
        self.assertEqual(len(scored.ignored), 1)

    def test_coverage_masks_only_keep_head_and_game_window(self):
        source = row('coverage')
        document = {'kind': 'neural-generalization-inputs-v1', 'records': [source]}
        with patch.object(inputs, 'read', return_value=document), patch.object(inputs, 'load_av',
                return_value=(self.times, self.values, 2.)):
            item = inputs.load_data('unused')['coverage'][0]
        self.assertFalse(item.mask[:, :3].any())
        self.assertTrue(np.array_equal(item.example.valid, (self.times >= .25) & (self.times < 1.75)))
        self.assertTrue(np.array_equal(item.mask[:, 3].astype(bool), item.example.valid))

    def test_eval_policy_cannot_silently_call_export_labels_rally_core(self):
        source = row('coverage')
        document = {'kind': 'neural-generalization-inputs-v1', 'records': [source]}
        with patch.object(inputs, 'read', return_value=document), patch.object(inputs, 'load_av',
                return_value=(self.times, self.values, 2.)):
            e = inputs.load_inference_examples('unused')[0]
            with self.assertRaisesRegex(ValueError, 'policy'):
                inputs.make_examples_for_evaluation([e], 'unused')
            scored = inputs.make_examples_for_evaluation([e], 'unused', scoring_policy='export-coverage')[0]
        self.assertIs(scored.valid, e.valid)
        self.assertEqual(len(scored.ignored), 3)

    @unittest.skipUnless(Path('/mnt/freenas').is_mount(), 'NAS-backed filesystem identity test')
    def test_verified_memo_detects_mutation(self):
        root = Path(private_value('private-reference-0071'))
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path = Path(folder)/'cache.bin'
            path.write_bytes(b'one')
            reference = {'path': str(path), 'sha256': inputs.sha(path)}
            inputs.verified(reference)
            with patch.object(inputs, 'sha', side_effect=AssertionError('must reuse verified identity')):
                inputs.verified(reference)
            stat = path.stat()
            path.write_bytes(b'two')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError, 'immutable feature changed'):
                inputs.verified(reference)


if __name__ == '__main__':
    unittest.main()
