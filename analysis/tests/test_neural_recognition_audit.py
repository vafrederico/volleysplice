from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def auditor():
    path = Path(__file__).resolve().parents[2]/'scripts/audit-neural-recognition.py'
    spec = importlib.util.spec_from_file_location('recognition_independent_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecognitionIntervalAuditTests(unittest.TestCase):
    def test_strict_three_second_gap_ignored_subtraction_and_complete_losses(self):
        from analysis.neural_evaluation import evaluate_predictions
        from analysis.schema import Interval
        module = auditor()
        _, helper = module.helpers()
        rows = [{'id': 'one', 'sourceGroup': 'court', 'durationSeconds': 50.,
                 'rallies': [Interval(2, 4), Interval(7, 9), Interval(12, 14), Interval(25, 27), Interval(45, 48)],
                 'predictions': [Interval(2, 4), Interval(7, 9), Interval(12, 14), Interval(29, 30)],
                 'ignoredIntervals': [Interval(8, 8.5)]}]
        result = module.independent_metrics(rows, evaluate_predictions(rows), helper)
        self.assertEqual([row['paddingSecondsBeforeAndAfter'] for row in result['padding']], [0., 1., 2., 3.])
        self.assertEqual(helper.padded_union([(2, 4), (7, 9)], 50, 0), [(2, 4), (7, 9)])
        self.assertEqual(helper.padded_union([(2, 4), (6.999, 9)], 50, 0), [(2, 9)])
        record = helper.parse_record(module.serialized_rows(rows)[0])
        exported = helper.export_union(record, 'predictions', 2)
        self.assertTrue(any(end == 8 for _, end in exported))
        self.assertTrue(any(start == 8.5 for start, _ in exported))
        self.assertEqual(result['coverage']['primaryExportCoverage']['completeRallyLosses'], 2)
        self.assertEqual(result['shortCompleteRallyLosses'], 2)

    def test_changed_hash_bound_helper_is_rejected_before_loading(self):
        module = auditor()
        with patch.object(module, 'HELPERS', {'audit-neural-context-tensors.py': 'wrong'}):
            with self.assertRaisesRegex(ValueError, 'helper changed'):
                module.helpers()


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch is not installed')
class RecognitionTensorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from analysis import neural_expanded_development as expanded
        from analysis import neural_recognition_development as development
        from analysis.recognition_temporal_model import RecognitionConfig, model_metadata
        from analysis.tests.test_neural_expanded_development import example
        cls.module = auditor()
        cls.tensor_helper, cls.interval_helper = cls.module.helpers()
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        cls.benchmark = torch.backends.cudnn.benchmark
        cls.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        cls.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = RecognitionConfig()
        cls.groups = ['a', 'b', 'c', 'd']
        cls.data = {'exact': [expanded.exact_supervision(example(f'exact-{g}-{i}', g, 137, j*2+i))
                             for j, g in enumerate(cls.groups) for i in range(2)]}
        for tier in ('draft', 'coverage'):
            cls.data[tier] = [expanded.auxiliary_supervision(example(f'{tier}-{g}', g, 137, j+17), tier)
                             for j, g in enumerate((*cls.groups, 'aux-only'))]
        contract = {'config': asdict(cls.config), 'groups': cls.groups, 'population': {
            tier+'Rows': [row.example.id for row in cls.data[tier]] for tier in cls.module.TIERS},
            'cohort': 'reviewed_export', 'lossArm': 'short_boost', 'checkpointEpochs': [1, 2],
            'decoderCandidates': expanded.decoder_candidates(), 'model': model_metadata(cls.config)}
        cls.registration = {'sha256': cls.module.canonical(contract), 'contract': contract}
        cls.result = development.run_cell(cls.data, cls.registration, cls.root, 3407, 'cpu')

    @classmethod
    def tearDownClass(cls):
        import torch
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)
        torch.backends.cudnn.benchmark = cls.benchmark
        torch.backends.cudnn.allow_tf32 = cls.cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = cls.matmul_tf32

    def fit_audit(self, folder=None):
        return self.module.audit_fit(folder or self.root/'fits/3407/inner-0-1', self.data, {'a', 'b'},
            (1, 2), 3407, self.registration, self.config, self.tensor_helper)

    def test_complete_seed_rebuilds_selection_and_metrics_without_fitting(self):
        from analysis import neural_recognition_fit as fit
        with patch.object(fit, 'fit_model', side_effect=AssertionError('auditor fitted')):
            checked = self.module.audit_seed(self.root, self.data, self.registration, self.config, 3407,
                self.tensor_helper, self.interval_helper)
        self.assertEqual(len(checked['fits']), 10)
        self.assertEqual(len(checked['logicalViews']), 12)
        self.assertEqual(checked['selections'], self.result['selections'])
        self.assertEqual(len(checked['independent']['padding']), 4)
        self.assertTrue(all(fit['cpuReplay'] for fit in checked['fits']))
        self.assertGreater(sum(row['checkedTicks'] for fit in checked['fits'] for row in fit['cpuReplay']), 0)

    def test_group_leak_or_sampler_history_edit_is_rejected(self):
        path = self.root/'fits/3407/inner-0-1/completed.json'
        original = path.read_bytes()
        try:
            for change in ('membership', 'sampling'):
                meta = json.loads(original)
                if change == 'membership':
                    meta['auxiliaryGroups']['coverage'].append('a')
                else:
                    meta['history'][0]['exposureSha256']['exact'] = 'wrong'
                path.write_text(json.dumps(meta))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.fit_audit()
        finally:
            path.write_bytes(original)

    def test_forged_checkpoint_hash_does_not_hide_changed_fold_scaler_or_probabilities(self):
        folder = self.root/'fits/3407/inner-0-1'
        meta_path = folder/'completed.json'
        original_meta = meta_path.read_bytes()
        for stem in ('weights', 'predictions'):
            path = folder/f'{stem}-1.npz'
            original = path.read_bytes()
            try:
                with np.load(path, allow_pickle=False) as cache:
                    tensors = {name: cache[name].copy() for name in cache.files}
                if stem == 'weights':
                    tensors['mean'][0] += 1
                else:
                    key = next(iter(tensors))
                    tensors[key][0] = 1-tensors[key][0]
                np.savez_compressed(path, **tensors)
                meta = json.loads(original_meta)
                meta['artifacts'][path.name] = self.module.digest(path)
                meta_path.write_text(json.dumps(meta))
                with self.subTest(stem=stem), self.assertRaisesRegex(ValueError, 'scaler differs|CPU mismatch'):
                    self.fit_audit()
            finally:
                path.write_bytes(original)
                meta_path.write_bytes(original_meta)

    def test_reselection_rejects_an_outer_record_inserted_into_inner_view(self):
        contract = self.registration['contract']
        examples = [row.example for row in self.data['exact'] if row.example.group != 'a']
        probabilities = {epoch: {example.id: np.zeros((len(example.times), 4), np.float32)
                                for example in examples} for epoch in (1, 2)}
        probabilities[1]['exact-a-0'] = np.zeros((137, 4), np.float32)
        with self.assertRaisesRegex(ValueError, 'Logical view'):
            self.module.reselect(examples, probabilities, contract, self.interval_helper)

    def test_saved_selection_is_checked_against_reconstructed_candidates(self):
        path = self.root/'result-3407.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['selections'][0]['epoch'] = 99
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'saved result.selections'):
                self.module.audit_seed(self.root, self.data, self.registration, self.config, 3407,
                    self.tensor_helper, self.interval_helper)
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
