"""Small mutations exercising the independent numerical audit's rejection gates."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('numerical_audit', ROOT/'scripts/audit-neural-generalization-numerics.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class GeneralizationNumericalAuditTest(unittest.TestCase):
    def test_inference_requires_correct_weights_and_label_blind_owner(self):
        directory = Path('/nas/fit')
        task = {'path': '/nas/task.json', 'sha256': 'a'*64}
        panel_ref = {'path': '/nas/panel.json', 'sha256': 'b'*64}
        panel = {'manifest': {'path': '/nas/inputs.json', 'sha256': 'c'*64},
                 'features': {'path': '/nas/features.json', 'sha256': 'd'*64}}
        meta = {'artifacts': {f'weights-{e}.npz': str(e)*64 for e in audit.EPOCHS}}
        receipt = {'task': task, 'id': 'video', 'precision': 'fp32', **panel,
                   'labelsUsed': False, 'panel': panel_ref, 'ignoredLabelsUsedForContextOrDecoding': False,
                   'epochs': list(audit.EPOCHS), 'weights': {
                       str(e): {'path': str(directory/'temporal'/f'weights-{e}.npz'),
                                'sha256': meta['artifacts'][f'weights-{e}.npz']} for e in audit.EPOCHS}}
        def check(value):
            audit.inference_owner(value, task, panel_ref, panel, 'video', 'fp32', directory, meta)
        check(receipt)
        wrong = copy.deepcopy(receipt)
        wrong['weights']['15']['path'] = '/nas/another-fit/temporal/weights-15.npz'
        with self.assertRaisesRegex(ValueError, 'checkpoint owner'):
            check(wrong)
        for key in ('labelsUsed', 'ignoredLabelsUsedForContextOrDecoding'):
            wrong = copy.deepcopy(receipt); del wrong[key]
            with self.assertRaisesRegex(ValueError, 'owner or label use'):
                check(wrong)

    def test_score_schema_rejects_truncated_nonfinite_or_nonprobability_timeline(self):
        times = np.arange(260, dtype=np.float64)/4
        scores = np.full((len(times), 4), .5, np.float32)
        audit.score_schema(scores, times)
        mutations = [scores[:-1], scores.astype(np.float16)]
        for value in (np.nan, 1.01, -.01):
            changed = scores.copy(); changed[120, 2] = value; mutations.append(changed)
        for changed in mutations:
            with self.assertRaisesRegex(ValueError, 'probability array'):
                audit.score_schema(changed, times)

    def test_cpu_forward_detects_changed_saved_prediction_and_model_weight(self):
        import torch
        from analysis.neural_recognition_fit import predict
        torch.set_num_threads(2)
        old, _, _ = audit.helpers()
        config = audit.MODELS['av-tcn']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8)
            model = audit.model_for(config).cpu().eval()
        example = SimpleNamespace(id='fixture', values=np.random.default_rng(8).normal(size=(260, 104)).astype(np.float32),
                                  times=np.arange(260)/4, valid=np.ones(260, bool))
        mean, scale = np.zeros(104, np.float32), np.ones(104, np.float32)
        saved = predict(model, example, mean, scale, config, 'cpu')
        old.replay_sample(model, example, saved, mean, scale, config)
        changed = saved.copy(); changed[130, 0] = 1. if saved[130, 0] < .5 else 0.
        with self.assertRaisesRegex(ValueError, 'CPU mismatch'):
            old.replay_sample(model, example, changed, mean, scale, config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        with self.assertRaisesRegex(ValueError, 'CPU mismatch'):
            old.replay_sample(model, example, saved, mean, scale, config)

    def test_teaching_population_excludes_invalid_time_and_rejects_shifted_grid(self):
        _, _, student = audit.helpers()
        valid_times = np.arange(800)/4
        times = valid_times[::2]
        valid = np.ones(800, bool); valid[200:400] = False
        selected = student.independent_teaching_indexes(times, valid_times, valid)
        self.assertEqual(len(selected), 128)
        self.assertTrue(valid[np.arange(0, 800, 2)[selected]].all())
        self.assertFalse(np.array_equal(selected, student.independent_teaching_indexes(times, valid_times, np.ones(800, bool))))
        with self.assertRaisesRegex(ValueError, 'grids differ'):
            student.independent_teaching_indexes(times+1., valid_times, valid)


if __name__ == '__main__':
    unittest.main()
