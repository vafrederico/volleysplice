from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis import neural_generalization_reuse as adapter

PATH = Path(__file__).resolve().parents[2]/'scripts/audit-neural-generalization-reuse.py'
SPEC = importlib.util.spec_from_file_location('reuse_audit_tests', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    records = [
        {'id': 'a', 'sourceGroup': 'one', 'labelTier': 'exact'},
        {'id': 'b', 'sourceGroup': 'two', 'labelTier': 'exact'},
        {'id': 'c', 'sourceGroup': 'three', 'labelTier': 'draft'},
        {'id': 'd', 'sourceGroup': 'four', 'labelTier': 'coverage'},
    ]
    task = {'taskId': 'first', 'registration': {'path': 'registration', 'sha256': 'r'},
        'registrationSha256': 'contract', 'manifest': {'path': 'manifest', 'sha256': 'm'},
        'features': {'path': 'features', 'sha256': 'f'}, 'model': 'av-tcn', 'seed': 3407,
        'epochs': [5, 15, 30, 60], 'lossArm': 'short_boost', 'trainIds': ['c', 'a', 'd'],
        'calibrationIds': ['b'], 'variant': 'medium', 'splitSeed': 1}
    configs = {'av-tcn': {'family': 'av', 'head': 'tcn'}}
    return task, records, configs


def test_recipe_and_first_owner_preserve_order_but_not_calibration(self):
    task, records, configs = fixture()
    expected = audit.recipe(task, records, configs)
    assert expected == adapter.training_recipe(task, records, configs)
    assert expected['tierOrderedIds'] == {'exact': ['a'], 'draft': ['c'], 'coverage': ['d']}
    assert expected['excludedStudentGroups'] == ['two']
    second = {**task, 'taskId': 'second', 'calibrationIds': [], 'variant': 'wide', 'splitSeed': 99}
    third = {**second, 'taskId': 'third', 'trainIds': ['a', 'c', 'd']}
    derived = audit.owners([task, second, third], {'m': records}, configs)
    assert [r[2] for r in derived] == ['first', 'first', 'third']
    assert derived == adapter.first_owners([task, second, third], records, configs)


def test_training_owner_cannot_cross_recipe_identity(self):
    task, records, configs = fixture()
    original = audit.io.canonical(audit.recipe(task, records, configs))
    for field, value in [('seed', 1729), ('registrationSha256', 'other'),
            ('features', {'path': 'features', 'sha256': 'different'}),
            ('manifest', {'path': 'manifest', 'sha256': 'different'}),
            ('registration', {'path': 'another-registration', 'sha256': 'r'})]:
        with self.subTest(field=field):
            altered = {**task, field: value}
            assert audit.io.canonical(audit.recipe(altered, records, configs)) != original


def test_inherited_training_history_and_bn_cannot_change(self):
    source = {'history': [{'epoch': 5, 'loss': .25}], 'scalerTrainIds': ['a'], 'bn': [0., 1.], 'validationIds': ['b']}
    target = {**source, 'validationIds': ['z'], 'trainingExecution': {'trainingPerformed': False}}
    allowed = {'validationIds', 'trainingExecution'}
    audit.check_inherited(source, target, allowed)
    for field in ('history', 'scalerTrainIds', 'bn'):
        changed = copy.deepcopy(target); changed[field] = []
        with self.assertRaisesRegex(ValueError, 'Inherited'):
            audit.check_inherited(source, changed, allowed)


class Archive(dict):
    @property
    def files(self):
        return list(self)


def test_shared_scores_require_exact_values_and_no_vacuous_positive_claim(self):
    values = np.full((3, 4), .5, np.float32)
    source = Archive(a=values)
    assert audit.shared_scores(source, Archive(a=values.copy(), b=values), ['a'], ['b', 'a']) == ['a']
    assert audit.shared_scores(source, Archive(b=values), ['a'], ['b']) == []
    changed = values.copy(); changed[1, 2] = np.nextafter(changed[1, 2], np.float32(1))
    with self.assertRaisesRegex(ValueError, 'Shared'):
        audit.shared_scores(source, Archive(a=changed), ['a'], ['a'])
    with self.assertRaisesRegex(ValueError, 'Shared'):
        audit.shared_scores(source, Archive(a=values.astype(np.float64)), ['a'], ['a'])
    with self.assertRaisesRegex(ValueError, 'population'):
        audit.shared_scores(source, Archive(a=values, extra=values), ['a'], ['a'])


def test_feature_archive_equivalence_is_dtype_and_value_exact(self):
    with tempfile.TemporaryDirectory() as temporary:
        left, right = Path(temporary)/'a.npz', Path(temporary)/'b.npz'
        values = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        np.savez(left, tokens=values, times=np.array([0., .5], np.float64))
        np.savez_compressed(right, tokens=values, times=np.array([0., .5], np.float64))
        assert left.read_bytes() != right.read_bytes()
        audit.exact_archives(left, right)
        np.savez_compressed(right, tokens=values.astype(np.float32), times=np.array([0., .5], np.float64))
        with self.assertRaisesRegex(ValueError, 'Array bytes'):
            audit.exact_archives(left, right)


class Documents:
    def __init__(self, records): self.records = records
    def document(self, reference): return self.records[reference['path']]
    def closure(self, document): pass


def test_numerical_gates_require_exact_task_checkpoint_and_panel_ownership(self):
    ref = lambda value: {'path': value, 'sha256': value}
    task, fit, completed, gate_ref, panel, fit_audit = map(ref, ('task', 'fit', 'completed', 'gate', 'panel', 'fit-audit'))
    gate = {'kind': 'independent-generalization-inference-numerical-audit-v1', 'passed': True,
        'task': task, 'taskId': 'task-owner', 'fitResult': fit, 'completed': completed, 'fitAudit': fit_audit,
        'panel': panel, 'precision': 'fp32', 'checkpointEpochs': [5, 15, 30, 60], 'auditor': audit.io.identity(audit.NUMERICAL)}
    docs = Documents({'task': {'taskId': 'task-owner'}, 'gate': gate})
    assert audit.inference_gate(gate_ref, task, fit, completed, fit_audit, panel, 'fp32', docs) is gate
    for field, wrong in [('taskId', 'other-owner'), ('panel', ref('different-panel')),
                         ('completed', ref('other-checkpoint-bank')), ('precision', 'fp16'), ('checkpointEpochs', [60])]:
        docs.records['gate'] = {**gate, field: wrong}
        with self.assertRaisesRegex(ValueError, 'numerical gate'):
            audit.inference_gate(gate_ref, task, fit, completed, fit_audit, panel, 'fp32', docs)


class ReuseAuditTests(unittest.TestCase):
    test_recipe_and_first_owner_preserve_order_but_not_calibration = test_recipe_and_first_owner_preserve_order_but_not_calibration
    test_training_owner_cannot_cross_recipe_identity = test_training_owner_cannot_cross_recipe_identity
    test_inherited_training_history_and_bn_cannot_change = test_inherited_training_history_and_bn_cannot_change
    test_shared_scores_require_exact_values_and_no_vacuous_positive_claim = test_shared_scores_require_exact_values_and_no_vacuous_positive_claim
    test_feature_archive_equivalence_is_dtype_and_value_exact = test_feature_archive_equivalence_is_dtype_and_value_exact
    test_numerical_gates_require_exact_task_checkpoint_and_panel_ownership = test_numerical_gates_require_exact_task_checkpoint_and_panel_ownership


if __name__ == '__main__':
    unittest.main()
