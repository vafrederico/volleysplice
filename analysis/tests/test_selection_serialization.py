from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_recall_sweep as io
from analysis import neural_selection_serialization as adapter
from analysis.schema import Interval

REPO = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


gate = load('independent_serialization_test', 'scripts/audit-neural-selection-serialization.py')


def example(endpoint=119):
    return io.SweepExample('source', 'group', 200., np.array([.125, .375]), np.ones(2, bool),
        (Interval(endpoint, 120., ('ace',)),), (Interval(0, 8),))


def fixture(endpoint=119):
    before = adapter.gold([example(endpoint)])
    after = adapter.gold(adapter.normalize_examples([example(endpoint)]))
    return before, {'beforeGold': before, 'afterGold': after,
        'beforeGoldSha256': io.canonical(before), 'afterGoldSha256': io.canonical(after),
        'changes': adapter.field_changes(before, after), 'replacedSymbols': ['calibration_examples'],
        'otherFunctionBindingsUnchanged': True, 'selectionBytesUnchanged': True}


class SerializationTests(unittest.TestCase):
    def test_exact_type_changes_and_all_other_arrays_preserved(self):
        before, proof = fixture()
        changes = gate.check_proof(proof, before, proof['afterGoldSha256'])
        self.assertEqual(len(changes), 3)
        normalized = adapter.normalize_examples([example()])[0]
        self.assertEqual(normalized.truth[0].tags, ('ace',))
        self.assertEqual(normalized.truth[0].start, 119)
        self.assertIs(type(normalized.truth[0].start), float)
        self.assertNotEqual(proof['beforeGoldSha256'], proof['afterGoldSha256'])

    def test_unrepresentable_integer_rejected_by_both_implementations(self):
        with self.assertRaisesRegex(ValueError, 'exact value'):
            adapter.normalize_examples([example(2**53+1)])
        original = adapter.gold([example(2**53+1)])
        with self.assertRaisesRegex(ValueError, 'exact endpoint value'): gate.normalize_gold(original)

    def test_any_other_field_value_type_or_order_change_fails(self):
        before, proof = fixture()
        mutations = [lambda p: p['afterGold'][0]['rallies'][0].update(tags=['service-fault']),
            lambda p: p['afterGold'][0]['times'].reverse(),
            lambda p: p['afterGold'][0]['valid'].__setitem__(0, False),
            lambda p: p['afterGold'][0].update(durationSeconds=201.),
            lambda p: p['afterGold'][0]['rallies'][0].update(start=119.00000000001),
            lambda p: p['changes'].pop(), lambda p: p.update(replacedSymbols=['calibration_examples', 'audit_candidates']),
            lambda p: p.update(otherFunctionBindingsUnchanged=False)]
        for mutation in mutations:
            value = deepcopy(proof); mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate.check_proof(value, before, proof['afterGoldSha256'])

    def test_zero_difference_adaptation_is_valid(self):
        value = adapter.normalize_examples([example()])
        before = adapter.gold(value); after = deepcopy(before)
        proof = {'beforeGold': before, 'afterGold': after, 'beforeGoldSha256': io.canonical(before),
            'afterGoldSha256': io.canonical(after), 'changes': [], 'replacedSymbols': ['calibration_examples'],
            'otherFunctionBindingsUnchanged': True, 'selectionBytesUnchanged': True}
        self.assertEqual(gate.check_proof(proof, before, io.canonical(before)), [])

    def test_external_evaluation_rejects_partial_global_freeze(self):
        workflow = load('serialization_workflow_test', 'scripts/generalization-selection-serialization.py')
        value = {'kind': workflow.GLOBAL_KIND, 'passed': True, 'taskCount': 161, 'selections': []}
        with patch.object(workflow.io, 'read', return_value=value), self.assertRaisesRegex(ValueError, 'global162'):
            workflow.global_gate(Path('dummy'))

    def test_publication_cannot_use_containment_only_audit(self):
        workflow = load('serialization_workflow_publication_test', 'scripts/generalization-selection-serialization.py')
        with patch.object(workflow.io, 'read', return_value={'kind': workflow.old.RESULT_GATE_KIND, 'passed': True}):
            with self.assertRaisesRegex(ValueError, 'serialization/global-selection'):
                workflow.report(SimpleNamespace(audit=Path('dummy'), index=Path('dummy'), output=Path('dummy')))

    def test_stripped_adapter_base_is_rejected_by_execution_preflight(self):
        before, proof = fixture()
        source = REPO/'scripts/audit-neural-generalization-selection.py'
        reference = {'path': 'selection', 'sha256': '0'*64}
        task_ref = {'path': 'task', 'sha256': '1'*64}
        ordinary = {'kind': 'independent-generalization-selection-audit-v1', 'passed': True,
            'auditor': io.identity(source), 'selection': reference, 'task': task_ref}
        selection = {'task': task_ref, 'selectionGoldSha256': proof['afterGoldSha256']}
        unchanged = SimpleNamespace(calibration_examples=lambda *a: [example()], Evidence=lambda: None)
        def read(path): return ordinary if path == 'ordinary' else selection if path == 'selection' else {}
        with patch.object(adapter.io, 'read', side_effect=read), patch.object(adapter, 'verified', side_effect=lambda r: r['path']), \
             patch.object(adapter, 'load_module', return_value=unchanged), \
             patch('analysis.neural_generalization_experiment.load_task', return_value=({'variant': 'original-medium'}, 'manifest', 'features', None)):
            with self.assertRaisesRegex(ValueError, 'adapter may have been stripped'):
                adapter.verify_execution('ordinary')


if __name__ == '__main__': unittest.main()
