from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_recall_sweep as io
from analysis import neural_selection_duration_view as adapter

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_duration_view_tests', REPO/'scripts/audit-neural-selection-duration-view.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def fixture():
    source = {'exactRows': [], 'other': {'literal': 1, 'ordered': ['a', 'b']}}
    records, caches = [], {}
    for index in range(8):
        key = str(index); frames = 6000 + 30*index + (1 if index else 0); duration = frames / 30.
        ref = {'path': f'cache-{index}', 'sha256': str(index)*64}
        metadata = {'duration': duration, 'frame_count': frames, 'fps': 30.}
        times = np.array([0., .25])
        row = {'id': key, 'sourceGroup': 'group', 'durationSeconds': round(duration, 6),
            'rallies': [{'start': 1., 'end': 2., 'tags': ['ace', 'reviewed']}],
            'ignoredIntervals': [{'start': 0., 'end': .2}],
            'featureCaches': {'audiovisual': {**ref, 'metadata': deepcopy(metadata)}}}
        source['exactRows'].append(row)
        records.append({**deepcopy(row), 'durationSeconds': duration,
                        'timestamps': times.tolist(), 'valid': [False, True]})
        caches[key] = (metadata, times, ref)
    comparison, changes = adapter.comparison_view(source, records,
        lambda ref: next((metadata, times) for metadata, times, cached in caches.values() if cached == ref))
    source_ref = {'path': 'source', 'sha256': 'f'*64}
    proof = {'originalSourceSha256': io.canonical(source), 'comparisonSourceSha256': io.canonical(comparison),
        'comparisonSource': comparison, 'changes': changes, 'interceptedDocuments': [source_ref],
        'replacedSymbols': [], 'evidenceOverride': 'Evidence.document',
        'otherFunctionBindingsUnchanged': True, 'otherEvidenceMethodsUnchanged': True,
        'selectionBytesUnchanged': True, 'metricDurationChanged': False,
        'externalOutcomesRead': False, 'trainingPerformed': False}
    return source, records, caches, source_ref, proof


class DurationViewAuditTests(unittest.TestCase):
    def test_seven_comparison_values_changed_without_mutating_source_or_records(self):
        source, records, caches, reference, proof = fixture()
        original = io.canonical(source); stored = io.canonical(records)
        changes, checks = gate.check_proof(proof, source, records, caches, reference)
        self.assertEqual(len(changes), 7); self.assertEqual(len(checks), 8)
        self.assertEqual(io.canonical(source), original); self.assertEqual(io.canonical(records), stored)
        self.assertEqual(gate.POLICY, adapter.POLICY)

    def test_labels_types_order_and_extra_view_fields_are_rejected(self):
        source, records, caches, reference, proof = fixture()
        mutations = [lambda x: x['comparisonSource']['exactRows'][1]['rallies'][0].update(end=2.001),
            lambda x: x['comparisonSource']['exactRows'][1]['rallies'][0]['tags'].reverse(),
            lambda x: x['comparisonSource']['exactRows'].reverse(),
            lambda x: x['comparisonSource']['other'].update(literal=1.),
            lambda x: x['comparisonSource'].update(extra='hidden'),
            lambda x: x['changes'].pop(),
            lambda x: x['changes'][0]['audiovisualCache'].update(path='other-cache')]
        for mutate in mutations:
            value = deepcopy(proof); mutate(value)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                gate.check_proof(value, source, records, caches, reference)

    def test_extra_interception_or_numerical_mutation_rejected(self):
        source, records, caches, reference, proof = fixture()
        for key, value in [('interceptedDocuments', [reference, reference]), ('replacedSymbols', ['historical_inputs']),
            ('otherEvidenceMethodsUnchanged', False), ('otherFunctionBindingsUnchanged', False),
            ('metricDurationChanged', True), ('selectionBytesUnchanged', False), ('externalOutcomesRead', True)]:
            mutated = {**proof, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.check_proof(mutated, source, records, caches, reference)

    def test_rounded_value_approximation_is_not_accepted(self):
        source, records, caches, _, _ = fixture()
        source['exactRows'][1]['durationSeconds'] += 1e-12
        with self.assertRaisesRegex(ValueError, 'exact six-decimal'):
            gate.expected_view(source, records, caches)

    def test_preexisting_cross_document_endpoint_types_are_not_cast_in_view(self):
        source, records, caches, _, _ = fixture()
        source['exactRows'][1]['rallies'][0]['start'] = 1
        view, _, _ = gate.expected_view(source, records, caches)
        self.assertIs(type(view['exactRows'][1]['rallies'][0]['start']), int)
        self.assertIs(type(records[1]['rallies'][0]['start']), float)

    def test_wrong_cache_frame_timeline_and_validity_are_rejected(self):
        source, records, caches, _, _ = fixture()
        for mode in ('identity', 'frames', 'timestamps', 'valid'):
            current_records, current_caches = deepcopy(records), deepcopy(caches)
            if mode == 'identity': current_caches['1'][2]['path'] = 'another-recording'
            elif mode == 'frames': current_caches['1'][0]['frame_count'] += 1
            elif mode == 'timestamps': current_caches['1'][1][1] = .251
            else: current_records[1]['valid'][0] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                gate.expected_view(source, current_records, current_caches)

    def test_stripped_duration_adapter_fails_even_when_metric_gold_is_unchanged(self):
        task_ref = {'path': 'task', 'sha256': '1'*64}
        stripped = {'task': task_ref, 'kind': 'independent-generalization-selection-audit-v1', 'passed': True}
        def read(path): return stripped if path == 'ordinary' else {'variant': 'original-corpus'}
        with patch.object(adapter.io, 'read', side_effect=read), patch.object(adapter, 'verified', return_value='task'):
            with self.assertRaisesRegex(ValueError, 'missing or stripped'):
                adapter.verify_execution('ordinary')


if __name__ == '__main__': unittest.main()
