from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_selection_duration_view as adapter
from analysis import neural_recall_sweep as io
from analysis.neural_selection_serialization import load_module

REPO = Path(__file__).resolve().parents[2]


def fixture():
    duration = 100 / 3
    ref = {'path': 'cache.npz', 'sha256': 'a'*64}
    source = {'unrelated': {'keep': [3, 4.0, True]}, 'exactRows': [{
        'id': 'video', 'sourceGroup': 'source', 'durationSeconds': round(duration, 6),
        'featureCaches': {'audiovisual': {**ref, 'metadata': {'duration': duration}}},
        'rallies': [{'start': 10, 'end': 12.5, 'tags': ['ace']}], 'ignoredIntervals': [{'start': 0, 'end': 1}]}]}
    row = deepcopy(source['exactRows'][0]); row.pop('featureCaches'); row['durationSeconds'] = duration
    row.update(timestamps=[.5, 1.5], valid=[False, True])
    metadata = {'duration': duration, 'frame_count': 100, 'fps': 3}
    return source, [row], lambda _: (metadata, np.asarray(row['timestamps']))


class DurationViewTests(unittest.TestCase):
    def test_comparison_only_change_preserves_source_and_records(self):
        source, rows, loader = fixture()
        before_source, before_rows = io.canonical(source), io.canonical(rows)
        view, changes = adapter.comparison_view(source, rows, loader)
        self.assertEqual(len(changes), 1)
        self.assertEqual(view['exactRows'][0]['durationSeconds'], rows[0]['durationSeconds'])
        self.assertEqual(io.canonical(source), before_source)
        self.assertEqual(io.canonical(rows), before_rows)
        view['exactRows'][0]['durationSeconds'] = source['exactRows'][0]['durationSeconds']
        self.assertEqual(io.canonical(view), before_source)

    def test_wrong_frame_duration_or_nonexact_rounding_rejected(self):
        for mutation in ('frames', 'rounding', 'stored', 'metadata'):
            source, rows, loader = fixture(); metadata, times = loader(None)
            if mutation == 'frames': metadata['frame_count'] += 1
            elif mutation == 'rounding': source['exactRows'][0]['durationSeconds'] += 1e-10
            elif mutation == 'stored': rows[0]['durationSeconds'] += 1e-10
            else: source['exactRows'][0]['featureCaches']['audiovisual']['metadata']['duration'] += 1e-10
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                adapter.comparison_view(source, rows, lambda _: (metadata, times))

    def test_time_validity_and_label_mutations_rejected(self):
        for mutation in ('times', 'valid', 'rally', 'tag'):
            source, rows, loader = fixture(); metadata, times = loader(None)
            if mutation == 'times': times[0] += .01
            elif mutation == 'valid': rows[0]['valid'][0] = True
            elif mutation == 'rally': rows[0]['rallies'][0]['start'] += .01
            else: rows[0]['rallies'][0]['tags'] = ['changed']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                adapter.comparison_view(source, rows, lambda _: (metadata, times))

    def test_original_delegated_receipt_cannot_strip_execution(self):
        ordinary = {'task': {'path': 'task', 'sha256': '0'*64}}
        with patch.object(adapter.io, 'read', side_effect=lambda p: ordinary if p == 'ordinary' else {'variant': 'original-corpus'}), \
             patch.object(adapter, 'verified', side_effect=lambda r: r['path']):
            with self.assertRaisesRegex(ValueError, 'missing or stripped'): adapter.verify_execution('ordinary')

    def test_nonhistorical_dispatch_uses_existing_frozen_execution_gate(self):
        ordinary = {'task': {'path': 'task', 'sha256': '0'*64}}
        with patch.object(adapter.io, 'read', side_effect=lambda p: ordinary if p == 'ordinary' else {'variant': 'expanded-medium'}), \
             patch.object(adapter, 'verified', side_effect=lambda r: r['path']), \
             patch.object(adapter.serialization, 'verify_execution', return_value={'existing': True}) as delegated:
            self.assertEqual(adapter.verify_execution('ordinary'), {'existing': True})
            delegated.assert_called_once_with('ordinary')

    def test_composite_workflow_still_requires_complete_global_freeze(self):
        workflow = load_module('duration_workflow_test', REPO/'scripts/generalization-selection-audit-execution.py')
        with patch.object(workflow.io, 'read', return_value={'kind': workflow.GLOBAL_KIND, 'passed': True,
            'taskCount': 161, 'selections': []}):
            with self.assertRaisesRegex(ValueError, 'global162'): workflow.global_gate('incomplete')
        with patch.object(workflow.io, 'read', return_value={'kind': 'independent-generalization-result-serialization-audit-v1', 'passed': True}):
            with self.assertRaisesRegex(ValueError, 'explicit-execution/global-selection'):
                workflow.report(SimpleNamespace(audit='old', index='index', output='output'))


if __name__ == '__main__': unittest.main()
