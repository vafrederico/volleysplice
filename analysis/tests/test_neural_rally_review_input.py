import importlib.util
import hashlib
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2]/'scripts'/'prepare-neural-rally-review-input.py'
SPEC = importlib.util.spec_from_file_location('rally_review_input', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RallyReviewInputTests(unittest.TestCase):
    def test_retains_touching_and_overlapping_event_identities(self):
        source = [{'start': 1, 'end': 3}, {'start': 2, 'end': 4}, {'start': 4, 'end': 7}]
        result = MODULE.events(source, 'model:record:event', 10)
        self.assertEqual(len(result), 3)
        self.assertEqual([r['id'] for r in result], ['model:record:event:1', 'model:record:event:2', 'model:record:event:3'])
        self.assertEqual([r['sourceIndex'] for r in result], [0, 1, 2])
        self.assertEqual(MODULE.interval_identity(result), MODULE.interval_identity(source))
        self.assertNotIn('id', source[0])

    def test_rejects_reordered_duplicate_and_invalid_source_events(self):
        bad = [[{'start': 2, 'end': 3}, {'start': 1, 'end': 2}],
               [{'start': 1, 'end': 2, 'id': 'x'}, {'start': 3, 'end': 4, 'id': 'x'}],
               [{'start': 1, 'end': 1}], [{'start': -1, 'end': 2}],
               [{'start': 1, 'end': 11}], [{'start': 1, 'end': float('inf')}]]
        for source in bad:
            with self.assertRaises(ValueError):
                MODULE.events(source, 'event', 10)

    def test_recovers_actual_production_ids_not_new_sequence_ids(self):
        union = [{'id': 'R001', 'start': 1., 'end': 3., 'confidence': .9},
                 {'id': 'R002', 'start': 5., 'end': 9., 'confidence': .8}]
        core = [{'start': 5., 'end': 9.}]
        row = {'durationSeconds': 10, 'currentDefault': core, 'cores': {'productionDefault': core},
               'productReplay': {'unionWithConfidence': union, 'components': {
                   'previousProduction': [{'id': 'PP-007', 'start': 5., 'end': 9.}],
                   'allLabelsV2': [{'id': 'AV2-001', 'start': 5., 'end': 6.},
                                   {'id': 'AV2-002', 'start': 7., 'end': 9.},
                                   {'id': 'AV2-003', 'start': 9., 'end': 10.}]}}}
        event = MODULE.production_events(row)[0]
        self.assertEqual((event['id'], event['sourceUnionIndex'], event['confidence']), ('R002', 1, .8))
        self.assertEqual(event['sourceComponentIds']['allLabelsV2'], ['AV2-001', 'AV2-002'])
        row['cores']['productionDefault'] = row['currentDefault'] = [{'start': 5., 'end': 8.}]
        with self.assertRaises(ValueError):
            MODULE.production_events(row)

    def test_keeps_all_four_heads_and_rejects_dtype_shape_and_probability_errors(self):
        times = np.array([0., .266666666666, .5], dtype=np.float64)
        MODULE.validate_scores(np.zeros((3, 4), np.float32), times)
        for scores in [np.zeros((3, 1), np.float32), np.zeros((4, 4), np.float32),
                       np.zeros((3, 4), np.float64), np.full((3, 4), np.nan, np.float32),
                       np.full((3, 4), 1.01, np.float32)]:
            with self.assertRaises(ValueError):
                MODULE.validate_scores(scores, times)

    def test_exact_frame_quantized_timeline_rejects_arange_retiming(self):
        times = np.array([0., .266666666666, .5], dtype=np.float64)
        record = {'tickCount': 3, 'timelineBytesSha256': hashlib.sha256(times.astype('<f8').tobytes()).hexdigest()}
        MODULE.validate_timeline(times, record)
        for value in [np.arange(3, dtype=np.float64)/4, times.astype(np.float32), times[::-1]]:
            with self.assertRaises(ValueError):
                MODULE.validate_timeline(value, record)

    def test_rejects_exact_auxiliary_and_head_order_leakage(self):
        rows = {'a': {'sourceGroup': 'held'}, 'b': {'sourceGroup': 'train'}}
        membership = {'validationGroups': ['held'], 'validationIds': ['a'], 'trainGroups': ['train'],
                      'trainIds': ['b'], 'auxiliaryGroups': {'draft': ['aux']}, 'auxiliaryIds': {'draft': ['c']}}
        fit = {'sourceGroup': 'held', 'seed': 1, 'sourceContractSha256': 'contract', 'membership': membership}
        completed = {**membership, 'seed': 1, 'contractSha256': 'contract',
                     'model': {'headNames': ['live', 'serve', 'end', 'keep']}}
        self.assertEqual(MODULE.validate_membership(completed, fit, rows), {'a'})
        for key, value in [('trainGroups', ['held']), ('trainIds', ['a']),
                           ('auxiliaryGroups', {'draft': ['held']}), ('auxiliaryIds', {'draft': ['a']}),
                           ('model', {'headNames': ['serve', 'live', 'end', 'keep']})]:
            with self.assertRaises(ValueError):
                MODULE.validate_membership({**completed, key: value}, fit, rows)


if __name__ == '__main__':
    unittest.main()
