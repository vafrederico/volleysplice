import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from analysis import neural_rally_review_proposals as builder


SOURCE = Path(__file__).resolve().parents[2]/'scripts'/'audit-neural-rally-review-candidates.py'
SPEC = importlib.util.spec_from_file_location('candidate_reconstruction_audit', SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class GoldPoison(dict):
    def __getitem__(self, key):
        if key not in ('durationSeconds', 'ignoredIntervals'):
            raise AssertionError('Read gold/metadata: '+str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in ('durationSeconds', 'ignoredIntervals'):
            raise AssertionError('Read gold/metadata: '+str(key))
        return super().get(key, default)


class CandidateAuditTests(unittest.TestCase):
    def check(self, record, base, neural, times, scores, inventory, mode):
        candidates = builder.proposals(record, base, neural, times, scores, inventory, mode)
        receipt = AUDIT.audit_candidates(record, base, neural, times, scores, inventory, mode, candidates)
        self.assertTrue(receipt['passed'])
        return candidates

    def fixture(self):
        record = GoldPoison(durationSeconds=20., ignoredIntervals=[{'start': 9.1, 'end': 9.9}])
        times = np.array([i/4 + (1/60 if i%2 else 0) for i in range(80)], np.float64)
        scores = np.zeros((80, 4), np.float32)
        scores[:, 0] = .2
        scores[8:20, 0] = .85
        scores[28:36, 0] = .7
        scores[48:60, 0] = .8
        scores[9, 1] = .6
        scores[18, 2] = .75
        scores[50, 1] = .8
        scores[57, 2] = .9
        return record, times, scores

    def test_reconstructs_all_six_paths_without_gold_access(self):
        record, times, scores = self.fixture()
        base, neural = [[2., 8.], [10., 15.]], [[2., 4.], [5., 8.], [11., 14.]]
        for inventory in builder.INVENTORIES:
            for mode in ('production', 'individual'):
                self.check(record, base, neural, times, scores, inventory, mode)

    def test_detects_missing_extra_geometry_reason_and_priority_mutations(self):
        record, times, scores = self.fixture()
        base, neural = [[2., 8.]], [[2., 4.], [5., 8.]]
        candidates = self.check(record, base, neural, times, scores, 'local_heads', 'production')
        mutations = [candidates[:-1], [*candidates, candidates[0]]]
        for key, replacement in [('start', .123), ('reasons', ['invented']), ('priority', 123.), ('id', 'wrong')]:
            changed = copy.deepcopy(candidates)
            changed[0][key] = replacement
            mutations.append(changed)
        for values in mutations:
            with self.assertRaises(ValueError):
                AUDIT.audit_candidates(record, base, neural, times, scores, 'local_heads', 'production', values)

    def test_touching_events_still_produce_split_boundary_even_same_union(self):
        record = GoldPoison(durationSeconds=20., ignoredIntervals=[])
        times = np.arange(80, dtype=np.float64)/4
        scores = np.full((80, 4), .1, np.float32)
        candidates = self.check(record, [[2., 8.]], [[2., 5.], [5., 8.]], times, scores, 'local_events', 'production')
        self.assertEqual(len(candidates), 1)
        self.assertEqual((candidates[0]['start'], candidates[0]['end'], candidates[0]['reasons']),
                         (4.5, 5.5, ['split_merge']))

    def test_actual_midpoint_edges_not_nominal_tick_grid(self):
        record = GoldPoison(durationSeconds=3., ignoredIntervals=[])
        times = np.array([0., .266666666666, .5, .766666666666, 1., 1.266666666666, 1.5, 1.766666666666])
        scores = np.zeros((len(times), 4), np.float32)
        scores[1:6, 0] = .8
        candidates = self.check(record, [], [], times, scores, 'local_events', 'individual')
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]['start'], (times[0]+times[1])/2)
        self.assertAlmostEqual(candidates[0]['end'], (times[5]+times[6])/2)

    def test_ignored_score_changes_cannot_change_any_candidate(self):
        record, times, scores = self.fixture()
        altered = scores.copy()
        altered[(times >= 9.1) & (times < 9.9)] = 1.
        for inventory in builder.INVENTORIES:
            for mode in ('production', 'individual'):
                args = (record, [[2., 8.], [10., 15.]], [[2., 4.], [5., 8.]], times)
                first = self.check(*args, scores, inventory, mode)
                second = self.check(*args, altered, inventory, mode)
                self.assertEqual(first, second)

    def test_high_live_evidence_cannot_cross_ignored_gap_with_no_tick(self):
        record = GoldPoison(durationSeconds=3., ignoredIntervals=[{'start': 1.01, 'end': 1.09}])
        times = np.array([0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75])
        scores = np.full((len(times), 4), .2, np.float32)
        scores[1, 0] = .4
        candidates = self.check(record, [], [], times, scores, 'local_events', 'individual')
        self.assertEqual(len(candidates), 1)
        self.assertLessEqual(candidates[0]['end'], 1.01)

    def test_boundary_peak_uses_local_valid_component(self):
        record = GoldPoison(durationSeconds=8., ignoredIntervals=[{'start': 3., 'end': 4.}])
        times = np.arange(32, dtype=np.float64)/4
        scores = np.zeros((len(times), 4), np.float32)
        scores[18, 1] = .9
        candidates = self.check(record, [[2., 2.75]], [], times, scores, 'local_heads', 'production')
        self.assertFalse(any('boundary_start' in row['reasons'] for row in candidates))

    def test_randomized_crosscheck_including_ignored_touching_and_splits(self):
        rng = np.random.default_rng(618723)
        for _ in range(16):
            record = GoldPoison(durationSeconds=24., ignoredIntervals=[{'start': 11.1, 'end': 11.2}])
            times = np.array([i/4 + (1/60 if i%2 else 0) for i in range(96)], np.float64)
            scores = rng.uniform(0, 1, (96, 4)).astype(np.float32)
            base, neural = [[2., 8.], [10., 15.], [17., 21.]], [[1.8, 4.], [5., 8.], [12., 15.], [15., 20.]]
            for inventory in builder.INVENTORIES:
                for mode in ('production', 'individual'):
                    self.check(record, base, neural, times, scores, inventory, mode)


if __name__ == '__main__':
    unittest.main()
