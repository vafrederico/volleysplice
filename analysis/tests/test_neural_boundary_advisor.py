import copy
import unittest

import numpy as np

from analysis import neural_boundary_advisor as advisor
from analysis import neural_production_combinations as iv


def record(ignored=()):
    return {'id': 'synthetic', 'sourceGroup': 'fixture', 'durationSeconds': 60.,
            'ignoredIntervals': [{'start': x, 'end': y} for x, y in ignored]}


class BoundaryAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.parent = [{'id': 'p', 'start': 5., 'end': 40., 'sourceIds': ['old-1']}]
        self.neural = [{'id': 'n1', 'start': 8., 'end': 16.},
                       {'id': 'n2', 'start': 24., 'end': 32.}]
        self.times = np.arange(0., 60., .25)
        self.scores = np.zeros((len(self.times), 4))
        self.scores[:, 0] = .8

    def plan(self, policy, rec=None, base=None, neural=None):
        return advisor.plan(rec or record(), self.parent if base is None else base,
                            self.neural if neural is None else neural, self.times, self.scores, policy)

    def peak(self, time, head, value):
        self.scores[self.times == time, head] = value

    def test_first_start_is_a_replacement_not_an_extra_rally(self):
        out = self.plan('first_start')
        self.assertEqual([(e['start'], e['end']) for e in out['events']], [(8., 40.)])
        self.assertEqual([c['type'] for c in out['eventCandidates']], ['initial_start'])
        self.assertEqual([p['kind'] for p in out['proposals']], ['initial_start'])
        self.assertTrue(out['events'][0]['endObserved'])
        self.assertEqual(out['events'][0]['endSource'], 'production-end')

    def test_typed_starts_distinguishes_initial_and_additional(self):
        out = self.plan('typed_starts')
        self.assertEqual([(e['start'], e['end']) for e in out['events']], [(8., 24.), (24., 40.)])
        self.assertEqual([c['type'] for c in out['eventCandidates']], ['initial_start', 'additional_start'])
        self.assertFalse(out['events'][0]['endObserved'])
        self.assertEqual(out['events'][0]['endSource'], 'partition-only')
        self.assertNotIn('end', [p['kind'] for p in out['proposals']])

    def test_separate_ends_creates_real_dead_time_gap(self):
        out = self.plan('separate_ends')
        self.assertEqual([(e['start'], e['end']) for e in out['events']], [(8., 16.), (24., 32.)])
        self.assertTrue(all(e['endObserved'] for e in out['events']))
        self.assertEqual([p['kind'] for p in out['proposals']], ['initial_start', 'end', 'additional_start', 'end'])
        self.assertGreater(iv.duration(iv.difference(self.parent, out['events'])), 0)

    def test_clipped_candidate_boundaries_are_unobserved(self):
        out = self.plan('separate_ends', neural=[{'id': 'wide', 'start': 2., 'end': 45.}])
        candidate = out['eventCandidates'][0]
        self.assertEqual((candidate['start'], candidate['end']), (5., 40.))
        self.assertFalse(candidate['startObserved'])
        self.assertFalse(candidate['endObserved'])
        self.assertEqual(candidate['startSource'], 'parent-clipped')
        self.assertEqual(out['proposals'], [])

    def test_native_boundaries_at_parent_edges_remain_observed(self):
        out = self.plan('separate_ends', neural=[{'id': 'same', 'start': 5., 'end': 40.}])
        self.assertTrue(out['eventCandidates'][0]['startObserved'])
        self.assertTrue(out['eventCandidates'][0]['endObserved'])

    def test_unsupported_parent_is_retained_without_candidate(self):
        original = copy.deepcopy(self.parent)
        for policy in advisor.POLICIES:
            out = self.plan(policy, neural=[])
            self.assertEqual(out['eventCandidates'], [])
            self.assertEqual(out['proposals'], [])
            self.assertEqual([(e['id'], e['start'], e['end']) for e in out['events']], [('p', 5., 40.)])
            self.assertTrue(out['events'][0]['fallback'])
        self.assertEqual(original, self.parent)

    def test_partial_support_retains_other_valid_component_as_fallback(self):
        out = self.plan('separate_ends', rec=record([(20, 22)]), neural=self.neural[:1])
        self.assertEqual([(e['start'], e['end']) for e in out['events']], [(8., 16.), (22., 40.)])
        fallback = out['events'][1]
        self.assertTrue(fallback['fallback'])
        self.assertFalse(fallback['startObserved'])
        self.assertTrue(fallback['endObserved'])

    def test_typing_resets_after_ignored_component(self):
        out = self.plan('separate_ends', rec=record([(20, 22)]))
        self.assertEqual([c['type'] for c in out['eventCandidates']], ['initial_start', 'initial_start'])
        self.assertNotEqual(out['eventCandidates'][0]['componentId'], out['eventCandidates'][1]['componentId'])

    def test_ignored_clipped_endpoints_are_unobserved(self):
        out = self.plan('separate_ends', rec=record([(20, 22)]), neural=[{'id': 'span', 'start': 10., 'end': 30.}])
        self.assertFalse(out['eventCandidates'][0]['endObserved'])
        self.assertFalse(out['eventCandidates'][1]['startObserved'])
        self.assertEqual(out['eventCandidates'][1]['startSource'], 'ignored-clipped')

    def test_association_requires_half_second_of_same_component_overlap(self):
        self.assertEqual(self.plan('typed_starts', neural=[(4, 5.499)])['eventCandidates'], [])
        self.assertEqual(len(self.plan('typed_starts', neural=[(4, 5.5)])['eventCandidates']), 1)
        out = self.plan('separate_ends', rec=record([(5.4, 5.6)]), neural=[(5, 6)])
        self.assertEqual(out['eventCandidates'], [])

    def test_no_gold_access_or_gold_dependent_typing(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key in ('rallies', 'serveMarkers'):
                    raise AssertionError('Gold was read')
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in ('rallies', 'serveMarkers'):
                    raise AssertionError('Gold was read')
                return super().get(key, default)

        for policy in advisor.POLICIES:
            self.assertEqual(self.plan(policy), self.plan(policy, rec=Poison(record())))
        out = self.plan('typed_starts', neural=self.neural[1:])
        self.assertEqual(out['eventCandidates'][0]['type'], 'initial_start')

    def test_head_refinement_chooses_strongest_feasible_then_nearest_then_earliest(self):
        self.peak(7., 1, .5)
        self.peak(9., 1, .5)
        self.assertEqual(self.plan('head_refined')['eventCandidates'][0]['start'], 7.)
        self.peak(10.5, 1, .9)
        self.assertEqual(self.plan('head_refined')['eventCandidates'][0]['start'], 10.5)
        self.peak(15.75, 1, 1.)  # Too far and would violate minimum length.
        self.assertEqual(self.plan('head_refined')['eventCandidates'][0]['start'], 10.5)

    def test_head_refinement_searches_feasible_peaks_before_selection(self):
        neural = [(8, 10)]
        self.peak(9.75, 1, 1.)
        self.peak(9.5, 1, .7)
        out = self.plan('head_refined', neural=neural)
        self.assertEqual(out['eventCandidates'][0]['start'], 9.5)
        self.assertEqual(out['eventCandidates'][0]['end'], 10.)

    def test_head_sequential_constraints_prevent_overlap(self):
        neural = [(8, 16), (18, 26)]
        self.peak(18., 2, .9)
        self.peak(17., 1, 1.)
        self.peak(18.25, 1, .7)
        out = self.plan('head_refined', neural=neural)
        self.assertEqual([(e['start'], e['end']) for e in out['events']], [(8., 18.), (18.25, 26.)])

    def test_head_no_qualifying_peak_keeps_original_endpoints(self):
        self.peak(9, 1, .249)
        self.assertEqual([(e['start'], e['end']) for e in self.plan('head_refined')['events']],
                         [(8., 16.), (24., 32.)])

    def test_head_exact_threshold_and_radius_are_inclusive(self):
        self.peak(11., 1, .25)
        self.peak(19., 2, .25)
        out = self.plan('head_refined')
        self.assertEqual((out['eventCandidates'][0]['start'], out['eventCandidates'][0]['end']), (11., 19.))

    def test_head_scores_in_ignored_time_cannot_affect_output(self):
        rec = record([(17, 23)])
        before = self.plan('head_refined', rec=rec)
        self.scores[(self.times >= 17) & (self.times < 23)] = [0, 1, 1, 1]
        self.assertEqual(before, self.plan('head_refined', rec=rec))

    def test_head_at_real_parent_endpoint_is_observed_and_ignored_endpoint_is_not_used(self):
        self.peak(5., 1, .9)
        self.peak(40., 2, .9)
        out = self.plan('head_refined', neural=[(2, 45)])
        self.assertTrue(out['eventCandidates'][0]['startObserved'])
        self.assertTrue(out['eventCandidates'][0]['endObserved'])
        self.assertEqual(out['eventCandidates'][0]['startSource'], 'head-serve')
        rec = record([(40, 42)])
        out = self.plan('head_refined', rec=rec, neural=[(2, 45)])
        self.assertFalse(out['eventCandidates'][0]['endObserved'])

    def test_actionable_shift_threshold_and_unobserved_partition_ends(self):
        out = self.plan('first_start', neural=[(5.249, 20)])
        self.assertEqual(out['proposals'], [])
        out = self.plan('first_start', neural=[(5.25, 20)])
        self.assertEqual([p['kind'] for p in out['proposals']], ['initial_start'])
        out = self.plan('separate_ends', neural=[(5, 39.75)])
        self.assertEqual([p['kind'] for p in out['proposals']], ['end'])

    def test_lineage_candidate_ids_priorities_and_original_inputs_preserved(self):
        original = copy.deepcopy((self.parent, self.neural))
        plans = [self.plan(policy) for policy in advisor.POLICIES]
        identity = plans[0]['eventCandidates'][0]['id']
        for out in plans:
            self.assertEqual(out['eventCandidates'][0]['id'], identity)
            self.assertEqual(out['events'][0]['sourceIds'], ['old-1'])
            self.assertAlmostEqual(out['eventCandidates'][0]['priority'], .8)
            self.assertTrue(all(p['candidateId'] in {c['id'] for c in out['eventCandidates']} for p in out['proposals']))
        self.assertEqual(original, (self.parent, self.neural))

    def test_fallback_original_observation_flags_are_retained(self):
        parent = [{**self.parent[0], 'startObserved': False, 'endObserved': False}]
        out = self.plan('typed_starts', base=parent, neural=[])
        self.assertFalse(out['events'][0]['startObserved'])
        self.assertFalse(out['events'][0]['endObserved'])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.plan('unknown')
        with self.assertRaises(ValueError):
            self.plan('first_start', neural=[(8, 16), (15, 20)])
        self.scores[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.plan('first_start')


if __name__ == '__main__':
    unittest.main()
