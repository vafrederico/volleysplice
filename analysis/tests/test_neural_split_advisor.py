import unittest

import numpy as np

from analysis import neural_production_combinations as iv
from analysis import neural_split_advisor as advisor


def record(ignored=(), duration=60.):
    return {'id': 'video', 'sourceGroup': 'group', 'durationSeconds': duration,
            'ignoredIntervals': [{'start': a, 'end': b} for a, b in ignored]}


class SplitAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0., 60., .25)
        self.scores = np.zeros((len(self.times), 4))
        self.scores[:, 0] = .8
        self.parent = [{'id': 'R001', 'start': 5., 'end': 40., 'confidence': .9,
                        'sourceComponentIds': {'old': ['PP-1', 'PP-2']}}]

    def proposals(self, policy, neural=(), rec=None, base=None):
        return advisor.split_proposals(rec or record(), self.parent if base is None else base,
                                       neural, self.times, self.scores, policy)

    def peak(self, time, serve=.8, end=.3):
        self.scores[self.times == time, 1] = serve
        self.scores[self.times == time - .25, 2] = end

    def test_single_shifted_neural_start_is_not_a_split(self):
        self.assertEqual(self.proposals('event_starts', [(20, 35)]), [])

    def test_event_start_keeps_actual_time_and_source_evidence(self):
        rows = self.proposals('event_starts', [{'id': 'a', 'start': 6, 'end': 15},
                                              {'id': 'b', 'start': 20.125, 'end': 30}])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['time'], 20.125)
        self.assertEqual((row['parentId'], row['parentStart'], row['parentEnd']), ('R001', 5., 40.))
        self.assertEqual(row['evidence']['previousNeuralEventId'], 'a')
        self.assertEqual(row['evidence']['neuralEventId'], 'b')
        self.assertAlmostEqual(row['priority'], .8)

    def test_prior_neural_evidence_does_not_cross_ignored_barrier(self):
        self.assertEqual(self.proposals('event_starts', [(6, 10), (20, 30)],
                                        rec=record([(12, 14)])), [])

    def test_internal_edges_are_strict_and_overlap_threshold_is_inclusive(self):
        self.assertEqual(self.proposals('event_starts', [(5, 6), (7, 10)]), [])
        rows = self.proposals('event_starts', [(5, 5.5), (20, 20.5)])
        self.assertEqual([x['time'] for x in rows], [20.])
        self.assertEqual(self.proposals('event_starts', [(5, 5.499), (20, 30)]), [])

    def test_event_nms_uses_live_strength_and_strict_one_second_radius(self):
        neural = [(6, 10), (20, 20.5), (20.75, 21.5), (22, 25)]
        self.scores[(self.times >= 20) & (self.times < 20.5), 0] = .4
        self.scores[(self.times >= 20.75) & (self.times < 21.5), 0] = .9
        self.assertEqual([x['time'] for x in self.proposals('event_starts', neural)], [20.75, 22.])
        self.assertEqual([x['time'] for x in self.proposals('event_starts', [(6, 10), (20, 20.5), (21, 25)])],
                         [20., 21.])

    def test_head_requires_preceding_end_or_live_valley(self):
        self.peak(20, end=0)
        self.assertEqual(self.proposals('head_evidence'), [])
        self.scores[self.times == 19, 0] = .349
        self.assertEqual([x['time'] for x in self.proposals('head_evidence')], [20.])
        self.scores[self.times == 19, 0] = .35
        self.assertEqual(self.proposals('head_evidence'), [])
        self.scores[self.times == 19, 2] = .25
        self.assertEqual(len(self.proposals('head_evidence')), 1)

    def test_head_plateau_has_one_earliest_peak(self):
        self.scores[(self.times >= 20) & (self.times <= 22), 1] = .8
        self.scores[self.times == 19.75, 2] = .3
        self.assertEqual([x['time'] for x in self.proposals('head_evidence')], [20.])

    def test_head_minimum_threshold_and_nms(self):
        self.peak(20, serve=.35)
        self.peak(20.5, serve=.9)
        self.peak(21.5, serve=.8)
        self.assertEqual([x['time'] for x in self.proposals('head_evidence')], [20.5, 21.5])

    def test_preceding_head_evidence_is_parent_and_component_local(self):
        self.peak(20, end=0)
        self.scores[self.times == 19.5, 2] = .9
        self.assertEqual(self.proposals('head_evidence', rec=record([(19.6, 19.8)])), [])
        self.scores[self.times == 19.5, 2] = 0
        self.peak(7.25, end=0)
        self.scores[self.times == 4.5, 2] = .9
        self.assertEqual(self.proposals('head_evidence'), [])

    def test_ignored_tick_scores_cannot_create_or_change_peak(self):
        self.peak(30)
        rec = record([(19.75, 20.25)])
        before = self.proposals('head_evidence', rec=rec)
        self.scores[(self.times >= 19.75) & (self.times < 20.25)] = [0., 1., 1., 1.]
        self.assertEqual(before, self.proposals('head_evidence', rec=rec))

    def test_nms_does_not_suppress_across_ignored_barrier(self):
        rows = advisor._nms([
            {'time': 20., 'strength': .8, 'component': (0., 20.1), 'evidence': {}},
            {'time': 20.75, 'strength': .9, 'component': (20.2, 60.), 'evidence': {}}])
        self.assertEqual([row['time'] for row in rows], [20., 20.75])

    def test_generation_requires_strict_two_seconds_from_valid_component_edges(self):
        rec = record([(17, 19)])
        for time in (20., 21.):
            self.assertEqual(self.proposals('event_starts', [(19, 19.5), (time, 30)], rec=rec), [])
            self.peak(time)
            self.assertEqual(self.proposals('head_evidence', rec=rec), [])
            self.scores[:, 1:3] = 0
        rows = self.proposals('event_starts', [(19, 20), (21.25, 30)], rec=rec)
        self.assertEqual([row['time'] for row in rows], [21.25])
        self.peak(21.25)
        self.assertEqual([row['time'] for row in self.proposals('head_evidence', rec=rec)], [21.25])

    def test_irregular_sample_times_are_not_replaced_by_fixed_grid(self):
        times = np.array([5., 7., 10.1, 14.37, 18.02, 23.4, 35.])
        scores = np.zeros((len(times), 4))
        scores[:, 0] = .8
        scores[4, 2] = .3
        scores[5, 1] = .9
        # No preceding evidence within three seconds: the large sampling gap matters.
        rows = advisor.split_proposals(record(), self.parent, [], times, scores, 'head_evidence')
        self.assertEqual(rows, [])
        times[4] = 22.6
        rows = advisor.split_proposals(record(), self.parent, [], times, scores, 'head_evidence')
        self.assertEqual([x['time'] for x in rows], [23.4])

    def test_corroborated_uses_head_timestamp_and_minimum_strength(self):
        self.peak(21, serve=.9)
        rows = self.proposals('corroborated', [(6, 12), (20, 30)])
        self.assertEqual([x['time'] for x in rows], [21.])
        self.assertAlmostEqual(rows[0]['priority'], .8)
        self.assertEqual(rows[0]['evidence']['corroborationDistanceSeconds'], 1.)
        self.scores[self.times == 21, 1] = 0
        self.peak(21.25)
        self.assertEqual(self.proposals('corroborated', [(6, 12), (20, 30)]), [])

    def test_corroboration_cannot_cross_even_narrow_ignored_barrier(self):
        self.peak(20.75)
        self.assertEqual(self.proposals('corroborated', [(6, 12), (20, 30)],
                                        rec=record([(20.1, 20.2)])), [])

    def test_corroboration_is_one_to_one(self):
        self.peak(20.5)
        rows = self.proposals('corroborated', [(6, 12), (20, 20.5), (21, 30)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['evidence']['neuralStart'], 20.)

    def test_label_blindness(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key in ('rallies', 'serveMarkers', 'hardNegatives'):
                    raise AssertionError('Labels were read')
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in ('rallies', 'serveMarkers', 'hardNegatives'):
                    raise AssertionError('Labels were read')
                return super().get(key, default)

        self.peak(20)
        rec = Poison(record())
        for policy in advisor.POLICIES:
            self.assertEqual(self.proposals(policy, [(6, 12), (20, 30)], rec=rec),
                             self.proposals(policy, [(6, 12), (20, 30)]))
        advisor.cleanup_proposals(rec, self.parent, [(6, 12)])

    def test_cleanup_threshold_is_strict_and_support_is_dilated_two_seconds(self):
        base = [(10, 30)]
        self.assertEqual(advisor.cleanup_proposals(record(), base, [(12, 18)]), [])
        rows = advisor.cleanup_proposals(record(), base, [(12, 17)])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['supportFraction'], .45)
        self.assertAlmostEqual(rows[0]['priority'], .55)

    def test_cleanup_does_not_join_positive_support_gaps(self):
        row = advisor.cleanup_proposals(record(), [(5, 45)], [(7, 10), (15, 18)])[0]
        self.assertEqual(row['supportedSeconds'], 14.)

    def test_cleanup_ignored_parent_is_ineligible_and_denominator_masks_ignored(self):
        self.assertEqual(advisor.cleanup_proposals(record([(5, 40)]), self.parent, []), [])
        rows = advisor.cleanup_proposals(record([(20, 30)]), [(10, 30)], [(11, 12)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['parentValidSeconds'], 10.)
        self.assertEqual(rows[0]['supportFraction'], .4)

    def test_cleanup_dilation_cannot_bridge_ignored_time_or_use_ignored_events(self):
        rows = advisor.cleanup_proposals(record([(18, 19)]), [(19, 23)], [(17, 18)])
        self.assertEqual(rows[0]['supportFraction'], 0.)
        rows = advisor.cleanup_proposals(record([(18, 22)]), [(15, 25)], [(19, 21)])
        self.assertEqual(rows[0]['supportFraction'], 0.)

    def test_partition_preserves_union_all_padding_cases_and_source_lineage(self):
        self.peak(20)
        proposals = self.proposals('head_evidence')
        children = advisor.apply_splits(self.parent, proposals)
        self.assertEqual([(x['start'], x['end']) for x in children], [(5., 20.), (20., 40.)])
        self.assertTrue(all(x['parentId'] == 'R001' for x in children))
        self.assertEqual(children[0]['sourceComponentIds'], self.parent[0]['sourceComponentIds'])
        self.assertEqual(children[1]['proposalIds'], ['s00000'])
        self.assertFalse(children[0]['endObserved'])
        self.assertTrue(children[1]['startObserved'])
        rec = record([(25, 26)])
        for padding in iv.PADS:
            self.assertEqual(iv.export(self.parent, rec, padding), iv.export(children, rec, padding))

    def test_original_observed_flags_and_touching_parent_identity_survive(self):
        base = [{'id': 'a', 'start': 5, 'end': 20, 'startObserved': False},
                {'id': 'b', 'start': 20, 'end': 40, 'endObserved': False}]
        children = advisor.apply_splits(base, [{'id': 's', 'parentId': 'b', 'time': 30}])
        self.assertEqual([x['parentId'] for x in children], ['a', 'b', 'b'])
        self.assertFalse(children[0]['startObserved'])
        self.assertFalse(children[-1]['endObserved'])
        self.assertEqual(children[0]['id'], 'a')

    def test_coincident_split_proposals_combine_traceable_ids(self):
        rows = advisor.apply_splits(self.parent, [{'id': 'x', 'parentId': 'R001', 'time': 20},
                                                 {'id': 'y', 'parentId': 'R001', 'time': 20}])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['proposalIds'], ['x', 'y'])
        self.assertEqual(rows[0]['endProposalIds'], ['x', 'y'])

    def test_invalid_inputs_and_unbounded_splits_fail(self):
        for proposal in ({'id': 's', 'parentId': 'wrong', 'time': 20},
                         {'id': 's', 'parentId': 'R001', 'time': 40},
                         {'id': 's', 'parentId': 'R001', 'time': float('nan')}):
            with self.assertRaises(ValueError):
                advisor.apply_splits(self.parent, [proposal])
        with self.assertRaises(ValueError):
            advisor.event_rows([(5, 20), (10, 30)])
        with self.assertRaises(ValueError):
            self.proposals('unknown')
        self.scores[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.proposals('event_starts')


if __name__ == '__main__':
    unittest.main()
