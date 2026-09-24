import unittest

import numpy as np

from analysis import neural_rally_review_proposals as r


def record(truth=(), ignored=(), duration=100.):
    return {'id': 'x', 'sourceGroup': 'g', 'durationSeconds': duration,
            'rallies': [{'start': a, 'end': b} for a, b in truth],
            'ignoredIntervals': [{'start': a, 'end': b} for a, b in ignored]}


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0., 100., .25)
        self.scores = np.full((len(self.times), 4), .1)

    def test_touching_split_flag_without_temporal_disagreement(self):
        rows = r.proposals(record(), [(10, 40)], [(10, 20), (20, 40)], self.times,
                           self.scores, 'local_events', 'production')
        self.assertEqual([(x['start'], x['end'], x['reasons']) for x in rows],
                         [(19.5, 20.5, ['split_merge'])])

    def test_small_gap_split_survives_tolerance(self):
        rows = r.proposals(record(), [(10, 40)], [(10, 20), (21, 40)], self.times,
                           self.scores, 'local_events', 'production')
        self.assertIn('split_merge', rows[0]['reasons'])
        old = r.proposals(record(), [(10, 40)], [(10, 20), (21, 40)], self.times,
                          self.scores, 'legacy', 'production')
        self.assertEqual(old, [])

    def test_flags_do_not_read_labels(self):
        args = ([(10, 40)], [(10, 20), (30, 40)], self.times, self.scores, 'local_heads', 'production')
        self.assertEqual(r.proposals(record(), *args), r.proposals(record([(2, 9), (50, 60)]), *args))

    def test_actual_time_cells_and_ignored_break(self):
        times = np.array([0., .2, .5, .8, 1.1, 1.4, 1.7, 2.])
        scores = np.ones((8, 4))*.6
        rows = r.live_components(record(ignored=[(.8, 1.1)], duration=2.2), times, scores)
        self.assertEqual(rows[0], {'start': 0., 'end': .65})
        self.assertAlmostEqual(rows[1]['start'], 1.1)
        self.assertEqual(rows[1]['end'], 2.15)

    def test_ignored_tick_values_do_not_change_heads(self):
        rec = record(ignored=[(20, 25)])
        altered = self.scores.copy()
        altered[(self.times >= 20) & (self.times < 25)] = [0, 1, 1, 1]
        args = (rec, [(10, 40)], [(10, 40)], self.times)
        self.assertEqual(r.proposals(*args, self.scores, 'local_heads', 'production'),
                         r.proposals(*args, altered, 'local_heads', 'production'))

    def test_nested_budget_counts_union_context(self):
        rec = record(duration=1000)
        rows = [{'id': str(i), 'start': 10.+i, 'end': 20.+i, 'priority': 1.} for i in range(10)]
        queues = r.budget_queues(rec, rows, 'evidence')
        self.assertEqual(queues[0]['proposalsSelected'], 10)
        self.assertEqual(queues[0]['reviewSeconds'], 27.)
        for a, b in zip(queues, queues[1:]):
            self.assertLessEqual(set(a['selectedIds']), set(b['selectedIds']))
            self.assertLessEqual(b['reviewSeconds'], b['budgetSeconds'])

    def test_empty_review_preserves_touching_events(self):
        out = r.edit_events(record(), [(10, 20), (20, 40)], [])
        self.assertEqual([(x['start'], x['end']) for x in out['events']], [(10, 20), (20, 40)])
        self.assertTrue(all(x['startObserved'] and x['endObserved'] for x in out['events']))

    def test_edit_restores_touching_split_with_identical_coverage(self):
        out = r.edit_events(record([(10, 20), (20, 40)]), [(10, 40)], [(19, 21)])
        self.assertEqual([(x['start'], x['end']) for x in out['events']], [(10, 20), (20, 40)])

    def test_edit_removes_false_split(self):
        out = r.edit_events(record([(10, 40)]), [(10, 20), (20, 40)], [(19, 21)])
        self.assertEqual([(x['start'], x['end']) for x in out['events']], [(10, 40)])

    def test_unseen_boundaries_are_not_imported(self):
        out = r.edit_events(record([(10, 40)]), [], [(20, 30)])
        self.assertEqual([(x['start'], x['end']) for x in out['events']], [(20, 30)])
        self.assertFalse(out['events'][0]['startObserved'])
        self.assertFalse(out['events'][0]['endObserved'])
        self.assertEqual((out['censoredStarts'], out['censoredEnds']), (1, 1))

    def test_ignored_base_occupancy_unchanged(self):
        out = r.edit_events(record([(10, 40)], [(20, 25)]), [(10, 40)], [(15, 30)])
        self.assertEqual([(x['start'], x['end']) for x in out['events']], [(10, 40)])

    def test_overlapping_event_identity_rejected(self):
        with self.assertRaises(ValueError):
            r.event_rows([(10, 30), (20, 40)])


if __name__ == '__main__':
    unittest.main()
