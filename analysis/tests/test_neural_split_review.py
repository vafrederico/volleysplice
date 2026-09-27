import copy
import unittest
from analysis import neural_split_review as r
from analysis import neural_split_advisor as a
from analysis import neural_split_metrics as m


def record():
    return {'id': 'synthetic', 'sourceGroup': 'fixture', 'durationSeconds': 100,
            'productionEvents': [{'id': 'p1', 'start': 10., 'end': 30.},
                                 {'id': 'p2', 'start': 50., 'end': 54.}],
            'rallies': [{'start': 10., 'end': 16.}, {'start': 20., 'end': 27.}],
            'ignoredIntervals': []}


def split(time=20.4, name='s1'):
    return {'id': name, 'parentId': 'p1', 'time': time, 'start': time-1, 'end': time+1, 'priority': .8}


def cleanup():
    return {'id': 'c2', 'parentId': 'p2', 'start': 50., 'end': 54., 'priority': 1., 'supportFraction': 0.}


class ReviewTests(unittest.TestCase):
    def test_whole_parent_cost_not_small_split_band(self):
        rec = record(); jobs = r.jobs(rec, rec['productionEvents'], [split()], [], 'split_only')
        self.assertEqual(jobs[0]['standalonePlaybackSeconds'], 24.)
        q = r.queues(rec, jobs, 'chronological', [.1, .3])
        self.assertEqual(q[0]['reviewJobs'], 0)
        self.assertEqual(q[1]['reviewSeconds'], 24.)

    def test_combined_deduplicates_parent_jobs(self):
        rec = record(); c = {**cleanup(), 'parentId': 'p1', 'id': 'c1'}
        rows = r.jobs(rec, rec['productionEvents'], [split(), split(25,'s2')], [c], 'combined')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['splitIds'], ['s1','s2'])
        self.assertEqual(rows[0]['cleanupIds'], ['c1'])

    def test_skip_oversized_nested_and_evidence_order(self):
        rec = record(); rows = r.jobs(rec, rec['productionEvents'], [split()], [cleanup()], 'combined')
        qs = r.queues(rec, rows, 'chronological', [.1,.2,.4])
        self.assertEqual(qs[0]['selectedParentIds'], ['p2'])
        self.assertTrue(set(qs[0]['selectedParentIds']) <= set(qs[-1]['selectedParentIds']))
        self.assertEqual(r.queues(rec, rows, 'evidence', [.4])[0]['selectedParentIds'][0], 'p2')

    def test_ignore_barriers_not_rejoined(self):
        rec = record(); rec['ignoredIntervals'] = [{'start': 19.8, 'end': 20.2}]
        rows = r.jobs(rec, rec['productionEvents'], [split()], [], 'split_only')
        q = r.queues(rec, rows, 'chronological', [.4])[0]
        self.assertEqual(len(q['playbackWindows']), 2)
        self.assertAlmostEqual(q['reviewSeconds'], 23.6)

    def test_human_accepts_snaps_rejects_and_keeps_original_coverage(self):
        rec = record(); ss = [split(), split(25,'false')]
        rows = r.jobs(rec, rec['productionEvents'], ss, [], 'split_only')
        q = r.queues(rec, rows, 'chronological', [.4])[0]
        out = r.human_edit(rec, ss, [], q, a, m)
        self.assertEqual(out['acceptedSplitCount'], 1)
        self.assertEqual(out['rejectedSplitCount'], 1)
        self.assertEqual(out['acceptedSplits'][0]['time'], 20.)
        self.assertEqual(out['eventTimelineRemovedSeconds'], 0.)
        self.assertEqual(out['eventTimelineAddedSeconds'], 0.)

    def test_false_parent_cleanup_only_changes_event_timeline(self):
        rec = record(); original = copy.deepcopy(rec)
        rows = r.jobs(rec, rec['productionEvents'], [], [cleanup()], 'cleanup_only')
        q = r.queues(rec, rows, 'chronological', [.1])[0]
        out = r.human_edit(rec, [], [cleanup()], q, a, m)
        self.assertEqual(out['removedFalseParentIds'], ['p2'])
        self.assertEqual(out['eventTimelineRemovedSeconds'], 4.)
        self.assertEqual(rec, original)
        self.assertEqual(len(rec['productionEvents']), 2)

    def test_unreviewed_splits_are_unapplied(self):
        rec = record(); ss = [split()]
        rows = r.jobs(rec, rec['productionEvents'], ss, [], 'split_only')
        q = r.queues(rec, rows, 'chronological', [.05])[0]
        out = r.human_edit(rec, ss, [], q, a, m)
        self.assertEqual(out['acceptedSplitCount'], 0)
        self.assertEqual(len(out['events']), 2)

    def test_mixed_parent_cannot_be_cleaned(self):
        rec = record(); cc = [{**cleanup(),'id':'c1','parentId':'p1'}]
        rows = r.jobs(rec, rec['productionEvents'], [], cc, 'cleanup_only')
        q = r.queues(rec, rows, 'chronological', [.4])[0]
        out = r.human_edit(rec, [], cc, q, a, m)
        self.assertEqual(out['removedFalseParents'], 0)
        self.assertEqual(r.cleanup_yield(rec,cc)['realOrMixedParents'], 1)


if __name__ == '__main__':
    unittest.main()
