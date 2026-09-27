import copy
import unittest
from analysis import neural_boundary_review as r
from analysis import neural_production_combinations as iv


class FullParentReviewTests(unittest.TestCase):
    def record(self):
        return {'id':'x','sourceGroup':'g','durationSeconds':100.,
                'productionEvents':[{'id':'p','start':10.,'end':40.},{'id':'q','start':60.,'end':65.}],
                'rallies':[{'start':8.,'end':15.},{'start':25.,'end':30.}], 'ignoredIntervals':[]}

    def test_unproposed_rallies_separate_with_clipped_unseen_boundary(self):
        rec=self.record(); original=copy.deepcopy(rec)
        out=r.full_parent_edit(rec,['p'])['events']
        self.assertEqual([(e['start'],e['end']) for e in out],[(10.,15.),(25.,30.),(60.,65.)])
        self.assertFalse(out[0]['startObserved']); self.assertTrue(out[0]['endObserved'])
        self.assertEqual(out[-1],rec['productionEvents'][-1]); self.assertEqual(rec,original)

    def test_false_selected_parent_removed_from_timeline_only(self):
        rec=self.record(); out=r.full_parent_edit(rec,['q'])['events']
        self.assertEqual(out,[rec['productionEvents'][0]])
        self.assertEqual(len(rec['productionEvents']),2)

    def test_touching_gold_remains_two_events(self):
        rec=self.record(); rec['rallies']=[{'start':10.,'end':20.},{'start':20.,'end':30.}]
        out=r.full_parent_edit(rec,['p'])['events']
        self.assertEqual(len(out),3); self.assertEqual(out[0]['end'],out[1]['start'])

    def test_ignored_barrier_and_gold_coverage(self):
        rec=self.record(); rec['ignoredIntervals']=[{'start':12.,'end':13.}]
        out=r.full_parent_edit(rec,['p'])['events']
        self.assertEqual([(e['start'],e['end']) for e in out[:3]],[(10.,12.),(13.,15.),(25.,30.)])
        before=iv.intersection(iv.difference(rec['productionEvents'],rec['ignoredIntervals']),rec['rallies'])
        after=iv.intersection(out,rec['rallies'])
        self.assertEqual(before,after)

    def test_full_parent_cost_and_budget(self):
        rec=self.record(); plan={'proposals':[{'id':'flag','parentId':'p','start':25.,'end':26.,'priority':.8}]}
        rows=r.jobs(rec,plan); q=r.queues(rec,rows,'evidence',[.1,.4])
        self.assertEqual(q[0]['reviewJobs'],0); self.assertEqual(q[1]['reviewSeconds'],34.)
        self.assertEqual(q[1]['selectedProposalIds'],['flag'])


if __name__=='__main__': unittest.main()
