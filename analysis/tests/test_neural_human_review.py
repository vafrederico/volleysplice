import unittest

from analysis import neural_human_review as review


def record(rallies=(), ignored=(), duration=30):
    return {'id':'a','sourceGroup':'g','durationSeconds':duration,
            'rallies':list(rallies),'ignoredIntervals':list(ignored)}


class ReviewTests(unittest.TestCase):
    def test_flags_ignore_gold(self):
        a=record([{'start':10,'end':15}])
        b=record([{'start':1,'end':29}])
        for rule in review.COMBO_POLICIES:
            self.assertEqual(review.combination_plan(a,[(8,16)],[(10,14)],rule,[(8,16)],[]),
                             review.combination_plan(b,[(8,16)],[(10,14)],rule,[(8,16)],[]))
        for rule in review.SOLO_POLICIES:
            self.assertEqual(review.individual_plan(a,[(8,16)],list(range(30)),[.6]*30,rule),
                             review.individual_plan(b,[(8,16)],list(range(30)),[.6]*30,rule))

    def test_half_is_strict_and_tolerance_has_no_gap_join(self):
        r=record()
        # Neural 4..6 expands to 2..8, covering exactly half of 2..14.
        self.assertFalse(review.combination_plan(r,[(2,14)],[(4,6)],'suppression_half')[0]['flagged'])
        self.assertTrue(review.combination_plan(r,[(2,14.1)],[(4,6)],'suppression_half')[0]['flagged'])
        self.assertTrue(review.combination_plan(r,[(0,12)],[(2,3),(8,9)],'suppression_any')[0]['flagged'])

    def test_guard_protects_full_both_source_component(self):
        r=record()
        p=[(4,12)]
        plan=review.combination_plan(r,p,[],'guarded_trim',[(4,8)],[(8,12)])
        self.assertFalse(plan[0]['flagged'])

    def test_ignored_span_splits_decisions(self):
        r=record(ignored=[{'start':8,'end':12}])
        plan=review.combination_plan(r,[(4,16)],[],'suppression_zero')
        self.assertEqual([(x['start'],x['end']) for x in plan],[(4,8),(12,16)])
        outcome=review.human_outcomes(r,[(4,16)],plan)
        self.assertEqual(outcome['padding'][2]['flaggedCandidates'],2)
        self.assertFalse(any(a<12 and b>8 for a,b in outcome['padding'][2]['playbackIntervals']))

    def test_grid_covers_valid_time_without_overlapping_positive(self):
        r=record(ignored=[{'start':8,'end':9}],duration=12)
        plan=review.individual_plan(r,[(2,4),(9,11)],list(range(12)),[.5]*12,'all_candidates')
        self.assertEqual([(p['start'],p['end'],p['kind']) for p in plan],
                         [(0,2,'negative'),(2,4,'positive'),(4,5,'negative'),(5,8,'negative'),
                          (9,11,'positive'),(11,12,'negative')])
        self.assertAlmostEqual(sum(p['end']-p['start'] for p in plan),11)

    def test_no_tick_is_reviewed_and_threshold_equality_is_confident(self):
        r=record(duration=5)
        plan=review.individual_plan(r,[(0,1),(2,3)],[.5,4],[.8,.2],'uncertain_medium')
        self.assertFalse(plan[0]['flagged'])
        self.assertTrue(next(p for p in plan if p['start']==2)['flagged'])
        self.assertFalse(next(p for p in plan if p['start']==3)['flagged'])

    def test_binary_keeps_mixed_candidate_boundary_edit_fixes_only_permission(self):
        r=record([{'start':10,'end':12},{'start':20,'end':22}])
        plan=review.combination_plan(r,[(8,14),(18,24)],[(18,24)],'suppression_zero')
        result=review.human_outcomes(r,[(8,14),(18,24)],plan)
        self.assertEqual(result['binaryRaw'],[[8.,14.],[18.,24.]])
        self.assertEqual(result['padding'][0]['mixedCandidates'],1)
        self.assertEqual(result['boundaryExports']['0'],[[10.,12.],[18.,24.]])

    def test_negative_recovery_and_context_do_not_enlarge_edit_permission(self):
        r=record([{'start':7,'end':8},{'start':12,'end':13}],duration=20)
        plan=review.individual_plan(r,[],list(range(20)),[.8 if 5<=i<10 else .01 for i in range(20)],'uncertain_medium')
        out=review.human_outcomes(r,[],plan)
        self.assertEqual(out['binaryRaw'],[[5.,10.]])
        self.assertEqual(out['boundaryExports']['0'],[[7.,8.]])
        self.assertEqual(out['padding'][2]['reviewedTrueRallyIds'],[0])
        self.assertEqual(out['padding'][2]['playbackTrueRallyIds'],[0,1])


if __name__=='__main__':
    unittest.main()
