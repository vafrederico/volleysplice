from __future__ import annotations
import json
import unittest

from analysis.neural_boundary_metrics import boundary_targets,evaluate_boundary_proposals,match_boundary_proposals


def candidate(start=10.,end=20.,kind='initial_start',parent='p',identity='c',**extra):
    return {'id':identity,'parentId':parent,'type':kind,'start':start,'end':end,
            'startObserved':True,'endObserved':True,**extra}


def record(proposals=(),gold=((10.,20.),(30.,40.)),parents=((8.,42.),),ignored=(),predictions=None,identity='r',group='g'):
    r={'id':identity,'sourceGroup':group,'durationSeconds':100.,'rallies':list(gold),
       'productionEvents':[{'id':'p' if i==0 else f'p{i}','start':s,'end':e} for i,(s,e) in enumerate(parents)],
       'ignoredIntervals':list(ignored),'boundaryProposals':list(proposals)}
    if predictions is not None:r['predictions']=predictions
    return r


def score(**kw):return evaluate_boundary_proposals([record(**kw)])['pooled']


class BoundaryMetricsTests(unittest.TestCase):
    def test_first_and_later_targets_typed(self):
        t=boundary_targets(record())
        self.assertEqual([x['type'] for x in t],['initial_start','additional_start'])
        self.assertEqual([x['start'] for x in t],[10,30])

    def test_perfect_two_events_start_and_end(self):
        s=score(proposals=[candidate(),candidate(30,40,'additional_start',identity='d')])
        self.assertEqual(s['typedStartLocalization']['1']['f1'],1)
        self.assertEqual(s['samePairBoundaries']['1']['f1'],1)

    def test_wrong_types_detected_separately(self):
        s=score(proposals=[candidate(kind='additional_start'),candidate(30,40,'initial_start',identity='d')])
        self.assertEqual(s['typedStartLocalization']['1']['matched'],0)
        self.assertEqual(s['untypedStartLocalization']['1']['matched'],2)
        self.assertEqual(s['typeDiagnostics']['1']['wrongType'],2)
        self.assertEqual(s['typeDiagnostics']['1']['initialProposedForAdditional'],1)
        self.assertEqual(s['typeDiagnostics']['1']['additionalProposedForInitial'],1)

    def test_end_cannot_match_another_gold_identity(self):
        s=score(proposals=[candidate(10,40)])
        self.assertEqual(s['typedStartLocalization']['1']['matched'],1)
        self.assertEqual(s['samePairBoundaries']['1']['matched'],0)

    def test_synthetic_end_not_observed_even_exact(self):
        s=score(proposals=[candidate(endObserved=False)])
        self.assertEqual(s['samePairBoundaries']['1']['matched'],0)
        self.assertEqual(s['unobservedCandidateEnds'],1)
        self.assertIsNone(s['samePairBoundaries']['1']['conditionalEndAccuracy'])

    def test_synthetic_start_not_observed_even_exact(self):
        s=score(proposals=[candidate(startObserved=False)])
        self.assertEqual(s['typedStartLocalization']['1']['predicted'],0)
        self.assertEqual(s['typedStartLocalization']['1']['true'],2)

    def test_duplicate_candidate_is_false_positive(self):
        s=score(proposals=[candidate(identity='a'),candidate(identity='b')])
        self.assertEqual(s['typedStartLocalization']['1']['matched'],1)
        self.assertEqual(s['typedStartLocalization']['1']['falsePositive'],1)

    def test_nearest_start_chosen_even_if_other_end_is_better(self):
        s=score(proposals=[candidate(10.1,25,identity='a'),candidate(10.2,20,identity='b')])
        self.assertEqual(s['typedStartLocalization']['1']['matched'],1)
        self.assertEqual(s['samePairBoundaries']['1']['matched'],0)

    def test_outside_parent_gold_endpoints_stay_denominator(self):
        s=score(gold=((7,20),(30,45)))
        self.assertEqual(s['targetCount'],2)
        self.assertEqual(s['inaccessibleTargetStarts'],1)
        self.assertEqual(s['inaccessibleTargetEnds'],1)

    def test_first_start_resets_after_ignored_component(self):
        t=boundary_targets(record(ignored=((21,29),)))
        self.assertEqual([x['type'] for x in t],['initial_start','initial_start'])

    def test_no_start_matching_across_ignored_gap(self):
        s=score(gold=((10,20),(30,40)),ignored=((29,30),),proposals=[candidate(28.5,29,identity='a')])
        self.assertEqual(s['untypedStartLocalization']['2']['matched'],0)

    def test_gold_censored_for_identity_but_valid_core_loss_still_counted(self):
        s=score(gold=((10,30),),ignored=((18,22),),predictions=[[22,42]])
        self.assertEqual(s['targetCount'],0)
        self.assertEqual(s['ignoredTouchedTrueRallies'],1)
        self.assertEqual(s['rawCoreSecondsLostFromBaseline'],8)
        self.assertEqual(s['additionalCompleteMisses'],0)

    def test_physical_partial_loss_and_new_complete_miss(self):
        s=score(predictions=[[15,25]])
        self.assertEqual(s['rawCoreSecondsLostFromBaseline'],15)
        self.assertEqual(s['additionalCompleteMisses'],1)

    def test_physical_coverage_clips_to_video_bounds(self):
        s=score(gold=((-5,20),),parents=((-10,42),),predictions=[[10,42]])
        self.assertEqual(s['rawCoreSecondsLostFromBaseline'],10)
        self.assertEqual(s['rawSelectedSecondsLostFromBaseline'],10)

    def test_oracle_accepts_wrong_type_when_start_and_overlap_valid(self):
        r=record();c=[candidate(kind='additional_start')]
        self.assertEqual(match_boundary_proposals(r,c)['pairs'],[(0,0)])

    def test_oracle_rejects_near_start_without_material_overlap(self):
        r=record();c=[candidate(9.5,10.1)]
        self.assertEqual(match_boundary_proposals(r,c)['pairs'],[])
        self.assertEqual(score(proposals=c)['untypedStartLocalization']['1']['matched'],0)

    def test_oracle_rejects_incidental_iou_without_start_match(self):
        self.assertEqual(match_boundary_proposals(record(),[candidate(8,42)])['pairs'],[])

    def test_oracle_preserves_original_candidate_indices(self):
        c=[candidate(15,18,identity='bad'),candidate(identity='good')]
        self.assertEqual(match_boundary_proposals(record(),c)['pairs'],[(1,0)])

    def test_tolerance_sensitivity_inclusive(self):
        s=score(proposals=[candidate(11,21)])
        self.assertEqual(s['typedStartLocalization']['0.5']['matched'],0)
        self.assertEqual(s['typedStartLocalization']['1']['matched'],1)
        self.assertEqual(s['samePairBoundaries']['1']['matched'],1)

    def test_pooled_counts_and_sourcegroups(self):
        result=evaluate_boundary_proposals([record(proposals=[candidate()],identity='a'),record(gold=((10,20),),identity='b',group='other')])
        self.assertEqual(result['pooled']['typedStartLocalization']['1']['recall'],1/3)
        self.assertEqual(result['sourceGroups']['other']['typedStartLocalization']['1']['recall'],0)
        json.dumps(result,allow_nan=False)

    def test_raw_core_recall_pools_seconds_before_rates(self):
        result=evaluate_boundary_proposals([
            record(gold=((10,30),),predictions=[[20,42]],identity='a'),
            record(gold=((10,12),),identity='b')])['pooled']
        self.assertEqual(result['coreHumanSeconds'],22)
        self.assertEqual(result['baselineRawCoreCoveredSeconds'],22)
        self.assertEqual(result['resultRawCoreCoveredSeconds'],12)
        self.assertEqual(result['resultRawCoreRecall'],12/22)

    def test_null_target_rates_not_perfect(self):
        s=score(gold=(),proposals=[candidate()])
        self.assertIsNone(s['typedStartLocalization']['1']['recall'])
        self.assertIsNone(s['typedStartLocalization']['1']['f1'])
        self.assertEqual(s['typedStartLocalization']['1']['falsePositive'],1)

    def test_untyped_pair_receipts_include_indices_and_types(self):
        r=evaluate_boundary_proposals([record(proposals=[candidate(kind='additional_start')])])['recordings'][0]
        self.assertEqual(r['typeDiagnostics']['1']['pairs'][0]['proposalIndex'],0)
        self.assertEqual(r['typeDiagnostics']['1']['pairs'][0]['targetType'],'initial_start')

    def test_outside_parent_or_bad_type_candidate_rejected(self):
        for c in (candidate(7,20),candidate(10,43),candidate(kind='bad')):
            with self.subTest(c=c),self.assertRaises(ValueError):score(proposals=[c])


if __name__=='__main__':unittest.main()
