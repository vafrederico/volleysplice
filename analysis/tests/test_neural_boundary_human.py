from __future__ import annotations
import unittest

from analysis.neural_boundary_human import human_edit
from analysis.neural_boundary_metrics import evaluate_boundary_proposals
from analysis.tests.test_neural_boundary_metrics import candidate,record


class BoundaryHumanTests(unittest.TestCase):
    def test_unselected_parent_exactly_preserved(self):
        r=record();h=human_edit(r,[candidate()],[])
        self.assertEqual(h['events'],r['productionEvents'])
        self.assertEqual(h['reviewedCandidates'],0)

    def test_all_rejected_parent_exactly_preserved(self):
        r=record();h=human_edit(r,[candidate(15,20)],['p'])
        self.assertEqual(h['events'],r['productionEvents'])
        self.assertEqual(h['rejectedCount'],1)

    def test_initial_start_replaces_baseline_start_not_extra_rally(self):
        r=record();h=human_edit(r,[candidate(10.5,21)],['p'])
        self.assertEqual([(e['start'],e['end']) for e in h['events']],[(10,42)])
        self.assertTrue(h['events'][0]['startObserved'])

    def test_additional_only_keeps_baseline_fallback(self):
        r=record();h=human_edit(r,[candidate(30.5,40,'additional_start')],['p'])
        self.assertEqual([(e['start'],e['end']) for e in h['events']],[(8,30),(30,42)])
        self.assertFalse(h['events'][0]['endObserved'])
        self.assertEqual(h['events'][0]['startKind'],'baseline_fallback')

    def test_starts_only_partitions_at_accepted_start(self):
        c=[candidate(),candidate(30,40,'additional_start',identity='d')]
        h=human_edit(record(),c,['p'])
        self.assertEqual([(e['start'],e['end']) for e in h['events']],[(10,30),(30,42)])
        self.assertFalse(h['events'][0]['endObserved'])
        self.assertTrue(h['events'][1]['endObserved'])

    def test_paired_corrects_only_accepted_endpoints(self):
        c=[candidate(10.5,25),candidate(30.5,41,'additional_start',identity='d')]
        h=human_edit(record(),c,['p'],correct_ends=True)
        self.assertEqual([(e['start'],e['end']) for e in h['events']],[(10,20),(30,40)])
        self.assertTrue(all(e['startObserved'] and e['endObserved'] for e in h['events']))

    def test_paired_can_lose_unproposed_rally_and_metrics_report_loss(self):
        r=record();h=human_edit(r,[candidate()],['p'],correct_ends=True)
        s=evaluate_boundary_proposals([{**r,'predictions':h['events']}])['pooled']
        self.assertEqual(h['acceptedCount'],1)
        self.assertEqual(s['rawCoreSecondsLostFromBaseline'],10)
        self.assertEqual(s['additionalCompleteMisses'],1)

    def test_wrong_type_corrected_by_human(self):
        h=human_edit(record(),[candidate(kind='additional_start')],['p'])
        self.assertEqual(h['correctedTypeCount'],1)
        self.assertEqual(len(h['events']),1)
        self.assertEqual(h['events'][0]['start'],10)

    def test_unobserved_candidate_start_cannot_be_oracle_accepted(self):
        h=human_edit(record(),[candidate(startObserved=False)],['p'],correct_ends=True)
        self.assertEqual(h['acceptedCount'],0)

    def test_outside_gold_start_clipped_without_observed_marker(self):
        r=record(gold=((7.5,20),))
        h=human_edit(r,[candidate(8.1,21)],['p'],correct_ends=True)
        self.assertEqual(h['events'][0]['start'],8)
        self.assertFalse(h['events'][0]['startObserved'])

    def test_outside_gold_end_clipped_without_observed_marker(self):
        r=record(gold=((10,45),))
        h=human_edit(r,[candidate(10,41)],['p'],correct_ends=True)
        self.assertEqual(h['events'][0]['end'],42)
        self.assertFalse(h['events'][0]['endObserved'])

    def test_accepted_parent_components_keep_unproposed_component(self):
        r=record(ignored=((21,29),))
        h=human_edit(r,[candidate()],['p'],correct_ends=True)
        self.assertEqual([(e['start'],e['end']) for e in h['events']],[(10,20),(29,42)])
        self.assertFalse(h['events'][1]['startObserved'])

    def test_two_components_initial_starts_independently_replaced(self):
        r=record(ignored=((21,29),))
        c=[candidate(),candidate(30,40,'initial_start',identity='d')]
        h=human_edit(r,c,['p'],correct_ends=True)
        self.assertEqual([(e['start'],e['end']) for e in h['events']],[(10,20),(30,40)])

    def test_duplicate_candidates_only_one_accepted(self):
        h=human_edit(record(),[candidate(identity='a'),candidate(identity='b')],['p'])
        self.assertEqual(h['acceptedCount'],1)
        self.assertEqual(h['rejectedCount'],1)

    def test_reviewed_indices_preserve_global_input_indices(self):
        r=record(parents=((8,42),(50,80)),gold=((10,20),(60,70)))
        c=[candidate(60,70,parent='p1',identity='other'),candidate(identity='here')]
        h=human_edit(r,c,['p'])
        self.assertEqual(h['acceptedCandidates'][0]['candidateIndex'],1)
        self.assertEqual(h['reviewedCandidates'],1)

    def test_unknown_selection_rejected(self):
        with self.assertRaises(ValueError):human_edit(record(),[],['no-such-parent'])


if __name__=='__main__':unittest.main()
