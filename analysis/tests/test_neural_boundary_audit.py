"""Independent boundary-adviser audit qualification."""
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('boundary_audit', Path(__file__).resolve().parents[2]
                                            / 'scripts/audit-neural-boundary-advisor.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record(**extra):
    return {'id': 'v', 'sourceGroup': 'g', 'durationSeconds': 100., 'ignoredIntervals': [],
            'productionEvents': [{'id': 'p', 'start': 5., 'end': 45.}],
            'rallies': [{'start': 10., 'end': 20.}, {'start': 30., 'end': 40.}], **extra}


class RawCoverageTests(unittest.TestCase):
    def test_shortening_event_can_lose_core_without_changing_export(self):
        from analysis import neural_production_combinations as intervals
        rec = record(predictions=[{'start': 12., 'end': 18.}, {'start': 30., 'end': 40.}])
        fixed = {'v': {str(p): audit.scalar.export(rec['productionEvents'], rec, p) for p in range(4)}}
        rows = intervals.duration_rows([rec], fixed)
        self.assertTrue(audit.audit_fixed_export([rec], rows, fixed)['passed'])
        self.assertEqual(audit.raw_coverage(rec, rec['predictions'])['rawCoreSecondsLostFromBaseline'], 4.)

    def test_complete_event_loss_is_reported_separately(self):
        rec = record(); result = audit.raw_coverage(rec, [{'start': 10., 'end': 20.}])
        self.assertEqual(result['additionalCompleteMisses'], 1)
        self.assertEqual(result['additionalCompleteMissIndexes'], [1])
        self.assertEqual(result['rawCoreSecondsLostFromBaseline'], 10.)

    def test_valid_core_of_censored_gold_still_counts_duration_loss(self):
        rec = record(ignoredIntervals=[[14., 16.]])
        result = audit.raw_coverage(rec, [{'start': 30., 'end': 40.}])
        self.assertEqual(result['rawCoreSecondsLostFromBaseline'], 8.)
        self.assertEqual(result['ignoredTouchedGoldIdentities'], 1)
        self.assertEqual(result['additionalCompleteMisses'], 0)

    def test_fixed_export_cannot_be_rebuilt_from_trimmed_events(self):
        from analysis import neural_production_combinations as intervals
        rec = record(predictions=[{'start': 10., 'end': 20.}])
        wrong = {'v': {str(p): audit.scalar.export(rec['predictions'], rec, p) for p in range(4)}}
        rows = intervals.duration_rows([rec], wrong)
        with self.assertRaisesRegex(ValueError, 'Export changed'):
            audit.audit_fixed_export([rec], rows, wrong)


class IndependentPeakTests(unittest.TestCase):
    def test_feasibility_precedes_peak_selection(self):
        times = [1., 2., 3., 4., 5.]
        scores = [[0., x, 0., 0.] for x in (.1, .3, .9, .4, .1)]
        self.assertEqual(audit.feasible_peak(times, scores, 1, 3., 3.5, 5., (0., 6.)), (4., True))

    def test_nearest_then_earliest_tie_break(self):
        times = [1., 2., 4., 5.]
        scores = [[0., .8, 0., 0.] for _ in times]
        self.assertEqual(audit.feasible_peak(times, scores, 1, 3., 0., 6., (0., 6.)), (2., True))

    def test_no_feasible_threshold_peak_keeps_original(self):
        self.assertEqual(audit.feasible_peak([1., 2., 3.], [[0., .24, 0., 0.]]*3,
                                            1, 2., 0., 5., (0., 5.)), (2., False))

    def test_no_mask_edge_peak(self):
        self.assertEqual(audit.feasible_peak([2., 3., 4.], [[0., x, 0., 0.] for x in (.1, .1, .9)],
                                            1, 3., 0., 4., (0., 4.)), (3., False))


class IndependentCandidateTests(unittest.TestCase):
    def fixture(self):
        rec = record(); nn = [{'id': 'n1', 'start': 8., 'end': 20.}, {'id': 'n2', 'start': 30., 'end': 42.}]
        times = list(range(100)); scores = [[.8, 0., 0., 0.] for _ in times]
        return rec, nn, times, scores

    def test_first_start_replaces_early_start_without_extra_event(self):
        rec, nn, times, scores = self.fixture()
        result = audit.reconstruct_core_candidates(rec, rec['productionEvents'], nn, times, scores, 'first_start')
        self.assertEqual([(x['start'], x['end']) for x in result], [(8., 45.)])
        self.assertEqual(result[0]['type'], 'initial_start')

    def test_separate_end_removes_dead_interval_from_event_timeline(self):
        rec, nn, times, scores = self.fixture()
        result = audit.reconstruct_core_candidates(rec, rec['productionEvents'], nn, times, scores, 'separate_ends')
        self.assertEqual([(x['start'], x['end']) for x in result], [(8., 20.), (30., 42.)])

    def test_typed_starts_preserve_gap_but_not_synthetic_end_observation(self):
        rec, nn, times, scores = self.fixture()
        result = audit.reconstruct_core_candidates(rec, rec['productionEvents'], nn, times, scores, 'typed_starts')
        self.assertEqual([(x['start'], x['end']) for x in result], [(8., 30.), (30., 45.)])
        self.assertFalse(result[0]['endObserved'])
        self.assertTrue(result[1]['endObserved'])


class BoundaryIntegrationAuditTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        from analysis import neural_boundary_advisor as advisor
        rec = record()
        nn = [{'id': 'n1', 'start': 10.2, 'end': 19.5}]
        times = np.arange(0., 100., .25); scores = np.zeros((len(times), 4)); scores[:, 0] = .8
        return rec, nn, times, scores, advisor

    def test_all_four_candidate_policies_reconstructed(self):
        rec, nn, times, scores, advisor = self.fixture()
        for policy in advisor.POLICIES:
            plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, policy)
            self.assertTrue(audit.audit_plan(rec, rec['productionEvents'], nn, times, scores, policy, plan)['passed'])

    def test_invented_candidate_endpoint_rejected(self):
        rec, nn, times, scores, advisor = self.fixture()
        plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends')
        plan['eventCandidates'][0]['end'] += .2
        with self.assertRaisesRegex(ValueError, 'eventCandidates'):
            audit.audit_plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends', plan)

    def test_unobserved_clipped_native_boundaries(self):
        rec, nn, times, scores, advisor = self.fixture()
        rec['ignoredIntervals'] = [[15., 25.]]
        nn = [{'id': 'n1', 'start': 10., 'end': 15.}, {'id': 'n2', 'start': 25., 'end': 35.}]
        plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends')
        self.assertTrue(audit.audit_plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends', plan)['passed'])
        self.assertFalse(plan['eventCandidates'][0]['endObserved'])
        self.assertFalse(plan['eventCandidates'][1]['startObserved'])

    def test_proposal_confirmation_reports_lost_unproposed_rally(self):
        from analysis import neural_boundary_human as human
        rec, nn, times, scores, advisor = self.fixture()
        plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends')
        edited = human.human_edit(rec, plan['eventCandidates'], ['p'], correct_ends=True)
        receipt = audit.audit_proposal_human(rec, plan, ['p'], edited)
        self.assertEqual(receipt['rawCoverage']['additionalCompleteMisses'], 1)
        self.assertEqual(receipt['rawCoverage']['rawCoreSecondsLostFromBaseline'], 10.)

    def test_full_parent_can_recover_unproposed_rally(self):
        from analysis import neural_boundary_review as review
        rec, _, _, _, _ = self.fixture()
        edited = review.full_parent_edit(rec, ['p'])
        receipt = audit.audit_full_parent_human(rec, ['p'], edited)
        self.assertEqual(len(edited['events']), 2)
        self.assertEqual(receipt['rawCoverage']['rawCoreSecondsLostFromBaseline'], 0)

    def test_false_human_end_observation_rejected(self):
        from analysis import neural_boundary_review as review
        rec, _, _, _, _ = self.fixture()
        rec['productionEvents'][0]['start'] = 12.
        edited = review.full_parent_edit(rec, ['p'])
        edited['events'][0]['startObserved'] = True
        with self.assertRaisesRegex(ValueError, 'startObserved'):
            audit.audit_full_parent_human(rec, ['p'], edited)

    def test_all_typed_metrics_and_physical_coverage(self):
        from analysis import neural_boundary_metrics as metrics
        rec, nn, times, scores, advisor = self.fixture()
        plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends')
        records = [{**rec, 'predictions': plan['events'], 'boundaryProposals': plan['eventCandidates']}]
        values = metrics.evaluate_boundary_proposals(records)
        self.assertTrue(audit.audit_typed_metrics(records, values)['passed'])
        values['pooled']['resultRawCoreRecall'] += .01
        with self.assertRaisesRegex(ValueError, 'resultRawCoreRecall'):
            audit.audit_typed_metrics(records, values)

    def test_wrong_type_is_verified_from_untyped_pair_evidence(self):
        from analysis import neural_boundary_metrics as metrics
        rec, nn, times, scores, advisor = self.fixture()
        plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, 'separate_ends')
        candidates = [dict(plan['eventCandidates'][0], type='additional_start')]
        records = [{**rec, 'boundaryProposals': candidates}]
        values = metrics.evaluate_boundary_proposals(records)
        self.assertTrue(audit.audit_typed_metrics(records, values)['passed'])
        self.assertEqual(values['pooled']['typeDiagnostics']['1']['wrongType'], 1)
        values['recordings'][0]['typeDiagnostics']['1']['wrongType'] = 0
        with self.assertRaisesRegex(ValueError, 'wrongType'):
            audit.audit_typed_metrics(records, values)

    def test_review_workload_counts_all_selected_parent_candidates(self):
        from analysis import neural_boundary_review as review
        rec, nn, times, scores, advisor = self.fixture()
        plan = advisor.plan(rec, rec['productionEvents'], nn, times, scores, 'first_start')
        queue = review.queues(rec, review.jobs(rec, plan), 'evidence', budgets=(1.,))[0]
        edited = review.full_parent_edit(rec, queue['selectedParentIds'])
        work = review.workload(rec, plan, queue, edited)
        self.assertTrue(audit.audit_workload(rec, plan, queue, edited, work)['passed'])
        self.assertEqual(work['reviewedTrueRallies'], 2)


if __name__ == '__main__':
    unittest.main()
