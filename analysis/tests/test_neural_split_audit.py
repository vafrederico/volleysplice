import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('split_audit', Path(__file__).resolve().parents[2]
                                            / 'scripts/audit_neural_split_advisor.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record(**extra):
    return {'id': 'video', 'sourceGroup': 'group', 'durationSeconds': 30.,
            'ignoredIntervals': [], 'rallies': [{'start': 2., 'end': 5.}, {'start': 12., 'end': 18.}], **extra}


def parent():
    return {'id': 'P1', 'start': 1., 'end': 21., 'confidence': .9, 'included': True}


def children():
    p = parent()
    return [{**p, 'id': 'P1.0', 'parentId': 'P1', 'end': 12., 'startObserved': True, 'endObserved': False},
            {**p, 'id': 'P1.1', 'parentId': 'P1', 'start': 12., 'startObserved': True, 'endObserved': True}]


class GeometryTests(unittest.TestCase):
    def test_strict_gap(self):
        self.assertEqual(audit.export([[1, 3], [6, 8]], record(), 0), [(1., 3.), (6., 8.)])
        self.assertEqual(audit.export([[1, 3], [5.999, 8]], record(), 0), [(1., 8.)])

    def test_no_rejoin_across_ignored(self):
        self.assertEqual(audit.export([[1, 8]], record(ignoredIntervals=[[4, 5]]), 1),
                         [(0., 4.), (5., 9.)])

    def test_children_preserve_all_exports(self):
        result = audit.audit_coverage_lineage(record(), [parent()], children())
        self.assertTrue(result['passed'])
        self.assertEqual(result['paddingCases'], 4)

    def test_gap_is_rejected(self):
        bad = children(); bad[1]['start'] += .001
        with self.assertRaisesRegex(ValueError, 'gap or overlap'):
            audit.audit_coverage_lineage(record(), [parent()], bad)

    def test_overlap_is_rejected(self):
        bad = children(); bad[1]['start'] -= .001
        with self.assertRaisesRegex(ValueError, 'gap or overlap'):
            audit.audit_coverage_lineage(record(), [parent()], bad)

    def test_false_lineage_is_rejected(self):
        bad = children(); bad[0]['parentId'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'Unknown parent'):
            audit.audit_coverage_lineage(record(), [parent()], bad)

    def test_parent_metadata_preserved(self):
        bad = children(); bad[1]['confidence'] = .1
        with self.assertRaisesRegex(ValueError, 'metadata'):
            audit.audit_coverage_lineage(record(), [parent()], bad)

    def test_unsplit_identity_must_survive(self):
        bad = [{**parent(), 'id': 'new', 'parentId': 'P1'}]
        with self.assertRaisesRegex(ValueError, 'Unsplit parent identity'):
            audit.audit_coverage_lineage(record(), [parent()], bad)


class DurationTests(unittest.TestCase):
    def test_union_not_identity_count(self):
        rec = record(predictions=children())
        same = record(predictions=[parent()])
        self.assertEqual(audit.duration_rows([rec]), audit.duration_rows([same]))

    def test_frozen_export_survives_empty_event_list(self):
        rec = record(predictions=[])
        overrides = {'video': {str(p): audit.export([parent()], rec, p) for p in range(4)}}
        self.assertEqual(audit.duration_rows([rec], overrides),
                         audit.duration_rows([record(predictions=[parent()])]))

    def test_independently_matches_canonical_metrics(self):
        from analysis import neural_production_combinations as actual
        data = [record(predictions=[parent()], ignoredIntervals=[[4, 6], [20, 21]])]
        self.assertTrue(audit.audit_duration_records(data, actual.duration_rows(data))['passed'])

    def test_wrong_duration_detected(self):
        records = [record(predictions=[parent()])]
        bad = copy.deepcopy(audit.duration_rows(records)); bad[2]['P_pad'] += .01
        with self.assertRaisesRegex(ValueError, 'P_pad'):
            audit.audit_duration_records(records, bad)


class CandidateOracleTests(unittest.TestCase):
    def test_secondary_neural_event_and_gold_independence(self):
        times = list(range(30)); scores = [[.8, .0, .0, .0] for _ in times]
        nn = [{'id': 'n1', 'start': 3., 'end': 7.}, {'id': 'n2', 'start': 12., 'end': 17.}]
        result = audit.reconstruct_candidates(record(), [parent()], nn, times, scores, 'event_starts')
        self.assertEqual(result, [{'parentId': 'P1', 'time': 12., 'priority': .8}])
        self.assertEqual(result, audit.reconstruct_candidates(record(rallies=[]), [parent()], nn,
                                                              times, scores, 'event_starts'))

    def test_neural_first_start_is_not_a_split(self):
        times = list(range(30)); scores = [[.8, .0, .0, .0] for _ in times]
        self.assertEqual(audit.reconstruct_candidates(record(), [parent()],
                         [{'id': 'n1', 'start': 12., 'end': 17.}], times, scores, 'event_starts'), [])

    def test_heads_require_preceding_end_or_valley(self):
        times = list(range(30)); scores = [[.8, .0, .0, .0] for _ in times]
        scores[12][1] = .8
        self.assertEqual(audit.reconstruct_candidates(record(), [parent()], [], times, scores, 'head_evidence'), [])
        scores[11][2] = .3
        self.assertEqual(audit.reconstruct_candidates(record(), [parent()], [], times, scores, 'head_evidence'),
                         [{'parentId': 'P1', 'time': 12., 'priority': .8}])

    def test_plateau_uses_earliest_tick(self):
        times = list(range(30)); scores = [[.1, .0, .0, .0] for _ in times]
        scores[12][1] = scores[13][1] = .8
        self.assertEqual(audit.reconstruct_candidates(record(), [parent()], [], times, scores, 'head_evidence'),
                         [{'parentId': 'P1', 'time': 12., 'priority': .8}])

    def test_ignored_component_never_supplies_prior_event(self):
        times = list(range(30)); scores = [[.8, .0, .0, .0] for _ in times]
        nn = [{'id': 'n1', 'start': 3., 'end': 7.}, {'id': 'n2', 'start': 12., 'end': 17.}]
        self.assertEqual(audit.reconstruct_candidates(record(ignoredIntervals=[[8, 10]]), [parent()],
                                                     nn, times, scores, 'event_starts'), [])

    def test_cleanup_support_half_is_not_flagged(self):
        p = {'id': 'P1', 'start': 0., 'end': 20.}
        got = audit.cleanup_support(record(), [p], [{'start': 2., 'end': 8.}])['P1']
        self.assertEqual(got, {'supportFraction': .5, 'flagged': False, 'priority': .5})

    def test_cleanup_does_not_dilate_across_ignored_barrier(self):
        p = {'id': 'P1', 'start': 10., 'end': 12.}
        got = audit.cleanup_support(record(ignoredIntervals=[[9., 10.]]), [p], [{'start': 5., 'end': 9.5}])['P1']
        self.assertEqual(got['supportFraction'], 0.)


class IntegrationTests(unittest.TestCase):
    def fixture(self):
        from analysis import neural_split_advisor as advisor
        from analysis import neural_split_review as review
        from analysis import neural_split_metrics as metrics
        import numpy as np
        rec = record(durationSeconds=120., productionEvents=[parent(), {'id': 'false', 'start': 40., 'end': 42.}])
        times = np.arange(0., 120., .25)
        scores = np.zeros((len(times), 4), dtype=float)
        scores[:, 0] = .8; scores[times == 12.25, 1] = .8; scores[times == 11.5, 2] = .5
        neural = [{'id': 'n1', 'start': 2., 'end': 5.}, {'id': 'n2', 'start': 12.5, 'end': 18.}]
        return advisor, review, metrics, rec, times, scores, neural

    def test_scalar_candidates_agree_all_policies(self):
        advisor, _, _, rec, times, scores, neural = self.fixture()
        for policy in advisor.POLICIES:
            actual = advisor.split_proposals(rec, rec['productionEvents'], neural, times, scores, policy)
            self.assertTrue(audit.audit_candidates(rec, rec['productionEvents'], neural,
                                                  times, scores, policy, actual)['passed'])
            output = advisor.apply_splits(rec['productionEvents'], actual)
            self.assertTrue(audit.audit_apply_splits(rec, rec['productionEvents'], actual, output)['passed'])

    def test_cleanup_and_queue_geometry(self):
        advisor, review, _, rec, times, scores, neural = self.fixture()
        parents = rec['productionEvents']
        splits = advisor.split_proposals(rec, parents, neural, times, scores, 'event_starts')
        cleanup = advisor.cleanup_proposals(rec, parents, neural)
        self.assertTrue(audit.audit_cleanup(rec, parents, neural, cleanup)['passed'])
        for inventory in ('split_only', 'cleanup_only', 'combined'):
            jobs = review.jobs(rec, parents, splits, cleanup, inventory)
            for ranker in ('chronological', 'evidence'):
                queues = review.queues(rec, jobs, ranker)
                self.assertTrue(audit.audit_jobs_queues(rec, parents, splits, cleanup, inventory,
                                                       ranker, jobs, queues)['passed'])

    def test_perfect_human_has_no_unproposed_split_and_no_true_core_loss(self):
        advisor, review, metrics, rec, times, scores, neural = self.fixture()
        parents = rec['productionEvents']
        splits = advisor.split_proposals(rec, parents, neural, times, scores, 'event_starts')
        cleanup = advisor.cleanup_proposals(rec, parents, neural)
        jobs = review.jobs(rec, parents, splits, cleanup, 'combined')
        queues = review.queues(rec, jobs, 'evidence')
        for queue in queues:
            actual = review.human_edit(rec, splits, cleanup, queue, advisor, metrics)
            self.assertTrue(audit.audit_human(rec, splits, cleanup, queue, actual)['passed'])
        self.assertEqual(queues[-1]['selectedParentIds'], ['false', 'P1'])
        self.assertEqual(actual['acceptedSplitCount'], 1)
        self.assertEqual(actual['removedFalseParentIds'], ['false'])

    def test_invented_human_start_rejected(self):
        advisor, review, metrics, rec, times, scores, neural = self.fixture()
        parents = rec['productionEvents']
        splits = advisor.split_proposals(rec, parents, neural, times, scores, 'event_starts')
        cleanup = advisor.cleanup_proposals(rec, parents, neural)
        jobs = review.jobs(rec, parents, splits, cleanup, 'combined')
        queue = review.queues(rec, jobs, 'evidence')[-1]
        actual = review.human_edit(rec, splits, cleanup, queue, advisor, metrics)
        actual['acceptedSplits'][0]['time'] += .1
        with self.assertRaisesRegex(ValueError, 'acceptedSplit.time'):
            audit.audit_human(rec, splits, cleanup, queue, actual)

    def test_bad_queue_workload_rejected(self):
        advisor, review, _, rec, times, scores, neural = self.fixture()
        parents = rec['productionEvents']
        splits = advisor.split_proposals(rec, parents, neural, times, scores, 'event_starts')
        cleanup = advisor.cleanup_proposals(rec, parents, neural)
        jobs = review.jobs(rec, parents, splits, cleanup, 'combined')
        queues = review.queues(rec, jobs, 'evidence'); queues[-1]['reviewSeconds'] += 1.
        with self.assertRaisesRegex(ValueError, 'reviewSeconds'):
            audit.audit_jobs_queues(rec, parents, splits, cleanup, 'combined', 'evidence', jobs, queues)

    def test_split_metrics_all_tolerances_and_pooled_scopes(self):
        advisor, _, metrics, rec, times, scores, neural = self.fixture()
        proposals = advisor.split_proposals(rec, rec['productionEvents'], neural, times, scores, 'event_starts')
        data = [{**rec, 'splitProposals': proposals, 'predictions': advisor.apply_splits(rec['productionEvents'], proposals)}]
        actual = metrics.evaluate_split_proposals(data)
        self.assertTrue(audit.audit_split_metrics(data, actual)['passed'])
        actual['pooled']['splitLocalization']['1']['matched'] += 1
        with self.assertRaisesRegex(ValueError, 'matched'):
            audit.audit_split_metrics(data, actual)

    def test_split_metrics_duplicates_masks_and_no_target(self):
        from analysis import neural_split_metrics as metrics
        base = [{'id': 'p', 'start': 0., 'end': 25.}]
        rec = record(productionEvents=base, predictions=base,
                     rallies=[{'start': 1., 'end': 3.}, {'start': 7., 'end': 10.}, {'start': 14., 'end': 19.}],
                     ignoredIntervals=[[9., 11.]],
                     splitProposals=[{'id': 's1', 'parentId': 'p', 'time': 7.},
                                     {'id': 's2', 'parentId': 'p', 'time': 14.},
                                     {'id': 's3', 'parentId': 'p', 'time': 14.}])
        actual = metrics.evaluate_split_proposals([rec])
        self.assertTrue(audit.audit_split_metrics([rec], actual)['passed'])


if __name__ == '__main__':
    unittest.main()
