import importlib.util
import copy
from pathlib import Path
import random
import unittest

from analysis import neural_rally_review_proposals as study
from analysis.neural_rally_identity_metrics import evaluate_rally_identities


PATH = Path(__file__).resolve().parents[2] / 'scripts/audit-neural-rally-review-proposals.py'
SPEC = importlib.util.spec_from_file_location('rally_review_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class IntervalPrimitiveTests(unittest.TestCase):
    def test_export_joins_strictly_below_three_seconds(self):
        rec = {'durationSeconds': 30., 'ignoredIntervals': []}
        self.assertEqual(audit.export([(2., 4.), (7., 9.)], rec, 0), [(2., 4.), (7., 9.)])
        self.assertEqual(audit.export([(2., 4.), (6.999, 9.)], rec, 0), [(2., 9.)])

    def test_ignored_hole_is_not_rejoined(self):
        rec = {'durationSeconds': 30., 'ignoredIntervals': [[4., 5.]]}
        self.assertEqual(audit.export([(2., 7.)], rec, 1), [(1., 4.), (5., 8.)])

    def test_union_cannot_stand_for_events(self):
        events = [(1., 3.), (3., 5.)]
        self.assertEqual(audit.union(events), [(1., 5.)])
        self.assertEqual(audit.event_matches(events, events), [(0, 0), (1, 1)])
        self.assertEqual(len(audit.event_matches(events, audit.union(events))), 1)

    def test_start_matching_is_one_to_one(self):
        truth = [(1., 5.), (1.5, 8.)]
        predictions = [(1.25, 6.)]
        self.assertEqual(len(audit.start_matches(truth, predictions, .5)), 1)

    def test_start_tolerance_is_inclusive(self):
        self.assertEqual(audit.start_matches([(1., 5.)], [(1.5, 5.)], .5), [(0, 0)])
        self.assertEqual(audit.start_matches([(1., 5.)], [(1.500001, 5.)], .5), [])

    def test_matching_cardinality_precedes_quality(self):
        truth = [(0., 4.), (4., 8.)]
        predictions = [(0., 2.), (0., 8.)]
        self.assertEqual(audit.event_matches(truth, predictions), [(0, 0), (1, 1)])


class EventEditorTests(unittest.TestCase):
    def test_empty_permission_preserves_touching_identities(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 5.]], 'ignoredIntervals': []}
        self.assertEqual(audit.reconstruct_events(rec, [[1., 3.], [3., 5.]], [])['events'],
                         [(1., 3.), (3., 5.)])

    def test_split_identity_survives_continuous_occupancy(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 3.], [3., 5.]], 'ignoredIntervals': []}
        self.assertEqual(audit.reconstruct_events(rec, [[1., 5.]], [[2., 4.]])['events'],
                         [(1., 3.), (3., 5.)])

    def test_incorrect_split_can_be_removed(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 5.]], 'ignoredIntervals': []}
        self.assertEqual(audit.reconstruct_events(rec, [[1., 3.], [3., 5.]], [[2., 4.]])['events'],
                         [(1., 5.)])

    def test_gold_boundaries_outside_permission_are_not_imported(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 8.]], 'ignoredIntervals': []}
        result = audit.reconstruct_events(rec, [], [[3., 5.]])
        self.assertEqual(result['events'], [(3., 5.)])
        self.assertEqual(result['goldMarkersInsidePermission'], [])

    def test_base_false_tail_outside_permission_stays(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 3.]], 'ignoredIntervals': []}
        result = audit.reconstruct_events(rec, [[1., 8.]], [[2., 5.]])
        self.assertEqual(result['events'], [(1., 3.), (5., 8.)])

    def test_closed_permission_removes_exact_base_marker(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 8.]], 'ignoredIntervals': []}
        result = audit.reconstruct_events(rec, [[1., 3.], [3., 8.]], [[3., 5.]])
        self.assertEqual(result['events'], [(1., 8.)])

    def test_ignored_time_never_becomes_event(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 8.]], 'ignoredIntervals': [[3., 5.]]}
        result = audit.reconstruct_events(rec, [], [[0., 10.]])
        self.assertEqual(result['events'], [(1., 3.), (5., 8.)])

    def test_overlapping_base_event_identity_is_rejected(self):
        rec = {'durationSeconds': 30., 'rallies': [], 'ignoredIntervals': []}
        with self.assertRaisesRegex(ValueError, 'identities overlap'):
            audit.reconstruct_events(rec, [[1., 5.], [3., 8.]], [])

    def test_audit_rejects_merged_touching_events(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 3.], [3., 5.]], 'ignoredIntervals': []}
        with self.assertRaisesRegex(ValueError, 'interval count differs'):
            audit.audit_edit_events(rec, [[1., 5.]], [[2., 4.]], [[1., 5.]])

    def test_audit_rejects_imported_unseen_boundary(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 8.]], 'ignoredIntervals': []}
        with self.assertRaisesRegex(ValueError, 'edited events'):
            audit.audit_edit_events(rec, [], [[3., 5.]], [[1., 8.]])

    def test_raw_ignored_base_occupancy_is_preserved(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 8.]], 'ignoredIntervals': [[3., 5.]]}
        self.assertEqual(audit.reconstruct_events(rec, [[2., 6.]], [[0., 10.]])['events'], [(1., 8.)])

    def test_full_editor_evidence_and_observed_flags(self):
        rec = {'durationSeconds': 30., 'rallies': [[1., 8.]], 'ignoredIntervals': []}
        queue = {'editWindows': [[3., 5.]]}
        result = study.edit_events(rec, [], queue['editWindows'])
        self.assertTrue(audit.audit_editor(rec, [], queue, result)['passed'])
        self.assertEqual(result['unobservedStarts'], 1)
        changed = copy.deepcopy(result)
        changed['events'][0]['startObserved'] = True
        with self.assertRaisesRegex(ValueError, 'startObserved'):
            audit.audit_editor(rec, [], queue, changed)

    def test_randomized_editor_permissions_are_local(self):
        generator = random.Random(1729)
        for _ in range(70):
            base = [[a, a + generator.uniform(.1, 6)] for a in range(1, 91, 10)]
            gold = [[a, a + generator.uniform(.1, 6)] for a in range(2, 92, 10)]
            rec = {'durationSeconds': 100., 'rallies': gold, 'ignoredIntervals': [[30., 32.]]}
            proposals = [[generator.uniform(0, 95), 0] for _ in range(8)]
            for row in proposals:
                row[1] = min(100., row[0] + generator.uniform(.1, 4))
            windows = audit.review_permission(rec, proposals)
            queue = {'editWindows': [list(x) for x in windows]}
            result = study.edit_events(rec, base, queue['editWindows'])
            self.assertTrue(audit.audit_editor(rec, base, queue, result)['passed'])


class BudgetAuditTests(unittest.TestCase):
    def setUp(self):
        self.record = {'durationSeconds': 200., 'ignoredIntervals': [[95., 100.]]}
        self.candidates = [
            {'id': 'q00000', 'start': 2., 'end': 3., 'priority': .4},
            {'id': 'q00001', 'start': 40., 'end': 50., 'priority': .8},
            {'id': 'q00002', 'start': 48., 'end': 52., 'priority': .6},
            {'id': 'q00003', 'start': 94., 'end': 101., 'priority': .9},
            {'id': 'q00004', 'start': 170., 'end': 171., 'priority': .5},
        ]

    def test_all_budget_queues_reconstruct(self):
        for ranker in ('chronological', 'evidence'):
            queues = study.budget_queues(self.record, self.candidates, ranker)
            self.assertTrue(audit.audit_budget_queues(self.record, self.candidates, ranker, queues)['passed'])

    def test_context_cannot_be_free(self):
        queues = study.budget_queues(self.record, self.candidates, 'evidence')
        queues[0]['reviewSeconds'] -= 1
        with self.assertRaisesRegex(ValueError, 'reviewSeconds'):
            audit.audit_budget_queues(self.record, self.candidates, 'evidence', queues)

    def test_selection_order_is_checked(self):
        queues = study.budget_queues(self.record, self.candidates, 'evidence')
        queues[-1]['selectedIds'].reverse()
        with self.assertRaisesRegex(ValueError, 'selected IDs'):
            audit.audit_budget_queues(self.record, self.candidates, 'evidence', queues)

    def test_randomized_nested_budget_selections(self):
        generator = random.Random(17)
        for _ in range(40):
            candidates = []
            for i in range(25):
                start = generator.uniform(0, 190)
                candidates.append({'id': f'q{i:05d}', 'start': start,
                                   'end': start + generator.uniform(.1, 8),
                                   'priority': generator.random()})
            for ranker in ('chronological', 'evidence'):
                queues = study.budget_queues(self.record, candidates, ranker)
                self.assertTrue(audit.audit_budget_queues(self.record, candidates, ranker, queues)['passed'])


class IdentityMetricAuditTests(unittest.TestCase):
    def record(self, truth, prediction, ignored=()):
        return {'id': 'fixture', 'sourceGroup': 'group', 'durationSeconds': 100.,
                'rallies': truth, 'predictions': prediction, 'ignoredIntervals': list(ignored)}

    def test_min_cost_matching_requires_alternating_path(self):
        # Greedy picks (0,0)=1 and loses cardinality; correct result is two .5s.
        edges = {(0, 0): 1., (0, 1): .5, (1, 0): .5}
        self.assertEqual(audit.optimal_bipartite(edges, 2, 2), (2, 1.))

    def test_independent_general_matching_against_exhaustive(self):
        generator = random.Random(91)
        for _ in range(50):
            rows, cols = generator.randrange(1, 5), generator.randrange(1, 5)
            edges = {(i, j): generator.randrange(11) / 10 for i in range(rows) for j in range(cols)
                     if generator.random() < .6}

            def enumerate_matchings(index=0, used=frozenset()):
                if index == rows:
                    return 0, 0.
                best = enumerate_matchings(index + 1, used)
                for j in range(cols):
                    if (index, j) in edges and j not in used:
                        count, weight = enumerate_matchings(index + 1, used | {j})
                        best = max(best, (count + 1, weight + edges[index, j]))
                return best

            expected = enumerate_matchings()
            got = audit.optimal_bipartite(edges, rows, cols)
            self.assertEqual(expected[0], got[0])
            self.assertAlmostEqual(expected[1], got[1])

    def test_touches_merges_splits_and_nested_predictions(self):
        cases = [
            self.record([[1., 3.], [3., 5.]], [[1., 3.], [3., 5.]]),
            self.record([[1., 3.], [4., 6.]], [[1., 6.]]),
            self.record([[1., 6.]], [[1., 3.], [3., 6.]]),
            self.record([[1., 4.], [6., 10.]], [[1., 10.], [1., 4.], [6., 10.]]),
            self.record([[1., 4.], [6., 10.]], [[1., 10.]], [[2., 3.]]),
            self.record([], [[1., 10.]]),
            self.record([[1., 10.]], []),
        ]
        for record in cases:
            result = evaluate_rally_identities([record])
            self.assertTrue(audit.audit_identity_metrics([record], result)['passed'])

    def test_observed_flags_and_masked_boundary_denominators(self):
        record = self.record([[10., 20.], [40., 50.]],
                             [dict(start=10., end=20., startObserved=False),
                              dict(start=39., end=50., endObserved=False)])
        result = evaluate_rally_identities([record])
        self.assertTrue(audit.audit_identity_metrics([record], result)['passed'])
        broken = copy.deepcopy(result)
        broken['recordings'][0]['observedStartLocalization']['1']['matched'] += 1
        with self.assertRaisesRegex(ValueError, 'observedStartLocalization'):
            audit.audit_identity_metrics([record], broken)

    def test_randomized_coverage_and_matching_cross_check(self):
        generator = random.Random(320)
        for _ in range(40):
            truth = [[a, a + generator.uniform(.2, 4.)] for a in range(1, 91, 10)]
            predicted = []
            for _ in range(generator.randrange(1, 17)):
                start = generator.uniform(0, 95)
                predicted.append([start, min(100., start + generator.uniform(.2, 15.))])
            record = self.record(truth, predicted, [[35., 36.]])
            result = evaluate_rally_identities([record])
            self.assertTrue(audit.audit_identity_metrics([record], result)['passed'])


if __name__ == '__main__':
    unittest.main()
