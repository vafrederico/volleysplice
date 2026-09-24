from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2]/'scripts/audit-neural-short-boost-intervals.py'
SPEC = importlib.util.spec_from_file_location('independent_short_boost_interval_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(truth, predictions=(), ignored=(), *, rid='sample', duration=30):
    def intervals(values):
        return [dict(value) if isinstance(value, dict) else {'start': value[0], 'end': value[1]} for value in values]
    return {'id': rid, 'sourceGroup': 'group-'+rid, 'durationSeconds': duration,
            'rallies': intervals(truth), 'predictions': intervals(predictions), 'ignoredIntervals': intervals(ignored)}


class IndependentIntervalArithmeticTests(unittest.TestCase):
    def test_serialized_gold_omits_only_nonmodel_annotation_metadata(self):
        gold = [{'start': 1, 'end': 2, 'tags': ['ace', 'reviewed'], 'notes': 'Manual review'},
                {'start': 3, 'end': 4, 'tags': [], 'notes': 'Another rally'}]
        stored = [{'start': 1., 'end': 2., 'tags': ['ace', 'reviewed']}, {'start': 3., 'end': 4.}]
        audit.verify_serialized_gold(stored, gold)
        audit.verify_serialized_gold([{'start': 0., 'end': 1.}], [{'start': 0, 'end': 1, 'reason': 'Off camera'}])
        audit.verify_serialized_gold([], [])

    def test_serialized_gold_rejects_boundary_tag_order_count_and_ignored_changes(self):
        gold = [{'start': 1, 'end': 2, 'tags': ['ace', 'reviewed'], 'notes': 'Manual review'},
                {'start': 3, 'end': 4, 'notes': 'Another rally'}]
        stored = [{'start': 1., 'end': 2., 'tags': ['ace', 'reviewed']}, {'start': 3., 'end': 4.}]
        mutations = []
        for field, value in [('start', 1.000000001), ('end', 2.000000001), ('tags', ['ace']),
                             ('tags', ['reviewed', 'ace']), ('tags', ['ace', 'reviewed', 'ace'])]:
            changed = copy.deepcopy(stored)
            changed[0][field] = value
            mutations.append(changed)
        mutations.extend([list(reversed(stored)), stored[:-1], stored+[stored[-1]],
                          [{**stored[0], 'notes': 'Unexpected serialized field'}, stored[1]]])
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit.verify_serialized_gold(changed, gold)
        ignored = [{'start': 0, 'end': 1, 'reason': 'Off camera'}]
        for changed in [[], [{'start': 0, 'end': 1.000000001}], [{'start': 0, 'end': 1, 'tags': ['ace']}]]:
            with self.subTest(ignored=changed), self.assertRaises(ValueError):
                audit.verify_serialized_gold(changed, ignored)

    def test_endpoint_sweep_handles_overlap_touching_duplicates_and_nested_holes(self):
        left = [(0, 3), (1, 2), (3, 4), (7, 9), (7, 9), (11, 11)]
        right = [(1, 2), (1.5, 3.5), (8, 10)]
        self.assertEqual(audit.boolean_intervals(left), [(0., 4.), (7., 9.)])
        self.assertEqual(audit.boolean_intervals(left, right, 'intersection'), [(1., 3.5), (8., 9.)])
        self.assertEqual(audit.boolean_intervals(left, right, 'difference'), [(0., 1.), (3.5, 4.), (7., 8.)])
        self.assertEqual(audit.boolean_intervals([], right, 'intersection'), [])
        self.assertEqual(audit.boolean_intervals([], [], 'difference'), [])
        with self.assertRaisesRegex(ValueError, 'Reversed'):
            audit.boolean_intervals([(2, 1)])

    def test_zero_padding_still_joins_strictly_less_than_three_not_equal(self):
        self.assertEqual(audit.padded_union([(0, 1), (4, 5)], 10, 0), [(0., 1.), (4., 5.)])
        self.assertEqual(audit.padded_union([(0, 1), (3.999, 5)], 10, 0), [(0., 5.)])
        self.assertEqual(audit.padded_union([(0, 1), (1, 2)], 10, 0), [(0., 2.)])
        self.assertEqual(audit.padded_union([(0, 1), (6, 7)], 10, 1), [(0., 2.), (5., 8.)])
        self.assertEqual(audit.padded_union([(0, 1), (5.999, 7)], 10, 1), [(0., 8.)])

    def test_original_ranges_are_clipped_before_padding_and_outside_events_disappear(self):
        record = audit.parse_record(row([(-1, 2), (8, 12)], [(-3, -1), (-1, 1), (9, 12), (11, 12)], duration=10))
        self.assertEqual(audit.ranges(record['predictions']), [(0., 1.), (9., 10.)])
        self.assertEqual(audit.export_union(record, 'predictions', 2), [(0., 3.), (7., 10.)])
        self.assertEqual(audit.export_union(record, 'rallies', 2), [(0., 10.)])

    def test_ignored_subtraction_occurs_after_joining_and_never_rejoins_holes(self):
        record = audit.parse_record(row([(0, 4)], [(0, 1), (3, 4)], [(1.5, 2.5)], duration=5))
        self.assertEqual(audit.export_union(record, 'predictions', 0), [(0., 1.5), (2.5, 4.)])
        metric = audit.pooled_metric([record], 0)
        self.assertEqual(metric['paddedModelExportSeconds'], 3)
        self.assertEqual(metric['coreHumanSeconds'], 3)
        self.assertEqual(metric['outputCropCount'], 2)
        self.assertEqual(metric['inputCropCount'], 2)
        self.assertEqual(metric['F1_padP_coreR'], 1)

    def test_pooled_ratios_are_duration_weighted_not_means_of_recording_scores(self):
        long = audit.parse_record(row([(0, 80)], [(0, 80)], rid='long', duration=100))
        short = audit.parse_record(row([(0, 20)], [], rid='short', duration=100))
        metric = audit.pooled_metric([long, short], 0)
        self.assertEqual(metric['P_pad'], 1)
        self.assertEqual(metric['R_core'], .8)
        self.assertAlmostEqual(metric['F1_padP_coreR'], 8/9)
        self.assertEqual(metric['paddedModelExportSeconds'], 80)
        self.assertEqual(metric['paddedHumanExportSeconds'], 100)
        self.assertEqual(metric['exportDurationDifferenceSeconds'], -20)
        self.assertNotEqual(metric['F1_padP_coreR'], .5)

    def test_all_four_padding_cases_have_analytic_durations_and_precision(self):
        record = audit.parse_record(row([(8, 12)], [(6, 10)], duration=20))
        for padding, model_seconds, intersection, recall in ((0, 4, 2, .5), (1, 6, 4, .75), (2, 8, 6, 1), (3, 10, 8, 1)):
            with self.subTest(padding=padding):
                metric = audit.pooled_metric([record], padding)
                precision = intersection/model_seconds
                self.assertEqual(metric['paddedModelExportSeconds'], model_seconds)
                self.assertEqual(metric['paddedHumanExportSeconds'], model_seconds)
                self.assertEqual(metric['paddedPrecisionIntersectionSeconds'], intersection)
                self.assertEqual(metric['coreHumanSeconds'], 4)
                self.assertEqual(metric['R_core'], recall)
                self.assertEqual(metric['P_pad'], precision)
                self.assertAlmostEqual(metric['F1_padP_coreR'], 2*precision*recall/(precision+recall))

    def test_no_predictions_and_invalid_evaluation_universes(self):
        record = audit.parse_record(row([(1, 3)]))
        metric = audit.pooled_metric([record], 2)
        self.assertEqual((metric['P_pad'], metric['R_core'], metric['F1_padP_coreR']), (0, 0, 0))
        for records in ([], [record, record], [audit.parse_record(row([(1, 3)], ignored=[(0, 4)]))]):
            with self.assertRaises(ValueError):
                audit.pooled_metric(records, 2)
        for malformed in (row([(1, 3), (2, 4)]), row([(1, float('nan'))]), row([(True, 4)])):
            with self.assertRaises(ValueError):
                audit.parse_record(malformed)

    def test_original_rally_identity_survives_ignored_splits_and_short_slice_uses_original_length(self):
        record = audit.parse_record(row([
            {'start': 0, 'end': 10, 'tags': ['ace']},
            {'start': 12, 'end': 14, 'tags': ['service-fault']},
            {'start': 16, 'end': 18, 'tags': ['ace']}], [(0, 1)], [(1, 9), (12, 14)], duration=20))
        coverage = audit.event_coverage([record], 'coreCoverage')
        self.assertEqual([r['truthIndex'] for r in coverage['rallies']], [0, 2])
        self.assertEqual(coverage['originalRallies'], 3)
        self.assertEqual(coverage['fullyIgnoredRallies'], 1)
        self.assertEqual(coverage['evaluableRallies'], 2)
        self.assertEqual(coverage['rallies'][0]['evaluableCoreSeconds'], 2)
        self.assertEqual(coverage['rallies'][0]['coverage'], .5)
        self.assertTrue(coverage['rallies'][0]['partiallyLost'])
        self.assertTrue(coverage['rallies'][1]['completelyLost'])
        slices = audit.slices(coverage)
        self.assertEqual(slices['duration_gt_3s']['evaluableRallies'], 1)
        self.assertEqual(slices['duration_le_3s']['evaluableRallies'], 1)
        self.assertEqual(slices['ace']['evaluableRallies'], 2)
        self.assertIsNone(slices['service_fault']['coreRecall'])

    def test_full_partial_and_complete_losses_use_two_second_exports(self):
        record = audit.parse_record(row([(4, 6), (12, 20), (25, 27)], [(4, 5), (12, 13)], duration=30))
        coverage = audit.event_coverage([record], 'primaryExportCoverage')
        self.assertEqual((coverage['fullyCoveredRallies'], coverage['partialRallyLosses'], coverage['completeRallyLosses']), (1, 1, 1))
        self.assertEqual([r['retainedCoreSeconds'] for r in coverage['rallies']], [2, 3, 0])
        self.assertEqual(coverage['coreRecall'], 5/12)
        self.assertEqual(coverage['coreRecall'], audit.pooled_metric([record], 2)['R_core'])

    def test_raw_core_coverage_does_not_apply_zero_padding_gap_join(self):
        record = audit.parse_record(row([(1.5, 2.5)], [(0, 1), (3, 4)], duration=6))
        self.assertEqual(audit.event_coverage([record], 'coreCoverage')['coreRecall'], 0)
        self.assertEqual(audit.pooled_metric([record], 0)['R_core'], 1)
        self.assertEqual(audit.event_coverage([record], 'primaryExportCoverage')['coreRecall'], 1)

    def test_independent_results_compare_full_canonical_schema_and_detect_corruption(self):
        # This comparison checks adapter compatibility; arithmetic is independently
        # established by the analytic cases above, not inferred from this oracle.
        from analysis.neural_evaluation import evaluate_predictions
        source = [row([(1, 4), (8, 13), (20, 22)], [(0, 2), (4, 5), (11, 13)], [(2, 3), (21, 22)])]
        records = [audit.parse_record(value) for value in source]
        expected = evaluate_predictions(source)
        result = audit.compare_scope(records, expected, 'fixture')
        self.assertEqual(len(result['padding']), 4)
        corrupted = copy.deepcopy(expected)
        corrupted['padding'][0]['paddedModelExportSeconds'] += .01
        with self.assertRaisesRegex(ValueError, 'paddedModelExportSeconds'):
            audit.compare_scope(records, corrupted, 'corrupt-duration')
        corrupted = copy.deepcopy(expected)
        corrupted['guardrails']['primaryExportCoverage']['rallies'][0]['truthIndex'] = 999
        with self.assertRaisesRegex(ValueError, 'original event missing'):
            audit.compare_scope(records, corrupted, 'corrupt-identity')

    def test_final_audit_refuses_to_read_or_write_without_completed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'requires completed report'):
                audit.audit(root/'nonexistent-manifest.json', root/'nonexistent-study')
            self.assertEqual(list(root.iterdir()), [])

    def test_pts_corrected_coverage_requires_new_refits_but_exact_and_draft_reuse(self):
        reference = {'path': '/old/study', 'reportSha256': 'old-report', 'contractSha256': 'old-contract'}
        for cohort in ('exact', 'draft'):
            result = {'cohort': cohort, 'kind': 'tcn', 'lossArm': 'baseline', 'seed': 3407,
                      'origin': {'type': 'reused-reference', 'fitRoot': f'/old/study/fits/{cohort}/tcn/3407',
                                 'studyPath': reference['path'], 'reportSha256': reference['reportSha256'],
                                 'referenceContractSha256': reference['contractSha256']}}
            self.assertTrue(audit.refit_origin(result, Path('/new/study'), reference)[1])
        result = {'cohort': 'reviewed_export', 'kind': 'tcn', 'lossArm': 'baseline', 'seed': 3407,
                  'origin': {'type': 'trained', 'fitRoot': '/new/study/fits/reviewed_export/tcn/baseline/3407'}}
        self.assertFalse(audit.refit_origin(result, Path('/new/study'), reference)[1])
        result['origin'] = {'type': 'reused-reference', 'fitRoot': '/old/study/fits/reviewed_export/tcn/3407'}
        with self.assertRaisesRegex(ValueError, 'origin path'):
            audit.refit_origin(result, Path('/new/study'), reference)


if __name__ == '__main__':
    unittest.main()
