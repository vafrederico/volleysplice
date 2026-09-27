"""Independent final metric/mean audit regression cases."""
import copy
import importlib.util
from pathlib import Path
import unittest

from analysis.neural_generalization_results import metric_rows
from analysis.neural_generalization_report import summarize_draws

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('generalization_results_audit', ROOT/'scripts/audit-neural-generalization-results.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
helper = audit.module('generalization_result_metric_oracle', ROOT/'scripts/audit-neural-short-boost-intervals.py')


class GeneralizationResultsAuditTest(unittest.TestCase):
    def test_reuse_flags_cannot_replace_a_physical_owner_or_hide_a_duplicate(self):
        original = {'taskId': 'original', 'variant': 'original-corpus'}
        self.assertFalse(audit.reuse_required(original, {}, None))
        with self.assertRaisesRegex(ValueError, 'Physical/reused'):
            audit.reuse_required(original, {'trainingReuse': {}}, None)
        expanded = {'taskId': 'target', 'variant': 'expanded-wide-validation'}
        mapping = {'taskId': 'target', 'physicalOwnerTaskId': 'source'}
        self.assertTrue(audit.reuse_required(expanded, {'trainingReuse': {}}, mapping))
        with self.assertRaisesRegex(ValueError, 'Physical/reused'):
            audit.reuse_required(expanded, {}, mapping)
        with self.assertRaisesRegex(ValueError, 'lacks'):
            audit.reuse_required(expanded, {}, None)
        owner = {'taskId': 'target', 'physicalOwnerTaskId': 'target'}
        with self.assertRaisesRegex(ValueError, 'Physical/reused'):
            audit.reuse_required(expanded, {'trainingReuse': {}}, owner)

    def test_completed_evaluation_retains_only_task_display_evidence(self):
        result = {'taskId': 'fit', 'model': 'av-tcn', 'variant': 'original-corpus', 'draw': 3407,
                  'precision': 'fp32', 'panels': [{'recordingIds': ['video']}], 'task': {'path': 'task'},
                  'selection': {'path': 'selection'}, 'operatingPoints': {'large': {'raw': [1, 2, 3]}},
                  'floors': list(range(11))}
        compact = audit.task_display_evidence(result)
        self.assertEqual(set(compact), {'taskId', 'model', 'variant', 'draw', 'precision', 'panels', 'task', 'selection'})
        self.assertEqual(compact['panels'], result['panels'])
        self.assertIn('operatingPoints', result)

    def test_projection_cannot_narrow_original_eight_or_drop_a_seed(self):
        ids = [f'original-{i}' for i in range(8)]
        panel = {'panelId': 'historical-nested-exact', 'labelPolicy': 'exact-rallies', 'reviewStatus': 'manually-reviewed',
                 'sourcePolicy': 'source-held', 'expectedRecordingIds': ids,
                 'result': {'floors': [{'floorPercent': f} for f in range(90, 101)]}}
        models = {'dino_tcn_short_boost', 'dino_transformer'}
        report = {'cells': [{'model': model, 'seed': seed, 'precision': 'fp16', 'precisionSpecificSelection': False,
                            'panels': {'historical-nested-exact': copy.deepcopy(panel)}}
                           for model in models for seed in (3407, 1729, 20260918)]}
        audit.check_historical_scope(report, ids, models, 'fp16')
        altered = copy.deepcopy(report); altered['cells'][0]['panels']['historical-nested-exact']['expectedRecordingIds'] = ids[:-1]
        with self.assertRaisesRegex(ValueError, 'narrowed or changed'):
            audit.check_historical_scope(altered, ids, models, 'fp16')
        report['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'model/seed'):
            audit.check_historical_scope(report, ids, models, 'fp16')

    def test_all_four_padding_and_three_label_policies_independent(self):
        base = {'id': 'fixture', 'sourceGroup': 'a', 'durationSeconds': 30.,
            'ignoredIntervals': [{'start': 7., 'end': 8.}],
            'predictions': [{'start': 2., 'end': 5.}, {'start': 8., 'end': 10.}, {'start': 16., 'end': 18.}]}
        exact = [{'start': 2.1, 'end': 5.}, {'start': 7.5, 'end': 10.}, {'start': 23., 'end': 27., 'tags': ['ace']}]
        for policy, field in [('exact-rallies', 'rallies'), ('reviewed-draft', 'reviewedLiveIntervals'), ('reviewed-export', 'humanExportIntervals')]:
            rows = [{**base, field: exact}]
            independent, guards = audit.independent_metrics(helper, rows, policy)
            audit.compare(metric_rows(rows, policy), independent)
            self.assertEqual(len(independent), 4)
            self.assertEqual(bool(guards), policy == 'exact-rallies')
            if guards:
                self.assertEqual(guards[0]['slices']['duration_gt_3s']['completeRallyLosses'], 1)
            else:
                self.assertTrue(all(r['missedCoreSeconds'] is None and r['eventF1'] is None for r in independent))

    def test_failed_draw_never_yields_favorable_mean_and_scopes_do_not_mix(self):
        rows = []
        for draw in (3407, 1729, 20260918):
            rows.append({'model': 'av-tcn', 'variant': 'original-corpus', 'precision': 'fp32',
                'panelId': 'common-unseen', 'labelPolicy': 'exact-rallies', 'productionFilter': 'all',
                'floorPercent': 99, 'paddingSeconds': 2, 'draw': draw, 'status': 'available', 'scopeId': 'one',
                **{key: .9 for key in audit.METRICS}})
        expected = {('original-corpus', 'av-tcn', 'fp32'): [3407, 1729, 20260918]}
        audit.compare(audit.complete_draw_summaries(rows, expected), summarize_draws(rows, expected))
        rows[2]['status'] = 'infeasible-inner-recall'
        result, = audit.complete_draw_summaries(rows, expected)
        self.assertEqual(result['status'], 'incomplete-registered-draws')
        self.assertIsNone(result['f1Value'])
        rows[2]['scopeId'] = 'other'
        self.assertEqual(audit.complete_draw_summaries(rows, expected), [])
        with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
            audit.complete_draw_summaries(rows[:2], expected)

    def test_unknown_and_calibration_exposure_are_not_strictly_clean(self):
        p = {'directTrainingHeads': [], 'sameSourceTrainingHeads': [], 'sameGroupTrainingHeads': [],
             'directCalibrationHeads': [], 'sameGroupCalibrationHeads': ['suppression'], 'unknownScopeHeads': [],
             'trainingOrRelated': False, 'primaryTrainingClean': True, 'strictNoFitOrCalibration': False}
        row = {'productionExposure': {'rallyPipeline': p}}
        self.assertTrue(audit.clean(row, 'no-production-training'))
        self.assertFalse(audit.clean(row, 'no-production-training-or-calibration'))
        p['unknownScopeHeads'] = ['untraced']; p['primaryTrainingClean'] = False
        self.assertFalse(audit.clean(row, 'no-production-training'))
        p['primaryTrainingClean'] = True
        with self.assertRaisesRegex(ValueError, 'lineage summary'):
            audit.clean(row, 'no-production-training')


if __name__ == '__main__': unittest.main()
