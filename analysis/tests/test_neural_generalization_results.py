import unittest
import numpy as np

from analysis.neural_generalization_results import manifest_example, metric_rows, is_clean
from analysis.neural_recall_sweep import panel_rows


class GeneralizationResultsTests(unittest.TestCase):
    def row(self, policy='exact-core'):
        return {'id': 'clip', 'sourceGroup': 'court-a', 'durationSeconds': 30.,
                'scoringPolicy': policy, 'rallies': [{'start': 10., 'end': 12.}],
                'keepTargets': [{'start': 8., 'end': 14.}], 'ignoredIntervals': [],
                'gameWindow': {'start': 5., 'end': 25.}}

    def test_ignored_gold_never_resets_inference_validity(self):
        row = self.row(); row['ignoredIntervals'] = [{'start': 10., 'end': 11.}]
        times = np.arange(0., 30., .25)
        example = manifest_example(row, times)
        self.assertTrue(example.valid.all())
        self.assertEqual(example.ignored[0].start, 10.)
        self.assertIs(example.times, times)

    def test_export_reference_is_not_padded_again(self):
        example = manifest_example(self.row('export-coverage'), np.arange(0., 30., .25))
        rows = panel_rows([example], {'clip': [type(example.truth[0])(10., 12.)]}, 'reviewed-export')
        metrics = metric_rows(rows, 'reviewed-export')
        self.assertEqual(metrics[0]['humanExportSeconds'], 6.)
        self.assertAlmostEqual(metrics[0]['recallValue'], 1/3)
        self.assertEqual(metrics[2]['humanExportSeconds'], 6.)
        self.assertEqual(metrics[2]['f1Value'], 1.)
        self.assertEqual(metrics[2]['correctlyRemovedSeconds'], 14.)
        self.assertIsNone(metrics[2]['completeRallyLosses'])

    def test_rally_loss_guardrail_tracks_each_padding(self):
        example = manifest_example(self.row(), np.arange(0., 30., .25))
        rows = panel_rows([example], {'clip': [type(example.truth[0])(13., 15.)]}, 'exact-rallies')
        metrics = metric_rows(rows, 'exact-rallies')
        self.assertEqual(metrics[0]['completeRallyLosses'], 1)
        self.assertEqual(metrics[2]['completeRallyLosses'], 0)
        self.assertEqual(metrics[2]['missedCoreSeconds'], 1.)
        self.assertEqual(metrics[2]['wantedExportOmittedSeconds'], 3.)
        self.assertEqual(metrics[2]['incorrectExportSeconds'], 3.)
        self.assertEqual(metrics[2]['correctlyRemovedSeconds'], 21.)

    def test_ignored_time_subtracted_after_join_without_rejoining(self):
        row = self.row(); row['rallies'] = [{'start': 4., 'end': 5.}, {'start': 8., 'end': 9.}]
        row['ignoredIntervals'] = [{'start': 5., 'end': 8.}]
        example = manifest_example(row, np.arange(0., 30., .25))
        rows = panel_rows([example], {'clip': list(example.truth)}, 'exact-rallies')
        metrics = metric_rows(rows, 'exact-rallies')
        self.assertEqual(metrics[0]['exportSeconds'], 2.)
        self.assertEqual(metrics[1]['exportSeconds'], 4.)
        self.assertEqual(metrics[2]['exportSeconds'], 6.)
        self.assertEqual(metrics[2]['f1Value'], 1.)

    def test_calibration_related_is_training_clean_but_not_strict_clean(self):
        record = {'productionExposure': {'rallyPipeline': {
            'primaryTrainingClean': True, 'strictNoFitOrCalibration': False}}}
        self.assertTrue(is_clean(record, 'all'))
        self.assertTrue(is_clean(record, 'no-production-training'))
        self.assertFalse(is_clean(record, 'no-production-training-or-calibration'))


if __name__ == '__main__':
    unittest.main()
