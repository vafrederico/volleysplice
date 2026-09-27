import unittest
from analysis.neural_generalization_report import summarize_draws, point_rows, METRICS


class GeneralizationReportTests(unittest.TestCase):
    def rows(self):
        return [{'model': 'av-tcn', 'variant': 'expanded-medium', 'precision': 'fp32',
            'panelId': 'common-unseen', 'labelPolicy': 'exact-rallies', 'productionFilter': 'all',
            'floorPercent': 99, 'paddingSeconds': 2, 'draw': draw, 'scopeId': 'same-video',
            'status': 'available', **{name: .5+index*.1 for name in METRICS}}
            for index, draw in enumerate((3407, 1729, 20260918, 20260923))]

    def expected(self):
        return {('expanded-medium', 'av-tcn', 'fp32'): [3407, 1729, 20260918, 20260923]}

    def test_all_draws_mean_has_matched_population_and_range(self):
        result, = summarize_draws(self.rows(), self.expected())
        self.assertAlmostEqual(result['f1Value'], .65)
        self.assertEqual(result['f1ValueRange'], [.5, .8])
        self.assertEqual(result['status'], 'available')

    def test_infeasible_draw_cannot_be_dropped_from_mean(self):
        rows = self.rows(); rows[-1].update(status='infeasible-inner-recall', f1Value=None)
        result, = summarize_draws(rows, self.expected())
        self.assertIsNone(result['f1Value'])
        self.assertEqual(result['status'], 'incomplete-registered-draws')
        self.assertEqual(len(result['completeDraws']), 3)

    def test_missing_draw_fails_instead_of_summarizing(self):
        with self.assertRaisesRegex(ValueError, 'population incomplete'):
            summarize_draws(self.rows()[:-1], self.expected())

    def test_unlike_video_populations_are_not_averaged(self):
        rows = self.rows(); rows[-1]['scopeId'] = 'another-video'
        self.assertEqual(summarize_draws(rows, self.expected()), [])

    def test_infeasible_point_retains_expected_scope_and_four_paddings(self):
        scopes = {}
        result = {'model': 'av-tcn', 'variant': 'expanded-medium', 'precision': 'fp32', 'draw': 3407,
            'operatingPoints': {}, 'floors': [{'floorPercent': 100, 'status': 'infeasible-inner-recall', 'operatingPointKey': None}],
            'panels': [{'panelId': 'common-unseen', 'labelPolicy': 'exact-rallies', 'productionFilter': 'all', 'recordingIds': ['video']} ]}
        rows = point_rows(result, scopes)
        self.assertEqual(len(rows), 4)
        self.assertEqual(scopes[rows[0]['scopeId']], ['video'])
        self.assertTrue(all(r['f1Value'] is None for r in rows))


if __name__ == '__main__':
    unittest.main()
