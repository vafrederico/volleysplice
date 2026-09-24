import unittest

from analysis import neural_production_combinations as c


class CombinationTests(unittest.TestCase):
    def test_protected_component_includes_nonoverlapping_tails(self):
        old, new = [[1, 4]], [[8, 10]]
        # Expanded intervals touch at 6; full original component is protected.
        self.assertEqual(c.serial(c.combine(old+new, [], old, new, 30, 'guarded_trim')), old+new)

    def test_exact_agreement_gap_remains_separate(self):
        groups = c.agreement_components([[1, 4]], [[8.5, 10]], 30)
        self.assertEqual([g['both'] for g in groups], [False, False])

    def test_component_preserving_and_trimming_are_distinct(self):
        old = [[2, 10]]
        n = [[5, 6]]
        self.assertEqual(c.serial(c.combine(old, n, old, [], 20, 'guarded_trim')), [[3., 8.]])
        self.assertEqual(c.serial(c.combine(old, n, old, [], 20, 'guarded_component_rejection')), old)

    def test_guard_never_restores_already_suppressed_production(self):
        old, new = [[1, 4]], [[8, 10]]
        self.assertEqual(c.serial(c.combine([[1, 3]], [], old, new, 30, 'guarded_trim')), [[1., 3.]])

    def test_dilated_support_has_no_export_gap_join(self):
        self.assertEqual(c.serial(c.dilate([[5, 6], [11, 12]], 2, 20)), [[3., 8.], [9., 14.]])

    def test_majority(self):
        self.assertEqual(c.serial(c.majority([[0, 5]], [[3, 8]], [[7, 10]])), [[3., 5.], [7., 8.]])

    def test_accounting_ignored_and_exact_three_gap(self):
        r = {'id': 'r', 'sourceGroup': 'g', 'durationSeconds': 20,
             'rallies': [[2, 5]], 'predictions': [[2, 5], [8, 10]], 'ignoredIntervals': [[9, 11]]}
        row = c.duration_rows([r])[0]
        self.assertEqual(row['evaluableVideoSeconds'], 18)
        self.assertEqual(row['paddedModelExportSeconds'], 4)
        self.assertEqual(row['incorrectExportSeconds'], 1)
        self.assertEqual(row['correctlyRemovedSeconds'], 14)
        self.assertEqual(row['incorrectlyRemovedSeconds'], 0)

    def test_review_is_label_blind_and_oracle_bound_is_limited(self):
        r = {'id': 'r', 'sourceGroup': 'g', 'durationSeconds': 30,
             'rallies': [[2, 5], [20, 21]], 'predictions': [[2, 5], [10, 11]], 'ignoredIntervals': []}
        n = {'r': [[2, 5], [20, 21]]}
        a = c.review_diagnostic([r], n, 'bidirectional_review')
        b = c.review_diagnostic([{**r, 'rallies': [[12, 13]]}], n, 'bidirectional_review')
        for x, y in zip(a['queue'], b['queue']):
            self.assertEqual(x['perRecording'][0]['disputedIntervals'], y['perRecording'][0]['disputedIntervals'])
            self.assertEqual(x['perRecording'][0]['reviewIntervals'], y['perRecording'][0]['reviewIntervals'])
        self.assertEqual(a['oracleDurationMetrics'][0]['F1_padP_coreR'], 1)
        suppress = c.review_diagnostic([r], n, 'suppression_review')
        self.assertEqual(suppress['oracleDurationMetrics'][0]['missedCoreSeconds'], 1)

    def test_override_is_not_padded_or_rejoined(self):
        r = {'id': 'r', 'sourceGroup': 'g', 'durationSeconds': 20,
             'rallies': [[2, 5]], 'predictions': [], 'ignoredIntervals': [[3, 4]]}
        overrides = {'r': {str(p): [[2, 3], [4, 5], [6, 7]] for p in c.PADS}}
        self.assertEqual([x['paddedModelExportSeconds'] for x in c.duration_rows([r], overrides)], [3]*4)


if __name__ == '__main__':
    unittest.main()
