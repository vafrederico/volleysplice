import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[2]/'scripts/audit-neural-human-review-summary.py'
SPEC = importlib.util.spec_from_file_location('human_review_summary_audit', PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class SummaryAuditTests(unittest.TestCase):
    def test_pooling_uses_summed_seconds_not_mean_precision(self):
        a = dict.fromkeys(MOD.TOTALS, 0.)
        b = dict(a)
        a.update(paddedModelExportSeconds=1, paddedIntersectionSeconds=1, coreHumanSeconds=1, coreIntersectionSeconds=1)
        b.update(paddedModelExportSeconds=99, paddedIntersectionSeconds=0, coreHumanSeconds=99, coreIntersectionSeconds=99)
        pooled = MOD.pooled([a, b])
        self.assertEqual(pooled['P_pad'], .01)
        self.assertEqual(pooled['R_core'], 1)
        self.assertAlmostEqual(pooled['F1_padP_coreR'], .02/1.01)

    def test_three_seed_averaging_does_not_sum_workload(self):
        self.assertEqual(MOD.mean_values([{'x': 10}, {'x': 20}, {'x': 30}], ('x',)), {'x': 20})
        with self.assertRaises(ValueError):
            MOD.mean_values([{'x': 10}], ('x',))

    def test_rally_counts_are_distinct_ignore_holes_and_exclude_touching(self):
        rec = {'rallies': [{'start': 1, 'end': 4}, {'start': 5, 'end': 8}, {'start': 10, 'end': 12}],
               'ignoredIntervals': [{'start': 5, 'end': 8}]}
        self.assertEqual(MOD.true_rallies(rec, [(1, 2), (3, 4), (5, 8), (8, 10)]), [0])
        self.assertEqual(MOD.true_rallies(rec, [(3, 11)]), [0, 2])

    def test_padding_inventory_rejects_missing_duplicate_or_replaced_case(self):
        rows = [{'paddingSecondsBeforeAndAfter': x} for x in (0, 1, 2, 3)]
        self.assertEqual(set(MOD.by_pad(rows)), {0, 1, 2, 3})
        for value in (rows[:3], rows+[rows[0]], rows[:3]+[{'paddingSecondsBeforeAndAfter': 4}],
                      [rows[0], {'paddingSecondsBeforeAndAfter': 1.5}, rows[2], rows[3]]):
            with self.assertRaises(ValueError):
                MOD.by_pad(value)

    def test_metric_comparison_rejects_nonfinite_or_inflated_count(self):
        MOD.close(3., 3, 'test')
        for value in (float('nan'), float('inf'), True, 3.01):
            with self.assertRaises(ValueError):
                MOD.close(value, 3, 'test')

    def test_seed_statistics_are_population_statistics(self):
        result = MOD.seed_statistics([{'x': 1}, {'x': 2}, {'x': 3}], ('x',))['x']
        self.assertEqual({k: result[k] for k in ('min', 'max', 'mean')}, {'min': 1, 'max': 3, 'mean': 2})
        self.assertAlmostEqual(result['seedPopulationStddev'], (2/3)**.5)

    def test_negative_slivers_deduplicate_across_policy_replicas(self):
        cell = {'configuration': {'family': 'individual', 'neuralId': 'model'}, 'seed': 1,
                'recordings': [{'id': 'a', 'candidates': [{'id': 'c0', 'kind': 'negative', 'start': 5-1e-12, 'end': 5},
                                                       {'id': 'c1', 'kind': 'negative', 'start': 5, 'end': 10}]}]}
        result = MOD.negative_grid_slivers([cell, cell])
        self.assertEqual(result['policyCandidateOccurrences'], 2)
        self.assertEqual(result['uniqueModelSeedRecordingCandidates'], 1)


if __name__ == '__main__':
    unittest.main()
