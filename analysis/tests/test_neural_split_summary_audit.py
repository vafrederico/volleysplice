"""Post-freeze qualification of the independent summary/report auditor."""
import copy
import importlib.util
from pathlib import Path
import unittest

from analysis.tests.test_neural_split_summary import fixture, summary


SPEC = importlib.util.spec_from_file_location('independent_split_summary', Path(__file__).resolve().parents[2]
                                            / 'scripts/audit-neural-split-summary.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class IndependentSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.cells = fixture()
        cls.aggregated = summary.aggregate(cls.contract, cls.cells)

    def test_entire_synthetic_matrix_and_all_tables(self):
        receipt = audit.audit_aggregate(self.contract, self.cells, self.aggregated)
        self.assertEqual(receipt['comparisonArms'], 59)
        self.assertEqual(receipt['scopeSummaries'], 300)
        self.assertTrue(audit.audit_tables(self.aggregated, summary.markdown(self.aggregated))['passed'])

    def test_independent_statistics_handle_unavailable_and_population_variance(self):
        samples = [{'a': None, 'b': 1.}, {'a': None, 'b': 2.}, {'a': None, 'b': 6.}]
        expected = summary.tree_stats(samples)
        self.assertEqual(audit.verify_statistics(samples, summary.means(expected), expected, 'toy'), 2)
        expected['b']['seedPopulationStddev'] *= 1.1
        with self.assertRaisesRegex(ValueError, 'Stddev'):
            audit.verify_statistics(samples, summary.means(expected), expected, 'toy')

    def test_missing_comparison_arm_rejected(self):
        bad = copy.deepcopy(self.aggregated); bad['reviewed'].pop()
        with self.assertRaisesRegex(ValueError, 'arm scope'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_pooled_rally_precision_tamper_rejected(self):
        bad = copy.deepcopy(self.aggregated); bad['automatic'][0]['identity']['eventPrecision'] += .001
        with self.assertRaisesRegex(ValueError, 'eventPrecision'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_fourth_padding_slice_tamper_rejected(self):
        bad = copy.deepcopy(self.aggregated)
        bad['baseline']['recordings']['a']['padding'][3]['metrics']['incorrectExportSeconds'] += 1
        with self.assertRaisesRegex(ValueError, 'incorrectExportSeconds'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_workload_range_tamper_rejected(self):
        bad = copy.deepcopy(self.aggregated)
        bad['reviewed'][0]['sourceGroups']['g1']['workloadSeedStatistics']['reviewSeconds']['max'] += 1
        with self.assertRaisesRegex(ValueError, 'reviewSeconds'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_markdown_value_tamper_rejected(self):
        document = summary.markdown(self.aggregated)
        first = next(line for line in document.splitlines() if line.startswith('| 0 |'))
        bad = document.replace(first, first.replace('100.00', '99.99', 1))
        if bad == document:
            bad = document.replace(first, first.replace('0.00', '0.01', 1))
        self.assertNotEqual(bad, document)
        with self.assertRaisesRegex(ValueError, 'table row'):
            audit.audit_tables(self.aggregated, bad)

    def test_cleanup_statistic_tamper_rejected(self):
        bad = copy.deepcopy(self.aggregated)
        bad['cleanupYield']['pooled']['metrics']['flaggedParents'] += 1
        with self.assertRaisesRegex(ValueError, 'flaggedParents'):
            audit.audit_aggregate(self.contract, self.cells, bad)


if __name__ == '__main__':
    unittest.main()
