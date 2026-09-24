import copy
import importlib.util
from pathlib import Path
import unittest

from analysis.neural_production_combinations import duration_rows
from analysis.neural_rally_identity_metrics import evaluate_rally_identities
from analysis.tests.test_neural_rally_review_summary import fixture, summary as producer


PATH = Path(__file__).resolve().parents[2] / 'scripts/audit-neural-rally-review-summary.py'
SPEC = importlib.util.spec_from_file_location('rally_review_summary_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def records():
    return [dict(id=f'r{i}', sourceGroup=f'g{i//2}', durationSeconds=100.,
                 rallies=[[5., 10.], [25., 30.]], predictions=[[5., 10.], [26., 29.]],
                 ignoredIntervals=[]) for i in range(8)]


class SummaryAuditTests(unittest.TestCase):
    def test_all_four_duration_paddings_validate(self):
        rec = records()
        audit.audit_time(duration_rows(rec), [x['id'] for x in rec])

    def test_pooled_rates_are_not_mean_recording_rates(self):
        rec = records()
        rec[0]['predictions'] = [[1., 90.]]
        rows = duration_rows(rec)
        original = rows[2]['P_pad']
        rows[2]['P_pad'] = sum(x['P_pad'] for x in rows[2]['perRecording']) / 8
        self.assertNotAlmostEqual(original, rows[2]['P_pad'])
        with self.assertRaisesRegex(ValueError, 'P_pad'):
            audit.audit_time(rows, [x['id'] for x in rec])

    def test_missing_padding_rejected(self):
        rec = records()
        with self.assertRaisesRegex(ValueError, 'Four fixed'):
            audit.audit_time(duration_rows(rec)[1:], [x['id'] for x in rec])

    def test_false_export_partition_mutation_rejected(self):
        rec = records()
        rows = duration_rows(rec)
        rows[1]['perRecording'][0]['incorrectExportSeconds'] += 1
        with self.assertRaisesRegex(ValueError, 'incorrectExportSeconds'):
            audit.audit_time(rows, [x['id'] for x in rec])

    def test_identity_and_localization_pooling(self):
        rec = records()
        result = evaluate_rally_identities(rec)
        audit.audit_identities(result, {x['id']: x for x in rec})
        result['sourceGroups']['g0']['observedStartLocalization']['1']['matched'] += 1
        with self.assertRaisesRegex(ValueError, 'observedStartLocalization'):
            audit.audit_identities(result, {x['id']: x for x in rec})

    def test_identity_counts_cannot_hide_missing_matches(self):
        rec = records()
        result = evaluate_rally_identities(rec)
        result['recordings'][0]['matches'].pop()
        with self.assertRaisesRegex(ValueError, 'Stored matching'):
            audit.audit_identities(result, {x['id']: x for x in rec})

    def test_workload_uses_dataset_seconds(self):
        rows = [{key: 1. for key in audit.WORKLOAD} for _ in range(8)]
        for row in rows:
            row['reviewSeconds'] = 12.
            row['budgetSeconds'] = 20.
        result = audit.workload_totals(rows, 800.)
        self.assertEqual(result['reviewSeconds'], 96.)
        self.assertEqual(result['reviewFractionOfVideo'], .12)
        self.assertEqual(result['budgetUtilization'], .6)

    def test_null_boundary_metric_is_not_zero(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            audit.compare({'metric': 0.}, {'metric': None}, 'case')

    def test_all_summary_statistics_reconstructed(self):
        contract, cells, refs = fixture()
        result = producer.aggregate(contract, cells, refs)
        counters = audit.audit_summary(contract, result, cells, refs)
        self.assertEqual(counters['armsAudited'], 4)

    def test_summary_mean_mutation_is_detected(self):
        contract, cells, refs = fixture()
        result = producer.aggregate(contract, cells, refs)
        result['arms'][0]['primary']['F1_padP_coreR'] += .01
        with self.assertRaisesRegex(ValueError, 'F1_padP_coreR'):
            audit.audit_summary(contract, result, cells, refs)

    def test_seed_deviation_and_available_counts_checked(self):
        contract, cells, refs = fixture()
        result = producer.aggregate(contract, cells, refs)
        result['arms'][0]['identitySeedStatistics']['eventF1']['seedPopulationStddev'] += .001
        with self.assertRaisesRegex(ValueError, 'seedPopulationStddev'):
            audit.audit_summary(contract, result, cells, refs)

    def test_guardrail_screen_cannot_override_event_regression(self):
        contract, cells, refs = fixture()
        result = producer.aggregate(contract, cells, refs)
        screen = result['arms'][0]['guardrailScreen']
        screen['passed'] = not screen['passed']
        with self.assertRaisesRegex(ValueError, 'guardrailScreen'):
            audit.audit_summary(contract, result, cells, refs)

    def test_rankings_cannot_drop_or_repeat_arms(self):
        contract, cells, refs = fixture()
        result = producer.aggregate(contract, cells, refs)
        result['rankingsByDeclaredBudgetAndMode']['production--budget-05'].clear()
        with self.assertRaisesRegex(ValueError, 'rankings differ'):
            audit.audit_summary(contract, result, cells, refs)

    def test_null_statistics_preserve_unavailable_status(self):
        result = audit.statistics([None, 2., 4.])
        self.assertEqual(result['mean'], 3.)
        self.assertEqual(result['availableSeeds'], 2)
        self.assertEqual(result['seedPopulationStddev'], 1.)
        self.assertIsNone(audit.statistics([None, None, None])['mean'])


if __name__ == '__main__':
    unittest.main()
