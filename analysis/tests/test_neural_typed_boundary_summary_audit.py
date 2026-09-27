"""Post-freeze tests for the independent typed-boundary summary audit."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from analysis.tests.test_neural_typed_boundary_summary import fixture, summary

SPEC = importlib.util.spec_from_file_location('typed_aggregate_audit', Path(__file__).resolve().parents[2]
                                            / 'scripts/audit-neural-typed-boundary-summary.py')
audit = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(audit)


class TypedAggregateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.cells, _, _ = fixture()
        cls.result = summary.aggregate(cls.contract, cls.cells)

    def test_all69arms_scopes_and_tables(self):
        receipt = audit.audit_aggregate(self.contract, self.cells, self.result)
        self.assertEqual(receipt['aggregateArms'], 69)
        self.assertEqual(receipt['scopeSummaries'], 345)
        self.assertTrue(audit.audit_tables(self.result, summary.markdown(self.result))['passed'])

    def test_missing_arm_rejected(self):
        bad = deepcopy(self.result); bad['automatic'].pop()
        with self.assertRaisesRegex(ValueError, 'arm scope'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_physical_core_alias_tamper_rejected(self):
        bad = deepcopy(self.result); bad['automatic'][0]['coverage']['resultRawCoreRecall'] = .1
        with self.assertRaisesRegex(ValueError, 'coverage alias'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_typed_pair_metric_tamper_rejected(self):
        bad = deepcopy(self.result)
        bad['reviewed'][0]['typed']['samePairBoundaries']['1']['predicted'] += 1
        with self.assertRaisesRegex(ValueError, 'samePairBoundaries'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_full_parent_acceptance_null_is_not_zero(self):
        bad = deepcopy(self.result)
        arm = next(r for r in bad['reviewed'] if r['humanMode'] == 'full_parent')
        arm['humanActions']['acceptedCount'] = 0
        with self.assertRaisesRegex(ValueError, 'acceptedCount'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_recording_padding_range_tamper_rejected(self):
        bad = deepcopy(self.result)
        bad['baseline']['recordings']['a']['padding'][3]['seedStatistics']['P_pad']['max'] -= .1
        with self.assertRaisesRegex(ValueError, 'P_pad'):
            audit.audit_aggregate(self.contract, self.cells, bad)

    def test_report_numeric_tamper_rejected(self):
        document = summary.markdown(self.result)
        row = next(line for line in document.splitlines() if line.startswith('| 0 |'))
        tampered = document.replace(row, row.replace('100.00', '99.99', 1))
        self.assertNotEqual(document, tampered)
        with self.assertRaisesRegex(ValueError, 'table row'):
            audit.audit_tables(self.result, tampered)

    def test_report_candidate_human_distinction_required(self):
        document = summary.markdown(self.result).replace('not gold-corrected output', 'output')
        with self.assertRaisesRegex(ValueError, 'interpretation limit'):
            audit.audit_tables(self.result, document)


if __name__ == '__main__':
    unittest.main()
