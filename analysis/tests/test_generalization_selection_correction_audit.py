"""Mutation checks for the independent containment-only calibration gate."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.schema import Interval

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/audit-neural-generalization-selection-correction.py'
SPEC = importlib.util.spec_from_file_location('correction_audit_under_test', SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def candidates():
    return [{'epoch': 5, 'decoder': {'index': i}, 'innerR_core': .99,
             'innerF1_padP_coreR': .8} for i in range(192)]


def true_proof():
    return {'recordings': [{'id': 'one', 'coreUnion': [[2., 3.]], 'retainedUnion': [[0., 5.]],
                           'missingCoreIntervals': [], 'coreExact': {'numerator': 1, 'denominator': 1},
                           'intersectionExact': {'numerator': 1, 'denominator': 1}}],
            'pooledCoreExact': {'numerator': 1, 'denominator': 1},
            'pooledIntersectionExact': {'numerator': 1, 'denominator': 1}, 'fullCoreContainment': True}


def corrected_case():
    raw = candidates()
    raw[0]['innerR_core'] = math.nextafter(1., math.inf)
    changed = deepcopy(raw)
    changed[0]['innerR_core'] = 1.
    proof = true_proof()
    changes = [{'candidateIndex': 0, 'rawRecall': raw[0]['innerR_core'], 'correctedRecall': 1., 'proof': proof}]
    return raw, changed, changes, proof


class CorrectionAuditTests(unittest.TestCase):
    def test_only_proven_overshoot_changes_and_valid_f1_is_preserved(self):
        raw, changed, changes, proof = corrected_case()
        before = deepcopy(raw)
        self.assertEqual(audit.check_tables(raw, changed, changes, lambda _: proof), changes)
        self.assertEqual(raw, before)
        self.assertTrue(audit.io.strict_selection(changed, 1.)['feasible'])
        raw[0]['innerR_core'] = 1. + 4 * math.ulp(1.)
        changes[0]['rawRecall'] = raw[0]['innerR_core']
        self.assertEqual(audit.check_tables(raw, changed, changes, lambda _: proof), changes)
        unchanged = candidates()
        unchanged[0]['innerR_core'] = math.nextafter(1., 0.)
        self.assertEqual(audit.check_tables(unchanged, deepcopy(unchanged), [], lambda _: self.fail()), [])
        self.assertFalse(audit.io.strict_selection(unchanged, 1.)['feasible'])

    def test_below_one_f1_order_and_boolean_mutations_fail(self):
        raw, changed, changes, proof = corrected_case()
        for mutation in ('below', 'f1', 'order', 'boolean'):
            wrong = deepcopy(changed)
            if mutation == 'below': wrong[1]['innerR_core'] = 1.
            if mutation == 'f1': wrong[0]['innerF1_padP_coreR'] = math.nextafter(.8, math.inf)
            if mutation == 'order': wrong[1], wrong[2] = wrong[2], wrong[1]
            if mutation == 'boolean': wrong[0]['innerR_core'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.check_tables(raw, wrong, changes, lambda _: proof)

    def test_invalid_f1_large_overshoot_and_missing_inventory_fail(self):
        raw, changed, changes, proof = corrected_case()
        invalid = deepcopy(raw); invalid[1]['innerF1_padP_coreR'] = math.nextafter(1., math.inf)
        with self.assertRaises(ValueError): audit.check_tables(invalid, changed, changes, lambda _: proof)
        invalid = deepcopy(raw); invalid[0]['innerR_core'] = 1. + 5 * math.ulp(1.)
        with self.assertRaises(ValueError): audit.check_tables(invalid, changed, changes, lambda _: proof)
        with self.assertRaises(ValueError): audit.check_tables(raw, changed, [], lambda _: proof)
        wrong = deepcopy(changes); wrong[0]['proof']['coreUnion'] = []
        with self.assertRaises(ValueError): audit.check_tables(raw, changed, wrong, lambda _: proof)

    def test_a_positive_sub_femtosecond_gap_cannot_be_promoted(self):
        example = SimpleNamespace(id='one', duration=10., truth=(Interval(2., 3.),), ignored=())
        raw, changed, changes, _ = corrected_case()
        # Padding leaves a positive gap smaller than an ordinary decimal tolerance.
        decoded = [Interval(math.nextafter(4., math.inf), 4.5)]
        with patch.object(audit.decoder, 'decode', return_value=decoded):
            proof = audit.coverage_proof([example], {5: {'one': None}}, raw[0])
        self.assertFalse(proof['fullCoreContainment'])
        missing = proof['recordings'][0]['missingCoreIntervals']
        self.assertTrue(missing)
        self.assertGreater(missing[0][1] - missing[0][0], 0.)
        self.assertLess(missing[0][1] - missing[0][0], 1e-14)
        with self.assertRaises(ValueError): audit.check_tables(raw, changed, changes, lambda _: proof)

    def test_exact_gap_and_ignored_interval_rules_are_preserved(self):
        self.assertEqual(audit.retained([Interval(2., 3.), Interval(10., 11.)], 20., []),
                         [[0., 5.], [8., 13.]])
        self.assertEqual(audit.retained([Interval(2., 3.), Interval(math.nextafter(10., 0.), 11.)], 20., []),
                         [[0., 13.]])
        self.assertEqual(audit.retained([Interval(2., 8.)], 20., [[4., 6.]]), [[0., 4.], [6., 10.]])

    def test_fabricated_proof_and_zero_record_proof_fail(self):
        raw, changed, changes, proof = corrected_case()
        corrupt = deepcopy(changes)
        corrupt[0]['proof']['recordings'][0]['coreExact']['numerator'] = 2
        with self.assertRaises(ValueError): audit.check_tables(raw, changed, corrupt, lambda _: proof)
        empty = deepcopy(proof); empty['recordings'] = []
        with self.assertRaises(ValueError): audit.check_tables(raw, changed, changes, lambda _: empty)


if __name__ == '__main__':
    unittest.main()
