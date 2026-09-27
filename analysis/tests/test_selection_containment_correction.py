import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis import neural_selection_containment_correction as correction
from analysis.neural_recall_operating_point import strict_selection
from analysis.schema import Interval


def candidate(recall, f1=.7, marker='unchanged'):
    return {'epoch': 5, 'decoder': {'smoothing': 1., 'enter': .2, 'minimum': .25, 'boundary': False},
            'innerR_core': recall, 'innerF1_padP_coreR': f1, 'extraField': marker}


class ContainmentCorrectionTests(unittest.TestCase):
    def proof(self, end=3.):
        example = SimpleNamespace(id='synthetic', truth=(Interval(0., end),), ignored=(), duration=10.)
        with patch.object(correction.frozen.base, 'decode', return_value=[Interval(0., 1.)]):
            return correction.containment_proof([example], {5: {'synthetic': None}}, candidate(math.nextafter(1., 2.)))

    def test_only_invalid_recall_changes_and_raw_order_ties_stay(self):
        raw = [candidate(math.nextafter(1., 0.), .8, 'below'), candidate(math.nextafter(1., 2.), .7, 'first'),
               candidate(math.nextafter(1., 2.), .7, 'second'), candidate(.99, .9, 'floor')]
        before = copy.deepcopy(raw)
        result, changes = correction.correct_candidates(raw, lambda *_: self.proof())
        self.assertEqual(raw, before)
        self.assertEqual(result[0], raw[0]); self.assertEqual(result[3], raw[3])
        self.assertEqual([v['candidateIndex'] for v in changes], [1, 2])
        for index in (1, 2):
            self.assertEqual(result[index], {**raw[index], 'innerR_core': 1.})
        self.assertEqual(strict_selection(result, 1.)['selected']['extraField'], 'first')
        self.assertEqual(strict_selection(result, .99)['selected']['extraField'], 'floor')
        self.assertEqual(strict_selection(result, 1.)['eligibleCandidateCount'], 2)

    def test_positive_single_ulp_missed_span_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty omitted core'):
            self.proof(math.nextafter(3., 4.))

    def test_envelope_and_f1_fail_closed(self):
        for value in (-.01, math.nan, math.inf, 1. + 5*math.ulp(1.)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                correction.correct_candidates([candidate(value)], lambda *_: self.proof())
        with self.assertRaisesRegex(ValueError, 'F1 outside'):
            correction.correct_candidates([candidate(1., math.nextafter(1., 2.))], lambda *_: self.proof())

    def test_zero_corrections_requires_no_proof_and_changes_nothing(self):
        raw = [candidate(.9), candidate(1.)]
        with patch('builtins.input', side_effect=AssertionError):
            fixed, corrections = correction.correct_candidates(raw, lambda *_: self.fail('Unexpected proof'))
        self.assertEqual(fixed, raw); self.assertEqual(corrections, [])

    def test_publication_rejects_only_the_old_numerical_gate(self):
        source = Path(__file__).resolve().parents[2]/'scripts/generalization-selection-containment.py'
        spec = importlib.util.spec_from_file_location('containment_workflow_test', source)
        workflow = importlib.util.module_from_spec(spec); spec.loader.exec_module(workflow)
        with patch.object(workflow.io, 'read', return_value={'kind': 'independent-generalization-result-report-audit-v1', 'passed': True}):
            with self.assertRaisesRegex(ValueError, 'joint full-result and correction audit'):
                workflow.report(SimpleNamespace(audit=Path('dummy'), index=Path('dummy'), output=Path('dummy')))


if __name__ == '__main__': unittest.main()
