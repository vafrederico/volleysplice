from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/audit-neural-context-preflight.py'
SPEC = importlib.util.spec_from_file_location('context_preflight_tested', SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class ContextPreflightTests(unittest.TestCase):
    def test_bit_parity_rejects_signed_zero_and_dtype_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory)/name for name in ('a.npz', 'b.npz'))
            np.savez(a, x=np.array([0.], dtype=np.float32))
            np.savez(b, x=np.array([-0.], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Parity values'):
                preflight.compare_npz(a, b)
            np.savez(b, x=np.array([0.], dtype=np.float64))
            with self.assertRaisesRegex(ValueError, 'Parity values'):
                preflight.compare_npz(a, b)

    def test_union_predictions_compare_only_the_declared_view(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory)/name for name in ('a.npz', 'b.npz'))
            np.savez(a, held_a=np.ones(3, dtype=np.float32), held_b=np.zeros(2, dtype=np.float32))
            np.savez(b, held_b=np.zeros(2, dtype=np.float32))
            self.assertTrue(preflight.compare_npz(a, b, ['held_b'])['bitExact'])
            with self.assertRaisesRegex(ValueError, 'inventory'):
                preflight.compare_npz(a, b, ['held_a'])

    def test_reference_audit_requires_kind_contract_path_and_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, audit = root/'report.json', root/'audit.json'
            report.write_text('{}', encoding='utf-8')
            correct = {'kind': preflight.AUDIT_KINDS[0], 'passed': True, 'contractSha256': 'contract',
                       'report': preflight.identity(report)}
            audit.write_text(json.dumps(correct), encoding='utf-8')
            self.assertEqual(preflight.verify_reference_audit(audit, report, preflight.AUDIT_KINDS[0], 'contract'),
                             preflight.identity(audit))
            for field, bad in (('kind', preflight.AUDIT_KINDS[1]), ('passed', False), ('contractSha256', 'another')):
                audit.write_text(json.dumps({**correct, field: bad}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Unpassed or unrelated'):
                    preflight.verify_reference_audit(audit, report, preflight.AUDIT_KINDS[0], 'contract')
            audit.write_text(json.dumps({**correct, 'report': {**correct['report'], 'path': str(root/'other.json')}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unpassed or unrelated'):
                preflight.verify_reference_audit(audit, report, preflight.AUDIT_KINDS[0], 'contract')


if __name__ == '__main__':
    unittest.main()
