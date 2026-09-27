"""Lossless transport parity; no metrics or row population may change."""
import base64
import gzip
import importlib.util
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('report_transport', ROOT/'scripts/render-neural-generalization-report-v2.py')
renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)


class GeneralizationReportTransportTest(unittest.TestCase):
    def test_numeric_categorical_missing_null_and_nested_round_trip(self):
        rows = [{'id': f'video-{i%4}', 'value': i*.23456789, 'floorPercent': 90+i%11,
                 'nested': {'range': [i%3, None], 'note': 'é </script> &'}, 'flag': i%2 == 0,
                 **({'optional': None} if i%4 == 0 else {'optional': i/9} if i%4 == 1 else {})}
                for i in range(180)]
        rows[10]['value'] = -0.; rows[11]['value'] = None; del rows[12]['value']
        original = json.loads(json.dumps(rows))
        packed = renderer.pack_series(rows)
        self.assertEqual(rows, original, 'packing must not mutate source')
        self.assertTrue(any(c['kind'] == 'float64' for c in packed['columns']))
        self.assertTrue(any(c['kind'] == 'dictionary' for c in packed['columns']))
        restored = renderer.unpack_series(packed)
        self.assertEqual(restored, rows)
        self.assertEqual(math.copysign(1, restored[10]['value']), -1)
        self.assertNotIn('value', restored[12])
        self.assertIsNone(restored[11]['value'])

    def test_exact_original_bytes_and_audit_payload_digest_survive_embedding(self):
        data = {'metadata': {'syntheticFixture': True}, 'inventory': [], 'tasks': [], 'scopes': {'s': ['video']},
                'series': [{'model': 'a', 'floorPercent': 90, 'precisionValue': .8, 'scopeId': 's',
                            'note': '</script><script>window.INJECTED=true</script>'}]}
        raw = (json.dumps(data, ensure_ascii=False, indent=3)+'\n\n').encode()
        html, receipt = renderer.render(raw, 'f'*64)
        encoded = re.search(r'id="canonicalReport"[^>]*>([^<]+)</script>', html).group(1)
        self.assertEqual(gzip.decompress(base64.b64decode(encoded)), raw)
        packed_text = re.search(r'id="packedReport"[^>]*>([^<]+)</script>', html).group(1)
        packed = json.loads(gzip.decompress(base64.b64decode(packed_text)))
        reconstructed = {**packed['header'], 'series': renderer.unpack_series(packed['series'])}
        self.assertEqual(reconstructed, data)
        self.assertEqual(receipt['sourceSha256'], renderer.digest(raw))
        self.assertEqual(receipt['reportContentSha256'], renderer.audit_digest(data))
        self.assertNotIn(data['series'][0]['note'], html)
        self.assertIn('class ColumnTable', html)
        self.assertIn('DINO embedding precision', html)

    def test_unrepresentable_integer_and_unaudited_real_input_fail(self):
        with self.assertRaisesRegex(ValueError, 'represented exactly'):
            renderer.pack_series([{'value': 2**53}])
        with self.assertRaisesRegex(ValueError, 'auditPassed'):
            renderer.render(json.dumps({'metadata': {}, 'series': []}).encode(), 'f'*64)


if __name__ == '__main__':
    unittest.main()
