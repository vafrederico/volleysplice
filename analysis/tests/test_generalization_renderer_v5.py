import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from analysis.tests.test_generalization_renderer_v4 import membership_fixture, embedded

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('deployment_report_v5_tests', REPO/'scripts/render-neural-generalization-report-v5.py')
renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)


class RendererV5Tests(unittest.TestCase):
    def test_note_is_visible_and_all_packed_data_and_download_bytes_match_v4(self):
        raw = json.dumps(membership_fixture(), indent=2).encode()
        previous, prior_receipt = renderer.load_v4().render(raw, 'fixture')
        html, receipt = renderer.render(raw, 'fixture')
        for key in ('packedReport', 'canonicalReport'):
            self.assertEqual(embedded(html, key), embedded(previous, key))
        self.assertEqual(gzip.decompress(base64.b64decode(embedded(html, 'canonicalReport'))), raw)
        self.assertEqual(receipt['reportContentSha256'], prior_receipt['reportContentSha256'])
        self.assertEqual(receipt['sourceSha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt['kind'], 'deployment-disclosure-report-view-v5')
        self.assertEqual(html.count('id="runtimeQualification"'), 1)
        self.assertIn('<div id="runtimeQualification" class="notice" role="note">'+renderer.NOTICE+'</div>', html)
        self.assertIn('presentationVersion:5,', html)

    def test_unqualified_real_data_and_changed_parent_source_are_rejected(self):
        value = membership_fixture(); value['metadata'].pop('syntheticFixture')
        with self.assertRaisesRegex(ValueError, 'auditPassed'):
            renderer.render(json.dumps(value).encode(), 'fixture')
        self.assertEqual(renderer.sha((REPO/'scripts/render-neural-generalization-report-v4.py').read_bytes()), renderer.V4_SHA)


if __name__ == '__main__': unittest.main()
