import copy
import json
import tempfile
import unittest
from pathlib import Path

from analysis import neural_original_manifest_view as view
from analysis import neural_recall_sweep as io


def document():
    rows = [{'id': f'r{i}', 'sourceGroup': f'g{i % 7}',
             'payload': {'boundary': [i, i + 0.125], 'keep': i % 2 == 0}} for i in range(18)]
    return {'kind': 'neural-expanded-quality-tier-manifest-v1',
            'exactRows': rows[:8], 'draftRows': rows[8:11], 'coverageRows': rows[11:],
            'unrelated': {'nested': [1, 1.0, None, False]}}


class ManifestView(unittest.TestCase):
    def test_full_rows_and_existing_fields_are_lossless(self):
        original = document(); before = copy.deepcopy(original)
        result = view.projection(original)
        self.assertEqual(original, before)
        self.assertEqual({k: v for k, v in result.items() if k != 'records'}, original)
        self.assertEqual(io.canonical({k: v for k, v in result.items() if k != 'records'}), io.canonical(original))
        self.assertEqual(result['records'], original['exactRows'] + original['draftRows'] + original['coverageRows'])
        self.assertEqual(len({r['sourceGroup'] for r in result['records']}), 7)

    def test_partial_duplicate_or_already_projected_scope_rejected(self):
        values = [document(), document(), document(), document()]
        values[0]['draftRows'].pop()
        values[1]['coverageRows'][0]['id'] = values[1]['exactRows'][0]['id']
        values[2]['records'] = []
        values[3]['kind'] = 'other'
        for value in values:
            with self.assertRaises(ValueError): view.projection(value)

    def context(self, folder, stage='neural-evaluation'):
        path = Path(folder)/'legacy.json'; path.write_text(json.dumps(document()))
        reference = io.identity(path)
        plan = {'legacyManifest': reference, 'recordsCanonicalSha256': io.canonical(view.projection(document())['records'])}
        return path, view._installed(plan, {'path': 'plan', 'sha256': 'p'},
                                    {'path': 'qualification', 'sha256': 'q'}, stage=stage, argv=['frozen', 'evaluate'])

    def test_exact_path_only_and_reader_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manager = self.context(folder); other = Path(folder)/'other.json'
            other.write_text(json.dumps(document())); reader = io.read; identity = io.identity(path)
            with manager as proof:
                self.assertIn('records', io.read(path))
                self.assertNotIn('records', io.read(other))
            self.assertIs(io.read, reader)
            self.assertEqual(io.identity(path), identity)
            self.assertTrue(proof['passed']); self.assertEqual(proof['readCount'], 1)

    def test_exception_restores_reader_and_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manager = self.context(folder); reader = io.read
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with manager as proof:
                    io.read(path); raise RuntimeError('fixture')
            self.assertIs(io.read, reader); self.assertFalse(proof['passed'])

    def test_changed_legacy_bytes_fail_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manager = self.context(folder); reader = io.read
            with self.assertRaises(ValueError):
                with manager:
                    changed = document(); changed['unrelated'] = 'changed'
                    path.write_text(json.dumps(changed)); io.read(path)
            self.assertIs(io.read, reader)

    def test_absent_view_read_cannot_certify_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            _, manager = self.context(folder)
            with self.assertRaises(ValueError):
                with manager: pass

    def test_changed_evidence_binding_is_rejected(self):
        class Evidence:
            def document(self, value): return value
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'legacy.json'; path.write_text(json.dumps(document()))
            plan = {'legacyManifest': io.identity(path), 'recordsCanonicalSha256': io.canonical(view.projection(document())['records'])}
            reader = io.read; previous = Evidence.document
            try:
                with self.assertRaises(ValueError):
                    with view._installed(plan, {}, {}, stage='final-audit', argv=['frozen'],
                                         extra_bindings={'Evidence.document': (Evidence, 'document')}):
                        io.read(path); Evidence.document = lambda self, value: None
                self.assertIs(io.read, reader)
            finally: Evidence.document = previous


if __name__ == '__main__': unittest.main()
