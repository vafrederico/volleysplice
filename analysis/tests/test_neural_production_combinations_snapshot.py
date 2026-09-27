"""Archive safety and evidence binding, independent of experimental outcomes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('combination_snapshot',
    Path(__file__).resolve().parents[2]/'scripts/snapshot-neural-production-combinations.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SnapshotTests(unittest.TestCase):
    def test_paths_cannot_escape_or_include_sensitive_trees(self):
        for name in ('../x.py', '/x.py', 'a/../b.py', 'a\\b.py', 'C:/x.py',
                     'x/.env.local', 'x/.git/config', 'x/fits/model.npz'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                M.safe_name(name)
        self.assertEqual(M.safe_name('repository/analysis/model.py'), 'repository/analysis/model.py')

    def test_hash_binding_and_recheck_detect_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); p = root/'a.py'; p.write_bytes(b'original')
            inv = M.Inventory(); ref = M.identity(p)
            inv.add(p, root, 'repository/a.py', ref)
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                inv.recheck()
            with self.assertRaises(ValueError):
                M.Inventory().add(p, root, 'a.py', ref)

    def test_symlink_and_external_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); folder = root/'base'; folder.mkdir()
            p = root/'external.py'; p.write_bytes(b'x')
            with self.assertRaises(ValueError):
                M.Inventory().add(p, folder, 'x.py')
            link = folder/'link.py'
            try:
                link.symlink_to(p)
            except OSError:
                return  # Windows may prohibit symlink creation; WSL runs this case.
            with self.assertRaises(ValueError):
                M.Inventory().add(link, folder, 'x.py')

    def test_arrays_require_explicit_small_artifact_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); p = root/'x.npz'; p.write_bytes(b'\x00\xffbinary')
            with self.assertRaises(ValueError):
                M.Inventory().add(p, root, 'x.npz')
            M.Inventory().add(p, root, 'x.npz', allow_npz=True)

    def test_exact_binary_restore_and_entry_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'a.zip'
            contents = {'nested/x.npz': b'\x00\xff\x80\r\n\x00', 'source.py': b'x=1\n'}
            with zipfile.ZipFile(archive, 'w') as z:
                for name, data in contents.items():
                    z.writestr(name, data)
            M.zip_roundtrip(archive, contents, root)
            with self.assertRaises(ValueError):
                M.zip_roundtrip(archive, {**contents, 'missing.txt': b''}, root)
            with self.assertRaises(ValueError):
                M.zip_roundtrip(archive, {**contents, 'source.py': b'changed'}, root)

    def test_collision_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); a = root/'a.py'; b = root/'b.py'
            a.write_bytes(b'a'); b.write_bytes(b'b')
            inv = M.Inventory(); inv.add(a, root, 'source.py')
            with self.assertRaises(ValueError):
                inv.add(b, root, 'source.py')
            inv = M.Inventory(); inv.total = M.MAX_TOTAL
            with self.assertRaises(ValueError):
                inv.add(a, root, 'a.py')

    def test_closed_directory_rejects_new_unbound_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); p = root/'a.py'; p.write_bytes(b'a')
            inv = M.Inventory(); inv.add(p, root, 'a.py'); inv.close_directory(root, {'a.py'})
            (root/'late.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):
                inv.recheck()

    def test_completion_gate_requires_audits_and_full_matrix(self):
        c = {'code': dict.fromkeys(M.REGISTERED), 'newFits': 0,
             'protectedTestOpened': False, 'productionPromotionAllowed': False}
        signature = M.canonical(c)
        report_id = {'path': '/study/report.json', 'sha256': 'a'*64, 'sizeBytes': 10}
        audit_id = {'path': '/study/recipe-audit-v1.json', 'sha256': 'b'*64, 'sizeBytes': 20}
        report = {'status': 'completed-fixed-combinations', 'contractSha256': signature,
                  'protectedTestOpened': False, 'productionChanged': False, 'newFits': 0,
                  'counts': {'configurations': 107, 'automaticCells': 303, 'reviewCells': 90,
                             'accountingScopesPerCell': 13, 'paddingCasesPerScope': 4},
                  'automaticResults': [None]*303, 'reviewResults': [None]*90}
        summary = {'kind': 'audited-production-combinations-summary-v1', 'passed': True,
                   'contractSha256': signature, 'protectedTestOpened': False,
                   'productionChanged': False, 'productionPromotionAllowed': False,
                   'artifacts': {'report': report_id, 'recipeAudit': audit_id},
                   'automatic': [None]*107, 'reviews': [None]*30}
        audit = {'kind': 'independent-production-combination-recipe-audit-v1', 'passed': True,
                 'contractSha256': signature, 'protectedTestOpened': False, 'report': report_id,
                 'automaticCellsAudited': 303, 'automaticRecordingConstructionsAudited': 2424,
                 'reviewCellsAudited': 90, 'reviewRecordingPaddingRowsAudited': 2880,
                 'actualAppOverrideRecordingPaddingRowsAudited': 128}
        with patch.object(M, 'CONTRACT', signature):
            args = ({'contract': c, 'sha256': signature}, report, summary, audit, report_id, audit_id)
            M.completion_gate(*args)
            audit['passed'] = False
            with self.assertRaises(ValueError):
                M.completion_gate(*args)
            audit['passed'] = True; report['automaticResults'].pop()
            with self.assertRaises(ValueError):
                M.completion_gate(*args)


if __name__ == '__main__':
    unittest.main()
