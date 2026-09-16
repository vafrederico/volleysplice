import importlib.util
from pathlib import Path
import plistlib
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('audit_release', Path(__file__).parents[1] / 'audit-release.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReleaseResourceTests(unittest.TestCase):
    def setUp(self):
        self.manifest_path = Path(__file__).parents[2] / 'App/PrivacyInfo.xcprivacy'
        self.manifest = plistlib.loads(self.manifest_path.read_bytes())
        self.files = {name: b'{}' for name in audit.MODELS}
        self.files.update({'PrivacyInfo.xcprivacy': self.manifest_path.read_bytes(),
                           'Info.plist': plistlib.dumps({'CFBundleExecutable': 'VolleySplice',
                                                        'ITSAppUsesNonExemptEncryption': False}),
                           'volleysplice_logo.png': b'logo', 'VolleySplice': b'executable'})

    def test_export_compliance_requires_boolean_false(self):
        for extra in [{}, {'ITSAppUsesNonExemptEncryption': True}, {'ITSAppUsesNonExemptEncryption': 'false'}]:
            files = self.files | {'Info.plist': plistlib.dumps({'CFBundleExecutable': 'VolleySplice'} | extra)}
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'NonExemptEncryption'):
                audit.audit(files, files.__getitem__, self.manifest)

    def test_archive_and_exported_ipa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'App.xcarchive'
            app = archive / 'Products/Applications/VolleySplice.app'
            app.mkdir(parents=True)
            ipa = root / 'App.ipa'
            with zipfile.ZipFile(ipa, 'w') as exported:
                for name, content in self.files.items():
                    (app / name).write_bytes(content)
                    exported.writestr('Payload/VolleySplice.app/' + name, content)
            audit.audit_release(archive, ipa, self.manifest_path, root / 'audit.json')
            # A fixture introduced during export must fail even if the archive is clean.
            with zipfile.ZipFile(ipa, 'a') as exported:
                exported.writestr('Payload/VolleySplice.app/base.bin', b'test')
            with self.assertRaisesRegex(ValueError, 'test resources'):
                audit.audit_release(archive, ipa, self.manifest_path, root / 'audit.json')

    def test_fixture_and_unreviewed_resource_rejection(self):
        for name in ['golden.json', 'base.bin', 'ios-editor-fixture.mp4', 'Tests/input.txt',
                     'PlugIns/Test.xctest/Info.plist', 'unknown-model.json']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                files = self.files | {name: b'test'}
                audit.audit(files, files.__getitem__, self.manifest)

    def test_missing_manifest_or_production_model(self):
        for name in ['PrivacyInfo.xcprivacy', next(iter(audit.MODELS))]:
            files = self.files.copy()
            del files[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.audit(files, files.__getitem__, self.manifest)

    def test_altered_and_nested_tracking_manifest(self):
        changed = self.manifest | {'NSPrivacyTracking': True}
        for name in ['PrivacyInfo.xcprivacy', 'Frameworks/SDK.framework/PrivacyInfo.xcprivacy']:
            files = self.files | {name: plistlib.dumps(changed)}
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.audit(files, files.__getitem__, self.manifest)


if __name__ == '__main__':
    unittest.main()
