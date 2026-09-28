import importlib.util
import copy
import hashlib
import json
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
        self.neural = copy.deepcopy(audit.expected_neural_resources())
        for variant in self.neural['variants'].values():
            for role, asset in variant['files'].items():
                payload = ('synthetic-' + variant['directory'] + '-' + role).encode()
                asset['sha256'] = hashlib.sha256(payload).hexdigest()
                asset['sizeBytes'] = len(payload)
                self.files['rally-models/' + variant['directory'] + '/' + asset['name']] = payload
        self.files[audit.NEURAL_MANIFEST] = json.dumps(self.neural).encode()
        self.licenses = {'licenses/' + name: ('Synthetic license: ' + name).encode() for name in audit.LICENSE_NAMES}
        self.licenses['THIRD_PARTY_NOTICES.md'] = b'Synthetic notices'
        self.files.update(self.licenses)

    def checked(self, files):
        return audit.audit(files, files.__getitem__, self.manifest, self.neural, self.licenses)

    def test_export_compliance_requires_boolean_false(self):
        for extra in [{}, {'ITSAppUsesNonExemptEncryption': True}, {'ITSAppUsesNonExemptEncryption': 'false'}]:
            files = self.files | {'Info.plist': plistlib.dumps({'CFBundleExecutable': 'VolleySplice'} | extra)}
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'NonExemptEncryption'):
                self.checked(files)

    def test_archive_and_exported_ipa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'App.xcarchive'
            app = archive / 'Products/Applications/VolleySplice.app'
            app.mkdir(parents=True)
            ipa = root / 'App.ipa'
            with zipfile.ZipFile(ipa, 'w') as exported:
                for name, content in self.files.items():
                    (app / name).parent.mkdir(parents=True, exist_ok=True)
                    (app / name).write_bytes(content)
                    exported.writestr('Payload/VolleySplice.app/' + name, content)
            models = root / 'models.json'
            models.write_text(json.dumps(self.neural), encoding='utf-8')
            audit.audit_release(archive, ipa, self.manifest_path, root / 'audit.json', models, self.licenses)
            report = json.loads((root / 'audit.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['ipa']['neural_assets']), 6)
            self.assertEqual(report['archive']['neural_assets'], report['ipa']['neural_assets'])
            # A fixture introduced during export must fail even if the archive is clean.
            with zipfile.ZipFile(ipa, 'a') as exported:
                exported.writestr('Payload/VolleySplice.app/base.bin', b'test')
            with self.assertRaisesRegex(ValueError, 'test resources'):
                audit.audit_release(archive, ipa, self.manifest_path, root / 'audit.json', models, self.licenses)

    def test_fixture_and_unreviewed_resource_rejection(self):
        for name in ['golden.json', 'base.bin', 'ios-editor-fixture.mp4', 'Tests/input.txt',
                     'PlugIns/Test.xctest/Info.plist', 'unknown-model.json', 'unexpected.onnx',
                     'rally-models/recall/extra.onnx', 'rally-models/private.txt', 'licenses/unreviewed.txt']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                files = self.files | {name: b'test'}
                self.checked(files)

    def test_missing_manifest_or_production_model(self):
        for name in ['PrivacyInfo.xcprivacy', next(iter(audit.MODELS)), audit.NEURAL_MANIFEST,
                     'rally-models/recall/encoder.onnx', 'licenses/DINOv2-Apache-2.0.txt']:
            files = self.files.copy()
            del files[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.checked(files)

    def test_altered_and_nested_tracking_manifest(self):
        changed = self.manifest | {'NSPrivacyTracking': True}
        for name in ['PrivacyInfo.xcprivacy', 'Frameworks/SDK.framework/PrivacyInfo.xcprivacy']:
            files = self.files | {name: plistlib.dumps(changed)}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.checked(files)

    def test_graph_config_size_hash_and_manifest_are_pinned(self):
        for name in ['rally-models/recall/encoder.onnx', 'rally-models/f1/temporal.onnx',
                     'rally-models/f1/pipeline.json']:
            for payload in [self.files[name] + b'x', b'X' * len(self.files[name])]:
                with self.subTest(name=name, size=len(payload)), self.assertRaisesRegex(ValueError, 'checksum or size'):
                    self.checked(self.files | {name: payload})
        modified = copy.deepcopy(self.neural)
        modified['defaultVariant'] = 'high-f1'
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            self.checked(self.files | {audit.NEURAL_MANIFEST: json.dumps(modified).encode()})

    def test_notice_content_and_manifest_cannot_expand_the_allowlist(self):
        for name in self.licenses:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'license notice differs'):
                self.checked(self.files | {name: self.files[name] + b'changed'})
        invalid = copy.deepcopy(self.neural)
        invalid['variants']['high-f1']['files']['encoder']['name'] = '../unreviewed.onnx'
        with self.assertRaisesRegex(ValueError, 'Invalid pinned neural asset'):
            audit.neural_assets(invalid)
        invalid = copy.deepcopy(self.neural)
        invalid['variants']['unexpected'] = invalid['variants']['high-f1']
        with self.assertRaisesRegex(ValueError, 'Incompatible pinned neural manifest'):
            audit.neural_assets(invalid)

    def test_pinned_graphs_and_notices_match_the_source_tree(self):
        repository = Path(__file__).resolve().parents[3]
        for name, asset in audit.neural_assets(audit.expected_neural_resources()).items():
            content = (repository / 'prod/public/runtime' / name).read_bytes()
            self.assertEqual(len(content), asset['sizeBytes'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), asset['sha256'])
        self.assertEqual(set(audit.expected_license_resources()), set(self.licenses))

    def test_clean_sdk_privacy_resource_is_reviewed_without_allowing_models(self):
        files = self.files | {'Frameworks/onnxruntime.framework/PrivacyInfo.xcprivacy': plistlib.dumps(self.manifest)}
        self.assertIn('Frameworks/onnxruntime.framework/PrivacyInfo.xcprivacy', self.checked(files)['privacy_manifests'])


if __name__ == '__main__':
    unittest.main()
