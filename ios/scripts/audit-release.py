"""Check archived/exported resources and inventory privacy declarations.

This is a bundle audit, not Xcode's generated privacy report or an API scanner.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import plistlib
import zipfile

MODELS = {
    'model-1ca43e38eefc.json', 'model-9c92b8e9333f.json',
    'suppression-overlap-exclusion-retrained.json', 'serving-side-85bc3325fbd4.json',
    'side-switch-c2570481c30d.json',
}
NEURAL_MANIFEST = 'rally-models/manifest.json'
LICENSE_NAMES = {
    'TorchVision-0.26.0-BSD-3-Clause.txt', 'DINOv2-Apache-2.0.txt',
    'ONNX-Runtime-1.24.2-MIT.txt', 'ONNX-Runtime-1.24.2-ThirdPartyNotices.txt',
}


def expected_neural_resources(model_manifest=None):
    """The upload helper retains the repository-pinned manifest alongside staged assets."""
    source = Path(__file__).resolve().parents[1]
    canonical = source.parent / 'models/distilled-large/ios-manifest.json'
    path = model_manifest or (canonical if canonical.is_file() else source / 'Fixtures/rally-models/manifest.json')
    return json.loads(path.read_text(encoding='utf-8'))


def expected_license_resources():
    source = Path(__file__).resolve().parents[1]
    canonical = source.parent / 'models/distilled-large/licenses'
    folder = canonical if canonical.is_dir() else source / 'Fixtures/licenses'
    notices = source.parent / 'THIRD_PARTY_NOTICES.md'
    if not notices.is_file():
        notices = source / 'Fixtures/THIRD_PARTY_NOTICES.md'
    return {'licenses/' + name: (folder / name).read_bytes() for name in LICENSE_NAMES} | {
        'THIRD_PARTY_NOTICES.md': notices.read_bytes(),
    }


def neural_assets(manifest):
    if (manifest.get('schemaVersion') != 1 or manifest.get('precision') != 'fp32'
            or manifest.get('family') != 'distilled-mobilenet-v3-large-tcn'
            or manifest.get('defaultVariant') != 'high-recall'
            or set(manifest.get('variants', {})) != {'high-recall', 'high-f1'}):
        raise ValueError('Incompatible pinned neural manifest')
    result = {}
    for selection, directory, model_id in [
            ('high-recall', 'recall', 'distilled-large-recall-v1'),
            ('high-f1', 'f1', 'distilled-large-f1-v1')]:
        variant = manifest['variants'][selection]
        if (variant.get('id') != model_id or variant.get('directory') != directory
                or variant.get('embeddingPrecision') != 'fp32'
                or set(variant.get('files', {})) != {'encoder', 'temporal', 'pipeline'}):
            raise ValueError('Incompatible pinned neural variant')
        for role, filename in [('encoder', 'encoder.onnx'), ('temporal', 'temporal.onnx'), ('pipeline', 'pipeline.json')]:
            asset = variant['files'][role]
            digest = asset.get('sha256')
            if (asset.get('name') != filename or type(asset.get('sizeBytes')) is not int or asset['sizeBytes'] <= 0
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(character not in '0123456789abcdef' for character in digest)):
                raise ValueError('Invalid pinned neural asset')
            result[f'rally-models/{directory}/{filename}'] = asset
    return result


def audit(files, read, expected_manifest, expected_neural=None, expected_licenses=None):
    files = sorted(files)
    if len(files) != len(set(files)):
        raise ValueError('Duplicate Release bundle resources')
    expected_neural = expected_neural if expected_neural is not None else expected_neural_resources()
    expected_licenses = expected_licenses if expected_licenses is not None else expected_license_resources()
    required_licenses = {'licenses/' + name for name in LICENSE_NAMES} | {'THIRD_PARTY_NOTICES.md'}
    if set(expected_licenses) != required_licenses:
        raise ValueError('Incomplete pinned license resource inventory')
    assets = neural_assets(expected_neural)
    neural_files = set(assets) | {NEURAL_MANIFEST}
    forbidden = []
    for name in files:
        path = PurePosixPath(name)
        lower = name.lower()
        if (path.is_absolute() or '..' in path.parts or '\\' in name
                or any(word in lower for word in ('fixture', 'golden', '.xctest', 'xctestrunner'))
                or any(part.lower() in ('tests', 'test', 'plugIns'.lower()) for part in path.parts)
                or path.suffix.lower() in ('.bin', '.b64', '.mp4', '.swift', '.py', '.sh')
                or (path.suffix.lower() in ('.json', '.onnx', '.ort') and name not in MODELS | neural_files)
                or (path.parts[0] == 'rally-models' and name not in neural_files)
                or (path.parts[0] == 'licenses' and name not in required_licenses)):
            forbidden.append(name)
    if forbidden:
        raise ValueError('Unexpected/test resources in Release bundle: ' + ', '.join(forbidden))
    for required in sorted(MODELS | neural_files | required_licenses | {'Info.plist', 'PrivacyInfo.xcprivacy', 'volleysplice_logo.png'}):
        if required not in files:
            raise ValueError('Missing Release resource: ' + required)
    if json.loads(read(NEURAL_MANIFEST)) != expected_neural:
        raise ValueError('Bundled neural manifest differs from pinned source')
    verified_assets = {}
    for name, expected in assets.items():
        content = read(name)
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != expected['sizeBytes'] or digest != expected['sha256']:
            raise ValueError('Neural asset checksum or size mismatch: ' + name)
        verified_assets[name] = {'sha256': digest, 'size_bytes': len(content)}
    for name, expected in expected_licenses.items():
        if read(name) != expected:
            raise ValueError('Bundled license notice differs from pinned source: ' + name)
    info = plistlib.loads(read('Info.plist'))
    if info.get('ITSAppUsesNonExemptEncryption') is not False:
        raise ValueError('Release must declare ITSAppUsesNonExemptEncryption as Boolean false')
    executable = info.get('CFBundleExecutable')
    if not isinstance(executable, str) or executable not in files or '/' in executable:
        raise ValueError('Missing or invalid app executable')
    manifests = {}
    for name in files:
        if name.endswith('.xcprivacy'):
            manifest = plistlib.loads(read(name))
            if (manifest.get('NSPrivacyTracking', False) is not False
                    or manifest.get('NSPrivacyTrackingDomains', [])
                    or manifest.get('NSPrivacyCollectedDataTypes', [])):
                raise ValueError('Review unexpected tracking/data collection declaration: ' + name)
            manifests[name] = manifest
    if manifests['PrivacyInfo.xcprivacy'] != expected_manifest:
        raise ValueError('Bundled app privacy manifest differs from reviewed source')
    return {'files': files, 'privacy_manifests': manifests, 'test_resources_found': False,
            'neural_assets': verified_assets, 'license_notices_verified': sorted(required_licenses),
            'bundle_id': info.get('CFBundleIdentifier'), 'version': info.get('CFBundleShortVersionString'),
            'build': info.get('CFBundleVersion'), 'uses_non_exempt_encryption': False,
            'executable_sha256': hashlib.sha256(read(executable)).hexdigest()}


def audit_release(archive, ipa, source_manifest, output, model_manifest=None, license_resources=None):
    expected = plistlib.loads(source_manifest.read_bytes())
    expected_neural = expected_neural_resources(model_manifest)
    expected_licenses = license_resources if license_resources is not None else expected_license_resources()
    app = archive / 'Products/Applications/VolleySplice.app'
    archive_files = [p.relative_to(app).as_posix() for p in app.rglob('*') if p.is_file()]
    archive_report = audit(archive_files, lambda name: (app / name).read_bytes(), expected, expected_neural, expected_licenses)
    with zipfile.ZipFile(ipa) as exported:
        prefix = 'Payload/VolleySplice.app/'
        names = [n[len(prefix):] for n in exported.namelist() if n.startswith(prefix) and not n.endswith('/')]
        ipa_report = audit(names, lambda name: exported.read(prefix + name), expected, expected_neural, expected_licenses)
    report = {
        'scope': 'Release archive and exported IPA resource/privacy manifest inventory',
        'limitations': 'Not an API/symbol scan or Xcode Organizer privacy report; review SDK behavior separately.',
        'archive': archive_report, 'ipa': ipa_report,
        'ipa_sha256': hashlib.sha256(ipa.read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Release archive and IPA verified: privacy manifest present; no test resources found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--ipa', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--models-manifest', type=Path, help='Pinned iOS neural manifest; defaults to repository or prepared source copy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_release(args.archive, args.ipa, args.manifest, args.output, args.models_manifest)
