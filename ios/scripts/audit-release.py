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


def audit(files, read, expected_manifest):
    files = sorted(files)
    forbidden = []
    for name in files:
        path = PurePosixPath(name)
        lower = name.lower()
        if (any(word in lower for word in ('fixture', 'golden', '.xctest', 'xctestrunner'))
                or any(part.lower() in ('tests', 'test', 'plugIns'.lower()) for part in path.parts)
                or path.suffix.lower() in ('.bin', '.b64', '.mp4', '.swift', '.py', '.sh')
                or (path.suffix == '.json' and name not in MODELS)):
            forbidden.append(name)
    if forbidden:
        raise ValueError('Unexpected/test resources in Release bundle: ' + ', '.join(forbidden))
    for required in sorted(MODELS | {'Info.plist', 'PrivacyInfo.xcprivacy', 'volleysplice_logo.png'}):
        if required not in files:
            raise ValueError('Missing Release resource: ' + required)
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
            'bundle_id': info.get('CFBundleIdentifier'), 'version': info.get('CFBundleShortVersionString'),
            'build': info.get('CFBundleVersion'), 'uses_non_exempt_encryption': False,
            'executable_sha256': hashlib.sha256(read(executable)).hexdigest()}


def audit_release(archive, ipa, source_manifest, output):
    expected = plistlib.loads(source_manifest.read_bytes())
    app = archive / 'Products/Applications/VolleySplice.app'
    archive_files = [p.relative_to(app).as_posix() for p in app.rglob('*') if p.is_file()]
    archive_report = audit(archive_files, lambda name: (app / name).read_bytes(), expected)
    with zipfile.ZipFile(ipa) as exported:
        prefix = 'Payload/VolleySplice.app/'
        names = [n[len(prefix):] for n in exported.namelist() if n.startswith(prefix) and not n.endswith('/')]
        ipa_report = audit(names, lambda name: exported.read(prefix + name), expected)
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
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_release(args.archive, args.ipa, args.manifest, args.output)
