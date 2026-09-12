"""Explicit source manifest. No credentials, video corpus, or build output is uploaded."""
import argparse
import base64
import gzip
import io
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.output, 'w:gz') as archive:
    for folder in ('Sources', 'Tests', 'App', 'scripts', 'VolleySplice.xcodeproj'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and (path.suffix in ('.swift', '.py', '.sh', '.ps1', '.plist', '.pbxproj', '.xcscheme', '.h', '.mm') or ('Assets.xcassets' in path.parts and path.suffix in ('.json', '.png'))):
                if path.suffix == '.sh':
                    # Windows checkouts may use CRLF; uploaded shell scripts must execute on macOS.
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    entry = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
                else:
                    archive.add(path, arcname=path.relative_to(root))
    archive.add(root / 'Package.swift', arcname='Package.swift')
    archive.add(root.parent / 'android/app/src/main/res/drawable-xxxhdpi/volleysplice_logo.png', arcname='Fixtures/volleysplice_logo.png')
    for path in (root.parent / 'android/app/src/main/assets').glob('*.json'):
        archive.add(path, arcname='Fixtures/' + path.name)
    archive.add(root.parent / 'tests/fixtures/on-device-y9-golden.json', arcname='Fixtures/golden.json')
    archive.add(root.parent / 'tests/fixtures/suppression-policy-golden.json', arcname='Fixtures/suppression-policy-golden.json')
    # Exact tiny media fixtures from Android's rotation/export instrumentation.
    fixture_root = root.parent / 'android/app/src/androidTest/assets'
    for name in ('overlay-fixture', 'overlay-fixture-90', 'overlay-fixture-180', 'overlay-fixture-270'):
        video = base64.b64decode((fixture_root / (name + '.mp4.b64')).read_bytes())
        video_info = tarfile.TarInfo('Fixtures/' + name + '.mp4')
        video_info.size = len(video)
        archive.addfile(video_info, io.BytesIO(video))
    # Tiny UI/export fixtures derived from Android's instrumentation clip; see Tests/Fixtures/README.md.
    for fixture in ('ios-editor-fixture', 'ios-export-fixture'):
        video = base64.b64decode((root / ('Tests/Fixtures/' + fixture + '.mp4.b64')).read_bytes())
        fixture_info = tarfile.TarInfo('Fixtures/' + fixture + '.mp4')
        fixture_info.size = len(video)
        archive.addfile(fixture_info, io.BytesIO(video))
    raw = gzip.decompress((root.parent / 'tests/fixtures/on-device-y9-base-features.bin.gz').read_bytes())
    info = tarfile.TarInfo('Fixtures/base.bin')
    info.size = len(raw)
    archive.addfile(info, io.BytesIO(raw))
print(args.output)
