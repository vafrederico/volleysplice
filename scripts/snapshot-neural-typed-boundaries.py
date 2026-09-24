#!/usr/bin/env python3
"""Preserve completed typed-boundary research; verify exact archive bytes."""
from analysis.private_ledger import private_value
from pathlib import Path
import hashlib
import json
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0079'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': digest(data), 'sizeBytes': len(data)}


def verify_references(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            actual = ref(Path(value['path']))
            assert actual['sha256'] == value['sha256'], value['path']
            if 'sizeBytes' in value:
                assert actual['sizeBytes'] == value['sizeBytes'], value['path']
        for child in value.values():
            verify_references(child)
    elif isinstance(value, list):
        for child in value:
            verify_references(child)


def main():
    reg = json.loads((ROOT/'registration.json').read_text())
    contract = reg['contract']
    assert digest(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) == reg['sha256']
    for entry in [*contract['sources'].values(), *contract['sourceCopies'].values(),
                  contract['input'], contract['probabilities'], contract['protocol'], contract['qualification']]:
        assert ref(Path(entry['path']))['sha256'] == entry['sha256'], entry['path']
    for name in ('report.json', 'summary-audit-v1.json', 'interpretation-audit-v1.json'):
        assert json.loads((ROOT/name).read_text())['passed'], name
    # The final interpretation receipt binds the final document; the earlier
    # summary receipt deliberately retains the pre-narrative document hash.
    for name in ('report.json', 'interpretation-audit-v1.json', 'figures/manifest.json',
                 'observed-start-diagnosis.json', 'clipped-start-resolution-diagnosis.json'):
        verify_references(json.loads((ROOT/name).read_text()))
    supplement = [
        'docs/research/neural-typed-boundary-results-2026-09-19.md',
        'docs/research/neural-typed-boundary-protocol-2026-09-19.md',
        'docs/research/README.md',
        'scripts/snapshot-neural-typed-boundaries.py',
        'scripts/diagnose-neural-boundary-observed-starts.py',
        'scripts/diagnose-neural-boundary-clipping-resolution.py',
    ]
    for pattern in ('scripts/*typed-boundar*.py', 'analysis/tests/*typed_boundar*.py'):
        supplement.extend(str(p.relative_to(REPO)) for p in REPO.glob(pattern))
    files = {str(p.relative_to(ROOT)): p for p in ROOT.rglob('*')
             if p.is_file() and 'reproducibility' not in p.relative_to(ROOT).parts}
    files.update({'supplement/'+name: REPO/name for name in sorted(set(supplement))})
    entries = [{'name': name, **ref(path)} for name, path in sorted(files.items())]
    manifest = {
        'kind': 'compact-typed-boundary-byte-snapshot-v1', 'contractSha256': reg['sha256'],
        'registeredSourceCount': len(contract['sources']), 'fileCount': len(entries), 'entries': entries,
        'reproductionNote': 'Includes frozen numerical sources, normalized inputs, results and supplementary reporting. Paths describe the original environment. Archive byte verification is not a relocated replay.'}
    dest = ROOT/'reproducibility'; dest.mkdir(exist_ok=False)
    manifest_path = dest/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    archive = dest/'research-source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for entry in entries:
            z.write(entry['path'], entry['name'])
        z.write(manifest_path, 'snapshot-manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {x['name'] for x in entries}|{'snapshot-manifest.json'}
        for entry in entries:
            data = z.read(entry['name'])
            assert len(data) == entry['sizeBytes'] and digest(data) == entry['sha256'], entry['name']
        assert z.read('snapshot-manifest.json') == manifest_path.read_bytes()
    for entry in entries:
        assert ref(Path(entry['path']))['sha256'] == entry['sha256'], entry['path']
    receipt = {'passed': True, 'contractSha256': reg['sha256'], 'filesVerified': len(entries),
               'zip': ref(archive), 'manifest': ref(manifest_path),
               'sourceBytesUnchanged': True, 'exactArchiveByteVerification': True,
               'relocatedReplayExecuted': False}
    (dest/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
