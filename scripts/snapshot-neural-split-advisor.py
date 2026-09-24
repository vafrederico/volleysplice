#!/usr/bin/env python3
"""Archive completed research bytes without changing registered numerical inputs."""
from analysis.private_ledger import private_value
from pathlib import Path
import hashlib
import json
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0093'))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha_bytes(path.read_bytes()), 'sizeBytes': path.stat().st_size}


def main():
    reg = json.loads((ROOT/'registration.json').read_text())
    for entry in [*reg['contract']['sources'].values(), *reg['contract']['sourceCopies'].values(),
                  reg['contract']['input'], reg['contract']['probabilities']]:
        assert sha_bytes(Path(entry['path']).read_bytes()) == entry['sha256'], entry['path']
    assert json.loads((ROOT/'report.json').read_text())['passed']
    assert json.loads((ROOT/'summary-audit-v1.json').read_text())['passed']
    interpretation = json.loads((ROOT/'interpretation-audit-v1.json').read_text())
    assert interpretation['passed']
    for key in ('summary', 'diagnosis', 'document', 'readme', 'auditor'):
        entry = interpretation[key]
        assert sha_bytes(Path(entry['path']).read_bytes()) == entry['sha256'], key
    supplement = [
        'docs/research/neural-split-advisor-results-2026-09-19.md',
        'docs/research/neural-split-advisor-protocol-2026-09-19.md',
        'docs/research/README.md',
        'scripts/plot-neural-split-advisor.py', 'scripts/diagnose-neural-split-errors.py',
        'scripts/audit-neural-split-summary.py', 'scripts/snapshot-neural-split-advisor.py',
        'scripts/audit-neural-split-interpretation.py',
    ]
    # Supplementary post-freeze audit tests, if present, belong in the archive too.
    supplement.extend(str(p.relative_to(REPO)) for p in (REPO/'analysis/tests').glob('*split*summary*audit*.py'))
    files = {str(p.relative_to(ROOT)): p for p in ROOT.rglob('*')
             if p.is_file() and 'reproducibility' not in p.relative_to(ROOT).parts}
    files.update({'supplement/'+name: REPO/name for name in supplement})
    entries = [{'name': name, **ref(path)} for name, path in sorted(files.items())]
    manifest = {'kind': 'compact-split-adviser-byte-snapshot-v1',
      'contractSha256': reg['sha256'], 'registeredSourceCount': len(reg['contract']['sources']),
      'fileCount': len(entries), 'entries': entries,
      'reproductionNote': 'Frozen numerical source, normalized inputs and all results included. Registered paths record the original environment. This receipt verifies archive bytes; it does not claim a relocated replay was executed.'}
    dest = ROOT/'reproducibility'; dest.mkdir(exist_ok=False)
    manifest_path = dest/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    archive = dest/'research-source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for entry in entries:
            z.write(entry['path'], entry['name'])
        z.write(manifest_path, 'snapshot-manifest.json')
    with zipfile.ZipFile(archive, 'r') as z:
        assert set(z.namelist()) == {x['name'] for x in entries}|{'snapshot-manifest.json'}
        for entry in entries:
            value = z.read(entry['name'])
            assert len(value) == entry['sizeBytes'] and sha_bytes(value) == entry['sha256'], entry['name']
        assert z.read('snapshot-manifest.json') == manifest_path.read_bytes()
    for entry in entries:
        assert sha_bytes(Path(entry['path']).read_bytes()) == entry['sha256'], 'Changed during archive: '+entry['path']
    result = {'passed': True, 'contractSha256': reg['sha256'], 'filesVerified': len(entries),
              'zip': ref(archive), 'manifest': ref(manifest_path),
              'sourceBytesUnchanged': True, 'exactArchiveByteVerification': True}
    (dest/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
