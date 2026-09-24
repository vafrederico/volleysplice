#!/usr/bin/env python3
"""Archive the completed fixed combination study, never import model code.

Run only after final document review. Default validates in memory; --create writes
an immutable snapshot, restores every ZIP entry and compares exact binary bytes.
Prior large results remain explicitly hash-bound; feature/video arrays are omitted.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0083'))
CONTRACT = 'c0656f85658e6eade8f4e8feacbeb88453186c3f1c3b7d38ab332eb105f47b27'
REGISTERED = {
    'analysis/neural_production_combinations.py',
    'scripts/run-neural-production-combinations.py',
    'scripts/audit-neural-combination-accounting.py',
    'scripts/prepare-neural-production-comparison.py',
    'analysis/neural_evaluation.py', 'analysis/crop_evaluation.py',
    'analysis/schema.py', 'analysis/metrics.py',
    'analysis/tests/test_neural_production_combinations.py',
    'analysis/tests/test_neural_combination_accounting.py',
    'analysis/tests/test_neural_production_comparison_input.py',
    'analysis/tests/test_neural_combination_runner.py',
}
EXTRA_SOURCE = {
    'scripts/audit-neural-combination-recipes.py',
    'analysis/tests/test_neural_combination_recipes.py',
    'scripts/summarize-neural-production-combinations.py',
    'scripts/snapshot-neural-production-combinations.py',
    'analysis/tests/test_neural_production_combinations_snapshot.py',
    'docs/research/neural-production-combinations-protocol-2026-09-19.md',
    'docs/research/neural-production-combinations-results-2026-09-19.md',
    'docs/research/README.md', 'docs/model-ranking-metric.md',
    'analysis/__init__.py', 'analysis/requirements.txt',
}
FORBIDDEN = {'.git', 'node_modules', '__pycache__', 'fits', 'videos'}
SOURCE_SUFFIXES = {'.py', '.ts', '.java', '.kt', '.json', '.md', '.txt', '.log'}
MAX_FILE = 128 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def safe_name(name):
    require(isinstance(name, str) and name, 'Empty archive path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and '\\' not in name
            and ':' not in name and '..' not in p.parts
            and not any(x in FORBIDDEN or x.startswith('.env') for x in p.parts),
            'Unsafe archive path: ' + name)
    return name


def safe_path(path, base):
    path, base = Path(path), Path(base)
    require(path.is_absolute() and base.is_absolute(), 'Absolute source paths required')
    require(not any(x.is_symlink() for x in [path, *path.parents]), 'Symlink source path')
    require(path.resolve().is_relative_to(base.resolve()), 'Source escapes allowed root')
    safe_name(path.relative_to(base).as_posix())
    require(path.is_file(), 'Missing regular source: ' + str(path))
    return path


def identity(path):
    path = Path(path)
    require(path.is_file() and not any(x.is_symlink() for x in [path, *path.parents]),
            'Missing or symlink evidence: ' + str(path))
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Evidence changed while hashing: ' + str(path))
    return {'path': str(path), 'sha256': h.hexdigest(), 'sizeBytes': after.st_size}


def matches(actual, expected):
    return all(actual.get(k) == expected[k] for k in ('path', 'sha256', 'sizeBytes') if k in expected)


class Inventory:
    def __init__(self):
        self.entries = {}
        self.bound = {}
        self.external = []
        self.closed_directories = {}
        self.total = 0

    def bind(self, path, expected=None):
        row = identity(path)
        require(expected is None or matches(row, expected), 'Bound evidence changed: ' + str(path))
        require(str(path) not in self.bound or self.bound[str(path)] == row, 'Evidence changed during capture')
        self.bound[str(path)] = row
        return row

    def add(self, path, base, name, expected=None, allow_npz=False):
        path = safe_path(path, base)
        require(path.suffix in SOURCE_SUFFIXES or allow_npz and path.suffix == '.npz',
                'Disallowed source type: ' + str(path))
        row = self.bind(path, expected)
        require(row['sizeBytes'] <= MAX_FILE, 'Source exceeds individual budget')
        data = path.read_bytes()
        require(sha(data) == row['sha256'], 'Source changed during read')
        name = safe_name(name)
        if name in self.entries:
            require(self.entries[name]['identity'] == row, 'Archive path collision')
            return row
        self.total += len(data)
        require(self.total <= MAX_TOTAL, 'Archive exceeds total uncompressed budget')
        self.entries[name] = {'identity': row, 'data': data}
        return row

    def read(self, path, expected=None):
        self.bind(path, expected)
        data = Path(path).read_bytes()
        require(sha(data) == self.bound[str(path)]['sha256'], 'JSON changed during read')
        return json.loads(data)

    def external_binding(self, ref, role, verify=True):
        require(set(('path', 'sha256')) <= set(ref), 'External binding is incomplete')
        require(len(ref['sha256']) == 64, 'External hash missing')
        if verify:
            row = self.bind(Path(ref['path']), ref)
        else:
            row = {k: ref[k] for k in ('path', 'sha256', 'sizeBytes')}
        self.external.append({'role': role, 'verifiedDuringArchive': verify, **row})

    def recheck(self):
        for path, row in list(self.bound.items()):
            require(identity(path) == row, 'Source changed before archive completion: ' + path)
        for folder, names in self.closed_directories.items():
            require({p.name for p in Path(folder).iterdir()} == names,
                    'Closed evidence directory changed: ' + folder)

    def close_directory(self, folder, expected):
        require({p.name for p in Path(folder).iterdir()} == expected,
                'Unbound files in closed evidence directory: ' + str(folder))
        self.closed_directories[str(folder)] = set(expected)


def completion_gate(reg, report, summary, audit, report_id, audit_id):
    c = reg['contract']
    require(canonical(c) == reg['sha256'] == CONTRACT, 'Wrong combination contract')
    require(set(c['code']) == REGISTERED and c['newFits'] == 0
            and c['protectedTestOpened'] is False and c['productionPromotionAllowed'] is False,
            'Registered scope differs')
    require(report['status'] == 'completed-fixed-combinations'
            and report['contractSha256'] == summary['contractSha256'] == audit['contractSha256'] == CONTRACT,
            'Study is incomplete or unrelated')
    require(summary['kind'] == 'audited-production-combinations-summary-v1' and summary['passed'] is True
            and audit['kind'] == 'independent-production-combination-recipe-audit-v1' and audit['passed'] is True,
            'Completed summary and independent recipe audit required')
    require(all(x['protectedTestOpened'] is False for x in (report, summary, audit))
            and report['productionChanged'] is summary['productionChanged'] is False
            and summary['productionPromotionAllowed'] is False and report['newFits'] == 0,
            'Forbidden test, production or training scope')
    require(matches(report_id, audit['report'])
            and matches(report_id, summary['artifacts']['report'])
            and matches(audit_id, summary['artifacts']['recipeAudit']), 'Audit/summary binding differs')
    require(report['counts'] == {'configurations': 107, 'automaticCells': 303, 'reviewCells': 90,
                                'accountingScopesPerCell': 13, 'paddingCasesPerScope': 4}
            and len(report['automaticResults']) == 303 and len(report['reviewResults']) == 90
            and len(summary['automatic']) == 107 and len(summary['reviews']) == 30,
            'Incomplete fixed matrix')
    require([audit[k] for k in ('automaticCellsAudited', 'automaticRecordingConstructionsAudited',
            'reviewCellsAudited', 'reviewRecordingPaddingRowsAudited',
            'actualAppOverrideRecordingPaddingRowsAudited')] == [303, 2424, 90, 2880, 128],
            'Independent recipe coverage is incomplete')


def native_git(repo):
    """Use Git for Windows: WSL Git cannot resolve this Windows worktree."""
    exe = '/mnt/c/Program Files/Git/cmd/git.exe'
    windows = subprocess.check_output(['wslpath', '-w', str(repo)], text=True).strip()
    def run(*args):
        return subprocess.check_output([exe, '-C', windows, *args]).decode('utf-8').replace('\r\n', '\n')
    return {'implementation': exe, 'repository': windows,
            'version': run('--version').strip(), 'head': run('rev-parse', 'HEAD').strip(),
            'statusPorcelainV1': run('status', '--porcelain=v1', '--untracked-files=all')}


def root_metadata_files(root):
    """Only known final outputs and bounded top-level administrative evidence."""
    names = {'registration.json', 'protocol-initial.md', 'qualification-tests-v1.json',
             'qualification-tests-v1.txt', 'production-input.json', 'neural-input.json',
             'adapter-attempt-1.json', 'report.json', 'recipe-audit-v1.json', 'summary.json'}
    for p in root.iterdir():
        if p.is_file() and p.suffix in {'.json', '.txt', '.log', '.md'}:
            names.add(p.name)
    return names


def build_inventory(root, repo):
    inv = Inventory()
    reg = inv.read(root/'registration.json')
    c = reg['contract']
    require(reg['sha256'] == CONTRACT and canonical(c) == CONTRACT, 'Wrong study registration')
    report = inv.read(root/'report.json')
    summary = inv.read(root/'summary.json')
    audit = inv.read(root/'recipe-audit-v1.json')
    interpretation = inv.read(root/'interpretation-audit-v1.json')
    completion_gate(reg, report, summary, audit, inv.bound[str(root/'report.json')],
                    inv.bound[str(root/'recipe-audit-v1.json')])
    for rel, ref in c['code'].items():
        require(str(repo/rel) == ref['path'], 'Registered repository path differs')
        inv.add(repo/rel, repo, 'repository/'+rel, ref)
    for rel in sorted(EXTRA_SOURCE):
        inv.add(repo/rel, repo, 'repository/'+rel)
    require(interpretation['kind'] == 'independent-production-combination-interpretation-audit-v1'
            and interpretation['passed'] is True and interpretation['contractSha256'] == CONTRACT
            and interpretation['protectedTestOpened'] is False
            and interpretation['productionChanged'] is False
            and interpretation['productionPromotionAllowed'] is False
            and interpretation['independentAggregation']['numericComparisons'] == 33229
            and interpretation['independentAggregation']['rankingPassed'] is True
            and interpretation['independentAggregation']['conservativeScreensPassedAudit'] is True,
            'Final independent interpretation audit is incomplete')
    for ref in interpretation['artifacts'].values():
        require(ref['path'] in inv.bound and matches(inv.bound[ref['path']], ref),
                'Interpretation audit final source/output binding differs')
    for key in ('productionInput', 'neuralInput', 'protocol'):
        inv.bind(c[key]['path'], c[key])
    require(matches(inv.bound[str(root/'registration.json')], summary['artifacts']['registration'])
            and matches(inv.bound[str(repo/'scripts/summarize-neural-production-combinations.py')],
                        summary['artifacts']['summarizer'])
            and matches(inv.bound[str(repo/'scripts/audit-neural-combination-recipes.py')], audit['implementation']),
            'Summary/audit source binding differs')
    for name in sorted(root_metadata_files(root)):
        inv.add(root/name, root, 'study/'+name)
    qualification = inv.read(root/'qualification-tests-v1.json')
    require(qualification['passed'] is True and qualification['exitCode'] == 0, 'Qualification incomplete')
    inv.bind(qualification['log']['path'], qualification['log'])
    for field, folder in [('automaticResults', 'results'), ('reviewResults', 'reviews')]:
        refs = report[field]
        expected = set()
        for ref in refs:
            path = Path(ref['path'])
            require(path.parent == root/folder and path.suffix == '.json', 'Unexpected result path')
            require(path.name not in expected, 'Duplicate result reference')
            expected.add(path.name)
            inv.add(path, root, 'study/'+path.relative_to(root).as_posix(), ref)
        inv.close_directory(root/folder, expected)
    production = inv.read(c['productionInput']['path'], c['productionInput'])
    neural = inv.read(c['neuralInput']['path'], c['neuralInput'])
    for ref in production['sourceCode']:
        p = Path(ref['path'])
        inv.add(p, repo, 'repository/'+p.relative_to(repo).as_posix(), ref)
    for group in ('browserRuntimes', 'androidRuntimes'):
        for ref in production[group].values():
            p = Path(ref['path'])
            inv.add(p, repo, 'repository/'+p.relative_to(repo).as_posix(), ref)
    for key in ('baseline', 'repairedManifest', 'exactManifest'):
        ref = production[key]
        inv.add(ref['path'], root.parent, 'provenance/production/'+key+'.json', ref)
    for row in production['refittedArtifactVerification']:
        for key in ('model', 'weights'):
            ref = row[key]
            rel = '/'.join([row['heldGroup'], row['bundle'], row['role'], Path(ref['path']).name])
            inv.add(ref['path'], root.parent, 'provenance/refitted/'+rel, ref, allow_npz=key == 'weights')
    expected_probabilities = set()
    for rec in production['recordings']:
        ref = rec['probabilities']
        p = Path(ref['path'])
        require(p.parent == root/'production-probabilities' and ref['sizeBytes'] <= 1024*1024,
                'Unexpected or oversized probability archive')
        expected_probabilities.add(p.name)
        inv.add(p, root, 'study/production-probabilities/'+p.name, ref, allow_npz=True)
        inv.external_binding(rec['featureCache'], 'feature cache; existing input-provenance binding, array omitted', verify=False)
    inv.close_directory(root/'production-probabilities', expected_probabilities)
    attempt = root/'adapter-attempt-1-duration-rounding'
    require(attempt.is_dir() and not attempt.is_symlink(), 'Missing preserved failed adapter attempt')
    for p in sorted(attempt.iterdir()):
        require(p.is_file() and p.suffix == '.npz' and p.stat().st_size <= 1024*1024,
                'Unexpected failed-attempt artifact')
        inv.add(p, root, 'study/'+p.relative_to(root).as_posix(), allow_npz=True)
    inv.close_directory(attempt, {p.name for p in attempt.iterdir()})
    for name, bindings in c['referenceStudies'].items():
        for label, ref in bindings.items():
            safe_path(ref['path'], root.parent)
            require(Path(ref['path']).suffix == '.json', 'Unexpected external metadata type')
            inv.external_binding(ref, 'completed reference '+name+'/'+label)
    for ref in neural['sourceBindings']:
        safe_path(ref['path'], root.parent)
        require(Path(ref['path']).suffix == '.json', 'Unexpected original neural result type')
        inv.external_binding(ref, 'original neural result '+ref['model']+'/'+str(ref['seed']))
    return inv, reg


def zip_roundtrip(archive, contents, restore_parent):
    """Restore only prevalidated names and compare exact bytes, including NPZ."""
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix='.snapshot-restore-', dir=restore_parent) as tmp:
        require(len(z.namelist()) == len(contents) and set(z.namelist()) == set(contents), 'ZIP entry inventory differs')
        require(z.testzip() is None, 'ZIP integrity failure')
        for name, original in contents.items():
            safe_name(name)
            restored = Path(tmp)/name
            restored.parent.mkdir(parents=True, exist_ok=True)
            restored.write_bytes(z.read(name))
            require(restored.read_bytes() == original, 'ZIP binary round-trip mismatch: ' + name)


def main(root, repo, create):
    require(root.is_dir() and not root.is_symlink(), 'Missing experiment root')
    target = root/'source-snapshot'
    require(not target.exists(), 'Immutable snapshot already exists')
    before_meta = root_metadata_files(root)
    git = native_git(repo)
    inv, reg = build_inventory(root, repo)
    manifest = {'kind': 'production-combinations-reproducible-snapshot-v1',
                'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': reg['sha256'],
                'registeredSourcesVerified': 12, 'nativeGit': git,
                'protectedTestOpened': False, 'productionChanged': False,
                'files': [{'archivePath': name, **value['identity']} for name, value in sorted(inv.entries.items())],
                'externalBindings': inv.external,
                'omissions': 'Videos and feature arrays omitted; feature identities retained from frozen input provenance. Prior large study/results remain at verified hash-bound paths.',
                'uncompressedSourceBytes': inv.total}
    manifest_bytes = encoded(manifest)
    contents = {name: row['data'] for name, row in inv.entries.items()}
    contents['snapshot-manifest.json'] = manifest_bytes
    inv.recheck()
    require(native_git(repo) == git and root_metadata_files(root) == before_meta, 'Source tree or evidence inventory changed')
    receipt = {'kind': 'production-combinations-source-snapshot-verification-v1',
               'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
               'mode': 'created' if create else 'validated', 'passed': True,
               'registeredSourcesVerified': 12, 'fileCount': len(inv.entries),
               'sourceBytes': inv.total, 'manifestSha256': sha(manifest_bytes),
               'allSourceBytesStable': True, 'fullStudyComplete': True, 'nativeGitHead': git['head']}
    if create:
        # New-only output: never overwrite a prior archive, including a failed attempt.
        target.mkdir()
        (target/'snapshot-manifest.json').write_bytes(manifest_bytes)
        zpath = target/'research-source.zip'
        with zipfile.ZipFile(zpath, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 19, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, data)
        zip_roundtrip(zpath, contents, root)
        inv.recheck()
        require(native_git(repo) == git and root_metadata_files(root) == before_meta,
                'Source tree or evidence inventory changed during ZIP verification')
        receipt.update(zipSha256=identity(zpath)['sha256'], zipSizeBytes=zpath.stat().st_size,
                       zipRoundtripPassed=True, restoredBinaryEntries=len(contents))
        with (target/'verification.json').open('xb') as f:
            f.write(encoded(receipt))
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    main(args.root, args.repo, args.create)
