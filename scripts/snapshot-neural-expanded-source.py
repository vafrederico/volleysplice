#!/usr/bin/env python3
"""Snapshot bounded neural research sources after every editor has finished.

Default mode validates and prints an inventory without writing. --create is an
explicit final orchestration action: it requires a completed report and a new
output directory. No model weights, video, secrets, caches, or home trees are
included. Saved worker source versions remain distinct from final helper code.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0070'))
PATTERNS = (
    'analysis/*neural*.py', 'analysis/compact_temporal_model.py', 'analysis/expanded_temporal_model.py',
    'analysis/requirements-neural.txt', 'analysis/tests/test_*neural*.py',
    'analysis/tests/test_compact_temporal_model.py', 'analysis/tests/test_expanded_temporal_model.py',
    'scripts/*neural*.py', 'scripts/*neural*.mjs', 'docs/research/neural-*.md',
    'docs/research/neural-*.json', 'docs/research/neural-*.lock', 'docs/research/README.md',
)
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def git_prefix():
    native = shutil.which('git')
    if native:
        prefix = [native, '-C', str(REPO)]
        if subprocess.run([*prefix, 'rev-parse', '--show-toplevel'], capture_output=True).returncode == 0:
            return prefix
    # The Windows worktree .git pointer contains an absolute Windows path;
    # Linux git cannot interpret it. Use Windows git without changing its config.
    windows_git = shutil.which('git.exe') or '/mnt/c/Program Files/Git/cmd/git.exe'
    require(Path(windows_git).is_file(), 'No git executable can read this worktree')
    windows_repo = subprocess.check_output(['wslpath', '-w', str(REPO)], text=True).strip()
    return [windows_git, '-C', windows_repo]


def git(prefix, *args):
    return subprocess.check_output([*prefix, *args])


def safe_read(path, root):
    require(path.is_file() and not path.is_symlink(), f'Not a regular nonsymlink file: {path}')
    require(path.resolve().is_relative_to(root.resolve()), f'File escapes allowed root: {path}')
    require(path.stat().st_size <= MAX_FILE_BYTES, f'Source unexpectedly large: {path}')
    require(not any(part.startswith('.env') or part in ('.git', 'node_modules', '__pycache__')
                    for part in path.relative_to(root).parts), f'Excluded path: {path}')
    data = path.read_bytes()
    require(b'\x00' not in data, f'Binary file excluded: {path}')
    data.decode('utf-8')
    return data


def research_allowed(name):
    # fnmatch '*' also matches '/', so constrain these deliberately shallow roots.
    parts = Path(name).parts
    return len(parts) in (2, 3) and any(fnmatch.fnmatchcase(name, pattern) for pattern in PATTERNS)


def inventory(preregistration, *, require_completed):
    registration = load(preregistration)
    contract = registration['contract']
    encoded = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
    require(digest(encoded) == registration['sha256'], 'Invalid preregistration contract hash')
    require(len(contract['code']) == 11, 'Expected the eleven registered analysis sources')
    study = preregistration.parent
    expanded = study.parent
    if require_completed:
        report = load(study / 'report.json')
        require(report['contractSha256'] == registration['sha256']
                and report['status'] == 'completed-expanded-development-screen', 'Study is not complete')
    prefix = git_prefix()
    head = git(prefix, 'rev-parse', 'HEAD').decode().strip()
    changed = set(git(prefix, 'diff', '--name-only', '-z', 'HEAD', '--').decode().split('\x00')) - {''}
    untracked = set(git(prefix, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\x00')) - {''}
    tracked = set(git(prefix, 'ls-files', '-z').decode().split('\x00')) - {''}
    registered = {f'analysis/{name}': sha for name, sha in contract['code'].items()}
    candidates = {name for name in tracked | untracked if research_allowed(name)} | set(registered)
    # All tracked analysis Python sources are small explicit dependency files.
    # Store their bytes too, so checkout line-ending rules cannot impair recovery.
    dependencies = {name for name in tracked if name.startswith('analysis/') and name.endswith('.py')}
    require(not ((changed & dependencies) - candidates), 'Changed analysis dependency falls outside research allowlist')
    candidates |= dependencies
    blobs = {}
    for item in git(prefix, 'ls-tree', '-r', '-z', 'HEAD', '--', 'analysis').split(b'\x00'):
        if item:
            attributes, name = item.split(b'\t', 1)
            blobs[name.decode()] = attributes.decode().split()[2]
    sources, rows, origins = {}, [], {}
    for name in sorted(candidates):
        path = REPO / name
        data = safe_read(path, REPO)
        sha = digest(data)
        if name in registered:
            require(sha == registered[name], f'Registered source changed: {name}')
        member = 'repository/' + name
        sources[member] = data
        origins[member] = path
        row = {'archivePath': member, 'repositoryPath': name, 'sha256': sha, 'sizeBytes': len(data),
               'registered': name in registered,
               'gitState': 'untracked' if name in untracked else 'modified' if name in changed else 'unchanged'}
        if name in blobs:
            row['headGitBlob'] = blobs[name]
        rows.append(row)
    worker_versions = []
    # Scheduling workers use several historical leaf-name conventions. Inspect
    # only immediate children and require the exact pair of provenance files.
    for leaf in sorted(expanded.iterdir()):
        source, execution_path = leaf / 'worker-source.py', leaf / 'execution.json'
        if not source.is_file() or not execution_path.is_file():
            continue
        execution = load(execution_path)
        require(execution['contractSha256'] == registration['sha256'], f'Worker contract mismatch: {leaf.name}')
        source_data = safe_read(source, leaf)
        require(digest(source_data) == execution['worker']['sha256'], f'Worker source snapshot mismatch: {leaf.name}')
        version = {'leaf': str(leaf), 'workerSha256': digest(source_data), 'files': []}
        for filename in ('worker-source.py', 'worker-source.json', 'execution.json'):
            path = leaf / filename
            if not path.exists():
                continue
            data = safe_read(path, leaf)
            member = f'worker-source-versions/{leaf.name}/{filename}'
            sources[member] = data
            origins[member] = path
            info = {'archivePath': member, 'sourcePath': str(path), 'sha256': digest(data), 'sizeBytes': len(data)}
            rows.append(info)
            version['files'].append(info)
        worker_versions.append(version)
    data = safe_read(preregistration, study)
    sources['provenance/preregistration.json'] = data
    origins['provenance/preregistration.json'] = preregistration
    rows.append({'archivePath': 'provenance/preregistration.json', 'sourcePath': str(preregistration),
                 'sha256': digest(data), 'sizeBytes': len(data)})
    require(sum(len(value) for value in sources.values()) <= MAX_TOTAL_BYTES, 'Snapshot exceeds source-only size budget')
    manifest = {'schemaVersion': 1, 'kind': 'expanded-neural-research-source-snapshot',
                'createdAt': datetime.now(timezone.utc).isoformat(), 'gitHead': head,
                'repositoryPath': str(REPO), 'contractSha256': registration['sha256'],
                'registeredSourcesVerified': len(registered), 'allowlistPatterns': list(PATTERNS),
                'dependencyPolicy': 'Include tracked analysis/*.py recursively; clean dependencies are identified by git HEAD and blob ID. Preserve exact working bytes independently of checkout line endings.',
                'restore': 'Checkout gitHead, overlay repository/ files, restore worker-source-versions separately. NAS data/model artifacts remain external and retain their original hash bindings.',
                'scope': 'Research source/docs only. No secrets, datasets, videos, models, cache, environment directory, or whole-home snapshot.',
                'workerSourceVersions': worker_versions, 'files': rows,
                'untrackedOutsideAllowlistExcluded': sorted(untracked - candidates),
                'changedOutsideAllowlistExcluded': sorted(changed - candidates),
                'sourceFileCount': len(sources), 'sourceBytes': sum(map(len, sources.values()))}
    return manifest, sources, origins, prefix


def zip_roundtrip(sources, manifest_bytes):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(sources.items()):
            archive.writestr(name, data)
        archive.writestr('snapshot-manifest.json', manifest_bytes)
    with zipfile.ZipFile(buffer) as archive:
        require(archive.testzip() is None, 'ZIP CRC roundtrip failed')
        require(len(archive.namelist()) == len(sources) + 1, 'Duplicate archive members')
        for name, data in sources.items():
            require(archive.read(name) == data, f'ZIP bytes changed: {name}')
        require(archive.read('snapshot-manifest.json') == manifest_bytes, 'ZIP manifest changed')
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, default=ROOT / 'study/preregistration.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'source-snapshot')
    parser.add_argument('--create', action='store_true', help='Final orchestrator trigger only, after all source edits finish')
    args = parser.parse_args()
    require(not args.create or not args.output.exists(), f'Output exists; immutable snapshot cannot overwrite: {args.output}')
    manifest, sources, origins, prefix = inventory(args.preregistration, require_completed=args.create)
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    zip_bytes = zip_roundtrip(sources, manifest_bytes)
    # Refuse a moving source tree, including the helper itself, before writing.
    for member, path in origins.items():
        require(path.read_bytes() == sources[member], f'Source changed during snapshot: {path}')
    require(git(prefix, 'rev-parse', 'HEAD').decode().strip() == manifest['gitHead'], 'git HEAD changed during snapshot')
    verification = {'mode': 'created' if args.create else 'validation-only-no-writes',
                    'sourceFileCount': len(sources), 'registeredSourcesVerified': manifest['registeredSourcesVerified'],
                    'workerVersionsVerified': len(manifest['workerSourceVersions']),
                    'sourceBytes': manifest['sourceBytes'], 'zipBytes': len(zip_bytes),
                    'zipSha256': digest(zip_bytes), 'manifestSha256': digest(manifest_bytes),
                    'allSourceBytesStable': True, 'zipRoundtripPassed': True, 'gitHead': manifest['gitHead']}
    if args.create:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, data in (('research-source.zip', zip_bytes), ('snapshot-manifest.json', manifest_bytes),
                            ('verification.json', (json.dumps(verification, indent=2) + '\n').encode())):
            with (args.output / name).open('xb') as stream:
                stream.write(data)
        with zipfile.ZipFile(args.output / 'research-source.zip') as archive:
            require(archive.testzip() is None, 'Saved ZIP CRC failed')
            for name, data in sources.items():
                require(archive.read(name) == data, f'Saved ZIP bytes changed: {name}')
        require(digest((args.output / 'research-source.zip').read_bytes()) == verification['zipSha256'], 'Saved ZIP hash failed')
        verification['output'] = str(args.output)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
