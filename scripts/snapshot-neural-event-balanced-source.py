#!/usr/bin/env python3
"""Validate or finally archive the event-balanced experiment's research sources.

The default is a no-write dry run. --create is reserved for the root agent's
explicit final trigger, after all editors finish and the full study completes.
Weights, videos, feature arrays, environments and secret files are excluded.
Historical worker versions remain in the verified prior immutable archive.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0088'))
EXPECTED_CONTRACT = '9f1ce53e50c5bd21a4365e0c30d62bb43dd7e72962d2d7a3341f720489aaebda'
PRIOR_ZIP_SHA256 = 'af7481edd9545cdd228127199e7727ef5737197bc177e09595b1cd10a894e914'
PRIOR_MANIFEST_SHA256 = 'e3f5aa5734705d66789396ac906113123e77441c092ecf609c731276c4d47c7f'
PREFLIGHT_REPORT_SHA256 = '370d41d60f05399bf1baf1d3b606a209d6642618323a75ca495546cddcb59f2d'

# Reuse the already-reviewed narrow path, Git and ZIP utilities without editing
# that historical helper. Its current source bytes are included in this archive.
spec = importlib.util.spec_from_file_location('prior_source_snapshot',
                                             REPO/'scripts/snapshot-neural-expanded-source.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
require, digest, load, safe_read = prior.require, prior.digest, prior.load, prior.safe_read


def contract_hash(contract):
    return digest(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def file_identity(path, expected=None):
    require(path.is_file() and not path.is_symlink(), f'Not a regular file: {path}')
    data = path.read_bytes()
    sha = digest(data)
    require(expected is None or sha == expected, f'External reference changed: {path}')
    return {'path': str(path), 'sha256': sha, 'sizeBytes': len(data)}


def verify_prior_archive(contract):
    reference = contract['referenceStudy']
    study = Path(reference['path'])
    registration = load(study/'preregistration.json')
    require(contract_hash(registration['contract']) == registration['sha256'] == reference['contractSha256'],
            'Invalid historical preregistration')
    bindings = {filename: file_identity(study/filename, reference[field]) for filename, field in (
        ('preregistration.json', 'preregistrationFileSha256'),
        ('report.json', 'reportSha256'), ('summary.json', 'summarySha256'))}
    leaf = study.parent/'source-snapshot'
    manifest_path, verification_path, zip_path = (leaf/name for name in (
        'snapshot-manifest.json', 'verification.json', 'research-source.zip'))
    verification, manifest = load(verification_path), load(manifest_path)
    require(verification['mode'] == 'created' and verification['zipRoundtripPassed'] is True,
            'Historical archive is not finalized and verified')
    require(verification['zipSha256'] == PRIOR_ZIP_SHA256
            and verification['manifestSha256'] == PRIOR_MANIFEST_SHA256,
            'Historical archive differs from recorded identity')
    zip_identity = file_identity(zip_path, PRIOR_ZIP_SHA256)
    manifest_identity = file_identity(manifest_path, PRIOR_MANIFEST_SHA256)
    require(manifest['contractSha256'] == reference['contractSha256'], 'Historical archive contract differs')
    require(manifest['gitHead'] == verification['gitHead'], 'Historical Git identity differs')
    with zipfile.ZipFile(zip_path) as archive:
        require(archive.testzip() is None, 'Historical archive CRC failed')
        names = archive.namelist()
        expected = {row['archivePath'] for row in manifest['files']} | {'snapshot-manifest.json'}
        require(len(names) == len(set(names)) and set(names) == expected, 'Historical ZIP inventory differs')
        require(archive.read('snapshot-manifest.json') == manifest_path.read_bytes(), 'Historical manifest bytes differ')
        for row in manifest['files']:
            data = archive.read(row['archivePath'])
            require(digest(data) == row['sha256'] and len(data) == row['sizeBytes'],
                    f'Historical archive member changed: {row["archivePath"]}')
        for name, sha in registration['contract']['code'].items():
            require(digest(archive.read('repository/analysis/'+name)) == sha,
                    f'Historical registered source differs: {name}')
        for version in manifest['workerSourceVersions']:
            members = {Path(row['archivePath']).name: row['archivePath'] for row in version['files']}
            source = archive.read(members['worker-source.py'])
            execution = json.loads(archive.read(members['execution.json']))
            require(digest(source) == version['workerSha256'] == execution['worker']['sha256'],
                    f'Historical worker version changed: {version["leaf"]}')
            require(execution['contractSha256'] == reference['contractSha256'], 'Historical worker contract differs')
    return {'archive': zip_identity, 'manifest': manifest_identity,
            'verification': file_identity(verification_path), 'gitHead': manifest['gitHead'],
            'contractSha256': reference['contractSha256'], 'registeredSourcesVerified': len(registration['contract']['code']),
            'workerVersionsVerified': len(manifest['workerSourceVersions']),
            'sourceFilesVerified': len(manifest['files']), 'zipRoundtripPassed': True,
            'referenceStudyBindings': bindings,
            'restore': 'Historical worker-source-versions remain separate in this prior ZIP; never replace them with current repository helper versions.'}


def inventory(preregistration, *, require_completed):
    registration = load(preregistration)
    contract = registration['contract']
    require(contract_hash(contract) == registration['sha256'] == EXPECTED_CONTRACT,
            'Invalid or unexpected event-balanced registration')
    require(len(contract['code']) == 13 and contract['experiment'] == 'per-original-rally-live-loss-v1',
            'Expected thirteen registered event-balanced sources')
    study, root = preregistration.parent, preregistration.parent.parent
    report_path = study/'report.json'
    report_identity = None
    if report_path.exists():
        report = load(report_path)
        expected_jobs = {(c, 'tcn', s) for c in contract['cohorts'] for s in contract['seeds']}
        actual_jobs = {(r['cohort'], r['kind'], r['seed']) for r in report['results']}
        require(report['status'] == 'completed-event-balanced-development'
                and report['contractSha256'] == registration['sha256']
                and report['manifestSha256'] == contract['manifestSha256']
                and len(report['results']) == len(expected_jobs) and actual_jobs == expected_jobs
                and not report['protectedTestOpened'] and not report['productionPromotionAllowed'],
                'Final study report is incomplete or has a different scope')
        require(all(r['contractSha256'] == registration['sha256'] for r in report['results']),
                'Final result partition contract differs')
        report_identity = file_identity(report_path)
    require(not require_completed or report_identity is not None, 'Full completed study report required before --create')
    historical = verify_prior_archive(contract)
    prefix = prior.git_prefix()
    head = prior.git(prefix, 'rev-parse', 'HEAD').decode().strip()
    changed = set(prior.git(prefix, 'diff', '--name-only', '-z', 'HEAD', '--').decode().split('\x00'))-{''}
    untracked = set(prior.git(prefix, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\x00'))-{''}
    tracked = set(prior.git(prefix, 'ls-files', '-z').decode().split('\x00'))-{''}
    registered = {'analysis/'+name: sha for name, sha in contract['code'].items()}
    candidates = {name for name in tracked | untracked if prior.research_allowed(name)} | set(registered)
    dependencies = {name for name in tracked if name.startswith('analysis/') and name.endswith('.py')}
    require(not ((changed & dependencies)-candidates), 'Changed analysis dependency falls outside research allowlist')
    candidates |= dependencies
    blobs = {}
    for item in prior.git(prefix, 'ls-tree', '-r', '-z', 'HEAD', '--', 'analysis').split(b'\x00'):
        if item:
            attributes, name = item.split(b'\t', 1)
            blobs[name.decode()] = attributes.decode().split()[2]
    sources, origins, rows = {}, {}, []

    def add(path, allowed_root, member, expected=None, **extra):
        require(member not in sources, f'Duplicate archive path: {member}')
        data = safe_read(path, allowed_root)
        sha = digest(data)
        require(expected is None or sha == expected, f'Source hash differs: {path}')
        sources[member], origins[member] = data, path
        rows.append({'archivePath': member, 'sourcePath': str(path), 'sha256': sha,
                     'sizeBytes': len(data), **extra})

    for name in sorted(candidates):
        extra = {'repositoryPath': name, 'registered': name in registered,
                 'gitState': 'untracked' if name in untracked else 'modified' if name in changed else 'unchanged'}
        if name in blobs:
            extra['headGitBlob'] = blobs[name]
        add(REPO/name, REPO, 'repository/'+name, registered.get(name), **extra)
    add(preregistration, study, 'provenance/preregistration.json')
    preflight = root/'preflight'
    preflight_path = preflight/'preflight-report.json'
    preflight_report = load(preflight_path)
    control_path = preflight/'control-contract.json'
    control = load(control_path)
    require(preflight_report['passed'] is True and preflight_report['kind'] == 'event-balanced-uniform-control-preflight-v1',
            'Uniform-control preflight did not pass')
    require(contract_hash(control['contract']) == control['sha256'] == preflight_report['controlContractSha256']
            and control['contract'] == preflight_report['controlContract'], 'Preflight control contract differs')
    require(control['contract']['code'] == contract['code'] == preflight_report['code']
            and control['contract']['environment'] == contract['environment']
            and control['contract']['manifestSha256'] == contract['manifestSha256']
            and control['contract']['historicalContractSha256'] == contract['referenceStudy']['contractSha256']
            and control['contract']['weighting'] == 'uniform', 'Preflight source/environment/manifest binding differs')
    add(preflight_path, preflight, 'preflight/preflight-report.json', PREFLIGHT_REPORT_SHA256)
    add(control_path, preflight, 'preflight/control-contract.json')
    script_path = preflight/'preflight-source.py'
    require(Path(preflight_report['scriptSnapshot']['path']) == script_path, 'Unexpected preflight script snapshot path')
    require(preflight_report['scriptSnapshot']['sha256'] == preflight_report['script']['sha256'],
            'Preflight script and saved source differ')
    add(script_path, preflight, 'preflight/preflight-source.py', preflight_report['scriptSnapshot']['sha256'])
    snapshots = preflight_report['codeSnapshots']
    require(len(snapshots) == len(contract['code']) and
            {Path(row['path']).name for row in snapshots} == set(contract['code']), 'Preflight source inventory differs')
    for row in snapshots:
        path = Path(row['path'])
        require(path == preflight/'sources'/path.name and row['sha256'] == contract['code'][path.name],
                'Unexpected preflight source path or identity')
        add(path, preflight, 'preflight/sources/'+path.name, row['sha256'])
    comparisons = preflight_report['comparisons']
    require(len(comparisons) == 3 and {row['cohort'] for row in comparisons} == set(contract['cohorts']),
            'Preflight cohort coverage differs')
    for row in comparisons:
        require(all(row[key] is True for key in ('historyBitExact', 'exposureBitExact', 'optimizerStepsIdentical'))
                and row['weights']['bitExact'] is True and row['predictions']['bitExact'] is True,
                'Uniform control was not bit-exact')
        completed = Path(row['replayCompleted']['path'])
        require(completed == preflight/'control-replay'/row['cohort']/'completed.json',
                'Unexpected preflight completed metadata path')
        add(completed, preflight, 'preflight/control-replay/'+row['cohort']+'/completed.json',
            row['replayCompleted']['sha256'])
        require(load(completed)['contractSha256'] == control['sha256'], 'Preflight fit contract differs')
    historical_leaf = Path(historical['archive']['path']).parent
    for filename in ('snapshot-manifest.json', 'verification.json'):
        add(historical_leaf/filename, historical_leaf, 'historical-archive/'+filename)
    require(sum(map(len, sources.values())) <= prior.MAX_TOTAL_BYTES, 'Snapshot exceeds source-only size budget')
    manifest = {'schemaVersion': 1, 'kind': 'event-balanced-neural-research-source-snapshot',
        'createdAt': datetime.now(timezone.utc).isoformat(), 'gitHead': head, 'repositoryPath': str(REPO),
        'contractSha256': registration['sha256'], 'registeredSourcesVerified': len(registered),
        'completedStudyReport': report_identity, 'historicalArchive': historical,
        'preflightControlContractSha256': control['sha256'], 'preflightSourcesVerified': len(snapshots),
        'preflightReportSha256': PREFLIGHT_REPORT_SHA256, 'allowlistPatterns': list(prior.PATTERNS),
        'dependencyPolicy': 'Include all tracked analysis Python dependencies with exact working bytes, Git HEAD and available blob IDs.',
        'restore': 'Checkout gitHead and overlay repository/. Keep preflight source versions separate. Use the verified prior archive for historical worker versions. Model/data artifacts remain at their bound NAS locations.',
        'scope': 'Research code/docs and preflight provenance only; no videos, feature arrays, tensor weights, secrets, environment directories or home trees.',
        'files': rows, 'sourceFileCount': len(sources), 'sourceBytes': sum(map(len, sources.values())),
        'untrackedOutsideAllowlistExcluded': sorted(untracked-candidates),
        'changedOutsideAllowlistExcluded': sorted(changed-candidates)}
    return manifest, sources, origins, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, default=ROOT/'study/preregistration.json')
    parser.add_argument('--output', type=Path, default=ROOT/'source-snapshot')
    parser.add_argument('--create', action='store_true', help='Only after the root agent explicitly says all local docs and source edits are final')
    args = parser.parse_args()
    require(not args.create or not args.output.exists(), f'Immutable output already exists: {args.output}')
    manifest, sources, origins, prefix = inventory(args.preregistration, require_completed=args.create)
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    zip_bytes = prior.zip_roundtrip(sources, manifest_bytes)
    for member, path in origins.items():
        require(path.read_bytes() == sources[member], f'Source changed during snapshot: {path}')
    require(prior.git(prefix, 'rev-parse', 'HEAD').decode().strip() == manifest['gitHead'], 'Git HEAD changed during snapshot')
    historical = manifest['historicalArchive']
    for identity in (historical['archive'], historical['manifest'], historical['verification'],
                     *historical['referenceStudyBindings'].values()):
        file_identity(Path(identity['path']), identity['sha256'])
    if manifest['completedStudyReport']:
        identity = manifest['completedStudyReport']
        file_identity(Path(identity['path']), identity['sha256'])
    verification = {'mode': 'created' if args.create else 'validation-only-no-writes',
        'contractSha256': manifest['contractSha256'], 'registeredSourcesVerified': manifest['registeredSourcesVerified'],
        'preflightSourcesVerified': manifest['preflightSourcesVerified'],
        'historicalRegisteredSourcesVerified': historical['registeredSourcesVerified'],
        'historicalWorkerVersionsVerified': historical['workerVersionsVerified'],
        'historicalArchiveSha256': historical['archive']['sha256'], 'fullStudyComplete': manifest['completedStudyReport'] is not None,
        'sourceFileCount': len(sources), 'sourceBytes': manifest['sourceBytes'], 'zipBytes': len(zip_bytes),
        'zipSha256': digest(zip_bytes), 'manifestSha256': digest(manifest_bytes),
        'allSourceBytesStable': True, 'zipRoundtripPassed': True, 'gitHead': manifest['gitHead']}
    if args.create:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, data in (('research-source.zip', zip_bytes), ('snapshot-manifest.json', manifest_bytes),
                           ('verification.json', (json.dumps(verification, indent=2)+'\n').encode())):
            with (args.output/name).open('xb') as stream:
                stream.write(data)
        with zipfile.ZipFile(io.BytesIO((args.output/'research-source.zip').read_bytes())) as archive:
            require(archive.testzip() is None, 'Saved ZIP CRC failed')
            for member, data in sources.items():
                require(archive.read(member) == data, f'Saved ZIP bytes differ: {member}')
            require(archive.read('snapshot-manifest.json') == manifest_bytes, 'Saved manifest bytes differ')
        file_identity(args.output/'research-source.zip', verification['zipSha256'])
        file_identity(args.output/'snapshot-manifest.json', verification['manifestSha256'])
        verification['output'] = str(args.output)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
