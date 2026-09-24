#!/usr/bin/env python3
"""Validate a completed context/keep-rescue source archive; write only with --create.

No training modules are imported and no tensor, feature or video files are read.
--create also requires at least one explicitly final repository research document.
Large reports stay at their hash-bound locations rather than entering the ZIP.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('followup_archive_primitives',
    REPO/'scripts/snapshot-neural-expanded-source.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
require, digest = prior.require, prior.digest
BASE_SOURCES = {
    'neural_expanded_development.py', 'expanded_temporal_model.py', 'neural_development.py',
    'compact_temporal_model.py', 'neural_evaluation.py', 'crop_evaluation.py', 'decoder.py',
    'features.py', 'config.py', 'schema.py', 'metrics.py', 'neural_event_balanced_development.py',
    'neural_event_weighting.py', 'neural_short_boost_transfer.py', 'neural_short_boost_weighting.py',
    'transfer_temporal_model.py',
}
EXPERIMENTS = {
    'short-context-development-v1': ('context', {
        'short_context_temporal_model.py', 'neural_context_fit.py', 'neural_context_development.py'}),
    'keep-head-short-rescue-development-v1': ('keep-rescue', {
        'neural_keep_rescue.py', 'neural_keep_rescue_development.py'}),
}
REFERENCE_AUDITS = {'neural-short-boost-independent-tensor-audit-v1',
                    'independent-short-boost-transfer-interval-audit-v1'}
COHORTS, KINDS = ['exact', 'draft', 'reviewed_export'], ['tcn', 'dino_tcn']
CPU_DIAGNOSIS = 'data/reports/neural-context-cpu-runtime-v1/diagnosis.json'
EXPLICIT_TESTS = {'analysis/tests/test_transfer_temporal_model.py',
                  'analysis/tests/test_short_context_temporal_model.py'}
CONTEXT_OPERATIONAL_DIRS = ('resource-monitor-v1', 'audit-stage/dependency-repair', 'runtime-restoration')
CONTEXT_OPERATIONAL_SUFFIXES = {'.py', '.ps1', '.json', '.jsonl', '.log', '.txt'}


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def binding(value):
    return {'path': value['path'], 'sha256': value['sha256']}


def contains(value, expected):
    if isinstance(value, dict):
        if value.get('path') == expected['path'] and value.get('sha256') == expected['sha256']:
            return True
        return any(contains(child, expected) for child in value.values())
    return isinstance(value, list) and any(contains(child, expected) for child in value)


class Evidence:
    """Lock explicit metadata/archive paths only; never follow arbitrary JSON paths."""
    def __init__(self):
        self.files = {}
        self.closed_directories = {}

    def identity(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), f'Not a regular evidence file: {path}')
        require(path.suffix in ('.json', '.md', '.py', '.log', '.txt', '.zip'),
                f'Nonmetadata evidence is forbidden: {path}')
        require(not any(part.startswith('.env') or part in ('.git', 'node_modules', '__pycache__')
                        for part in path.parts), f'Excluded evidence path: {path}')
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                h.update(block)
        row = {'path': str(path), 'sha256': h.hexdigest(), 'sizeBytes': path.stat().st_size}
        require(expected is None or row['sha256'] == expected, f'Evidence changed: {path}')
        previous = self.files.get(str(path))
        require(previous is None or previous == row, f'Evidence changed during inventory: {path}')
        self.files[str(path)] = row
        return row

    def read(self, path, expected=None):
        row = self.identity(path, expected)
        require(Path(path).suffix == '.json', 'Expected JSON evidence')
        return json.loads(Path(path).read_text(encoding='utf-8')), row

    def verified(self, value):
        return self.read(value['path'], value['sha256'])

    def recheck(self):
        for row in list(self.files.values()):
            self.identity(row['path'], row['sha256'])
        for folder, closed in self.closed_directories.items():
            require(operational_files(Path(folder), closed['root']) == closed['files'],
                    f'Closed operational evidence changed during snapshot: {folder}')


def operational_files(folder, task_root):
    """Read only text evidence inside one explicitly named, nonsymlink root."""
    folder, task_root = Path(folder), Path(task_root)
    require(task_root.is_dir() and not task_root.is_symlink(), 'Experiment evidence root must not be a symlink')
    require(folder.is_dir() and not folder.is_symlink(), f'Missing or symlink operational directory: {folder}')
    require(folder.resolve().is_relative_to(task_root.resolve()), 'Operational evidence escapes experiment root')
    relative = folder.relative_to(task_root)
    require(all(not (task_root/Path(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts)+1)), 'Operational evidence traverses a symlink directory')
    files = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), f'Symlink operational evidence is forbidden: {path}')
        name = path.relative_to(folder)
        require(not any(part.startswith('.env') or part in ('.git', 'node_modules', '__pycache__', 'fits')
                        for part in name.parts), f'Excluded operational evidence path: {path}')
        if path.is_dir():
            continue
        require(path.is_file() and path.suffix in CONTEXT_OPERATIONAL_SUFFIXES,
                f'Nontext operational evidence is forbidden: {path}')
        data = prior.safe_read(path, folder)
        files[name.as_posix()] = {'path': str(path), 'sha256': digest(data), 'sizeBytes': len(data)}
    return files


def verify_closed_inventory(files, marker_name, declared, size_key):
    expected = {}
    require(isinstance(declared, list), 'Closed evidence inventory must be a list')
    for row in declared:
        name = row['relativePath']
        path = PurePosixPath(name)
        require(isinstance(name, str) and name == path.as_posix() and not path.is_absolute()
                and '..' not in path.parts and '\\' not in name and name != marker_name,
                'Unsafe closed evidence relative path')
        require(name not in expected and type(row[size_key]) is int and row[size_key] >= 0,
                'Duplicate or invalid closed evidence inventory row')
        expected[name] = {'sha256': row['sha256'], 'sizeBytes': row[size_key]}
    actual = {name: {'sha256': row['sha256'], 'sizeBytes': row['sizeBytes']}
              for name, row in files.items() if name != marker_name}
    require(expected == actual, 'Closed evidence inventory differs from files')


def context_operational_evidence(study, evidence, add):
    """Include the three known evidence roots only after their writers finish."""
    root = Path(study).parent
    folders = [root/name for name in CONTEXT_OPERATIONAL_DIRS]
    if not any(path.exists() or path.is_symlink() for path in folders):
        return []
    require(all(path.is_dir() and not path.is_symlink() for path in folders),
            'Context operational evidence requires monitor, dependency repair and runtime restoration')
    scopes = []
    for label, folder in zip(CONTEXT_OPERATIONAL_DIRS, folders):
        files = operational_files(folder, root)
        marker_name = ('completion-evidence.json' if label == 'resource-monitor-v1' else
                       'verification.json' if label == 'audit-stage/dependency-repair' else 'completed.json')
        require(marker_name in files, f'Operational evidence is not completed: {label}')
        marker, marker_id = evidence.read(folder/marker_name, files[marker_name]['sha256'])
        if label == 'resource-monitor-v1':
            require(marker['kind'] == 'context-resource-observer-completion-evidence-v1'
                    and marker['wslExitCode'] == 0 and marker['hostExitCode'] == 0
                    and marker['allObserverProcessesExited'] is True and marker['finalTelemetryFilesClosed'] is True,
                    'Resource observers are not successfully stopped with closed telemetry')
            verify_closed_inventory(files, marker_name, marker['evidenceInventory'], 'bytes')
        elif label == 'audit-stage/dependency-repair':
            require({'diagnosis-before-fix.json', 'focused-tests.log', 'corrected-audit-neural-context-tensors.py'} <= set(files)
                    and marker['exitCode'] == 0 and marker['testCount'] > 0
                    and marker['registeredAnalysisSourceCount'] == 19 and marker['changedRegisteredAnalysisSources'] == []
                    and marker['correctedAuditorSha256'] == files['corrected-audit-neural-context-tensors.py']['sha256'],
                    'Dependency repair verification is incomplete or changes registered sources')
        else:
            require(marker['status'] == 'completed' and marker['trainingComplete'] is True
                    and marker['restorationVerified'] is True and marker['wslRestarted'] is False
                    and set(files) == {'completed.json', 'restoration.json'},
                    'Runtime restoration is not completed or contains unexpected files')
            verify_closed_inventory(files, marker_name, marker['files'], 'sizeBytes')
        for name, row in files.items():
            add(row['path'], folder, 'provenance/context-operation/'+label+'/'+name, row['sha256'])
        evidence.closed_directories[str(folder)] = {'root': root, 'files': files}
        scopes.append({'relativeDirectory': label, 'completion': marker_id, 'fileCount': len(files),
                       'fileInventorySha256': canonical(files)})
    return scopes


def validate_scope(registration, report, summary, report_id, tensor=None):
    """Metadata-only completion gate; quality calculations belong to bound audits."""
    c, sha = registration['contract'], registration['sha256']
    require(canonical(c) == sha and c['experiment'] in EXPERIMENTS, 'Invalid follow-up registration')
    mode, added = EXPERIMENTS[c['experiment']]
    require(set(c['code']) == BASE_SOURCES | added, 'Registered 19/18 source inventory differs')
    require(c['kinds'] == KINDS and len(c['seeds']) == len(set(c['seeds'])) == 3
            and len(c['groups']) == len(set(c['groups'])) == 4, 'Unexpected fold/seed/kind scope')
    require(private_value('source-group-008') not in c['groups'], 'Protected source group is forbidden')
    require(c['primaryMetric'] == 'F1_padP_coreR' and c['targetPaddingSeconds'] == 2
            and c['joinGapSeconds'] == 3 and c['paddingSweep'] == [0, 1, 2, 3], 'Metric contract differs')
    for value in (c, report, summary):
        require(value['protectedTestOpened'] is False and value['productionPromotionAllowed'] is False,
                'Protected-test/promotion status is forbidden')
    require(report['contractSha256'] == summary['contractSha256'] == sha
            and report['manifestSha256'] == c['manifestSha256'] and report['records'] == 8
            and report['sourceGroups'] == c['groups'] and summary['passed'] is True
            and contains(summary, report_id) and summary['referenceStudy'] == c['referenceStudy'],
            'Completed report/summary is missing or unrelated')
    if mode == 'context':
        require(c['cohort'] in COHORTS and c['cohorts'] == [c['cohort']]
                and c['contexts'] == ['original', 'short']
                and c['lossArm'] in ('baseline', 'short_boost', 'global_control'), 'Context scope differs')
        expected = {(c['cohort'], k, c['lossArm'], v, s) for k in KINDS
                    for v in c['contexts'] for s in c['seeds']}
        actual = {(r['cohort'], r['kind'], r['lossArm'], r['context'], r['seed']) for r in report['results']}
        require(report['status'] == 'completed-context-development'
                and summary['kind'] == 'neural-context-summary-v1'
                and summary['status'] == 'completed-context-development-audit'
                and report['execution']['physicalFreshFits'] == 60
                and report['execution']['logicalFreshFits'] == 96, 'Context completion differs')
        counts = {'physicalFreshFits': 60, 'physicalInnerOwners': 36, 'outerRefits': 24,
                  'logicalInnerViews': 72, 'logicalFreshFits': 96, 'freshCheckpoints': 168,
                  'freshNPZArtifacts': 336, 'referenceResultCells': 6}
        require(tensor is not None and tensor['kind'] == 'independent-context-tensor-audit-v1'
                and tensor['passed'] is True and tensor['contractSha256'] == sha
                and binding(tensor['report']) == binding(report_id) and tensor['code'] == c['code']
                and all(tensor['counts'][key] == value for key, value in counts.items()),
                'Missing, unrelated or incomplete context tensor audit')
        expected_audit = {'canonicalMetricReplays': 12, 'reusedPayloadsVerified': 6,
                          'freshSelectedInnerCandidatesReplayed': 24, 'freshSelectedRefitsReplayed': 24,
                          'fullInnerGridSelectionsReplayed': 24, 'independentPaddingScopeRows': 624}
    else:
        require(c['cohorts'] and c['cohorts'] == [name for name in COHORTS if name in c['cohorts']]
                and c['variants'] == ['reference', 'selected'] and c['baselineLossArm'] == 'baseline'
                and c['rescueOptions'] == [None, .35, .5, .65, .8], 'Keep-rescue scope differs')
        n = len(c['cohorts'])
        expected = {(cohort, k, 'baseline', v, s) for cohort in c['cohorts'] for k in KINDS
                    for v in c['variants'] for s in c['seeds']}
        actual = {(r['cohort'], r['kind'], r['lossArm'], r['rescueVariant'], r['seed']) for r in report['results']}
        require(report['status'] == 'completed-keep-rescue-development'
                and summary['kind'] == 'independent-neural-keep-rescue-summary-v1'
                and summary['status'] == 'completed-keep-rescue-development-audit'
                and report['execution']['newFits'] == summary['newFits'] == 0
                and report['execution']['candidateCells'] == report['execution']['referenceCells'] == 6*n
                and report['execution']['device'] == 'cpu' and summary['audit']['freshFits'] == 0
                and summary['audit']['freshNPZArtifacts'] == 0, 'Keep-rescue must be completed with zero new fits')
        counts = {'physicalFreshFits': 0, 'freshNPZArtifacts': 0, 'resultCells': 12*n}
        expected_audit = {'canonicalMetricReplays': 12*n, 'matchedBaselinePayloadsVerified': 6*n,
                          'outerRescueSelectionsReplayed': 24*n,
                          'innerOptionScoresIndependentlyReplayed': 120*n,
                          'selectedRefitsDecoded': 24*n, 'independentFinalPaddingScopeRows': 624*n}
        require(len(summary['audit']['originalCompletedFitsReferenced']) == 96*n
                and len(summary['audit']['originalNPZArtifactsReferenced']) == 192*n,
                'Referenced keep-rescue checkpoint inventory differs')
    require(actual == expected and len(report['results']) == len(expected)
            and all(r['contractSha256'] == sha for r in report['results'])
            and summary['scope']['resultCells'] == len(expected)
            and all(summary['audit']['counts'][k] == v for k, v in expected_audit.items()),
            'Result partition or passed-summary counts differ')
    return mode, counts


def verify_reference_archive(reference, evidence):
    """Verify the actual final transfer ZIP, including its report/summary bindings."""
    study = Path(reference['path'])
    reg, reg_id = evidence.read(study/'preregistration.json', reference['preregistrationFileSha256'])
    c = reg['contract']
    require(canonical(c) == reg['sha256'] == reference['contractSha256']
            and c['experiment'] == 'bounded-short-boost-dino-transfer-v1'
            and set(c['code']) == BASE_SOURCES, 'Invalid preceding transfer registration')
    report, report_id = evidence.read(study/'report.json', reference['reportSha256'])
    summary, summary_id = evidence.read(study/'summary.json', reference['summarySha256'])
    expected = {(cohort, kind, arm, seed) for cohort in COHORTS for kind in KINDS
                for arm in ('baseline', 'short_boost', 'global_control') for seed in c['seeds']}
    require(report['status'] == 'completed-short-boost-transfer-development'
            and summary['status'] == 'completed-short-boost-transfer-audit'
            and report['contractSha256'] == summary['contractSha256'] == reg['sha256']
            and len(report['results']) == len(expected) == 54
            and {(r['cohort'], r['kind'], r['lossArm'], r['seed']) for r in report['results']} == expected
            and report['protectedTestOpened'] is False and report['productionPromotionAllowed'] is False
            and contains(summary, report_id), 'Preceding transfer report/summary not completed and bound')
    audits = [evidence.verified(row)[0] for row in reference['independentAudits']]
    require(len(audits) == 2 and {a['kind'] for a in audits} == REFERENCE_AUDITS
            and all(a['passed'] is True and a['contractSha256'] == reg['sha256']
                    and contains(a, report_id) for a in audits), 'Preceding independent audits differ')
    leaf = study.parent/'source-snapshot'
    verification, verification_id = evidence.read(leaf/'verification.json')
    manifest, manifest_id = evidence.read(leaf/'snapshot-manifest.json', verification['manifestSha256'])
    zip_id = evidence.identity(leaf/'research-source.zip', verification['zipSha256'])
    require(verification['mode'] == 'created' and verification['zipRoundtripPassed'] is True
            and verification['allSourceBytesStable'] is True and verification['fullStudyComplete'] is True
            and manifest['kind'] == 'short-boost-transfer-research-source-snapshot'
            and manifest['contractSha256'] == verification['contractSha256'] == reg['sha256']
            and manifest['gitHead'] == verification['gitHead']
            and manifest['registeredSourcesVerified'] == verification['registeredSourcesVerified'] == 16
            and binding(manifest['externalBindings']['studyReport']) == binding(report_id)
            and binding(manifest['externalBindings']['summary']) == binding(summary_id),
            'Preceding source archive is not final or binds different artifacts')
    rows = manifest['files']
    require(len(rows) == len({r['archivePath'] for r in rows}) == manifest['sourceFileCount']
            and sum(r['sizeBytes'] for r in rows) == manifest['sourceBytes'] <= prior.MAX_TOTAL_BYTES,
            'Preceding archive inventory/count/budget differs')
    with zipfile.ZipFile(zip_id['path']) as archive:
        expected_names = {r['archivePath'] for r in rows} | {'snapshot-manifest.json'}
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == expected_names, 'Preceding ZIP inventory differs')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
                    'Unsafe preceding ZIP member')
            # The completed transfer archive also preserves the native Windows
            # resource observers used during WSL recovery. These are source
            # evidence, read as ZIP bytes only; do not allow arbitrary scripts.
            monitor_powershell = (p.suffix == '.ps1' and len(p.parts) >= 4
                and p.parts[0] == 'provenance'
                and ((p.parts[1] == 'study' and p.parts[2].startswith('resource-monitor-v'))
                     or p.parts[1].startswith('runtime-recovery-v')))
            require((p.suffix in ('.py', '.mjs', '.md', '.json', '.jsonl', '.txt', '.log', '.lock')
                     or monitor_powershell)
                    and not any(part.startswith('.env') or part in ('.git', 'node_modules', '__pycache__')
                                for part in p.parts), 'Nonresearch preceding ZIP member')
        require(sum(i.file_size for i in archive.infolist()) <= prior.MAX_TOTAL_BYTES+prior.MAX_FILE_BYTES,
                'Preceding ZIP expanded size exceeds source budget')
        require(archive.testzip() is None, 'Preceding ZIP CRC failed')
        require(archive.read('snapshot-manifest.json') == Path(manifest_id['path']).read_bytes(),
                'Embedded preceding manifest differs')
        for row in rows:
            data = archive.read(row['archivePath'])
            require(len(data) == row['sizeBytes'] and digest(data) == row['sha256'], 'Preceding ZIP member changed')
        for name, sha in c['code'].items():
            require(digest(archive.read('repository/analysis/'+name)) == sha, 'Preceding registered source differs')
    return c, {'archive': zip_id, 'manifest': manifest_id, 'verification': verification_id,
               'preregistration': reg_id, 'report': report_id, 'summary': summary_id,
               'contractSha256': reg['sha256'], 'registeredSourcesVerified': 16, 'zipRoundtripPassed': True}


def inventory(study, final_docs=(), source_copy_dirs=()):
    study = Path(study)
    evidence = Evidence()
    reg, reg_id = evidence.read(study/'preregistration.json')
    c = reg['contract']
    require(canonical(c) == reg['sha256'] and c['experiment'] in EXPERIMENTS, 'Invalid follow-up registration')
    report, report_id = evidence.read(study/'report.json')
    summary, summary_id = evidence.read(study/'summary.json')
    tensor = None
    if c['experiment'] == 'short-context-development-v1':
        tensor_ref = summary['audit']['tensorAudit']
        require(Path(tensor_ref['path']).resolve() == (study/'tensor-audit-v1.json').resolve(),
                'Noncanonical context tensor audit')
        tensor, _ = evidence.verified(tensor_ref)
    mode, counts = validate_scope(reg, report, summary, report_id, tensor)
    require(mode != 'keep-rescue' or not (study/'fits').exists(), 'Zero-fit study contains a fits directory')
    previous, historical = verify_reference_archive(c['referenceStudy'], evidence)
    require(all(c['code'][name] == sha for name, sha in previous['code'].items()), 'Frozen inherited source changed')
    require(c['manifestSha256'] == previous['manifestSha256']
            and c['manifest']['sha256'] == c['manifestSha256']
            and c['dinoManifest'] == previous['dinoManifest'], 'Follow-up data identity differs')
    manifest, _ = evidence.verified(c['manifest'])
    evidence.verified(c['dinoManifest'])
    require(len(manifest['exactRows']) == 8 and len(manifest['draftRows']) == 3
            and len(manifest['coverageRows']) == 7
            and sorted({r['sourceGroup'] for r in manifest['exactRows']}) == c['groups'], 'Dataset scope differs')
    for row in manifest['exactRows']+manifest['draftRows']+manifest['coverageRows']:
        require(row['consent']['train'] is True and row['environment'] in ('grass', 'indoor')
                and row['sourceGroup'] not in set(manifest['protectedSourceGroups']) | {private_value('source-group-008')},
                'Forbidden dataset row')
    preflight, _ = evidence.verified(c['preflight'])
    require(preflight['passed'] is True and preflight['code'] == c['code']
            and preflight['manifest'] == c['manifest']
            and preflight['referenceContractSha256'] == c['referenceStudy']['contractSha256']
            and binding(preflight['referenceReport']) == binding(historical['report'])
            and preflight['referenceAudits'] == c['referenceStudy']['independentAudits'], 'Preflight association differs')
    if mode == 'context':
        require(preflight['kind'] == 'context-engineering-preflight-v1'
                and preflight['ownerReuseQualified'] is True and preflight['originalProfileReplayQualified'] is True
                and preflight['dinoManifest'] == c['dinoManifest'] and preflight['cohort'] == c['cohort']
                and preflight['lossArm'] == c['lossArm'], 'Context engineering qualification differs')
        plan_ref = preflight['planFile']
    else:
        require(preflight['kind'] == 'keep-rescue-engineering-preflight-v1'
                and preflight['noOpReplayQualified'] is True and preflight['selectionIsolationQualified'] is True
                and preflight['cohorts'] == c['cohorts']
                and preflight['executionEnvironment'] == c['executionEnvironment']
                and preflight['testExitCode'] == 0, 'Keep-rescue engineering qualification differs')
        plan_ref = preflight['plan']
    plan, _ = evidence.verified(plan_ref)
    require(all(preflight[key] == value for key, value in plan.items()) and plan['code'] == c['code'],
            'Preflight report differs from prospective plan')
    if mode == 'context':
        require(canonical(plan) == preflight['planSha256'], 'Context preflight plan hash differs')
    protocol_id = evidence.identity(c['protocolSnapshot']['path'], c['protocolSnapshot']['sha256'])
    for name in ('summary.md', 'loss-identities.json'):
        evidence.identity(study/name)
    prefix = prior.git_prefix()
    head = prior.git(prefix, 'rev-parse', 'HEAD').decode().strip()
    tracked = set(prior.git(prefix, 'ls-files', '-z').decode().split('\0'))-{''}
    untracked = set(prior.git(prefix, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0'))-{''}
    changed = set(prior.git(prefix, 'diff', '--name-only', '-z', 'HEAD', '--').decode().split('\0'))-{''}
    registered = {'analysis/'+name: sha for name, sha in c['code'].items()}
    candidates = {p for p in tracked | untracked if prior.research_allowed(p)} | set(registered)
    candidates |= {p for p in EXPLICIT_TESTS if (REPO/p).is_file()}
    dependencies = {p for p in tracked if p.startswith('analysis/') and p.endswith('.py')}
    require(not ((changed & dependencies)-candidates), 'Changed dependency outside source allowlist')
    candidates |= dependencies
    final_names = []
    for doc in final_docs:
        path = Path(doc) if Path(doc).is_absolute() else REPO/doc
        require(path.resolve().is_relative_to((REPO/'docs/research').resolve()) and path.suffix == '.md',
                'Final document must be repository docs/research Markdown')
        name = path.resolve().relative_to(REPO.resolve()).as_posix()
        candidates.add(name)
        final_names.append(name)
    sources, origins, rows = {}, {}, []

    def add(path, root, member, expected=None, **extra):
        require(member not in sources, f'Duplicate archive member: {member}')
        data = prior.safe_read(Path(path), Path(root))
        require(expected is None or digest(data) == expected, f'Source changed: {path}')
        sources[member], origins[member] = data, Path(path)
        rows.append({'archivePath': member, 'sourcePath': str(path), 'sha256': digest(data),
                     'sizeBytes': len(data), **extra})

    def metadata(ref, member):
        row = evidence.identity(ref['path'], ref['sha256'])
        # Large preflight reports/audits remain explicitly hash-bound, never truncated.
        if row['sizeBytes'] <= prior.MAX_FILE_BYTES:
            add(row['path'], Path(row['path']).parent, member, row['sha256'])

    for name in sorted(candidates):
        add(REPO/name, REPO, 'repository/'+name, registered.get(name), registered=name in registered,
            gitState='untracked' if name in untracked else 'modified' if name in changed else 'unchanged')
    metadata(reg_id, 'provenance/preregistration.json')
    metadata(protocol_id, 'provenance/protocol/'+Path(protocol_id['path']).name)
    metadata(c['preflight'], 'preflight/report.json')
    metadata(plan_ref, 'preflight/plan.json')
    script_ref = preflight['script']
    require(Path(script_ref['path']).resolve() == (REPO/f'scripts/audit-neural-{mode}-preflight.py').resolve(),
            'Unexpected preflight script path')
    metadata(script_ref, 'preflight/preflight-source.py')
    if mode == 'keep-rescue':
        metadata(preflight['syntheticTestLog'], 'preflight/tests.log')
        for index, test in enumerate(preflight['syntheticTests']):
            require(Path(test['path']).resolve().is_relative_to((REPO/'analysis/tests').resolve()),
                    'Preflight test escapes research tests')
            metadata(test, f'preflight/tests/{index}-'+Path(test['path']).name)
    for key in ('manifest', 'verification'):
        metadata(historical[key], 'reference-source-snapshot/'+Path(historical[key]['path']).name)
    for name in ('sources', 'registered-sources'):
        if (study/name).exists():
            source_copy_dirs = (*source_copy_dirs, study/name)
    preflight_root = Path(c['preflight']['path']).parent
    if (preflight_root/'sources').exists():
        source_copy_dirs = (*source_copy_dirs, preflight_root/'sources')
    for folder in source_copy_dirs:
        require(Path(folder).is_dir() and not Path(folder).is_symlink(), 'Invalid registered source-copy directory')
    copy_dirs = sorted({Path(p).resolve() for p in source_copy_dirs})
    for index, folder in enumerate(copy_dirs):
        require(folder.is_dir() and not folder.is_symlink(), 'Invalid registered source-copy directory')
        files = list(folder.iterdir())
        require({p.name for p in files} == set(c['code']) and len(files) == len(c['code']),
                'Registered source copies must contain exactly the registered sources')
        for path in sorted(files):
            add(path, folder, f'registered-source-copies/{index}/'+path.name, c['code'][path.name])
    if mode == 'context' and (REPO/CPU_DIAGNOSIS).exists():
        add(REPO/CPU_DIAGNOSIS, REPO, 'provenance/runtime/diagnosis.json')
    operational = context_operational_evidence(study, evidence, add) if mode == 'context' else []
    total = sum(map(len, sources.values()))
    require(total <= prior.MAX_TOTAL_BYTES, 'Source snapshot exceeds 32 MiB budget')
    output = {'schemaVersion': 1, 'kind': 'neural-followup-research-source-snapshot',
              'experiment': c['experiment'], 'createdAt': datetime.now(timezone.utc).isoformat(),
              'contractSha256': reg['sha256'], 'gitHead': head, 'repositoryPath': str(REPO),
              'registeredSourcesVerified': len(registered), 'completionCounts': counts,
              'completedStudyReport': report_id, 'passedSummary': summary_id,
              'referenceArchive': historical, 'finalDocuments': final_names,
              'registeredSourceCopyDirectories': [str(p) for p in copy_dirs],
              'externalBindings': list(evidence.files.values()), 'files': rows,
              'sourceFileCount': len(rows), 'sourceBytes': total,
              'allowlistPatterns': list(prior.PATTERNS), 'explicitRepositoryFiles': sorted(EXPLICIT_TESTS),
              'changedOutsideAllowlistExcluded': sorted(changed-candidates),
              'untrackedOutsideAllowlistExcluded': sorted(untracked-candidates),
              'protectedTestOpened': False, 'productionPromotionAllowed': False,
              'scope': 'Research source/docs and bounded provenance only. No model weights, NPZs, feature arrays, videos, secrets or environment directories.',
              'limits': 'This checks archive integrity and completed audit bindings, not metrics, model outputs or fresh fitting. Large reports and the preceding ZIP remain external immutable hash bindings.',
              'restore': 'Checkout gitHead and overlay repository/. Retain separately named preflight and registered source copies. The verified preceding transfer ZIP retains its own historical archive bindings.'}
    if mode == 'context':
        output['contextOperationalEvidence'] = operational
    return output, sources, origins, prefix, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--final-doc', type=Path, action='append', default=[],
                        help='Final docs/research Markdown file; repeatable and required for --create')
    parser.add_argument('--source-copy-dir', type=Path, action='append', default=[],
                        help='Additional directory containing exactly the registered source copies')
    parser.add_argument('--create', action='store_true', help='Only after completed audits and final docs; immutable output')
    args = parser.parse_args(argv)
    output = args.study/'source-snapshot'
    require(not args.create or args.final_doc, '--create requires an explicitly final --final-doc')
    require(not args.create or not output.exists(), f'Immutable output already exists: {output}')
    manifest, sources, origins, prefix, evidence = inventory(args.study, args.final_doc, args.source_copy_dir)
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    zip_bytes = prior.zip_roundtrip(sources, manifest_bytes)
    for member, path in origins.items():
        require(prior.safe_read(path, path.parent) == sources[member], f'Source changed during snapshot: {path}')
    evidence.recheck()
    require(prior.git(prefix, 'rev-parse', 'HEAD').decode().strip() == manifest['gitHead'], 'Git HEAD changed')
    verification = {'mode': 'created' if args.create else 'validation-only-no-writes',
                    'contractSha256': manifest['contractSha256'], 'gitHead': manifest['gitHead'],
                    'registeredSourcesVerified': manifest['registeredSourcesVerified'],
                    'fullStudyComplete': True, 'passedSummaryVerified': True,
                    'finalDocumentsSpecified': bool(args.final_doc), 'completionCounts': manifest['completionCounts'],
                    'sourceFileCount': len(sources), 'sourceBytes': manifest['sourceBytes'], 'zipBytes': len(zip_bytes),
                    'zipSha256': digest(zip_bytes), 'manifestSha256': digest(manifest_bytes),
                    'allSourceBytesStable': True, 'zipRoundtripPassed': True, 'output': str(output)}
    if args.create:
        output.mkdir(parents=False, exist_ok=False)
        for name, data in (('research-source.zip', zip_bytes), ('snapshot-manifest.json', manifest_bytes),
                           ('verification.json', (json.dumps(verification, indent=2)+'\n').encode())):
            with (output/name).open('xb') as handle:
                handle.write(data)
        require((output/'research-source.zip').read_bytes() == zip_bytes
                and (output/'snapshot-manifest.json').read_bytes() == manifest_bytes, 'Saved archive readback differs')
    print(json.dumps(verification, indent=2))
    return verification


if __name__ == '__main__':
    main()
