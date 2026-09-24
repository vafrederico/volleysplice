#!/usr/bin/env python3
"""Capture a completed, independently audited rally-review study in bounded ZIP.

Only --create writes an archive, after exact final-document SHA authorization.
Historical videos/features/checkpoints remain explicit provenance references.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0092'))
CONTRACT = '7386abe4befe96ba4c9607454f5f6936b312dd8459c5dae1a453d4cadc93440d'
HELPER = REPO/'scripts/snapshot-neural-production-combinations.py'
HELPER_SHA = 'd2fda0db04c8573f75ded3c8be059e8cada0b5b1ad1499355e1ce6ac00e9a41c'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('Qualified archive helper changed')
SPEC = importlib.util.spec_from_file_location('rally_archive_qualified_helpers', HELPER)
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)
require = H.require
REGISTERED = {
    'analysis/neural_rally_review_proposals.py', 'analysis/neural_rally_identity_metrics.py',
    'scripts/run-neural-rally-review-proposals.py', 'scripts/prepare-neural-rally-review-input.py',
    'scripts/audit-neural-rally-review-proposals.py', 'scripts/audit-neural-rally-review-candidates.py',
    'analysis/tests/test_neural_rally_review_proposals.py', 'analysis/tests/test_neural_rally_identity_metrics.py',
    'analysis/tests/test_neural_rally_review_audit.py', 'analysis/tests/test_neural_rally_review_candidates_audit.py',
    'analysis/tests/test_neural_rally_review_input.py', 'analysis/neural_human_review.py',
    'analysis/tests/test_neural_rally_review_runner.py',
    'analysis/neural_production_combinations.py', 'scripts/audit-neural-combination-accounting.py',
    'analysis/crop_evaluation.py', 'analysis/schema.py',
}
EXTRA = {
    'analysis/__init__.py', 'analysis/version.py', 'analysis/requirements.txt',
    'scripts/snapshot-neural-production-combinations.py',
    'analysis/tests/test_neural_production_combinations_snapshot.py',
    'scripts/snapshot-neural-rally-review-proposals.py',
    'scripts/replay-neural-rally-review-snapshot.py',
    'analysis/tests/test_neural_rally_review_snapshot.py',
    'docs/model-ranking-metric.md', 'docs/research/README.md',
    'docs/research/neural-rally-review-proposals-protocol-2026-09-19.md',
    'docs/research/neural-rally-review-proposals-results-2026-09-19.md',
    'docs/research/neural-rally-review-replay-2026-09-19.md',
    'scripts/summarize-neural-rally-review-proposals.py',
    'scripts/audit-neural-rally-review-summary.py',
    'analysis/tests/test_neural_rally_review_summary_audit.py',
    'analysis/tests/test_neural_rally_review_summary.py',
}


def metadata_files(root):
    return {p.name for p in root.iterdir() if p.is_file() and p.suffix in {'.json', '.md', '.txt', '.log'}}


def completion_gate(registration, report):
    c = registration['contract']
    require(H.canonical(c) == registration['sha256'] == CONTRACT, 'Wrong study registration')
    require(set(c['sources']) == REGISTERED and c['models'] == ['compact_boost', 'dino_global', 'dino_boost']
            and c['seeds'] == [3407, 1729, 20260918], 'Registered source/model scope differs')
    require(c['configurationSeedRuns'] == 108 and c['outcomeCells'] == 432 and len(c['configurations']) == 36
            and c['budgetFractions'] == [.05, .10, .20, .40]
            and c['primaryMetric'] == 'F1_padP_coreR' and c['targetPaddingSeconds'] == 2
            and c['joinGapSeconds'] == 3 and c['paddingCases'] == [0, 1, 2, 3]
            and c['editMarginSeconds'] == 2 and c['viewContextSeconds'] == 2, 'Registered matrix or metric differs')
    require(c['goldUsedForProposalsOrRanking'] is False and c['hypotheticalHumanOutcomes'] is True,
            'Gold access scope differs')
    for value in (c, report):
        require(value['newDetectorFits'] == value['learnedReviewFits'] == 0
                and value['protectedTestOpened'] is value['productionChanged'] is False,
                'Forbidden detector fitting, protected-test or production operation')
    require(report['kind'] == c['kind'] == 'fixed-budget-rally-review-proposals-v1'
            and report['status'] == 'completed-fixed-rally-review-proposals'
            and report['contractSha256'] == CONTRACT and report['configurationSeedRuns'] == 108
            and report['outcomeCells'] == 432 and report['candidatePlansAudited'] == 864
            and report['editedRecordingsAudited'] == 3456 and len(report['results']) == 108,
            'Full matrix and independent audits are incomplete')


def evaluation_gate(value):
    require(len(value['durationMetrics']) == 4 and value['durationAudit'] == {'passed': True, 'scopeCount': 13, 'paddingCases': 4},
            'Independent export accounting coverage differs')
    audit = value['identityAudit']
    require(audit['passed'] is True and audit['recordingsAudited'] == 8 and audit['sourceGroupsAudited'] == 4
            and audit['observedBoundaryFlagsAudited'] is True, 'Independent identity accounting is incomplete')


def result_inventory(inv, root, registration, report):
    configs = {x['id']: x for x in registration['contract']['configurations']}
    expected = {(name, seed) for name in configs for seed in registration['contract']['seeds']}
    seen, filenames = set(), set()
    for ref in report['results']:
        path = Path(ref['path'])
        require(path.parent == root/'results' and path.suffix == '.json' and path.name not in filenames,
                'Unexpected or duplicate result path')
        filenames.add(path.name)
        inv.add(path, root, 'study/results/'+path.name, ref)
        value = inv.read(path, ref)
        key = value['configuration']['id'], value['seed']
        require(key in expected and key not in seen and value['configuration'] == configs[key[0]]
                and value['contractSha256'] == CONTRACT, 'Result configuration partition differs')
        seen.add(key)
        require(len(value['plans']) == 8 and len(value['outcomes']) == 4, 'Incomplete plan/outcome count')
        for row in value['plans']:
            require(row['candidateAudit']['passed'] is True and row['candidateAudit']['goldFieldsRead'] is False
                    and row['candidateAudit']['candidateCount'] == len(row['candidates'])
                    and row['queueAudit']['passed'] is True and row['queueAudit']['budgetsAudited'] == 4
                    and row['queueAudit']['labelDataRead'] is False and row['queueAudit']['nestedSelections'] is True,
                    'Candidate/queue independent reconstruction is incomplete')
        evaluation_gate(value['automatic'])
        require([x['budgetFraction'] for x in value['outcomes']] == registration['contract']['budgetFractions'],
                'Outcome budget partition differs')
        for row in value['outcomes']:
            evaluation_gate(row)
            require(len(row['recordings']) == 8 and all(r['editorAudit']['passed'] is True
                    and r['editorAudit']['goldMarkersImportedOutsidePermission'] == 0 for r in row['recordings']),
                    'Independent local editor reconstruction is incomplete')
    require(seen == expected, 'Missing result cells')
    inv.close_directory(root/'results', filenames)


def binding_refs(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        for nested in value.values():
            yield from binding_refs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from binding_refs(nested)


def add_audit_sources(inv, repo, values):
    for value in values:
        for ref in binding_refs(value):
            path = Path(ref['path'])
            if path.is_relative_to(repo):
                inv.add(path, repo, 'repository/'+path.relative_to(repo).as_posix(), ref)


def audit_gate(registration, summary, audit, interpretation, identities, final_doc):
    require(summary['kind'] == 'fixed-rally-review-proposals-summary-v1'
            and summary['contractSha256'] == registration['sha256']
            and len(summary['arms']) == 144 and len(summary['automaticBaselines']) == 4
            and summary['targetPaddingSeconds'] == 2 and summary['joinGapSeconds'] == 3,
            'Complete target-padding summary required')
    for value in (audit, interpretation):
        require(value['passed'] is True and value['contractSha256'] == registration['sha256'],
                'Completed summary and final independent audits required')
    require(audit.get('failures', []) == [] and interpretation.get('failures', []) == [], 'Independent audit has failures')
    require(interpretation['kind'] == 'independent-rally-review-interpretation-audit-v2'
            and interpretation['generatedMarkdownDataRowsVerified'] == 1748
            and bool(interpretation['manualNarrativeReview']), 'Final prose and complete tables were not reviewed')
    require(audit['kind'] == 'independent-rally-review-summary-audit-v1'
            and all(audit[key] == value for key, value in {
                'configurationSeedRunsAudited': 108, 'outcomeCellsAudited': 432, 'candidatePlansAudited': 864,
                'editedRecordingsAudited': 3456, 'automaticBaselinesAudited': 4, 'armsAudited': 144,
                'sourceGroupArmSummariesAudited': 576, 'paddingCasesAudited': 4, 'rankingListsAudited': 8,
                'guardrailScreensAudited': 720}.items())
            and audit['numericalReconstructionAuditsRequiredPassed'] is True and audit['seedStatisticsReconstructed'] is True,
            'Independent summary audit coverage differs')
    for name, value in [('summary', summary), ('summaryAudit', audit), ('interpretation', interpretation)]:
        for ref in binding_refs(value):
            require(ref['path'] in identities and H.matches(identities[ref['path']], ref),
                    name+' source/output binding differs')
    doc_refs = [x for x in binding_refs(interpretation) if x['path'] == str(final_doc)]
    require(len(doc_refs) >= 1 and all(H.matches(identities[str(final_doc)], ref) for ref in doc_refs),
            'Final interpretation audit does not bind the exact final document')


def prior_inventory(inv, root, data):
    experiments = root.parent
    for key in ('priorProbabilityInput', 'previousRegistration', 'exactManifest'):
        ref = data[key]
        inv.add(ref['path'], experiments, 'provenance/'+key+'.json', ref)
    old_root = Path(data['priorProbabilityInput']['path']).parent
    previous = inv.read(old_root/'registration.json')
    require(H.canonical(previous['contract']) == previous['sha256']
            and previous['sha256'] == '7f47932b9a18e030840449b7e4eceec489ac474d8eb5fa9dee13ee0a32a71b56',
            'Previous review registration differs')
    inv.add(old_root/'registration.json', experiments, 'provenance/previous-human-review-registration.json')
    for relative, ref in previous['contract']['sources'].items():
        inv.bind(ref['path'], ref)
        if relative == 'analysis/metrics.py':
            inv.add(ref['path'], REPO, 'repository/'+relative, ref)
    for key in ('previousNeuralInput', 'previousProductionInput', 'independentTensorAudit'):
        inv.external_binding(data[key], 'Verified original input provenance; normalized replacement is included')
    for row in data['sourceFits']:
        for key in ('completed', 'weights', 'probabilities', 'sourceResult'):
            inv.external_binding(row[key], 'Original selected '+key+' provenance only; normalized scores/events included', verify=False)
    for row in data['records']:
        for key in ('featureCache', 'labelSource'):
            inv.external_binding(row[key], 'Original '+key+' provenance only; normalized timestamps/gold included', verify=False)
    return previous['sha256']


def add_binary(inv, path, base, name, expected):
    path = H.safe_path(path, base)
    require(path.suffix in {'.png', '.svg'} and path.stat().st_size <= 5*1024*1024, 'Unexpected figure type or size')
    row = inv.bind(path, expected)
    data = path.read_bytes()
    require(H.sha(data) == row['sha256'], 'Figure changed during capture')
    require(name not in inv.entries, 'Duplicate figure archive path')
    inv.entries[H.safe_name(name)] = {'identity': row, 'data': data}
    inv.total += len(data)
    require(inv.total <= H.MAX_TOTAL, 'Archive exceeds bounded size')


def figures_inventory(inv, root):
    folder = root/'figures'
    if not folder.exists():
        return None
    manifest = inv.read(folder/'manifest.json')
    require(manifest['kind'] == 'rally-review-quality-vs-workload-figures-v1'
            and manifest['contractSha256'] == CONTRACT and manifest['assetCount'] == len(manifest['assets']) == 4,
            'Figure manifest incomplete')
    names = {'manifest.json'}
    for ref in manifest['assets']:
        path = Path(ref['path'])
        require(path.parent == folder and path.name not in names, 'Unexpected figure path')
        names.add(path.name)
        add_binary(inv, path, folder, 'study/figures/'+path.name, ref)
    inv.add(folder/'manifest.json', root, 'study/figures/manifest.json')
    inv.close_directory(folder, names)
    add_audit_sources(inv, REPO, [manifest])
    for ref in binding_refs(manifest):
        if ref['path'] in inv.bound:
            require(H.matches(inv.bound[ref['path']], ref), 'Figure source/result binding differs')
    return inv.bound[str(folder/'manifest.json')]


def failure_inventory(inv, root, registration):
    """A closed, outcome-free initial operational attempt is required for v2."""
    ref = registration['contract']['previousFailedAttempt']
    value = inv.read(ref['path'], ref)
    folder = Path(ref['path']).parent
    require(value['status'] == 'closed-failed-before-result-publication'
            and value['completedResultFiles'] == 0 and value['outcomeTablesRead'] is False
            and value['numericalPolicyChanges'] is False and value['nextAttempt'] == str(root),
            'Initial operational attempt is not closed without outcome selection')
    require(not (folder/'report.json').exists() and not list((folder/'results').glob('*.json')),
            'Initial attempt unexpectedly contains quality result files')
    previous = inv.read(folder/'registration.json')
    require(H.canonical(previous['contract']) == previous['sha256'] == value['contractSha256']
            == '51db181deb9f5339bcabd97be92848d180cea47b6914b1bebb0dc26858665aef',
            'Initial attempt registration differs')
    original = previous['contract']['sources']
    require(len(original) == len(value['preservedSources']) == 16
            and {x['relativePath'] for x in value['preservedSources']} == set(original),
            'Initial source preservation is incomplete')
    seen_files = set()
    for row in value['preservedSources']:
        path, relative = Path(row['snapshotPath']), row['relativePath']
        require(path == folder/'initial-source'/relative and row['sha256'] == original[relative]['sha256'],
                'Initial source copy differs from frozen source')
        inv.add(path, folder, 'provenance/initial-attempt/initial-source/'+relative,
                {'path': str(path), 'sha256': row['sha256'], 'sizeBytes': original[relative]['sizeBytes']})
        seen_files.add(path)
        if relative != 'scripts/run-neural-rally-review-proposals.py':
            require(registration['contract']['sources'][relative]['sha256'] == row['sha256'],
                    'Numerical policy changed between operational attempts')
    require(set(p for p in (folder/'initial-source').rglob('*') if p.is_file()) == seen_files,
            'Unexpected initial source-copy contents')
    for directory in [folder/'initial-source', *(p for p in (folder/'initial-source').rglob('*') if p.is_dir())]:
        inv.close_directory(directory, {p.name for p in directory.iterdir()})
    for name in metadata_files(folder):
        inv.add(folder/name, folder, 'provenance/initial-attempt/'+name)
    return {'role': 'closed operational failure before result publication', 'closure': ref,
            'sourceFilesPreserved': 16, 'numericalPolicyChanges': False}


def prose_history_inventory(inv, root, interpretation_name):
    receipt_path = root/'interpretation-supersession-v1.json'
    value = inv.read(receipt_path)
    require(value['passed'] is True and value['kind'] == 'rally-review-prose-supersession-v1'
            and value['status'] == 'closed-prose-only-correction' and value['contractSha256'] == CONTRACT
            and value['provisionalAuditAcceptedAsFinal'] is False and value['registeredSourcesUnchanged'] == 17
            and value['numericalSummaryUnchanged'] is value['numericalReportUnchanged'] is True,
            'Provisional prose audit supersession is not closed')
    require(value['finalAudit']['path'] == str(root/interpretation_name), 'Different final prose audit selected')
    for key in ('provisionalAudit', 'finalAudit', 'report', 'summary', 'registration'):
        inv.bind(value[key]['path'], value[key])
    for key in ('provisionalAuditSource', 'finalAuditSource'):
        ref = value[key]
        path = Path(ref['path'])
        inv.add(path, REPO, 'repository/'+path.relative_to(REPO).as_posix(), ref)
    original = inv.read(value['provisionalAudit']['path'], value['provisionalAudit'])
    folder, seen = root/'prose-initial', set()
    require(len(value['preservedBindings']) == 2
            and {row['role'] for row in value['preservedBindings']} == {'document', 'readme'}, 'Incomplete initial prose copies')
    for row in value['preservedBindings']:
        before, copy, after = row['originalBinding'], row['preservedCopy'], row['finalBinding']
        require(before == original[row['role']] and before['sha256'] == copy['sha256']
                and before['sizeBytes'] == copy['sizeBytes'], 'Initial prose differs from provisional audit binding')
        path = Path(copy['path'])
        require(path.is_relative_to(folder) and path.suffix == '.md', 'Unexpected initial prose copy')
        inv.add(path, root, 'study/'+path.relative_to(root).as_posix(), copy)
        inv.bind(after['path'], after)
        seen.add(path)
    require({p for p in folder.rglob('*') if p.is_file()} == seen
            and sum(p.stat().st_size for p in seen) < 5*1024*1024, 'Initial prose directory is not closed/bounded')
    for directory in [folder, *(p for p in folder.rglob('*') if p.is_dir())]:
        inv.close_directory(directory, {p.name for p in directory.iterdir()})
    return {'role': 'superseded provisional prose audit; final v2 accepted',
            'closure': inv.bound[str(receipt_path)], 'preservedProseFiles': 2}


def build_inventory(root, repo, document_sha, interpretation_name):
    inv = H.Inventory()
    reg = inv.read(root/'registration.json')
    report = inv.read(root/'report.json')
    summary = inv.read(root/'summary.json')
    audit = inv.read(root/'summary-audit-v1.json')
    require(Path(interpretation_name).name == interpretation_name and interpretation_name.startswith('interpretation-audit-')
            and interpretation_name.endswith('.json'), 'Explicit final interpretation receipt filename required')
    interpretation_path = root/interpretation_name
    interpretation = inv.read(interpretation_path)
    completion_gate(reg, report)
    for relative, ref in reg['contract']['sources'].items():
        require(ref['path'] == str(repo/relative), 'Registered source path differs')
        inv.add(repo/relative, repo, 'repository/'+relative, ref)
    for relative in sorted(EXTRA):
        inv.add(repo/relative, repo, 'repository/'+relative)
    add_audit_sources(inv, repo, [summary, audit, interpretation])
    for name in sorted(metadata_files(root)):
        inv.add(root/name, root, 'study/'+name)
    document = repo/'docs/research/neural-rally-review-proposals-results-2026-09-19.md'
    require(len(document_sha) == 64 and inv.bound[str(document)]['sha256'] == document_sha,
            'Explicit final document hash differs')
    for name in ('input', 'probabilities', 'protocol', 'goldSemantics', 'qualification'):
        inv.bind(reg['contract'][name]['path'], reg['contract'][name])
    data = inv.read(reg['contract']['input']['path'], reg['contract']['input'])
    require(data['headNames'] == ['live', 'serve', 'end', 'keep'] and data['roundTripExact'] is True
            and data['selectedOuterFitCount'] == 36 and len(data['entries']) == 72 and len(data['records']) == 8,
            'Normalized four-head input differs')
    require(0 < data['npz']['sizeBytes'] < 8*1024*1024, 'Normalized input archive exceeds bounded size')
    inv.add(data['npz']['path'], root.parent, 'study/probabilities.npz', data['npz'], allow_npz=True)
    require(inv.read(reg['contract']['qualification']['path'])['passed'] is True, 'Qualification failed')
    previous = prior_inventory(inv, root, data)
    result_inventory(inv, root, reg, report)
    failure = failure_inventory(inv, root, reg)
    figures = figures_inventory(inv, root)
    prose_history = prose_history_inventory(inv, root, interpretation_name)
    audit_gate(reg, summary, audit, interpretation, inv.bound, document)
    return inv, reg, {'finalDocument': inv.bound[str(document)],
        'independentSummaryAudit': inv.bound[str(root/'summary-audit-v1.json')],
        'independentInterpretationAudit': inv.bound[str(interpretation_path)],
        'previousHumanReviewContractSha256': previous, 'initialOperationalFailure': failure,
        'provisionalProseAuditSupersession': prose_history,
        'figuresManifest': figures,
        'omissions': 'Videos, feature arrays, original checkpoints and previous full study ZIPs are omitted. All exact normalized scores, timestamps, events, gold and numerical replay sources are included.'}


def capture(root, repo, inv, registration, details, create):
    import numpy as np
    target = root/'source-snapshot'
    require(not target.exists(), 'Immutable snapshot already exists')
    names, git = metadata_files(root), H.native_git(repo)
    manifest = {'kind': 'rally-review-proposals-reproducible-snapshot-v1',
        'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
        'registeredSourcesVerified': 17, 'qualifiedHelperSha256': HELPER_SHA,
        'nativeGit': git, 'protectedTestOpened': False, 'productionChanged': False,
        'environment': {'pythonVersion': platform.python_version(), 'pythonImplementation': platform.python_implementation(),
                        'numpyVersion': np.__version__, 'platform': platform.platform()},
        'files': [{'archivePath': name, **entry['identity']} for name, entry in sorted(inv.entries.items())],
        'externalBindings': inv.external, 'uncompressedSourceBytes': inv.total, **details}
    content = {name: row['data'] for name, row in inv.entries.items()}
    content['snapshot-manifest.json'] = H.encoded(manifest)
    inv.recheck()
    require(H.native_git(repo) == git and metadata_files(root) == names, 'Source/evidence changed during validation')
    receipt = {'kind': 'rally-review-proposals-snapshot-verification-v1',
        'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
        'mode': 'created' if create else 'validated', 'passed': True, 'registeredSourcesVerified': 17,
        'fileCount': len(inv.entries), 'sourceBytes': inv.total, 'manifestSha256': H.sha(content['snapshot-manifest.json']),
        'fullStudyComplete': True, 'allSourceBytesStable': True, 'nativeGitHead': git['head']}
    if create:
        target.mkdir()
        (target/'snapshot-manifest.json').write_bytes(content['snapshot-manifest.json'])
        archive = target/'research-source.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for name, data in sorted(content.items()):
                item = zipfile.ZipInfo(name, date_time=(2026, 9, 19, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                stream.writestr(item, data)
        H.zip_roundtrip(archive, content, root)
        with tempfile.TemporaryDirectory(prefix='rally-archive-import-', dir=root) as tmp:
            restored = Path(tmp)
            with zipfile.ZipFile(archive) as stream:
                for name in content:
                    destination = restored/name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(stream.read(name))
            portable = restored/'repository/scripts/replay-neural-rally-review-snapshot.py'
            result = subprocess.run([sys.executable, '-B', str(portable), '--snapshot', str(restored)],
                                    check=True, capture_output=True, text=True)
            portable_receipt = json.loads(result.stdout)
            require(portable_receipt['passed'] is True and portable_receipt['importsResolvedFromArchive'] is True,
                    'Portable replay import/input verification failed')
        inv.recheck()
        require(H.native_git(repo) == git and metadata_files(root) == names, 'Source/evidence changed during restore')
        receipt.update(zipSha256=H.identity(archive)['sha256'], zipSizeBytes=archive.stat().st_size,
                       zipRoundtripPassed=True, restoredBinaryEntries=len(content), portableImportVerification=portable_receipt)
        with (target/'verification.json').open('xb') as stream:
            stream.write(H.encoded(receipt))
    print(json.dumps(receipt, indent=2))
    return receipt


def main(root, repo, final_document_sha256, interpretation_audit_name, create=False):
    require(root.is_dir() and not root.is_symlink() and not (root/'source-snapshot').exists(), 'Invalid existing archive root')
    before = metadata_files(root)
    inventory, registration, details = build_inventory(root, repo, final_document_sha256, interpretation_audit_name)
    require(metadata_files(root) == before, 'Metadata changed during capture')
    return capture(root, repo, inventory, registration, details, create)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--final-document-sha256', required=True)
    parser.add_argument('--interpretation-audit-name', required=True,
                        help='Explicit final reviewed receipt filename; provisional receipts must not be selected')
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    main(args.root, args.repo, args.final_document_sha256, args.interpretation_audit_name, args.create)
