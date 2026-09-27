#!/usr/bin/env python3
"""Bounded reproducible archive for the completed perfect-human review study.

No model or review-recipe module is imported. The independently qualified path,
inventory, native-Git and exact binary ZIP restore helpers are hash-pinned below.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0091'))
CONTRACT = '7f47932b9a18e030840449b7e4eceec489ac474d8eb5fa9dee13ee0a32a71b56'
HELPER = REPO/'scripts/snapshot-neural-production-combinations.py'
HELPER_SHA = 'd2fda0db04c8573f75ded3c8be059e8cada0b5b1ad1499355e1ce6ac00e9a41c'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('Qualified archive helper changed')
SPEC = importlib.util.spec_from_file_location('qualified_combination_archive', HELPER)
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)
require, sha, canonical, encoded = H.require, H.sha, H.canonical, H.encoded
MODELS = ['compact_boost', 'compact_keep', 'dino_global', 'dino_boost', 'dino_keep']
SEEDS = [3407, 1729, 20260918]
REGISTERED = {
    'analysis/neural_human_review.py', 'scripts/run-neural-perfect-human-review.py',
    'scripts/audit-neural-human-review.py', 'scripts/prepare-neural-human-review-probabilities.py',
    'analysis/tests/test_neural_human_review.py', 'analysis/tests/test_neural_human_review_audit.py',
    'analysis/tests/test_neural_human_review_probability_input.py',
    'analysis/neural_production_combinations.py', 'scripts/audit-neural-combination-accounting.py',
    'analysis/neural_evaluation.py', 'analysis/crop_evaluation.py', 'analysis/schema.py', 'analysis/metrics.py',
}
EXTRA_SOURCE = {
    'scripts/summarize-neural-perfect-human-review.py',
    'scripts/plot-neural-perfect-human-review.py',
    'scripts/audit-neural-human-review-summary.py',
    'scripts/audit-neural-human-review-document.py',
    'analysis/tests/test_neural_human_review_summary_audit.py',
    'scripts/snapshot-neural-perfect-human-review.py',
    'analysis/tests/test_neural_perfect_human_review_snapshot.py',
    'scripts/snapshot-neural-production-combinations.py',
    'analysis/tests/test_neural_production_combinations_snapshot.py',
    'analysis/tests/test_neural_production_combinations.py',
    'analysis/tests/test_neural_combination_accounting.py',
    'analysis/__init__.py', 'analysis/requirements.txt', 'docs/model-ranking-metric.md',
    'docs/research/neural-perfect-human-review-protocol-2026-09-19.md',
    'docs/research/neural-perfect-human-review-results-2026-09-19.md', 'docs/research/README.md',
}


def metadata_files(root):
    """Only top-level metadata; explicitly handled directories are closed later."""
    return {p.name for p in root.iterdir()
            if p.is_file() and p.suffix in {'.json', '.md', '.txt', '.log'}}


def external(inv, ref, base, role, verify=True):
    p = Path(ref['path'])
    require(p.is_absolute() and p.resolve().is_relative_to(base.resolve())
            and not any(x.is_symlink() for x in [p, *p.parents]),
            'External reference escapes the study provenance roots')
    require(p.suffix in {'.json', '.npz', '.zip'}, 'Unexpected provenance reference type')
    inv.external_binding(ref, role, verify=verify)


def probability_gate(value):
    require(value['kind'] == 'held-source-group-human-review-probabilities-v1'
            and value['models'] == MODELS and value['seeds'] == SEEDS
            and value['recordingCount'] == 8 and value['probabilityArrayCount'] == 120
            and value['selectedOuterFitCount'] == 60 and value['roundTripExact'] is True
            and value['probabilityDtype'] == 'float32' and value['timesDtype'] == 'float64'
            and value['retainedHeadIndex'] == 0,
            'Probability scope or normalization differs')
    require(all(value[k] is False for k in ('protectedTestOpened', 'trainingPerformed',
                'featureExtractionPerformed', 'goldUsedForQueueSelection')),
            'Forbidden probability provenance operation')
    require(len(value['records']) == 8 and len(value['sourceFits']) == 60
            and len(value['entries']) == 120 and len(value['sourceResults']) == 15,
            'Incomplete normalized probability provenance')


def probability_inventory(inv, root, repo, expected):
    value = inv.read(root/'probability-input.json', expected)
    probability_gate(value)
    ref = value['npz']
    require(ref['path'] == str(root/'probabilities.npz') and 0 < ref['sizeBytes'] < 4*1024*1024,
            'Normalized probability archive is not the bounded expected file')
    inv.add(root/'probabilities.npz', root, 'study/probabilities.npz', ref, allow_npz=True)
    source = value['sourceScript']
    require(source['path'] == str(repo/'scripts/prepare-neural-human-review-probabilities.py'),
            'Probability adapter source path differs')
    inv.add(source['path'], repo, 'repository/scripts/prepare-neural-human-review-probabilities.py', source)
    for name in ('previousRegistration', 'previousNeuralInput', 'exactManifest'):
        ref = value[name]
        inv.add(ref['path'], root.parent, 'provenance/probability/'+name+'.json', ref)
    for name, ref in value['sourceReports'].items():
        external(inv, ref, root.parent, 'preceding neural report '+name)
    external(inv, value['independentTensorAudit'], root.parent, 'preceding independent tensor audit')
    for ref in value['sourceResults']:
        external(inv, ref, root.parent, 'preceding selected neural result '+ref['model']+'/'+str(ref['seed']))
    for row in value['sourceFits']:
        external(inv, row['completed'], root.parent, 'selected outer fit completion '+row['fitKey'])
        for key in ('weights', 'probabilities'):
            external(inv, row[key], root.parent,
                     'original '+key+'; normalized-input provenance only, original arrays omitted', verify=False)
    for row in value['records']:
        external(inv, row['featureCache'], root.parent.parent,
                 'exact timestamp feature source; normalized-input provenance only, array omitted', verify=False)
    return value


def report_gate(registration, report):
    c = registration['contract']
    require(canonical(c) == registration['sha256'] == CONTRACT
            and c['kind'] == 'fixed-perfect-human-review-development-v1'
            and set(c['sources']) == REGISTERED, 'Registered contract or source set differs')
    require(c['models'] == MODELS and c['seeds'] == SEEDS
            and c['policyCount'] == 120 and c['resultCells'] == 360
            and c['legacyPolicies'] == 30 and c['legacyCells'] == 90
            and c['recordings'] == 8 and c['rallies'] == 322
            and c['primaryMetric'] == 'F1_padP_coreR' and c['targetPaddingSeconds'] == 2
            and c['joinGapSeconds'] == 3 and c['paddingCases'] == [0, 1, 2, 3],
            'Registered policy, scope or metric differs')
    require(all(c[k] is False for k in ('protectedTestOpened', 'productionChanged',
                'productionPromotionAllowed', 'candidateFlagsUseGold'))
            and c['newFits'] == 0 and c['hypotheticalHumanDecisionsUseGold'] is True,
            'Forbidden registration scope')
    require(report['kind'] == c['kind'] and report['contractSha256'] == CONTRACT
            and report['status'] == 'completed-perfect-human-review'
            and report['policyCount'] == 120 and report['resultCells'] == 360
            and len(report['results']) == 360 and len(report['legacyResults']) == 90
            and report['independentCandidatePlans'] == 2880
            and report['independentRecordingOutcomes'] == 2880
            and report['durationScopesPerOutcome'] == 13 and report['paddingCasesPerScope'] == 4,
            'Incomplete independent construction/accounting coverage')
    require(report['protectedTestOpened'] is False and report['productionChanged'] is False
            and report['newFits'] == 0 and report['hypotheticalHumanOutcomes'] is True,
            'Forbidden report scope')


def result_inventory(inv, root, registration, report):
    c = registration['contract']
    configs = {x['id']: x for x in c['configurations']}
    expected_cells = {(name, seed) for name in configs for seed in SEEDS}
    require(len(configs) == 120 and len(expected_cells) == 360, 'Duplicate registered policies')
    seen = set()
    legacy_seen = set()
    for field, folder in [('results', 'results'), ('legacyResults', 'legacy')]:
        filenames = set()
        for ref in report[field]:
            p = Path(ref['path'])
            require(p.parent == root/folder and p.suffix == '.json' and p.name not in filenames,
                    'Unexpected or duplicate result path')
            filenames.add(p.name)
            inv.add(p, root, 'study/'+p.relative_to(root).as_posix(), ref)
            value = inv.read(p, ref)
            if field == 'results':
                key = value['configuration']['id'], value['seed']
                require(key in expected_cells and key not in seen
                        and value['configuration'] == configs[key[0]]
                        and value['contractSha256'] == CONTRACT,
                        'Result configuration/seed partition differs')
                seen.add(key)
                require(value['independentHumanReview']['passed'] is True
                        and len(value['independentHumanReview']['recordings']) == 8
                        and all(row['candidatePlan']['passed'] is True and row['outcome']['passed'] is True
                                for row in value['independentHumanReview']['recordings']),
                        'Independent candidate/outcome reconstruction is incomplete')
                for name in ('independentBinaryAccounting', 'independentBoundaryAccounting'):
                    audit = value[name]
                    require(audit['passed'] is True and audit['scopeCount'] == 13 and len(audit['checks']) == 13
                            and all(x['passed'] is True for x in audit['checks']),
                            'Independent binary/boundary accounting is incomplete')
            else:
                key = value['anchor'], value['neuralId'], value['mode'], value['seed']
                require(key not in legacy_seen and value['kind'] == 'legacy-export-disagreement-reference',
                        'Duplicate or invalid legacy result')
                legacy_seen.add(key)
                external(inv, value['source'], root.parent, 'original completed fine-grained review cell')
        inv.close_directory(root/folder, filenames)
    expected_legacy = {(anchor, model, mode, seed) for anchor in c['anchors'] for model in MODELS
                       for mode in ('suppression_review', 'bidirectional_review') for seed in SEEDS}
    require(seen == expected_cells and legacy_seen == expected_legacy, 'Incomplete result cell partition')


def prior_inventory(inv, root, registration):
    c = registration['contract']
    for name, ref in c['priorStudy'].items():
        inv.add(ref['path'], root.parent, 'provenance/production-combinations/'+name, ref)
    for key in ('productionInput', 'neuralInput'):
        ref = c[key]
        inv.add(ref['path'], root.parent, 'provenance/production-combinations/'+Path(ref['path']).name, ref)
    prior_root = Path(c['priorStudy']['registration.json']['path']).parent
    previous = inv.read(prior_root/'registration.json', c['priorStudy']['registration.json'])
    require(canonical(previous['contract']) == previous['sha256'] == H.CONTRACT,
            'Prior production registration differs')
    leaf = prior_root/'source-snapshot'
    receipt = inv.read(leaf/'verification.json')
    require(receipt['passed'] is True and receipt['mode'] == 'created'
            and receipt['zipRoundtripPassed'] is True and receipt['allSourceBytesStable'] is True
            and receipt['fullStudyComplete'] is True and receipt['contractSha256'] == previous['sha256'],
            'Prior reproducible archive is incomplete')
    inv.add(leaf/'verification.json', root.parent, 'provenance/production-combinations/source-snapshot/verification.json')
    inv.add(leaf/'snapshot-manifest.json', root.parent, 'provenance/production-combinations/source-snapshot/snapshot-manifest.json',
            {'path': str(leaf/'snapshot-manifest.json'), 'sha256': receipt['manifestSha256']})
    ref = {'path': str(leaf/'research-source.zip'), 'sha256': receipt['zipSha256'], 'sizeBytes': receipt['zipSizeBytes']}
    external(inv, ref, root.parent, 'previous complete source/runtime/evidence ZIP; verified external archive')
    return ref


def summary_gate(registration, summary, identities):
    require(summary['kind'] == 'perfect-human-review-summary-v1' and summary['passed'] is True
            and summary['contractSha256'] == registration['sha256']
            and summary['hypotheticalHumanOutcomes'] is True
            and all(summary[k] is False for k in ('protectedTestOpened', 'productionChanged',
                                                 'productionPromotionAllowed')),
            'Final hypothetical summary is incomplete or unrelated')
    require(summary['targetPaddingSeconds'] == 2 and summary['joinGapSeconds'] == 3
            and summary['paddingCases'] == [0, 1, 2, 3]
            and summary['recordings'] == 8 and summary['rallies'] == 322
            and summary['seedCount'] == 3 and summary['sourceGroups'] == registration['contract']['sourceGroups']
            and len(summary['configurations']) == 120 and len(summary['legacy']) == 30,
            'Summary policy/scope/metric differs')
    for name, ref in summary['artifacts'].items():
        require(ref['path'] in identities and H.matches(identities[ref['path']], ref),
                'Summary artifact binding differs: ' + name)


def summary_audit_gate(registration, audit, identities):
    require(audit['kind'] == 'independent-perfect-human-review-summary-audit-v1'
            and audit['passed'] is True and audit['contractSha256'] == registration['sha256']
            and audit['protectedTestOpened'] is False and audit['newFits'] == 0 and audit['failures'] == [],
            'Independent summary audit is incomplete or unrelated')
    expected = {'resultCells': 360, 'policies': 120, 'legacyCells': 90, 'legacyPolicies': 30,
                'paddingCases': 4, 'sourceGroupSummaryChecks': 480, 'uncertaintyNestingChecks': 120,
                'seedStatisticChecks': 28320, 'paretoFrontiers': 8, 'seedCount': 3, 'recordings': 8, 'rallies': 322}
    require(all(audit['counts'][k] == v for k, v in expected.items()),
            'Independent summary audit coverage differs')
    for name in ('registration', 'report', 'summary', 'sourceScript', 'tests', 'finalDocument'):
        if name == 'finalDocument' and name not in audit:
            continue  # Explicit final-capture document SHA is required separately.
        ref = audit[name]
        require(ref['path'] in identities and H.matches(identities[ref['path']], ref),
                'Independent summary audit artifact/source binding differs: ' + name)


def add_figure(inv, root, name, expected, folder_name='figures'):
    require(folder_name in {'figures', 'figure-layout-attempt-1'}, 'Unexpected figure directory')
    require(name in {'perfect-human-review-quality-vs-workload.png',
                     'perfect-human-review-quality-vs-workload.svg'}, 'Unexpected figure name/type')
    p = H.safe_path(root/folder_name/name, root/folder_name)
    require(expected['path'] == str(p), 'Figure path differs from its manifest')
    row = inv.bind(p, expected)
    require(0 < row['sizeBytes'] <= 5*1024*1024, 'Figure exceeds bounded artifact budget')
    data = p.read_bytes()
    require(sha(data) == row['sha256'], 'Figure changed during capture')
    archive_name = H.safe_name('study/'+folder_name+'/'+name)
    require(archive_name not in inv.entries, 'Duplicate figure archive entry')
    inv.total += len(data)
    require(inv.total <= H.MAX_TOTAL, 'Archive exceeds total uncompressed budget')
    inv.entries[archive_name] = {'identity': row, 'data': data}


def figure_inventory(inv, root):
    folder = root/'figures'
    value = inv.read(folder/'manifest.json')
    require(value['kind'] == 'perfect-human-review-figures-v1' and value['passed'] is True
            and value['contractSha256'] == CONTRACT, 'Figures are incomplete or unrelated')
    for name in ('sourceScript', 'summary', 'summaryAudit', 'report'):
        ref = value[name]
        require(ref['path'] in inv.bound and H.matches(inv.bound[ref['path']], ref),
                'Figure provenance source/output binding differs')
    require(len(value['files']) == 2, 'Expected exactly the PNG and SVG figure')
    seen = set()
    for ref in value['files']:
        name = ref['name']
        require(name not in seen, 'Duplicate figure identity')
        seen.add(name)
        add_figure(inv, root, name, ref)
    require(seen == {'perfect-human-review-quality-vs-workload.png',
                     'perfect-human-review-quality-vs-workload.svg'}, 'Incomplete figure inventory')
    inv.add(folder/'manifest.json', root, 'study/figures/manifest.json')
    inv.close_directory(folder, seen | {'manifest.json'})
    return inv.bound[str(folder/'manifest.json')]


def superseded_layout_inventory(inv, root):
    folder = root/'figure-layout-attempt-1'
    receipt = inv.read(folder/'superseded-layout.json')
    require(receipt['kind'] == 'superseded-layout-figure-attempt-v1'
            and receipt['status'] == 'superseded-layout' and receipt['closed'] is True,
            'Superseded layout evidence is not closed')
    final = receipt['supersededBy']
    require(final['path'] == str(root/'figures/manifest.json')
            and final['path'] in inv.bound and H.matches(inv.bound[final['path']], final),
            'Superseded layout is not bound to the final figure manifest')
    names = {'perfect-human-review-quality-vs-workload.png', 'perfect-human-review-quality-vs-workload.svg',
             'manifest.json', 'plot-neural-perfect-human-review.py'}
    require(len(receipt['files']) == 4 and {Path(x['path']).name for x in receipt['files']} == names,
            'Superseded layout file inventory differs')
    require({p.name for p in folder.iterdir()} == names | {'superseded-layout.json'}
            and sum(p.stat().st_size for p in folder.iterdir()) < 5*1024*1024,
            'Superseded layout is oversized or contains unexpected files')
    for ref in receipt['files']:
        p = Path(ref['path'])
        require(p.parent == folder, 'Superseded artifact escapes its directory')
        if p.suffix in {'.png', '.svg'}:
            add_figure(inv, root, p.name, ref, folder_name=folder.name)
        else:
            inv.add(p, root, 'study/'+p.relative_to(root).as_posix(), ref)
    original = inv.read(folder/'manifest.json')
    require(original['contractSha256'] == CONTRACT
            and original['sourceScript']['sha256'] == inv.bound[str(folder/'plot-neural-perfect-human-review.py')]['sha256'],
            'Superseded original source is not preserved exactly')
    inv.add(folder/'superseded-layout.json', root, 'study/figure-layout-attempt-1/superseded-layout.json')
    inv.close_directory(folder, names | {'superseded-layout.json'})
    return {'role': 'superseded-layout', 'fileCount': 5,
            'receipt': inv.bound[str(folder/'superseded-layout.json')]}


def build_inventory(root, repo, final_document_sha256):
    inv = H.Inventory()
    # Read completion metadata before opening any individual quality result.
    reg = inv.read(root/'registration.json')
    report = inv.read(root/'report.json')
    summary = inv.read(root/'summary.json')
    audit = inv.read(root/'summary-audit-v1.json')
    interpretation = inv.read(root/'interpretation-audit-v1.json')
    report_gate(reg, report)
    c = reg['contract']
    for rel, ref in c['sources'].items():
        require(ref['path'] == str(repo/rel), 'Registered source path differs')
        inv.add(repo/rel, repo, 'repository/'+rel, ref)
    for rel in sorted(EXTRA_SOURCE):
        inv.add(repo/rel, repo, 'repository/'+rel)
    document = repo/'docs/research/neural-perfect-human-review-results-2026-09-19.md'
    require(len(final_document_sha256) == 64
            and inv.bound[str(document)]['sha256'] == final_document_sha256,
            'Reviewed final document bytes differ')
    summary_gate(reg, summary, inv.bound)
    summary_audit_gate(reg, audit, inv.bound)
    require(interpretation['kind'] == 'independent-perfect-human-review-interpretation-audit-v1'
            and interpretation['passed'] is True and interpretation['contractSha256'] == CONTRACT
            and interpretation['protectedTestOpened'] is False and interpretation['failures'] == []
            and interpretation['tableRowsVerified'] == 1470
            and interpretation['executiveNarrativeReview']['requiredCorrections'] == [],
            'Final document interpretation audit is incomplete')
    for name in ('finalDocument', 'summary', 'summaryAudit', 'report', 'implementation'):
        ref = interpretation[name]
        require(ref['path'] in inv.bound and H.matches(inv.bound[ref['path']], ref),
                'Interpretation audit document/source binding differs')
    for name in metadata_files(root):
        inv.add(root/name, root, 'study/'+name)
    qualification = inv.read(root/'qualification-tests-v1.json')
    require(qualification['passed'] is True and qualification['exitCode'] == 0, 'Qualification failed')
    for name in ('protocol', 'probabilityInput', 'probabilities'):
        inv.bind(c[name]['path'], c[name])
    prior_archive = prior_inventory(inv, root, reg)
    probability_inventory(inv, root, repo, c['probabilityInput'])
    result_inventory(inv, root, reg, report)
    figures = figure_inventory(inv, root)
    superseded = superseded_layout_inventory(inv, root)
    return inv, reg, {
        'finalDocument': inv.bound[str(document)],
        'finalDocumentApprovalBinding': 'Explicit SHA supplied only after root final capture signal',
        'independentSummaryAudit': inv.bound[str(root/'summary-audit-v1.json')],
        'independentInterpretationAudit': inv.bound[str(root/'interpretation-audit-v1.json')],
        'previousCompleteSourceArchive': prior_archive,
        'figuresManifest': figures,
        'supersededLayoutEvidence': superseded,
        'omissions': 'Historical model/checkpoint arrays, videos and feature arrays are not copied. Exact normalized live probabilities/times are included; original array identities remain explicit input-provenance bindings.',
    }


def write_snapshot(root, repo, inv, registration, extra_manifest, create):
    """Write a new-only snapshot after a study-specific completion gate passes."""
    target = root/'source-snapshot'
    require(not target.exists(), 'Immutable snapshot already exists')
    git = H.native_git(repo)
    names = metadata_files(root)
    manifest = {'kind': 'perfect-human-review-reproducible-snapshot-v1',
                'createdAt': datetime.now(timezone.utc).isoformat(),
                'contractSha256': registration['sha256'], 'nativeGit': git,
                'registeredSourcesVerified': len(registration['contract']['sources']),
                'qualifiedHelperSha256': HELPER_SHA,
                'protectedTestOpened': False, 'productionChanged': False,
                'files': [{'archivePath': name, **entry['identity']}
                          for name, entry in sorted(inv.entries.items())],
                'externalBindings': inv.external,
                'uncompressedSourceBytes': inv.total, **extra_manifest}
    mb = encoded(manifest)
    contents = {name: value['data'] for name, value in inv.entries.items()}
    contents['snapshot-manifest.json'] = mb
    inv.recheck()
    require(H.native_git(repo) == git and metadata_files(root) == names,
            'Source or top-level evidence changed during validation')
    receipt = {'kind': 'perfect-human-review-source-snapshot-verification-v1',
               'createdAt': datetime.now(timezone.utc).isoformat(),
               'contractSha256': registration['sha256'], 'mode': 'created' if create else 'validated',
               'passed': True, 'registeredSourcesVerified': len(registration['contract']['sources']),
               'fileCount': len(inv.entries), 'sourceBytes': inv.total,
               'manifestSha256': sha(mb), 'fullStudyComplete': True,
               'allSourceBytesStable': True, 'nativeGitHead': git['head']}
    if create:
        target.mkdir()
        (target/'snapshot-manifest.json').write_bytes(mb)
        archive = target/'research-source.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 19, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, data)
        H.zip_roundtrip(archive, contents, root)
        inv.recheck()
        require(H.native_git(repo) == git and metadata_files(root) == names,
                'Source or evidence inventory changed during binary restore verification')
        receipt.update(zipSha256=H.identity(archive)['sha256'], zipSizeBytes=archive.stat().st_size,
                       zipRoundtripPassed=True, restoredBinaryEntries=len(contents))
        with (target/'verification.json').open('xb') as f:
            f.write(encoded(receipt))
    print(json.dumps(receipt, indent=2))
    return receipt


def main(root, repo, final_document_sha256, create=False):
    require(root.is_dir() and not root.is_symlink(), 'Missing experiment root')
    require(not (root/'source-snapshot').exists(), 'Immutable snapshot already exists')
    before = metadata_files(root)
    inv, reg, details = build_inventory(root, repo, final_document_sha256)
    require(metadata_files(root) == before, 'Top-level metadata inventory changed during capture')
    return write_snapshot(root, repo, inv, reg, details, create)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--final-document-sha256', required=True,
                        help='Exact reviewed final-document hash supplied with final capture authorization')
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    main(args.root, args.repo, args.final_document_sha256, args.create)
