"""Synthetic metadata/source archives only; no real studies or tensor reads."""
from analysis.private_ledger import private_value
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('followup_snapshot_tests',
    Path(__file__).resolve().parents[2]/'scripts/snapshot-neural-followup-source.py')
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.encode() if isinstance(value, str) else (json.dumps(value, indent=2)+'\n').encode()
    path.write_bytes(data)
    return {'path': str(path), 'sha256': snapshot.digest(data)}


def fixture(root, mode='context', cohorts=None, reference_extra=None):
    repo, study, reference = root/'repository', root/'followup', root/'transfer/study'
    experiment = next(name for name, row in snapshot.EXPERIMENTS.items() if row[0] == mode)
    added = snapshot.EXPERIMENTS[experiment][1]
    code = {name: put(repo/'analysis'/name, '# synthetic '+name+'\n')['sha256']
            for name in snapshot.BASE_SOURCES | added}
    put(repo/'scripts'/f'audit-neural-{mode}-preflight.py', '# synthetic qualification\n')
    final_doc = repo/'docs/research/neural-followup-final.md'
    put(final_doc, 'Synthetic final research document.\n')
    groups, seeds = ['g0', 'g1', 'g2', 'g3'], [3407, 1729, 20260918]
    def row(i, group):
        return {'id': str(i), 'sourceGroup': group, 'environment': 'grass', 'consent': {'train': True}}
    manifest = {'exactRows': [row(i, groups[i//2]) for i in range(8)],
                'draftRows': [row(8+i, 'draft') for i in range(3)],
                'coverageRows': [row(11+i, 'coverage') for i in range(7)],
                'protectedSourceGroups': [private_value('source-group-008')]}
    manifest_ref = put(root/'data/manifest.json', manifest)
    dino_ref = put(root/'data/dino.json', {'records': []})
    old = {'experiment': 'bounded-short-boost-dino-transfer-v1',
           'code': {name: code[name] for name in snapshot.BASE_SOURCES}, 'seeds': seeds,
           'manifestSha256': manifest_ref['sha256'], 'dinoManifest': dino_ref}
    old_sha = snapshot.canonical(old)
    reg_ref = put(reference/'preregistration.json', {'contract': old, 'sha256': old_sha})
    old_report = {'status': 'completed-short-boost-transfer-development', 'contractSha256': old_sha,
                  'protectedTestOpened': False, 'productionPromotionAllowed': False,
                  'results': [{'cohort': c, 'kind': k, 'lossArm': a, 'seed': s}
                              for c in snapshot.COHORTS for k in snapshot.KINDS
                              for a in ('baseline', 'short_boost', 'global_control') for s in seeds]}
    old_report_ref = put(reference/'report.json', old_report)
    old_summary_ref = put(reference/'summary.json', {
        'status': 'completed-short-boost-transfer-audit', 'contractSha256': old_sha, 'inputs': [old_report_ref]})
    audits = [put(reference/f'audit-{i}.json', {'kind': kind, 'passed': True,
              'contractSha256': old_sha, 'report': old_report_ref})
              for i, kind in enumerate(sorted(snapshot.REFERENCE_AUDITS))]
    reference_bound = {'path': str(reference), 'contractSha256': old_sha,
                       'preregistrationFileSha256': reg_ref['sha256'], 'reportSha256': old_report_ref['sha256'],
                       'summarySha256': old_summary_ref['sha256'], 'independentAudits': audits}
    archived = {'repository/analysis/'+name: (repo/'analysis'/name).read_bytes() for name in snapshot.BASE_SOURCES}
    archived.update(reference_extra or {})
    archive_manifest = {'kind': 'short-boost-transfer-research-source-snapshot', 'contractSha256': old_sha,
                        'gitHead': 'head', 'registeredSourcesVerified': 16,
                        'externalBindings': {'studyReport': old_report_ref, 'summary': old_summary_ref},
                        'files': [{'archivePath': name, 'sha256': snapshot.digest(data), 'sizeBytes': len(data)}
                                  for name, data in archived.items()],
                        'sourceFileCount': len(archived), 'sourceBytes': sum(map(len, archived.values()))}
    archive_root = reference.parent/'source-snapshot'
    archive_manifest_ref = put(archive_root/'snapshot-manifest.json', archive_manifest)
    zip_bytes = snapshot.prior.zip_roundtrip(archived, (archive_root/'snapshot-manifest.json').read_bytes())
    (archive_root/'research-source.zip').write_bytes(zip_bytes)
    put(archive_root/'verification.json', {'mode': 'created', 'zipRoundtripPassed': True,
        'allSourceBytesStable': True, 'fullStudyComplete': True, 'contractSha256': old_sha,
        'gitHead': 'head', 'registeredSourcesVerified': 16,
        'zipSha256': snapshot.digest(zip_bytes), 'manifestSha256': archive_manifest_ref['sha256']})
    protocol_ref = put(study/'protocol.md', 'Synthetic prospective protocol.\n')
    preflight = {'kind': f'{mode}-engineering-preflight-v1', 'code': code, 'manifest': manifest_ref,
                 'referenceContractSha256': old_sha, 'referenceReport': old_report_ref,
                 'referenceAudits': audits,
                 'script': snapshot.binding(snapshot.Evidence().identity(repo/'scripts'/f'audit-neural-{mode}-preflight.py'))}
    if mode == 'context':
        preflight.update(dinoManifest=dino_ref, cohort='draft', lossArm='baseline')
    else:
        preflight.update(cohorts=cohorts or ['draft'], executionEnvironment={'device': 'cpu'}, syntheticTests=[])
    plan_ref = put(study/'preflight/plan.json', preflight)
    if mode == 'context':
        preflight.update(passed=True, planFile=plan_ref, planSha256=snapshot.canonical(preflight),
                         ownerReuseQualified=True, originalProfileReplayQualified=True)
    else:
        preflight.update(passed=True, plan=plan_ref, noOpReplayQualified=True, selectionIsolationQualified=True,
                         testExitCode=0, syntheticTestLog=put(study/'preflight/tests.log', 'OK\n'))
    preflight_ref = put(study/'preflight/report.json', preflight)
    c = {'experiment': experiment, 'code': code, 'kinds': snapshot.KINDS, 'seeds': seeds, 'groups': groups,
         'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
         'paddingSweep': [0, 1, 2, 3], 'protectedTestOpened': False, 'productionPromotionAllowed': False,
         'manifest': manifest_ref, 'manifestSha256': manifest_ref['sha256'], 'dinoManifest': dino_ref,
         'referenceStudy': reference_bound, 'protocolSnapshot': protocol_ref, 'preflight': preflight_ref}
    if mode == 'context':
        c.update(cohort='draft', cohorts=['draft'], lossArm='baseline', contexts=['original', 'short'])
        grid = [{'cohort': 'draft', 'kind': k, 'lossArm': 'baseline', 'context': v, 'seed': s}
                for k in snapshot.KINDS for v in c['contexts'] for s in seeds]
        execution = {'physicalFreshFits': 60, 'logicalFreshFits': 96}
        counts = {'canonicalMetricReplays': 12, 'reusedPayloadsVerified': 6,
                  'freshSelectedInnerCandidatesReplayed': 24, 'freshSelectedRefitsReplayed': 24,
                  'fullInnerGridSelectionsReplayed': 24, 'independentPaddingScopeRows': 624}
    else:
        c.update(cohorts=cohorts or ['draft'], variants=['reference', 'selected'], baselineLossArm='baseline',
                 rescueOptions=[None, .35, .5, .65, .8], executionEnvironment={'device': 'cpu'})
        grid = [{'cohort': co, 'kind': k, 'lossArm': 'baseline', 'rescueVariant': v, 'seed': s}
                for co in c['cohorts'] for k in snapshot.KINDS for v in c['variants'] for s in seeds]
        n = len(c['cohorts'])
        execution = {'newFits': 0, 'candidateCells': 6*n, 'referenceCells': 6*n, 'device': 'cpu'}
        counts = {'canonicalMetricReplays': 12*n, 'matchedBaselinePayloadsVerified': 6*n,
                  'outerRescueSelectionsReplayed': 24*n, 'innerOptionScoresIndependentlyReplayed': 120*n,
                  'selectedRefitsDecoded': 24*n, 'independentFinalPaddingScopeRows': 624*n}
    sha = snapshot.canonical(c)
    reg = {'contract': c, 'sha256': sha}
    put(study/'preregistration.json', reg)
    report = {'status': f'completed-{mode}-development', 'contractSha256': sha,
              'manifestSha256': manifest_ref['sha256'], 'records': 8, 'sourceGroups': groups,
              'protectedTestOpened': False, 'productionPromotionAllowed': False,
              'execution': execution, 'results': [{**r, 'contractSha256': sha} for r in grid]}
    report_ref = put(study/'report.json', report)
    summary = {'kind': 'neural-context-summary-v1' if mode == 'context' else 'independent-neural-keep-rescue-summary-v1',
               'status': f'completed-{mode}-development-audit', 'contractSha256': sha, 'passed': True,
               'protectedTestOpened': False, 'productionPromotionAllowed': False,
               'inputs': [report_ref], 'referenceStudy': reference_bound,
               'scope': {'resultCells': len(grid)}, 'audit': {'counts': counts}}
    tensor = None
    if mode == 'context':
        tensor = {'kind': 'independent-context-tensor-audit-v1', 'passed': True, 'contractSha256': sha,
                  'report': report_ref, 'code': code, 'counts': {'physicalFreshFits': 60, 'physicalInnerOwners': 36,
                  'outerRefits': 24, 'logicalInnerViews': 72, 'logicalFreshFits': 96,
                  'freshCheckpoints': 168, 'freshNPZArtifacts': 336, 'referenceResultCells': 6}}
        summary['audit']['tensorAudit'] = put(study/'tensor-audit-v1.json', tensor)
    else:
        summary['newFits'] = 0
        summary['audit'].update(freshFits=0, freshNPZArtifacts=0,
            originalCompletedFitsReferenced={str(i): 'hash' for i in range(96*n)},
            originalNPZArtifactsReferenced={str(i): 'hash' for i in range(192*n)})
    put(study/'summary.json', summary)
    put(study/'summary.md', 'Synthetic completed summary.\n')
    put(study/'loss-identities.json', {})
    return repo, study, reg, report, summary, report_ref, tensor


def operational_fixture(values):
    root = values[1].parent
    monitor = root/'resource-monitor-v1'
    for name in ('observe.py', 'host.ps1', 'continuous/telemetry.jsonl', 'observer.log', 'notes.txt'):
        put(monitor/name, '# synthetic closed evidence\n')
    rows = [{'relativePath': path.relative_to(monitor).as_posix(), 'bytes': path.stat().st_size,
             'sha256': snapshot.digest(path.read_bytes())}
            for path in sorted(monitor.rglob('*')) if path.is_file()]
    put(monitor/'completion-evidence.json', {'kind': 'context-resource-observer-completion-evidence-v1',
        'wslExitCode': 0, 'hostExitCode': 0, 'allObserverProcessesExited': True,
        'finalTelemetryFilesClosed': True, 'evidenceInventory': rows})
    repair = root/'audit-stage/dependency-repair'
    corrected = put(repair/'corrected-audit-neural-context-tensors.py', '# corrected evidence reader\n')
    put(repair/'diagnosis-before-fix.json', {'synthetic': True})
    put(repair/'focused-tests.log', 'OK\n')
    put(repair/'verification.json', {'exitCode': 0, 'testCount': 13,
        'registeredAnalysisSourceCount': 19, 'changedRegisteredAnalysisSources': [],
        'correctedAuditorSha256': corrected['sha256']})
    restoration = root/'runtime-restoration'
    receipt = put(restoration/'restoration.json', {'synthetic': True, 'restored': True})
    put(restoration/'completed.json', {'status': 'completed', 'trainingComplete': True,
        'restorationVerified': True, 'wslRestarted': False,
        'files': [{'relativePath': 'restoration.json', 'sha256': receipt['sha256'],
                   'sizeBytes': (restoration/'restoration.json').stat().st_size}]})
    return monitor, repair, restoration


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def git(self, repo):
        def call(prefix, *args):
            if args[0] == 'rev-parse':
                return b'head\n'
            if args[0] == 'diff' or '--others' in args:
                return b''
            return ('\0'.join(p.relative_to(repo).as_posix() for p in repo.rglob('*') if p.is_file())+'\0').encode()
        return call

    def run_inventory(self, values):
        repo, study, *_ = values
        with patch.object(snapshot, 'REPO', repo), patch.object(snapshot.prior, 'git_prefix', return_value=['git']), \
                patch.object(snapshot.prior, 'git', side_effect=self.git(repo)):
            return snapshot.inventory(study)

    def test_context_complete_inventory_without_writes(self):
        values = fixture(self.root)
        result = self.run_inventory(values)
        manifest, sources = result[:2]
        self.assertEqual(manifest['registeredSourcesVerified'], 19)
        self.assertEqual(manifest['completionCounts']['freshNPZArtifacts'], 336)
        self.assertFalse((values[1]/'source-snapshot').exists())
        self.assertTrue(all(not name.endswith(('.npz', '.mp4')) for name in sources))

    def test_keep_subset_and_full_counts(self):
        for i, cohorts in enumerate((['draft'], snapshot.COHORTS)):
            values = fixture(self.root/str(i), 'keep-rescue', cohorts)
            manifest = self.run_inventory(values)[0]
            self.assertEqual(manifest['registeredSourcesVerified'], 18)
            self.assertEqual(manifest['completionCounts']['resultCells'], 12*len(cohorts))
            self.assertEqual(manifest['completionCounts']['physicalFreshFits'], 0)

    def test_completion_scope_rejections(self):
        _, _, reg, report, summary, report_id, tensor = fixture(self.root)
        for mutation in ('incomplete', 'missingcell', 'promotion', 'auditcount', 'wrongtensor'):
            r, s, t = copy.deepcopy(report), copy.deepcopy(summary), copy.deepcopy(tensor)
            if mutation == 'incomplete': r['status'] = 'running'
            if mutation == 'missingcell': r['results'].pop()
            if mutation == 'promotion': r['productionPromotionAllowed'] = True
            if mutation == 'auditcount': s['audit']['counts']['canonicalMetricReplays'] = 11
            if mutation == 'wrongtensor': t['counts']['physicalFreshFits'] = 59
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                snapshot.validate_scope(reg, r, s, report_id, t)

    def test_wrong_source_inventory_rejected(self):
        _, _, reg, report, summary, report_id, tensor = fixture(self.root)
        reg['contract']['code']['extra.py'] = '0'*64
        reg['sha256'] = snapshot.canonical(reg['contract'])
        with self.assertRaisesRegex(ValueError, 'source inventory'):
            snapshot.validate_scope(reg, report, summary, report_id, tensor)

    def test_keep_new_fits_rejected(self):
        values = fixture(self.root, 'keep-rescue')
        (values[1]/'fits').mkdir()
        with self.assertRaisesRegex(ValueError, 'fits directory'):
            self.run_inventory(values)

    def test_reference_archive_corruption_rejected(self):
        values = fixture(self.root)
        archive = self.root/'transfer/source-snapshot/research-source.zip'
        archive.write_bytes(archive.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError, 'Evidence changed'):
            self.run_inventory(values)

    def test_reference_monitor_powershell_roundtrip_allowed(self):
        extras = {
            'provenance/study/resource-monitor-v5/host-monitor.ps1': b'# synthetic observer\n',
            'provenance/runtime-recovery-v4/validation-observer/host-observe.ps1': b'# synthetic observer\n',
        }
        values = fixture(self.root, reference_extra=extras)
        manifest = self.run_inventory(values)[0]
        self.assertEqual(manifest['registeredSourcesVerified'], 19)

    def test_context_closed_operational_evidence_allowlist(self):
        values = fixture(self.root)
        operational_fixture(values)
        put(self.root/'unrelated/weights.npz', 'not included\n')
        result = self.run_inventory(values)
        manifest, sources = result[:2]
        self.assertEqual([r['relativeDirectory'] for r in manifest['contextOperationalEvidence']],
                         list(snapshot.CONTEXT_OPERATIONAL_DIRS))
        self.assertIn('provenance/context-operation/resource-monitor-v1/host.ps1', sources)
        self.assertIn('provenance/context-operation/resource-monitor-v1/continuous/telemetry.jsonl', sources)
        self.assertIn('provenance/context-operation/runtime-restoration/restoration.json', sources)
        self.assertFalse(any(name.endswith('.npz') or 'unrelated' in name for name in sources))
        result[4].recheck()

    def test_keep_does_not_read_context_operational_directories(self):
        values = fixture(self.root, 'keep-rescue')
        put(self.root/'resource-monitor-v1/weights.npz', 'unrelated active evidence\n')
        result = self.run_inventory(values)
        self.assertNotIn('contextOperationalEvidence', result[0])
        self.assertFalse(any('context-operation/' in name for name in result[1]))

    def test_context_rejects_unfinished_changed_or_unsafe_evidence(self):
        for index, mutation in enumerate(('missing-restoration', 'running', 'changed-telemetry',
                                         'npz', 'fits', 'restoration-unverified', 'unsafe-inventory')):
            values = fixture(self.root/str(index))
            monitor, _, restoration = operational_fixture(values)
            if mutation == 'missing-restoration': (restoration/'completed.json').unlink()
            elif mutation == 'changed-telemetry': put(monitor/'continuous/telemetry.jsonl', 'changed\n')
            elif mutation == 'npz': put(monitor/'predictions.npz', 'not read\n')
            elif mutation == 'fits': put(monitor/'fits/completed.json', {})
            elif mutation == 'restoration-unverified':
                marker = json.loads((restoration/'completed.json').read_text())
                marker['restorationVerified'] = False
                put(restoration/'completed.json', marker)
            else:
                marker = json.loads((monitor/'completion-evidence.json').read_text())
                if mutation == 'running': marker['allObserverProcessesExited'] = False
                else: marker['evidenceInventory'][0]['relativePath'] = '../outside.json'
                put(monitor/'completion-evidence.json', marker)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_inventory(values)

    def test_context_rechecks_closed_inventory_for_new_files_and_symlinks(self):
        values = fixture(self.root)
        monitor, _, _ = operational_fixture(values)
        result = self.run_inventory(values)
        put(monitor/'late.json', {})
        with self.assertRaisesRegex(ValueError, 'changed during snapshot'):
            result[4].recheck()
        (monitor/'late.json').unlink()
        original = Path.is_symlink
        with patch.object(Path, 'is_symlink', lambda path: path == monitor/'host.ps1' or original(path)):
            with self.assertRaisesRegex(ValueError, 'Symlink operational'):
                self.run_inventory(values)

    def test_nonmonitor_powershell_reference_member_rejected(self):
        for index, name in enumerate(('repository/scripts/arbitrary.ps1',
                                      'provenance/unrelated/arbitrary.ps1',
                                      'provenance/study/resource-monitor-v5/.env/private.ps1')):
            values = fixture(self.root/str(index), reference_extra={name: b'# synthetic source\n'})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Nonresearch preceding'):
                self.run_inventory(values)

    def test_unpassed_reference_audit_rejected(self):
        values = fixture(self.root)
        audit = Path(values[2]['contract']['referenceStudy']['independentAudits'][0]['path'])
        data = json.loads(audit.read_text())
        data['passed'] = False
        put(audit, data)
        with self.assertRaisesRegex(ValueError, 'Evidence changed'):
            self.run_inventory(values)

    def test_registered_copy_hash_rejected(self):
        values = fixture(self.root)
        folder = values[1]/'sources'
        for name in values[2]['contract']['code']:
            put(folder/name, (values[0]/'analysis'/name).read_text())
        put(folder/'neural_context_fit.py', 'changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.run_inventory(values)

    def test_large_preflight_bound_not_archived(self):
        values = fixture(self.root)
        # Simulate the same budget boundary without allocating multi-megabyte fixtures.
        with patch.object(snapshot.prior, 'MAX_FILE_BYTES', 2500):
            # Registered source/preregistration is also external-bound when oversized.
            result = self.run_inventory(values)
        self.assertNotIn('preflight/report.json', result[1])
        self.assertIn(str(values[1]/'preflight/report.json'), result[4].files)

    def test_forbidden_and_escaping_paths(self):
        put(self.root/'.env.json', '{}')
        put(self.root/'weights.npz', 'not even binary')
        for name in ('.env.json', 'weights.npz'):
            with self.assertRaises(ValueError):
                snapshot.Evidence().identity(self.root/name)
        put(self.root/'outside.py', '# outside')
        with self.assertRaises(ValueError):
            snapshot.prior.safe_read(self.root/'outside.py', self.root/'allowed')

    def test_create_requires_final_doc_and_never_overwrites(self):
        values = fixture(self.root)
        repo, study = values[:2]
        with self.assertRaisesRegex(ValueError, 'final-doc'):
            snapshot.main(['--study', str(study), '--create'])
        with patch.object(snapshot, 'REPO', repo), patch.object(snapshot.prior, 'git_prefix', return_value=['git']), \
                patch.object(snapshot.prior, 'git', side_effect=self.git(repo)), redirect_stdout(io.StringIO()):
            snapshot.main(['--study', str(study)])
            self.assertFalse((study/'source-snapshot').exists())
            args = ['--study', str(study), '--create', '--final-doc', 'docs/research/neural-followup-final.md']
            verification = snapshot.main(args)
            self.assertTrue(verification['zipRoundtripPassed'])
            with zipfile.ZipFile(study/'source-snapshot/research-source.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn('repository/docs/research/neural-followup-final.md', archive.namelist())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                snapshot.main(args)


if __name__ == '__main__':
    unittest.main()
