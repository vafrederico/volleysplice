"""Archive completion, provenance and portable replay safety on synthetic data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts'/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


M = load('rally_snapshot', 'snapshot-neural-rally-review-proposals.py')
R = load('rally_portable', 'replay-neural-rally-review-snapshot.py')


class RallyReviewSnapshotTests(unittest.TestCase):
    def contract(self):
        return {'sources': dict.fromkeys(M.REGISTERED), 'models': ['compact_boost', 'dino_global', 'dino_boost'],
                'seeds': [3407, 1729, 20260918], 'configurationSeedRuns': 108, 'outcomeCells': 432,
                'configurations': [None]*36, 'budgetFractions': [.05, .10, .20, .40],
                'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
                'paddingCases': [0, 1, 2, 3], 'editMarginSeconds': 2, 'viewContextSeconds': 2,
                'goldUsedForProposalsOrRanking': False, 'hypotheticalHumanOutcomes': True,
                'newDetectorFits': 0, 'learnedReviewFits': 0, 'protectedTestOpened': False,
                'productionChanged': False, 'kind': 'fixed-budget-rally-review-proposals-v1'}

    def test_rejects_incomplete_or_forbidden_registration_report(self):
        c = self.contract()
        sha = M.H.canonical(c)
        reg = {'contract': c, 'sha256': sha}
        report = {key: c[key] for key in ('kind', 'newDetectorFits', 'learnedReviewFits', 'protectedTestOpened', 'productionChanged')}
        report.update(status='completed-fixed-rally-review-proposals', contractSha256=sha,
                      configurationSeedRuns=108, outcomeCells=432, candidatePlansAudited=864,
                      editedRecordingsAudited=3456, results=[None]*108)
        with patch.object(M, 'CONTRACT', sha):
            M.completion_gate(reg, report)
            for key, value in [('candidatePlansAudited', 863), ('editedRecordingsAudited', 3455),
                               ('protectedTestOpened', True), ('newDetectorFits', 1), ('status', 'running')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    M.completion_gate(reg, {**report, key: value})
            with self.assertRaises(ValueError):
                M.completion_gate({'sha256': sha, 'contract': {**c, 'viewContextSeconds': 0}}, report)

    def evaluation(self):
        return {'durationMetrics': [None]*4,
                'durationAudit': {'passed': True, 'scopeCount': 13, 'paddingCases': 4},
                'identityAudit': {'passed': True, 'recordingsAudited': 8, 'sourceGroupsAudited': 4,
                                  'observedBoundaryFlagsAudited': True}}

    def test_requires_identity_and_all_padding_accounting_audits(self):
        good = self.evaluation()
        M.evaluation_gate(good)
        for key, value in [('durationAudit', {'passed': True, 'scopeCount': 12, 'paddingCases': 4}),
                           ('identityAudit', {**good['identityAudit'], 'observedBoundaryFlagsAudited': False})]:
            with self.assertRaises(ValueError):
                M.evaluation_gate({**good, key: value})

    def test_metadata_discovery_excludes_large_arrays_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'report.json').write_text('{}')
            (root/'probabilities.npz').write_bytes(b'array')
            (root/'fits').mkdir()
            (root/'fits/completed.json').write_text('{}')
            self.assertEqual(M.metadata_files(root), {'report.json'})

    def audit_fixture(self):
        doc = Path('/fixture/final.md')
        ref = {'path': str(doc), 'sha256': 'a'*64, 'sizeBytes': 10}
        summary = {'kind': 'fixed-rally-review-proposals-summary-v1', 'contractSha256': M.CONTRACT,
                   'arms': [None]*144, 'automaticBaselines': [None]*4, 'targetPaddingSeconds': 2, 'joinGapSeconds': 3}
        audit = {'passed': True, 'contractSha256': M.CONTRACT, 'kind': 'independent-rally-review-summary-audit-v1',
                 'configurationSeedRunsAudited': 108, 'outcomeCellsAudited': 432, 'candidatePlansAudited': 864,
                 'editedRecordingsAudited': 3456, 'automaticBaselinesAudited': 4, 'armsAudited': 144,
                 'sourceGroupArmSummariesAudited': 576, 'paddingCasesAudited': 4, 'rankingListsAudited': 8,
                 'guardrailScreensAudited': 720, 'numericalReconstructionAuditsRequiredPassed': True,
                 'seedStatisticsReconstructed': True}
        interpretation = {'passed': True, 'contractSha256': M.CONTRACT, 'finalDocument': ref,
                          'kind': 'independent-rally-review-interpretation-audit-v2',
                          'generatedMarkdownDataRowsVerified': 1748, 'manualNarrativeReview': 'checked'}
        return doc, summary, audit, interpretation, {str(doc): ref}

    def test_final_document_binding_and_summary_coverage_required(self):
        doc, summary, audit, interpretation, identities = self.audit_fixture()
        reg = {'sha256': M.CONTRACT}
        M.audit_gate(reg, summary, audit, interpretation, identities, doc)
        for altered in [{**audit, 'outcomeCellsAudited': 431}, {**audit, 'seedStatisticsReconstructed': False}]:
            with self.assertRaises(ValueError):
                M.audit_gate(reg, summary, altered, interpretation, identities, doc)
        with self.assertRaises(ValueError):
            M.audit_gate(reg, summary, audit, {**interpretation, 'finalDocument': {**interpretation['finalDocument'], 'sha256': 'b'*64}}, identities, doc)
        with self.assertRaises(ValueError):
            M.audit_gate(reg, summary, audit, {**interpretation, 'unbound': {'path': '/unknown', 'sha256': 'c'*64}}, identities, doc)

    def test_recursive_refs_retain_embedded_seed_bindings(self):
        ref = {'path': '/a', 'sha256': 'a'*64}
        self.assertEqual(list(M.binding_refs({'arm': [{'seedResults': [ref]}]})), [ref])

    def test_figure_bound_bytes_type_size_and_new_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root/'plot.png'; path.write_bytes(b'PNG\x00data')
            ref = M.H.identity(path)
            inv = M.H.Inventory()
            M.add_binary(inv, path, root, 'study/figures/plot.png', ref)
            self.assertEqual(inv.entries['study/figures/plot.png']['data'], b'PNG\x00data')
            with self.assertRaises(ValueError):
                M.add_binary(inv, path, root, 'study/figures/plot.png', ref)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                M.add_binary(M.H.Inventory(), path, root, 'study/figures/plot.png', ref)

    def test_portable_entries_reject_traversal_absolute_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root/'file.json'; path.write_text('{}')
            self.assertEqual(R.safe_entry(root, 'file.json'), path)
            for name in ('../file.json', '/file.json', 'C:/file.json', 'folder\\file.json'):
                with self.assertRaises(ValueError):
                    R.safe_entry(root, name)
            link = root/'link.json'; link.symlink_to(path)
            with self.assertRaises(ValueError):
                R.safe_entry(root, 'link.json')

    def snapshot_fixture(self, root):
        refs, entries = {}, []
        for key, name in [('input', 'rally-review-input.json'), ('probabilities', 'probabilities.npz'),
                          ('protocol', 'protocol-initial.md'), ('qualification', 'qualification-v1.json'),
                          ('goldSemantics', 'gold-semantics-audit-v1.json')]:
            path = root/'study'/name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'fixture'+name.encode())
            refs[key] = M.H.identity(path)
            entries.append({'archivePath': 'study/'+name, **refs[key]})
        source = root/'repository/source.py'; source.parent.mkdir(); source.write_bytes(b'x=1\n')
        source_ref = M.H.identity(source)
        entries.append({'archivePath': 'repository/source.py', **source_ref})
        c = {'sources': {'source.py': source_ref}, **refs}
        signature = R.canonical(c)
        path = root/'study/registration.json'
        path.write_text(json.dumps({'contract': c, 'sha256': signature}))
        entries.append({'archivePath': 'study/registration.json', **M.H.identity(path)})
        manifest = {'kind': 'rally-review-proposals-reproducible-snapshot-v1', 'contractSha256': signature, 'files': entries}
        (root/'snapshot-manifest.json').write_text(json.dumps(manifest))
        return signature, manifest

    def test_portable_snapshot_verifies_every_original_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            signature, _ = self.snapshot_fixture(root)
            with patch.object(R, 'CONTRACT', signature):
                self.assertEqual(len(R.verify_snapshot(root)[2]), 7)
                (root/'repository/source.py').write_bytes(b'x=2\n')
                with self.assertRaises(ValueError):
                    R.verify_snapshot(root)

    def test_portable_snapshot_rejects_duplicate_or_unregistered_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            signature, manifest = self.snapshot_fixture(root)
            manifest['files'].append(manifest['files'][0])
            (root/'snapshot-manifest.json').write_text(json.dumps(manifest))
            with patch.object(R, 'CONTRACT', signature), self.assertRaises(ValueError):
                R.verify_snapshot(root)


if __name__ == '__main__':
    unittest.main()
