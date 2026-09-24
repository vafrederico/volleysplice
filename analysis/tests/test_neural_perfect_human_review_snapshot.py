"""Probability provenance and archive safety without reading study outcomes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('human_review_snapshot',
    Path(__file__).resolve().parents[2]/'scripts/snapshot-neural-perfect-human-review.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class HumanReviewSnapshotTests(unittest.TestCase):
    def valid_probability(self):
        return {'kind': 'held-source-group-human-review-probabilities-v1',
                'models': M.MODELS, 'seeds': M.SEEDS, 'recordingCount': 8,
                'probabilityArrayCount': 120, 'selectedOuterFitCount': 60,
                'roundTripExact': True, 'probabilityDtype': 'float32', 'timesDtype': 'float64',
                'retainedHeadIndex': 0, 'protectedTestOpened': False, 'trainingPerformed': False,
                'featureExtractionPerformed': False, 'goldUsedForQueueSelection': False,
                'records': [None]*8, 'sourceFits': [None]*60, 'entries': [None]*120,
                'sourceResults': [None]*15}

    def test_probability_gate_rejects_recreated_times_or_gold_flags(self):
        value = self.valid_probability(); M.probability_gate(value)
        for field, bad in [('timesDtype', 'float32'), ('roundTripExact', False),
                           ('goldUsedForQueueSelection', True), ('selectedOuterFitCount', 59)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                M.probability_gate({**value, field: bad})

    def test_external_array_provenance_is_explicitly_not_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            ref = {'path': str(root/'original.npz'), 'sha256': 'a'*64, 'sizeBytes': 100}
            inv = M.H.Inventory()
            M.external(inv, ref, root, 'original array omitted', verify=False)
            self.assertFalse(inv.external[0]['verifiedDuringArchive'])
            self.assertFalse(inv.entries)
            with self.assertRaises(ValueError):
                M.external(inv, {**ref, 'path': str(root.parent/'escape.npz')}, root, 'unsafe', verify=False)

    def test_metadata_inventory_does_not_recurse_into_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'report.json').write_text('{}'); (root/'probabilities.npz').write_bytes(b'NPZ')
            folder = root/'fits'; folder.mkdir(); (folder/'completed.json').write_text('{}')
            self.assertEqual(M.metadata_files(root), {'report.json'})

    def test_report_gate_requires_complete_hypothetical_audited_scope(self):
        c = {'kind': 'fixed-perfect-human-review-development-v1', 'sources': dict.fromkeys(M.REGISTERED),
             'models': M.MODELS, 'seeds': M.SEEDS, 'policyCount': 120, 'resultCells': 360,
             'legacyPolicies': 30, 'legacyCells': 90, 'recordings': 8, 'rallies': 322,
             'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
             'paddingCases': [0,1,2,3], 'protectedTestOpened': False, 'productionChanged': False,
             'productionPromotionAllowed': False, 'candidateFlagsUseGold': False,
             'newFits': 0, 'hypotheticalHumanDecisionsUseGold': True}
        signature = M.canonical(c)
        report = {'kind': c['kind'], 'contractSha256': signature, 'status': 'completed-perfect-human-review',
                  'policyCount': 120, 'resultCells': 360, 'results': [None]*360, 'legacyResults': [None]*90,
                  'independentCandidatePlans': 2880, 'independentRecordingOutcomes': 2880,
                  'durationScopesPerOutcome': 13, 'paddingCasesPerScope': 4,
                  'protectedTestOpened': False, 'productionChanged': False, 'newFits': 0,
                  'hypotheticalHumanOutcomes': True}
        reg = {'contract': c, 'sha256': signature}
        with patch.object(M, 'CONTRACT', signature):
            M.report_gate(reg, report)
            for field, bad in [('independentCandidatePlans', 2879), ('hypotheticalHumanOutcomes', False),
                               ('protectedTestOpened', True), ('status', 'running')]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    M.report_gate(reg, {**report, field: bad})

    def test_summary_audit_requires_full_coverage_and_exact_bindings(self):
        names = ('registration', 'report', 'summary', 'sourceScript', 'tests')
        refs = {name: {'path': '/fixture/'+name, 'sha256': 'a'*64, 'sizeBytes': 10} for name in names}
        identities = {ref['path']: dict(ref) for ref in refs.values()}
        audit = {'kind': 'independent-perfect-human-review-summary-audit-v1', 'passed': True,
                 'contractSha256': M.CONTRACT, 'protectedTestOpened': False, 'newFits': 0, 'failures': [],
                 'counts': {'resultCells': 360, 'policies': 120, 'legacyCells': 90, 'legacyPolicies': 30,
                            'paddingCases': 4, 'sourceGroupSummaryChecks': 480, 'uncertaintyNestingChecks': 120,
                            'seedStatisticChecks': 28320, 'paretoFrontiers': 8, 'seedCount': 3,
                            'recordings': 8, 'rallies': 322}, **refs}
        reg = {'sha256': M.CONTRACT}
        M.summary_audit_gate(reg, audit, identities)
        with self.assertRaises(ValueError):
            M.summary_audit_gate(reg, {**audit, 'counts': {**audit['counts'], 'sourceGroupSummaryChecks': 479}}, identities)
        identities[refs['summary']['path']]['sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            M.summary_audit_gate(reg, audit, identities)

    def test_figure_allowlist_hash_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); folder = root/'figures'; folder.mkdir()
            name = 'perfect-human-review-quality-vs-workload.png'
            p = folder/name; p.write_bytes(b'\x89PNG\r\n\x00test')
            ref = M.H.identity(p)
            inv = M.H.Inventory(); M.add_figure(inv, root, name, ref)
            self.assertEqual(inv.entries['study/figures/'+name]['data'], p.read_bytes())
            with self.assertRaises(ValueError):
                M.add_figure(M.H.Inventory(), root, '../unrelated.png', ref)
            with self.assertRaises(ValueError):
                M.add_figure(M.H.Inventory(), root, 'arbitrary.html', ref)
            with self.assertRaises(ValueError):
                M.add_figure(M.H.Inventory(), root, name, ref, folder_name='arbitrary')
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                M.add_figure(M.H.Inventory(), root, name, ref)

    def test_superseded_layout_is_closed_and_bound_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); old = root/'figure-layout-attempt-1'; old.mkdir()
            final = root/'figures'; final.mkdir(); (final/'manifest.json').write_text('{}')
            source = old/'plot-neural-perfect-human-review.py'; source.write_bytes(b'original source\n')
            (old/'perfect-human-review-quality-vs-workload.png').write_bytes(b'original png')
            (old/'perfect-human-review-quality-vs-workload.svg').write_bytes(b'<svg/>')
            (old/'manifest.json').write_text(json.dumps({'contractSha256': M.CONTRACT,
                                                       'sourceScript': M.H.identity(source)}))
            receipt = {'kind': 'superseded-layout-figure-attempt-v1', 'status': 'superseded-layout',
                       'closed': True, 'supersededBy': M.H.identity(final/'manifest.json'),
                       'files': [M.H.identity(p) for p in sorted(old.iterdir())]}
            marker = old/'superseded-layout.json'; marker.write_text(json.dumps(receipt))
            inv = M.H.Inventory(); inv.bind(final/'manifest.json')
            result = M.superseded_layout_inventory(inv, root)
            self.assertEqual(result['role'], 'superseded-layout')
            self.assertEqual(result['fileCount'], 5)
            self.assertTrue(all(name.startswith('study/figure-layout-attempt-1/') for name in inv.entries))
            receipt['closed'] = False; marker.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                M.superseded_layout_inventory(M.H.Inventory(), root)


if __name__ == '__main__':
    unittest.main()
