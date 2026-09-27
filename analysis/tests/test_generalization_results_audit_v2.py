from analysis.private_ledger import private_value
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OLD = REPO/'scripts/audit-neural-generalization-results.py'
NEW = REPO/'scripts/audit-neural-generalization-results-v2.py'
spec = importlib.util.spec_from_file_location('independent_results_audit_v2_tests', NEW)
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(str(path).encode()).hexdigest()}


class Evidence:
    def __init__(self, documents=None):
        self.documents = documents or {}; self.opened = []; self.bound = []
    def capture(self, path):
        result = ref(path)
        if Path(path).name == 'audit-duration-provenance.json': result['sha256'] = gate.DURATION_PROOF_SHA
        self.bound.append(result); return result
    def document(self, reference):
        self.opened.append(reference['path'])
        if reference['path'] not in self.documents: raise FileNotFoundError(reference['path'])
        return deepcopy(self.documents[reference['path']])
    def bind(self, reference):
        self.bound.append(reference); return Path(reference['path'])
    def closure(self, value):
        pass


class Payload(dict):
    def __enter__(self): return self
    def __exit__(self, *_): pass


def duration_fixture():
    root = Path(private_value('private-reference-0064'))
    protocol_ref = ref(root/'protocol.json')
    protocol = {'records': ref(root/'records.json'), 'source': ref(root/'source.json')}
    stored, source, checks, payloads, inventory = [], [], [], {}, {}
    for i in range(8):
        name = 'recording-'+str(i); frame_count = 300 if i == 0 else 298+3*i
        duration = frame_count/30.; display = round(duration, 6)
        cache_ref = ref(root/(name+'.npz')); metadata = {'duration': duration, 'frame_count': frame_count, 'fps': 30.}
        row = {'id': name, 'sourceGroup': 'group-'+str(i), 'durationSeconds': duration,
            'timestamps': [0., .5], 'valid': [False, True],
            'rallies': [{'start': 1., 'end': 2.}], 'ignoredIntervals': [{'start': 0., 'end': .1}]}
        stored.append(row)
        source.append({**deepcopy(row), 'durationSeconds': display,
            'featureCaches': {'audiovisual': {**cache_ref, 'metadata': metadata}}})
        inventory[name] = {**deepcopy(row), 'durationSeconds': display}
        payloads[cache_ref['path']] = Payload(metadata_json=np.asarray(json.dumps(metadata)), times=np.asarray(row['timestamps']))
        checks.append({'id': name, 'sourceGroup': row['sourceGroup'], 'audiovisualCache': cache_ref,
            'storedMetricDurationSeconds': duration, 'cacheMetadataDurationSeconds': duration,
            'manifestDurationSeconds': display, 'frameCount': frame_count, 'fps': 30.,
            'manifestMinusStoredSeconds': display-duration,
            'storedEqualsBothCacheMetadataDurations': True, 'storedEqualsFrameCountDivFps': True,
            'manifestEqualsExactSixDecimalRound': True, 'timestampCount': 2,
            'timestampsExactlyCache': True, 'sourceGroupAndIntervalValuesAndTagsExact': True,
            'validMaskExactlyIgnoredSubtraction': True})
    proof = {'kind': 'independent-historical-duration-provenance-audit-v1', 'passed': True,
        'protocol': protocol_ref, **protocol, 'recordingCount': 8, 'roundedManifestDifferenceCount': 7,
        'auditor': {'path': str(REPO/'scripts/audit-neural-historical-duration-provenance.py'), 'sha256': gate.DURATION_AUDITOR_SHA},
        'checks': checks, **{key: False for key in ('metricDurationChanged', 'selectionChanged', 'trainingPerformed', 'gpuUsed', 'rawVideoOpened')}}
    documents = {str(root/'audit-duration-provenance.json'): proof,
        protocol['records']['path']: {'records': stored}, protocol['source']['path']: {'exactRows': source}}
    return protocol_ref, protocol, documents, inventory, payloads


class FinalAuditV2Tests(unittest.TestCase):
    def test_frozen_source_and_all_numerical_oracle_ast_unchanged(self):
        # The immutable receipt binds the archived source, before publication sanitization.
        frozen = Path(private_value('private-reference-0206'))
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), gate.ORIGINAL_AUDITOR_SHA)
        old, new = ast.parse(OLD.read_text()), ast.parse(NEW.read_text())
        named = lambda tree: {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        a, b = named(old), named(new)
        self.assertEqual(set(b)-set(a), {'historical_report_gates', 'historical_duration_checks', 'validate_historical_duration'})
        for name, node in a.items():
            with self.subTest(name=name):
                self.assertEqual(ast.dump(node.args) if isinstance(node, ast.FunctionDef) else node.name,
                                 ast.dump(b[name].args) if isinstance(node, ast.FunctionDef) else b[name].name)
                if name not in ('audit_historical', 'audit'):
                    self.assertEqual(ast.dump(node, include_attributes=False), ast.dump(b[name], include_attributes=False))
        # Runtime CLI argument parsing and every preexisting constant/import stay unchanged.
        added = {'ORIGINAL_AUDITOR_SHA', 'DURATION_PROOF_SHA', 'DURATION_AUDITOR_SHA', 'REVISION_POLICY'}
        def remainder(tree):
            return [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in added for t in node.targets))]
        self.assertEqual([ast.dump(n) for n in remainder(old)], [ast.dump(n) for n in remainder(new)])

    def test_only_declared_historical_and_receipt_source_edits(self):
        old, new = OLD.read_text(), NEW.read_text()
        a = ast.parse(old); b = ast.parse(new)
        functions = lambda tree, source: {n.name: ast.get_source_segment(source, n)
            for n in tree.body if isinstance(n, ast.FunctionDef)}
        original, current = functions(a, old), functions(b, new)
        historical = original['audit_historical'].replace(
            "    original_ids = [r['id'] for r in evidence.document(original_protocol['records'])['records']]",
            "    original_ids = [r['id'] for r in evidence.document(original_protocol['records'])['records']]\n"
            "    duration_checks = historical_duration_checks(original_protocol_ref, original_protocol, evidence, inventory)")
        historical = historical.replace('    candidates = []', '    original_report_ref, candidates = historical_report_gates(historical_dir, evidence)')
        start = historical.index("        for p in path.parent.glob('audit*.json'):")
        end = historical.index("    io.require(candidates,", start)
        historical = historical[:start]+"        io.require(entry['result'] == original_report_ref, 'Historical FP32 report authority differs')\n"+historical[end:]
        historical = historical.replace("                    compare(raw['durationSeconds'], authoritative['durationSeconds'], 'Historical duration')",
            "                    validate_historical_duration(raw, authoritative, duration_checks[raw['id']])")
        self.assertEqual(current['audit_historical'], historical)
        audit = original['audit'].replace(
            '    evidence = Evidence(); index_ref = evidence.capture(index_path); index = evidence.document(index_ref)',
            "    evidence = Evidence(); index_ref = evidence.capture(index_path); index = evidence.document(index_ref)\n"
            "    original_auditor = {'path': str(REPO/'scripts/audit-neural-generalization-results.py'), 'sha256': ORIGINAL_AUDITOR_SHA}\n"
            '    evidence.bind(original_auditor)')
        audit = audit.replace("    result = {'kind': 'independent-generalization-result-report-audit-v1', 'passed': True,",
            "    result = {'kind': 'independent-generalization-result-report-audit-v2', 'passed': True,\n"
            "        'originalAuditor': original_auditor, 'auditContractRevision': REVISION_POLICY,")
        self.assertEqual(current['audit'], audit)

    def test_two_audit_authorities_have_distinct_required_schemas(self):
        root = Path(private_value('private-reference-0065')); report = ref(root/'report-complete.json')
        documents = {}
        specs = [('audit-complete.json', 'independent-recall-floor-sweep-audit-v1', 'audit-neural-recall-sweep.py'),
                 ('audit-controls-complete.json', 'independent-historical-recall-control-replay-v1', 'audit-neural-historical-controls.py')]
        for filename, kind, auditor in specs:
            documents[str(root/filename)] = {'passed': True, 'kind': kind, 'report': report, 'auditor': ref(REPO/'scripts'/auditor)}
        documents[str(root/'audit-informational.json')] = {'passed': True, 'report': report, 'kind': 'unrelated'}
        evidence = Evidence(documents)
        with patch.object(gate.io, 'identity', side_effect=ref):
            result, gates = gate.historical_report_gates(root, evidence)
            self.assertEqual(result, report); self.assertEqual(len(gates), 2)
            self.assertNotIn(str(root/'audit-informational.json'), evidence.opened)
            for key, value in [('kind', 'independent-recall-floor-sweep-audit-v1'), ('passed', False),
                               ('report', ref(root/'other-report.json')), ('auditor', ref(root/'wrong.py'))]:
                bad = deepcopy(documents); bad[str(root/'audit-controls-complete.json')][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError): gate.historical_report_gates(root, Evidence(bad))
            del documents[str(root/'audit-controls-complete.json')]
            with self.assertRaises(FileNotFoundError): gate.historical_report_gates(root, Evidence(documents))

    def test_duration_pair_rejects_tiny_changes_without_tolerance(self):
        _, _, documents, inventory, _ = duration_fixture()
        stored = next(d['records'] for d in documents.values() if 'records' in d and isinstance(d['records'], list))
        proof = next(d for d in documents.values() if 'checks' in d)
        row, check = stored[1], proof['checks'][1]; authoritative = inventory[row['id']]
        gate.validate_historical_duration(row, authoritative, check)
        for side, key, value in [('raw', 'durationSeconds', np.nextafter(row['durationSeconds'], float('inf'))),
                ('inventory', 'durationSeconds', np.nextafter(authoritative['durationSeconds'], float('inf'))),
                ('check', 'cacheMetadataDurationSeconds', 99.), ('check', 'frameCount', check['frameCount']+1),
                ('check', 'fps', 29.), ('raw', 'sourceGroup', 'wrong'), ('inventory', 'id', 'wrong')]:
            values = {'raw': deepcopy(row), 'inventory': deepcopy(authoritative), 'check': deepcopy(check)}
            values[side][key] = value
            with self.subTest(side=side, key=key), self.assertRaises(ValueError):
                gate.validate_historical_duration(values['raw'], values['inventory'], values['check'])

    def test_full_duration_proof_replayed_without_mutating_gold(self):
        p, protocol, documents, inventory, payloads = duration_fixture()
        before = deepcopy((documents, inventory))
        with patch.object(gate.np, 'load', side_effect=lambda path, **_: payloads[str(path)]):
            result = gate.historical_duration_checks(p, protocol, Evidence(documents), inventory)
        self.assertEqual(len(result), 8); self.assertEqual((documents, inventory), before)

    def test_duration_proof_rejects_source_scope_cache_and_mask_tampering(self):
        for mutation in ('proof-source', 'proof-count', 'proof-metric-change', 'cache-duration', 'cache-frames',
                         'timestamps', 'mask', 'label', 'inventory-display', 'proof-cache-ref', 'proof-byte-identity'):
            p, protocol, documents, inventory, payloads = duration_fixture(); evidence = Evidence(documents)
            proof = next(d for d in documents.values() if 'checks' in d)
            stored = next(d['records'] for d in documents.values() if 'records' in d and isinstance(d['records'], list))
            payload = payloads[proof['checks'][1]['audiovisualCache']['path']]
            if mutation == 'proof-source': proof['source'] = ref(private_value('private-reference-0066'))
            elif mutation == 'proof-count': proof['recordingCount'] = 7
            elif mutation == 'proof-metric-change': proof['metricDurationChanged'] = True
            elif mutation.startswith('cache-'):
                meta = json.loads(str(payload['metadata_json'].item()))
                meta['duration' if mutation == 'cache-duration' else 'frame_count'] += 1
                payload['metadata_json'] = np.asarray(json.dumps(meta))
            elif mutation == 'timestamps': payload['times'] = np.asarray([0., .6])
            elif mutation == 'mask': stored[1]['valid'][0] = True
            elif mutation == 'label': stored[1]['rallies'][0]['end'] += .01
            elif mutation == 'inventory-display': inventory[stored[1]['id']]['durationSeconds'] += 1e-6
            elif mutation == 'proof-cache-ref': proof['checks'][1]['audiovisualCache'] = ref(private_value('private-reference-0067'))
            else:
                capture = evidence.capture
                evidence.capture = lambda path: {**capture(path), 'sha256': 'b'*64}
            with self.subTest(mutation=mutation), patch.object(gate.np, 'load', side_effect=lambda path, **_: payloads[str(path)]):
                with self.assertRaises(ValueError): gate.historical_duration_checks(p, protocol, evidence, inventory)


if __name__ == '__main__': unittest.main()
