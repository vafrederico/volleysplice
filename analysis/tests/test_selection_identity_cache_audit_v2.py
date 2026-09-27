from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_cache_cold_tests_v2', REPO/'scripts/audit-neural-selection-identity-cache-v2.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def ref(name):
    return {'path': '/synthetic-nas/'+name, 'sha256': 'a'*64}


def file_row(name='weights.npz'):
    return {**ref(name), 'stat': {'dev': 1, 'ino': 2, 'size': 3, 'mtimeNs': 4, 'ctimeNs': 5}}


def stage_fixture():
    plan = {'tasks': []}; inventory = {'stages': []}
    for i in range(18):
        task = ref(f'task-{i}.json'); stages = []
        for stage in gate.STAGES:
            output = ref(f'fit-{i}/{stage}.json'); preexisting = output if i == 0 else None
            stages.append({'stage': stage, 'output': output['path'], 'preexistingOutput': preexisting})
            inventory['stages'].append({'task': task, 'stage': stage, 'output': output,
                'preexistingOutput': preexisting, 'execution': None if i == 0 else ref(f'{i}-{stage}-execution.json')})
        plan['tasks'].append({'task': task, 'stages': stages})
    return plan, inventory


class ColdIdentityAuditTests(unittest.TestCase):
    def test_v2_changes_only_registered_workflow_and_auditor_keys(self):
        original = (REPO/'scripts/audit-neural-selection-identity-cache.py').read_bytes()
        expected = original.replace(b'scripts/audit-neural-selection-identity-cache.py',
                                    b'scripts/audit-neural-selection-identity-cache-v2.py')
        expected = expected.replace(b'scripts/generalization-selection-cache-execution.py',
                                    b'scripts/generalization-selection-cache-execution-v2.py')
        self.assertEqual((REPO/'scripts/audit-neural-selection-identity-cache-v2.py').read_bytes(), expected)

    def test_registered_relative_docs_resolve_only_inside_repository(self):
        self.assertEqual(gate.resolve_reference('docs/research/protocol.md'), REPO/'docs/research/protocol.md')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            gate.resolve_reference('../outside-protocol.json')

    def test_complete_original_population_and_three_preexisting_stages_required(self):
        plan, inventory = stage_fixture(); gate.validate_stage_inventory(plan, inventory)
        for mutation in ('dropped', 'reordered', 'hidden-preexisting', 'wrong-output'):
            bad = deepcopy(inventory)
            if mutation == 'dropped': bad['stages'].pop()
            elif mutation == 'reordered': bad['stages'][3], bad['stages'][6] = bad['stages'][6], bad['stages'][3]
            elif mutation == 'hidden-preexisting': bad['stages'][3]['execution'] = None
            else: bad['stages'][3]['output']['path'] = '/synthetic-nas/another-selection.json'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate.validate_stage_inventory(plan, bad)

    def test_file_union_cannot_hide_duplicate_or_changed_stat_identity(self):
        one, two = file_row('a.npz'), file_row('b.npz')
        self.assertEqual(gate.inventory_union([[one], [one, two]]), [one, two])
        changed = deepcopy(one); changed['stat']['ctimeNs'] += 1
        for rows in ([[one, one]], [[two, one]], [[one], [changed]], [[]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError): gate.inventory_union(rows)
        complete = gate.validate_final_union([one, two], [[one], [two]])
        with self.assertRaisesRegex(ValueError, 'narrowed'):
            gate.validate_final_union(complete[:-1], [[one], [two]])

    def test_only_three_hash_bindings_and_valid_stage_statistics(self):
        plan, task, output = ref('plan.json'), ref('task.json'), ref('selection.json')
        stage = {'kind': 'explicit-selection-identity-cache-stage-v1', 'passed': True,
            'plan': plan, 'task': task, 'stage': 'select', 'output': output, 'bindings': gate.BINDINGS,
            'functionBindingsRestored': True, 'numericalFunctionsUnchanged': True, 'inputs': [plan, task],
            'files': [file_row()], 'statistics': {'coldHashCount': 1, 'cacheHitCount': 0, 'coldBytes': 3, 'reusedBytes': 0},
            'startedAtUTC': '2026-09-23T00:00:00Z', 'finishedAtUTC': '2026-09-23T00:00:01Z'}
        gate.validate_execution(stage, plan, task, 'select', output)
        for key, value in [('bindings', gate.BINDINGS+['analysis.decoder.decode']), ('numericalFunctionsUnchanged', False),
                           ('functionBindingsRestored', False), ('inputs', [task]), ('files', [])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.validate_execution({**stage, key: value}, plan, task, 'select', output)
        stage['statistics']['cacheHitCount'] = -1
        with self.assertRaises(ValueError): gate.validate_execution(stage, plan, task, 'select', output)

    def test_cold_hash_reads_bytes_and_rejects_pre_post_stat_race(self):
        stat = file_row()['stat']
        with patch.object(gate, 'signature', return_value=stat), patch.object(Path, 'open', mock_open(read_data=b'abc')):
            digest, observed = gate.cold_hash('/synthetic-nas/file')
            self.assertEqual(digest, hashlib.sha256(b'abc').hexdigest()); self.assertEqual(observed, stat)
        changed = {**stat, 'ctimeNs': 6}
        with patch.object(gate, 'signature', side_effect=[stat, changed]), patch.object(Path, 'open', mock_open(read_data=b'abc')):
            with self.assertRaisesRegex(ValueError, 'during independent cold hash'):
                gate.cold_hash('/synthetic-nas/file')

    def test_cold_evidence_rejects_hash_or_cached_metadata_mismatch(self):
        row = file_row(); stat = row['stat']
        with patch.object(gate, 'cold_hash', return_value=('b'*64, stat)), patch.object(gate, 'signature', return_value=stat):
            with self.assertRaisesRegex(ValueError, 'Cold identity mismatch'):
                gate.ColdEvidence().bind(row, stat)
        with patch.object(gate, 'cold_hash', return_value=(row['sha256'], stat)), patch.object(gate, 'signature', return_value=stat):
            with self.assertRaisesRegex(ValueError, 'immutable metadata'):
                gate.ColdEvidence().bind(row, {**stat, 'mtimeNs': 3})

    def test_declared_evidence_is_bound_but_informational_lineage_is_not_expanded(self):
        root, source, array, unrelated = ref('audit.json'), ref('source.json'), ref('weights.npz'), ref('old-raw.mp4')
        document = {'kind': 'independent-generalization-selection-audit-v1', 'passed': True,
            'task': source, 'selection': source, 'auditor': source, 'references': [array, source],
            'informationalLineage': {'obsolete': unrelated}}
        stat = file_row()['stat']; evidence = gate.ColdEvidence()
        def load(reference):
            evidence.bind(reference)
            self.assertEqual(reference, root)  # Source JSON is pinned, never opened.
            return document
        with patch.object(gate, 'cold_hash', return_value=('a'*64, stat)) as cold, \
             patch.object(gate, 'signature', return_value=stat), patch.object(evidence, 'document', side_effect=load):
            gate.bind_declared_document(evidence, root, document['kind']); evidence.finish()
            self.assertEqual(set(evidence.checked), {r['path'] for r in (root, source, array)})
            self.assertEqual(cold.call_count, 3)

    def test_missing_required_evidence_and_unsupported_schema_fail_closed(self):
        root = ref('audit.json'); evidence = gate.ColdEvidence()
        document = {'kind': 'independent-generalization-selection-audit-v1', 'passed': True,
                    'task': root, 'selection': root, 'auditor': root}
        with patch.object(evidence, 'document', return_value=document):
            with self.assertRaisesRegex(ValueError, 'Missing required numerical'):
                gate.bind_declared_document(evidence, root, document['kind'])
            with self.assertRaisesRegex(ValueError, 'Unsupported numerical'):
                gate.bind_declared_document(evidence, root, 'unregistered-historical-schema')

    def test_declared_required_ref_cannot_escape_cold_hashing(self):
        root, numerical = ref('audit.json'), ref('missing-checkpoint.npz')
        document = {'kind': 'independent-generalization-selection-audit-v1', 'passed': True,
            'task': root, 'selection': root, 'auditor': root, 'references': [numerical]}
        evidence = gate.ColdEvidence()
        def bind(reference, expected_stat=None):
            if reference == numerical:
                raise FileNotFoundError('Required numerical checkpoint missing')
            return Path(reference['path'])
        with patch.object(evidence, 'document', return_value=document), patch.object(evidence, 'bind', side_effect=bind):
            with self.assertRaisesRegex(FileNotFoundError, 'numerical checkpoint'):
                gate.bind_declared_document(evidence, root, document['kind'])


if __name__ == '__main__': unittest.main()
