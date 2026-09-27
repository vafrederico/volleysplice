import copy
import importlib.util
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts'/filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


cold = load('student_cold_publication_tests', 'audit-neural-student-inference-cache.py')
final = load('student_finalization_v3_tests', 'finalize-neural-generalization-v3.py')


def fixture():
    tasks, rows = [], []
    for index in range(27):
        folder = '/registered/fit-'+str(index)
        task = {'taskId': 'student/'+str(index), 'task': {'path': '/tasks/'+str(index), 'sha256': 'a'*64},
                'fitDirectory': folder}
        tasks.append(task)
        path = folder+'/inference-numerical-audit-fp32.json'
        rows.append({**task, 'mode': 'preexisting-uncached' if index < 2 else 'cached-required',
            'numericAuditPath': path, 'numericAudit': {'path': path, 'sha256': 'b'*64} if index < 2 else None,
            'preexistingCompletion': {'path': '/previous/'+str(index), 'sha256': 'c'*64} if index < 2 else None,
            'executionCompanion': None if index < 2 else folder+'/student-inference-cache-execution-fp32.json'})
    plan = {'kind': 'registered-streaming-student-panel-queue-v3', 'tasks': tasks, 'numericAuditClassification': rows}
    reference = {'path': '/plan.json', 'sha256': 'd'*64}
    completed = {'kind': 'streaming-student-full-panel-completed-v3', 'passed': True,
        'plan': reference, 'logicalTaskCount': 27, 'all27Full42All4NumericAndApplicableReuseGatesPassed': True,
        'numericExecutionCompanions': [{'taskId': row['taskId'], 'companion': {
            'path': row['executionCompanion'], 'sha256': 'e'*64}} for row in rows[2:]]}
    return plan, reference, completed


class StudentCachePublicationTests(unittest.TestCase):
    def test_all27_partition_and_all25_new_companions(self):
        plan, reference, completed = fixture()
        rows = cold.classifications(plan)
        self.assertEqual(len(cold.completed_companions(reference, completed, rows)), 25)

    def test_missing_and_duplicate_classification_rejected(self):
        for mutation in ('missing', 'duplicate', 'order'):
            plan, _, _ = fixture()
            if mutation == 'missing': plan['numericAuditClassification'].pop()
            if mutation == 'duplicate': plan['tasks'][-1] = plan['tasks'][0]
            if mutation == 'order': plan['numericAuditClassification'].reverse()
            with self.assertRaises(ValueError): cold.classifications(plan)

    def test_new_audit_cannot_claim_preexisting_without_receipt(self):
        plan, _, _ = fixture(); row = plan['numericAuditClassification'][2]
        row['mode'] = 'preexisting-uncached'
        row['numericAudit'] = {'path': row['numericAuditPath'], 'sha256': 'b'*64}
        row['executionCompanion'] = None
        with self.assertRaises(ValueError): cold.classifications(plan)

    def test_new_audit_requires_canonical_companion_path(self):
        plan, _, _ = fixture(); plan['numericAuditClassification'][2]['executionCompanion'] = '/wrong.json'
        with self.assertRaises(ValueError): cold.classifications(plan)

    def test_missing_extra_reordered_companions_rejected(self):
        for mutation in ('missing', 'extra', 'order', 'wrong-plan', 'incomplete'):
            plan, reference, completed = fixture()
            if mutation == 'missing': completed['numericExecutionCompanions'].pop()
            if mutation == 'extra': completed['numericExecutionCompanions'].append(completed['numericExecutionCompanions'][0])
            if mutation == 'order': completed['numericExecutionCompanions'].reverse()
            if mutation == 'wrong-plan': completed['plan'] = {**reference, 'sha256': 'f'*64}
            if mutation == 'incomplete': completed['all27Full42All4NumericAndApplicableReuseGatesPassed'] = False
            with self.assertRaises(ValueError):
                cold.completed_companions(reference, completed, plan['numericAuditClassification'])

    def test_omitted_or_wrong_hash_cold_evidence_rejected(self):
        helper = cold.helper()
        reference = {'path': str(REPO/'known-input.npy'), 'sha256': 'a'*64}
        key = str(Path(reference['path']).resolve())
        cold.require_covered({'nested': [reference]}, {key: reference}, helper)
        for files in ({}, {key: {**reference, 'sha256': 'b'*64}}):
            with self.assertRaises(ValueError): cold.require_covered({'nested': [reference]}, files, helper)

    def test_final_wrapper_preserves_every_delegated_field(self):
        original = {'kind': 'independent-generalization-finalization-audit-v1', 'passed': True,
                    'reportContentSha256': 'a'*64, 'metrics': {'recall': 0.99}}
        outer = {**copy.deepcopy(original), 'kind': final.KIND,
                 **{key: {'path': '/proof', 'sha256': 'b'*64} for key in final.EXTRA}}
        final.restore_delegated(outer, original)
        for key, value in (('reportContentSha256', 'c'*64), ('passed', False), ('metrics', {'recall': 1.0})):
            changed = {**outer, key: value}
            with self.assertRaises(ValueError): final.restore_delegated(changed, original)
        with self.assertRaises(ValueError): final.restore_delegated({**outer, 'unexpected': True}, original)

    def test_process_stage_duplicates_gaps_and_warm_import_rejected(self):
        first = {'pid': 123, 'bootId': 'boot', 'processStartTimeTicks': 456,
                 'processStageOrdinal': 1, 'statistics': {'coldHashCount': 20}}
        second = {**first, 'processStageOrdinal': 2, 'statistics': {'coldHashCount': 0}}
        cold.validate_process_stages([second, first])
        for rows in ([first, first], [second], [first, {**second, 'processStageOrdinal': 3}],
                     [{**first, 'statistics': {'coldHashCount': 0}}]):
            with self.assertRaises(ValueError): cold.validate_process_stages(rows)


if __name__ == '__main__':
    unittest.main()
