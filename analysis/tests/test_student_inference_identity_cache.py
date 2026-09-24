"""Mutation and dispatch checks; no model fitting or inference runs here."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from analysis import neural_student_inference_identity_cache as body
from analysis import neural_generalization_inputs as inputs
from analysis import neural_recall_sweep as io


def student_helper():
    spec = importlib.util.spec_from_file_location('cache_test_frozen_student_helper', body.STUDENT_HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def numeric_fixture(helper):
    module = ModuleType('numeric_fixture')
    module.helpers = lambda: (SimpleNamespace(), None, helper)
    module.audit_inference = lambda *args: None
    module.main = lambda: None
    return module


class CacheBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.file = self.root/'array.bin'
        self.file.write_bytes(b'a fixed test input')
        self.reference = body.identity(self.file)

    def test_evidence_and_input_loader_share_one_physical_hash_and_restore(self):
        helper = student_helper(); module = numeric_fixture(helper)
        original = (helper.digest, module.helpers, inputs.sha, io.identity, helper.Evidence.bind)
        cache = body.IdentityCache(); cache.begin()
        with body.installed_student_cache(cache, module):
            _, _, active = module.helpers()
            active.Evidence().bind(self.reference)
            inputs.verified(self.reference)
        inventory = cache.finish()
        self.assertEqual(inventory['statistics']['coldHashCount'], 1)
        self.assertEqual(inventory['statistics']['cacheHitCount'], 1)
        self.assertEqual(inventory['statistics']['coldBytes'], self.file.stat().st_size)
        self.assertEqual((helper.digest, module.helpers, inputs.sha, io.identity, helper.Evidence.bind), original)

    def test_second_stage_is_warm_and_bad_expected_digest_is_still_rejected(self):
        helper = student_helper(); module = numeric_fixture(helper); cache = body.IdentityCache()
        inventories = []
        for _ in range(2):
            cache.begin()
            with body.installed_student_cache(cache, module):
                _, _, active = module.helpers()
                active.Evidence().bind(self.reference)
                inputs.verified(self.reference)
                with self.assertRaises(ValueError):
                    active.Evidence().bind({**self.reference, 'sha256': '0'*64})
            inventories.append(cache.finish())
        self.assertEqual(inventories[1]['statistics']['coldHashCount'], 0)
        self.assertEqual(inventories[0]['files'], inventories[1]['files'])

    def test_failure_restores_declared_bindings(self):
        helper = student_helper(); module = numeric_fixture(helper)
        original = (helper.digest, module.helpers, inputs.sha, io.identity)
        cache = body.IdentityCache(); cache.begin()
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with body.installed_student_cache(cache, module):
                module.helpers()
                raise RuntimeError('deliberate')
        self.assertEqual((helper.digest, module.helpers, inputs.sha, io.identity), original)

    def test_numerical_function_or_evidence_method_replacement_fails(self):
        for mutation in ('numeric', 'evidence'):
            helper = student_helper(); module = numeric_fixture(helper)
            cache = body.IdentityCache(); cache.begin()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                with body.installed_student_cache(cache, module):
                    module.helpers()
                    if mutation == 'numeric': module.audit_inference = lambda *args: 'changed'
                    else: helper.Evidence.bind = lambda *args: 'changed'

    def test_changed_ctime_is_not_accepted_as_same_size_mtime(self):
        cache = body.IdentityCache(); cache.begin(); cache.digest(self.file); cache.finish()
        original = body.signature(self.file); changed = {**original, 'ctimeNs': original['ctimeNs']+1}
        cache.begin()
        with patch('analysis.neural_selection_identity_cache.signature', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'metadata changed'): cache.digest(self.file)

    def test_new_inode_or_after_read_mutation_fails(self):
        cache = body.IdentityCache(); cache.begin(); cache.digest(self.file); cache.finish()
        original = body.signature(self.file); changed = {**original, 'ino': original['ino']+1}
        cache.begin()
        with patch('analysis.neural_selection_identity_cache.signature', return_value=changed):
            with self.assertRaises(ValueError): cache.digest(self.file)
        cache = body.IdentityCache(); cache.begin()
        with patch('analysis.neural_selection_identity_cache.signature', side_effect=[original, original, changed]):
            with self.assertRaisesRegex(ValueError, 'changed while cold hashing'): cache.digest(self.file)


class DispatchTests(unittest.TestCase):
    def command(self):
        return [sys.executable, str(body.NUMERIC), '--phase', 'inference', '--task', '/task.json',
            '--fit', '/fit', '--panel', '/panel.json', '--precision', 'fp32',
            '--fit-audit', '/fit/fit-numerical-audit.json', '--output', '/out.json']

    def test_only_exact_frozen_fp32_inference_cli_is_intercepted(self):
        command = self.command()
        self.assertEqual(body.parse_command(command)['--phase'], 'inference')
        mutations = [command+['--extra', 'x'], command[:-2], [*command[:2], '--phase', 'fit', *command[4:]],
            [*command[:10], '--precision', 'int8', *command[12:]],
            [command[0], '/other.py', *command[2:]], [*command[:4], '--phase', 'inference', *command[6:]]]
        for candidate in mutations:
            with self.subTest(command=candidate), self.assertRaises(ValueError): body.parse_command(candidate)

    def test_failed_process_is_poisoned_and_argv_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'audit.json'; receipt = root/'execution.json'
            command = self.command(); command[-1] = str(output)
            helper = student_helper(); numeric = numeric_fixture(helper)
            def fail():
                numeric.helpers()
                raise RuntimeError('failed numeric replay')
            numeric.main = fail
            executor = object.__new__(body.StudentInferenceAuditCache)
            executor.numeric = numeric; executor.cache = body.IdentityCache(); executor.calls = 0; executor.poisoned = False
            previous = sys.argv; ref = {'path': '/reference', 'sha256': 'x'}
            kwargs = dict(plan_ref=ref, task_ref=ref, panel_ref=ref, fit_audit_ref=ref,
                          receipt_path=receipt, log_path=root/'log.txt')
            with patch.object(body, 'STUDY', root), patch.object(body, 'verify_plan', return_value={}), \
                 patch.object(body, '_invocation', return_value={}), patch.object(body, 'memory_state', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'failed numeric replay'): executor.execute(command, **kwargs)
                self.assertIs(sys.argv, previous)
                self.assertTrue(executor.poisoned)
                self.assertFalse(receipt.exists())
                with self.assertRaisesRegex(ValueError, 'failed cache process'): executor.execute(command, **kwargs)


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.input = self.root/'input.bin'; self.input.write_bytes(b'input')
        self.output = self.root/'audit.json'; self.output.write_text('{}')
        self.companion = self.root/'execution.json'; self.ref = {'path': '/reference', 'sha256': 'fixed'}
        self.command = DispatchTests().command(); self.command[-1] = str(self.output)
        self.payload = {'evidence': [body.identity(self.input)]}
        self.value = {'kind': body.EXECUTION_KIND, 'passed': True, 'plan': self.ref,
            'adapter': body.identity(body.__file__), 'policy': body.POLICY,
            'numericAuditor': body.numeric_source(), 'baseIdentityCache': body.identity(body.BASE_CACHE),
            'studentHelper': body.identity(body.STUDENT_HELPER), 'task': self.ref, 'panel': self.ref,
            'fitAudit': self.ref, 'command': self.command, 'output': body.identity(self.output),
            'persistentCacheImported': False, 'functionBindingsRestored': True,
            'numericalFunctionsAndEvidenceMethodsUnchanged': True, 'all42All4ReplayRetained': True,
            'calibrationOrOutcomeMetricsExecuted': False, 'independentFinalColdClosureStillRequired': True,
            'numericalEvidence': self.payload['evidence'],
            'pid': 1, 'processStageOrdinal': 1, 'processStartTimeTicks': 1234, 'bootId': 'fixed',
            'files': [{**body.identity(self.input), 'stat': body.signature(self.input)}],
            'statistics': {'coldHashCount': 1, 'cacheHitCount': 0, 'coldBytes': 5, 'reusedBytes': 0}}

    def verify(self):
        with patch.object(body, 'verify_plan', return_value={}), patch.object(body, '_invocation', return_value={}), \
             patch.object(body, '_numeric_payload', return_value=self.payload):
            return body.verify_execution(self.companion, plan_ref=self.ref, task_ref=self.ref,
                panel_ref=self.ref, fit_audit_ref=self.ref, output_path=self.output, command=self.command)

    def save(self, value=None):
        self.companion.write_text(json.dumps(self.value if value is None else value))

    def test_original_numeric_output_alone_is_not_cached_execution_proof(self):
        with self.assertRaises(FileNotFoundError): self.verify()
        self.save(); self.assertTrue(self.verify()['passed'])

    def test_changed_output_omitted_evidence_and_relaxed_claims_fail(self):
        cases = [('output', self.ref), ('numericalEvidence', []), ('persistentCacheImported', True),
            ('functionBindingsRestored', False), ('numericAuditor', self.ref),
            ('calibrationOrOutcomeMetricsExecuted', True), ('independentFinalColdClosureStillRequired', False)]
        for key, value in cases:
            changed = copy.deepcopy(self.value); changed[key] = value; self.save(changed)
            with self.subTest(key=key), self.assertRaises(ValueError): self.verify()

    def test_changed_stat_duplicate_file_or_impossible_counters_fail(self):
        for mutation in ('stat', 'duplicate', 'counter', 'too-many-cold-bytes', 'cold-bytes-without-read'):
            changed = copy.deepcopy(self.value)
            if mutation == 'stat': changed['files'][0]['stat']['ctimeNs'] += 1
            if mutation == 'duplicate': changed['files'].append(changed['files'][0])
            if mutation == 'counter': changed['statistics']['coldHashCount'] = 2
            if mutation == 'too-many-cold-bytes': changed['statistics']['coldBytes'] = 6
            if mutation == 'cold-bytes-without-read':
                changed['statistics'].update(coldHashCount=0, cacheHitCount=1)
            self.save(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.verify()


class QualificationTests(unittest.TestCase):
    def test_exact_control_source_cold_warm_and_process_identity_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fit = root/'fit-numerical-audit.json'; fit.write_text('{}')
            ref = body.identity(fit); source = body.identity(body.REPO/'scripts/student-inference-identity-cache.py')
            target = {'fitDirectory': str(root), 'task': ref, 'panel': ref, 'outputPath': str(root/'audit.json')}
            plan = {'sources': {'scripts/student-inference-identity-cache.py': source},
                'pythonExecutable': sys.executable,
                'qualification': {'controlAudit': ref, 'targets': [target, target]}}
            proof = {'kind': body.QUALIFICATION_KIND, 'passed': True, 'plan': ref,
                'adapter': body.identity(body.__file__), 'source': source, 'controlAudit': ref,
                'executions': [ref, ref], 'byteExactColdParity': True, 'byteExactWarmParity': True,
                'productionOutputsModified': False, 'calibrationOrOutcomeMetricsExecuted': False}
            stages = [{'output': ref, 'statistics': {'coldHashCount': 1, 'cacheHitCount': 1},
                'files': ['same'], 'pid': 1, 'bootId': 'same', 'processStartTimeTicks': 1234, 'processStageOrdinal': 1},
                {'output': ref, 'statistics': {'coldHashCount': 0, 'cacheHitCount': 2},
                'files': ['same'], 'pid': 1, 'bootId': 'same', 'processStartTimeTicks': 1234, 'processStageOrdinal': 2}]
            path = root/'proof.json'
            for mutation in ('none', 'source', 'control', 'warm-cold', 'inventory', 'pid', 'boot', 'start', 'ordinal', 'missing'):
                candidate = copy.deepcopy(proof); replay = copy.deepcopy(stages)
                if mutation == 'source': candidate['source'] = ref
                if mutation == 'control': replay[1]['output'] = {**ref, 'sha256': '0'*64}
                if mutation == 'warm-cold': replay[1]['statistics']['coldHashCount'] = 1
                if mutation == 'inventory': replay[1]['files'] = ['different']
                if mutation == 'pid': replay[1]['pid'] = 2
                if mutation == 'boot': replay[1]['bootId'] = 'another'
                if mutation == 'start': replay[1]['processStartTimeTicks'] += 1
                if mutation == 'ordinal': replay[1]['processStageOrdinal'] = 1
                if mutation == 'missing': candidate['executions'] = [ref]
                path.write_text(json.dumps(candidate))
                with self.subTest(mutation=mutation), patch.object(body, 'verify_plan', return_value=plan), \
                     patch.object(body, 'verify_execution', side_effect=replay):
                    if mutation == 'none': self.assertTrue(body.verify_qualification(body.identity(path), ref)['passed'])
                    else:
                        with self.assertRaises(ValueError): body.verify_qualification(body.identity(path), ref)


if __name__ == '__main__': unittest.main()
