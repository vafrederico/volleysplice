"""Disclosed hash reuse around the unchanged, CPU-only student inference auditor.

The cache is born empty in this process. No numeric function, Evidence method,
array loader, tolerance or saved audit payload is replaced.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
STUDY = Path(private_value('private-reference-0061'))
BASE_CACHE = REPO/'analysis/neural_selection_identity_cache.py'
BASE_CACHE_SHA = 'b3c5f281cebe440649e35930816cdc5257ad99699731432dec2b1d61d619d4d0'
NUMERIC = REPO/'scripts/audit-neural-generalization-numerics.py'
NUMERIC_SHA = '6714e430a89f91e24e174f44eeed49c741d61100013a19005ae6423fd155efcd'
STUDENT_HELPER = REPO/'scripts/audit-neural-mobile-distillation.py'
STUDENT_HELPER_SHA = '090167d06859469be5ae1d06ee7eddcb7a380c28f2d74d6838a967ac4b09000e'
PLAN_KIND = 'registered-student-inference-identity-cache-execution-v1'
EXECUTION_KIND = 'explicit-student-inference-identity-cache-execution-v1'
QUALIFICATION_KIND = 'student-inference-identity-cache-byte-exact-qualification-v1'
COMPANION_NAME = 'student-inference-cache-execution-fp32.json'
POLICY = {
    'kind': 'student-inference-process-local-identity-cache-v1',
    'baseCacheBindings': ['analysis.neural_recall_sweep.identity',
        'analysis.neural_recall_sweep_adapters.identity', 'analysis.neural_generalization_inputs.sha'],
    'additionalHashBinding': 'fresh audit-neural-mobile-distillation module.digest',
    'helperFactoryDispatchBinding': 'frozen numeric module.helpers delegates original then installs only digest',
    'keyFields': ['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'],
    'firstProcessReadColdHashed': True, 'persistentCacheImportAllowed': False,
    'prePostHashStatCheck': True, 'changedMetadataRehashThenFail': True,
    'allNumericalFunctionsAndEvidenceMethodsUnchanged': True,
    'frozenNumericMainAndArguments': True, 'torchCpuThreads': 2,
    'onlyStudentFp32InferenceAudits': True,
    'executionCompanionRequiredForNewAudits': True,
    'independentFinalColdClosureBeforePublication': True,
    'calibrationOrOutcomeMetricsExecuted': False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def verified(reference):
    require(set(reference) == {'path', 'sha256'} and identity(reference['path']) == reference,
            'Bound execution identity changed: '+str(reference.get('path')))
    return Path(reference['path'])


require(identity(BASE_CACHE)['sha256'] == BASE_CACHE_SHA, 'Original identity-cache source changed')
from analysis.neural_selection_identity_cache import IdentityCache, installed, signature


def numeric_source():
    reference = identity(NUMERIC)
    require(reference['sha256'] == NUMERIC_SHA, 'Frozen numeric auditor changed')
    return reference


def source_contract():
    numeric_source()
    require(identity(BASE_CACHE)['sha256'] == BASE_CACHE_SHA
            and identity(STUDENT_HELPER)['sha256'] == STUDENT_HELPER_SHA, 'Frozen hash/numeric helper changed')
    names = ['analysis/neural_student_inference_identity_cache.py',
        'analysis/neural_selection_identity_cache.py',
        'scripts/student-inference-identity-cache.py',
        'scripts/audit-neural-generalization-numerics.py',
        'scripts/audit-neural-mobile-distillation.py',
        'analysis/tests/test_student_inference_identity_cache.py',
        'docs/research/neural-student-inference-identity-cache-2026-09-23.md']
    return {name: identity(REPO/name) for name in names}


def _target(row, panel, output=None, receipt=None):
    folder = Path(row['fitDirectory'])
    return {'task': row['task'], 'taskId': row['taskId'], 'fitDirectory': str(folder),
        'panel': panel, 'outputPath': str(output or folder/'inference-numerical-audit-fp32.json'),
        'companionPath': str(receipt or folder/COMPANION_NAME)}


def make_plan(student_plan_ref, qualification_root):
    source = read(verified(student_plan_ref)); rows = source['tasks']
    require(len(rows) == 27 and len({r['taskId'] for r in rows}) == 27
            and len({r['physicalOwnerTaskId'] for r in rows}) == 20,
            'Complete registered student population required')
    root = Path(qualification_root).resolve()
    require(root.is_relative_to(STUDY), 'Qualification must stay on study NAS')
    first = rows[0]
    require(first['taskId'] == 'original-corpus/distilled-mobile-tcn/seed-3407', 'Fixed qualification control changed')
    control = Path(first['fitDirectory'])/'inference-numerical-audit-fp32.json'
    return {'kind': PLAN_KIND, 'policy': POLICY, 'sources': source_contract(),
        'studentPlan': student_plan_ref, 'pythonExecutable': sys.executable,
        'targets': [_target(r, source['panel']) for r in rows],
        'qualification': {'controlAudit': identity(control), 'outputRoot': str(root),
            'targets': [_target(first, source['panel'], root/(phase+'-audit.json'),
                        root/(phase+'-execution.json')) for phase in ('cold', 'warm')]},
        'preexistingClassificationOwnedByQueuePlan': True,
        'immutableSourcesAndOutputPopulation': True}


def verify_plan(plan_ref):
    plan = read(verified(plan_ref))
    require(plan['kind'] == PLAN_KIND and plan['policy'] == POLICY
            and plan['sources'] == source_contract(), 'Numeric cache execution plan/source differs')
    expected = make_plan(plan['studentPlan'], plan['qualification']['outputRoot'])
    require(plan == expected, 'Numeric cache task/control population changed')
    for target in plan['targets'] + plan['qualification']['targets']:
        task = read(verified(target['task']))
        require(task['taskId'] == target['taskId'] and task['model'] == 'distilled-mobile-tcn',
                'Numeric cache target is not its registered student task')
        verified(target['panel'])
    return plan


def parse_command(command):
    require(isinstance(command, (list, tuple)) and len(command) == 16
            and all(isinstance(x, str) for x in command), 'Exact numeric inference argv required')
    require(Path(command[1]).resolve() == NUMERIC.resolve(), 'Only frozen numeric auditor may be intercepted')
    flags = command[2::2]; values = command[3::2]
    expected = {'--phase', '--task', '--fit', '--panel', '--precision', '--fit-audit', '--output'}
    require(len(flags) == len(set(flags)) and set(flags) == expected,
            'Unknown, duplicate or incomplete numeric inference arguments')
    args = dict(zip(flags, values, strict=True))
    require(args['--phase'] == 'inference' and args['--precision'] == 'fp32',
            'Only FP32 inference auditing is authorized')
    return args


def command_for(target, fit_audit, executable=None):
    return [executable or sys.executable, str(NUMERIC), '--phase', 'inference',
        '--task', target['task']['path'], '--fit', target['fitDirectory'],
        '--panel', target['panel']['path'], '--precision', 'fp32',
        '--fit-audit', str(fit_audit), '--output', target['outputPath']]


def _invocation(plan, command, task_ref, panel_ref, fit_audit_ref, output_path, receipt_path):
    args = parse_command(command)
    require(command[0] == plan['pythonExecutable'], 'Numeric Python executable differs')
    targets = [t for t in plan['targets'] + plan['qualification']['targets']
               if t['outputPath'] == str(output_path)]
    require(len(targets) == 1, 'Numeric output is outside registered targets')
    target = targets[0]
    require(target['task'] == task_ref and target['panel'] == panel_ref
            and target['companionPath'] == str(receipt_path)
            and args == {'--phase': 'inference', '--task': task_ref['path'],
                '--fit': target['fitDirectory'], '--panel': panel_ref['path'],
                '--precision': 'fp32', '--fit-audit': fit_audit_ref['path'],
                '--output': str(output_path)}, 'Numeric execution ownership/arguments differ')
    require(Path(fit_audit_ref['path']) == Path(target['fitDirectory'])/'fit-numerical-audit.json',
            'Numeric execution fit-audit path differs')
    for reference in (task_ref, panel_ref, fit_audit_ref): verified(reference)
    fit = read(fit_audit_ref['path'])
    require(fit['kind'] == 'independent-generalization-fit-numerical-audit-v1'
            and fit['passed'] is True and fit['task'] == task_ref and fit['auditor'] == numeric_source(),
            'Passing frozen fit numerical audit required')
    return target


def _functions(module):
    return {k: v for k, v in vars(module).items() if inspect.isfunction(v)}


def _evidence_methods(cls):
    return {k: v for k, v in vars(cls).items() if inspect.isfunction(v)}


@contextmanager
def installed_student_cache(cache, numeric):
    """The sole extra dispatch hook delegates the frozen helper factory first."""
    before = _functions(numeric); original_helpers = numeric.helpers; created = []

    def bound_helpers():
        old, tensor, student = original_helpers()
        require(Path(student.__file__).resolve() == STUDENT_HELPER.resolve()
                and identity(student.__file__)['sha256'] == STUDENT_HELPER_SHA,
                'The numeric helper factory returned another student auditor')
        saved = _functions(student); cls = student.Evidence; methods = _evidence_methods(cls)
        original_digest = student.digest; student.digest = cache.digest
        created.append((student, saved, cls, methods, original_digest))
        return old, tensor, student

    numeric.helpers = bound_helpers
    problem = None
    try:
        with installed(cache):
            yield
    finally:
        try:
            require(numeric.helpers is bound_helpers, 'Declared helper dispatch binding changed')
            require(all(vars(numeric).get(k) is v for k, v in before.items() if k != 'helpers'),
                    'A frozen numeric function was replaced')
            for student, saved, cls, methods, original_digest in created:
                require(student.digest == cache.digest and student.Evidence is cls
                        and _evidence_methods(cls) == methods
                        and all(vars(student).get(k) is v for k, v in saved.items() if k != 'digest'),
                        'A student numeric function or Evidence method was replaced')
        except BaseException as error:
            problem = error
        finally:
            numeric.helpers = original_helpers
            for student, _, _, _, original_digest in created: student.digest = original_digest
        if problem is not None: raise problem


def memory_state():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:')):
            result[line.split(':')[0]+'Bytes'] = int(line.split()[1])*1024
    return result


def _numeric_payload(path, task_ref, panel_ref, fit_audit_ref):
    value = read(path); panel = read(panel_ref['path']); identifiers = panel['recordingIds']
    require(value['kind'] == 'independent-generalization-inference-numerical-audit-v1'
            and value['passed'] is True and value['task'] == task_ref
            and value['panel'] == panel_ref and value['fitAudit'] == fit_audit_ref
            and value['precision'] == 'fp32' and value['auditor'] == numeric_source()
            and value['checkpointEpochs'] == [5, 15, 30, 60]
            and value['trainingPerformed'] is False and value['gpuUsed'] is False,
            'Delegated frozen numerical audit identity differs')
    require(len(identifiers) == 42 and len(set(identifiers)) == 42
            and [(r['recordingId'], r['epoch']) for r in value['cpuReplay']]
                == [(key, epoch) for key in identifiers for epoch in (5, 15, 30, 60)]
            and [r['id'] for r in value['studentFeatureReplay']] == identifiers
            and len(value['inferenceReceipts']) == 42, 'Full42/all4 numerical replay is required')
    return value


def _write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


class StudentInferenceAuditCache:
    def __init__(self):
        numeric_source()
        spec = importlib.util.spec_from_file_location('explicit_cached_frozen_student_numeric', NUMERIC)
        self.numeric = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.numeric)
        require(inspect.getsourcefile(self.numeric.main) == str(NUMERIC)
                and inspect.getsourcefile(self.numeric.audit_inference) == str(NUMERIC),
                'Frozen numeric entry points changed')
        self.cache = IdentityCache(); self.calls = 0; self.poisoned = False

    def execute(self, command, *, plan_ref, task_ref, panel_ref, fit_audit_ref, receipt_path, log_path):
        require(not self.poisoned, 'A failed cache process may not continue')
        plan = verify_plan(plan_ref); args = parse_command(command); output = Path(args['--output'])
        receipt_path = Path(receipt_path); log_path = Path(log_path)
        _invocation(plan, command, task_ref, panel_ref, fit_audit_ref, output, receipt_path)
        require(all(p.resolve().is_relative_to(STUDY) for p in (output, receipt_path, log_path)),
                'All audit output/log/companions must be on study NAS')
        require(not output.exists() and not receipt_path.exists(), 'Existing/partial audit must not be overwritten')
        receipt_path.parent.mkdir(parents=True, exist_ok=True); log_path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat(); wall = time.monotonic(); before = memory_state()
        old_argv = sys.argv; before_functions = _functions(self.numeric); self.cache.begin()
        try:
            with log_path.open('a') as log:
                with redirect_stdout(log), redirect_stderr(log), installed_student_cache(self.cache, self.numeric):
                    sys.argv = list(command[1:])
                    self.numeric.main()
            sys.argv = old_argv
            require(_functions(self.numeric) == before_functions, 'Numeric functions did not restore exactly')
            inventory = self.cache.finish()
            value = _numeric_payload(output, task_ref, panel_ref, fit_audit_ref)
            self.calls += 1; gc.collect()
            companion = {'kind': EXECUTION_KIND, 'passed': True, 'plan': plan_ref,
                'adapter': identity(__file__), 'policy': POLICY, 'numericAuditor': numeric_source(),
                'baseIdentityCache': identity(BASE_CACHE), 'studentHelper': identity(STUDENT_HELPER),
                'task': task_ref, 'panel': panel_ref, 'fitAudit': fit_audit_ref,
                'command': list(command), 'output': identity(output),
                'numericalEvidence': value['evidence'], **inventory,
                'pid': os.getpid(), 'processStageOrdinal': self.calls,
                'bootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'processStartTimeTicks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                'persistentCacheImported': False, 'functionBindingsRestored': True,
                'numericalFunctionsAndEvidenceMethodsUnchanged': True,
                'all42All4ReplayRetained': True, 'calibrationOrOutcomeMetricsExecuted': False,
                'startedAtUTC': started, 'finishedAtUTC': datetime.now(timezone.utc).isoformat(),
                'wallSeconds': time.monotonic()-wall, 'memoryBefore': before,
                'memoryAfterRelease': memory_state(),
                'memoryInterpretation': 'RSS is after GC; HWM is process lifetime, not a per-stage peak.',
                'independentFinalColdClosureStillRequired': True}
            _write_new(receipt_path, companion)
            verify_execution(receipt_path, plan_ref=plan_ref, task_ref=task_ref, panel_ref=panel_ref,
                fit_audit_ref=fit_audit_ref, output_path=output, command=command)
            return identity(receipt_path)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            sys.argv = old_argv


def verify_execution(companion_path, *, plan_ref, task_ref, panel_ref, fit_audit_ref, output_path, command):
    """Validate disclosure/stat closure; independent final physical rehash is separate."""
    plan = verify_plan(plan_ref); companion_path = Path(companion_path); output_path = Path(output_path)
    _invocation(plan, command, task_ref, panel_ref, fit_audit_ref, output_path, companion_path)
    value = read(companion_path)
    require(value['kind'] == EXECUTION_KIND and value['passed'] is True
            and value['plan'] == plan_ref and value['adapter'] == identity(__file__)
            and value['policy'] == POLICY and value['numericAuditor'] == numeric_source()
            and value['baseIdentityCache'] == identity(BASE_CACHE)
            and value['studentHelper'] == identity(STUDENT_HELPER)
            and value['task'] == task_ref and value['panel'] == panel_ref
            and value['fitAudit'] == fit_audit_ref and value['command'] == list(command)
            and value['output'] == identity(output_path), 'Numeric cache execution companion differs')
    require(value['persistentCacheImported'] is False
            and value['functionBindingsRestored'] is True
            and value['numericalFunctionsAndEvidenceMethodsUnchanged'] is True
            and value['all42All4ReplayRetained'] is True
            and value['calibrationOrOutcomeMetricsExecuted'] is False
            and value['independentFinalColdClosureStillRequired'] is True,
            'Numeric cache execution policy was relaxed')
    payload = _numeric_payload(output_path, task_ref, panel_ref, fit_audit_ref)
    require(value['numericalEvidence'] == payload['evidence'], 'Named numeric evidence closure differs')
    require(all(type(value[k]) is int and value[k] > 0
                for k in ('pid', 'processStageOrdinal', 'processStartTimeTicks'))
            and isinstance(value['bootId'], str) and bool(value['bootId']), 'Invalid execution process identity')
    files = value['files']; paths = [r['path'] for r in files]
    require(paths == sorted(set(paths)) and all(set(r) == {'path', 'sha256', 'stat'} for r in files),
            'Cache touched-file inventory is duplicate or unordered')
    for row in files:
        require(isinstance(row['sha256'], str) and len(row['sha256']) == 64
                and set(row['sha256']) <= set('0123456789abcdef')
                and signature(row['path']) == row['stat'], 'Cached immutable file metadata changed')
    stats = value['statistics']
    require(set(stats) == {'coldHashCount', 'cacheHitCount', 'coldBytes', 'reusedBytes'}
            and all(type(x) is int and x >= 0 for x in stats.values())
            and stats['coldHashCount'] <= len(files) <= stats['coldHashCount']+stats['cacheHitCount']
            and stats['coldBytes'] <= sum(r['stat']['size'] for r in files)
            and (stats['coldHashCount'] != 0 or stats['coldBytes'] == 0)
            and stats['coldBytes']+stats['reusedBytes'] >= sum(r['stat']['size'] for r in files),
            'Cache counters/touched closure are inconsistent')
    return value


def verify_qualification(qualification_ref, plan_ref):
    plan = verify_plan(plan_ref); proof = read(verified(qualification_ref))
    require(proof['kind'] == QUALIFICATION_KIND and proof['passed'] is True
            and proof['plan'] == plan_ref and proof['adapter'] == identity(__file__)
            and proof['source'] == plan['sources']['scripts/student-inference-identity-cache.py']
            and proof['controlAudit'] == plan['qualification']['controlAudit']
            and proof['byteExactColdParity'] is True and proof['byteExactWarmParity'] is True
            and proof['productionOutputsModified'] is False
            and proof['calibrationOrOutcomeMetricsExecuted'] is False,
            'Passing byte-exact cold/warm numeric qualification required')
    require(len(proof['executions']) == 2, 'Both qualification stages are required')
    stages = []
    for target, reference in zip(plan['qualification']['targets'], proof['executions'], strict=True):
        verified(reference); fit_ref = identity(Path(target['fitDirectory'])/'fit-numerical-audit.json')
        command = command_for(target, fit_ref['path'], plan['pythonExecutable'])
        stage = verify_execution(reference['path'], plan_ref=plan_ref, task_ref=target['task'],
            panel_ref=target['panel'], fit_audit_ref=fit_ref,
            output_path=target['outputPath'], command=command)
        require(stage['output']['sha256'] == proof['controlAudit']['sha256'], 'Delegated audit bytes differ')
        stages.append(stage)
    require(stages[0]['statistics']['coldHashCount'] > 0
            and stages[1]['statistics']['coldHashCount'] == 0
            and stages[1]['statistics']['cacheHitCount'] > 0
            and stages[0]['files'] == stages[1]['files']
            and stages[0]['pid'] == stages[1]['pid']
            and stages[0]['bootId'] == stages[1]['bootId']
            and stages[0]['processStartTimeTicks'] == stages[1]['processStartTimeTicks']
            and [s['processStageOrdinal'] for s in stages] == [1, 2],
            'Qualification does not prove cold then warm process-local execution')
    return proof
