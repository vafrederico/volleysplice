#!/usr/bin/env python3
"""Independently cold-hash all27 student audits and declared cached execution evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO/'scripts/audit-neural-selection-identity-cache-v2.py'
HELPER_SHA = 'c88e6608058a3ba051eed7b21ca8ae42ed8c7a7b08f88964f50946104328bfd0'
NUMERICAL = REPO/'scripts/audit-neural-generalization-numerics.py'
NUMERICAL_SHA = '6714e430a89f91e24e174f44eeed49c741d61100013a19005ae6423fd155efcd'
KIND = 'independent-student-inference-cache-cold-audit-v1'
STAT_KEYS = {'dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'}
NUMERICAL_REFERENCE_FIELDS = ('task', 'fitResult', 'completed', 'panel', 'fitAudit',
    'inferenceReceipts', 'cpuReplay', 'studentFeatureReplay', 'evidence', 'auditor')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def signature(path):
    value = Path(path).stat()
    return {'dev': value.st_dev, 'ino': value.st_ino, 'size': value.st_size,
            'mtimeNs': value.st_mtime_ns, 'ctimeNs': value.st_ctime_ns}


def identity(path):
    path = Path(path); before = signature(path); result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    require(signature(path) == before, 'File changed during independent hash: '+str(path))
    return {'path': str(path), 'sha256': result.hexdigest()}


def helper():
    require(identity(HELPER)['sha256'] == HELPER_SHA, 'Independent cold hashing helper changed')
    spec = importlib.util.spec_from_file_location('independent_student_cold_hashing_helper', HELPER)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def classifications(plan):
    require(plan['kind'] == 'registered-streaming-student-panel-queue-v3', 'Wrong student queue plan')
    tasks = plan['tasks']; rows = plan['numericAuditClassification']
    require(len(tasks) == len(rows) == 27 and len({row['taskId'] for row in tasks}) == 27,
            'Exactly27 distinct registered student tasks required')
    for task, row in zip(tasks, rows, strict=True):
        require(all(row[key] == task[key] for key in ('taskId', 'task', 'fitDirectory')),
                'Student task classification order or owner differs')
        expected = str(Path(row['fitDirectory'])/'inference-numerical-audit-fp32.json')
        require(row['numericAuditPath'] == expected, 'Wrong numerical audit path')
        if row['mode'] == 'preexisting-uncached':
            require(isinstance(row['numericAudit'], dict) and row['numericAudit']['path'] == expected
                and isinstance(row['preexistingCompletion'], dict) and row['executionCompanion'] is None,
                'Incomplete preexisting uncached audit binding')
        elif row['mode'] == 'cached-required':
            require(row['numericAudit'] is None and row['preexistingCompletion'] is None
                and row['executionCompanion'] == str(Path(row['fitDirectory'])/'student-inference-cache-execution-fp32.json'),
                'Missing cached-required execution path or incorrect preexisting claim')
        else:
            raise ValueError('Unknown student numeric execution classification')
    return rows


def numerical_scope(value, row, panel_ref, identifiers):
    require(value['kind'] == 'independent-generalization-inference-numerical-audit-v1'
        and value['passed'] is True and value['taskId'] == row['taskId'] and value['task'] == row['task']
        and value['panel'] == panel_ref and value['precision'] == 'fp32'
        and value['checkpointEpochs'] == [5, 15, 30, 60]
        and value['trainingPerformed'] is False and value['gpuUsed'] is False
        and value['auditor'] == {'path': str(NUMERICAL), 'sha256': NUMERICAL_SHA},
        'Wrong independent student numerical scope')
    folder = Path(row['fitDirectory'])
    require(value['fitResult']['path'] == str(folder/'fit-result.json')
        and value['fitAudit']['path'] == str(folder/'fit-numerical-audit.json')
        and value['completed']['path'] == str(folder/'temporal/completed.json'), 'Numerical fit owner differs')
    output = folder/'inference'/panel_ref['sha256'][:16]/'fp32'
    require([ref['path'] for ref in value['inferenceReceipts']] == [str(output/(key+'.json')) for key in identifiers]
        and [(item['recordingId'], item['epoch']) for item in value['cpuReplay']]
            == [(key, epoch) for key in identifiers for epoch in (5, 15, 30, 60)]
        and [item['id'] for item in value['studentFeatureReplay']] == identifiers,
        'Missing full42/all4 or sampled student feature replay')
    require(all(key in value for key in NUMERICAL_REFERENCE_FIELDS), 'Missing named numerical evidence')


def completed_companions(plan_ref, completed, rows):
    require(completed['kind'] == 'streaming-student-full-panel-completed-v3'
        and completed['passed'] is True and completed['plan'] == plan_ref
        and completed['logicalTaskCount'] == 27
        and completed['all27Full42All4NumericAndApplicableReuseGatesPassed'] is True,
        'Complete v3 queue gate required')
    required = [row for row in rows if row['mode'] == 'cached-required']
    bindings = completed['numericExecutionCompanions']
    require([item['taskId'] for item in bindings] == [row['taskId'] for row in required]
        and all(item['companion']['path'] == row['executionCompanion'] for item, row in zip(bindings, required, strict=True)),
        'Completed queue has missing, reordered or unexpected execution companions')
    return {item['taskId']: item['companion'] for item in bindings}


def require_covered(tree, by_path, cold):
    for reference in cold.references(tree):
        key = str(cold.resolve_reference(reference['path']))
        require(key in by_path and by_path[key]['sha256'] == reference['sha256'],
                'Required evidence is absent from the cold closure: '+key)


def validate_process_stages(stages):
    groups = {}
    for value in stages:
        key = (value['bootId'], value['pid'], value['processStartTimeTicks'])
        groups.setdefault(key, []).append(value)
    for values in groups.values():
        ordered = sorted(values, key=lambda item: item['processStageOrdinal'])
        require([item['processStageOrdinal'] for item in ordered] == list(range(1, len(ordered)+1))
            and ordered[0]['statistics']['coldHashCount'] > 0,
            'Cached process stages are missing, duplicated or did not start cold')


def verify_publication(cold_path, queue_plan_path):
    """Small hashes plus the independently established complete immutable stat closure."""
    gate_ref = identity(cold_path); gate = read(cold_path)
    plan_ref = identity(queue_plan_path); plan = read(queue_plan_path); rows = classifications(plan)
    require(gate['kind'] == KIND and gate['passed'] is True and gate['source'] == identity(__file__)
        and gate['coldHelper'] == identity(HELPER) and gate['coldHelper']['sha256'] == HELPER_SHA
        and gate['numericalAuditor'] == identity(NUMERICAL) and gate['numericalAuditor']['sha256'] == NUMERICAL_SHA
        and gate['queuePlan'] == plan_ref and gate['logicalTaskCount'] == 27
        and gate['coldCacheImported'] is False and gate['allNamedNumericalEvidenceRehashed'] is True
        and gate['allTouchedEvidenceRehashed'] is True, 'Complete independent student cold gate required')
    require(identity(gate['queueCompleted']['path']) == gate['queueCompleted'], 'Queue completion changed')
    bindings = completed_companions(plan_ref, read(gate['queueCompleted']['path']), rows)
    require([item['taskId'] for item in gate['taskChecks']] == [row['taskId'] for row in rows],
            'Cold gate task population differs')
    for row, checked in zip(rows, gate['taskChecks'], strict=True):
        require(checked['mode'] == row['mode'] and checked['numericAudit'] == identity(row['numericAuditPath']),
                'Numerical audit changed after cold verification')
        if row['mode'] == 'preexisting-uncached':
            require(checked['numericAudit'] == row['numericAudit'] and checked['executionCompanion'] is None,
                    'Preexisting audit classification changed')
        else:
            require(checked['executionCompanion'] == bindings[row['taskId']]
                and identity(row['executionCompanion']) == checked['executionCompanion'],
                'Cached execution companion changed or missing')
    files = gate['checkedFiles']
    require(files and len(files) == gate['fileCount'] and len({item['path'] for item in files}) == len(files),
            'Incomplete or duplicate cold file inventory')
    cold = helper(); by_path = {item['path']: item for item in files}

    def covered(tree):
        require_covered(tree, by_path, cold)

    covered({'source': gate['source'], 'coldHelper': gate['coldHelper'],
             'numericalAuditor': gate['numericalAuditor'], 'plan': plan_ref,
             'completed': gate['queueCompleted']})
    covered(gate['previousV2StopEvidence'])
    covered(plan); covered(read(gate['queueCompleted']['path']))
    execution = read(plan['numericExecutionPlan']['path']); covered(execution)
    qualification = read(plan['numericExecutionQualification']['path']); covered(qualification)
    for reference in qualification['executions']:
        stage = read(reference['path']); covered(stage)
        covered(read(stage['output']['path'])['evidence'])
    for row in rows:
        numeric = read(row['numericAuditPath'])
        covered({key: numeric[key] for key in NUMERICAL_REFERENCE_FIELDS})
        if row['mode'] == 'cached-required':
            covered(read(row['executionCompanion']))
        else:
            covered(read(row['preexistingCompletion']['path']))
    for item in files:
        require(set(item['stat']) == STAT_KEYS and signature(item['path']) == item['stat'],
                'Cold-verified immutable file metadata changed: '+item['path'])
    require(identity(cold_path) == gate_ref and identity(queue_plan_path) == plan_ref, 'Publication gate changed during inspection')
    return {'coldAudit': gate_ref, 'queuePlan': plan_ref, 'queueCompleted': gate['queueCompleted'],
            'logicalTaskCount': 27, 'cachedTaskCount': len(bindings), 'all27PublicationGatesPassed': True}


def validate_companion(value, *, execution_plan, execution_plan_ref, adapter_ref,
                       row, panel_ref, audit_ref, numeric, output_path=None):
    output_path = str(output_path or row['numericAuditPath'])
    sources = execution_plan['sources']
    require(value['kind'] == 'explicit-student-inference-identity-cache-execution-v1'
        and value['passed'] is True and value['plan'] == execution_plan_ref
        and value['adapter'] == adapter_ref and value['policy'] == execution_plan['policy']
        and value['numericAuditor'] == {'path': str(NUMERICAL), 'sha256': NUMERICAL_SHA}
        and value['baseIdentityCache'] == sources['analysis/neural_selection_identity_cache.py']
        and value['studentHelper'] == sources['scripts/audit-neural-mobile-distillation.py']
        and value['task'] == row['task'] and value['panel'] == panel_ref
        and value['fitAudit'] == numeric['fitAudit'] and value['output'] == audit_ref
        and value['numericalEvidence'] == numeric['evidence'], 'Cached execution ownership/evidence differs')
    policy = value['policy']
    require(policy['kind'] == 'student-inference-process-local-identity-cache-v1'
        and policy['baseCacheBindings'] == ['analysis.neural_recall_sweep.identity',
            'analysis.neural_recall_sweep_adapters.identity', 'analysis.neural_generalization_inputs.sha']
        and policy['additionalHashBinding'] == 'fresh audit-neural-mobile-distillation module.digest'
        and policy['helperFactoryDispatchBinding'] == 'frozen numeric module.helpers delegates original then installs only digest'
        and policy['keyFields'] == ['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs']
        and policy['torchCpuThreads'] == 2 and policy['persistentCacheImportAllowed'] is False
        and policy['calibrationOrOutcomeMetricsExecuted'] is False,
        'Unexpected hash binding, process or numerical policy')
    for key in ('firstProcessReadColdHashed', 'prePostHashStatCheck', 'changedMetadataRehashThenFail',
                'allNumericalFunctionsAndEvidenceMethodsUnchanged', 'frozenNumericMainAndArguments',
                'onlyStudentFp32InferenceAudits', 'executionCompanionRequiredForNewAudits',
                'independentFinalColdClosureBeforePublication'):
        require(policy[key] is True, 'Relaxed cache execution policy: '+key)
    for key in ('functionBindingsRestored', 'numericalFunctionsAndEvidenceMethodsUnchanged',
                'all42All4ReplayRetained', 'independentFinalColdClosureStillRequired'):
        require(value[key] is True, 'Incomplete cache execution proof: '+key)
    require(value['persistentCacheImported'] is False and value['calibrationOrOutcomeMetricsExecuted'] is False,
            'Undeclared cache import or outcome access')
    require(type(value['pid']) is int and value['pid'] > 0
        and type(value['processStageOrdinal']) is int and value['processStageOrdinal'] > 0
        and type(value['processStartTimeTicks']) is int and value['processStartTimeTicks'] > 0
        and isinstance(value['bootId'], str) and value['bootId'], 'Incomplete cached process identity')
    command = value['command']
    require(isinstance(command, list) and len(command) == 16
        and command[:2] == [execution_plan['pythonExecutable'], str(NUMERICAL)], 'Wrong numeric executable/command')
    flags = command[2::2]
    require(len(flags) == len(set(flags)) and dict(zip(flags, command[3::2], strict=True)) == {
        '--phase': 'inference', '--task': row['task']['path'], '--fit': row['fitDirectory'],
        '--panel': panel_ref['path'], '--precision': 'fp32', '--fit-audit': numeric['fitAudit']['path'],
        '--output': output_path}, 'Cached numeric invocation changed')
    files = value['files']
    require(files and [item['path'] for item in files] == sorted({item['path'] for item in files}),
            'Missing, duplicate or unordered touched-file inventory')
    for item in files:
        require(set(item) == {'path', 'sha256', 'stat'} and set(item['stat']) == STAT_KEYS
            and all(type(v) is int and v >= 0 for v in item['stat'].values()), 'Malformed immutable file signature')
    stats = value['statistics']
    require(set(stats) == {'coldHashCount', 'cacheHitCount', 'coldBytes', 'reusedBytes'}
        and all(type(v) is int and v >= 0 for v in stats.values())
        and stats['coldHashCount'] <= len(files) <= stats['coldHashCount']+stats['cacheHitCount']
        and stats['coldBytes'] <= sum(item['stat']['size'] for item in files)
        and (stats['coldHashCount'] != 0 or stats['coldBytes'] == 0)
        and stats['coldBytes']+stats['reusedBytes'] >= sum(item['stat']['size'] for item in files),
        'Inconsistent cache statistics')
    return files


def audit(args):
    require(Path('/mnt/freenas').is_mount() and args.output.resolve().is_relative_to('/mnt/freenas')
        and not args.output.exists(), 'New direct NAS cold audit output required')
    cold = helper(); evidence = cold.ColdEvidence()
    own_ref = evidence.capture(Path(__file__).resolve())
    helper_ref = evidence.capture(HELPER); numerical_ref = evidence.capture(NUMERICAL)
    require(numerical_ref['sha256'] == NUMERICAL_SHA, 'Frozen numerical source changed')
    plan_ref = evidence.capture(args.queue_plan); plan = evidence.document(plan_ref)
    rows = classifications(plan)
    completed_ref = evidence.capture(args.queue_completed); completed = evidence.document(completed_ref)
    companions = completed_companions(plan_ref, completed, rows)
    snapshot = evidence.document(plan['classificationSnapshot'])
    require(snapshot['kind'] == 'student-numeric-execution-classification-v3'
        and snapshot['previousV2Plan'] == plan['previousV2Plan']
        and snapshot['previousV2Exit'] == plan['previousV2Exit']
        and snapshot['previousV2Output'] == plan['previousV2Output']
        and snapshot['numericAuditClassification'] == rows
        and snapshot['preexistingCount'] == sum(row['mode'] == 'preexisting-uncached' for row in rows)
        and snapshot['partialUnreceiptedOutputsPresent'] is False
        and completed['classificationSnapshot'] == plan['classificationSnapshot']
        and completed['newNumericGatesRequireCompanions'] is True,
        'Preexisting classification or companion requirement changed')
    previous_plan = evidence.document(plan['previousV2Plan'])
    previous_exit = evidence.document(plan['previousV2Exit'])
    previous_process = evidence.document(plan['previousV2ProcessIdentity'])
    previous_output = Path(plan['previousV2Output'])
    events_ref = evidence.capture(previous_output/'events.jsonl')
    events = [json.loads(line) for line in Path(events_ref['path']).read_text().splitlines()]
    stop_ref = evidence.capture(previous_output/'stop-request.json'); stop = evidence.document(stop_ref)
    require(previous_plan['kind'] == 'registered-streaming-student-panel-queue-v2'
        and previous_plan['tasks'] == plan['tasks'] and previous_plan['panel'] == plan['panel']
        and previous_plan['source'] == evidence.capture(REPO/'scripts/run-neural-student-panel-queue-v2.py')
        and previous_exit['kind'] == 'streaming-student-recovery-observed-exit-v2'
        and previous_exit['exitCode'] == 0 and previous_exit['all27Completed'] is False
        and previous_exit['previousWorkerExitStillUnknown'] is True
        and previous_process['plan'] == plan['previousV2Plan']
        and previous_process['grant'] == previous_exit['grant']
        and stop['rootAuthorized'] is True and stop['signalSent'] is False
        and stop['workerIdentity'] == plan['previousV2ProcessIdentity']
        and events[-1]['stage'] == 'worker-stop' and events[-1]['reason'] == 'stop-or-12h-limit'
        and events[-1]['completed'] == snapshot['preexistingCount']
        and 0 < snapshot['preexistingCount'] < 27 and not (previous_output/'completed.json').exists(),
        'Preexisting v2 task scope or observed graceful stop differs')
    evidence.closure(previous_plan); evidence.closure(previous_exit); evidence.closure(previous_process); evidence.closure(stop)
    previous_stop_evidence = {'plan': plan['previousV2Plan'], 'exit': plan['previousV2Exit'],
        'process': plan['previousV2ProcessIdentity'], 'events': events_ref, 'stop': stop_ref}
    evidence.closure(plan); evidence.closure(completed)
    # Hash explicit direct binding trees only; never expand unrelated JSON lineage.
    for key in ('source', 'cacheAdapter', 'numericExecutionPlan', 'numericExecutionQualification',
                'previousV2Plan', 'previousV2Exit', 'previousV2ProcessIdentity', 'classificationSnapshot'):
        evidence.closure(plan[key])
    evidence.closure(completed.get('grant'))
    panel_ref = plan['panel']; panel = evidence.document(panel_ref)
    identifiers = panel['recordingIds']
    require(len(identifiers) == len(set(identifiers)) == 42 and panel['inferenceUsesLabels'] is False
        and panel['allInferenceTicksValid'] is True, 'Frozen full42 blind panel required')
    execution_plan_ref = plan['numericExecutionPlan']; execution_plan = evidence.document(execution_plan_ref)
    evidence.closure(execution_plan)
    require(execution_plan['kind'] == 'registered-student-inference-identity-cache-execution-v1'
        and execution_plan['preexistingClassificationOwnedByQueuePlan'] is True
        and execution_plan['immutableSourcesAndOutputPopulation'] is True, 'Wrong numeric execution plan')
    evidence.closure(execution_plan['sources']); evidence.closure(execution_plan['studentPlan'])
    require(execution_plan['sources']['analysis/neural_student_inference_identity_cache.py'] == plan['cacheAdapter'],
            'Queue and numeric adapter source differ')
    targets = execution_plan['targets']
    require(len(targets) == 27 and [item['taskId'] for item in targets] == [row['taskId'] for row in rows],
            'Numeric cache target population differs')
    for row, target in zip(rows, targets, strict=True):
        require(target['task'] == row['task'] and target['panel'] == panel_ref
            and target['fitDirectory'] == row['fitDirectory'] and target['outputPath'] == row['numericAuditPath']
            and target['companionPath'] == str(Path(row['fitDirectory'])/'student-inference-cache-execution-fp32.json'),
            'Numeric target owner/output differs')
    qualification = evidence.document(plan['numericExecutionQualification'])
    evidence.closure(qualification)
    require(qualification['kind'] == 'student-inference-identity-cache-byte-exact-qualification-v1'
        and qualification['passed'] is True and qualification['plan'] == execution_plan_ref
        and qualification['adapter'] == plan['cacheAdapter']
        and qualification['source'] == execution_plan['sources']['scripts/student-inference-identity-cache.py']
        and qualification['controlAudit'] == execution_plan['qualification']['controlAudit']
        and qualification['byteExactColdParity'] is True and qualification['byteExactWarmParity'] is True
        and qualification['productionOutputsModified'] is False
        and qualification['calibrationOrOutcomeMetricsExecuted'] is False,
        'Byte-exact cold and warm qualification required')
    evidence.bind(qualification['controlAudit'])
    require(len(qualification['executions']) == 2 and len(execution_plan['qualification']['targets']) == 2,
            'Both qualification stages required')
    stages = []
    for target, reference in zip(execution_plan['qualification']['targets'], qualification['executions'], strict=True):
        value = evidence.document(reference); audit_ref = value['output']; numeric = evidence.document(audit_ref)
        evidence.closure(value)
        require(audit_ref == {'path': target['outputPath'], 'sha256': qualification['controlAudit']['sha256']}
            and reference['path'] == target['companionPath'], 'Qualification audit bytes/owner changed')
        files = validate_companion(value, execution_plan=execution_plan, execution_plan_ref=execution_plan_ref,
            adapter_ref=plan['cacheAdapter'], row=rows[0], panel_ref=panel_ref,
            audit_ref=audit_ref, numeric=numeric, output_path=target['outputPath'])
        for item in files:
            evidence.bind({'path': item['path'], 'sha256': item['sha256']}, item['stat'])
        evidence.closure(numeric['evidence']); stages.append(value)
    require(stages[0]['statistics']['coldHashCount'] > 0 and stages[1]['statistics']['coldHashCount'] == 0
        and stages[1]['statistics']['cacheHitCount'] > 0 and stages[0]['files'] == stages[1]['files']
        and stages[0]['pid'] == stages[1]['pid'] and [item['processStageOrdinal'] for item in stages] == [1, 2],
        'Qualification is not one process with cold then warm execution')
    require(stages[0]['bootId'] == stages[1]['bootId']
        and stages[0]['processStartTimeTicks'] == stages[1]['processStartTimeTicks'],
        'Cold and warm qualification used different processes')
    checks, cached_stages = [], []
    for row in rows:
        task = evidence.document(row['task'])
        require(task['taskId'] == row['taskId'] and task['model'] == 'distilled-mobile-tcn', 'Wrong student task')
        audit_ref = evidence.capture(row['numericAuditPath']); numeric = evidence.document(audit_ref)
        numerical_scope(numeric, row, panel_ref, identifiers)
        for key in NUMERICAL_REFERENCE_FIELDS:
            evidence.closure(numeric[key])
        companion_ref = None
        if row['mode'] == 'preexisting-uncached':
            require(audit_ref == row['numericAudit'], 'Preexisting numerical audit changed')
            prior = evidence.document(row['preexistingCompletion'])
            require(prior['task'] == row['task'] and prior['completion']['numericalAudit'] == audit_ref,
                    'Preexisting v2 completion differs')
            evidence.closure(prior)
        else:
            companion_ref = companions[row['taskId']]; value = evidence.document(companion_ref)
            evidence.closure(value)
            files = validate_companion(value, execution_plan=execution_plan, execution_plan_ref=execution_plan_ref,
                adapter_ref=plan['cacheAdapter'], row=row, panel_ref=panel_ref, audit_ref=audit_ref, numeric=numeric)
            for item in files:
                evidence.bind({'path': item['path'], 'sha256': item['sha256']}, item['stat'])
            cached_stages.append(value)
        checks.append({'taskId': row['taskId'], 'mode': row['mode'],
                       'numericAudit': audit_ref, 'executionCompanion': companion_ref})
        print(json.dumps({'coldCheckedStudentTask': row['taskId'], 'completed': len(checks), 'total': 27}), flush=True)
    validate_process_stages(cached_stages)
    evidence.finish()
    result = {'kind': KIND, 'passed': True, 'source': own_ref, 'coldHelper': helper_ref,
        'numericalAuditor': numerical_ref, 'queuePlan': plan_ref, 'queueCompleted': completed_ref,
        'previousV2StopEvidence': previous_stop_evidence,
        'logicalTaskCount': 27, 'taskChecks': checks, 'coldCacheImported': False,
        'allNamedNumericalEvidenceRehashed': True, 'allTouchedEvidenceRehashed': True,
        'checkedFiles': [evidence.checked[key] for key in sorted(evidence.checked)],
        'fileCount': len(evidence.checked), 'totalBytes': evidence.total_bytes,
        'createdAtUTC': datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    verify_publication(args.output, args.queue_plan)
    print(json.dumps({'coldAudit': identity(args.output), 'files': result['fileCount'],
                      'bytes': result['totalBytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-plan', type=Path, required=True)
    parser.add_argument('--queue-completed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
