#!/usr/bin/env python3
"""Serialize registered student encoders for blind full-panel inference and audits."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0061'))
GIB = 1024**3
OBSERVER = REPO/'scripts/run-neural-student-reuse-profiled.py'
SPEC = importlib.util.spec_from_file_location('registered_student_resource_helpers', OBSERVER)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verified(reference):
    require(HELPERS.identity(reference['path']) == reference, 'Registered metadata changed: '+reference['path'])
    return Path(reference['path'])


def gate_helpers():
    from analysis.neural_selection_serialization import load_module
    from analysis.neural_selection_duration_view import verify_execution
    correction = load_module('student_panel_frozen_correction_gate', REPO/'scripts/generalization-selection-containment.py')
    reuse = load_module('student_panel_frozen_reuse_queue', REPO/'scripts/run-neural-generalization-reuse-queue.py')
    return correction, reuse, verify_execution


def preflight(plan):
    correction, reuse, verify_execution = gate_helpers()
    result, cache = [], {}
    for row in plan['tasks']:
        task_path = verified(row['task'])
        folder = Path(row['fitDirectory'])
        numerical_path = folder/'fit-numerical-audit.json'
        numeric = reuse.check_gate(numerical_path, task_path, folder, 'independent-generalization-fit-numerical-audit-v1')
        fitted = read(folder/'fit-result.json')
        require(numeric['taskId'] == row['taskId'] and numeric['checkpointEpochs'] == [5, 15, 30, 60]
                and numeric['completed'] == fitted['temporal'] and numeric['student'] is not None,
                'Student numerical fit scope differs')
        selection, ordinary, companion = folder/'selection.json', folder/'selection-audit.json', folder/'selection-correction-audit.json'
        correction.correction_gate(selection, ordinary, companion, cache)
        require(read(selection)['task'] == row['task'], 'Selection belongs to another student task')
        execution = verify_execution(ordinary)
        result.append({'task': row['task'], 'fitNumericalAudit': HELPERS.identity(numerical_path),
                       'selection': HELPERS.identity(selection), 'ordinaryAudit': HELPERS.identity(ordinary),
                       'correctionAudit': HELPERS.identity(companion), 'execution': execution})
    require(len(result) == 27, 'Incomplete student preflight')
    return result


def completion_gates(plan, before):
    correction, reuse, _ = gate_helpers()
    require(preflight(plan) == before, 'Student fit or selection gate changed during inference')
    result, cache = [], {}
    task_refs = {row['taskId']: row['task'] for row in plan['tasks']}
    for row, initial in zip(plan['tasks'], before, strict=True):
        task_path = verified(row['task'])
        folder = Path(row['fitDirectory'])
        fitted = read(folder/'fit-result.json')
        numerical_path = folder/'inference-numerical-audit-fp32.json'
        numeric = reuse.check_gate(numerical_path, task_path, folder, 'independent-generalization-inference-numerical-audit-v1')
        require(numeric['taskId'] == row['taskId'] and numeric['panel'] == plan['panel'] and numeric['precision'] == 'fp32'
                and numeric['checkpointEpochs'] == [5, 15, 30, 60] and numeric['completed'] == fitted['temporal']
                and numeric['fitAudit'] == initial['fitNumericalAudit'], 'Student inference numerical gate association differs')
        output = folder/'inference'/plan['panel']['sha256'][:16]/'fp32'
        expected_receipts = [HELPERS.identity(output/(key+'.json')) for key in plan['recordingIds']]
        require(numeric['inferenceReceipts'] == expected_receipts
                and {p.stem for p in output.glob('*.json')} == set(plan['recordingIds'])
                and {p.stem for p in output.glob('*.npz')} == set(plan['recordingIds']), 'Student inference full42 inventory differs')
        require([(r['recordingId'], r['epoch']) for r in numeric['cpuReplay']]
                == [(key, epoch) for key in plan['recordingIds'] for epoch in (5, 15, 30, 60)]
                and [r['id'] for r in numeric['studentFeatureReplay']] == plan['recordingIds'],
                'Student inference replay scope differs')
        correction.verify_closure(numeric, cache)
        copied_ref = None
        if row['physicalOwnerTaskId'] != row['taskId']:
            copied_path = folder/'reuse-inference-audit-fp32.json'
            copied = reuse.check_gate(copied_path, task_path, folder, 'independent-generalization-inference-reuse-audit-v1')
            expected_plan = next(ref for ref in plan['reusePlans'] if Path(ref['path']).parent == folder.parents[3])
            require(copied['plan'] == expected_plan and copied['sourceTask'] == task_refs[row['physicalOwnerTaskId']]
                    and copied['panel'] == plan['panel'] and copied['precision'] == 'fp32'
                    and copied['targetNumericalAudit'] == HELPERS.identity(numerical_path)
                    and copied['recordingIds'] == plan['recordingIds'] and copied['checkpointEpochs'] == [5, 15, 30, 60]
                    and copied['allRawScoreBytesExactlyEqual'] is True and copied['studentFeatureArrayEqualityCount'] == 42,
                    'Student inference reuse gate association/scope differs')
            correction.verify_closure(copied, cache)
            copied_ref = HELPERS.identity(copied_path)
        result.append({'task': row['task'], 'numericalAudit': HELPERS.identity(numerical_path), 'reuseAudit': copied_ref,
                       'recordingCount': 42, 'checkpointEpochs': [5, 15, 30, 60]})
        print(json.dumps({'studentPanelCompletionGate': row['taskId']}), flush=True)
    return result


def task_inventory():
    original = ROOT/'original-corpus-v1'
    tasks = []
    registration = read(original/'registration.json')
    for task_id in registration['contract']['taskIds']:
        path = original/'tasks'/(task_id.replace('/', '__')+'.json')
        task = read(path)
        if task['model'] == 'distilled-mobile-tcn':
            tasks.append({'taskId': task_id, 'task': HELPERS.identity(path), 'physicalOwnerTaskId': task_id,
                          'fitDirectory': str(original/'fits'/task['model']/f'seed-{task["seed"]}')})
    require(len(tasks) == 3, 'Original student population differs')
    for name, expected_logical, expected_physical in (('randomized-variants-v1', 16, 10), ('export-proxy-v1', 8, 7)):
        plan = read(ROOT/name/'reuse-plan.json')
        rows = []
        for row in plan['tasks']:
            task = read(verified(row['task']))
            if task['model'] == 'distilled-mobile-tcn':
                rows.append({key: row[key] for key in ('taskId', 'task', 'physicalOwnerTaskId', 'fitDirectory')})
        require(len(rows) == expected_logical and len({row['physicalOwnerTaskId'] for row in rows}) == expected_physical,
                'Registered student reuse population differs: '+name)
        tasks.extend(rows)
    require(len({row['taskId'] for row in tasks}) == 27, 'Duplicate logical student task')
    return tasks


def commands():
    panel = str(ROOT/'panels-v1/fp32-core.json')
    common = ['--model', 'distilled-mobile-tcn', '--panel', panel, '--precision', 'fp32', '--device', 'cuda']
    evaluation = str(REPO/'scripts/run-neural-generalization-evaluation-queue.py')
    reuse = str(REPO/'scripts/run-neural-generalization-reuse-queue.py')
    result = [[sys.executable, '-B', evaluation, phase, '--registration-dir', str(ROOT/'original-corpus-v1'), *common]
              for phase in ('infer', 'audit-inference')]
    result += [[sys.executable, '-B', reuse, '--plan', str(ROOT/name/'reuse-plan.json'), '--action', 'infer', *common]
               for name in ('randomized-variants-v1', 'export-proxy-v1')]
    return result


def contract():
    panel_path = ROOT/'panels-v1/fp32-core.json'
    panel = read(panel_path)
    require(panel['kind'] == 'frozen-independent-inference-panel-v1' and panel['inferenceUsesLabels'] is False
            and panel['allInferenceTicksValid'] is True and len(set(panel['recordingIds'])) == 42,
            'Invalid blind full42 panel')
    require(panel['precisionVariants'] == ['fp32'], 'Student inference must keep the original FP32 panel')
    return {'kind': 'registered-student-full-panel-queue-v1', 'source': HELPERS.identity(__file__),
        'resourceHelpers': HELPERS.identity(OBSERVER),
        'queueSources': [HELPERS.identity(REPO/'scripts'/name) for name in
                         ('run-neural-generalization-evaluation-queue.py', 'run-neural-generalization-reuse-queue.py')],
        'gateSources': [HELPERS.identity(REPO/name) for name in
            ('analysis/neural_selection_duration_view.py', 'analysis/neural_selection_serialization.py',
             'scripts/generalization-selection-containment.py', 'scripts/audit-neural-generalization-numerics.py',
             'scripts/audit-neural-generalization-reuse.py')],
        'originalRegistration': HELPERS.identity(ROOT/'original-corpus-v1/registration.json'),
        'reusePlans': [HELPERS.identity(ROOT/name/'reuse-plan.json') for name in ('randomized-variants-v1', 'export-proxy-v1')],
        'panel': HELPERS.identity(panel_path), 'panelBindings': {key: panel[key] for key in
                         ('manifest', 'features', 'featureAudit', 'inventory', 'registrar')},
        'recordingIds': panel['recordingIds'], 'tasks': task_inventory(), 'commands': commands(),
        'logicalTaskCount': 27, 'physicalEncoderCount': 20, 'cpuAffinity': [10, 11],
        'numericalRecipesChanged': False, 'imageEncoderConcurrency': 1,
        'studentFittingAllowed': False, 'teacherTargetsAllowed': False, 'evaluationMetricsAllowed': False,
        'preflightRequiresAll27FitOrdinaryCorrectionExecutionGates': True,
        'completionRequiresAll27Full42All4NumericalAndApplicableReuseGates': True,
        'selectionRequirement': 'Every raw inference task must have its own frozen audited selection. All162 global selections must freeze before any accuracy evaluation.',
        'reuseRule': 'Resume exact frozen encoder/image receipts; reuse logical predictions only through the existing independently audited reuse queue.',
        'startupMinimumAvailableRamBytes': 8*GIB, 'startupMinimumGpuFreeMiB': 5*1024,
        'startupMinimumWindowsPhysicalFreeBytes': 3*GIB, 'minimumWindowsCFreeBytes': 20*GIB}


def validate_plan(plan):
    require(plan == contract(), 'Student panel queue differs from its frozen contract')
    for reference in plan['panelBindings'].values():
        verified(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--grant-sha256')
    parser.add_argument('--fit-exit', type=Path)
    parser.add_argument('--fit-exit-sha256')
    parser.add_argument('--required-exited-pid', type=int, action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(Path('/mnt/freenas').is_mount() and args.plan.resolve().is_relative_to(ROOT), 'Expected mounted NAS plan')
    if args.action == 'register':
        value = contract()
        validate_plan(value)
        HELPERS.write_new(args.plan, value)
        print(json.dumps({'registered': HELPERS.identity(args.plan)}), flush=True)
        return
    require(all((args.plan_sha256, args.grant, args.grant_sha256, args.fit_exit, args.fit_exit_sha256,
                 args.required_exited_pid, args.output)), 'Run requires a root execution grant and completed fit exit')
    for path, digest in ((args.plan, args.plan_sha256), (args.grant, args.grant_sha256), (args.fit_exit, args.fit_exit_sha256)):
        require(HELPERS.identity(path)['sha256'] == digest, 'Execution binding differs: '+str(path))
    plan = read(args.plan)
    validate_plan(plan)
    require(read(args.fit_exit).get('released') is True, 'Student fitting slot is not released')
    require(all(not (Path('/proc')/str(pid)).exists() for pid in args.required_exited_pid), 'Student fitting process is still present')
    expected_grant = {'kind': 'student-full-panel-queue-execution-grant-v1', 'source': HELPERS.identity(__file__),
        'plan': HELPERS.identity(args.plan), 'fitExit': HELPERS.identity(args.fit_exit),
        'requiredExitedPids': args.required_exited_pid, 'cpuAffinity': [10, 11],
        'imageEncoderConcurrency': 1, 'numericalRecipesChanged': False, 'evaluationMetricsAllowed': False}
    HELPERS.validate_grant(read(args.grant), expected_grant)
    folder = args.output.resolve()
    require(folder.is_relative_to(ROOT), 'Expected NAS execution outputs')
    os.sched_setaffinity(0, {10, 11})
    for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        path = folder/'runtime'/name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    process = Path('/proc')/str(os.getpid())
    parent, start = HELPERS.stat_identity((process/'stat').read_text())
    HELPERS.write_new(folder/'process-identity.json', {'pid': os.getpid(), 'parentPid': parent, 'startTimeTicks': start,
        'bootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'uid': os.getuid(),
        'command': [v.decode() for v in (process/'cmdline').read_bytes().split(b'\0') if v],
        'plan': HELPERS.identity(args.plan), 'grant': HELPERS.identity(args.grant), 'fitExit': HELPERS.identity(args.fit_exit)})
    selected_gates = preflight(plan)
    HELPERS.write_new(folder/'preflight.json', {'plan': HELPERS.identity(args.plan), 'passed': True, 'tasks': selected_gates})
    results = []
    for index, command in enumerate(plan['commands']):
        while True:
            state = HELPERS.resources()
            if (state['linuxAvailableBytes'] >= 8*GIB and state['windowsPhysicalFreeBytes'] >= 3*GIB
                    and state['windowsCFreeBytes'] >= 20*GIB and state['gpuFreeMiB'] >= 5*1024):
                break
            print(json.dumps({'studentPanelQueueWaiting': 'startup resource floor', **state}), flush=True)
            time.sleep(30)
        HELPERS.write_new(folder/f'launch-{index}.json', {'plan': HELPERS.identity(args.plan), 'command': command,
            'resources': state, 'preflight': HELPERS.identity(folder/'preflight.json')})
        result = subprocess.run(command, cwd=REPO)
        results.append({'command': command, 'returnCode': result.returncode})
        HELPERS.write_new(folder/f'result-{index}.json', results[-1])
        require(result.returncode == 0, 'Registered student panel queue failed; no next stage started')
    checked = completion_gates(plan, selected_gates)
    HELPERS.write_new(folder/'completed.json', {'kind': 'student-full-panel-queue-completed-v1',
        'plan': HELPERS.identity(args.plan), 'grant': HELPERS.identity(args.grant), 'queueResults': results,
        'logicalTaskCount': 27, 'physicalEncoderCount': 20, 'evaluationMetricsProduced': False,
        'completionGates': checked, 'all27Full42All4NumericAndApplicableReuseGatesPassed': True})
    print(json.dumps({'completed': HELPERS.identity(folder/'completed.json')}), flush=True)


if __name__ == '__main__':
    main()
