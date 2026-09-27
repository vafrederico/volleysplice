#!/usr/bin/env python3
"""Resume the interrupted FP32 orchestration without changing numerical recipes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_inputs import verified
from analysis.neural_selection_duration_view import verify_execution


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


queue = module('recovery_reuse', REPO/'scripts/run-neural-generalization-reuse-queue.py')
containment = module('recovery_containment', REPO/'scripts/generalization-selection-containment.py')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def emit(**values):
    print(json.dumps({'time': stamp(), **values}), flush=True)


def check_rows(rows, panel_path, *, inference, shared=None):
    shared = {} if shared is None else shared
    panel_ref = io.identity(panel_path)
    ids = io.read(panel_path)['recordingIds']
    assert len(ids) == len(set(ids)) == 42
    checks = []
    for row in rows:
        folder = Path(row['fitDirectory'])
        task = verified(row['task'])
        fitted = io.read(folder/'fit-result.json')
        fpath = folder/'fit-numerical-audit.json'
        fit = queue.check_gate(fpath, task, folder, 'independent-generalization-fit-numerical-audit-v1')
        assert fit['taskId'] == row['taskId'] and fit['completed'] == fitted['temporal']
        assert fit['checkpointEpochs'] == [5, 15, 30, 60] and fit['student'] is None
        containment.verify_closure(fit, shared)
        containment.correction_gate(folder/'selection.json', folder/'selection-audit.json',
                                    folder/'selection-correction-audit.json', shared)
        assert io.read(folder/'selection.json')['task'] == row['task']
        checked = {'task': row['task'], 'fitAudit': io.identity(fpath),
                   'selection': io.identity(folder/'selection.json'),
                   'ordinaryAudit': io.identity(folder/'selection-audit.json'),
                   'correctionAudit': io.identity(folder/'selection-correction-audit.json'),
                   'execution': verify_execution(folder/'selection-audit.json')}
        reused = bool(row.get('reusePlan')) and row['taskId'] != row['physicalOwnerTaskId']
        if reused:
            path = folder/'reuse-audit.json'
            gate = queue.check_gate(path, task, folder, 'independent-generalization-fit-reuse-audit-v1')
            assert gate['plan'] == row['reusePlan'] and gate['sourceTask'] == row['physicalOwnerTask']
            assert gate['targetNumericalAudit'] == io.identity(fpath)
            containment.verify_closure(gate, shared)
            checked['fitReuseAudit'] = io.identity(path)
        if inference:
            path = folder/'inference-numerical-audit-fp32.json'
            gate = queue.check_gate(path, task, folder, 'independent-generalization-inference-numerical-audit-v1')
            output = folder/'inference'/panel_ref['sha256'][:16]/'fp32'
            assert gate['taskId'] == row['taskId'] and gate['panel'] == panel_ref and gate['precision'] == 'fp32'
            assert gate['completed'] == fitted['temporal'] and gate['fitAudit'] == io.identity(fpath)
            assert gate['checkpointEpochs'] == [5, 15, 30, 60] and gate['studentFeatureReplay'] == []
            assert gate['inferenceReceipts'] == [io.identity(output/(key+'.json')) for key in ids]
            assert {p.stem for p in output.glob('*.json')} == {p.stem for p in output.glob('*.npz')} == set(ids)
            assert [(v['recordingId'], v['epoch']) for v in gate['cpuReplay']] == [
                (key, epoch) for key in ids for epoch in (5, 15, 30, 60)]
            containment.verify_closure(gate, shared)
            checked['inferenceAudit'] = io.identity(path)
            if reused:
                path = folder/'reuse-inference-audit-fp32.json'
                gate = queue.check_gate(path, task, folder, 'independent-generalization-inference-reuse-audit-v1')
                assert gate['plan'] == row['reusePlan'] and gate['sourceTask'] == row['physicalOwnerTask']
                assert gate['panel'] == panel_ref and gate['precision'] == 'fp32'
                assert gate['recordingIds'] == ids and gate['checkpointEpochs'] == [5, 15, 30, 60]
                assert gate['allRawScoreBytesExactlyEqual'] is True and gate['studentFeatureArrayEqualityCount'] == 0
                assert gate['targetNumericalAudit'] == checked['inferenceAudit']
                assert gate['fitReuseAudit'] == checked['fitReuseAudit']
                containment.verify_closure(gate, shared)
                checked['reuseAudit'] = io.identity(path)
        checks.append(checked)
        emit(checked=row['taskId'], inference=inference, completed=len(checks), total=len(rows))
    return checks


def register(args):
    old_path = args.root/'cached-fp32-remaining-v1/plan.json'
    old = io.read(old_path)
    assert [s['id'] for s in old['stages']] == [
        'proxy-av', 'original-nonstudent', 'random-nonstudent', 'proxy-nonstudent']
    inventory = io.read(verified(old['taskInventory']))
    av = [r for r in inventory['jobs'] if r['taskId'].split('/')[0] in (
        'original-medium', 'expanded-medium', 'expanded-large', 'expanded-wide-validation')
        and io.read(verified(r['task']))['model'] in ('av-tcn', 'av-transformer')]
    done = av + [r for s in old['stages'][:3] for r in s['jobs']]
    pending = old['stages'][3]['jobs']
    assert len(av) == 32 and len(done) == 111 and len(pending) == 24
    assert len({r['taskId'] for r in done + pending}) == 135
    # This recovery is deliberately specific to a fully completed third stage
    # and an unstarted fourth stage. Other interruption shapes fail closed.
    assert all((Path(r['fitDirectory'])/'inference-numerical-audit-fp32.json').exists() for r in done)
    assert all(not (Path(r['fitDirectory'])/'inference').exists() for r in pending)
    assert not (old_path.parent/'complete.json').exists()
    assert not (old_path.parent/'random-nonstudent-complete.json').exists()
    assert not (old_path.parent/'proxy-nonstudent-launch.json').exists()
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output/'interrupted-queue'
    archive.mkdir()
    copied = []
    for path in sorted(old_path.parent.iterdir()):
        assert path.is_file(), 'Inspect unexpected old queue subdirectory'
        target = archive/path.name
        shutil.copy2(path, target)
        assert io.identity(target)['sha256'] == io.identity(path)['sha256']
        copied.append({'original': io.identity(path), 'preserved': io.identity(target)})
    interruption = io.read(args.interruption)
    plan = {'kind': 'interrupted-cached-fp32-recovery-plan-v1', 'createdAt': stamp(),
            'source': io.identity(__file__), 'priorPlan': io.identity(old_path),
            'interruption': io.identity(args.interruption), 'preservedQueueFiles': copied,
            'priorObservedExitCode': None, 'panel': old['panel'], 'code': old['code'],
            'verifiedExistingJobs': done, 'pendingJobs': pending,
            'commands': old['stages'][3]['commands'],
            'policy': {'cpuAffinity': [12, 13], 'nice': 10,
                       'numericalRecipesChanged': False, 'externalMetricsOpened': False,
                       'imageEncodersAllowed': False, 'maximumConcurrentGpuWorkers': 3}}
    io.write_new(args.output/'plan.json', plan)
    emit(registered=io.identity(args.output/'plan.json'), completedJobs=111, pendingJobs=24)


def load_plan(path):
    plan = io.read(path)
    assert plan['kind'] == 'interrupted-cached-fp32-recovery-plan-v1'
    assert plan['source'] == io.identity(__file__)
    for ref in plan['code'].values():
        verified(ref)
    verified(plan['priorPlan'])
    verified(plan['interruption'])
    for row in plan['preservedQueueFiles']:
        verified(row['original'])
        verified(row['preserved'])
    return plan


def audit(args):
    plan = load_plan(args.plan)
    os.sched_setaffinity(0, {12, 13})
    os.nice(10)
    assert not args.output.exists()
    checks = check_rows(plan['verifiedExistingJobs'], verified(plan['panel']), inference=True)
    io.write_new(args.output, {'kind': 'interrupted-cached-fp32-existing-audit-v1',
        'passed': True, 'plan': io.identity(args.plan), 'source': io.identity(__file__),
        'checks': checks, 'logicalTasks': 111, 'all42All4': True, 'priorObservedExitCode': None,
        'externalMetricsOpened': False, 'createdAt': stamp()})
    emit(audit=io.identity(args.output))


def resources():
    mem = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith('MemAvailable:'))
    win = json.loads(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        '$o=Get-CimInstance Win32_OperatingSystem; @{physical=[long]$o.FreePhysicalMemory*1024;cFree=[long](Get-PSDrive C).Free}|ConvertTo-Json -Compress'], text=True))
    gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    return {'linuxAvailableBytes': mem, 'windows': win, 'gpuFreeMiB': gpu,
            'passed': mem >= 6*2**30 and win['physical'] >= 2*2**30 and win['cFree'] >= 20*2**30 and gpu >= 4096}


def run(args):
    plan = load_plan(args.plan)
    assert io.identity(args.grant)['sha256'] == args.grant_sha256
    grant = io.read(args.grant)
    assert grant['kind'] == 'interrupted-cached-fp32-recovery-grant-v1'
    assert grant['plan'] == io.identity(args.plan) and grant['maximumConcurrentGpuWorkers'] == 3
    assert grant['imageEncodersAllowed'] is False and grant['externalMetricsOpened'] is False
    audit_path = verified(grant['existingAudit'])
    audit_doc = io.read(audit_path)
    assert audit_doc['plan'] == io.identity(args.plan) and audit_doc['passed'] is True and audit_doc['logicalTasks'] == 111
    assert grant['interruption'] == plan['interruption']
    os.sched_setaffinity(0, {12, 13})
    os.nice(10)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        target = args.plan.parent/'runtime'/key.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    output = args.plan.parent/'run'
    output.mkdir(exist_ok=False)
    panel = verified(plan['panel'])
    checks = check_rows(plan['pendingJobs'], panel, inference=False)
    started = time.monotonic()
    while True:
        state = resources()
        if state['passed']:
            break
        assert time.monotonic()-started < 12*3600 and not (output/'STOP').exists()
        emit(waiting='resources', resources=state)
        time.sleep(30)
    io.write_new(output/'launch.json', {'plan': io.identity(args.plan), 'grant': io.identity(args.grant),
        'selectionChecks': checks, 'resources': state, 'createdAt': stamp(), 'pid': os.getpid(),
        'bootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'startTicks': Path('/proc/self/stat').read_text().split()[21]})
    for i, command in enumerate(plan['commands']):
        emit(starting='proxy-nonstudent', commandIndex=i)
        with (output/f'{i}.log').open('a') as log:
            result = subprocess.run([sys.executable, *command], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, 'Recovery command failed; inspect and preserve partial work'
    completed = check_rows(plan['pendingJobs'], panel, inference=True)
    # Rebind all previously completed evidence after the new work as well.
    previous = check_rows(plan['verifiedExistingJobs'], panel, inference=True)
    assert previous == audit_doc['checks']
    io.write_new(output/'complete.json', {'kind': 'interrupted-cached-fp32-recovery-complete-v1',
        'plan': io.identity(args.plan), 'grant': io.identity(args.grant), 'existingAudit': io.identity(audit_path),
        'newChecks': completed, 'existingChecksUnchanged': True, 'all135All42All4': True,
        'priorObservedExitCode': None, 'externalMetricsOpened': False, 'createdAt': stamp()})
    emit(completed=io.identity(output/'complete.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('register')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--interruption', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('audit-existing')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--grant', type=Path, required=True)
    p.add_argument('--grant-sha256', required=True)
    args = parser.parse_args()
    {'register': register, 'audit-existing': audit, 'run': run}[args.action](args)


if __name__ == '__main__':
    main()
