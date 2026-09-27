#!/usr/bin/env python3
"""Execute every logical fit while sharing only identical preregistered training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.neural_generalization_experiment import load_task
from analysis.neural_generalization_inputs import verified
from analysis.neural_recall_sweep import identity, read, require, write_new


def resource_gate(device='cuda'):
    available = next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()
                     if row.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    require(available >= 4*1024**3 and disk.f_bavail*disk.f_frsize >= 20*1024**3,
            'Resource floor reached before next logical task')
    result = {'availableRamBytes': available, 'windowsCFreeBytes': disk.f_bavail*disk.f_frsize}
    if device.startswith('cuda'):
        query = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                               check=True, capture_output=True, text=True, timeout=15)
        free = [int(row.strip()) for row in query.stdout.splitlines() if row.strip()]
        require(len(free) == 1 and free[0] >= 3*1024, 'Declared 3 GiB free-GPU floor reached')
        result['gpuFreeMiB'] = free[0]
    return result


def check_gate(path, task_path, fit, kind):
    result = read(path)
    require(result['kind'] == kind and result['passed'] is True
            and result['task'] == identity(task_path) and result['fitResult'] == identity(fit/'fit-result.json'),
            'Existing audit is stale or belongs to another task: '+str(path))
    auditor = REPO/('scripts/audit-neural-generalization-reuse.py' if 'reuse-audit' in kind
                    else 'scripts/audit-neural-generalization-numerics.py')
    require(result['auditor'] == identity(auditor), 'Gate was produced by a different auditor revision')
    # Rebind the checkpoint bank on resume. Large feature arrays are independently
    # rebound by the numerical auditor and by the frozen inference/data loaders.
    fitted = read(fit/'fit-result.json')
    completed = read(verified(fitted['temporal']))
    for name, digest in completed['artifacts'].items():
        verified({'path': str(fit/'temporal'/name), 'sha256': digest})
    if fitted.get('student'):
        student = read(verified(fitted['student']))
        verified(student['weights'])
    return result


def run_commands(commands, log_path, task_id):
    for command in commands:
        with log_path.open('a') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        require(result.returncode == 0, 'Logical task failed; inspect NAS reuse log: '+task_id)


def infer_task(args, row, path, task, logs):
    require(args.panel is not None, 'Inference requires a frozen panel')
    panel_ref = identity(args.panel); panel = read(args.panel)
    require(panel['kind'] == 'frozen-independent-inference-panel-v1'
            and panel['inferenceUsesLabels'] is False and panel['allInferenceTicksValid'] is True
            and len(set(panel['recordingIds'])) == len(panel['recordingIds']) == 42, 'Invalid inference panel')
    for name in ('manifest', 'features', 'featureAudit', 'inventory', 'registrar'):
        verified(panel[name])
    ids_path = args.panel.parent/'inference-recording-ids.json'
    if ids_path.exists():
        require(read(ids_path) == panel['recordingIds'], 'Inference population changed')
    else:
        write_new(ids_path, panel['recordingIds'])
    fit = Path(row['fitDirectory']); canonical = row['taskId'] == row['physicalOwnerTaskId']
    numeric_fit = fit/'fit-numerical-audit.json'
    check_gate(numeric_fit, path, fit, 'independent-generalization-fit-numerical-audit-v1')
    selection = fit/'selection.json'; selection_gate = read(fit/'selection-audit.json')
    require(selection_gate['passed'] is True and selection_gate['selection'] == identity(selection),
            'Freeze and audit each logical task selection before external inference')
    if not canonical:
        check_gate(fit/'reuse-audit.json', path, fit, 'independent-generalization-fit-reuse-audit-v1')
    for precision in args.precision or ['fp32']:
        if precision != 'fp32' and task['model'] not in ('dino-tcn', 'dino-transformer'):
            continue
        require(precision in panel['precisionVariants'], 'Precision absent from panel')
        resource_gate(args.device)
        numerical = fit/f'inference-numerical-audit-{precision}.json'
        reuse_gate = fit/f'reuse-inference-audit-{precision}.json'
        commands = []
        if not numerical.exists():
            if canonical:
                commands.append([sys.executable, str(REPO/'scripts/run-neural-generalization.py'), 'infer',
                    '--task', str(path), '--output', str(fit), '--ids', str(ids_path),
                    '--panel', str(args.panel), '--precision', precision, '--device', args.device])
            elif not (fit/f'inference-reuse-receipt-{precision}.json').exists():
                commands.append([sys.executable, str(REPO/'scripts/reuse-neural-generalization-inference.py'),
                    '--plan', str(args.plan), '--task', str(path), '--output', str(fit),
                    '--panel', str(args.panel), '--precision', precision])
            commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-numerics.py'),
                '--phase', 'inference', '--task', str(path), '--fit', str(fit), '--panel', str(args.panel),
                '--precision', precision, '--fit-audit', str(numeric_fit), '--output', str(numerical)])
        if not canonical and not reuse_gate.exists():
            commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-reuse.py'),
                '--phase', 'inference', '--plan', str(args.plan), '--task', str(path), '--fit', str(fit),
                '--panel', str(args.panel), '--precision', precision, '--output', str(reuse_gate)])
        print(json.dumps({'starting': task['taskId'], 'action': 'infer', 'precision': precision,
                          'physicalTrainingOwner': row['physicalOwnerTaskId'], 'commands': len(commands)}), flush=True)
        run_commands(commands, logs/(task['taskId'].replace('/', '__')+'-infer-'+precision+'.log'), task['taskId'])
        checked = check_gate(numerical, path, fit, 'independent-generalization-inference-numerical-audit-v1')
        require(checked['panel'] == panel_ref and checked['precision'] == precision,
                'Inference audit refers to a different panel or precision')
        if not canonical:
            checked = check_gate(reuse_gate, path, fit, 'independent-generalization-inference-reuse-audit-v1')
            require(checked['plan'] == identity(args.plan) and checked['panel'] == panel_ref
                    and checked['precision'] == precision and checked['sourceTask'] == row['physicalOwnerTask'],
                    'Inference reuse audit differs from task/panel/owner')
        print(json.dumps({'completed': task['taskId'], 'action': 'infer', 'precision': precision}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--action', choices=('fit', 'infer'), default='fit')
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), action='append')
    parser.add_argument('--model', action='append')
    parser.add_argument('--variant', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    plan = read(args.plan)
    require(plan['kind'] == 'generalization-training-reuse-plan-v1', 'Unknown companion reuse plan')
    registration = read(verified(plan['registration']))
    for reference in plan['code'].values():
        verified(reference)
    verified(plan['protocol']); verified(plan['qualification'])
    require([row['taskId'] for row in plan['tasks']] == registration['contract']['taskIds'],
            'Reuse plan does not preserve registered logical order')
    requested = []
    for row in plan['tasks']:
        path = verified(row['task'])
        task, _, _, _ = load_task(path)
        if args.model and task['model'] not in args.model or args.variant and task['variant'] not in args.variant:
            continue
        require(task['taskId'] == row['taskId'], 'Mapped task ID differs')
        requested.append((row, path, task))
    if args.limit is not None:
        requested = requested[:args.limit]
    logs = args.plan.parent/'reuse-queue-logs'
    logs.mkdir(exist_ok=True)
    write_new(logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json'),
        {'kind': 'generalization-reuse-queue-execution-v1', 'plan': identity(args.plan),
         'source': identity(__file__), 'taskIds': [task['taskId'] for _, _, task in requested],
         'device': args.device, 'action': args.action, 'panel': identity(args.panel) if args.panel else None,
         'precisions': args.precision or ['fp32'],
         'allFixedCheckpointsRequired': True, 'independentTargetCalibration': True})
    for row, path, task in requested:
        resources = resource_gate(args.device)
        if args.action == 'infer':
            infer_task(args, row, path, task, logs)
            continue
        fit = Path(row['fitDirectory'])
        canonical = row['taskId'] == row['physicalOwnerTaskId']
        commands = []
        if not (fit/'fit-result.json').exists():
            if canonical:
                command = [sys.executable, str(REPO/'scripts/run-neural-generalization.py'), 'fit',
                           '--task', str(path), '--output', str(fit), '--device', args.device]
            else:
                donor = Path(row['physicalOwnerFitDirectory'])
                check_gate(donor/'fit-numerical-audit.json', verified(row['physicalOwnerTask']), donor,
                           'independent-generalization-fit-numerical-audit-v1')
                command = [sys.executable, str(REPO/'scripts/reuse-neural-generalization-fit.py'),
                           '--plan', str(args.plan), '--task', str(path), '--output', str(fit), '--device', args.device]
            commands.append(command)
        numerical = fit/'fit-numerical-audit.json'
        if not numerical.exists():
            commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-numerics.py'),
                '--phase', 'fit', '--task', str(path), '--fit', str(fit), '--output', str(numerical)])
        reuse_audit = fit/'reuse-audit.json'
        if not canonical and not reuse_audit.exists():
            commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-reuse.py'),
                '--phase', 'fit', '--plan', str(args.plan), '--task', str(path),
                '--fit', str(fit), '--output', str(reuse_audit)])
        print(json.dumps({'starting': task['taskId'], 'physicalTrainingOwner': row['physicalOwnerTaskId'],
                          'commands': len(commands), 'resources': resources}), flush=True)
        run_commands(commands, logs/(task['taskId'].replace('/', '__')+'.log'), task['taskId'])
        check_gate(numerical, path, fit, 'independent-generalization-fit-numerical-audit-v1')
        fitted = read(fit/'fit-result.json')
        require(('trainingReuse' in fitted) is (not canonical), 'Physical/reused training status differs')
        if not canonical:
            checked = check_gate(reuse_audit, path, fit, 'independent-generalization-fit-reuse-audit-v1')
            require(checked['plan'] == identity(args.plan) and checked['sourceTask'] == row['physicalOwnerTask'],
                    'Independent reuse gate refers to another owner/plan')
        print(json.dumps({'completed': task['taskId'], 'fit': identity(fit/'fit-result.json'),
                          'numericalAudit': identity(numerical),
                          'reuseAudit': None if canonical else identity(reuse_audit)}), flush=True)


if __name__ == '__main__':
    main()
