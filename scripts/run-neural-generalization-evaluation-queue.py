#!/usr/bin/env python3
"""Run explicit frozen task lists through selection, blind inference, or evaluation."""
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
from analysis.neural_generalization_inputs import manifest_rows, verified
from analysis.neural_recall_sweep import identity, read, require, write_new


def resource_gate():
    available = next(int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    require(available >= 3*1024**3 and disk.f_bavail*disk.f_frsize >= 20*1024**3, 'Resource floor reached before next task')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit-fit', 'select', 'infer', 'audit-inference', 'evaluate'))
    p.add_argument('--registration-dir', type=Path, required=True)
    p.add_argument('--historical', type=Path)
    p.add_argument('--panel', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--original-manifest', type=Path)
    p.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), action='append')
    p.add_argument('--model', action='append')
    p.add_argument('--variant', action='append')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    registration = read(a.registration_dir/'registration.json')
    requested = []
    for task_id in registration['contract']['taskIds']:
        path = a.registration_dir/'tasks'/(task_id.replace('/', '__')+'.json')
        task, _, _, _ = load_task(path)
        if a.model and task['model'] not in a.model or a.variant and task['variant'] not in a.variant:
            continue
        fit = a.registration_dir/'fits'/task['model']/f'seed-{task["seed"]}' if task['variant'] == 'original-corpus' else (
            a.registration_dir/'fits'/task['variant']/task['model']/f'split-{task["splitSeed"]}')
        require((fit/'fit-result.json').exists(), 'Requested fit is incomplete: '+task_id)
        requested.append((path, task, fit))
    if a.limit is not None:
        requested = requested[:a.limit]
    logs = a.registration_dir/'evaluation-queue-logs'
    logs.mkdir(exist_ok=True)
    write_new(logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+a.action+'.json'),
        {'action': a.action, 'registration': identity(a.registration_dir/'registration.json'),
         'taskIds': [t['taskId'] for _, t, _ in requested], 'source': identity(__file__),
         'precision': a.precision or ['fp32'], 'panel': identity(a.panel) if a.panel else None})
    ids_path = None
    if a.action not in ('select', 'audit-fit'):
        require(a.panel is not None, 'Independent frozen panel required')
        panel = read(a.panel)
        _, population = manifest_rows(verified(panel['manifest']))
        verified(panel['features'])
        ids_path = a.panel.parent/'inference-recording-ids.json'
        ids = [r['id'] for r in population if 'infer' in r['eligibleRoles']]
        if ids_path.exists():
            require(read(ids_path) == ids, 'Inference population changed')
        else:
            write_new(ids_path, ids)
    for path, task, fit in requested:
        resource_gate()
        selection = fit/'selection.json'; audit = fit/'selection-audit.json'
        numeric_fit = fit/'fit-numerical-audit.json'
        commands = []
        if a.action in ('select', 'audit-fit') and not numeric_fit.exists():
            commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-numerics.py'),
                '--phase', 'fit', '--task', str(path), '--fit', str(fit), '--output', str(numeric_fit)])
        if a.action == 'audit-fit':
            pass
        elif a.action == 'select':
            if not selection.exists():
                command = [sys.executable, str(REPO/'scripts/evaluate-neural-generalization.py'), 'select',
                           '--task', str(path), '--fit', str(fit), '--output', str(selection)]
                if a.historical:
                    command += ['--historical', str(a.historical)]
                commands.append(command)
            if not audit.exists():
                commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-selection.py'),
                                 '--selection', str(selection), '--output', str(audit)])
        else:
            require(selection.exists() and audit.exists(), 'Freeze and audit selections before external inference/evaluation')
            require(read(audit)['passed'] and read(audit)['selection'] == identity(selection), 'Selection audit changed')
            require(numeric_fit.exists() and read(numeric_fit)['passed']
                and read(numeric_fit)['task'] == identity(path), 'Numerical fit audit absent or changed')
            precisions = [v for v in a.precision or ['fp32'] if v == 'fp32' or task['model'] in ('dino-tcn', 'dino-transformer')]
            for precision in precisions:
                if a.action == 'infer':
                    commands.append([sys.executable, str(REPO/'scripts/run-neural-generalization.py'), 'infer',
                        '--task', str(path), '--output', str(fit), '--ids', str(ids_path), '--panel', str(a.panel),
                        '--precision', precision, '--device', a.device])
                else:
                    numeric_inference = fit/f'inference-numerical-audit-{precision}.json'
                    if not numeric_inference.exists():
                        commands.append([sys.executable, str(REPO/'scripts/audit-neural-generalization-numerics.py'),
                            '--phase', 'inference', '--task', str(path), '--fit', str(fit), '--panel', str(a.panel),
                            '--precision', precision, '--fit-audit', str(numeric_fit), '--output', str(numeric_inference)])
                    if a.action == 'audit-inference':
                        continue
                    require(a.inventory is not None and a.original_manifest is not None, 'Evaluation inventory/original inputs required')
                    destination = fit/f'evaluation-{precision}.json'
                    if destination.exists():
                        previous = read(destination)
                        require(previous['selection'] == identity(selection) and previous['panel'] == identity(a.panel), 'Existing evaluation lineage differs')
                        continue
                    commands.append([sys.executable, str(REPO/'scripts/evaluate-neural-generalization.py'), 'evaluate',
                        '--task', str(path), '--fit', str(fit), '--output', str(destination),
                        '--selection', str(selection), '--selection-audit', str(audit), '--panel', str(a.panel),
                        '--inventory', str(a.inventory), '--original-manifest', str(a.original_manifest), '--precision', precision])
        print(json.dumps({'starting': task['taskId'], 'action': a.action, 'commands': len(commands)}), flush=True)
        for command in commands:
            with (logs/(task['taskId'].replace('/', '__')+'-'+a.action+'.log')).open('a') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            require(result.returncode == 0, 'Task stage failed; inspect NAS log: '+task['taskId'])
        print(json.dumps({'completed': task['taskId'], 'action': a.action}), flush=True)


if __name__ == '__main__':
    main()
