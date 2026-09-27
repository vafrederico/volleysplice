#!/usr/bin/env python3
"""Run a bounded serial subset of previously frozen fitting tasks on the NAS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.neural_context_development import identity, read


def memory_snapshot():
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1])*1024
    disk = os.statvfs('/mnt/c')
    return {'availableRamBytes': available, 'windowsCFreeBytes': disk.f_bavail*disk.f_frsize}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-dir', type=Path, required=True)
    parser.add_argument('--model', action='append')
    parser.add_argument('--variant', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    registration = read(args.registration_dir/'registration.json')
    tasks = []
    for task_id in registration['contract']['taskIds']:
        path = args.registration_dir/'tasks'/(task_id.replace('/', '__')+'.json')
        task = read(path)
        if args.model and task['model'] not in args.model:
            continue
        if args.variant and task['variant'] not in args.variant:
            continue
        if task['variant'] == 'original-corpus':
            output = args.registration_dir/'fits'/task['model']/f'seed-{task["seed"]}'
        else:
            output = args.registration_dir/'fits'/task['variant']/task['model']/f'split-{task["splitSeed"]}'
        if (output/'fit-result.json').exists():
            continue
        tasks.append((path, task, output))
    if args.limit is not None:
        tasks = tasks[:args.limit]
    queue = {'createdAt': datetime.now(timezone.utc).isoformat(),
             'registration': identity(args.registration_dir/'registration.json'),
             'source': identity(__file__), 'taskIds': [t['taskId'] for _, t, _ in tasks]}
    logs = args.registration_dir/'queue-logs'
    logs.mkdir(exist_ok=True)
    queue_path = logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
    with queue_path.open('x') as stream:
        json.dump(queue, stream, indent=2)
    for path, task, output in tasks:
        resources = memory_snapshot()
        if resources['availableRamBytes'] < 3*1024**3 or resources['windowsCFreeBytes'] < 20*1024**3:
            raise RuntimeError('Resource floor reached before next fit: '+json.dumps(resources))
        print(json.dumps({'starting': task['taskId'], 'resources': resources}), flush=True)
        log_path = logs/(task['taskId'].replace('/', '__')+'.log')
        with log_path.open('a') as log:
            result = subprocess.run([sys.executable, str(REPO/'scripts/run-neural-generalization.py'), 'fit',
                '--task', str(path), '--output', str(output), '--device', args.device], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Fit failed ({result.returncode}): {task["taskId"]}; inspect {log_path}')
        print(json.dumps({'completed': task['taskId'], 'receipt': identity(output/'fit-result.json')}), flush=True)


if __name__ == '__main__':
    main()
