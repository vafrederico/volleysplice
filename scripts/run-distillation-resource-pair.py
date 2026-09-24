#!/usr/bin/env python3
"""Resource-only parallel seed with a hard handoff guard; no fitting-code changes."""
from analysis.private_ledger import private_value
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def identity(pid):
    folder = Path('/proc') / str(pid)
    command = (folder / 'cmdline').read_bytes()
    if b'scripts/run-neural-mobile-distillation.py' not in command or b'--seed' in command:
        raise RuntimeError('PID is not the expected owned all-seed runner')
    if folder.stat().st_uid != os.getuid():
        raise RuntimeError('Runner has a different owner')
    start = (folder / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return {'pid': pid, 'startTicks': start, 'commandSha256': hashlib.sha256(command).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith(private_value('private-reference-0060')):
        raise RuntimeError('Resource artifacts must be on NAS')
    study = root / 'distilled-mobile-v1'
    parent = identity(args.parent_pid)
    registration = json.loads((study / 'preregistration.json').read_text())
    if (study / 'fits/1729').exists() or (study / 'fits/20260918').exists():
        raise RuntimeError('Guard must start before either later seed')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    amendment = {'kind': 'resource-amendment-2', 'createdAtUtc': datetime.now(timezone.utc).isoformat(),
        'sourceSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'registrationSha256': registration['sha256'], 'parent': parent,
        'concurrentGpuWorkersMaximum': 2, 'torchThreadsPerWorker': 2,
        'secondWorkerSeed': 20260918, 'numericalRecipeChanged': False,
        'guard': 'SIGSTOP original runner when fits/1729 appears; resume only after third-seed result verifies and worker exits0.',
        'timingLimit': 'Concurrent desktop timings; not exclusive throughput or phone measurements.',
        'resourceSnapshot': {'cFreeGiB': shutil.disk_usage('/mnt/c').free / 1024**3,
            'wslAvailableGiB': int(memory['MemAvailable'].split()[0]) / 1024**2, 'gpuUsedTotalMiB': gpu}}
    with (root / 'resource-amendment-2.json').open('x') as stream:
        json.dump(amendment, stream, indent=2)
        stream.write('\n')
    events = root / 'resource-amendment-2-events.jsonl'

    def event(kind, **values):
        row = {'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}
        with events.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def signal_parent(value):
        if identity(args.parent_pid) != parent:
            raise RuntimeError('Parent process identity changed')
        os.kill(args.parent_pid, value)

    repo = Path(__file__).resolve().parents[1]
    child = subprocess.Popen([sys.executable, str(repo / 'scripts/run-neural-mobile-distillation.py'),
        '--output', str(study), '--seed', '20260918'], cwd=repo)
    paused = False
    try:
        event('second-worker-started', pid=child.pid)
        while child.poll() is None:
            if not paused and (study / 'fits/1729').exists():
                signal_parent(signal.SIGSTOP)
                paused = True
                event('parent-suspended-before-second-seed-work')
            time.sleep(.5)
        if child.returncode != 0:
            event('second-worker-failed', returncode=child.returncode, parentSuspended=paused)
            raise RuntimeError('Worker failed; inspect artifacts before resuming a suspended parent')
        result = json.loads((study / 'result-20260918.json').read_text())
        if result['seed'] != 20260918 or result['contractSha256'] != registration['sha256']:
            raise RuntimeError('Third-seed completion identity differs; do not resume parent')
        event('second-worker-completion-verified', completeEvaluation=result['completeEvaluation'])
        if paused:
            signal_parent(signal.SIGCONT)
            paused = False
            event('parent-resumed')
    except BaseException:
        # A failed guardian must never leave two writers free to enter seed3.
        # Preserve both jobs' artifacts; root can inspect and recover explicitly.
        if not paused:
            signal_parent(signal.SIGSTOP)
            paused = True
        raise


if __name__ == '__main__':
    main()
