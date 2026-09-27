#!/usr/bin/env python3
"""Guard and record serial CPU precision stages without changing their numerical code."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io

GIB = 1024 ** 3
POLICY = {'cpuAffinity': [16, 17], 'minimumNice': 10, 'gpuUsed': False,
          'minimumWindowsPhysicalFreeBytes': 2 * GIB, 'minimumLinuxAvailableBytes': 8 * GIB,
          'minimumCFreeBytes': 20 * GIB, 'monitorSeconds': 15, 'serialStagesOnly': True,
          'resourcePolicyOnly': True, 'numericalSourceChanged': False}


def snapshot():
    raw = subprocess.check_output([
        '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-NonInteractive',
        '-Command', '[Console]::Write((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024)'], text=True)
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    values = {'windowsPhysicalFreeBytes': int(raw.strip()), 'linuxAvailableBytes': available,
              'windowsCFreeBytes': disk.f_bavail * disk.f_frsize}
    return {'atUTC': datetime.now(timezone.utc).isoformat(), **values,
            'startupFloorsPassed': values['windowsPhysicalFreeBytes'] >= 2 * GIB
                and values['linuxAvailableBytes'] >= 8 * GIB and values['windowsCFreeBytes'] >= 20 * GIB}


def run(root, stage):
    io.require(root.resolve().is_relative_to('/mnt/freenas') and sorted(os.sched_getaffinity(0)) == [16, 17]
               and os.getpriority(os.PRIO_PROCESS, 0) >= 10
               and os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'NAS CPU-only execution scope differs')
    for name in ('TMPDIR', 'TMP', 'TEMP', 'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH'):
        path = root / 'runtime' / name.lower(); path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    commands = {
        'replay': ('run-neural-historical-precision-sweep.py', ['run', '--output', str(root)]),
        'audit': ('audit-neural-historical-precision-sweep.py', ['--study', str(root), '--output', str(root/'audit.json')]),
        'project': ('project-neural-historical-precision-sweep.py', ['--study', str(root)])}
    name, arguments = commands[stage]
    protocol_ref = io.identity(root/'protocol.json'); protocol = io.read(protocol_ref['path'])
    script = io.identity(REPO/'scripts'/name)
    io.require(script in protocol['code'], 'Stage numerical script is not bound by the frozen protocol')
    receipt = root / f'cpu-{stage}-start.json'; completion = root / f'cpu-{stage}-completed.json'
    io.require(not receipt.exists() and not completion.exists(), 'Stage execution already registered')
    measured = snapshot()
    if not measured['startupFloorsPassed']:
        print(json.dumps({'stage': stage, 'status': 'resource-wait', 'snapshot': measured}), flush=True)
        return 2
    command = [sys.executable, '-B', script['path'], *arguments]
    io.write_new(receipt, {'kind': 'historical-precision-cpu-stage-resource-grant-v1',
        'stage': stage, 'protocol': protocol_ref, 'policy': POLICY, 'preflight': measured,
        'runner': io.identity(__file__), 'numericalScript': script, 'command': command,
        'noExclusiveTimingClaim': True})
    peak, samples = 0, 0
    with (root/f'cpu-{stage}-resources.jsonl').open('x') as log:
        child = subprocess.Popen(command, cwd=REPO)
        while True:
            try:
                code = child.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                try:
                    measured = snapshot()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    measured = {'atUTC': datetime.now(timezone.utc).isoformat(), 'monitorError': str(error)}
                status = Path(f'/proc/{child.pid}/status')
                rss = 0
                try:
                    rss = next((int(line.split()[1])*1024 for line in status.read_text().splitlines()
                                if line.startswith('VmRSS:')), 0)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                peak = max(peak, rss); samples += 1
                log.write(json.dumps({**measured, 'pid': child.pid, 'rssBytes': rss}) + '\n'); log.flush()
    io.write_new(completion, {'kind': 'historical-precision-cpu-stage-resource-completion-v1',
        'stage': stage, 'start': io.identity(receipt), 'pid': child.pid, 'exitCode': code,
        'monitorSamples': samples, 'maximumObservedRssBytes': peak,
        'resourceLog': io.identity(root/f'cpu-{stage}-resources.jsonl'),
        'runner': io.identity(__file__), 'atUTC': datetime.now(timezone.utc).isoformat()})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--stage', choices=('replay', 'audit', 'project'), required=True)
    args = parser.parse_args()
    io.require(args.study.resolve().is_relative_to('/mnt/freenas'), 'Stage lock must use NAS')
    with (args.study/'cpu-stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(args.study, args.stage))
