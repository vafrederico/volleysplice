#!/usr/bin/env python3
"""Run unchanged student reuse queues with external, observational Linux RSS sampling."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0061'))
GIB = 1024**3
FIELDS = ('VmRSS', 'VmHWM', 'VmSwap', 'RssAnon', 'RssFile', 'RssShmem')


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stat_identity(text):
    tail = text.rsplit(')', 1)[1].split()
    return int(tail[1]), int(tail[19])


def memory_status(text):
    result = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        if key in FIELDS:
            fields = rest.split()
            if fields[1:] != ['kB']:
                raise ValueError('Unexpected Linux memory units')
            result[key+'Bytes'] = int(fields[0])*1024
    return result


def argument(command, flag):
    return command[command.index(flag)+1] if flag in command else None


def artifact_phase(command):
    names = {Path(value).name for value in command}
    if 'run-neural-generalization.py' in names and 'fit' in command:
        output = argument(command, '--output')
        if not output:
            return 'fit-phase-unknown'
        folder = Path(output)
        if (folder/'temporal/completed.json').exists():
            return 'fit-finalization'
        if (folder/'temporal').exists():
            return 'temporal-input-preparation-or-training'
        if (folder/'student/completed.json').exists():
            return 'student-feature-extraction-or-transition'
        return 'student-input-loading-or-training'
    if 'audit-neural-generalization-numerics.py' in names:
        return 'independent-numerical-audit'
    if 'audit-neural-generalization-reuse.py' in names:
        return 'independent-reuse-audit'
    if 'reuse-neural-generalization-fit.py' in names:
        return 'reuse-and-calibration'
    if 'run-neural-generalization-reuse-queue.py' in names:
        return 'queue-controller'
    return 'supervisor-or-resource-inspection'


def connected_rows(rows, root_pid):
    """Retain only chains whose parents survived this same identity revalidation."""
    by_pid = {row['pid']: row for row in rows}
    connected = {root_pid} if root_pid in by_pid else set()
    while True:
        expanded = connected | {pid for pid, row in by_pid.items() if row['parentPid'] in connected}
        if expanded == connected:
            return [row for row in rows if row['pid'] in connected]
        connected = expanded


def process_tree(root_pid):
    processes = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            parent, start = stat_identity((directory/'stat').read_text())
            processes[int(directory.name)] = (parent, start)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    result = []
    for pid in sorted(selected):
        if pid not in processes:
            continue
        directory = Path('/proc')/str(pid)
        try:
            command = [v.decode(errors='replace') for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
            memory = memory_status((directory/'status').read_text())
            # A PID recycled during the sample cannot inherit the old identity.
            if stat_identity((directory/'stat').read_text()) != processes[pid]:
                continue
            parent, start = processes[pid]
            result.append({'pid': pid, 'parentPid': parent, 'startTimeTicks': start,
                           'command': command, 'artifactPhase': artifact_phase(command), **memory})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return connected_rows(result, root_pid)


def sample_memory(stop, folder, parent_pid, summary):
    with (folder/'samples.jsonl').open('x', encoding='utf-8') as stream:
        while not stop.is_set():
            now = datetime.now(timezone.utc).isoformat()
            try:
                records = process_tree(parent_pid)
                for row in records:
                    stream.write(json.dumps({'utc': now, **row})+'\n')
                    key = f"{row['pid']}:{row['startTimeTicks']}"
                    item = summary['processes'].setdefault(key, {'pid': row['pid'], 'startTimeTicks': row['startTimeTicks'],
                        'command': row['command'], 'firstSampleUtc': now, 'sampleCount': 0,
                        'maximumObservedVmHwmBytes': 0, 'maximumObservedSwapBytes': 0, 'phases': {}})
                    item['sampleCount'] += 1
                    item['lastSampleUtc'] = now
                    item['maximumObservedVmHwmBytes'] = max(item['maximumObservedVmHwmBytes'], row.get('VmHWMBytes', 0))
                    item['maximumObservedSwapBytes'] = max(item['maximumObservedSwapBytes'], row.get('VmSwapBytes', 0))
                    phase = item['phases'].setdefault(row['artifactPhase'], {'samples': 0, 'sampledMaximumRssBytes': 0})
                    phase['samples'] += 1
                    phase['sampledMaximumRssBytes'] = max(phase['sampledMaximumRssBytes'], row.get('VmRSSBytes', 0))
                stream.flush()
            except Exception as error:
                # Observation must never change, interrupt, or retry a fit.
                summary['inspectionErrors'].append({'utc': now, 'type': type(error).__name__, 'message': str(error)})
                stream.write(json.dumps({'utc': now, 'inspectionError': str(error)})+'\n')
                stream.flush()
            stop.wait(.5)


def resources():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                     text=True, timeout=15).strip())
    ps = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    win = int(subprocess.check_output([ps, '-NoProfile', '-NonInteractive', '-Command',
        '[Int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024'], text=True, timeout=30).strip())
    return {'linuxAvailableBytes': available, 'windowsPhysicalFreeBytes': win,
            'windowsCFreeBytes': disk.f_bavail*disk.f_frsize, 'gpuFreeMiB': gpu}


def validate_grant(grant, expected):
    for name, value in expected.items():
        if grant.get(name) != value:
            raise ValueError('Student execution grant differs: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--grant-sha256', required=True)
    parser.add_argument('--av-exit', type=Path, required=True)
    parser.add_argument('--av-exit-sha256', required=True)
    parser.add_argument('--required-exited-pid', type=int, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    assert folder.is_relative_to(ROOT) and Path('/mnt/freenas').is_mount()
    assert identity(args.grant)['sha256'] == args.grant_sha256
    assert identity(args.av_exit)['sha256'] == args.av_exit_sha256
    assert json.loads(args.av_exit.read_text())['released'] is True
    feature_exit = ROOT/'features-v1/gpu-final-run-v1/exit.json'
    assert json.loads(feature_exit.read_text())['released'] is True
    assert all(not (Path('/proc')/str(pid)).exists() for pid in args.required_exited_pid)
    queue = REPO/'scripts/run-neural-generalization-reuse-queue.py'
    plans = [ROOT/name/'reuse-plan.json' for name in ('randomized-variants-v1', 'export-proxy-v1')]
    expected_grant = {'kind': 'generalization-profiled-student-worker-grant-v1',
        'source': identity(__file__), 'queue': identity(queue), 'reusePlans': [identity(p) for p in plans],
        'featureExit': identity(feature_exit), 'avExit': identity(args.av_exit),
        'requiredExitedPids': args.required_exited_pid, 'cpuAffinity': [10, 11],
        'models': ['distilled-mobile-tcn'], 'sampleIntervalSeconds': .5,
        'startupMinimumAvailableRamBytes': 8*GIB, 'startupMinimumGpuFreeMiB': 5*1024,
        'startupMinimumWindowsPhysicalFreeBytes': 3*GIB, 'minimumWindowsCFreeBytes': 20*GIB,
        'numericalRecipesChanged': False}
    validate_grant(json.loads(args.grant.read_text()), expected_grant)
    os.sched_setaffinity(0, {10, 11})
    for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        path = folder/'runtime'/name.lower(); path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    commands = [[sys.executable, '-B', str(queue), '--plan', str(plan), '--action', 'fit',
                 '--model', 'distilled-mobile-tcn', '--device', 'cuda'] for plan in plans]
    contract = {'kind': 'observed-student-reuse-queue-v1', 'source': identity(__file__), 'queue': identity(queue),
        'grant': identity(args.grant), 'avExit': identity(args.av_exit), 'featureExit': identity(feature_exit),
        'requiredExitedPids': args.required_exited_pid, 'reusePlans': [identity(p) for p in plans], 'commands': commands,
        'cpuAffinity': [10, 11], 'pollSeconds': .5, 'numericalRecipesChanged': False,
        'observationalOnly': True, 'phaseDefinition': 'Existing student/temporal artifact existence, not instrumentation inside frozen fitting code.',
        'limits': 'Phase RSS maxima are sampled. VmHWM is cumulative for each PID/start identity and is not a phase-specific peak. RSS may include shared/reclaimable pages. No CUDA-memory or speed inference.'}
    write_new(folder/'plan.json', contract)
    summary = {'kind': 'student-process-memory-observations-v1', 'plan': identity(folder/'plan.json'),
               'processes': {}, 'inspectionErrors': [], 'queueResults': []}
    stop = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(stop, folder, os.getpid(), summary), daemon=True)
    sampler.start()
    try:
        for index, command in enumerate(commands):
            while True:
                state = resources()
                if (state['linuxAvailableBytes'] >= 8*GIB and state['windowsPhysicalFreeBytes'] >= 3*GIB
                        and state['windowsCFreeBytes'] >= 20*GIB and state['gpuFreeMiB'] >= 5*1024):
                    break
                print(json.dumps({'studentQueueWaiting': 'startup resource floor', **state}), flush=True)
                time.sleep(30)
            write_new(folder/f'launch-{index}.json', {'plan': summary['plan'], 'resources': state, 'command': command})
            result = subprocess.run(command, cwd=REPO)
            summary['queueResults'].append({'command': command, 'returnCode': result.returncode})
            if result.returncode:
                raise RuntimeError('Registered student queue failed; no retry or next queue started')
    finally:
        stop.set(); sampler.join()
        summary['completedBothQueues'] = len(summary['queueResults']) == 2 and all(v['returnCode'] == 0 for v in summary['queueResults'])
        summary['samples'] = identity(folder/'samples.jsonl')
        write_new(folder/'summary.json', summary)
        print(json.dumps({'memorySummary': identity(folder/'summary.json'), 'completedBothQueues': summary['completedBothQueues']}), flush=True)


if __name__ == '__main__':
    main()
