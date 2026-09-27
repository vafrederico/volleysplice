#!/usr/bin/env python3
"""Pause only our verified historical runner before an encoder handoff.

Scheduling only: no fit, score, checkpoint, or registered source is changed.
The parent approved SIGSTOP during task 63/71 and explicit later SIGCONT.
"""
from analysis.private_ledger import private_value
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def owned_identity(pid, study):
    root = Path('/proc') / str(pid)
    command_bytes = (root / 'cmdline').read_bytes()
    command = [part.decode() for part in command_bytes.split(b'\0') if part]
    expected = ['-B', 'scripts/run-neural-historical-recall-sweep.py', 'fit', '--output', str(study)]
    if root.stat().st_uid != os.getuid() or Path(command[0]).name != 'python' or command[1:] != expected:
        raise RuntimeError('PID does not identify the expected owned historical runner')
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'uid': os.getuid(), 'startTicks': fields[19],
            'commandSha256': hashlib.sha256(command_bytes).hexdigest()}


def state(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]


def resource_snapshot():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    return {'wslAvailableKiB': int(memory['MemAvailable'].split()[0]), 'gpuUsedTotalMiB': gpu}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'watch', 'resume'))
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--before-task', type=int, choices=(64, 72), required=True)
    parser.add_argument('--encoder-exit-receipt', type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    if not str(study).startswith(private_value('private-reference-0060')):
        raise RuntimeError('Handoff receipts must stay on NAS')
    seed = 3407 if args.before_task == 64 else 20260918
    previous = study / f'refits/distilled_mobile_tcn/{seed}/outer-2'
    next_fit = study / f'refits/distilled_mobile_tcn/{seed}/outer-3'
    dest = study / f'resource-handoffs/before-task-{args.before_task}'
    pause_path, resume_path = dest / 'paused.json', dest / 'resumed.json'
    own = owned_identity(args.pid, study)
    if args.action == 'inspect':
        print(json.dumps({'runner': own, 'state': state(args.pid), 'previousExists': previous.exists(),
                          'nextExists': next_fit.exists(), 'source': reference(__file__)}), flush=True)
        return
    if args.action == 'watch':
        if next_fit.exists() or pause_path.exists():
            raise RuntimeError('Guard is too late or has already paused this transition')
        registration = {'kind': 'historical-student-resource-handoff-v1', 'createdAtUtc': now(),
                        'runner': own, 'source': reference(__file__), 'protocol': reference(study / 'protocol.json'),
                        'beforeTask': args.before_task, 'pauseTrigger': str(previous),
                        'numericalRecipeChanged': False,
                        'authorization': 'Parent approved verified SIGSTOP during prior temporal task and explicit SIGCONT after encoder exit.'}
        write_new(dest / 'guard.json', registration)
        print(json.dumps({'event': 'handoff-guard-armed', 'beforeTask': args.before_task, 'runner': own}), flush=True)
        while not previous.exists():
            if owned_identity(args.pid, study) != own or next_fit.exists():
                raise RuntimeError('Runner identity or task order changed')
            time.sleep(.5)
        if owned_identity(args.pid, study) != own or next_fit.exists():
            raise RuntimeError('Unsafe handoff state before pause')
        os.kill(args.pid, signal.SIGSTOP)
        for _ in range(100):
            if state(args.pid) in ('T', 't'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('SIGSTOP did not reach stopped state')
        if next_fit.exists():
            raise RuntimeError('Next student directory appeared; leave runner stopped for inspection')
        write_new(pause_path, {'kind': 'historical-student-resource-paused-v1', 'atUtc': now(),
                              'guard': reference(dest / 'guard.json'), 'runner': own,
                              'state': state(args.pid), 'resources': resource_snapshot(),
                              'nextStudentDirectoryAbsent': True})
        print(json.dumps({'event': 'historical-runner-paused', 'receipt': reference(pause_path)}), flush=True)
    else:
        paused = json.loads(pause_path.read_text())
        guard = json.loads((dest / 'guard.json').read_text())
        if paused['runner'] != own or guard['runner'] != own or state(args.pid) not in ('T', 't'):
            raise RuntimeError('Paused runner identity/state differs')
        if guard['source'] != reference(__file__) or guard['protocol'] != reference(study / 'protocol.json'):
            raise RuntimeError('Guard source or protocol changed')
        if next_fit.exists() or resume_path.exists() or not args.encoder_exit_receipt:
            raise RuntimeError('Need one explicit encoder-exit receipt before first resume')
        exit_ref = reference(args.encoder_exit_receipt)
        write_new(resume_path, {'kind': 'historical-student-resource-resume-v1', 'atUtc': now(),
                               'paused': reference(pause_path), 'runner': own, 'encoderExitReceipt': exit_ref,
                               'resources': resource_snapshot(), 'numericalRecipeChanged': False})
        if owned_identity(args.pid, study) != own:
            raise RuntimeError('Runner identity changed before SIGCONT')
        os.kill(args.pid, signal.SIGCONT)
        print(json.dumps({'event': 'historical-runner-resumed', 'receipt': reference(resume_path)}), flush=True)


if __name__ == '__main__':
    main()
