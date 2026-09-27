#!/usr/bin/env python3
"""Temporary, opt-in host-pressure relief; no training or persistent settings.

Default/--once only observes. --watch --permit-cache-flush runs the registered
resource policy for at most12hours. STOP or study progress.status=complete exits.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024**3
INTERVAL = 30
COOLDOWN = 180
MAX_SECONDS = 12*60*60
REPO = Path(__file__).resolve().parents[1]
WINDOWS_PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
WINDOWS_WSL = '/mnt/c/Windows/System32/wsl.exe'
FLUSH_COMMAND = 'sync; echo 3 > /proc/sys/vm/drop_caches'
PS_QUERY = ("$ErrorActionPreference='Stop'; "
    "$system=Get-CimInstance Win32_OperatingSystem; "
    "$disk=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\"; "
    "@{windowsFreePhysicalBytes=([int64]$system.FreePhysicalMemory*1024); "
    "windowsCFreeBytes=[int64]$disk.FreeSpace} | ConvertTo-Json -Compress")
STOP_REQUESTED = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def decision(snapshot, *, seconds_since_attempt, elapsed_seconds, stop_requested=False, progress_status='running'):
    """Pure resource policy; cannot invoke a command or inspect model outputs."""
    if stop_requested:
        return {'action': 'stop', 'reason': 'STOP file or process signal'}
    if progress_status == 'complete':
        return {'action': 'stop', 'reason': 'study progress complete'}
    if elapsed_seconds >= MAX_SECONDS:
        return {'action': 'stop', 'reason': 'maximum12hours reached'}
    required = ('windowsFreePhysicalBytes', 'linuxCachedBytes', 'linuxAvailableBytes')
    if not all(isinstance(snapshot.get(k), (int, float)) and not isinstance(snapshot[k], bool)
               and math.isfinite(snapshot[k]) and snapshot[k] >= 0 for k in required):
        return {'action': 'observe', 'reason': 'invalid or unavailable resource sample'}
    gates = {'windowsPhysicalBelow2GiB': snapshot['windowsFreePhysicalBytes'] < 2*GIB,
             'linuxCachedAbove4GiB': snapshot['linuxCachedBytes'] > 4*GIB,
             'linuxAvailableAbove6GiB': snapshot['linuxAvailableBytes'] > 6*GIB}
    if not all(gates.values()):
        return {'action': 'observe', 'reason': 'pressure gates not all met', 'gates': gates}
    if seconds_since_attempt < COOLDOWN:
        return {'action': 'observe', 'reason': '180second minimum cooldown', 'gates': gates,
                'cooldownRemainingSeconds': max(0., COOLDOWN-seconds_since_attempt)}
    return {'action': 'flush', 'reason': 'all resource pressure gates and cooldown met', 'gates': gates}


def root_command(distro, command):
    require(isinstance(distro, str) and distro.strip(), 'Current WSL distribution is required')
    require(command in (FLUSH_COMMAND, 'id -u'), 'Unregistered privileged command')
    return [WINDOWS_WSL, '--distribution', distro, '--user', 'root', '--', 'sh', '-c', command]


def sample_resources():
    before = time.monotonic()
    process = subprocess.run([WINDOWS_PS, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', PS_QUERY],
                             capture_output=True, text=True, timeout=20, check=True)
    windows = json.loads(process.stdout.lstrip('\ufeff').strip())
    require(set(windows) == {'windowsFreePhysicalBytes', 'windowsCFreeBytes'}, 'Unexpected Windows resource schema')
    require(all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in windows.values()),
            'Invalid Windows resource values')
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'Cached', 'MemTotal', 'SwapFree', 'SwapTotal', 'Dirty', 'Writeback'):
            fields = value.split(); require(fields[1:] == ['kB'], 'Unexpected Linux memory units')
            memory[key] = int(fields[0])*1024
    return {**windows, 'linuxAvailableBytes': memory['MemAvailable'], 'linuxCachedBytes': memory['Cached'],
            'linuxMemory': memory, 'sampleUtc': utc(), 'querySeconds': time.monotonic()-before}


def prepare_plan(root, path):
    distro = os.environ.get('WSL_DISTRO_NAME')
    require(os.name == 'posix' and os.geteuid() != 0 and distro, 'Run as the regular WSL user')
    amendment_path = root/'execution-resource-amendment-v2.json'
    amendment = json.loads(amendment_path.read_text())
    protocol_ref = amendment['protocol']
    protocol_path = Path(protocol_ref['path'])
    if not protocol_path.is_absolute():
        protocol_path = REPO/protocol_path
    protocol = identity(protocol_path)
    require(protocol['sha256'] == protocol_ref['sha256'], 'Bound study protocol changed')
    contract = {'kind': 'temporary-host-memory-watch-contract-v1', 'root': str(root),
        'source': identity(__file__), 'tests': identity(REPO/'analysis/tests/test_host_memory_watch.py'),
        'resourceAmendment': identity(amendment_path), 'studyProtocol': protocol,
        'distribution': distro, 'regularUserUid': os.geteuid(), 'intervalSeconds': INTERVAL,
        'minimumSecondsBetweenAttempts': COOLDOWN, 'startupCooldownSeconds': COOLDOWN,
        'maximumRunSeconds': MAX_SECONDS,
        'thresholds': {'windowsFreePhysicalBytesStrictlyBelow': 2*GIB,
                       'linuxCachedBytesStrictlyAbove': 4*GIB, 'linuxAvailableBytesStrictlyAbove': 6*GIB},
        'windowsQueryExecutable': WINDOWS_PS, 'windowsQueryCommand': PS_QUERY,
        'allowedFlushCommand': root_command(distro, FLUSH_COMMAND),
        'allowedReadOnlyRootProbe': root_command(distro, 'id -u'),
        'stopFile': str(path.parent/'STOP'), 'progressPath': str(root/'progress.json'),
        'progressCompleteStatus': 'complete', 'commandTimeoutSeconds': 45,
        'failedFlushPolicy': 'Stop helper; never repeat or request elevation after failure/timeout.',
        'failedInspectionPolicy': 'Observe only; stop after3 consecutive resource/progress inspection errors.',
        'effects': 'sync then discard reclaimable Linux page cache, dentries and inodes; no source/data/model files deleted, no services restarted, no persistent settings changed',
        'trainingRecipeChanged': False, 'gpuUsed': False, 'labelsRead': False,
        'wallTimeComparisonsInvalidDuringConcurrentWork': True}
    if path.exists():
        saved = json.loads(path.read_text())
        require(saved['contract'] == contract, 'Resource plan changed; preserve old plan and create a reviewed new plan')
    else:
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'createdAtUtc': utc(), 'contract': contract}, stream, indent=2); stream.write('\n')
    return contract, identity(path)


def signal_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run(args):
    root = args.root.resolve()
    require(root.is_relative_to(private_value('private-reference-0060')), 'Study and all monitor artifacts must be on NAS')
    folder = root/'resource-watch-v1'; folder.mkdir(exist_ok=True)
    plan_path = args.plan.resolve() if args.plan else folder/'plan.json'
    require(plan_path.parent == folder and not plan_path.is_symlink(), 'Plan must be directly in the NAS watch directory')
    contract, plan_ref = prepare_plan(root, plan_path)
    require(not args.permit_cache_flush or args.watch, 'Inspection mode never permits cache flushing')
    require(not args.watch or args.permit_cache_flush, 'Long watch requires explicit --permit-cache-flush')
    # Held for the entire watch. Advisory ownership disappears at process exit;
    # the lock file stays on NAS, so no file deletion is needed.
    lock = None
    if args.watch:
        lock = (folder/'active.lock').open('a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+f'-pid{os.getpid()}'
    log_path = folder/(('watch-' if args.watch else 'inspection-')+suffix+'.jsonl')
    log = log_path.open('x', encoding='utf-8')

    def event(kind, **values):
        item = {'utc': utc(), 'event': kind, **values}
        line = json.dumps(item, allow_nan=False)
        log.write(line+'\n'); log.flush()
        print(line, flush=True)

    start = time.monotonic()
    last_attempt = start  # Conservatively respects a recent manual cache flush.
    failures, attempts = 0, 0
    stopped = 'inspection complete'
    signal.signal(signal.SIGTERM, signal_stop); signal.signal(signal.SIGINT, signal_stop)
    try:
        event('start', pid=os.getpid(), regularUserUid=os.geteuid(), mode='watch' if args.watch else 'inspection-only',
              plan=plan_ref, log=str(log_path), startupCooldownSeconds=COOLDOWN)
        if args.probe_root:
            probe = subprocess.run(contract['allowedReadOnlyRootProbe'], capture_output=True, text=True, timeout=20, check=True)
            require(probe.stdout.strip() == '0', 'Nested WSL read-only root identity differs')
            event('root-interop-readonly-probe', command='id -u', returnedUid=0, cacheFlushPerformed=False)
        while True:
            stop = STOP_REQUESTED or Path(contract['stopFile']).exists()
            gate = decision({}, seconds_since_attempt=time.monotonic()-last_attempt,
                            elapsed_seconds=time.monotonic()-start, stop_requested=stop)
            if gate['action'] == 'stop':
                stopped = gate['reason']; break
            try:
                status = json.loads(Path(contract['progressPath']).read_text())['status']
                require(isinstance(status, str), 'Invalid study progress status')
                gate = decision({}, seconds_since_attempt=time.monotonic()-last_attempt,
                                elapsed_seconds=time.monotonic()-start, stop_requested=stop, progress_status=status)
                if gate['action'] == 'stop':
                    stopped = gate['reason']; break
                snapshot = sample_resources()
                failures = 0
                gate = decision(snapshot, seconds_since_attempt=time.monotonic()-last_attempt,
                                elapsed_seconds=time.monotonic()-start, stop_requested=stop, progress_status=status)
                event('inspection', resources=snapshot, decision=gate, actionEnabled=args.watch)
            except Exception as error:
                failures += 1
                event('inspection-error', exceptionType=type(error).__name__, message=str(error), consecutiveFailures=failures)
                gate = {'action': 'observe'}
                if failures >= 3 or not args.watch:
                    stopped = 'inspection failed; no cache action'; break
            if gate['action'] == 'stop':
                stopped = gate['reason']; break
            if not args.watch:
                break
            if gate['action'] == 'flush':
                # Refresh pressure and stop state immediately before the only mutation.
                snapshot = sample_resources()
                status = json.loads(Path(contract['progressPath']).read_text())['status']
                require(isinstance(status, str), 'Invalid study progress status before mutation')
                gate = decision(snapshot, seconds_since_attempt=time.monotonic()-last_attempt,
                    elapsed_seconds=time.monotonic()-start,
                    stop_requested=STOP_REQUESTED or Path(contract['stopFile']).exists(), progress_status=status)
                event('pre-flush-recheck', resources=snapshot, decision=gate)
                if gate['action'] == 'stop':
                    stopped = gate['reason']; break
                if gate['action'] == 'flush':
                    last_attempt = time.monotonic(); attempts += 1
                    event('cache-flush-attempt', attempt=attempts, command=contract['allowedFlushCommand'])
                    try:
                        result = subprocess.run(contract['allowedFlushCommand'], capture_output=True, text=True, timeout=45, check=True)
                        event('cache-flush-complete', attempt=attempts, elapsedSeconds=time.monotonic()-last_attempt,
                              returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
                    except Exception as error:
                        stopped = 'cache flush failed or timed out; manual review required'
                        event('cache-flush-failed', attempt=attempts, exceptionType=type(error).__name__, message=str(error)); break
            deadline = time.monotonic()+INTERVAL
            while time.monotonic() < deadline:
                if STOP_REQUESTED or Path(contract['stopFile']).exists():
                    break
                time.sleep(min(1., max(0., deadline-time.monotonic())))
    except Exception as error:
        stopped = 'helper error; stopped without retry'
        event('helper-error', exceptionType=type(error).__name__, message=str(error))
        raise
    finally:
        event('exit', reason=stopped, attempts=attempts, elapsedSeconds=time.monotonic()-start)
        log.close()
        if lock:
            lock.close()
    return {'log': str(log_path), 'plan': plan_ref, 'attempts': attempts, 'exitReason': stopped}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    p.add_argument('--permit-cache-flush', action='store_true')
    p.add_argument('--probe-root', action='store_true', help='Read-only nested WSL id -u probe; never an elevation prompt')
    args = p.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
