#!/usr/bin/env python3
"""Publish/audit complete features with unchanged functions and one verified-file memo."""
from analysis.private_ledger import private_value
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(private_value('private-reference-0089'))
REPO = Path(__file__).resolve().parents[1]


def resources():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    win = int(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-NonInteractive', '-Command',
        '[Int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024'], text=True, timeout=30).strip())
    return {'linuxAvailableBytes': available, 'windowsPhysicalFreeBytes': win,
            'windowsCFreeBytes': disk.f_bavail*disk.f_frsize}


def gate():
    while True:
        state = resources()
        if (state['linuxAvailableBytes'] >= 6*1024**3 and state['windowsPhysicalFreeBytes'] >= 3*1024**3
                and state['windowsCFreeBytes'] >= 20*1024**3):
            return state
        print(json.dumps({'precisionPublicationPaused': 'resource floor', **state}), flush=True)
        time.sleep(30)


def main():
    assert Path('/mnt/freenas').is_mount()
    assert not Path('/proc/88183').exists(), 'Registered INT8 worker must have exited'
    assert (ROOT/'int8-queue-complete.json').is_file()
    names = ('features-complete.json', 'audit-features-complete.json', 'audit-core-complete-continuity-v1.json')
    assert not any((ROOT/name).exists() for name in names), 'Partial or completed precision publication already exists'
    panel = ROOT.parent/'panels-v1/precision-complete.json'
    assert not panel.exists(), 'Complete precision panel already exists'
    os.sched_setaffinity(0, {16, 17})
    for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        path = ROOT/'runtime'/name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    state = gate()
    sys.path.insert(0, str(REPO))
    from analysis.neural_context_development import identity, read, write_immutable
    from analysis.neural_generalization_inputs import verified
    from analysis import neural_generalization_feature_inference as encoders
    spec = importlib.util.spec_from_file_location('frozen_complete_feature_auditor', REPO/'scripts/audit-neural-generalization-features.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    registration = read(ROOT/'complete-publication-v1/registration.json')
    assert registration['source'] == identity(__file__)
    for reference in registration['dependencies']:
        verified(reference)
    assert read(ROOT/'int8-queue-complete.json')['plan'] == identity(ROOT/'int8-queue-plan.json')
    assert read(ROOT/'gpu-final-run-v1/exit.json')['released'] is True
    startup = {'kind': 'complete-precision-publication-execution-v1', 'at': datetime.now(timezone.utc).isoformat(),
        'source': identity(__file__), 'registration': identity(ROOT/'complete-publication-v1/registration.json'),
        'cpuAffinity': [16, 17], 'int8WorkerExitedPid': 88183,
        'int8QueueCompleted': identity(ROOT/'int8-queue-complete.json'),
        'gpuQueueCompleted': identity(ROOT/'gpu-queue-complete.json'),
        'stageQueueCompleted': identity(ROOT/'stage-queue-complete.json'),
        'auditPlan': identity(ROOT/'feature-audit-plan.json'), 'manifest': identity(ROOT/'inputs.json'),
        'resourceFloors': {'linuxAvailableBytes': 6*1024**3, 'windowsPhysicalFreeBytes': 3*1024**3,
                           'windowsCFreeBytes': 20*1024**3},
        'rootCpuGrantConfirmed': True, 'sameFrozenFunctionsAndChecks': True,
        'verifiedFileMemoRetainedAcrossPublicationAndAudit': True, 'gpuUsed': False, 'resources': state}
    write_immutable(ROOT/'complete-publication-v1/startup.json', startup)
    print(json.dumps({'startup': identity(ROOT/'complete-publication-v1/startup.json')}), flush=True)
    encoders.publish_index('complete', ROOT, scope='all')
    gate()
    auditor.audit(ROOT, 'complete', scope='all')
    gate()
    subprocess.run([sys.executable, '-B', str(REPO/'scripts/audit-neural-feature-index-continuity.py')],
                   cwd=REPO, check=True)
    gate()
    subprocess.run([sys.executable, '-B', str(REPO/'scripts/register-neural-generalization-panel.py'),
        '--manifest', str(ROOT/'inputs.json'), '--features', str(ROOT/'features-complete.json'),
        '--audit', str(ROOT/'audit-features-complete.json'),
        '--inventory', str(ROOT.parent/'inventory-v1/inventory-v2.json'), '--output', str(panel),
        '--precision-complete'], cwd=REPO, check=True)
    previous = read(ROOT.parent/'panels-v1/fp32-core.json')
    current = read(panel)
    assert current['precisionVariants'] == ['fp32', 'fp16', 'int8']
    assert current['features'] == identity(ROOT/'features-complete.json')
    assert current['featureAudit'] == identity(ROOT/'audit-features-complete.json')
    for key in ('features', 'featureAudit', 'precisionVariants'):
        previous.pop(key)
        current.pop(key)
    assert previous == current, 'Precision panel changed another panel field'
    output = ROOT/'complete-publication-v1/completed.json'
    write_immutable(output, {'kind': 'complete-precision-feature-publication-completed-v1', 'passed': True,
        'startup': identity(ROOT/'complete-publication-v1/startup.json'),
        'outputs': {name: identity(ROOT/name) for name in names}, 'precisionPanel': identity(panel),
        'corePanel': identity(ROOT.parent/'panels-v1/fp32-core.json'), 'allOtherPanelFieldsUnchanged': True})
    print(json.dumps({'completed': identity(output)}), flush=True)


if __name__ == '__main__':
    main()
