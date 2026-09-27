#!/usr/bin/env python3
"""One serial CPU staging queue, with teacher sources first and resource floors."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_stage as stage
from analysis.neural_context_development import identity, read, write_immutable


def resources():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    windows = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile',
                              '-Command', '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
                             check=True, capture_output=True, text=True)
    return {'ramAvailableBytes': available, 'windowsCFreeBytes': disk.f_bavail*disk.f_frsize,
            'windowsPhysicalFreeBytes': int(windows.stdout.strip())*1024}


def main():
    root = stage.environment(stage.ROOT)
    os.sched_setaffinity(0, set(range(12, 24)))
    plan = stage.verify_plan(root/'stage-plan.json')
    amendment = read(root/'timing-amendment-v1/protocol.json')
    gate = read(root/'timing-amendment-v1/engineering-parity.json')
    stage.require(gate['passed'] is True and gate['protocol'] == identity(root/'timing-amendment-v1/protocol.json'),
                  'Timing amendment full parity gate absent')
    manifest = read(root/'inputs.json')
    def priority(source):
        if not source['reuse']:
            return 0 if source['teaching']['allowed'] else 1 if source['sourceGroup'] in manifest['commonEvaluationGroups'] else 2
        return 3 if len(source['reuse']['dino']) < 3 else 4
    sources = sorted(plan['records'], key=lambda row: (priority(row), row['id']))
    floors = {'ramAvailableBytes': 4*1024**3, 'windowsCFreeBytes': 20*1024**3,
              'windowsPhysicalFreeBytes': int(1.5*1024**3)}
    registration = {'source': identity(__file__), 'stagePlan': identity(root/'stage-plan.json'),
                    'inputs': identity(root/'inputs.json'), 'timingAmendment': identity(root/'timing-amendment-v1/protocol.json'),
                    'timingEngineering': identity(root/'timing-amendment-v1/engineering-parity.json'),
                    'sourceOrder': [row['id'] for row in sources], 'cpuAffinity': list(range(12, 24)),
                    'minimumBeforeRecord': floors, 'workers': 1, 'outputs': 'NAS only; original and amended stagers unchanged.'}
    write_immutable(root/'stage-queue-plan.json', registration)
    results = []
    for source in sources:
        path = root/'staged'/source['id']/'receipt.json'
        if not path.exists():
            while True:
                state = resources()
                if all(state[key] >= minimum for key, minimum in floors.items()):
                    break
                print(json.dumps({'stageQueuePaused': 'resource floor', 'id': source['id'], **state}), flush=True)
                time.sleep(30)
            print(json.dumps({'stageQueueStart': source['id'], **state}), flush=True)
            if source['id'] in amendment['affectedRecordings']:
                subprocess.run([sys.executable, '-B', str(stage.REPO/'scripts/prepare-generalization-timing-amendment.py'), 'stage'], check=True)
            else:
                stage.stage_record(root/'stage-plan.json', source['id'])
        receipt = read(path)
        stage.require(receipt['lineage'] == {'plan': registration['stagePlan'], 'source': source}, 'Completed staging source differs')
        results.append({'id': source['id'], 'receipt': identity(path)})
        print(json.dumps({'stageQueueComplete': source['id'], 'remaining': len(sources)-len(results)}), flush=True)
    write_immutable(root/'stage-queue-complete.json', {'plan': identity(root/'stage-queue-plan.json'), 'records': results})


if __name__ == '__main__':
    main()
