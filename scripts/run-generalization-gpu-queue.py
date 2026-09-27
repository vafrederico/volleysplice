#!/usr/bin/env python3
"""Prioritize fit encoders before full-panel and FP16 extraction, under one GPU grant."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_inference as encoders
from analysis.neural_context_development import identity, read, write_immutable

ROOT = encoders.stage.ROOT


def resources():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    return {'ramAvailableBytes': available, 'windowsCFreeBytes': disk.f_bavail*disk.f_frsize}


def register():
    encoders.stage.environment(ROOT)
    encoders.verify_plan(ROOT)
    plan = encoders.stage.verify_plan(ROOT/'stage-plan.json')
    new_fit = [r['id'] for r in plan['records'] if r['teaching']['allowed'] and not r['reuse']]
    encoders.require(len(new_fit) == 14, 'Expected14 new fit-eligible sources')
    result = {'kind': 'prioritized-generalization-gpu-queue-v1', 'source': identity(__file__),
              'encoderPlan': identity(ROOT/'encoder-plan.json'), 'stagePlan': identity(ROOT/'stage-plan.json'),
              'samplingAmendment': identity(ROOT/'timing-amendment-v1/protocol.json'),
              'phases': [{'name': 'new-fit-core', 'ids': new_fit, 'arms': ['fp32', 'mobile']},
                         {'name': 'full-core', 'ids': [r['id'] for r in plan['records']], 'arms': ['fp32', 'mobile']},
                         {'name': 'fp16', 'ids': [r['id'] for r in plan['records']], 'arms': ['fp16']}],
              'cpuAffinity': [10, 11], 'torchThreads': 2, 'maximumTorchGpuReservedBytes': int(1.75*1024**3),
              'minimumBeforeRecord': {'ramAvailableBytes': 4*1024**3, 'windowsCFreeBytes': 20*1024**3,
                                      'gpuFreeBytes': 4*1024**3},
              'requiresParentFullGpuGrant': True, 'noEncoderOrSamplingChange': True,
              'rule': 'Reuse frozen core functions/checkpoints. One GPU queue; original18 core features reused without requiring unrelated pending precision staging.'}
    write_immutable(ROOT/'gpu-queue-plan.json', result)
    print(json.dumps({'registered': True, 'plan': identity(ROOT/'gpu-queue-plan.json')}), flush=True)


def verify():
    plan = read(ROOT/'gpu-queue-plan.json')
    for key in ('source', 'encoderPlan', 'stagePlan', 'samplingAmendment'):
        encoders.verified(plan[key])
    encoders.verify_plan(ROOT)
    gate = read(ROOT/'encoder-engineering.json')
    encoders.require(gate['passed'] is True and gate['encoderPlan'] == plan['encoderPlan'], 'GPU engineering gate differs')
    return plan


def run():
    import torch
    encoders.stage.environment(ROOT)
    plan = verify()
    os.sched_setaffinity(0, set(plan['cpuAffinity']))
    staging = encoders.stage.verify_plan(ROOT/'stage-plan.json')
    sources = {r['id']: r for r in staging['records']}
    free, total = torch.cuda.mem_get_info()
    encoders.require(free >= plan['minimumBeforeRecord']['gpuFreeBytes'], 'Insufficient free GPU memory before encoder load')
    torch.cuda.set_per_process_memory_fraction(plan['maximumTorchGpuReservedBytes']/total)
    torch.cuda.reset_peak_memory_stats()
    operations, backbone = encoders.gpu_backbones()
    encoder_plan = encoders.verify_plan(ROOT)
    all_phases = []
    for phase in plan['phases']:
        completed_path = ROOT/'gpu-queue'/f"{phase['name']}.json"
        pending, records = list(phase['ids']), []
        while pending:
            state = resources()
            state['gpuFreeBytes'] = torch.cuda.mem_get_info()[0]
            if any(state[key] < limit for key, limit in plan['minimumBeforeRecord'].items()):
                print(json.dumps({'gpuQueuePaused': 'resource floor', 'phase': phase['name'], **state}), flush=True)
                time.sleep(30)
                continue
            ready = None
            for key in pending:
                source = sources[key]
                reused = all((arm in source['reuse'].get('dino', {}) if arm != 'mobile'
                              else 'mobile' in source['reuse']) for arm in phase['arms'])
                if reused or (ROOT/'staged'/key/'receipt.json').exists():
                    ready = source
                    break
            if ready is None:
                print(json.dumps({'gpuQueueWaiting': 'staging', 'phase': phase['name'], 'remaining': len(pending)}), flush=True)
                time.sleep(30)
                continue
            key = ready['id']
            stage_path = ROOT/'staged'/key/'receipt.json'
            staged = read(stage_path) if stage_path.exists() else None
            if staged is not None:
                encoders.require(staged['lineage'] == {'plan': plan['stagePlan'], 'source': ready}, 'Staging receipt differs')
                if staged.get('samplingAmendment'):
                    encoders.require(staged['samplingAmendment'] == plan['samplingAmendment'], 'Unbound timing amendment')
                    gate = read(ROOT/'timing-amendment-v1/engineering-parity.json')
                    encoders.require(gate['passed'] is True and gate['protocol'] == plan['samplingAmendment'], 'Timing gate differs')
            print(json.dumps({'gpuQueueStart': key, 'phase': phase['name'], 'remaining': len(pending), **state}), flush=True)
            started = time.perf_counter()
            outputs = {}
            for arm in phase['arms']:
                if arm == 'mobile':
                    outputs[arm] = encoders.mobile_record(ready, staged, backbone, ROOT)
                else:
                    outputs[arm] = encoders.dino_record(ready, staged, arm, operations[arm], encoder_plan, ROOT)
            torch.cuda.synchronize()
            records.append({'id': key, 'outputs': outputs, 'stagingReceipt': identity(stage_path) if staged is not None else None,
                            'wallSecondsIncludingVerification': time.perf_counter()-started, 'resourcesBefore': state})
            pending.remove(key)
            print(json.dumps({'gpuQueueComplete': key, 'phase': phase['name'], 'remaining': len(pending)}), flush=True)
        result = {'plan': identity(ROOT/'gpu-queue-plan.json'), 'phase': phase, 'records': records,
                  'peakAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakReservedBytes': torch.cuda.max_memory_reserved()}
        if completed_path.exists():
            previous = read(completed_path)
            encoders.require(previous['plan'] == result['plan'] and previous['phase'] == result['phase']
                             and {r['id']: r['outputs'] for r in previous['records']} == {r['id']: r['outputs'] for r in records},
                             'Completed GPU phase output identities changed')
        else:
            write_immutable(completed_path, result)
        all_phases.append(identity(completed_path))
        print(json.dumps({'gpuPhaseComplete': phase['name'], 'receipt': identity(completed_path)}), flush=True)
    write_immutable(ROOT/'gpu-queue-complete.json', {'plan': identity(ROOT/'gpu-queue-plan.json'), 'phases': all_phases})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run'))
    args = parser.parse_args()
    {'register': register, 'run': run}[args.phase]()
