#!/usr/bin/env python3
"""Consume completed immutable frame stages with one bounded two-thread CPU encoder."""
from pathlib import Path
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_inference as encoders
from analysis.neural_context_development import identity, read, write_immutable


def resources():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    disk = os.statvfs('/mnt/c')
    return {'ramAvailableBytes': available, 'windowsCFreeBytes': disk.f_bavail*disk.f_frsize}


def main():
    root = encoders.stage.environment(encoders.stage.ROOT)
    os.sched_setaffinity(0, {8, 9})
    plan = encoders.verify_plan(root)
    staged_plan = encoders.stage.verify_plan(root/'stage-plan.json')
    gate = read(root/'encoder-engineering.json')
    encoders.require(gate['passed'] is True and gate['encoderPlan'] == identity(root/'encoder-plan.json'), 'Encoder gate differs')
    q = read(root/'int8-wrapper-engineering.json')
    encoders.require(q['passed'] is True, 'INT8 wrapper gate absent')
    registration = {'source': identity(__file__), 'encoderPlan': identity(root/'encoder-plan.json'),
                    'stagePlan': identity(root/'stage-plan.json'), 'wrapperGate': identity(root/'int8-wrapper-engineering.json'),
                    'affinity': [8, 9], 'ortThreads': 2, 'batch': 1,
                    'resourceFloors': {'ramBytes': 4*1024**3, 'windowsCFreeBytes': 20*1024**3},
                    'rule': 'One worker, completed staged records only, same frozen graph/encoder function; no retuning.'}
    write_immutable(root/'int8-queue-plan.json', registration)
    p = encoders.precision_module()
    runtime = p.session(encoders.verified(plan['quantizedGraph']), threads=2)
    def infer(values):
        return np.concatenate([runtime.run(None, {'image': values[i:i+1]})[0] for i in range(len(values))])
    pending = [source for source in staged_plan['records'] if 'int8' not in source.get('reuse', {}).get('dino', {})]
    pending.sort(key=lambda source: (not source['teaching']['allowed'], source['id']))
    records = []
    paused = None
    while pending:
        state = resources()
        low = state['ramAvailableBytes'] < 4*1024**3 or state['windowsCFreeBytes'] < 20*1024**3
        if low:
            if paused != 'resources':
                print(json.dumps({'int8QueuePaused': 'resource floor', **state}), flush=True)
            paused = 'resources'
            time.sleep(30)
            continue
        ready = next((source for source in pending if (root/'staged'/source['id']/'receipt.json').exists()), None)
        if ready is None:
            if paused != 'staging':
                print(json.dumps({'int8QueueWaiting': 'remaining records not staged', 'remaining': len(pending)}), flush=True)
            paused = 'staging'
            time.sleep(30)
            continue
        paused = None
        receipt_path = root/'staged'/ready['id']/'receipt.json'
        staged = read(receipt_path)
        encoders.require(staged['lineage'] == {'plan': identity(root/'stage-plan.json'), 'source': ready}, 'Ready stage binding differs')
        print(json.dumps({'int8QueueStart': ready['id'], 'remaining': len(pending), **state}), flush=True)
        encoders.dino_record(ready, staged, 'int8', infer, plan, root)
        receipt = root/'encoders/int8'/ready['id']/'receipt.json'
        records.append({'id': ready['id'], 'receipt': identity(receipt), 'resourcesBefore': state})
        pending.remove(ready)
        print(json.dumps({'int8QueueComplete': ready['id'], 'remaining': len(pending), **resources()}), flush=True)
    write_immutable(root/'int8-queue-complete.json', {'plan': identity(root/'int8-queue-plan.json'), 'records': records})


if __name__ == '__main__':
    main()
