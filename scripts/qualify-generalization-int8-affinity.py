#!/usr/bin/env python3
"""Qualify and apply a scheduling-only CPU affinity change to the existing worker."""
from analysis.private_ledger import private_value
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import os
import resource
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0089'))


def affinities(pid):
    return {int(path.name): sorted(os.sched_getaffinity(int(path.name))) for path in (Path('/proc')/str(pid)/'task').iterdir()}


def set_all(pid, cpus):
    for tid in affinities(pid):
        os.sched_setaffinity(tid, set(cpus))
    result = affinities(pid)
    if not all(value == sorted(cpus) for value in result.values()):
        raise ValueError('Not every worker thread has the requested affinity')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-pid', type=int, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {8, 9})
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location('frozen_precision', REPO/'scripts/evaluate-dino-precision.py')
    p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    import numpy as np
    p.ensure_nas(ROOT)
    pid = args.worker_pid
    command = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if b'scripts/run-generalization-int8-queue.py' not in command:
        raise ValueError('Only this experiment INT8 worker may be paused/remapped')
    start_ticks = (Path('/proc')/str(pid)/'stat').read_text().split()[21]
    before = affinities(pid)
    encoder_plan = p.read(ROOT/'encoder-plan.json')
    gate = p.read(ROOT/'encoder-engineering.json')
    staged = p.read(ROOT/'staged'/gate['recordingId']/'receipt.json')
    folder = ROOT/'int8-affinity-amendment-v1'
    folder.mkdir(exist_ok=True)
    plan = {'source': p.ident(__file__), 'encoderPlan': p.ident(ROOT/'encoder-plan.json'),
            'queuePlan': p.ident(ROOT/'int8-queue-plan.json'), 'workerPid': pid, 'workerStartTicks': start_ticks,
            'workerThreadsBefore': before, 'frameIndexes': gate['dinoFrameIndexes'],
            'stagingReceipt': p.ident(ROOT/'staged'/gate['recordingId']/'receipt.json'),
            'controlAffinity': [8, 9], 'candidateAffinity': [6, 8], 'ortThreads': 2, 'batch': 1,
            'graph': encoder_plan['quantizedGraph'], 'parentAuthorization': 'Bounded scheduling pilot; pause only our existing worker; one active inference process; resume mapped allTIDs only if bit-exact.',
            'labelsUsed': False, 'modelSelectionPerformed': False, 'mobileBenchmark': False,
            'timingContext': 'Concurrent historical/head fitting and one video stager; scheduling pilot only.'}
    p.write(folder/'protocol.json', plan)
    for reference in encoder_plan['parents']:
        if p.sha(reference['path']) != reference['sha256']:
            raise ValueError('Frozen encoder dependency changed')
    rgb = []
    for tick in gate['dinoFrameIndexes']:
        ref = staged['dinoRgbChunks'][tick//128]
        if p.sha(ref['path']) != ref['sha256']:
            raise ValueError('Pilot pixels changed')
        rgb.append(np.load(ref['path'], mmap_mode='r')[tick % 128].copy())
    pixels = p.normalized(np.stack(rgb)); del rgb
    changed = False
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(40):
            if '\nState:\tT' in (Path('/proc')/str(pid)/'status').read_text():
                break
            time.sleep(.05)
        else:
            raise ValueError('Existing worker did not stop')
        runtime = p.session(plan['graph']['path'], threads=2)
        outputs, measured = {}, {}
        for name, cpus in (('control', [8, 9]), ('candidate', [6, 8])):
            set_all(os.getpid(), cpus)
            runtime.run(None, {'image': pixels[:1]})
            started = time.perf_counter()
            outputs[name] = np.concatenate([runtime.run(None, {'image': row[None]})[0] for row in pixels])
            seconds = time.perf_counter()-started
            measured[name] = {'affinity': cpus, 'frames': len(pixels), 'wallSeconds': seconds,
                              'framesPerSecond': len(pixels)/seconds}
        if not np.array_equal(outputs['control'], outputs['candidate']):
            raise ValueError('Affinity pilot changed raw FP32 INT8-graph outputs')
        pilot = {'plan': p.ident(folder/'protocol.json'), 'passed': True, 'all32OutputsBitExact': True,
                 'storedFloat16AlsoBitExact': bool(np.array_equal(outputs['control'].astype(np.float16), outputs['candidate'].astype(np.float16))),
                 'measurementsUnderConcurrency': measured, 'maximumResidentKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'mobileBenchmark': False}
        p.write(folder/'pilot.json', pilot)
        if (Path('/proc')/str(pid)/'stat').read_text().split()[21] != start_ticks:
            raise ValueError('Worker process identity changed')
        after = set_all(pid, [6, 8]); changed = True
        p.write(folder/'applied.json', {'plan': p.ident(folder/'protocol.json'), 'pilot': p.ident(folder/'pilot.json'),
                'existingWorkerReused': True, 'threadsAfter': after, 'graphAndBatchAndOrtThreadCountUnchanged': True})
        print(json.dumps(pilot), flush=True)
        del runtime, outputs, pixels
        gc.collect()
    finally:
        if not changed:
            for tid, cpus in before.items():
                if (Path('/proc')/str(pid)/'task'/str(tid)).exists():
                    os.sched_setaffinity(tid, set(cpus))
        os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
