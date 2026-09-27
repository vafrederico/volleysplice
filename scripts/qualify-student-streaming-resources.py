#!/usr/bin/env python3
"""Measure one unchanged, label-blind batch64 student extraction in an isolated folder."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0061'))
HELPER_PATH = REPO/'scripts/run-neural-student-reuse-profiled.py'
SPEC = importlib.util.spec_from_file_location('frozen_resource_helpers', HELPER_PATH)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def environment(folder):
    require(folder.resolve().is_relative_to(ROOT) and Path('/mnt/freenas').is_mount(), 'Expected mounted NAS folder')
    os.sched_setaffinity(0, {2, 3})
    os.nice(10)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        path = folder/'runtime'/key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PATH'] += os.pathsep+'/usr/lib/wsl/lib'


def register(folder):
    task_path = ROOT/'original-corpus-v1/tasks/original-corpus__distilled-mobile-tcn__seed-3407.json'
    task = read(task_path)
    fit = ROOT/'original-corpus-v1/fits/distilled-mobile-tcn/seed-3407'
    feature_index = read(task['features']['path'])
    candidates = []
    for row in feature_index['records']:
        if row['recordingId'] not in task['trainIds']:
            continue
        reference = row['features']['imageInput']
        image = read(reference['path'])
        candidates.append((int(image['frames']), row['recordingId'], reference, image))
    frames, key, image_ref, image = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
    expected_path = fit/'student-features'/(key+'.json')
    expected = read(expected_path)
    require(expected['imageInput'] == image['arrays'], 'Existing image/output lineage differs')
    plan = {'kind': 'student-batch64-streaming-resource-qualification-v1',
        'source': HELPERS.identity(__file__), 'resourceHelpers': HELPERS.identity(HELPER_PATH),
        'numericSources': [HELPERS.identity(REPO/name) for name in
            ('analysis/neural_mobile_distillation.py', 'analysis/neural_generalization_inputs.py', 'analysis/mobile_visual_features.py')],
        'task': HELPERS.identity(task_path), 'student': HELPERS.identity(fit/'student/completed.json'),
        'fitNumericalAudit': HELPERS.identity(fit/'fit-numerical-audit.json'),
        'manifest': task['manifest'], 'features': task['features'], 'imageInput': image_ref,
        'expectedReceipt': HELPERS.identity(expected_path), 'expectedOutput': expected['output'],
        'recordingId': key, 'frames': frames,
        'selectionRule': 'Longest images224 recording among this original completed student training population; ascending ID breaks ties. No rally labels or outcomes read for selection.',
        'outputDirectory': str(folder/'extracted'), 'batchSize': 64, 'cpuAffinity': [2, 3], 'threads': 2, 'nice': 10,
        'temporaryFourGpuWorkerSchedulingExceptionAuthorized': True,
        'qualificationOnly': True, 'ownerDirectoryWritesAllowed': False, 'scoresOrMetricsAllowed': False,
        'determinism': {'cublasWorkspace': ':4096:8', 'cudnnBenchmark': False, 'cudnnTf32': False,
                        'matmulTf32': False, 'deterministicAlgorithms': True, 'encoderEval': True},
        'startupFloors': {'linuxAvailableBytes': 10*1024**3, 'windowsPhysicalFreeBytes': 3*1024**3,
                          'gpuFreeMiB': 5*1024, 'windowsCFreeBytes': 20*1024**3},
        'limits': 'Own Torch tensor allocation/reservation peaks exclude driver/context overhead; RSS includes imports and mapped image pages. Overlapped wall time is not a benchmark.'}
    HELPERS.write_new(folder/'plan.json', plan)
    print(json.dumps({'registered': HELPERS.identity(folder/'plan.json'), 'recordingId': key, 'frames': frames}), flush=True)


def run(folder, digest):
    require(HELPERS.identity(folder/'plan.json')['sha256'] == digest, 'Qualification plan changed')
    plan = read(folder/'plan.json')
    require(plan['source'] == HELPERS.identity(__file__), 'Qualification source changed')
    for reference in [plan['resourceHelpers'], *plan['numericSources'], plan['task'], plan['student'],
                      plan['fitNumericalAudit'], plan['manifest'], plan['features'], plan['imageInput'],
                      plan['expectedReceipt'], plan['expectedOutput']]:
        require(HELPERS.identity(reference['path']) == reference, 'Qualification input changed')
    require(not (folder/'report.json').exists() and not Path(plan['outputDirectory']).exists(), 'Qualification already started')
    while True:
        state = HELPERS.resources()
        if all(state[key] >= value for key, value in plan['startupFloors'].items()):
            break
        print(json.dumps({'studentResourceQualificationWaiting': state}), flush=True)
        time.sleep(30)
    HELPERS.write_new(folder/'startup.json', {'plan': HELPERS.identity(folder/'plan.json'), 'resources': state,
        'pid': os.getpid(), 'temporaryFourGpuWorkerException': True})
    import numpy as np
    import torch
    from analysis import neural_mobile_distillation as student
    from analysis.neural_generalization_inputs import load_images_teachers
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    meta = read(plan['student']['path'])
    fit_audit = read(plan['fitNumericalAudit']['path'])
    require(fit_audit['passed'] is True and fit_audit['task'] == plan['task']
            and fit_audit['student']['weights'] == meta['weights'], 'Original student fit gate differs')
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            samples.append({'elapsedSeconds': time.perf_counter()-started,
                            **HELPERS.memory_status(Path('/proc/self/status').read_text())})
            stop.wait(.1)
    started = time.perf_counter()
    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        images, teachers = load_images_teachers(plan['manifest']['path'], plan['features']['path'],
            recording_ids=[plan['recordingId']], for_training=False)
        require(not teachers, 'Teacher targets reached inference qualification')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        encoder = student.load_encoder(meta, 'cuda')
        receipt = student.extract_student_record(encoder, meta, images[plan['recordingId']],
                                                Path(plan['outputDirectory']), 'cuda')
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        del encoder
        torch.cuda.empty_cache()
        checks = []
        with np.load(receipt['output']['path'], allow_pickle=False) as actual, np.load(plan['expectedOutput']['path'], allow_pickle=False) as expected:
            require(set(actual.files) == set(expected.files), 'Output array schema changed')
            for key in actual.files:
                require(actual[key].dtype == expected[key].dtype and np.array_equal(actual[key], expected[key]),
                        'Unchanged extraction array differs: '+key)
                checks.append({'array': key, 'shape': list(actual[key].shape), 'dtype': str(actual[key].dtype), 'bitExact': True})
        require(HELPERS.identity(plan['expectedReceipt']['path']) == plan['expectedReceipt']
                and HELPERS.identity(plan['expectedOutput']['path']) == plan['expectedOutput'], 'Owner cache changed during qualification')
    finally:
        stop.set()
        watcher.join()
        HELPERS.write_new(folder/'rss-samples.json', samples)
    report = {'kind': 'student-batch64-streaming-resource-qualification-result-v1', 'passed': True,
        'plan': HELPERS.identity(folder/'plan.json'), 'startup': HELPERS.identity(folder/'startup.json'),
        'outputReceipt': HELPERS.identity(Path(plan['outputDirectory'])/(plan['recordingId']+'.json')),
        'arrayChecks': checks, 'rssSamples': HELPERS.identity(folder/'rss-samples.json'),
        'sampledMaximumRssBytes': max(v.get('VmRSSBytes', 0) for v in samples),
        'processVmHwmBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'sampledMaximumSwapBytes': max(v.get('VmSwapBytes', 0) for v in samples),
        'torchPeakAllocatedBytes': allocated, 'torchPeakReservedBytes': reserved,
        'wallSeconds': time.perf_counter()-started, 'overlappedTimingNotBenchmark': True,
        'ownerCacheUnchanged': True, 'teacherTargetsLoaded': False, 'scoresOrMetricsProduced': False}
    HELPERS.write_new(folder/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    environment(args.folder)
    if args.action == 'register':
        register(args.folder)
    else:
        require(args.plan_sha256 is not None, 'Run needs frozen plan hash')
        run(args.folder, args.plan_sha256)
