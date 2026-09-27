#!/usr/bin/env python3
"""Synthetic CPU proof that changing validation cannot change fixed-epoch training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.neural_context_development import canonical_hash, identity, read, write_immutable
from analysis.neural_development import Example
from analysis.neural_expanded_development import Supervised
from analysis.neural_generalization_experiment import EPOCHS, MODELS
from analysis.neural_generalization_reuse import Bindings, require
from analysis.neural_recognition_fit import fit_model
from analysis.schema import Interval


def synthetic_row(config, key, seed, tier='exact', ticks=24):
    times = np.arange(ticks, dtype=np.float64)/4
    values = np.random.default_rng(seed).normal(size=(ticks, config.input_dimension)).astype(np.float32)
    targets = np.zeros((ticks, 4), np.float32)
    targets[4:10, 0] = 1; targets[13:22, 0] = 1
    targets[[4, 13], 1] = 1; targets[[9, 21], 2] = 1
    targets[2:23, 3] = 1
    mask = np.ones_like(targets)
    truth = (Interval(1., 2.5), Interval(3.25, 5.5))
    if tier == 'draft': mask[:, 1:] = 0
    if tier == 'coverage':
        mask[:, :3] = 0; targets[:, :3] = 0; truth = ()
    example = Example(key, 'source-'+key, ticks/4, times, values, targets,
                      np.ones(ticks, bool), truth, (), 'indoor')
    return Supervised(example, mask, tier)


def qualify(destination, affinity=None):
    import torch
    if affinity is not None: os.sched_setaffinity(0, set(affinity))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(not destination.exists(), 'Qualification destination already exists')
    destination.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('reuse_qualification_registration', REPO/'scripts/register-neural-generalization.py')
    registrar = importlib.util.module_from_spec(spec); spec.loader.exec_module(registrar)
    evidence = Bindings()
    for reference in registrar.code_dependencies().values(): evidence.bind(reference)
    evidence.capture(__file__)
    configurations = {}
    for name, config in MODELS.items(): configurations.setdefault(canonical_hash(asdict(config)), (name, config))
    checks = []; started = time.perf_counter()
    for name, config in configurations.values():
        exact = [synthetic_row(config, 'exact', 8)]
        auxiliary = {tier: [synthetic_row(config, tier, index+10, tier)] for index, tier in enumerate(('draft', 'coverage'))}
        common = synthetic_row(config, 'shared-validation', 20).example
        added = synthetic_row(config, 'additional-validation', 21).example
        # The second population also changes order. No real labels/features or
        # external outcomes are read by this engineering qualification.
        variants = ([common], [added, common])
        folders = [destination/name/'narrow', destination/name/'wide']
        for folder, validation in zip(folders, variants, strict=True):
            fit_model(exact, auxiliary, validation, config, 3407, EPOCHS, folder, 'cpu', 'synthetic-validation-independence-v1', 'short_boost')
        left, right = (read(folder/'completed.json') for folder in folders)
        training_keys = [key for key in left if key not in ('validationIds', 'validationGroups', 'artifacts', 'wallSeconds', 'peakAllocatedCudaBytes')]
        require({k: left[k] for k in training_keys} == {k: right[k] for k in training_keys}, 'Validation changed training metadata/history')
        for epoch in EPOCHS:
            with np.load(folders[0]/f'weights-{epoch}.npz', allow_pickle=False) as a, np.load(folders[1]/f'weights-{epoch}.npz', allow_pickle=False) as b:
                require(a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files), 'Validation changed checkpoint/scaler arrays')
            with np.load(folders[0]/f'predictions-{epoch}.npz', allow_pickle=False) as a, np.load(folders[1]/f'predictions-{epoch}.npz', allow_pickle=False) as b:
                require(a.files == [common.id] and b.files == [added.id, common.id]
                        and np.array_equal(a[common.id], b[common.id]), 'Validation changed shared probabilities')
            for folder in folders:
                for stem in ('weights', 'predictions'): evidence.capture(folder/f'{stem}-{epoch}.npz')
        for folder in folders: evidence.capture(folder/'completed.json')
        checks.append({'name': name, 'config': asdict(config), 'epochs': list(EPOCHS),
            'checkpointHistoryAndSharedScoresBitExact': True,
            'validationPopulations': [[e.id for e in records] for records in variants],
            'metadata': [identity(folder/'completed.json') for folder in folders]})
        print(json.dumps({'qualifiedArchitecture': name, 'passed': True}), flush=True)
    result = {'kind': 'generalization-validation-independence-qualification-v1', 'passed': True,
        'syntheticOnly': True, 'realDataRead': False, 'device': 'cpu', 'torchThreads': 2,
        'affinity': sorted(os.sched_getaffinity(0)), 'architectures': checks,
        'scope': 'All five distinct temporal architectures, exact/draft/coverage streams, all60epochs and four checkpoint banks; student fit has no validation parameter and is checked separately by provenance gates.',
        'wallSeconds': time.perf_counter()-started, 'evidence': evidence.finish(), 'source': identity(__file__)}
    write_immutable(destination/'qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-affinity', help='Comma-separated allowed CPUs')
    args = parser.parse_args()
    result = qualify(args.output, [int(x) for x in args.cpu_affinity.split(',')] if args.cpu_affinity else None)
    print(json.dumps({'passed': result['passed'], 'architectures': len(result['architectures']), 'wallSeconds': result['wallSeconds']}))
