#!/usr/bin/env python3
"""Real-data recipe replay and attention engineering smoke; no quality selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from analysis import neural_short_boost_transfer as source
from analysis import neural_expanded_development as expanded
from analysis import neural_recognition_fit as candidate
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_recognition_development import MANIFEST, REFERENCE
from analysis.recognition_temporal_model import RecognitionConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    dino_path = Path(read(REFERENCE/'preregistration.json')['contract']['dinoManifest']['path'])
    data = source.load_data(MANIFEST, dino_path, with_dino=False)
    groups = sorted({r.example.group for r in data['exact']})
    excluded = set(groups[:2])
    train = [r for r in data['exact'] if r.example.group not in excluded]
    auxiliary = expanded.auxiliary_for_fold(data, 'reviewed_export', excluded)
    held = [r.example for r in data['exact'] if r.example.group in excluded]
    config = RecognitionConfig(family='av', head='tcn')
    old_path, new_path = args.output/'legacy-control', args.output/'adapter-control'
    old = source.fit_model(train, auxiliary, held, 'tcn', 3407, (1,), old_path, 'cuda', 'engineering-only', 'short_boost')
    new = candidate.fit_model(train, auxiliary, held, config, 3407, (1,), new_path, 'cuda', 'engineering-only', 'short_boost')
    if any(not np.array_equal(old[1][e.id], new[1][e.id]) for e in held):
        raise ValueError('Current TCN predictions did not replay exactly')
    with np.load(old_path/'weights-1.npz') as a, np.load(new_path/'weights-1.npz') as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
            raise ValueError('Current TCN checkpoint did not replay exactly')
    old_meta, new_meta = read(old_path/'completed.json'), read(new_path/'completed.json')
    for key in ('history', 'optimizerSteps', 'exposureSha256', 'positiveWeight'):
        if old_meta[key] != new_meta[key]:
            raise ValueError(f'Current TCN training recipe differs: {key}')
    attention_path = args.output/'attention-smoke'
    attention = candidate.fit_model(train, auxiliary, held, RecognitionConfig(), 3407, (1,),
        attention_path, 'cuda', 'engineering-only', 'short_boost')
    attention_meta = read(attention_path/'completed.json')
    if attention_meta['exposureSha256'] != old_meta['exposureSha256']:
        raise ValueError('Attention sample exposure changed')
    if any(not np.isfinite(attention[1][e.id]).all() or np.any(attention[1][e.id][~e.valid] != 0) for e in held):
        raise ValueError('Invalid attention probabilities')
    code = {name: expanded.base.file_sha256(REPO/'analysis'/name) for name in (
        'recognition_temporal_model.py', 'neural_recognition_fit.py', 'local_attention_temporal_model.py')}
    report = {'kind': 'recognition-real-data-engineering-v1', 'passed': True,
              'manifest': identity(MANIFEST), 'code': code, 'controlBitExact': True,
              'attentionExposureIdentical': True, 'heldGroups': sorted(excluded),
              'control': identity(new_path/'completed.json'), 'legacy': identity(old_path/'completed.json'),
              'attention': identity(attention_path/'completed.json'), 'protectedTestOpened': False,
              'qualitySelectionPerformed': False}
    write_immutable(args.output/'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
