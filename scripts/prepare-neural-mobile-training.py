#!/usr/bin/env python3
"""Bind a completed mobile cache index to its runtime helper and validate it.

Creates immutable NAS artifacts, loads the existing eighteen-record supervision
without changing it, and checks one untrained CPU forward batch. No fitting,
feature extraction, GPU use, or protected-test access occurs here.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path(private_value('private-reference-0095'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'immutable artifact differs: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(content)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='completed original full extraction index')
    parser.add_argument('--output', type=Path, required=True, help='new immutable training-index.json')
    parser.add_argument('--source-archive', type=Path, required=True, help='immutable mobile_visual_features.py copy')
    parser.add_argument('--source-manifest', type=Path, default=SOURCE_MANIFEST)
    parser.add_argument('--validation-output', type=Path, required=True)
    parser.add_argument('--temp-dir', type=Path, required=True, help='NAS scratch/cache directory')
    args = parser.parse_args()
    for key in ('index', 'output', 'source_archive', 'source_manifest', 'validation_output', 'temp_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    helper = REPO / 'analysis/mobile_visual_features.py'
    if len({args.index, args.output, args.source_archive, args.validation_output, helper.resolve()}) != 5:
        parser.error('input, output, archive, report and runtime helper must be separate files')
    if args.validation_output.exists():
        parser.error('validation output already exists; choose a new report path to revalidate')
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    for name in ('TMPDIR', 'TEMP', 'TMP', 'XDG_CACHE_HOME', 'TORCH_HOME'):
        os.environ[name] = str(args.temp_dir)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from analysis.neural_short_boost_transfer import load_data
    from analysis.neural_recognition_inputs import attach_features
    from analysis.neural_recognition_fit import fit_scaler, make_chunks
    from analysis.recognition_temporal_model import RecognitionConfig, model_for

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    index_digest = sha(args.index)
    index = json.loads(args.index.read_text())
    if (index.get('kind') != 'volleycut-mobile-visual-extraction-index' or len(index.get('records', [])) != 18
            or index.get('labelsUsed') is not False):
        raise ValueError('expected a completed full eighteen-record mobile extraction index')
    if (index.get('trainingEligible', True) is not True
            or any(row.get('trainingEligible', True) is not True for row in index['records'])):
        raise ValueError('upstream extraction index explicitly disallows training')
    helper_bytes = helper.read_bytes()
    write_immutable(args.source_archive, helper_bytes)
    index['upstreamExtractionIndex'] = {'path': str(args.index), 'sha256': index_digest}
    index['runtimeDependencies'] = [{'path': str(helper.resolve()), 'sha256': sha(helper),
                                     'archivePath': str(args.source_archive)}]
    index['trainingEligible'] = True
    for row in index['records']:
        row['trainingEligible'] = True
    write_immutable(args.output, json_bytes(index))
    data = load_data(args.source_manifest, None, with_dino=False)
    config = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8)
    attached = attach_features(data, args.output, config, args.source_manifest)
    records = []
    for tier in ('exact', 'draft', 'coverage'):
        for original, new in zip(data[tier], attached[tier], strict=True):
            a, b = original.example, new.example
            if (a.values.shape[1] != 104 or b.values.shape != (len(a.times), 2416)
                    or not np.array_equal(a.values, b.values[:, :104]) or not np.isfinite(b.values).all()
                    or original is new or a is b or original.mask is not new.mask
                    or any(getattr(a, field) is not getattr(b, field)
                           for field in ('times', 'targets', 'valid', 'truth', 'ignored'))
                    or (a.id, a.group, a.environment, original.tier) != (b.id, b.group, b.environment, new.tier)):
                raise ValueError(f'feature attachment changed existing inputs/supervision: {a.id}')
            records.append({'id': b.id, 'tier': tier, 'group': b.group, 'shape': list(b.values.shape),
                            'finite': True, 'maskAndSupervisionUnchanged': True,
                            'originalAvUnchanged': True, 'bytes': b.values.nbytes})
    groups = sorted({row.example.group for row in attached['exact']})
    training = [row for row in attached['exact'] if row.example.group != groups[0]]
    mean, scale = fit_scaler(training, config)
    # One recording's chunk views avoid copying the entire corpus into pools.
    first = make_chunks(training[:1], mean, scale, config, 'short_boost')[0]
    x, y, mask = [np.stack([first[i]]) for i in range(3)]
    if (x.shape[2] != 2416 or y.shape != mask.shape or y.shape != (1, x.shape[1], 4)
            or not all(np.isfinite(value).all() for value in (x, y, mask, first[4], mean, scale))):
        raise ValueError('nonfinite or incorrectly shaped first batch/scaler')
    torch.manual_seed(20260922)
    model = model_for(config).cpu().eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(x)).numpy()
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError('untrained CPU forward failed')
    if (sha(helper) != hashlib.sha256(helper_bytes).hexdigest() or sha(args.source_archive) != sha(helper)
            or sha(args.index) != index_digest):
        raise ValueError('source changed during validation')
    report = {'schemaVersion': 1, 'status': 'pass', 'createdAt': datetime.now(timezone.utc).isoformat(),
        'gpuUsed': False, 'trainingPerformed': False, 'protectedTestOpened': False,
        'scope': 'Real eighteen-record feature attachment; unchanged supervision; one untrained CPU forward batch',
        'sourceManifest': {'path': str(args.source_manifest), 'sha256': sha(args.source_manifest)},
        'trainingIndex': {'path': str(args.output), 'sha256': sha(args.output)},
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'runtimeDependencies': index['runtimeDependencies'],
        'bridgeSourceSha256': sha(REPO / 'analysis/neural_recognition_inputs.py'),
        'counts': {tier: len(attached[tier]) for tier in ('exact', 'draft', 'coverage')}, 'records': records,
        'totalAttachedArrayBytes': sum(row['bytes'] for row in records),
        'firstBatch': {'sourceId': training[0].example.id, 'inputShape': list(x.shape),
                       'targetAndLogitShape': list(y.shape), 'scalerShape': list(mean.shape), 'finite': True,
                       'trueLengthNoSyntheticPadding': True, 'cpuForwardOnly': True,
                       'modelState': 'seeded untrained', 'cpuThreads': 1}}
    write_immutable(args.validation_output, json_bytes(report))
    print(json.dumps({key: report[key] for key in
          ('status', 'trainingIndex', 'counts', 'totalAttachedArrayBytes', 'firstBatch')}), flush=True)


if __name__ == '__main__':
    main()
