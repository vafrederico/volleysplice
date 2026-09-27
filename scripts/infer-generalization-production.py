#!/usr/bin/env python3
"""Replay the fixed shipped ensemble/suppression on blind full-video features."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.config import FeatureConfig
from analysis.features import FeatureSequence, VideoMetadata, contextualize
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_inputs import feature_entries, manifest_rows, select_rows, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('frozen_production_replay', REPO/'scripts/prepare-neural-production-comparison.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    _, population = manifest_rows(args.manifest)
    rows = select_rows(population, 'infer')
    features = feature_entries(rows, args.features)
    assets = {}
    for name, (filename, expected_sha) in replay.ASSETS.items():
        reference = identity(REPO/'prod/public/runtime'/filename)
        if reference['sha256'] != expected_sha:
            raise ValueError('Production assets changed; register the new baseline before replay')
        assets[name] = reference
    runtimes = {k: read(v['path']) for k, v in assets.items()}
    config = FeatureConfig.from_dict(runtimes['previous']['featureConfig'])
    source_paths = ['scripts/infer-generalization-production.py', 'scripts/prepare-neural-production-comparison.py',
        'analysis/features.py', 'analysis/neural_generalization_inputs.py',
        'prod/src/lib/on-device/model.ts', 'prod/src/lib/on-device/ensemble.ts',
        'prod/src/lib/on-device/suppression-model.ts', 'prod/src/lib/on-device/suppression-policy.ts',
        'prod/src/lib/cut-draft.ts', 'prod/src/lib/score-tracking.ts']
    protocol = {'kind': 'blind-fixed-production-generalization-v1', 'manifest': identity(args.manifest),
                'features': identity(args.features), 'assets': assets,
                'sourceCode': {p: identity(REPO/p) for p in source_paths},
                'programSha256': hashlib.sha256(replay.NODE_REPLAY.encode()).hexdigest(),
                'modelIds': ['productionDefault', 'productionUnion'],
                'productionPolicy': 'aggressive whole-rally suppression', 'trainingPerformed': False,
                'labelsUsedForInference': False, 'recordingIds': [r['id'] for r in rows],
                'nodeVersion': subprocess.check_output([args.node, '--version'], text=True).strip()}
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output/'protocol.json', protocol)
    results = []
    for row in rows:
        receipt_path = args.output/(row['id']+'.json')
        expected = {'id': row['id'], 'sourceGroup': row['sourceGroup'],
                    'protocol': identity(args.output/'protocol.json'), 'featureCache': features[row['id']]['audiovisual'],
                    'labelsUsed': False}
        if receipt_path.exists():
            receipt = read(receipt_path)
            if any(receipt.get(k) != v for k, v in expected.items()):
                raise ValueError('Production replay resume lineage differs')
            verified(receipt['probabilities'])
            results.append(receipt)
            continue
        with np.load(verified(expected['featureCache']), allow_pickle=False) as cache:
            metadata = VideoMetadata(**json.loads(str(cache['metadata_json'].item())))
            sequence = FeatureSequence(cache['times'].astype(np.float64), cache['values'].astype(np.float32),
                                       tuple(str(v) for v in cache['names']), metadata)
        contextual, names = contextualize(sequence, config)
        if any(list(names) != runtime['featureNames'] for runtime in runtimes.values()):
            raise ValueError('Production feature schema mismatch')
        payload = {'repoUrl': replay.node_repo_url(REPO), 'id': row['id'], 'duration': metadata.duration,
                   'times': sequence.times.tolist(), 'contextual': contextual.ravel().tolist(), 'ignoredIntervals': []}
        result = subprocess.run([args.node, '--input-type=module', '-e', replay.NODE_REPLAY],
                                input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, check=True)
        decoded = json.loads(result.stdout)
        probability_path = args.output/(row['id']+'.npz')
        with probability_path.open('xb') as stream:
            np.savez_compressed(stream, times=sequence.times,
                **{k: np.asarray(v, np.float32) for k, v in decoded.pop('probabilities').items()})
        receipt = {**expected, 'durationSeconds': metadata.duration, 'probabilities': identity(probability_path),
                   'predictions': {'productionDefault': decoded['variants']['aggressive']['core'],
                                   'productionUnion': decoded['variants']['none']['core']},
                   'productReplay': decoded}
        write_immutable(receipt_path, receipt)
        results.append(receipt)
        print(json.dumps({'productionInferred': row['id']}), flush=True)
    for reference in protocol['sourceCode'].values():
        verified(reference)
    write_immutable(args.output/'report.json', {'protocol': identity(args.output/'protocol.json'), 'records': results})


if __name__ == '__main__':
    main()
