"""Registered nested source-held recognition comparisons; one bounded worker.

Six unordered excluded-group fits produce twelve isolated validation views.
Predictions on a held outer group are never consulted by that group's selection.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np

from . import neural_expanded_development as expanded
from . import neural_short_boost_transfer as source
from .neural_context_development import canonical_hash, identity, read, write_immutable
from .neural_evaluation import evaluate_predictions
from .neural_recognition_fit import fit_model
from .recognition_temporal_model import RecognitionConfig, model_metadata

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0057'))
REFERENCE = ROOT/'2026-09-19-short-boost-transfer/study'
MANIFEST = ROOT/'2026-09-19-short-boost-transfer/manifest-pts-v1.json'
PROTOCOL = REPO/'docs/research/neural-recognition-execution-2026-09-22.md'
ENGINEERING = ROOT/'2026-09-22-recognition/engineering/report.json'
SEEDS = (3407, 1729, 20260918)
EPOCHS = (5, 15, 30, 60)


def require(value, message):
    if not value:
        raise ValueError(message)


def dataset_contract(manifest):
    groups = sorted({r['sourceGroup'] for r in manifest['exactRows']})
    require(len(groups) == 4 and len(manifest['exactRows']) == 8, 'Exact population changed')
    rows = [r for key in ('exactRows', 'draftRows', 'coverageRows') for r in manifest[key]]
    require(len(rows) == 18 and len({r['id'] for r in rows}) == 18, 'Training corpus changed')
    require(not any(r['environment'] == 'beach' or r['split'] == 'test'
                    or r['sourceGroup'] in expanded.base.PROTECTED_GROUPS for r in rows), 'Protected/beach source present')
    return groups, {key: [r['id'] for r in manifest[key]] for key in ('exactRows', 'draftRows', 'coverageRows')}


def register(output, config, manifest_path, dino_path, feature_path=None):
    old = read(REFERENCE/'preregistration.json')
    require(canonical_hash(old['contract']) == old['sha256'], 'Reference contract changed')
    code = dict(old['contract']['code'])
    for name in ('local_attention_temporal_model.py', 'recognition_temporal_model.py',
                 'neural_recognition_fit.py', 'neural_recognition_development.py',
                 'neural_context_fit.py', 'neural_context_development.py'):
        code[name] = expanded.base.file_sha256(REPO/'analysis'/name)
    if feature_path:
        name = 'neural_recognition_inputs.py'
        code[name] = expanded.base.file_sha256(REPO/'analysis'/name)
    require(all(expanded.base.file_sha256(REPO/'analysis'/name) == sha for name, sha in code.items()),
            'Legacy dependency revision changed')
    engineering = read(ENGINEERING)
    require(engineering.get('passed') is True and engineering.get('controlBitExact') is True
            and engineering.get('attentionExposureIdentical') is True
            and engineering['manifest'] == identity(manifest_path)
            and all(code.get(name) == sha for name, sha in engineering['code'].items()),
            'Real-data engineering audit absent or stale')
    manifest = read(manifest_path)
    groups, population = dataset_contract(manifest)
    contract = {'experiment': 'mobile-recognition-local-attention-v1', 'manifest': identity(manifest_path),
                'dinoManifest': identity(dino_path) if config.family == 'dino' else None,
                'featureManifest': identity(feature_path) if feature_path else None,
                'referenceStudy': identity(REFERENCE/'preregistration.json'),
                'referenceReport': identity(REFERENCE/'report.json'),
                'engineeringAudit': identity(ENGINEERING),
                'config': asdict(config), 'model': model_metadata(config), 'code': code,
                'protocol': identity(PROTOCOL), 'groups': groups, 'population': population,
                'seeds': list(SEEDS), 'checkpointEpochs': list(EPOCHS),
                'cohort': 'reviewed_export', 'lossArm': 'short_boost',
                'decoderCandidates': expanded.decoder_candidates(),
                'selection': 'max F1_padP_coreR among inner R_core>=.95, otherwise max F1_padP_coreR',
                'targetPaddingSeconds': 2, 'paddingSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3,
                'ignoredPolicy': 'subtract after padding/joining, never rejoin',
                'trainingCoreTicks': 128, 'trainingHaloTicks': 62,
                'protectedTestOpened': False, 'productionPromotionAllowed': False}
    path = output/'preregistration.json'
    value = {'sha256': canonical_hash(contract), 'contract': contract}
    write_immutable(path, value)
    sources = output/'registered-sources'
    sources.mkdir(exist_ok=True)
    for name, sha in code.items():
        destination = sources/name
        data = (REPO/'analysis'/name).read_bytes()
        if destination.exists():
            require(expanded.base.file_sha256(destination) == sha, 'Source archive differs')
        else:
            with destination.open('xb') as stream:
                stream.write(data)
    protocol_copy = output/'protocol.md'
    if not protocol_copy.exists():
        protocol_copy.write_bytes(PROTOCOL.read_bytes())
    require(expanded.base.file_sha256(protocol_copy) == contract['protocol']['sha256'], 'Protocol snapshot differs')
    return value


def serializable_rows(rows):
    return [{**row, **{key: [item.to_dict() for item in row[key]]
                      for key in ('rallies', 'ignoredIntervals', 'predictions')}} for row in rows]


def run_cell(data, registration, output, seed, device, *, fit_fn=fit_model):
    contract, digest = registration['contract'], registration['sha256']
    config = RecognitionConfig(**contract['config'])
    result_path = output/f'result-{seed}.json'
    if result_path.exists():
        result = read(result_path)
        require(result['contractSha256'] == digest and result['seed'] == seed, 'Result identity differs')
        return result
    examples = {row.example.id: row.example for row in data['exact']}
    require(list(examples) == contract['population']['exactRows'], 'Loaded exact order differs')
    groups = contract['groups']
    owners = {}
    for left, right in combinations(range(len(groups)), 2):
        excluded = {groups[left], groups[right]}
        train = [r for r in data['exact'] if r.example.group not in excluded]
        auxiliary = expanded.auxiliary_for_fold(data, contract['cohort'], excluded)
        validation = [e for e in examples.values() if e.group in excluded]
        folder = output/'fits'/str(seed)/f'inner-{left}-{right}'
        write_immutable(folder.parent/f'plan-inner-{left}-{right}.json', {
            'excludedGroups': sorted(excluded), 'trainIds': [r.example.id for r in train],
            'auxiliaryIds': {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
            'validationIds': [e.id for e in validation], 'contractSha256': digest})
        print(f'FIT {config.family}/{config.head} seed={seed} exclude={left},{right}', flush=True)
        owners[(left, right)] = fit_fn(train, auxiliary, validation, config, seed,
            tuple(contract['checkpointEpochs']), folder, device, digest, contract['lossArm'])
    rows, selections = [], []
    for outer_index, outer in enumerate(groups):
        fitting = [r for r in data['exact'] if r.example.group != outer]
        held = [e for e in examples.values() if e.group == outer]
        probabilities = {epoch: {} for epoch in contract['checkpointEpochs']}
        for inner_index, inner in enumerate(groups):
            if inner == outer:
                continue
            owner = owners[tuple(sorted((outer_index, inner_index)))]
            allowed = {e.id for e in examples.values() if e.group == inner}
            for epoch in probabilities:
                require(not set(probabilities[epoch]) & allowed, 'Duplicate logical inner member')
                probabilities[epoch].update({key: value for key, value in owner[epoch].items() if key in allowed})
        expected = {r.example.id for r in fitting}
        require(all(set(p) == expected for p in probabilities.values()), 'Inner selection exposure differs')
        selected = expanded.choose_settings([r.example for r in fitting], probabilities)
        require(selected['decoder'] in contract['decoderCandidates'], 'Unregistered decoder')
        epoch = selected['epoch']
        auxiliary = expanded.auxiliary_for_fold(data, contract['cohort'], {outer})
        folder = output/'fits'/str(seed)/f'outer-{outer_index}'
        print(f'REFIT {config.family}/{config.head} seed={seed} outer={outer_index} epoch={epoch}', flush=True)
        predictions = fit_fn(fitting, auxiliary, held, config, seed, (epoch,), folder,
                             device, digest, contract['lossArm'])[epoch]
        rows.extend(e.row(expanded.base.decode(e, predictions[e.id], selected['decoder'])) for e in held)
        selections.append({'heldSourceGroup': outer, **selected})
    result = {'contractSha256': digest, 'family': config.family, 'head': config.head, 'seed': seed,
              'model': contract['model'], 'selections': selections, 'evaluation': evaluate_predictions(rows),
              'predictions': serializable_rows(rows), 'protectedTestOpened': False,
              'productionPromotionAllowed': False}
    write_immutable(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('av', 'dino', 'player', 'mobile'), required=True)
    parser.add_argument('--head', choices=('tcn', 'transformer'), required=True)
    parser.add_argument('--scalar-dimension', type=int, default=0)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--dino-manifest', type=Path)
    parser.add_argument('--feature-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    require(not args.device.startswith('cuda') or os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before deterministic GPU execution')
    config = RecognitionConfig(family=args.family, head=args.head, scalar_dimension=args.scalar_dimension)
    dino_path = args.dino_manifest or Path(read(REFERENCE/'preregistration.json')['contract']['dinoManifest']['path'])
    args.output.mkdir(parents=True, exist_ok=True)
    registration = register(args.output, config, args.manifest, dino_path, args.feature_manifest)
    if args.register_only:
        print(json.dumps({'contractSha256': registration['sha256'], 'model': registration['contract']['model']}))
        return
    started = time.perf_counter()
    data = source.load_data(args.manifest, dino_path, with_dino=config.family == 'dino')
    if config.family in ('player', 'mobile'):
        from .neural_recognition_inputs import attach_features
        data = attach_features(data, args.feature_manifest, config, args.manifest)
    write_immutable(args.output/'environment.json', {'torch': torch.__version__,
        'cuda': torch.version.cuda, 'device': args.device,
        'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'threads': torch.get_num_threads(), 'contractSha256': registration['sha256']})
    for seed in ((args.seed,) if args.seed else SEEDS):
        run_cell(data, registration, args.output, seed, args.device)
    paths = [args.output/f'result-{seed}.json' for seed in SEEDS]
    if all(path.exists() for path in paths):
        report = {'status': 'completed-recognition-development', 'contractSha256': registration['sha256'],
                  'results': [read(path) for path in paths], 'protectedTestOpened': False,
                  'productionPromotionAllowed': False}
        write_immutable(args.output/'report.json', report)
    print(json.dumps({'finishedAt': datetime.now(timezone.utc).isoformat(), 'wallSeconds': time.perf_counter()-started,
                      'completedSeeds': [seed for seed, path in zip(SEEDS, paths) if path.exists()]}), flush=True)


if __name__ == '__main__':
    main()
