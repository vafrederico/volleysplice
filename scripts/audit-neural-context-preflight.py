#!/usr/bin/env python3
"""Qualify optional context fitting and unordered-fold reuse before registration.

Requires the completed, audited preceding study. Predetermined engineering
replays compare tensors and predictions, never candidate quality metrics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
import numpy as np
from analysis import neural_short_boost_transfer as frozen
from analysis import neural_context_fit as context_fit

ROOT = Path(private_value('private-reference-0084'))
NEW_SOURCES = ('short_context_temporal_model.py', 'neural_context_fit.py', 'neural_context_development.py')
AUDIT_KINDS = ('neural-short-boost-independent-tensor-audit-v1',
               'independent-short-boost-transfer-interval-audit-v1')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    return {'path': str(path), 'sha256': frozen.base.file_sha256(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compare_npz(left, right, keys=None):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        selected = set(a.files) if keys is None else set(keys)
        require(selected <= set(a.files) and selected <= set(b.files), 'Parity tensor inventory missing')
        if keys is None:
            require(set(a.files) == set(b.files), 'Parity tensor inventories differ')
        for name in selected:
            require(a[name].dtype == b[name].dtype and a[name].shape == b[name].shape
                    and a[name].tobytes(order='C') == b[name].tobytes(order='C'), 'Parity values differ: '+name)
    return {'left': identity(left), 'right': identity(right), 'arraysCompared': len(selected), 'bitExact': True}


def verify_reference_audit(path, report_path, expected_kind, contract_sha):
    value = read(path)
    report_sha = frozen.base.file_sha256(report_path)
    resolved_report = report_path.resolve()
    # The independent auditors bind the completed report by path and hash in
    # their evidence envelope. Do not accept a passed unrelated audit.
    def contains_binding(node):
        if isinstance(node, dict):
            if (node.get('sha256') == report_sha
                    and Path(node.get('path', '')).resolve() == resolved_report):
                return True
            return any(contains_binding(child) for child in node.values())
        return isinstance(node, list) and any(contains_binding(child) for child in node)
    require(expected_kind in AUDIT_KINDS and value.get('kind') == expected_kind
            and value.get('contractSha256') == contract_sha
            and value.get('passed') is True and contains_binding(value), 'Unpassed or unrelated reference audit')
    return identity(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'study')
    parser.add_argument('--cohort', choices=frozen.COHORTS, required=True)
    parser.add_argument('--arm', choices=frozen.ARMS, required=True)
    parser.add_argument('--tensor-audit', type=Path, required=True)
    parser.add_argument('--interval-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    reg, report, summary = [read(args.reference/name) for name in ('preregistration.json', 'report.json', 'summary.json')]
    old = reg['contract']
    require(frozen.canonical_hash(old) == reg['sha256']
            and report['contractSha256'] == reg['sha256']
            and report['status'] == 'completed-short-boost-transfer-development'
            and len(report['results']) == 54
            and summary['status'] == 'completed-short-boost-transfer-audit'
            and summary['contractSha256'] == reg['sha256']
            and report['protectedTestOpened'] is False and report['productionPromotionAllowed'] is False,
            'Reference study not complete and audited')
    manifest = Path(old['dinoManifest']['path']).parent/'manifest-pts-v1.json'
    dino_manifest = Path(old['dinoManifest']['path'])
    require(frozen.base.file_sha256(manifest) == old['manifestSha256']
            and frozen.base.file_sha256(dino_manifest) == old['dinoManifest']['sha256'], 'Reference data changed')
    code = {**old['code'], **{name: frozen.base.file_sha256(REPO/'analysis'/name) for name in NEW_SOURCES}}
    require(all(frozen.base.file_sha256(REPO/'analysis'/name) == sha for name, sha in old['code'].items()),
            'Frozen reference sources changed')
    require(args.tensor_audit.resolve() != args.interval_audit.resolve(), 'Distinct independent audits are required')
    audits = [verify_reference_audit(path, args.reference/'report.json', kind, reg['sha256'])
              for path, kind in zip((args.tensor_audit, args.interval_audit), AUDIT_KINDS)]
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                   'device': 'cuda', 'gpu': torch.cuda.get_device_name()}
    require(environment == old['environment'], 'Engineering replay environment differs')
    seed, excluded = 3407, set(old['groups'][:2])
    plan = {'kind': 'context-engineering-preflight-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'referenceContractSha256': reg['sha256'], 'referenceReport': identity(args.reference/'report.json'),
            'referenceAudits': audits, 'code': code, 'script': identity(Path(__file__)),
            'manifest': identity(manifest), 'dinoManifest': identity(dino_manifest), 'environment': environment,
            'cohort': args.cohort, 'lossArm': args.arm, 'seed': seed, 'excludedGroups': sorted(excluded),
            'originalProfileEpoch': 5, 'shortProfileEpoch': 1, 'qualityMetricsInspected': False,
            'procedure': 'Original epoch5 union-owner weights and both held prediction sets versus preceding '
                         'swapped inner fits. Short epoch1 union versus one-view versus repeated union, '
                         'weights/shared predictions/history; then immutable resume.'}
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output/'plan.json', plan)
    plan_hash = frozen.canonical_hash(plan)
    checked = []
    for kind in old['kinds']:
        print('QUALIFY', kind, flush=True)
        data = frozen.load_data(manifest, dino_manifest, with_dino=kind == 'dino_tcn')
        train = [row for row in data['exact'] if row.example.group not in excluded]
        auxiliary = frozen.expanded.auxiliary_for_fold(data, args.cohort, excluded)
        views = [[row.example for row in data['exact'] if row.example.group == group] for group in old['groups'][:2]]
        union = [row.example for row in data['exact'] if row.example.group in excluded]
        source = next(row for row in report['results'] if
                      (row['cohort'], row['kind'], row['lossArm'], row['seed']) == (args.cohort, kind, args.arm, seed))
        origin = Path(source['origin']['fitRoot'])
        original = args.output/kind/'original-union'
        context_fit.fit_model(train, auxiliary, union, kind, seed, (5,), original, 'cuda', plan_hash,
                              args.arm, context='original')
        meta = read(original/'completed.json')
        original_checks = []
        reference_contract = (source['origin']['referenceContractSha256']
                              if source['origin']['type'] == 'reused-reference' else reg['sha256'])
        for owner_index, validation in ((0, views[1]), (1, views[0])):
            reference = origin/f'outer-{owner_index}'/'inner-0'
            old_meta = read(reference/'completed.json')
            require(old_meta['trainIds'] == meta['trainIds'] and old_meta['auxiliaryIds'] == meta['auxiliaryIds']
                    and old_meta['validationIds'] == [row.id for row in validation], 'Reference fold membership differs')
            require(old_meta['contractSha256'] == reference_contract and old_meta['kind'] == kind
                    and old_meta['seed'] == seed and old_meta.get('lossArm', 'baseline') == args.arm
                    and old_meta['trainGroups'] == meta['trainGroups']
                    and old_meta['auxiliaryGroups'] == meta['auxiliaryGroups']
                    and old_meta['history'][:5] == meta['history'], 'Reference metadata or training history differs')
            for stem in ('weights', 'predictions'):
                name = f'{stem}-5.npz'
                require(frozen.base.file_sha256(reference/name) == old_meta['artifacts'][name], 'Reference parity artifact changed')
            original_checks.append({'referenceCompletion': identity(reference/'completed.json'),
                                    'referenceContractSha256': reference_contract,
                                    'weights': compare_npz(original/'weights-5.npz', reference/'weights-5.npz'),
                                    'predictions': compare_npz(original/'predictions-5.npz', reference/'predictions-5.npz',
                                                               [row.id for row in validation])})
        paths = [args.output/kind/name for name in ('short-union', 'short-single', 'short-repeat')]
        for path, valid in zip(paths, (union, views[1], union)):
            context_fit.fit_model(train, auxiliary, valid, kind, seed, (1,), path, 'cuda', plan_hash,
                                  args.arm, context='short')
        short_checks = []
        first = read(paths[0]/'completed.json')
        require(meta['contextProfile'] == 'original' and meta['model']['receptiveFieldTicks'] == 125
                and meta['model']['haloTicks'] == 62
                and first['contextProfile'] == 'short' and first['model']['receptiveFieldTicks'] == 33
                and first['model']['haloTicks'] == 16 and first['parameters'] == meta['parameters'],
                'Context profile or parameter count differs')
        for path, ids in ((paths[1], [row.id for row in views[1]]), (paths[2], [row.id for row in union])):
            other = read(path/'completed.json')
            require(first['trainingIdentity'] == other['trainingIdentity'] and first['history'] == other['history']
                    and first['optimizerSteps'] == other['optimizerSteps'] and first['exposureSha256'] == other['exposureSha256'],
                    'Owner reuse changes training identity, history or exposure')
            short_checks.append({'weights': compare_npz(paths[0]/'weights-1.npz', path/'weights-1.npz'),
                                 'predictions': compare_npz(paths[0]/'predictions-1.npz', path/'predictions-1.npz', ids)})
        before = identity(paths[0]/'completed.json')
        context_fit.fit_model(train, auxiliary, union, kind, seed, (1,), paths[0], 'cuda', plan_hash,
                              args.arm, context='short')
        require(before == identity(paths[0]/'completed.json'), 'Resume rewrote completion metadata')
        expected = {key: first[key] for key in ('contractSha256', 'trainIds', 'auxiliaryIds', 'kind', 'seed', 'lossArm')}
        replay = context_fit.predict_checkpoint(paths[0], 1, union, kind=kind, context='short',
                                                expected_identity=expected, device='cuda')
        with np.load(paths[0]/'predictions-1.npz', allow_pickle=False) as saved:
            require(all(replay[key].dtype == saved[key].dtype and replay[key].shape == saved[key].shape
                        and replay[key].tobytes() == saved[key].tobytes() for key in replay),
                    'Saved short-context checkpoint replay differs')
        checked.append({'kind': kind, 'originalProfileReplay': original_checks, 'shortOwnerReplay': short_checks,
                        'resumeMetadataUnchanged': True, 'savedCheckpointReplayBitExact': True,
                        'originalModel': meta['model'], 'shortModel': first['model'],
                        'completionFiles': [identity(path/'completed.json') for path in (original, *paths)]})
        del data, train, auxiliary, views, union
        gc.collect()
        torch.cuda.empty_cache()
    require(all(frozen.base.file_sha256(REPO/'analysis'/name) == sha for name, sha in code.items()),
            'Sources changed during preflight')
    require(frozen.base.file_sha256(manifest) == plan['manifest']['sha256']
            and frozen.base.file_sha256(dino_manifest) == plan['dinoManifest']['sha256'], 'Data changed during preflight')
    result = {**plan, 'passed': True, 'planSha256': plan_hash, 'planFile': identity(args.output/'plan.json'),
              'ownerReuseQualified': True, 'originalProfileReplayQualified': True, 'checks': checked}
    write_new(args.output/'report.json', result)
    print(json.dumps({'passed': True, 'report': identity(args.output/'report.json')}, indent=2))


if __name__ == '__main__':
    main()
