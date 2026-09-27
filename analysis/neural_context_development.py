"""Optional, unregistered context experiment with explicit shared inner fits.

This module never chooses a cohort/arm or creates a registration. Execution
requires a separately frozen registration, completed reference-study audits,
and a new preflight. Six physical unordered-exclusion owners supply twelve
restricted logical inner views; four outer selections/refits remain distinct.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from . import neural_expanded_development as expanded
from . import neural_short_boost_transfer as source
from . import short_context_temporal_model as adapter
from .neural_evaluation import evaluate_predictions

REPO = Path(__file__).resolve().parents[1]
KINDS = ('tcn', 'dino_tcn')
CONTEXTS = ('original', 'short')
EXPERIMENT = 'short-context-development-v1'
FIT_REUSE = {'unorderedInnerExclusions': True, 'physicalInnerOwnersPerCell': 6,
             'logicalInnerViewsPerCell': 12, 'outerRefitsPerCell': 4,
             'trainingHaloTicks': 62, 'trainingCoreTicks': 128}
NEW_SOURCES = {'short_context_temporal_model.py', 'neural_context_fit.py', 'neural_context_development.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    return expanded.base.file_sha256(Path(path))


def identity(path):
    return {'path': str(path), 'sha256': digest(path)}


def write_immutable(path, value):
    """Allow exact resumes, reject conflicting content and concurrent writers."""
    path = Path(path)
    if path.exists():
        require(read(path) == value, f'Immutable artifact changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_verified(reference):
    require(digest(reference['path']) == reference['sha256'], f"Changed input: {reference['path']}")
    return read(reference['path'])


def contains_reference(value, reference):
    if isinstance(value, dict):
        if value.get('path') == reference['path'] and value.get('sha256') == reference['sha256']:
            return True
        return any(contains_reference(item, reference) for item in value.values())
    return isinstance(value, list) and any(contains_reference(item, reference) for item in value)


def completed_reference(reference):
    """No launch with partial results, missing independent audits, or stale hashes."""
    root = Path(reference['path'])
    reg = read_verified({'path': str(root/'preregistration.json'), 'sha256': reference['preregistrationFileSha256']})
    require(reg['sha256'] == canonical_hash(reg['contract']) == reference['contractSha256'], 'Reference registration differs')
    report_ref = {'path': str(root/'report.json'), 'sha256': reference['reportSha256']}
    report = read_verified(report_ref)
    summary = read_verified({'path': str(root/'summary.json'), 'sha256': reference['summarySha256']})
    require(report['status'] == 'completed-short-boost-transfer-development'
            and summary['status'] == 'completed-short-boost-transfer-audit'
            and report['contractSha256'] == summary['contractSha256'] == reg['sha256'], 'Reference study/audit is not complete')
    expected = {(c, k, a, s) for c in previous_cohorts(reg) for k in KINDS for a in source.ARMS for s in reg['contract']['seeds']}
    require(len(report['results']) == len(expected) == 54
            and {(r['cohort'], r['kind'], r['lossArm'], r['seed']) for r in report['results']} == expected
            and all(r['contractSha256'] == reg['sha256'] for r in report['results'])
            and not report['protectedTestOpened']
            and not report['productionPromotionAllowed'], 'Reference population/promotion status differs')
    require(contains_reference(summary, report_ref), 'Summary does not bind reference report')
    audit_kinds = set()
    for item in reference['independentAudits']:
        audit = read_verified(item)
        require(audit['passed'] is True and audit['contractSha256'] == reg['sha256']
                and contains_reference(audit, report_ref), 'Independent audit failed or covers another report')
        audit_kinds.add(audit['kind'])
    require({'neural-short-boost-independent-tensor-audit-v1', 'independent-short-boost-transfer-interval-audit-v1'} <= audit_kinds,
            'Both independent tensor and interval audits are required')
    return reg, report


def previous_cohorts(registration):
    require(registration['contract']['cohorts'] == list(expanded.COHORTS), 'Reference cohorts differ')
    return expanded.COHORTS


def validate_registration(path):
    """Root creates this concrete contract only after choosing the follow-up."""
    registration = read(path)
    contract = registration['contract']
    require(registration['sha256'] == canonical_hash(contract), 'Registration hash differs')
    require(contract['experiment'] == EXPERIMENT and contract['contexts'] == list(CONTEXTS)
            and contract['kinds'] == list(KINDS) and contract['fitReuse'] == FIT_REUSE, 'Unsupported context study')
    require(contract['cohort'] in expanded.COHORTS and contract['cohorts'] == [contract['cohort']]
            and contract['lossArm'] in source.ARMS, 'Cohort and arm must be explicitly locked')
    old, report = completed_reference(contract['referenceStudy'])
    previous = old['contract']
    for field in ('seeds', 'groups', 'checkpointEpochs', 'decoderCandidates', 'selection', 'primaryMetric',
                  'targetPaddingSeconds', 'joinGapSeconds', 'paddingSweep', 'training', 'environment',
                  'manifestSha256', 'dinoManifest', 'screen', 'retentionRecoveryScreen', 'evaluationPopulation'):
        require(contract[field] == previous[field], f'Context comparison changed {field}')
    require(len(contract['groups']) == 4 and len(contract['seeds']) == 3
            and contract['primaryMetric'] == 'F1_padP_coreR' and contract['targetPaddingSeconds'] == 2
            and contract['joinGapSeconds'] == 3 and contract['paddingSweep'] == [0, 1, 2, 3], 'Metric/group scope differs')
    require(set(contract['code']) == set(previous['code']) | NEW_SOURCES
            and all(contract['code'][name] == sha for name, sha in previous['code'].items()), 'Frozen reference sources changed')
    for name, sha in contract['code'].items():
        require(Path(name).name == name and digest(REPO/'analysis'/name) == sha, 'Registered source changed')
    require(contract['models'] == {kind: adapter.model_metadata(kind, context='short') for kind in KINDS}, 'Model adapter metadata differs')
    require(contract['contextModels'] == {context: {kind: adapter.model_metadata(kind, context=context) for kind in KINDS}
                                         for context in CONTEXTS}, 'Context profile metadata differs')
    require(digest(contract['protocolSnapshot']['path']) == contract['protocolSnapshot']['sha256'], 'Prospective protocol changed')
    manifest = read_verified(contract['manifest'])
    require(contract['manifest']['sha256'] == contract['manifestSha256'], 'Manifest identity differs')
    read_verified(contract['dinoManifest'])
    groups = sorted({row['sourceGroup'] for row in manifest['exactRows']})
    require(groups == contract['groups'], 'Exact evaluation source groups differ')
    preflight = read_verified(contract['preflight'])
    require(preflight['passed'] is True and preflight['ownerReuseQualified'] is True
            and preflight['originalProfileReplayQualified'] is True, 'Context preflight has not qualified reuse and no-op replay')
    require(preflight['code'] == contract['code'] and preflight['manifest'] == contract['manifest']
            and preflight['dinoManifest'] == contract['dinoManifest']
            and preflight['referenceContractSha256'] == old['sha256']
            and preflight['cohort'] == contract['cohort'] and preflight['lossArm'] == contract['lossArm'],
            'Preflight binds different code/data/reference/cohort/arm')
    require(preflight['referenceReport'] == {'path': str(Path(contract['referenceStudy']['path'])/'report.json'),
                                           'sha256': contract['referenceStudy']['reportSha256']}
            and preflight['referenceAudits'] == contract['referenceStudy']['independentAudits'],
            'Preflight report/audit bindings differ')
    return registration, manifest, report


def owner_id(excluded_groups):
    require(len(set(excluded_groups)) == 2, 'Inner owner requires two distinct excluded groups')
    return 'exclude-' + canonical_hash(sorted(excluded_groups))[:20]


def owner_plan(contract, contract_hash, manifest, kind, seed, excluded_groups):
    """Training identity is independent of which excluded group is called outer."""
    excluded = sorted(excluded_groups)
    require(set(excluded) <= set(contract['groups']) and len(excluded) == len(set(excluded)) == 2, 'Invalid excluded groups')
    require(kind in contract['kinds'] and seed in contract['seeds'], 'Cell outside registration')
    cohort = contract['cohort']
    tiers = () if cohort == 'exact' else ('draft',) if cohort == 'draft' else ('draft', 'coverage')
    names = {'exact': 'exactRows', 'draft': 'draftRows', 'coverage': 'coverageRows'}
    rows = {tier: [r for r in manifest[names[tier]] if r['sourceGroup'] not in excluded] for tier in ('exact', *tiers)}
    validation = [r for r in manifest['exactRows'] if r['sourceGroup'] in excluded]
    training = {'contractSha256': contract_hash, 'manifestSha256': contract['manifestSha256'],
                'cohort': cohort, 'kind': kind, 'contextProfile': 'short', 'lossArm': contract['lossArm'], 'seed': seed,
                'epochs': contract['checkpointEpochs'], 'model': contract['models'][kind],
                'trainIds': [r['id'] for r in rows['exact']],
                'auxiliaryIds': {tier: [r['id'] for r in rows[tier]] for tier in tiers},
                'trainGroups': sorted({r['sourceGroup'] for r in rows['exact']}),
                'auxiliaryGroups': {tier: sorted({r['sourceGroup'] for r in rows[tier]}) for tier in tiers},
                'orderedTrainingRowsSha256': canonical_hash(rows)}
    require(rows['exact'] and {r['sourceGroup'] for r in validation} == set(excluded), 'Empty training/held group')
    return {'ownerId': owner_id(excluded), 'excludedGroups': excluded, 'training': training,
            'trainingKeySha256': canonical_hash(training), 'validationIds': [r['id'] for r in validation]}


def fold_graph(contract, contract_hash, manifest, kind, seed):
    groups = contract['groups']
    owners = {plan['ownerId']: plan for excluded in combinations(groups, 2)
              for plan in [owner_plan(contract, contract_hash, manifest, kind, seed, excluded)]}
    views = []
    for outer_index, outer in enumerate(groups):
        for inner_index, inner in enumerate(g for g in groups if g != outer):
            views.append({'outerIndex': outer_index, 'outerHeldSourceGroup': outer,
                          'innerIndex': inner_index, 'innerValidationGroup': inner,
                          'ownerId': owner_id((outer, inner)),
                          'validationIds': [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] == inner]})
    require(len(owners) == 6 and len(views) == 12, 'Unexpected owner/logical-view count')
    return owners, views


def validate_completion(folder, expected):
    meta = read(folder/'completed.json')
    for field in ('contractSha256', 'kind', 'seed', 'contextProfile', 'lossArm', 'epochs', 'trainIds', 'auxiliaryIds',
                  'trainGroups', 'auxiliaryGroups', 'validationIds'):
        require(meta[field] == expected[field], f'Fit completion changed {field}')
    require(meta['model'] == expected['model'] and meta['scalerTrainIds'] == expected['trainIds'], 'Fit model/scaler provenance differs')
    require(meta['trainingIdentitySha256'] == canonical_hash(meta['trainingIdentity'])
            and all(meta.get(key) == value for key, value in meta['trainingIdentity'].items()), 'Fit training identity differs')
    required = {f'{stem}-{epoch}.npz' for epoch in expected['epochs'] for stem in ('weights', 'predictions')}
    require(set(meta['artifacts']) == required, 'Fit artifact inventory differs')
    for name, sha in meta['artifacts'].items():
        require(digest(folder/name) == sha, 'Fit artifact hash differs')
    return meta


def prediction_view(probabilities, owner_ids, requested_ids, examples):
    """A logical selector receives only its designated held group's scores."""
    require(set(probabilities) == set(owner_ids), 'Owner prediction inventory differs')
    require(len(requested_ids) == len(set(requested_ids)) and set(requested_ids) <= set(owner_ids), 'Logical view escapes owner scope')
    result = {}
    for identifier in requested_ids:
        values, example = probabilities[identifier], examples[identifier]
        require(values.shape == (len(example.times), 4) and values.dtype == np.float32
                and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), 'Invalid logical prediction array')
        require(np.all(values[~example.valid] == 0), 'Ignored ticks contain predictions')
        result[identifier] = values
    return result


def verify_logical_reference(view, plan, folder, meta):
    require(view['ownerId'] == plan['ownerId'] and sorted([view['outerHeldSourceGroup'], view['innerValidationGroup']]) == plan['excludedGroups'],
            'Logical view has wrong exclusion owner')
    return {**view, 'ownerPath': str(folder), 'ownerCompletedSha256': digest(folder/'completed.json'),
            'trainingKeySha256': plan['trainingKeySha256'], 'ownerValidationIds': plan['validationIds'],
            'trainingIdentitySha256': meta['trainingIdentitySha256'],
            'checkpointArtifacts': {name: sha for name, sha in meta['artifacts'].items()}}


def load_logical_predictions(reference, plan, examples, epoch):
    require(reference['trainingKeySha256'] == plan['trainingKeySha256']
            and reference['ownerValidationIds'] == plan['validationIds'], 'Logical reference training identity changed')
    require(reference['ownerId'] == plan['ownerId']
            and sorted([reference['outerHeldSourceGroup'], reference['innerValidationGroup']]) == plan['excludedGroups'], 'Logical reference changed fold role')
    expected_ids = [identifier for identifier, e in examples.items() if e.group == reference['innerValidationGroup']]
    require(reference['validationIds'] == expected_ids and all(examples[i].group != reference['outerHeldSourceGroup'] for i in expected_ids),
            'Logical validation population includes wrong/outer group')
    folder = Path(reference['ownerPath'])
    require(digest(folder/'completed.json') == reference['ownerCompletedSha256'], 'Logical owner metadata changed')
    require(read(folder/'completed.json')['trainingIdentitySha256'] == reference['trainingIdentitySha256'], 'Logical training identity differs')
    filename = f'predictions-{epoch}.npz'
    require(digest(folder/filename) == reference['checkpointArtifacts'][filename], 'Logical checkpoint changed')
    with np.load(folder/filename, allow_pickle=False) as cache:
        # NPZ is lazy: only the permitted logical group's arrays are materialized.
        require(set(cache.files) == set(plan['validationIds']), 'Owner prediction scope differs')
        values = {identifier: cache[identifier] for identifier in reference['validationIds']}
    return prediction_view(values, reference['validationIds'], reference['validationIds'], examples)


def run_cell(data, manifest, contract, contract_hash, kind, seed, output, device, *, fit_fn=None, choose_fn=None):
    if fit_fn is None:
        from .neural_context_fit import fit_model
        fit_fn = fit_model
    choose_fn = choose_fn or expanded.choose_settings
    owners, views = fold_graph(contract, contract_hash, manifest, kind, seed)
    cohort, arm = contract['cohort'], contract['lossArm']
    root = output/'fits'/cohort/kind/arm/'short'/str(seed)
    examples = {r.example.id: r.example for r in data['exact']}
    require(list(examples) == [r['id'] for r in manifest['exactRows']], 'Loaded exact scope/order differs')
    owner_meta = {}
    for key, plan in owners.items():
        excluded = set(plan['excludedGroups'])
        train = [r for r in data['exact'] if r.example.group not in excluded]
        auxiliary = expanded.auxiliary_for_fold(data, cohort, excluded)
        validation = [e for e in examples.values() if e.group in excluded]
        require([r.example.id for r in train] == plan['training']['trainIds']
                and {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()} == plan['training']['auxiliaryIds'], 'Loaded owner training membership differs')
        folder = root/'inner-owners'/key
        write_immutable(root/'owner-plans'/f'{key}.json', plan)
        print(f'FIT {cohort} {kind} {arm} short seed={seed} excluded={",".join(plan["excludedGroups"])}', flush=True)
        fit_fn(train, auxiliary, validation, kind, seed, tuple(contract['checkpointEpochs']), folder, device,
               contract_hash, arm, context='short')
        owner_meta[key] = validate_completion(folder, {**plan['training'], 'validationIds': plan['validationIds']})
    references = []
    for view in views:
        plan = owners[view['ownerId']]
        reference = verify_logical_reference(view, plan, root/'inner-owners'/view['ownerId'], owner_meta[view['ownerId']])
        path = root/'logical-views'/f"outer-{view['outerIndex']}"/f"inner-{view['innerIndex']}.json"
        write_immutable(path, reference)
        references.append({'path': str(path), 'sha256': digest(path)})
    outer_rows, selections = [], []
    for outer_index, outer in enumerate(contract['groups']):
        fitting = [r for r in data['exact'] if r.example.group != outer]
        held = [e for e in examples.values() if e.group == outer]
        expected_ids = {r.example.id for r in fitting}
        probabilities = {epoch: {} for epoch in contract['checkpointEpochs']}
        logical = [read_verified(ref) for ref in references if read(ref['path'])['outerIndex'] == outer_index]
        require(len(logical) == 3, 'Outer selection does not have three logical folds')
        for view in logical:
            for epoch in contract['checkpointEpochs']:
                scores = load_logical_predictions(view, owners[view['ownerId']], examples, epoch)
                require(not set(probabilities[epoch]) & set(scores), 'Duplicate validation row in inner pool')
                probabilities[epoch].update(scores)
        require(all(set(values) == expected_ids for values in probabilities.values()), 'Inner pool includes outer/missing rows')
        selected = choose_fn([r.example for r in fitting], probabilities)
        require(selected['epoch'] in contract['checkpointEpochs'] and selected['decoder'] in contract['decoderCandidates'], 'Selection outside frozen grid')
        require(selected['recallEligibilityFloor'] == .95
                and selected['recallEligibilityPassed'] == (selected['innerR_core'] >= .95), 'Selection feasibility differs')
        epoch = selected['epoch']
        auxiliary = expanded.auxiliary_for_fold(data, cohort, {outer})
        folder = root/f'outer-{outer_index}'/'refit'
        print(f'REFIT {cohort} {kind} {arm} short seed={seed} outer={outer} epoch={epoch}', flush=True)
        fit_fn(fitting, auxiliary, held, kind, seed, (epoch,), folder, device, contract_hash, arm, context='short')
        expected = {'contractSha256': contract_hash, 'kind': kind, 'seed': seed, 'contextProfile': 'short', 'lossArm': arm,
                    'epochs': [epoch], 'trainIds': [r.example.id for r in fitting],
                    'auxiliaryIds': {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                    'trainGroups': sorted({r.example.group for r in fitting}),
                    'auxiliaryGroups': {tier: sorted({r.example.group for r in rows}) for tier, rows in auxiliary.items()},
                    'validationIds': [e.id for e in held], 'model': contract['models'][kind]}
        meta = validate_completion(folder, expected)
        with np.load(folder/f'predictions-{epoch}.npz', allow_pickle=False) as cache:
            scores = prediction_view({name: cache[name] for name in cache.files}, expected['validationIds'], expected['validationIds'], examples)
        outer_rows.extend(e.row(expanded.base.decode(e, scores[e.id], selected['decoder'])) for e in held)
        selections.append({'heldSourceGroup': outer, **selected})
    result = {'cohort': cohort, 'kind': kind, 'architecture': kind, 'lossArm': arm, 'context': 'short', 'seed': seed,
              'contractSha256': contract_hash, 'origin': {'type': 'trained-context', 'fitRoot': str(root), 'innerOwnership': 'six-unordered-exclusions'},
              'logicalInnerViews': references, 'selections': selections, 'evaluation': evaluate_predictions(outer_rows),
              'predictions': [{**row, **{key: [i.to_dict() for i in row[key]] for key in ('rallies', 'ignoredIntervals', 'predictions')}} for row in outer_rows]}
    write_immutable(output/f'result-{cohort}-{kind}-{arm}-short-{seed}.json', result)
    return result


def reference_result(row, reference, contract_hash):
    result = deepcopy(row)
    result.update(context='original', contractSha256=contract_hash,
                  origin={'type': 'reused-context-reference', 'studyPath': reference['path'],
                          'reportSha256': reference['reportSha256'], 'referenceContractSha256': reference['contractSha256'],
                          'sourceOrigin': deepcopy(row['origin']),
                          'sourceResult': {key: row[key] for key in ('cohort', 'kind', 'lossArm', 'seed')}})
    return result


def run_registered_job(registration_path, kind, seed, device):
    import torch

    registration, manifest, _ = validate_registration(registration_path)
    contract = registration['contract']
    require(kind in KINDS and seed in contract['seeds'], 'Unregistered job')
    actual_environment = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                          'device': device, 'gpu': torch.cuda.get_device_name() if device.startswith('cuda') else None}
    require(actual_environment == contract['environment'], 'Execution environment differs from registration')
    torch.set_num_threads(2)
    data = source.load_data(Path(contract['manifest']['path']), Path(contract['dinoManifest']['path']), with_dino=kind == 'dino_tcn')
    return run_cell(data, manifest, contract, registration['sha256'], kind, seed, registration_path.parent, device)


def run_study(registration_path, device, workers):
    registration, _, report = validate_registration(registration_path)
    contract, output = registration['contract'], registration_path.parent
    require(not (output/'report.json').exists(), 'Completed context study exists')
    require(workers in (1, 2, 3), 'Unsupported worker count')
    selected = [r for r in report['results'] if r['cohort'] == contract['cohort'] and r['lossArm'] == contract['lossArm']]
    require({(r['kind'], r['seed']) for r in selected} == {(k, s) for k in KINDS for s in contract['seeds']} and len(selected) == 6,
            'Reference study does not contain exactly six matched cells')
    results = [reference_result(r, contract['referenceStudy'], registration['sha256']) for r in selected]
    for row in results:
        write_immutable(output/f"result-{row['cohort']}-{row['kind']}-{row['lossArm']}-original-{row['seed']}.json", row)
    jobs = [(kind, seed) for kind in KINDS for seed in contract['seeds']]
    started = time.perf_counter()
    execution_path = output/f'execution-{time.time_ns()}.json'
    execution = {'contractSha256': registration['sha256'], 'status': 'running',
                 'startedAt': datetime.now(timezone.utc).isoformat(), 'workers': workers,
                 'freshResultCells': 6, 'reusedResultCells': 6, 'jobOrder': [list(job) for job in jobs]}
    write_immutable(execution_path, execution)
    logs = output/'job-logs'
    logs.mkdir(exist_ok=True)
    def launch(job):
        kind, seed = job
        print(f'START {kind} short seed={seed} {datetime.now(timezone.utc).isoformat()}', flush=True)
        with (logs/f'{kind}-{seed}-{time.time_ns()}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'analysis.neural_context_development', '--registration', str(registration_path),
                            '--device', device, '--job', f'{kind}:{seed}'], check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f'DONE {kind} short seed={seed} {datetime.now(timezone.utc).isoformat()}', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(launch, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        for kind, seed in jobs:
            results.append(read(output/f"result-{contract['cohort']}-{kind}-{contract['lossArm']}-short-{seed}.json"))
        final_registration, _, _ = validate_registration(registration_path)
        require(final_registration == registration, 'Registration changed during execution')
        require(all(r['contractSha256'] == registration['sha256'] for r in results), 'Result contract changed')
        final = {'schemaVersion': 1, 'status': 'completed-context-development', 'contractSha256': registration['sha256'],
                 'manifestSha256': contract['manifestSha256'], 'records': report['records'], 'sourceGroups': contract['groups'],
                 'protectedTestOpened': False, 'productionPromotionAllowed': False,
                 'execution': {'started': identity(execution_path), 'workers': workers, 'wallSeconds': time.perf_counter()-started,
                               'completedAt': datetime.now(timezone.utc).isoformat(), 'physicalFreshFits': 60, 'logicalFreshFits': 96},
                 'results': results}
        write_immutable(output/'report.json', final)
    except BaseException as error:
        write_immutable(execution_path.with_name(execution_path.stem+'-failed.json'),
                        {**execution, 'status': 'failed', 'started': identity(execution_path),
                         'errorType': type(error).__name__, 'error': str(error),
                         'endedAt': datetime.now(timezone.utc).isoformat(), 'wallSeconds': time.perf_counter()-started})
        raise
    write_immutable(execution_path.with_name(execution_path.stem+'-completed.json'),
                    {**execution, 'status': 'completed', 'started': identity(execution_path),
                     'endedAt': datetime.now(timezone.utc).isoformat(), 'wallSeconds': time.perf_counter()-started,
                     'report': identity(output/'report.json')})
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--job')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.validate_only:
        registration, _, _ = validate_registration(args.registration)
        print(json.dumps({'validated': True, 'contractSha256': registration['sha256'], 'trained': False}))
    elif args.job:
        kind, seed = args.job.split(':')
        run_registered_job(args.registration, kind, int(seed), args.device)
    else:
        run_study(args.registration, args.device, args.workers)


if __name__ == '__main__':
    main()
