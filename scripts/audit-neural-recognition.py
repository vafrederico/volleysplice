#!/usr/bin/env python3
"""Final-only independent recognition-study audit. No fitting or GPU access.

Reconstructs source exclusions, scalers, sampling exposure, logical validation
views and decoder selection from saved owners. Reuses pinned independent
endpoint-sweep arithmetic, not the training evaluator's interval operations.
CPU checkpoint replay samples the first/middle/last real-context chunks of each
held recording at every checkpoint; this is explicitly not full neural replay.
Only --output is written, exclusively; all study/evidence inputs are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HELPERS = {
    'audit-neural-context-tensors.py': '06aec8f98164ea8a64a274999dba56de3c2b37921ed35f48cc933b32a5f2fb33',
    'audit-neural-short-boost-intervals.py': '8ed1ed75ebc2511c9603642c0b09ba5888554910072a6e1d1c5490a65864b81b',
}
TIERS = ('exact', 'draft', 'coverage')
REPLAY_RTOL, REPLAY_ATOL = 1e-4, 2e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    return {'path': str(path), 'sha256': digest(path)}


def verify_binding(binding):
    require(isinstance(binding, dict) and 'path' in binding and 'sha256' in binding, 'Missing file binding')
    require(digest(binding['path']) == binding['sha256'], 'Bound file changed: '+binding['path'])
    return Path(binding['path'])


def helpers():
    result = []
    for name, sha in HELPERS.items():
        path = REPO/'scripts'/name
        require(digest(path) == sha, 'Independent audit helper changed: '+name)
        spec = importlib.util.spec_from_file_location(name.replace('-', '_').replace('.', '_'), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return tuple(result)


def compare(actual, expected, label='value'):
    """Exact inventories/types with small tolerance only for real-valued metrics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label+': mapping inventory differs')
        for key in expected:
            compare(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+': list length differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, label+f'[{index}]')
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), label+': numeric mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, label+': value differs')


def arrays_identical(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), 'Engineering archive inventory differs')
        for name in a.files:
            require(a[name].dtype == b[name].dtype and a[name].shape == b[name].shape
                    and a[name].tobytes() == b[name].tobytes(), 'Engineering control not bit-exact: '+name)


def verify_artifacts(folder, meta):
    expected = {f'{stem}-{epoch}.npz' for epoch in meta['epochs'] for stem in ('weights', 'predictions')}
    require(set(meta['artifacts']) == expected, 'Checkpoint artifact inventory differs: '+str(folder))
    require({path.name for path in folder.glob('*.npz')} == expected, 'Unexpected checkpoint archive: '+str(folder))
    for name, sha in meta['artifacts'].items():
        require(digest(folder/name) == sha, 'Checkpoint artifact changed: '+str(folder/name))


def verify_engineering(binding, contract):
    path = verify_binding(binding)
    report = read(path)
    require(report['passed'] is True and report['controlBitExact'] is True
            and report['attentionExposureIdentical'] is True
            and report['protectedTestOpened'] is False
            and report['qualitySelectionPerformed'] is False, 'Engineering audit did not pass prospectively')
    require(report['manifest'] == contract['manifest'], 'Engineering input revision differs')
    require(all(contract['code'].get(name) == sha for name, sha in report['code'].items()), 'Engineering source revision differs')
    sources, folders = {}, {}
    for name in ('legacy', 'control', 'attention'):
        completed = verify_binding(report[name])
        folders[name] = completed.parent
        sources[name] = read(completed)
        verify_artifacts(completed.parent, sources[name])
    common = ('trainIds', 'auxiliaryIds', 'validationIds', 'scalerTrainIds', 'positiveWeight',
              'supervisedCounts', 'liveLossWeighting', 'epochs', 'optimizerSteps', 'exposureSha256')
    for name in ('control', 'attention'):
        for key in common:
            require(sources[name][key] == sources['legacy'][key], 'Engineering recipe differs: '+name+'/'+key)
        expected = [{k: row[k] for k in ('epoch', 'optimizerSteps', 'exposureSha256')} for row in sources['legacy']['history']]
        actual = [{k: row[k] for k in ('epoch', 'optimizerSteps', 'exposureSha256')} for row in sources[name]['history']]
        require(actual == expected, 'Engineering exposure history differs')
    require(sources['legacy']['history'] == sources['control']['history'], 'Engineering control history differs')
    for epoch in sources['legacy']['epochs']:
        for stem in ('weights', 'predictions'):
            arrays_identical(folders['legacy']/f'{stem}-{epoch}.npz', folders['control']/f'{stem}-{epoch}.npz')
        with np.load(folders['control']/f'weights-{epoch}.npz', allow_pickle=False) as control, \
                np.load(folders['attention']/f'weights-{epoch}.npz', allow_pickle=False) as attention:
            for key in ('mean', 'scale'):
                require(np.array_equal(control[key], attention[key]), 'Engineering transformer scaler differs')
    return {'report': identity(path), 'controlBitExactRecomputed': True,
            'attentionExposureRecomputed': True, 'checkpointEpochs': sources['legacy']['epochs']}


def verify_registration(study):
    registration = read(study/'preregistration.json')
    contract = registration['contract']
    require(canonical(contract) == registration['sha256'], 'Registration hash differs')
    require(contract['protectedTestOpened'] is False and contract['productionPromotionAllowed'] is False,
            'Protected-test/production scope changed')
    require(contract['cohort'] == 'reviewed_export' and contract['lossArm'] == 'short_boost', 'Unregistered cohort/loss')
    require(contract['targetPaddingSeconds'] == 2 and contract['paddingSeconds'] == [0, 1, 2, 3]
            and contract['joinGapSeconds'] == 3 and contract['trainingCoreTicks'] == 128
            and contract['trainingHaloTicks'] == 62, 'Metric/chunk contract differs')
    require(len(contract['groups']) == len(set(contract['groups'])) == 4
            and contract['groups'] == sorted(contract['groups']), 'Four distinct ordered source groups required')
    require(contract['checkpointEpochs'] == sorted(set(contract['checkpointEpochs']))
            and all(type(epoch) is int and epoch > 0 for epoch in contract['checkpointEpochs']), 'Invalid checkpoints')
    for name, sha in contract['code'].items():
        require(Path(name).name == name, 'Unsafe source snapshot name')
        require(digest(REPO/'analysis'/name) == sha and digest(study/'registered-sources'/name) == sha,
                'Registered current/archive source differs: '+name)
    require({path.name for path in (study/'registered-sources').iterdir()} == set(contract['code']), 'Source snapshot inventory differs')
    for name in ('manifest', 'protocol', 'referenceStudy', 'referenceReport'):
        verify_binding(contract[name])
    require(digest(study/'protocol.md') == contract['protocol']['sha256'], 'Protocol snapshot differs')
    for name in ('dinoManifest', 'featureManifest'):
        if contract.get(name):
            verify_binding(contract[name])
    reference = read(contract['referenceStudy']['path'])
    require(canonical(reference['contract']) == reference['sha256'], 'Reference contract differs')
    require(all(contract['code'].get(name) == sha for name, sha in reference['contract']['code'].items()), 'Legacy source recipe differs')
    reference_report = read(contract['referenceReport']['path'])
    require(reference_report['contractSha256'] == reference['sha256'], 'Reference report contract differs')
    engineering = verify_engineering(contract['engineeringAudit'], contract)
    return registration, engineering


def load_inputs(contract):
    from analysis import neural_short_boost_transfer as source
    from analysis.recognition_temporal_model import RecognitionConfig, model_metadata
    config = RecognitionConfig(**contract['config'])
    require(model_metadata(config) == contract['model'], 'Registered architecture differs')
    manifest_path = Path(contract['manifest']['path'])
    manifest = read(manifest_path)
    for tier in TIERS:
        require([row['id'] for row in manifest[tier+'Rows']] == contract['population'][tier+'Rows'], 'Population inventory differs')
    data = source.load_data(manifest_path, Path(contract['dinoManifest']['path']) if contract.get('dinoManifest') else None,
                            with_dino=config.family == 'dino')
    if config.family in ('player', 'mobile'):
        from analysis.neural_recognition_inputs import attach_features
        require(contract.get('featureManifest') is not None, 'Missing added-feature identity')
        data = attach_features(data, Path(contract['featureManifest']['path']), config, manifest_path)
    for tier in TIERS:
        require([row.example.id for row in data[tier]] == contract['population'][tier+'Rows'], 'Loaded population order differs')
        for row in data[tier]:
            require(row.example.values.dtype == np.float32 and row.example.values.shape == (len(row.example.times), config.input_dimension)
                    and np.isfinite(row.example.values).all(), 'Invalid loaded features')
    require(sorted({row.example.group for row in data['exact']}) == contract['groups'], 'Loaded source groups differ')
    return data, config


def fold_members(data, excluded):
    rows = {tier: [row for row in data[tier] if row.example.group not in excluded] for tier in TIERS}
    validation = [row.example for row in data['exact'] if row.example.group in excluded]
    expected = {'trainIds': [row.example.id for row in rows['exact']],
        'auxiliaryIds': {tier: [row.example.id for row in rows[tier]] for tier in TIERS[1:]},
        'trainGroups': sorted({row.example.group for row in rows['exact']}),
        'auxiliaryGroups': {tier: sorted({row.example.group for row in rows[tier]}) for tier in TIERS[1:]},
        'validationIds': [example.id for example in validation],
        'validationGroups': sorted({example.group for example in validation})}
    return rows, validation, expected


def scaler_indexes(config):
    start = 104 + (config.token_count * config.token_dimension if config.family == 'mobile' else 0)
    return np.r_[np.arange(104), np.arange(start, start + config.scalar_dimension)]


def independent_scaler(rows, config):
    # The historical code computes AV and scalar moments separately; retain the
    # same array layout/reduction order while excluding all nonexact/invalid rows.
    indexes = scaler_indexes(config)
    av = np.concatenate([row.example.values[row.example.valid, :104] for row in rows])
    mean = av.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(av.std(0, dtype=np.float64), 1e-4).astype(np.float32)
    if len(indexes) > 104:
        scalars = np.concatenate([row.example.values[row.example.valid][:, indexes[104:]] for row in rows])
        mean = np.r_[mean, scalars.mean(0, dtype=np.float64).astype(np.float32)]
        scale = np.r_[scale, np.maximum(scalars.std(0, dtype=np.float64), 1e-4).astype(np.float32)]
    return mean, scale


def independent_counts(rows):
    if not rows:
        return {'valid': [0]*4, 'positiveMass': [0]*4}
    return {'valid': np.sum([row.mask.sum(axis=0) for row in rows], axis=0).astype(int).tolist(),
            'positiveMass': np.sum([(row.example.targets*row.mask).sum(axis=0) for row in rows], axis=0).tolist()}


def real_chunks(example):
    edges = np.diff(np.r_[False, example.valid, False].astype(np.int8))
    result = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        for core in range(int(start), int(end), 128):
            finish = min(int(end), core+128)
            result.append((max(int(start), core-62), min(int(end), finish+62), core, finish))
    return result


def replay_sample(model, example, saved, mean, scale, config):
    import torch
    chunks = real_chunks(example)
    require(chunks, 'Held recording has no real context')
    selected = sorted({0, len(chunks)//2, len(chunks)-1})
    indexes = scaler_indexes(config)
    maximum, points = 0., 0
    locations = []
    with torch.inference_mode():
        for index in selected:
            left, right, core, finish = chunks[index]
            values = example.values[left:right].copy()
            values[:, indexes] = np.clip((values[:, indexes]-mean)/scale, -10, 10)
            output = torch.sigmoid(model(torch.from_numpy(values[None])))[0].numpy()[core-left:finish-left]
            expected = saved[core:finish]
            require(np.allclose(output, expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                    'Saved prediction/checkpoint CPU mismatch: '+example.id)
            maximum = max(maximum, float(np.max(np.abs(output-expected))))
            points += len(output)
            locations.append({'left': left, 'right': right, 'coreStart': core, 'coreEnd': finish})
    return {'recordingId': example.id, 'checkedTicks': points, 'maximumAbsoluteDifference': maximum,
            'sampledChunks': locations, 'totalChunks': len(chunks)}


def audit_fit(folder, data, excluded, epochs, seed, registration, config, tensor_helper):
    import torch
    from analysis import neural_short_boost_weighting as weighting
    from analysis.recognition_temporal_model import model_for
    contract = registration['contract']
    rows, validation, members = fold_members(data, excluded)
    meta = read(folder/'completed.json')
    expected = {**members, 'contractSha256': registration['sha256'], 'kind': f'{config.family}_{config.head}',
                'seed': seed, 'lossArm': contract['lossArm'], 'epochs': list(epochs), 'model': contract['model']}
    for key, value in expected.items():
        require(meta[key] == value, 'Fit identity differs: '+str(folder)+'/'+key)
    require(meta['scalerTrainIds'] == members['trainIds'], 'Scaler includes auxiliary/held records')
    counts = {tier: independent_counts(records) for tier, records in rows.items()}
    require(meta['supervisedCounts'] == counts, 'Supervision counts differ')
    positives = np.asarray(counts['exact']['positiveMass'])
    positive_weight = np.minimum(20., np.sqrt((np.asarray(counts['exact']['valid'])-positives)/np.maximum(positives, 1.)))
    require(meta['positiveWeight'] == positive_weight.tolist(), 'Positive weights differ')
    expected_weighting = {'mode': contract['lossArm'], 'rows': {
        tier: [weighting.live_event_weights(row, contract['lossArm'])[1] for row in records] for tier, records in rows.items()}}
    require(meta['liveLossWeighting'] == expected_weighting, 'Live-loss exposure differs')
    inputs = {row.example.id: {'group': row.example.group, 'valid': row.example.valid} for records in rows.values() for row in records}
    supervision = {row.example.id: {'mask': row.mask} for records in rows.values() for row in records}
    pools = {tier: tensor_helper.chunk_geometry([row.example.id for row in records], inputs, supervision)
             for tier, records in rows.items()}
    exposure = tensor_helper.independent_exposure(pools, seed, max(epochs))
    tensor_helper.validate_history(meta, exposure)
    verify_artifacts(folder, meta)
    mean, scale = independent_scaler(rows['exact'], config)
    with torch.random.fork_rng(devices=[]):
        model = model_for(config).cpu().eval()
    template = model.state_dict()
    prediction_sets, replayed = {}, []
    for epoch in epochs:
        with np.load(folder/f'weights-{epoch}.npz', allow_pickle=False) as cache:
            require(set(cache.files) == {'mean', 'scale', *('model::'+name for name in template)}, 'Weight tensor inventory differs')
            for name, reference in (('mean', mean), ('scale', scale)):
                require(cache[name].dtype == np.float32 and np.array_equal(cache[name], reference), 'Independent fold scaler differs')
            state = {}
            for name, tensor in template.items():
                value = cache['model::'+name]
                require(value.dtype == np.float32 and value.shape == tuple(tensor.shape) and np.isfinite(value).all(), 'Invalid model tensor: '+name)
                state[name] = torch.from_numpy(value.copy())
            model.load_state_dict(state, strict=True)
        with np.load(folder/f'predictions-{epoch}.npz', allow_pickle=False) as cache:
            require(set(cache.files) == {example.id for example in validation}, 'Prediction scope differs')
            predictions = {}
            for example in validation:
                value = cache[example.id]
                require(value.dtype == np.float32 and value.shape == (len(example.times), 4)
                        and np.isfinite(value).all() and np.all((0 <= value) & (value <= 1))
                        and np.all(value[~example.valid] == 0), 'Invalid prediction tensor: '+example.id)
                predictions[example.id] = value.copy()
                replayed.append({'epoch': epoch, **replay_sample(model, example, value, mean, scale, config)})
            prediction_sets[epoch] = predictions
    return prediction_sets, {'path': str(folder), 'completed': identity(folder/'completed.json'),
        'excludedGroups': sorted(excluded), 'epochs': list(epochs), 'checkpointArtifacts': meta['artifacts'],
        'members': members, 'optimizerSteps': exposure[-1]['optimizerSteps'],
        'exposureSha256': exposure[-1]['exposureSha256'], 'cpuReplay': replayed}


def serialized_rows(rows):
    return [{**row, **{key: [item.to_dict() for item in row[key]] for key in ('rallies', 'ignoredIntervals', 'predictions')}} for row in rows]


def reselect(examples, probabilities, contract, interval_helper):
    from analysis import neural_expanded_development as expanded
    from analysis.crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
    require(contract['decoderCandidates'] == expanded.decoder_candidates(), 'Decoder grid differs')
    candidates = []
    for epoch in sorted(probabilities):
        require(set(probabilities[epoch]) == {example.id for example in examples}, 'Logical view includes wrong records')
        for decoder in contract['decoderCandidates']:
            rows = [example.row(expanded.base.decode(example, probabilities[epoch][example.id], decoder)) for example in examples]
            recordings = [RecordingIntervals(row['id'], 'development', row['durationSeconds'], tuple(row['rallies']),
                          tuple(row['predictions']), tuple(row['ignoredIntervals'])) for row in rows]
            score = evaluate_f1_pad_p_core_r(recordings, [2.], 3.)[0]
            independent = interval_helper.pooled_metric([interval_helper.parse_record(row) for row in serialized_rows(rows)], 2)
            for key in ('F1_padP_coreR', 'R_core', 'P_pad'):
                compare(score[key], independent[key], 'inner independent '+key)
            candidates.append({'epoch': epoch, 'decoder': decoder,
                               'innerF1_padP_coreR': score['F1_padP_coreR'], 'innerR_core': score['R_core']})
    eligible = [candidate for candidate in candidates if candidate['innerR_core'] >= .95]
    selected = max(eligible or candidates, key=lambda candidate: candidate['innerF1_padP_coreR'])
    return {**selected, 'recallEligibilityPassed': bool(eligible), 'recallEligibilityFloor': .95}, candidates


def independent_metrics(rows, evaluated, helper):
    records = [helper.parse_record(row) for row in serialized_rows(rows)]
    padding = []
    for value in (0, 1, 2, 3):
        independent = helper.pooled_metric(records, value)
        canonical_row = next(row for row in evaluated['padding'] if row['paddingSecondsBeforeAndAfter'] == value)
        for key in (*helper.SUM_FIELDS, 'P_pad', 'R_core', 'F1_padP_coreR', 'exportDurationDifferenceSeconds'):
            compare(canonical_row[key], independent[key], 'independent padding '+key)
        padding.append({key: independent[key] for key in (*helper.SUM_FIELDS, 'P_pad', 'R_core', 'F1_padP_coreR',
            'exportDurationDifferenceSeconds', 'paddingSecondsBeforeAndAfter', 'joinGapSeconds')})
    coverage = {}
    for scope in ('coreCoverage', 'primaryExportCoverage'):
        coverage[scope] = helper.event_coverage(records, scope)
        compare(evaluated['guardrails'][scope], coverage[scope], 'independent '+scope)
    events = coverage['primaryExportCoverage']['rallies']
    return {'padding': padding, 'coverage': coverage,
        'shortCompleteRallyLosses': sum(row['completelyLost'] for row in events if row['end']-row['start'] <= 3),
        'longCompleteRallyLosses': sum(row['completelyLost'] for row in events if row['end']-row['start'] > 3)}


def audit_seed(study, data, registration, config, seed, tensor_helper, interval_helper):
    from analysis import neural_expanded_development as expanded
    from analysis.neural_evaluation import evaluate_predictions
    contract = registration['contract']
    groups = contract['groups']
    folder = study/'fits'/str(seed)
    expected_names = {f'inner-{left}-{right}' for left, right in combinations(range(4), 2)} | {f'outer-{index}' for index in range(4)}
    require({path.name for path in folder.iterdir() if path.is_dir()} == expected_names, 'Physical fit inventory differs')
    owners, fit_checks, views, selections, rows, grids = {}, [], [], [], [], []
    for left, right in combinations(range(4), 2):
        excluded = {groups[left], groups[right]}
        owners[(left, right)], checked = audit_fit(folder/f'inner-{left}-{right}', data, excluded,
            tuple(contract['checkpointEpochs']), seed, registration, config, tensor_helper)
        fit_checks.append(checked)
        _, _, members = fold_members(data, excluded)
        plan = {'excludedGroups': sorted(excluded), **{key: members[key] for key in ('trainIds', 'auxiliaryIds', 'validationIds')},
                'contractSha256': registration['sha256']}
        require(read(folder/f'plan-inner-{left}-{right}.json') == plan, 'Owner plan differs')
    for outer_index, outer in enumerate(groups):
        examples = [row.example for row in data['exact'] if row.example.group != outer]
        probabilities = {epoch: {} for epoch in contract['checkpointEpochs']}
        for inner_index, inner in enumerate(groups):
            if inner_index == outer_index:
                continue
            key = tuple(sorted((outer_index, inner_index)))
            allowed = {row.example.id for row in data['exact'] if row.example.group == inner}
            for epoch in probabilities:
                require(not set(probabilities[epoch]) & allowed, 'Duplicate logical inner scope')
                probabilities[epoch].update({identifier: owners[key][epoch][identifier] for identifier in allowed})
            views.append({'outerGroup': outer, 'innerGroup': inner,
                          'owner': f'inner-{key[0]}-{key[1]}', 'allowedIds': sorted(allowed)})
        selected, candidates = reselect(examples, probabilities, contract, interval_helper)
        selections.append({'heldSourceGroup': outer, **selected})
        grids.append({'heldSourceGroup': outer, 'candidates': candidates})
        predictions, checked = audit_fit(folder/f'outer-{outer_index}', data, {outer}, (selected['epoch'],),
                                         seed, registration, config, tensor_helper)
        fit_checks.append(checked)
        for row in data['exact']:
            if row.example.group == outer:
                rows.append(row.example.row(expanded.base.decode(row.example,
                    predictions[selected['epoch']][row.example.id], selected['decoder'])))
    require(len(fit_checks) == 10 and len(views) == 12, 'Fit/view count differs')
    evaluated = evaluate_predictions(rows)
    stored = read(study/f'result-{seed}.json')
    expected = {'contractSha256': registration['sha256'], 'family': config.family, 'head': config.head, 'seed': seed,
                'model': contract['model'], 'selections': selections, 'evaluation': evaluated,
                'predictions': serialized_rows(rows), 'protectedTestOpened': False, 'productionPromotionAllowed': False}
    compare(stored, expected, 'saved result')
    metrics = independent_metrics(rows, evaluated, interval_helper)
    return {'seed': seed, 'result': identity(study/f'result-{seed}.json'), 'fits': fit_checks,
            'logicalViews': views, 'selections': selections, 'selectionGrids': grids,
            'independent': metrics, 'evaluation': evaluated}


def audit(study):
    import torch
    tensor_helper, interval_helper = helpers()
    registration, engineering = verify_registration(study)
    contract = registration['contract']
    require((study/'report.json').exists(), 'Study incomplete: final report absent')
    report = read(study/'report.json')
    require(report['status'] == 'completed-recognition-development' and report['contractSha256'] == registration['sha256']
            and report['protectedTestOpened'] is False and report['productionPromotionAllowed'] is False, 'Final report identity differs')
    require([row['seed'] for row in report['results']] == contract['seeds'], 'Seed inventory differs')
    require({path.name for path in (study/'fits').iterdir() if path.is_dir()} == {str(seed) for seed in contract['seeds']}, 'Fit seed inventory differs')
    data, config = load_inputs(contract)
    before_threads = torch.get_num_threads()
    results = []
    try:
        torch.set_num_threads(1)
        for seed in contract['seeds']:
            print(f'AUDIT {config.family}/{config.head} seed={seed}', flush=True)
            results.append(audit_seed(study, data, registration, config, seed, tensor_helper, interval_helper))
            saved = next(row for row in report['results'] if row['seed'] == seed)
            compare(saved, read(study/f'result-{seed}.json'), 'report/result binding')
    finally:
        torch.set_num_threads(before_threads)
    return {'kind': 'independent-recognition-audit-v1', 'passed': True,
        'study': str(study), 'registration': identity(study/'preregistration.json'),
        'report': identity(study/'report.json'), 'contractSha256': registration['sha256'],
        'auditor': identity(Path(__file__)), 'helpers': HELPERS, 'engineering': engineering,
        'counts': {'seeds': len(results), 'physicalFits': 10*len(results), 'logicalInnerViews': 12*len(results)},
        'cpuReplayScope': 'first/middle/last real-context core chunk per held recording at every checkpoint; not full replay',
        'cpuReplayTolerance': {'rtol': REPLAY_RTOL, 'atol': REPLAY_ATOL},
        'inputScope': 'hash-bound frozen feature/label manifests and caches; raw video bytes are not rehashed',
        'results': results, 'protectedTestOpened': False, 'fittingPerformed': False,
        'gpuUsed': False, 'finishedAt': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Audit output already exists; refusing overwrite')
    result = audit(args.study.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'output': str(args.output), 'counts': result['counts']}))


if __name__ == '__main__':
    main()
