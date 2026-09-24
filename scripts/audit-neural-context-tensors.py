#!/usr/bin/env python3
"""Independent, final-only tensor audit for an explicitly registered context study.

No fitting, GPU work, automatic cohort selection or default dataset paths. This
script must not be run on real data until the optional experiment is authorized
and completed. Shared owners are physical fits; restricted views are references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LOW_LEVEL = 'audit-neural-short-boost-tensors.py'
LOW_LEVEL_SHA = 'da773784aea271f1f19c67f02fa6246c6dc5557d628ef53334ada892f24df8b4'
TIERS = ('exact', 'draft', 'coverage')
KINDS = ('tcn', 'dino_tcn')
EXTRA_CODE = {'short_context_temporal_model.py', 'neural_context_fit.py', 'neural_context_development.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {'path': str(path), 'sha256': digest(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified(reference):
    require(digest(reference['path']) == reference['sha256'], 'Changed bound file: '+reference['path'])
    return read(reference['path'])


def low_level():
    path = REPO/'scripts'/LOW_LEVEL
    require(digest(path) == LOW_LEVEL_SHA, 'Independent low-level verifier changed')
    spec = importlib.util.spec_from_file_location('context_independent_tensor_primitives', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner_key(excluded):
    require(len(excluded) == len(set(excluded)) == 2, 'Owner needs two distinct exclusions')
    return 'exclude-'+canonical(sorted(excluded))[:20]


def training_rows(manifest, cohort, excluded):
    require(cohort in ('exact', 'draft', 'reviewed_export'), 'Unknown cohort')
    active = TIERS[:{'exact': 1, 'draft': 2, 'reviewed_export': 3}[cohort]]
    return {tier: [r for r in manifest[tier+'Rows'] if r['sourceGroup'] not in excluded] for tier in active}


def membership(manifest, cohort, excluded):
    rows = training_rows(manifest, cohort, set(excluded))
    valid = [r for r in manifest['exactRows'] if r['sourceGroup'] in excluded]
    require(rows['exact'] and valid, 'Empty fitting or held scope')
    return {'trainIds': [r['id'] for r in rows['exact']],
            'auxiliaryIds': {tier: [r['id'] for r in records] for tier, records in rows.items() if tier != 'exact'},
            'trainGroups': sorted({r['sourceGroup'] for r in rows['exact']}),
            'auxiliaryGroups': {tier: sorted({r['sourceGroup'] for r in records}) for tier, records in rows.items() if tier != 'exact'},
            'validationIds': [r['id'] for r in valid], 'validationGroups': sorted(set(excluded))}


def independent_graph(contract, contract_hash, manifest, kind, seed):
    """Derive owners/views without calling runner owner_plan or fold_graph."""
    groups = contract['groups']
    require(len(groups) == len(set(groups)) == 4 and groups == sorted(groups), 'Four ordered distinct exact groups required')
    require(kind in KINDS and seed in contract['seeds'], 'Unregistered cell')
    owners, views = {}, []
    for excluded in combinations(groups, 2):
        members = membership(manifest, contract['cohort'], excluded)
        rows = training_rows(manifest, contract['cohort'], set(excluded))
        training = {'contractSha256': contract_hash, 'manifestSha256': contract['manifestSha256'],
                    'cohort': contract['cohort'], 'kind': kind, 'contextProfile': 'short',
                    'lossArm': contract['lossArm'], 'seed': seed, 'epochs': contract['checkpointEpochs'],
                    'model': contract['models'][kind],
                    **{k: members[k] for k in ('trainIds', 'auxiliaryIds', 'trainGroups', 'auxiliaryGroups')},
                    'orderedTrainingRowsSha256': canonical(rows)}
        key = owner_key(excluded)
        owners[key] = {'ownerId': key, 'excludedGroups': list(excluded), 'training': training,
                       'trainingKeySha256': canonical(training), 'validationIds': members['validationIds']}
    for outer_index, outer in enumerate(groups):
        for inner_index, held in enumerate(g for g in groups if g != outer):
            views.append({'outerIndex': outer_index, 'outerHeldSourceGroup': outer,
                          'innerIndex': inner_index, 'innerValidationGroup': held,
                          'ownerId': owner_key([outer, held]),
                          'validationIds': [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] == held]})
    require(len(owners) == 6 and len(views) == 12
            and Counter(v['ownerId'] for v in views) == Counter({key: 2 for key in owners}), 'Bad owner/view graph')
    return owners, views


def validate_membership(meta, expected, by_id):
    for key, value in expected.items():
        require(meta[key] == value, 'Fit membership differs: '+key)
    train = meta['trainIds']+[rid for ids in meta['auxiliaryIds'].values() for rid in ids]
    held = meta['validationIds']
    require(len(train) == len(set(train)) and len(held) == len(set(held))
            and not set(train) & set(held), 'Duplicate or leaking record membership')
    require(not {by_id[r]['sourceGroup'] for r in train} & {by_id[r]['sourceGroup'] for r in held}, 'Source-group leakage')
    require(meta['scalerTrainIds'] == meta['trainIds'], 'Scaler contains auxiliary/held data')


def validate_logical_reference(actual, view, plan, folder, meta):
    require(view['ownerId'] == plan['ownerId']
            and sorted([view['outerHeldSourceGroup'], view['innerValidationGroup']]) == plan['excludedGroups'], 'Wrong logical owner')
    expected = {**view, 'ownerPath': str(folder), 'ownerCompletedSha256': digest(folder/'completed.json'),
                'trainingKeySha256': plan['trainingKeySha256'], 'ownerValidationIds': plan['validationIds'],
                'trainingIdentitySha256': meta['trainingIdentitySha256'], 'checkpointArtifacts': meta['artifacts']}
    require(actual == expected, 'Logical view ownership, restriction or checkpoint binding differs')
    require(set(view['validationIds']) <= set(plan['validationIds']), 'Logical view escapes owner')


def validate_context_metadata(metadata, context='short'):
    require(context in ('original', 'short'), 'Unknown context profile')
    halo, dilations = (16, [1, 1, 2, 2, 2]) if context == 'short' else (62, [1, 2, 4, 8, 16])
    expected = {'contextProfile': context, 'dilations': dilations, 'kernelSize': 5,
                'haloTicks': halo, 'receptiveFieldTicks': 2*halo+1, 'temporalCenterSpanSeconds': 2*halo/4,
                'originalPairedTrainingHaloTicks': 62, 'originalPairedTrainingChunkTicks': 252}
    require(all(metadata.get(k) == v for k, v in expected.items()), 'Short context/RF/training-chunk contract differs')


def verify_context_bindings(contract, preflight):
    from analysis.short_context_temporal_model import model_metadata
    models = {context: {kind: model_metadata(kind, context=context) for kind in KINDS}
              for context in ('original', 'short')}
    for context, kinds in models.items():
        for metadata in kinds.values():
            validate_context_metadata(metadata, context)
    require(contract['contextModels'] == models, 'Both context model declarations must match registered factories')
    protocol = contract['protocolSnapshot']
    require(digest(protocol['path']) == protocol['sha256'], 'Prospective protocol snapshot changed')
    reference = contract['referenceStudy']
    require(preflight['referenceReport'] == {'path': str(Path(reference['path'])/'report.json'), 'sha256': reference['reportSha256']}
            and preflight['referenceAudits'] == reference['independentAudits'], 'Preflight reference report/audit identities differ')
    return protocol


def architecture_contract(low):
    """Allocate CPU templates only; do not call a fitter or consume caller RNG."""
    import torch
    from analysis.short_context_temporal_model import model_for, model_metadata
    shapes = low.parameter_contract()
    with torch.random.fork_rng(devices=[]):
        for kind in KINDS:
            meta = model_metadata(kind, context='short')
            validate_context_metadata(meta)
            model = model_for(kind, context='short').cpu()
            temporal = model.temporal if kind == 'dino_tcn' else model
            require([b.depthwise.dilation[0] for b in temporal.blocks] == [1, 1, 2, 2, 2]
                    and [b.depthwise.padding[0] for b in temporal.blocks] == [2, 2, 4, 4, 4]
                    and all(b.depthwise.kernel_size == (5,) for b in temporal.blocks)
                    and model.halo_ticks == 16 and model.receptive_field_ticks == 33, 'Actual temporal convolution profile differs')
            require({'model::'+n: list(v.shape) for n, v in model.state_dict().items()} == shapes[kind]['stateShapes']
                    and sum(p.numel() for p in model.parameters()) == shapes[kind]['parameterCount'], 'Context changed tensor shapes/count')
            shapes[kind]['metadata'] = meta
    return shapes


def chunk_geometry(ids, inputs, supervision):
    """Reconstruct sampler inputs using mask geometry, no training chunk code."""
    groups = Counter(inputs[rid]['group'] for rid in ids)
    output = []
    for rid in ids:
        valid, mask = inputs[rid]['valid'], supervision[rid]['mask']
        edges = np.diff(np.r_[False, valid, False].astype(np.int8))
        one = []
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            for core in range(int(start), int(end), 128):
                finish = min(int(end), core+128)
                if np.any(mask[core:finish]):
                    one.append({'id': rid, 'left': max(int(start), core-62), 'right': min(int(end), finish+62),
                                'coreStart': core, 'coreEnd': finish})
        for entry in one:
            output.append({**entry, 'samplingWeight': 1./(groups[inputs[rid]['group']]*len(one))})
    return output


def independent_exposure(pools, seed, epochs):
    """Replay integer sampling and hashes only; no tensors, dropout or optimizer."""
    pools = {tier: rows for tier, rows in pools.items() if rows}
    require(pools.get('exact'), 'No supervised exact chunks')
    probability = {}
    for tier, chunks in pools.items():
        p = np.asarray([x['samplingWeight'] for x in chunks], np.float64)
        probability[tier] = p/p.sum()
    rngs = {tier: np.random.default_rng(np.random.SeedSequence([seed, i])) for i, tier in enumerate(TIERS)}
    hashes = {tier: hashlib.sha256() for tier in pools}
    history, steps = [], 0
    for epoch in range(1, epochs+1):
        draw = rngs['exact'].choice(len(pools['exact']), size=len(pools['exact']), replace=True, p=probability['exact'])
        buckets = {}
        for index in draw:
            item = pools['exact'][index]
            buckets.setdefault(item['right']-item['left'], []).append(int(index))
        for indexes in buckets.values():
            for start in range(0, len(indexes), 16):
                exact = indexes[start:start+16]
                for tier in TIERS:
                    if tier not in pools:
                        continue
                    chosen = exact if tier == 'exact' else rngs[tier].choice(
                        len(pools[tier]), len(exact), replace=True, p=probability[tier]).tolist()
                    hashes[tier].update(json.dumps([epoch, steps, chosen], separators=(',', ':')).encode())
                steps += 1
        history.append({'epoch': epoch, 'optimizerSteps': steps,
                        'exposureSha256': {tier: h.hexdigest() for tier, h in hashes.items()}})
    return history


def validate_history(meta, expected):
    require(len(meta['history']) == len(expected) > 0, 'Epoch history length differs')
    for actual, row in zip(meta['history'], expected):
        require(all(actual[k] == v for k, v in row.items()) and np.isfinite(actual['loss']), 'Independent sampling/step replay differs')
    require(meta['optimizerSteps'] == expected[-1]['optimizerSteps']
            and meta['exposureSha256'] == expected[-1]['exposureSha256'], 'Final sampling exposure differs')


def paired_prefix(left, right):
    for key in ('trainIds', 'auxiliaryIds', 'trainGroups', 'auxiliaryGroups', 'scalerTrainIds', 'positiveWeight', 'supervisedCounts'):
        require(left[key] == right[key], 'Paired training scalar/membership differs: '+key)
    count = min(len(left['history']), len(right['history']))
    require(count > 0, 'No paired epoch prefix')
    for a, b in zip(left['history'][:count], right['history'][:count]):
        require(all(a[k] == b[k] for k in ('epoch', 'optimizerSteps', 'exposureSha256')), 'Paired sampling prefix differs')
    return count


def contains_binding(value, bound):
    if isinstance(value, dict):
        return (value.get('path') == bound['path'] and value.get('sha256') == bound['sha256']) or any(contains_binding(v, bound) for v in value.values())
    return isinstance(value, list) and any(contains_binding(v, bound) for v in value)


def reference_binding(reference):
    root = Path(reference['path'])
    files = {name: {'path': str(root/(name+'.json')), 'sha256': reference[field]} for name, field in
             (('preregistration', 'preregistrationFileSha256'), ('report', 'reportSha256'), ('summary', 'summarySha256'))}
    reg, report, summary = (verified(files[name]) for name in ('preregistration', 'report', 'summary'))
    require(reg['sha256'] == canonical(reg['contract']) == reference['contractSha256'], 'Reference registration differs')
    require(report['contractSha256'] == summary['contractSha256'] == reg['sha256']
            and report['status'] == 'completed-short-boost-transfer-development'
            and summary['status'] == 'completed-short-boost-transfer-audit'
            and report['protectedTestOpened'] is False and report['productionPromotionAllowed'] is False
            and contains_binding(summary, files['report']), 'Reference study/summary incomplete or unbound')
    expected = {(c, k, a, s) for c in ('exact', 'draft', 'reviewed_export') for k in KINDS
                for a in ('baseline', 'global_control', 'short_boost') for s in reg['contract']['seeds']}
    require(len(report['results']) == 54 and {(r['cohort'], r['kind'], r['lossArm'], r['seed']) for r in report['results']} == expected,
            'Reference factorial grid differs')
    kinds = set()
    for bound in reference['independentAudits']:
        item = verified(bound)
        require(item['passed'] is True and item['contractSha256'] == reg['sha256']
                and contains_binding(item, files['report']), 'Reference independent audit failed or unbound')
        kinds.add(item['kind'])
    require({'neural-short-boost-independent-tensor-audit-v1', 'independent-short-boost-transfer-interval-audit-v1'} <= kinds,
            'Missing independent reference audits')
    return reg, report, [*files.values(), *reference['independentAudits']]


def reference_copy(result, original, reference, contract_hash):
    origin = {'type': 'reused-context-reference', 'studyPath': reference['path'], 'reportSha256': reference['reportSha256'],
              'referenceContractSha256': reference['contractSha256'], 'sourceOrigin': original['origin'],
              'sourceResult': {k: original[k] for k in ('cohort', 'kind', 'lossArm', 'seed')}}
    require(result == {**original, 'context': 'original', 'contractSha256': contract_hash, 'origin': origin},
            'Reference payload or nested provenance changed')


def verify_loader(contract, manifest, inputs, supervision, diagnostics):
    """Compare both real loader representations to independently rebuilt inputs."""
    from analysis.neural_short_boost_transfer import load_data
    from analysis.neural_short_boost_weighting import live_event_weights
    dino_entries = {r['recordingId']: r for r in read(contract['dinoManifest']['path'])['records']}
    checked = []
    for kind in KINDS:
        loaded = load_data(Path(contract['manifest']['path']), Path(contract['dinoManifest']['path']), with_dino=kind == 'dino_tcn')
        for tier in TIERS:
            require([r.example.id for r in loaded[tier]] == [r['id'] for r in manifest[tier+'Rows']], 'Loader row order differs')
            for row in loaded[tier]:
                e, item = row.example, inputs[row.example.id]
                rebuilt = supervision[e.id]
                require(row.tier == tier and e.group == item['group']
                        and np.array_equal(e.times, item['times']) and np.array_equal(e.valid, rebuilt['valid'])
                        and np.array_equal(e.targets, rebuilt['targets']) and np.array_equal(row.mask, rebuilt['mask'])
                        and e.values.dtype == np.float32 and np.isfinite(e.values).all()
                        and np.array_equal(e.values[:, :104], item['normalized']), 'Loaded targets/masks/AV inputs differ')
                require(e.values.shape == (len(e.times), 3944 if kind == 'dino_tcn' else 104), 'Representation input shape differs')
                multipliers, actual_diagnostic = live_event_weights(row, contract['lossArm'])
                require(multipliers.shape == (len(e.times),) and multipliers.dtype == np.float32
                        and np.isfinite(multipliers).all() and actual_diagnostic == diagnostics[e.id]
                        and hashlib.sha256(multipliers.astype('<f4').tobytes()).hexdigest() == diagnostics[e.id]['liveMultiplierSha256'],
                        'Actual live loss multipliers differ from independent reconstruction')
                if kind == 'dino_tcn':
                    with np.load(dino_entries[e.id]['dinoPath'], allow_pickle=False) as cache:
                        times, tokens = cache['timestamps'].astype(np.float64), cache['tokens']
                        right = np.minimum(np.searchsorted(times, e.times), len(times)-1)
                        left = np.maximum(0, right-1)
                        nearest = np.where(np.abs(times[left]-e.times) <= np.abs(times[right]-e.times), left, right)
                        require(np.array_equal(e.values[:, 104:], tokens[nearest].reshape(len(e.times), 3840).astype(np.float32)),
                                'Loaded DINO token identity differs')
                checked.append({'kind': kind, 'id': e.id, 'targetSha256': hashlib.sha256(e.targets.tobytes()).hexdigest(),
                                'maskSha256': hashlib.sha256(row.mask.tobytes()).hexdigest(), 'shape': list(e.values.shape),
                                'liveMultiplierSha256': diagnostics[e.id]['liveMultiplierSha256']})
        del loaded
    return checked


def audit(registration_path):
    study = registration_path.parent
    require((study/'report.json').is_file(), 'Final audit requires completed context report; no partial/automatic run')
    low = low_level()
    verifier = low.independent_verifier()
    reg, report = read(registration_path), read(study/'report.json')
    c = reg['contract']
    require(reg['sha256'] == canonical(c) and c['experiment'] == 'short-context-development-v1', 'Bad context registration')
    require(c['contexts'] == ['original', 'short'] and c['kinds'] == list(KINDS)
            and c['cohorts'] == [c['cohort']] and c['cohort'] in ('exact', 'draft', 'reviewed_export')
            and c['lossArm'] in ('baseline', 'global_control', 'short_boost')
            and len(c['seeds']) == len(set(c['seeds'])) == 3 and c['checkpointEpochs'] == [5, 15, 30, 60], 'Unsupported scope')
    require(c['fitReuse'] == {'unorderedInnerExclusions': True, 'physicalInnerOwnersPerCell': 6,
            'logicalInnerViewsPerCell': 12, 'outerRefitsPerCell': 4, 'trainingHaloTicks': 62, 'trainingCoreTicks': 128}, 'Fit reuse contract differs')
    old_reg, old_report, locked = reference_binding(c['referenceStudy'])
    old = old_reg['contract']
    reference_fit_hashes = {}
    for bound in c['referenceStudy']['independentAudits']:
        previous_audit = verified(bound)
        if previous_audit['kind'] == 'neural-short-boost-independent-tensor-audit-v1':
            reference_fit_hashes = {Path(f['completed']['path']).resolve(): f['completed']['sha256']
                                    for f in previous_audit['fits']}
    require(len(reference_fit_hashes) == 864, 'Reference tensor audit fit inventory differs')
    for key in ('seeds', 'groups', 'checkpointEpochs', 'decoderCandidates', 'selection', 'primaryMetric',
                'targetPaddingSeconds', 'joinGapSeconds', 'paddingSweep', 'training', 'environment',
                'manifestSha256', 'dinoManifest', 'screen', 'retentionRecoveryScreen', 'evaluationPopulation'):
        require(c[key] == old[key], 'Reference comparison changed '+key)
    require(len(old['code']) == 16 and set(c['code']) == set(old['code']) | EXTRA_CODE
            and all(c['code'][k] == v for k, v in old['code'].items()), 'Frozen code inheritance differs')
    for name, sha in c['code'].items():
        require(Path(name).name == name and digest(REPO/'analysis'/name) == sha, 'Registered source changed')
    manifest = verified(c['manifest'])
    require(c['manifest']['sha256'] == c['manifestSha256'] == old['manifestSha256'], 'Dataset changed from reference')
    exact = verified(manifest['exactManifest'])
    require(exact['recordings'] == manifest['exactRows'], 'Exact evaluation inputs changed')
    preflight = verified(c['preflight'])
    require(preflight['passed'] is True and preflight['ownerReuseQualified'] is True
            and preflight['originalProfileReplayQualified'] is True and preflight['code'] == c['code']
            and preflight['manifest'] == c['manifest'] and preflight['dinoManifest'] == c['dinoManifest']
            and preflight['referenceContractSha256'] == old_reg['sha256']
            and preflight['cohort'] == c['cohort'] and preflight['lossArm'] == c['lossArm'], 'Missing/mismatched preflight qualification')
    protocol_binding = verify_context_bindings(c, preflight)
    require(report['status'] == 'completed-context-development' and report['contractSha256'] == reg['sha256']
            and report['manifestSha256'] == c['manifestSha256'] and report['records'] == 8
            and report['sourceGroups'] == c['groups'] and report['protectedTestOpened'] is False
            and report['productionPromotionAllowed'] is False, 'Report scope/completion differs')
    tiers = {tier: manifest[tier+'Rows'] for tier in TIERS}
    by_id = {r['id']: r for rows in tiers.values() for r in rows}
    require([len(tiers[t]) for t in TIERS] == [8, 3, 7] and len(by_id) == 18
            and sorted({r['sourceGroup'] for r in tiers['exact']}) == c['groups']
            and all(r['environment'] in ('grass', 'indoor') and r['consent']['train'] is True
                    and r['sourceGroup'] not in manifest['protectedSourceGroups'] for r in by_id.values()), 'Forbidden or changed dataset scope')
    locked += [identity(registration_path), identity(study/'report.json'), c['manifest'], c['dinoManifest'],
               manifest['exactManifest'], c['preflight'], protocol_binding,
               identity(Path(__file__)), identity(REPO/'scripts'/LOW_LEVEL)]
    inputs = verifier.load_cache_inputs(manifest)
    supervision, diagnostics = {}, {}
    for tier, rows in tiers.items():
        for row in rows:
            rid = row['id']
            with np.load(inputs[rid]['cache']['path'], allow_pickle=False) as cache:
                duration = float(json.loads(str(cache['metadata_json'].item()))['duration'])
            supervision[rid] = low.independent_supervision(row, tier, inputs[rid]['times'], duration)
            require(np.array_equal(supervision[rid]['valid'], inputs[rid]['valid']), 'Independent valid timeline differs')
            diagnostics[rid] = low.independent_weights(row, tier, inputs[rid]['times'], supervision[rid], c['lossArm'])
    # The previous contract binds the same repaired input manifest and its
    # originalManifest/protocolAmendment/featureRevision extraction provenance.
    dino_audit = low.verify_dino_inputs(old, Path(c['manifest']['path']), inputs)
    loader_audit = verify_loader(c, manifest, inputs, supervision, diagnostics)
    architecture = architecture_contract(low)
    require(c['models'] == {k: architecture[k]['metadata'] for k in KINDS}, 'Registered context models differ')
    results = report['results']
    key = lambda r: (r['kind'], r['context'], r['seed'])
    expected = {(k, p, s) for k in KINDS for p in ('original', 'short') for s in c['seeds']}
    require(len(results) == 12 and {key(r) for r in results} == expected
            and all(r['cohort'] == c['cohort'] and r['lossArm'] == c['lossArm']
                    and r['contractSha256'] == reg['sha256'] for r in results), 'Context result grid differs')
    selected_old = {(r['kind'], r['seed']): r for r in old_report['results'] if r['cohort'] == c['cohort'] and r['lossArm'] == c['lossArm']}
    require(len(selected_old) == 6, 'Reference cell count differs')
    reference_roots, original_wrappers = {}, []
    for r in results:
        path = study/f"result-{c['cohort']}-{r['kind']}-{c['lossArm']}-{r['context']}-{r['seed']}.json"
        require(read(path) == r, 'Result file differs from final report')
        if r['context'] == 'original':
            source = selected_old[(r['kind'], r['seed'])]
            reference_copy(r, source, c['referenceStudy'], reg['sha256'])
            root, nested = low.validate_origin(source, Path(c['referenceStudy']['path']), old['referenceStudy'])
            reference_roots[(r['kind'], r['seed'])] = root
            original_wrappers.append({'result': identity(path), 'origin': r['origin'], 'referenceFitRoot': str(root), 'nestedHistoricalReuse': nested})
    scaler_cache, exposure_cache, fits, metas, references = {}, {}, [], {}, []
    counts = Counter()
    expected_fit_paths, expected_view_paths, expected_plan_paths = set(), set(), set()
    reference_meta_cache = {}

    def check_fit(folder, members, kind, seed, epochs):
        path = folder/'completed.json'
        bound, meta = identity(path), read(path)
        validate_membership(meta, members, by_id)
        require(meta['contractSha256'] == reg['sha256'] and meta['kind'] == meta['architecture'] == kind
                and meta['seed'] == seed and meta['contextProfile'] == 'short' and meta['lossArm'] == c['lossArm']
                and meta['epochs'] == epochs and meta['model'] == c['models'][kind]
                and meta['parameters'] == architecture[kind]['parameterCount'], 'Fit recipe/context differs')
        ids = {'exact': members['trainIds'], **members['auxiliaryIds']}
        expected_diag = {'mode': c['lossArm'], 'rows': {tier: [diagnostics[rid] for rid in names] for tier, names in ids.items()}}
        require(meta['liveLossWeighting'] == expected_diag, 'Live loss multiplier diagnostics differ')
        training = {k: meta[k] for k in ('contractSha256', 'kind', 'seed', 'lossArm', 'contextProfile', 'model',
                                        'trainIds', 'auxiliaryIds', 'trainGroups', 'auxiliaryGroups', 'liveLossWeighting')}
        require(meta['trainingIdentity'] == training and meta['trainingIdentitySha256'] == canonical(training), 'Role-independent training identity differs')
        expected_counts = {tier: {field: (np.sum([supervision[rid]['counts'][field] for rid in names], axis=0).tolist() if names else [0]*4)
                                 for field in ('valid', 'positiveMass')} for tier, names in ids.items()}
        require(set(meta['supervisedCounts']) == set(expected_counts), 'Supervision tiers differ')
        for tier, count in expected_counts.items():
            require(all(np.allclose(meta['supervisedCounts'][tier][field], value, rtol=1e-6, atol=1e-4)
                        for field, value in count.items()), 'Independent supervision count differs')
        valid, positive = (np.asarray(expected_counts['exact'][field]) for field in ('valid', 'positiveMass'))
        pos_weight = np.minimum(20., np.sqrt((valid-positive)/np.maximum(positive, 1.)))
        require(np.allclose(meta['positiveWeight'], pos_weight, rtol=1e-7, atol=1e-7), 'Exact-only positive class weights differ')
        pool_key = canonical(ids)
        replay_key = (pool_key, seed, max(epochs))
        if replay_key not in exposure_cache:
            pools = {tier: chunk_geometry(names, inputs, supervision) for tier, names in ids.items()}
            exposure_cache[replay_key] = independent_exposure(pools, seed, max(epochs))
        validate_history(meta, exposure_cache[replay_key])
        scaler_key = tuple(members['trainIds'])
        if scaler_key not in scaler_cache:
            scaler_cache[scaler_key] = verifier.independent_scaler(members['trainIds'], inputs)
        mean, scale, ticks = scaler_cache[scaler_key]
        names = {f'{stem}-{epoch}.npz' for epoch in epochs for stem in ('weights', 'predictions')}
        require(set(meta['artifacts']) == names and {p.name for p in folder.glob('*.npz')} == names, 'Fresh artifact inventory differs')
        checkpoints = [low.audit_checkpoint(folder, epoch, meta, architecture[kind], mean, scale, inputs) for epoch in epochs]
        require(digest(path) == bound['sha256'], 'Fit metadata changed during audit')
        expected_fit_paths.add(folder.resolve())
        counts['physicalFreshFits'] += 1
        counts['freshCheckpoints'] += len(checkpoints)
        counts['freshNPZArtifacts'] += 2*len(checkpoints)
        fits.append({'completed': bound, 'kind': kind, 'seed': seed, 'members': members,
                     'scalerValidTicks': ticks, 'checkpoints': checkpoints, 'independentExposureReplayPassed': True})
        return meta

    def compare_reference(fresh, folder):
        path = folder/'completed.json'
        if path not in reference_meta_cache:
            require(reference_fit_hashes.get(path.resolve()) == digest(path), 'Reference completion differs from bound tensor audit')
            meta = read(path)
            # Reference tensors were independently audited by the bound prior
            # study. Recheck all file hashes now, without counting them as fresh.
            require(set(meta['artifacts']) == {f'{stem}-{epoch}.npz' for epoch in meta['epochs'] for stem in ('weights', 'predictions')},
                    'Reference artifact inventory differs')
            require(all(digest(folder/name) == sha for name, sha in meta['artifacts'].items()), 'Reference checkpoint changed')
            reference_meta_cache[path] = meta
        meta = reference_meta_cache[path]
        epochs = paired_prefix(fresh, meta)
        require(meta['kind'] == fresh['kind'] and meta['seed'] == fresh['seed'], 'Reference fit kind/seed differs')
        for epoch in meta['epochs']:
            # Scalar arrays, not model weights, must be identical under context.
            with np.load(folder/f'weights-{epoch}.npz', allow_pickle=False) as cache:
                mean, scale, _ = scaler_cache[tuple(fresh['trainIds'])]
                require(np.array_equal(cache['mean'], mean) and np.array_equal(cache['scale'], scale), 'Reference paired scaler differs')
        references.append({'completed': identity(path), 'pairedEpochs': epochs})
        counts['referencePrefixPairs'] += 1
        counts['referenceEpochPrefixes'] += epochs

    for r in results:
        if r['context'] != 'short':
            continue
        kind, seed = r['kind'], r['seed']
        root = study/'fits'/c['cohort']/kind/c['lossArm']/'short'/str(seed)
        require(r['origin'] == {'type': 'trained-context', 'fitRoot': str(root), 'innerOwnership': 'six-unordered-exclusions'}, 'Fresh fit root/provenance differs')
        owners, views = independent_graph(c, reg['sha256'], manifest, kind, seed)
        # Runner graph is used only as a comparison, never to derive audit scope.
        from analysis.neural_context_development import fold_graph
        require(fold_graph(c, reg['sha256'], manifest, kind, seed) == (owners, views), 'Runner graph differs from independent graph')
        for owner, plan in owners.items():
            plan_path = root/'owner-plans'/f'{owner}.json'
            require(read(plan_path) == plan, 'Owner plan or ordered training rows changed')
            expected_plan_paths.add(plan_path.resolve())
            members = membership(manifest, c['cohort'], plan['excludedGroups'])
            meta = check_fit(root/'inner-owners'/owner, members, kind, seed, c['checkpointEpochs'])
            metas[(kind, seed, owner)] = meta
            counts['physicalInnerOwners'] += 1
        bound_views = r['logicalInnerViews']
        require(len(bound_views) == 12, 'Logical view count differs')
        for view, bound in zip(views, bound_views):
            path = root/'logical-views'/f"outer-{view['outerIndex']}"/f"inner-{view['innerIndex']}.json"
            require(bound['path'] == str(path), 'Noncanonical/reordered logical view')
            actual = verified(bound)
            plan, meta = owners[view['ownerId']], metas[(kind, seed, view['ownerId'])]
            validate_logical_reference(actual, view, plan, root/'inner-owners'/view['ownerId'], meta)
            expected_view_paths.add(path.resolve())
            # Each view may expose ONLY its inner group. Owner NPZ inventory
            # legitimately includes both excluded groups; selectors may not.
            require(all(by_id[rid]['sourceGroup'] == view['innerValidationGroup'] for rid in actual['validationIds']),
                    'Logical validation exposes outer group')
            reference_root = reference_roots[(kind, seed)]
            compare_reference(meta, reference_root/f"outer-{view['outerIndex']}"/f"inner-{view['innerIndex']}")
            counts['logicalInnerViews'] += 1
        selections = {s['heldSourceGroup']: s for s in r['selections']}
        require(len(r['selections']) == len(selections) == 4 and set(selections) == set(c['groups']), 'Outer selections differ')
        for index, group in enumerate(c['groups']):
            selection = selections[group]
            require(selection['epoch'] in c['checkpointEpochs'] and selection['decoder'] in c['decoderCandidates'], 'Outer selection outside grid')
            members = membership(manifest, c['cohort'], [group])
            meta = check_fit(root/f'outer-{index}'/'refit', members, kind, seed, [selection['epoch']])
            metas[(kind, seed, 'outer-'+str(index))] = meta
            compare_reference(meta, reference_roots[(kind, seed)]/f'outer-{index}'/'refit')
            counts['outerRefits'] += 1
    for seed in c['seeds']:
        slots = [key[2] for key in metas if key[:2] == ('tcn', seed)]
        require(len(slots) == 10, 'Missing cross-representation physical fit')
        for slot in slots:
            counts['crossRepresentationEpochPrefixes'] += paired_prefix(metas[('tcn', seed, slot)], metas[('dino_tcn', seed, slot)])
            counts['crossRepresentationFitPairs'] += 1
    counts['logicalFreshFits'] = counts['logicalInnerViews']+counts['outerRefits']
    counts['referenceResultCells'] = len(original_wrappers)
    expected_counts = {'physicalFreshFits': 60, 'physicalInnerOwners': 36, 'outerRefits': 24, 'logicalInnerViews': 72,
                       'logicalFreshFits': 96, 'freshCheckpoints': 168, 'freshNPZArtifacts': 336,
                       'referenceResultCells': 6, 'referencePrefixPairs': 96, 'crossRepresentationFitPairs': 30}
    require(all(counts[k] == v for k, v in expected_counts.items()), 'Final physical/logical inventory differs')
    require({p.parent.resolve() for p in (study/'fits').rglob('completed.json')} == expected_fit_paths
            and len(list((study/'fits').rglob('*.npz'))) == 336
            and {p.resolve() for p in (study/'fits').glob('**/logical-views/**/*.json')} == expected_view_paths
            and {p.resolve() for p in (study/'fits').glob('**/owner-plans/*.json')} == expected_plan_paths, 'Unexpected physical fits/artifacts/views/plans')
    require(report['execution']['physicalFreshFits'] == 60 and report['execution']['logicalFreshFits'] == 96, 'Report counts differ')
    for name, sha in c['code'].items():
        require(digest(REPO/'analysis'/name) == sha, 'Code changed during audit')
    for bound in locked:
        require(digest(bound['path']) == bound['sha256'], 'Input changed during audit')
    return {'kind': 'independent-context-tensor-audit-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'passed': True, 'contractSha256': reg['sha256'], 'report': identity(study/'report.json'),
            'registration': identity(registration_path), 'manifest': c['manifest'], 'dinoManifest': c['dinoManifest'],
            'protocolSnapshot': protocol_binding, 'contextModels': c['contextModels'],
            'counts': dict(counts), 'expected': expected_counts, 'code': c['code'], 'architectures': architecture,
            'independentScalerPopulations': len(scaler_cache), 'scalerExactlyEqualCheckpoints': counts['freshCheckpoints'],
            'loaderTargetsMasksAndRepresentations': loader_audit, 'dinoInputs': dino_audit,
            'referenceBindings': locked[:len(c['referenceStudy']['independentAudits'])+3],
            'referenceResultWrappers': original_wrappers, 'referencePrefixChecks': references,
            'fits': fits, 'auditScript': identity(Path(__file__)), 'independentPrimitives': identity(REPO/'scripts'/LOW_LEVEL),
            'limits': ['CPU-only; no fitting, source-video reads or protected labels.',
                       'Stored prediction index/shape/finite/ignored zeros are checked, not full forward replay of every checkpoint.',
                       'Reference result payloads and nested origins are preserved; reference fits are not fabricated or counted as fresh.',
                       'Independent integer sampler replay checks exposure/steps. Dropout and optimizer computations rely on frozen code and qualified preflight, not a second training run.',
                       'Context profile is metadata plus verified factory convolution attributes; state_dict alone cannot identify dilation.',
                       'Metric, full decoder-grid selection and selected-probability-to-interval replay belong to the separate context summary.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.registration.parent/'tensor-audit-v1.json'
    require(not output.exists(), 'Refuse overwriting an immutable audit')
    result = audit(args.registration)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'counts': result['counts'], 'artifact': identity(output)}, indent=2))


if __name__ == '__main__':
    main()
