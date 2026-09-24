#!/usr/bin/env python3
"""Freeze original rally identities and four held-fold heads for review research.

No fitting, decoder changes, outcome evaluation, protected-test access or use of
gold for candidate generation. Never overwrites any existing input artifact.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(private_value('private-reference-0057'))
PRIOR = ROOT/'2026-09-19-perfect-human-review'/'probability-input.json'
PRIOR_SHA = '17c24623bb851d6abf90ed163386f8355bbb41c4a3e7537e8a1d7835aec75bc7'
DESTINATION = ROOT/'2026-09-19-rally-review-proposals'
MODELS = ('compact_boost', 'dino_global', 'dino_boost')
SEEDS = (3407, 1729, 20260918)
HEADS = ('live', 'serve', 'end', 'keep')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(ref):
    require(digest(ref['path']) == ref['sha256'], 'Changed bound file: '+ref['path'])
    if 'sizeBytes' in ref:
        require(Path(ref['path']).stat().st_size == ref['sizeBytes'], 'Bound size differs: '+ref['path'])


def verified(ref):
    verify(ref)
    return read(ref['path'])


def interval_identity(rows, tags=False):
    return [(float(r['start']), float(r['end']), tuple(r.get('tags', [])) if tags else ()) for r in rows]


def events(values, prefix, duration):
    """Retain every source event separately, including touching/overlapping ones."""
    out = []
    for index, row in enumerate(values):
        start, end = float(row['start']), float(row['end'])
        require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration+1e-9,
                'Invalid original event bounds')
        require(index == 0 or start >= out[-1]['start'], 'Original event ordering changed')
        out.append({**row, 'id': row.get('id', f'{prefix}:{index+1}'),
                    'sourceIndex': index, 'start': start, 'end': end})
    require(len({r['id'] for r in out}) == len(out), 'Duplicate original event identifier')
    return out


def production_events(row):
    """Recover actual current-default IDs; component boundaries stay separate."""
    union = row['productReplay']['unionWithConfidence']
    by_bounds = {(r['start'], r['end']): r for r in union}
    require(len(by_bounds) == len(union), 'Ambiguous production union identity')
    core = row['cores']['productionDefault']
    require(core == row['currentDefault'], 'Production default aliases differ')
    require(all((r['start'], r['end']) in by_bounds for r in core), 'Default event absent from actual app union')
    kept = [by_bounds[(r['start'], r['end'])] for r in core]
    output = events(kept, 'unused', row.get('featureMetadataDurationSeconds', row['durationSeconds']))
    for event in output:
        event['sourceUnionIndex'] = next(i for i, r in enumerate(union) if r['id'] == event['id'])
        event['sourceComponentIds'] = {name: [r['id'] for r in values
            if r['start'] < event['end'] and event['start'] < r['end']]
            for name, values in row['productReplay']['components'].items()}
    return output


def validate_scores(scores, times):
    require(scores.dtype == np.float32 and scores.shape == (len(times), 4),
            'Scores must remain original float32 [ticks,4]')
    require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid probability values')


def validate_membership(completed, fit, rows):
    group = fit['sourceGroup']
    expected = {rid for rid, row in rows.items() if row['sourceGroup'] == group}
    require(completed['validationGroups'] == [group] and set(completed['validationIds']) == expected,
            'Selected validation source differs')
    require(group not in completed['trainGroups'] and not expected.intersection(completed['trainIds']),
            'Exact training leakage')
    require(all(group not in values for values in completed['auxiliaryGroups'].values())
            and all(not expected.intersection(values) for values in completed['auxiliaryIds'].values()),
            'Auxiliary training leakage')
    require(all(completed[key] == value for key, value in fit['membership'].items()), 'Bound membership differs')
    require(completed['model']['headNames'] == list(HEADS), 'Head order changed')
    require(completed['seed'] == fit['seed'] and completed['contractSha256'] == fit['sourceContractSha256'],
            'Fit identity differs')
    return expected


def validate_timeline(times, expected):
    require(times.dtype == np.float64 and times.ndim == 1 and len(times) == expected['tickCount'],
            'Timeline shape or dtype differs')
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Invalid timeline')
    require(hashlib.sha256(times.astype('<f8', copy=False).tobytes()).hexdigest() == expected['timelineBytesSha256'],
            'Exact timestamp bytes changed')


def prepare(output):
    json_path, npz_path = output/'rally-review-input.json', output/'probabilities.npz'
    require(not json_path.exists() and not npz_path.exists(), 'Refusing to overwrite immutable input')
    require(digest(PRIOR) == PRIOR_SHA, 'Prior normalized probability input changed')
    prior = read(PRIOR)
    registration = verified(prior['previousRegistration'])
    require(canonical(registration['contract']) == registration['sha256'] == prior['previousContractSha256'],
            'Prior registration contract differs')
    contract = registration['contract']
    neural = verified(contract['neuralInput'])
    production = verified(contract['productionInput'])
    manifest = verified(prior['exactManifest'])
    verify(prior['npz'])
    rows = {r['id']: r for r in neural['records']}
    prod_rows = {r['id']: r for r in production['recordings']}
    manifest_rows = {r['id']: r for r in manifest['exactRows']}
    prior_records = {r['id']: r for r in prior['records']}
    require(len(rows) == 8 and set(rows) == set(prod_rows) == set(manifest_rows) == set(prior_records),
            'Recording inventory differs')
    selected = [f for f in prior['sourceFits'] if f['modelId'] in MODELS]
    groups = sorted({r['sourceGroup'] for r in rows.values()})
    require(len(selected) == 36 and {(f['modelId'], f['seed'], f['sourceGroup']) for f in selected}
            == {(m, s, g) for m in MODELS for s in SEEDS for g in groups}, 'Selected fit inventory differs')
    arrays, records, entries, results, source_refs = {}, [], [], {}, []
    old_npz = np.load(prior['npz']['path'], allow_pickle=False)
    try:
        for rid in sorted(rows):
            row, prod, exact, old = rows[rid], prod_rows[rid], manifest_rows[rid], prior_records[rid]
            require(exact['environment'] in ('grass', 'indoor')
                    and row['sourceGroup'] not in manifest['protectedSourceGroups'], 'Forbidden source')
            require(row['sourceGroup'] == prod['sourceGroup'] == exact['sourceGroup'] == old['sourceGroup'],
                    'Source-group revision differs')
            for other in (prod, exact):
                require(interval_identity(row['rallies'], True) == interval_identity(other['rallies'], True)
                        and interval_identity(row.get('ignoredIntervals', [])) == interval_identity(other.get('ignoredIntervals', [])),
                        'Gold or ignored revision differs')
            require(row['durationSeconds'] == prod['featureMetadataDurationSeconds'] == old['durationSeconds']
                    and abs(row['durationSeconds']-prod['durationSeconds']) < 1e-6,
                    'Exact duration or allowed prior display rounding differs')
            times = old_npz[old['timesKey']]
            validate_timeline(times, old)
            verify(old['featureCache'])
            with np.load(old['featureCache']['path'], allow_pickle=False) as features:
                require(np.array_equal(times, features['times'].astype(np.float64)), 'Feature timeline differs')
            key = 'times::'+rid
            arrays[key] = times.copy()
            label_doc = verified(exact['labelSource'])
            require(label_doc.get('serveMarkers', []) == exact.get('serveMarkers', []), 'Serve marker source differs')
            require(interval_identity(label_doc['rallies'], True) == interval_identity(row['rallies'], True)
                    and interval_identity(label_doc.get('ignoredIntervals', [])) == interval_identity(row.get('ignoredIntervals', [])),
                    'Original label document differs from frozen gold')
            source_refs.extend([old['featureCache'], exact['labelSource']])
            records.append({**row, 'environment': exact['environment'], 'timesKey': key,
                'tickCount': len(times), 'timelineBytesSha256': old['timelineBytesSha256'],
                'featureCache': old['featureCache'], 'productionEvents': production_events(prod),
                'productionComponents': prod['productReplay']['components'],
                'productionUnionEvents': prod['productReplay']['unionWithConfidence'],
                'serveMarkers': exact.get('serveMarkers', []), 'labelSource': exact['labelSource'],
                'annotation': exact.get('annotation'), 'targetContract': exact.get('targetContract'),
                'annotationPolicy': label_doc.get('annotationPolicy')})
        for fit in selected:
            completed = verified(fit['completed'])
            expected_ids = validate_membership(completed, fit, rows)
            for field in ('probabilities', 'weights'):
                verify(fit[field])
                require(completed['artifacts'][Path(fit[field]['path']).name] == fit[field]['sha256'],
                        'Checkpoint artifact differs from completed fit')
                source_refs.append(fit[field])
            source_refs.append(fit['completed'])
            cell_key = (fit['modelId'], fit['seed'])
            if cell_key not in results:
                result = verified(fit['sourceResult'])
                require(result['seed'] == fit['seed'] and result['kind'] == completed['kind']
                        and result['lossArm'] == completed['lossArm'], 'Selected result identity differs')
                results[cell_key] = {r['id']: r for r in result['predictions']}
                source_refs.append(fit['sourceResult'])
            with np.load(fit['probabilities']['path'], allow_pickle=False) as original:
                require(set(original.files) == expected_ids, 'Checkpoint recording inventory differs')
                for rid in sorted(expected_ids):
                    scores = original[rid]
                    times = arrays['times::'+rid]
                    validate_scores(scores, times)
                    old_key = f"{fit['modelId']}::{fit['seed']}::{rid}::live"
                    require(np.array_equal(scores[:, 0], old_npz[old_key]), 'Prior live scores differ')
                    key = f"{fit['modelId']}::{fit['seed']}::{rid}::scores"
                    arrays[key] = scores.copy()
                    original_events = results[cell_key][rid]['predictions']
                    require([[r['start'], r['end']] for r in original_events]
                            == neural['predictions'][fit['modelId']][str(fit['seed'])][rid],
                            'Original neural event sequence differs from previous input')
                    event_rows = events(original_events, f"{fit['modelId']}::{fit['seed']}::{rid}::event",
                                        rows[rid]['durationSeconds'])
                    entries.append({'modelId': fit['modelId'], 'seed': fit['seed'], 'recordingId': rid,
                        'sourceGroup': fit['sourceGroup'], 'fitKey': fit['fitKey'], 'scoresKey': key,
                        'timesKey': 'times::'+rid, 'tickCount': len(times), 'events': event_rows,
                        'scoresBytesSha256': hashlib.sha256(scores.astype('<f4', copy=False).tobytes()).hexdigest(),
                        'headBytesSha256': {h: hashlib.sha256(scores[:, i].astype('<f4', copy=False).tobytes()).hexdigest()
                                            for i, h in enumerate(HEADS)}})
    finally:
        old_npz.close()
    require(len(entries) == 72 and len(arrays) == 80, 'Normalized array inventory differs')
    for ref in [identity(PRIOR), prior['npz'], prior['previousRegistration'], contract['neuralInput'],
                contract['productionInput'], prior['exactManifest'], *source_refs]:
        verify(ref)
    require(digest(PRIOR) == PRIOR_SHA, 'Prior input changed during preparation')
    output.mkdir(parents=True, exist_ok=True)
    with npz_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(npz_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(arrays)
                and all(np.array_equal(archive[k], a) and archive[k].dtype == a.dtype for k, a in arrays.items()),
                'Exact normalized probability round-trip failed')
    evidence = {'schemaVersion': 1, 'kind': 'held-source-group-rally-review-input-v1',
        'createdAt': datetime.now(timezone.utc).isoformat(), 'protectedTestOpened': False,
        'trainingPerformed': False, 'featureExtractionPerformed': False, 'outcomeScoringPerformed': False,
        'goldUsedForProposalGeneration': False, 'models': list(MODELS), 'seeds': list(SEEDS),
        'sourceGroups': groups, 'recordingCount': 8, 'selectedOuterFitCount': 36, 'scoreArrayCount': 72,
        'timelineArrayCount': 8, 'headNames': list(HEADS), 'scoresDtype': 'float32', 'timesDtype': 'float64',
        'roundTripExact': True, 'probabilityMeaning': 'Uncalibrated sigmoid head scores, not needs-review probabilities',
        'timestampRule': 'Exact original float64 AV tick centers, frame-quantized nominal4Hz; never arange/4',
        'durationRule': 'Exact cached duration for every row; prior production display rounding differs by less than 1e-6 seconds',
        'eventIdentityRule': 'Original ordered events retained separately; no clipping, ignoring, union, padding or short-gap joining here',
        'neuralEventIdRule': 'Source results have no IDs; stable model/seed/recording/sourceIndex IDs identify exact original entries',
        'productionEventIdentityRule': 'Current-default core exact-matched to original product unionWithConfidence; original R IDs retained',
        'productionComponentRule': 'Original AV2-/PP- event lists retained separately; actual app strict-overlap ensemble already merges internal boundaries',
        'goldRole': 'Outcome-only; rally-start targets are serve-contact proxies, serveMarkers are sparse and not exhaustive',
        'priorProbabilityInput': identity(PRIOR), 'previousRegistration': prior['previousRegistration'],
        'previousNeuralInput': contract['neuralInput'], 'previousProductionInput': contract['productionInput'],
        'exactManifest': prior['exactManifest'], 'independentTensorAudit': prior['independentTensorAudit'],
        'sourceScript': identity(Path(__file__).resolve()), 'numpyVersion': np.__version__,
        'npz': identity(npz_path), 'records': records, 'entries': entries, 'sourceFits': selected,
        'validation': {'exactNeuralEventListsVerified': 72, 'exactSourceLiveArraysVerified': 72,
                       'allFourHeadsRetained': True, 'originalProductionIdsRecovered': True,
                       'heldGroupsAbsentFromExactAndAuxiliaryTraining': True}}
    with json_path.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'input': identity(json_path), 'npz': identity(npz_path),
                      'source': evidence['sourceScript'], 'selectedOuterFits': 36,
                      'scoreArrays': 72, 'timelineArrays': 8}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    prepare(parser.parse_args().output)
