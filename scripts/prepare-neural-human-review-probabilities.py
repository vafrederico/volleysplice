#!/usr/bin/env python3
"""Bind selected, held-source-group neural probabilities for review-only study.

No fitting, decoding, labels-as-features, threshold selection or quality scoring.
Original sources stay unchanged; output files are created exclusively.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(private_value('private-reference-0057'))
PREVIOUS = ROOT / '2026-09-19-production-combinations'
TRANSFER = ROOT / '2026-09-19-short-boost-transfer'
DESTINATION = ROOT / '2026-09-19-perfect-human-review'
MODELS = ('compact_boost', 'compact_keep', 'dino_global', 'dino_boost', 'dino_keep')
SEEDS = (3407, 1729, 20260918)
EXPECTED_MANIFEST = '817c01a8809d93d6d9d23bc273a3eb4a9e3bf756cb54abe84aa337bd5f25196a'
EXPECTED_TENSOR_AUDIT = '137b446f4f40f01dd1aa7db5c1d248ef11ea9e307f9c8f8496def02f28d15a4c'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(ref):
    require(digest(ref['path']) == ref['sha256'], 'Changed bound file: ' + ref['path'])


def verified(ref):
    verify(ref)
    return read(ref['path'])


def verify_reference_result(result, report):
    keys = ('cohort', 'kind', 'lossArm', 'seed', 'rescueVariant')
    matched = [row for row in report['results'] if all(row.get(k) == result.get(k) for k in keys)]
    require(len(matched) == 1 and matched[0] == result, 'Result differs from unique audited report payload')


def validate_scores(scores, times):
    require(scores.shape == (len(times), 4) and scores.dtype == np.float32,
            'Scores must be original aligned float32 [ticks,4]')
    require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)),
            'Scores must be finite sigmoid probabilities')


def validate_membership(completed, group, rows):
    require(completed['validationGroups'] == [group], 'Validation group differs')
    require(group not in completed['trainGroups'] and all(
        group not in groups for groups in completed['auxiliaryGroups'].values()), 'Held group leaked into fit')
    expected = {rid for rid, row in rows.items() if row['sourceGroup'] == group}
    require(set(completed['validationIds']) == expected, 'Outer validation recording identity differs')
    require(not set(completed['validationIds']) & set(completed['trainIds']), 'Validation recording leaked into fit')
    require(all(not expected & set(ids) for ids in completed['auxiliaryIds'].values()),
            'Validation recording leaked into auxiliary fit')
    return expected


def interval_identity(rows, *, include_tags):
    return [(float(row['start']), float(row['end']), tuple(row.get('tags', [])) if include_tags else ())
            for row in rows]


def prepare(output):
    require(not (output/'probabilities.npz').exists() and not (output/'probability-input.json').exists(),
            'Refusing to overwrite immutable probability input')
    registration_path = PREVIOUS/'registration.json'
    registration = read(registration_path)
    contract = registration['contract']
    require(canonical(contract) == registration['sha256'], 'Prior registration contract digest differs')
    source_input = verified(contract['neuralInput'])
    source_bindings = [b for b in source_input['sourceBindings'] if b['model'] in MODELS]
    require({(b['model'], b['seed']) for b in source_bindings} == {(m, s) for m in MODELS for s in SEEDS}
            and len(source_bindings) == 15, 'Model/seed inventory differs')
    result_cells = {(b['model'], b['seed']): verified(b) for b in source_bindings}
    reference_bindings = {}
    for study_name, refs in contract['referenceStudies'].items():
        report = verified(refs['report.json'])
        reference_bindings[study_name] = refs['report.json']
        for binding in source_bindings:
            if study_name in Path(binding['path']).parts:
                verify_reference_result(result_cells[(binding['model'], binding['seed'])], report)
    manifest_path = TRANSFER/'manifest-pts-v1.json'
    require(digest(manifest_path) == EXPECTED_MANIFEST, 'Exact timeline manifest changed')
    manifest = read(manifest_path)
    require(len(manifest['exactRows']) == 8, 'Expected eight exact recordings')
    rows = {row['id']: row for row in manifest['exactRows']}
    input_records = {row['id']: row for row in source_input['records']}
    require(set(rows) == set(input_records), 'Recording inventory differs')
    tensor_path = TRANSFER/'tensor-scaler-audit-v1.json'
    require(digest(tensor_path) == EXPECTED_TENSOR_AUDIT, 'Independent source tensor audit changed')
    tensor = read(tensor_path)
    require(tensor['passed'] and tensor['studyComplete'], 'Independent source audit did not pass')
    require(tensor['report']['sha256'] == reference_bindings[TRANSFER.name]['sha256'], 'Audit report binding differs')
    audited_fits = {fit['completed']['path']: fit for fit in tensor['fits']}
    arrays, record_entries, times = {}, [], {}
    feature_refs = []
    for rid in sorted(rows):
        row, input_record = rows[rid], input_records[rid]
        require(row['sourceGroup'] == input_record['sourceGroup'] and row['environment'] in ('grass', 'indoor')
                and row['sourceGroup'] not in manifest['protectedSourceGroups'], 'Forbidden/different evaluation source')
        require(interval_identity(row['rallies'], include_tags=True) ==
                interval_identity(input_record['rallies'], include_tags=True) and
                interval_identity(row.get('ignoredIntervals', []), include_tags=False) ==
                interval_identity(input_record.get('ignoredIntervals', []), include_tags=False),
                'Gold/ignored identity differs from prior study')
        cache = row['featureCaches']['audiovisual']
        verify(cache)
        with np.load(cache['path'], allow_pickle=False) as archive:
            timeline = archive['times'].astype(np.float64)
            metadata_text = str(archive['metadata_json'].item())
        metadata = json.loads(metadata_text)
        require(timeline.ndim == 1 and len(timeline) and np.isfinite(timeline).all()
                and np.all(np.diff(timeline) > 0) and np.all(np.abs(np.diff(timeline)-.25) <= .1),
                'Invalid exact feature timeline')
        require(float(metadata['duration']) == input_record['durationSeconds'], 'Exact duration differs')
        key = 'times::'+rid
        arrays[key] = timeline
        times[rid] = timeline
        feature_ref = {key: cache[key] for key in ('path', 'sha256', 'sizeBytes')}
        feature_refs.append(feature_ref)
        record_entries.append({'id': rid, 'sourceGroup': row['sourceGroup'],
            'durationSeconds': input_record['durationSeconds'], 'timesKey': key, 'tickCount': len(timeline),
            'featureCache': feature_ref, 'metadataJsonSha256': hashlib.sha256(metadata_text.encode()).hexdigest(),
            'metadataCanonicalSha256': canonical(metadata),
            'timelineBytesSha256': hashlib.sha256(timeline.astype('<f8', copy=False).tobytes()).hexdigest()})
    fit_entries, probability_entries = [], []
    groups = contract['sourceGroups']
    for binding in source_bindings:
        cell = result_cells[(binding['model'], binding['seed'])]
        require(len(cell['selections']) == 4 and {s['heldSourceGroup'] for s in cell['selections']} == set(groups),
                'Selected outer fold inventory differs')
        for selection in cell['selections']:
            group = selection['heldSourceGroup']
            index = groups.index(group)
            folder = Path(cell['origin']['fitRoot'])/f'outer-{index}'/'refit'
            completed_path = folder/'completed.json'
            audited = audited_fits[str(completed_path)]
            completed = verified(audited['completed'])
            expected_ids = validate_membership(completed, group, rows)
            require(completed['model']['headNames'] == ['live', 'serve', 'end', 'keep'], 'Probability head order differs')
            require(completed['kind'] == cell['kind'] and completed['lossArm'] == cell['lossArm']
                    and completed['seed'] == cell['seed'], 'Source fit model identity differs')
            epoch = selection['epoch']
            checkpoint = [c for c in audited['checkpoints'] if c['epoch'] == epoch]
            require(len(checkpoint) == 1, 'Selected epoch absent/duplicated in audit')
            pred_path, weight_path = folder/f'predictions-{epoch}.npz', folder/f'weights-{epoch}.npz'
            for path in (pred_path, weight_path):
                require(digest(path) == completed['artifacts'][path.name] == checkpoint[0]['artifactHashes'][path.name],
                        'Source checkpoint artifact hash differs')
            fit_key = f"{binding['model']}::{binding['seed']}::{group}"
            fit_entries.append({'fitKey': fit_key, 'modelId': binding['model'], 'seed': binding['seed'],
                'sourceGroup': group, 'outerFoldIndex': index, 'epoch': epoch,
                'completed': audited['completed'], 'probabilities': identity(pred_path), 'weights': identity(weight_path),
                'sourceResult': binding, 'decoder': selection['decoder'],
                'keepThreshold': selection.get('keepThreshold'), 'membership': {k: completed[k] for k in (
                    'trainIds', 'auxiliaryIds', 'validationIds', 'trainGroups', 'auxiliaryGroups', 'validationGroups')},
                'sourceContractSha256': completed['contractSha256']})
            with np.load(pred_path, allow_pickle=False) as archive:
                require(set(archive.files) == expected_ids, 'Prediction NPZ recording inventory differs')
                for rid in sorted(archive.files):
                    scores = archive[rid]
                    validate_scores(scores, times[rid])
                    key = f"{binding['model']}::{binding['seed']}::{rid}::live"
                    require(key not in arrays, 'Duplicate normalized probability key')
                    arrays[key] = scores[:, 0].copy()
                    probability_entries.append({'modelId': binding['model'], 'seed': binding['seed'],
                        'recordingId': rid, 'sourceGroup': group, 'fitKey': fit_key, 'liveKey': key,
                        'timesKey': 'times::'+rid, 'tickCount': len(times[rid]),
                        'liveBytesSha256': hashlib.sha256(arrays[key].astype('<f4', copy=False).tobytes()).hexdigest()})
    require(len(fit_entries) == 60 and len(probability_entries) == 120 and len(arrays) == 128, 'Final input count differs')
    # Recheck source identities immediately before creating normalized outputs.
    for ref in [contract['neuralInput'], *source_bindings, *feature_refs]:
        verify(ref)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output/'probabilities.npz'
    with archive_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(archive_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(arrays) and all(np.array_equal(archive[k], v) and archive[k].dtype == v.dtype
                for k, v in arrays.items()), 'Normalized NPZ round-trip changed values/dtypes')
    evidence = {'schemaVersion': 1, 'kind': 'held-source-group-human-review-probabilities-v1',
        'createdAt': datetime.now(timezone.utc).isoformat(), 'protectedTestOpened': False,
        'trainingPerformed': False, 'featureExtractionPerformed': False, 'goldUsedForQueueSelection': False,
        'models': list(MODELS), 'seeds': list(SEEDS), 'recordingCount': 8, 'probabilityArrayCount': 120,
        'selectedOuterFitCount': 60, 'headNamesInSource': ['live', 'serve', 'end', 'keep'], 'retainedHeadIndex': 0,
        'probabilityMeaning': 'Uncalibrated live-head sigmoid scores; not a learned probability of needing review',
        'timestampRule': 'Exact float64 values from bound AV feature caches; frame-quantized nominal4Hz, never arange/4',
        'samplingSupportConvention': 'Use actual tick centers; downstream policy must declare its interval convention',
        'probabilityDtype': 'float32', 'timesDtype': 'float64', 'roundTripExact': True,
        'previousRegistration': identity(registration_path), 'previousContractSha256': registration['sha256'],
        'previousNeuralInput': contract['neuralInput'], 'sourceReports': reference_bindings,
        'exactManifest': identity(manifest_path), 'independentTensorAudit': identity(tensor_path),
        'sourceScript': identity(Path(__file__).resolve()), 'numpyVersion': np.__version__,
        'npz': identity(archive_path), 'records': record_entries, 'entries': probability_entries,
        'sourceFits': fit_entries, 'sourceResults': source_bindings}
    with (output/'probability-input.json').open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'input': identity(output/'probability-input.json'),
                      'npz': identity(archive_path), 'source': evidence['sourceScript'],
                      'selectedOuterFits': 60, 'probabilityArrays': 120, 'timelineArrays': 8}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    prepare(parser.parse_args().output)
