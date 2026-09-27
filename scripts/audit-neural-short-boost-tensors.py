#!/usr/bin/env python3
"""Independent final tensor/scaler/supervision audit of the54-cell transfer study.

No training, GPU work, source videos, protected labels or metric selection. Final
mode requires all54 result cells and follows validated origin.fitRoot paths for
768 fresh and96 immutable reused fits. An existing output is never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0084'))
MANIFEST = ROOT/'manifest-pts-v1.json'
VERIFIER_SHA256 = 'c7460cf0dc1cea7f54031405a0c78713edb54ebb941be18fd0bd8ad82b4c500b'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def verify_reference_identity(declared, registered):
    """Bind optional size metadata without confusing it with file identity."""
    required, allowed = {'path', 'sha256'}, {'path', 'sha256', 'sizeBytes'}
    require(required <= set(declared) <= allowed and required <= set(registered) <= allowed,
            'Unexpected reference identity fields')
    require(all(declared[key] == registered[key] for key in required), 'Reference path/hash binding differs')
    actual = identity(Path(declared['path']))
    for reference in (declared, registered):
        require(all(actual[key] == value for key, value in reference.items()), 'Reference artifact identity differs')


def registration(path):
    result = read(path)
    encoded = json.dumps(result['contract'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == result['sha256'], 'Invalid registration hash')
    return result


def independent_verifier():
    path = REPO/'scripts/audit-neural-expanded-tensors.py'
    require(digest(path) == VERIFIER_SHA256, 'Frozen independent scaler verifier changed')
    spec = importlib.util.spec_from_file_location('transfer_independent_scaler_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels(times, ranges):
    result = np.zeros(len(times), np.float32)
    for start, end in ranges:
        result[(times >= start) & (times < end)] = 1
    return result


def merge(ranges, gap=0.):
    result = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and (start <= result[-1][1] or 0 < start-result[-1][1] < gap):
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract(ranges, ignored):
    result = []
    for start, end in ranges:
        pieces = [(start, end)]
        for left, right in ignored:
            next_pieces = []
            for a, b in pieces:
                if right <= a or left >= b:
                    next_pieces.append((a, b))
                else:
                    if a < left:
                        next_pieces.append((a, left))
                    if right < b:
                        next_pieces.append((right, b))
            pieces = next_pieces
        result.extend(pieces)
    return result


def independent_supervision(row, tier, times, duration):
    """Rebuild labels/masks from annotation layers, without training helpers."""
    truth = [(r['start'], r['end']) for r in row.get('rallies', [])]
    ignored = [(r['start'], r['end']) for r in row.get('ignoredIntervals', [])]
    valid = ~labels(times, ignored).astype(bool)
    target = np.zeros((len(times), 4), np.float32)
    mask = np.zeros_like(target)
    if tier == 'exact':
        target[:, 0] = labels(times, truth)
        for head, index in ((1, 0), (2, 1)):
            for interval in truth:
                distance = np.abs(times-interval[index])
                pulse = np.exp(-.5*(distance/.35)**2)
                pulse[distance > 1.] = 0
                target[:, head] = np.maximum(target[:, head], pulse.astype(np.float32))
        padded = merge([(max(0., a-2), min(duration, b+2)) for a, b in truth], 3.)
        target[:, 3] = labels(times, subtract(padded, ignored))
        mask[:] = valid[:, None]
        for start, end in ignored:
            mask[(times >= start-1.) & (times <= end+1.), 1:3] = 0
    elif tier == 'draft':
        target[:, 0] = labels(times, truth)
        mask[:, 0] = valid
        for interval in truth+ignored:
            for boundary in interval:
                mask[np.abs(times-boundary) <= 1., 0] = 0
    else:
        require(tier == 'coverage', 'Unknown supervision tier')
        window = row['gameWindow']
        valid &= (times >= window['start']) & (times < window['end'])
        target[:, 3] = labels(times, [(r['start'], r['end']) for r in row['keepTargets']])
        mask[:, 3] = valid
    return {'targets': target, 'mask': mask, 'valid': valid,
            'counts': {'valid': mask.sum(0).astype(int).tolist(),
                       'positiveMass': (target*mask).sum(0).tolist()}}


def independent_weights(row, tier, times, supervision, arm):
    """Reconstruct the complete saved diagnostic and its float32 vector hash."""
    require(arm in ('baseline', 'global_control', 'short_boost'), 'Unknown loss arm')
    mask, targets, valid = supervision['mask'], supervision['targets'], supervision['valid']
    eligible = valid & (mask[:, 0] > 0) & (targets[:, 0] > 0)
    ranges = row.get('rallies', [])
    selected = [((times >= r['start']) & (times < r['end'])) for r in ranges]
    ownership = np.sum(selected, axis=0) if selected else np.zeros(len(times), np.int64)
    require(np.all(ownership[eligible] == 1), 'Positive tick lacks unique original event')
    short_flags = [r['end']-r['start'] <= 3 for r in ranges]
    event_counts = [int((inside & eligible).sum()) for inside in selected]
    p = int(eligible.sum())
    s = sum(count for count, short in zip(event_counts, short_flags) if short)
    ideal = 1.+s/p if p else 1.
    global_multiplier = np.float32(ideal)
    vector = np.ones(len(times), np.float32)
    if arm == 'global_control':
        vector[eligible] = global_multiplier
    elif arm == 'short_boost':
        for inside, short in zip(selected, short_flags):
            if short:
                vector[inside & eligible] = 2.
    events = []
    for index, (rally, inside, short, count) in enumerate(zip(ranges, selected, short_flags, event_counts)):
        ticks = inside & eligible
        events.append({'eventIndex': index, 'start': float(rally['start']), 'end': float(rally['end']),
            'tags': rally.get('tags', []), 'positiveSupervisedTicks': count,
            'invalidTicksInsideEvent': int((inside & ~valid).sum()),
            'maskedValidTicksInsideEvent': int((inside & valid & (mask[:, 0] == 0)).sum()),
            'originalDurationSeconds': float(rally['end']-rally['start']), 'isShortOriginalEvent': short,
            'multiplier': float(vector[ticks][0]) if count else None,
            'weightedPositiveMass': float(vector[ticks].astype(np.float64).sum())})
    observed = float(vector[eligible].astype(np.float64).sum())
    expected = p+(s if arm != 'baseline' else 0)
    return {'recordingId': row['id'], 'sourceGroup': row['sourceGroup'], 'tier': tier,
        'originalEventCount': len(ranges), 'eligibleEventCount': sum(n > 0 for n in event_counts),
        'zeroSupervisedEventCount': sum(n == 0 for n in event_counts),
        'negativeSupervisedTicks': int((valid & (mask[:, 0] > 0) & (targets[:, 0] == 0)).sum()),
        'method': 'fixed-short-rally-live-boost-with-record-matched-control-v1', 'mode': arm,
        'shortDurationSeconds': 3., 'shortPositiveMultiplier': 2.,
        'positiveSupervisedTicks': p, 'shortPositiveSupervisedTicks': s, 'longPositiveSupervisedTicks': p-s,
        'shortOriginalEventCount': sum(short_flags),
        'eligibleShortEventCount': sum(short and n > 0 for short, n in zip(short_flags, event_counts)),
        'zeroSupervisedShortEventCount': sum(short and n == 0 for short, n in zip(short_flags, event_counts)),
        'idealGlobalPositiveMultiplier': ideal, 'globalPositiveMultiplierFloat32': float(global_multiplier),
        'expectedPositiveMass': expected, 'weightedPositiveMass': observed,
        'positiveMassRoundingError': observed-expected, 'positiveMassAbsoluteTolerance': p*float(np.finfo(np.float32).eps),
        'minimumPositiveMultiplier': float(vector[eligible].min()) if p else None,
        'maximumPositiveMultiplier': float(vector[eligible].max()) if p else None,
        'liveMultiplierSha256': hashlib.sha256(vector.astype('<f4').tobytes()).hexdigest(),
        'liveMultiplierHashEncoding': 'little-endian float32, C-order, one value per original cache tick', 'events': events}


def parameter_contract():
    import torch
    from analysis.transfer_temporal_model import model_for
    result = {}
    with torch.random.fork_rng(devices=[]):
        for kind, count in (('tcn', 29700), ('dino_tcn', 46868)):
            model = model_for(kind).cpu()
            require(sum(p.numel() for p in model.parameters()) == count, 'Unexpected parameter count')
            result[kind] = {'parameterCount': count,
                           'stateShapes': {'model::'+name: list(value.shape) for name, value in model.state_dict().items()}}
    return result


def validate_origin(result, study, reference):
    cohort, kind, arm, seed = (result[k] for k in ('cohort', 'kind', 'lossArm', 'seed'))
    reused = cohort in ('exact', 'draft') and kind == 'tcn' and arm == 'baseline'
    origin = result['origin']
    expected = (Path(reference['path'])/'fits'/cohort/kind/str(seed) if reused else
                study/'fits'/cohort/kind/arm/str(seed))
    require(Path(origin['fitRoot']).resolve() == expected.resolve(), 'Unexpected origin fitRoot')
    require(origin['type'] == ('reused-reference' if reused else 'trained'), 'Unexpected origin type')
    if reused:
        require(origin['studyPath'] == reference['path'] and origin['reportSha256'] == reference['reportSha256']
                and origin['referenceContractSha256'] == reference['contractSha256'], 'Reused baseline provenance differs')
    return expected, reused


def validate_dataset_revision(manifest, original):
    """Only seven coverage AV caches may change; all annotation bytes stay fixed."""
    for key in ('exactRows', 'draftRows', 'exactManifest', 'protectedSourceGroups'):
        require(manifest[key] == original[key], 'Unchanged dataset scope differs: '+key)
    previous, current = original['coverageRows'], manifest['coverageRows']
    require(len(previous) == len(current) == 7
            and [r['id'] for r in previous] == [r['id'] for r in current], 'Coverage revision population/order differs')
    revised = []
    for old, new in zip(previous, current):
        require({k: v for k, v in old.items() if k != 'featureCaches'}
                == {k: v for k, v in new.items() if k != 'featureCaches'}, 'Coverage labels/source metadata changed')
        require({k: v for k, v in old['featureCaches'].items() if k != 'audiovisual'}
                == {k: v for k, v in new['featureCaches'].items() if k != 'audiovisual'}, 'Unrelated feature cache changed')
        left, right = old['featureCaches']['audiovisual'], new['featureCaches']['audiovisual']
        require(left['path'] != right['path'] and left['sha256'] != right['sha256'], 'Coverage AV cache was not independently replaced')
        revised.append({'id': new['id'], 'originalAV': left, 'correctedAV': right})
    restored = {key: value for key, value in manifest.items()
                if key not in ('originalManifest', 'protocolAmendment', 'featureRevision')}
    restored['coverageRows'] = [{**new, 'featureCaches': {**new['featureCaches'], 'audiovisual': old['featureCaches']['audiovisual']}}
                                for old, new in zip(previous, current)]
    require(restored == original, 'Unrelated manifest fields changed during PTS revision')
    return revised


def expected_membership(tiers, cohort, outer, held):
    excluded = {outer, held}
    train = [r['id'] for r in tiers['exact'] if r['sourceGroup'] not in excluded]
    active = () if cohort == 'exact' else ('draft',) if cohort == 'draft' else ('draft', 'coverage')
    auxiliary = {tier: [r['id'] for r in tiers[tier] if r['sourceGroup'] not in excluded] for tier in active}
    validating = [r['id'] for r in tiers['exact'] if r['sourceGroup'] == held]
    return {'trainIds': train, 'auxiliaryIds': auxiliary, 'validationIds': validating, 'scalerTrainIds': train}


def audit_exposure(left, right):
    for key in ('trainIds', 'auxiliaryIds', 'validationIds', 'scalerTrainIds', 'positiveWeight', 'supervisedCounts'):
        require(left[key] == right[key], f'Paired {key} differs')
    require(left['history'] and right['history'], 'Missing exposure history')
    for a, b in zip(left['history'], right['history']):
        require(a['epoch'] == b['epoch'] and a['optimizerSteps'] == b['optimizerSteps'], 'Paired optimizer steps differ')
        require(a['exposureSha256'] == b['exposureSha256'], 'Paired sampling exposure differs')
        require(set(a['exposureSha256']) == {'exact', *left['auxiliaryIds']}, 'Missing exposure stream')
    return min(len(left['history']), len(right['history']))


def audit_checkpoint(folder, epoch, meta, architecture, mean, scale, inputs):
    wp, pp = (folder/f'{stem}-{epoch}.npz' for stem in ('weights', 'predictions'))
    hashes = {p.name: digest(p) for p in (wp, pp)}
    require(all(meta['artifacts'][name] == sha for name, sha in hashes.items()), 'Changed checkpoint artifact')
    with np.load(wp, allow_pickle=False) as cache:
        expected = set(architecture['stateShapes']) | {'mean', 'scale'}
        require(len(cache.files) == len(expected) and set(cache.files) == expected, 'Model tensor inventory differs')
        for name in cache.files:
            value = cache[name]
            shape = [104] if name in ('mean', 'scale') else architecture['stateShapes'][name]
            require(list(value.shape) == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    f'Invalid model/scaler tensor: {name}')
        require(np.all(cache['scale'] >= np.float32(1e-4)), 'Scale below floor')
        mean_error = float(np.max(np.abs(cache['mean'].astype(np.float64)-mean)))
        scale_error = float(np.max(np.abs(cache['scale'].astype(np.float64)-scale)))
        require(np.array_equal(cache['mean'], mean) and np.array_equal(cache['scale'], scale),
                f'Independent exact-only scaler differs: mean={mean_error}, scale={scale_error}')
    predictions = []
    with np.load(pp, allow_pickle=False) as cache:
        require(len(cache.files) == len(meta['validationIds']) and set(cache.files) == set(meta['validationIds']),
                'Prediction recording inventory differs')
        for rid in meta['validationIds']:
            source, probabilities = inputs[rid], cache[rid]
            require(source['tier'] == 'exact', 'Auxiliary recording entered evaluation')
            require(probabilities.shape == (len(source['times']), 4) and probabilities.dtype == np.float32
                    and np.isfinite(probabilities).all(), 'Prediction shape/dtype/finite contract differs')
            require(np.all((probabilities >= 0) & (probabilities <= 1)), 'Prediction outside probability bounds')
            require(np.all(probabilities[~source['valid']] == 0), 'Ignored prediction ticks are nonzero')
            predictions.append({'id': rid, 'shape': list(probabilities.shape), 'ignoredTicks': int((~source['valid']).sum())})
    return {'epoch': epoch, 'artifactHashes': hashes, 'scalerExactlyEqual': True,
            'meanMaximumAbsoluteError': mean_error, 'scaleMaximumAbsoluteError': scale_error, 'predictions': predictions}


def validate_dino_association(entry, manifest, source, tier, metadata, wrapper):
    roi = [float(source['roi'][key]) for key in ('x', 'y', 'width', 'height')] if source.get('roi') else None
    require(entry['recordingId'] == source['id'] and entry['tier'] == tier
            and entry['sourceGroup'] == source['sourceGroup'] and entry.get('passed') is True
            and entry['audiovisualSha256'] == source['featureCaches']['audiovisual']['sha256']
            and entry['sourceVideoVerified']['sha256'] == source['contentSha256'], 'DINO source association differs')
    require(wrapper['recordingId'] == source['id'] and wrapper['cacheMetadata'] == metadata
            and wrapper['cache']['sha256'] == entry['dinoSha256']
            and metadata['recordingId'] == source['id'] and metadata['recordingContentSha256'] == source['contentSha256']
            and metadata['extractorConfigSha256'] == manifest['extractorConfigSha256']
            and metadata['roi'] == roi and metadata.get('labelsUsed') is False
            and metadata.get('completed') is True
            and all(metadata['backbone'][key] == value for key, value in manifest['semanticBackbone'].items()),
            'DINO embedded metadata association/recipe differs')


def validate_pts_vectors(arrays, selection, av_times, dino_times):
    """Check nearest media-frame selection without calling the repair extractor."""
    expected = {'times', 'packetPtsTicks', 'presentationTimes', 'selectedOrdinals',
                'selectedPresentationTimes', 'observedPresentationTimes', 'selectedFrameSha256'}
    require(set(arrays) == expected, 'PTS selection array inventory differs')
    times, ticks, pts, indexes, chosen, observed, hashes = (arrays[key] for key in (
        'times', 'packetPtsTicks', 'presentationTimes', 'selectedOrdinals',
        'selectedPresentationTimes', 'observedPresentationTimes', 'selectedFrameSha256'))
    duration = selection['videoDurationSeconds']
    require(np.isfinite(duration) and duration > 0, 'PTS duration invalid')
    grid = np.arange(int(np.ceil(duration*4-1e-9)), dtype=np.float64)/4
    grid = grid[grid < duration]
    require(times.dtype == np.float64 and np.array_equal(times, grid)
            and np.array_equal(av_times, times) and np.array_equal(dino_times, times), 'Corrected AV/DINO media grids differ')
    require(ticks.dtype == np.int64 and ticks.ndim == 1 and len(ticks) == selection['videoFrameCount']
            and len(ticks) > 0 and np.all(np.diff(ticks) > 0), 'PTS packet inventory invalid')
    numerator, denominator = (int(value) for value in selection['timeBase'].split('/'))
    require(numerator > 0 and denominator > 0 and pts.dtype == np.float64
            and np.array_equal(pts, ticks.astype(np.float64)*numerator/denominator)
            and np.isfinite(pts).all() and abs(pts[0]) <= 1e-6, 'Presentation clock differs from packet time base')
    require(indexes.dtype == np.int64 and indexes.shape == times.shape and len(times) == selection['sampleCount']
            and np.all((indexes >= 0) & (indexes < len(pts))) and np.all(np.diff(indexes) > 0), 'Selected frame ordinal inventory invalid')
    require(chosen.dtype == observed.dtype == np.float64 and chosen.shape == observed.shape == times.shape
            and np.array_equal(chosen, pts[indexes]) and np.isfinite(observed).all(), 'Selected/decoded presentation times invalid')
    error = np.abs(chosen-times)
    prior, following = indexes > 0, indexes < len(pts)-1
    # Every point on a sorted timeline is globally nearest iff neither adjacent
    # point is closer. A tied earlier neighbor disqualifies the chosen point.
    require(np.all(error[prior] < np.abs(times[prior]-pts[indexes[prior]-1]))
            and np.all(error[following] <= np.abs(times[following]-pts[indexes[following]+1])), 'Nearest-PTS selection or earlier-tie rule differs')
    decoded_error = np.abs(observed-chosen)
    require(float(error.max()) <= .125+1e-9 and float(decoded_error.max()) <= 2e-6, 'Decoded frame/tick PTS error exceeds contract')
    require(hashes.shape == times.shape and all(len(str(value)) == 64 and set(str(value)) <= set('0123456789abcdef') for value in hashes),
            'Selected frame pixel hashes invalid')
    for key, value, dtype in (('gridSha256', times, '<f8'), ('presentationTimesSha256', pts, '<f8'),
                              ('selectedOrdinalsSha256', indexes, '<i8'), ('selectedPresentationTimesSha256', chosen, '<f8')):
        require(hashlib.sha256(value.astype(dtype).tobytes()).hexdigest() == selection[key], 'PTS vector hash differs: '+key)
    require(hashlib.sha256('\n'.join(str(value) for value in hashes).encode()).hexdigest() == selection['frameHashesSha256'],
            'Selected pixel identity vector hash differs')
    require(np.isclose(error.max(), selection['maximumFrameSelectionErrorSeconds'], rtol=0, atol=1e-12)
            and np.isclose(decoded_error.max(), selection['maximumDecodedPtsErrorSeconds'], rtol=0, atol=1e-12), 'PTS error diagnostic differs')
    return {'frames': len(pts), 'samples': len(times), 'maximumSelectionErrorSeconds': float(error.max()),
            'maximumDecodedPtsErrorSeconds': float(decoded_error.max()), 'sharedMediaGridVerified': True,
            'independentNearestFrameSelectionVerified': True}


def verify_media_pts_binding(entry, manifest, source, av_times, dino_times, metadata):
    decoder = 'sequential-opencv-nearest-media-pts-v1'
    av = source['featureCaches']['audiovisual']
    binding = av['decoderSelection']
    require(binding == metadata['decoderSelection'] and binding['decoder'] == av['videoDecoder'] == decoder,
            'AV/DINO decoder selection bindings differ')
    require(entry['decoderSelectionPath'] == binding['selection']['path']
            and entry['decoderSelectionSha256'] == binding['selection']['sha256'], 'DINO row decoder selection differs')
    selected_path, array_path = Path(binding['selection']['path']), Path(binding['arrays']['path'])
    require(digest(selected_path) == binding['selection']['sha256'] and digest(array_path) == binding['arrays']['sha256'],
            'PTS selection provenance changed')
    selection = read(selected_path)
    require(selection['recordingId'] == source['id'] and selection['source']['sha256'] == source['contentSha256']
            and selection['decoder'] == decoder and selection['arrays'] == binding['arrays']
            and selection['repairPlan'] == manifest['repairPlan'], 'PTS recording/repair-plan association differs')
    with np.load(av['path'], allow_pickle=False) as data:
        require(str(data['video_decoder'].item()) == decoder
                and json.loads(str(data['decoder_selection_json'].item())) == binding, 'AV embedded decoder provenance differs')
    with np.load(array_path, allow_pickle=False) as data:
        result = validate_pts_vectors({key: data[key] for key in data.files}, selection, av_times, dino_times)
    audio = selection['audio']
    require(audio['sampleRate'] == 16000 and audio['formulaUnchanged'] is True
            and abs(float(audio['stream']['start_time'])-audio['startOffsetSeconds']) < 1e-12
            and abs(audio['stream']['firstDecodedFramePtsSeconds']-audio['startOffsetSeconds']) <= 1/float(audio['stream']['sample_rate'])
            and audio['alignmentSamples'] == round(audio['startOffsetSeconds']*audio['sampleRate']), 'Media audio alignment differs')
    return {**result, 'selection': identity(selected_path), 'arrays': identity(array_path), 'audioOriginOffsetSeconds': audio['startOffsetSeconds']}


def verify_dino_inputs(contract, manifest_path, inputs):
    path = Path(contract['dinoManifest']['path'])
    require(digest(path) == contract['dinoManifest']['sha256'], 'DINO manifest changed')
    manifest = read(path)
    require(manifest['expandedManifestPath'] == str(manifest_path)
            and manifest['expandedManifestSha256'] == digest(manifest_path)
            and manifest.get('protectedTestOpened') is False and manifest.get('beachIncluded') is False
            and manifest.get('embeddingLabelsUsed') is False, 'DINO dataset binding differs')
    require(manifest['originalManifest'] == contract['originalManifest']
            and manifest['protocolAmendment'] == contract['protocolAmendment']
            and manifest['featureRevision'] == contract['featureRevision'], 'DINO repair revision differs')
    repair_plan_path = Path(manifest['repairPlan']['path'])
    require(digest(repair_plan_path) == manifest['repairPlan']['sha256'], 'PTS repair plan changed')
    repair_plan = read(repair_plan_path)
    verify_reference_identity(repair_plan['originalManifest'], contract['originalManifest'])
    verify_reference_identity(repair_plan['protocolAmendment'], contract['protocolAmendment'])
    require(repair_plan['sourceCode'] == manifest['sourceCode'], 'PTS repair provenance differs')
    for item in repair_plan['sourceCode'].values():
        require(digest(Path(item['path'])) == item['sha256'], 'PTS extraction source changed')
    source_manifest = read(manifest_path)
    sources = {row['id']: row for key in ('exactRows', 'draftRows', 'coverageRows') for row in source_manifest[key]}
    entries = manifest['records']
    require(len(entries) == len(inputs) == 18 and {r['recordingId'] for r in entries} == set(inputs), 'DINO population differs')
    result = []
    for entry in entries:
        rid = entry['recordingId']
        cache_path, metadata_path = Path(entry['dinoPath']), Path(entry['metadataPath'])
        require(digest(cache_path) == entry['dinoSha256'] and digest(metadata_path) == entry['metadataSha256'], 'DINO artifact changed')
        with np.load(cache_path, allow_pickle=False) as cache:
            times, tokens = cache['timestamps'].astype(np.float64), cache['tokens']
            metadata = json.loads(str(cache['metadata_json'].item()))
            validate_dino_association(entry, manifest, sources[rid], inputs[rid]['tier'], metadata, read(metadata_path))
            require(times.ndim == 1 and len(times) and np.isfinite(times).all() and np.all(np.diff(times) > 0), 'DINO timeline invalid')
            require(tokens.shape == (len(times), 10, 384) and tokens.dtype == np.float16 and np.isfinite(tokens).all(), 'DINO token contract differs')
            av = inputs[rid]['times']
            right = np.clip(np.searchsorted(times, av), 0, len(times)-1)
            left = np.maximum(0, right-1)
            error = np.minimum(np.abs(times[right]-av), np.abs(times[left]-av))
            require(float(error.max()) <= .125+1e-8, 'DINO/AV timeline mismatch')
            nearest = np.where(np.abs(times[left]-av) <= np.abs(times[right]-av), left, right)
            for key, value in (('avTimesSha256', av.astype('<f8')), ('dinoTimesSha256', times.astype('<f8')),
                               ('nearestIndexesSha256', nearest.astype('<i8'))):
                require(hashlib.sha256(value.tobytes()).hexdigest() == entry['nearestAlignment'][key],
                        f'DINO alignment identity differs: {rid}/{key}')
            pts_verification = (verify_media_pts_binding(entry, manifest, sources[rid], av, times, metadata)
                                if inputs[rid]['tier'] == 'coverage' else None)
        result.append({'recordingId': rid, 'dino': identity(cache_path), 'metadata': identity(metadata_path),
                       'maximumNearestTimestampError': float(error.max()), 'tokens': len(times),
                       'mediaPtsVerification': pts_verification})
        del tokens
    return result


def audit(manifest_path, study):
    require((study/'report.json').is_file(), 'Final tensor audit requires completed report.json')
    verifier = independent_verifier()
    reg, report = registration(study/'preregistration.json'), read(study/'report.json')
    contract = reg['contract']
    require(contract['experiment'] == 'bounded-short-boost-dino-transfer-v1' and len(contract['code']) == 16, 'Unexpected source contract')
    require(contract['kinds'] == ['tcn', 'dino_tcn'] and contract['lossArms'] == ['baseline', 'global_control', 'short_boost']
            and len(contract['cohorts']) == 3 and len(contract['seeds']) == 3 and len(contract['groups']) == 4
            and contract['checkpointEpochs'] == [5, 15, 30, 60], 'Unexpected factorial grid')
    require(report['status'] == 'completed-short-boost-transfer-development' and report['contractSha256'] == reg['sha256']
            and report['manifestSha256'] == contract['manifestSha256']
            and not report['protectedTestOpened'] and not report['productionPromotionAllowed'], 'Incomplete or mismatched report')
    locked_inputs = [identity(path) for path in (manifest_path, study/'preregistration.json', study/'report.json', Path(__file__))]
    require(digest(manifest_path) == contract['manifestSha256'], 'Expanded manifest changed')
    for name, sha in contract['code'].items():
        require(digest(REPO/'analysis'/name) == sha, f'Frozen source changed: {name}')
    reference = contract['referenceStudy']
    reference_root = Path(reference['path'])
    reference_bindings = []
    for filename, field in (('preregistration.json', 'preregistrationFileSha256'), ('report.json', 'reportSha256'), ('summary.json', 'summarySha256')):
        value = identity(reference_root/filename)
        require(value['sha256'] == reference[field], 'Historical reference changed')
        reference_bindings.append(value)
    old_reg = registration(reference_root/'preregistration.json')
    require(old_reg['sha256'] == reference['contractSha256'] and len(old_reg['contract']['code']) == 11, 'Historical registration differs')
    require(all(contract['code'][name] == sha for name, sha in old_reg['contract']['code'].items()), 'Historical source bindings differ')
    preflight_path = Path(contract['preflight']['path'])
    require(digest(preflight_path) == contract['preflight']['sha256'], 'Preflight report changed')
    preflight = read(preflight_path)
    require(preflight['passed'] is True and preflight['code'] == contract['code'], 'Preflight source binding differs')
    locked_inputs += reference_bindings + [identity(preflight_path)]
    historical = {(r['cohort'], r['kind'], r['seed']): r for r in read(reference_root/'report.json')['results']}
    manifest = read(manifest_path)
    original_path = Path(contract['originalManifest']['path'])
    require(digest(original_path) == contract['originalManifest']['sha256'] == old_reg['contract']['manifestSha256'],
            'Original dataset reference changed')
    original_manifest = read(original_path)
    revised_caches = validate_dataset_revision(manifest, original_manifest)
    locked_inputs.append(identity(original_path))
    require(manifest['originalManifest'] == contract['originalManifest']
            and manifest['protocolAmendment'] == contract['protocolAmendment']
            and manifest['featureRevision'] == contract['featureRevision'], 'Dataset amendment differs from registered revision')
    amendment_path = Path(contract['protocolAmendment']['path'])
    require(digest(amendment_path) == contract['protocolAmendment']['sha256'], 'PTS protocol amendment changed')
    locked_inputs.append(identity(amendment_path))
    exact_path = Path(manifest['exactManifest']['path'])
    require(digest(exact_path) == manifest['exactManifest']['sha256'], 'Exact manifest changed')
    exact = read(exact_path)
    locked_inputs.append(identity(exact_path))
    require(exact['recordings'] == manifest['exactRows'], 'Exact label/feature revision differs')
    tiers = {tier: manifest[key] for tier, key in (('exact', 'exactRows'), ('draft', 'draftRows'), ('coverage', 'coverageRows'))}
    by_id = {r['id']: r for rows in tiers.values() for r in rows}
    require(len(by_id) == sum(map(len, tiers.values())) == 18, 'Dataset identity differs')
    require(all(r['consent']['train'] is True and r['environment'] in ('grass', 'indoor')
                and r['sourceGroup'] not in manifest['protectedSourceGroups'] for r in by_id.values()), 'Excluded source or missing consent')
    require(report['records'] == len(tiers['exact']) == 8 and report['sourceGroups'] == contract['groups'], 'Evaluation population differs')
    inputs = verifier.load_cache_inputs(manifest)
    supervision, expected_weights = {}, {}
    for tier, rows in tiers.items():
        for row in rows:
            rid, item = row['id'], inputs[row['id']]
            with np.load(item['cache']['path'], allow_pickle=False) as cache:
                duration = float(json.loads(str(cache['metadata_json'].item()))['duration'])
            supervision[rid] = independent_supervision(row, tier, item['times'], duration)
            require(np.array_equal(supervision[rid]['valid'], item['valid']), 'Independent validity differs')
            for arm in contract['lossArms']:
                expected_weights[(rid, arm)] = independent_weights(row, tier, item['times'], supervision[rid], arm)
    dino_inputs = verify_dino_inputs(contract, manifest_path, inputs)
    global_weights = read(study/'weight-diagnostics.json')
    locked_inputs += [identity(study/'weight-diagnostics.json'), identity(Path(contract['dinoManifest']['path']))]
    require(global_weights['contractSha256'] == reg['sha256'], 'Global weighting contract differs')
    expected_global = {arm: {'mode': arm, 'rows': {tier: [expected_weights[(r['id'], arm)] for r in rows]
                                                 for tier, rows in tiers.items()}} for arm in contract['lossArms']}
    require(global_weights['arms'] == expected_global, 'Global live weights differ from independent reconstruction')
    expected_cells = {(c, k, a, s) for c in contract['cohorts'] for k in contract['kinds'] for a in contract['lossArms'] for s in contract['seeds']}
    key_for = lambda r: (r['cohort'], r['kind'], r['lossArm'], r['seed'])
    results = report['results']
    require(len(results) == len(expected_cells) == 54 and {key_for(r) for r in results} == expected_cells, 'Final54-cell grid is incomplete')
    architecture = parameter_contract()
    scaler_cache, metas, checked_paths = {}, {}, set()
    fits, failures, counts = [], [], Counter()
    groups = contract['groups']
    for result in results:
        cohort, kind, arm, seed = key_for(result)
        require(result['contractSha256'] == reg['sha256'] and result['architecture'] == kind, 'Result identity differs')
        result_file = study/f'result-{cohort}-{kind}-{arm}-{seed}.json'
        require(read(result_file) == result, 'Result file differs from final report')
        fit_root, reused = validate_origin(result, study, reference)
        if reused:
            old = historical[(cohort, kind, seed)]
            require(all(result[field] == old[field] for field in ('selections', 'evaluation', 'predictions')), 'Reused baseline result changed')
        selections = {row['heldSourceGroup']: row for row in result['selections']}
        require(len(result['selections']) == len(selections) == 4 and set(selections) == set(groups), 'Missing outer selection')
        for outer_index, outer in enumerate(groups):
            folds = [(f'inner-{i}', held) for i, held in enumerate(g for g in groups if g != outer)] + [('refit', outer)]
            for fold, held in folds:
                folder = fit_root/f'outer-{outer_index}'/fold
                require(folder.resolve() not in checked_paths, 'Fit path reused by multiple factorial cells')
                checked_paths.add(folder.resolve())
                try:
                    path = folder/'completed.json'
                    completed_identity = identity(path)
                    meta = read(path)
                    expected_contract = old_reg['sha256'] if reused else reg['sha256']
                    require(meta['contractSha256'] == expected_contract and meta['kind'] == kind and meta['seed'] == seed, 'Fit contract/kind/seed differs')
                    membership = expected_membership(tiers, cohort, outer, held)
                    require(all(meta[key] == value for key, value in membership.items()), 'Fit population or source-fold leakage')
                    ids_by_tier = {'exact': meta['trainIds'], **meta['auxiliaryIds']}
                    require(meta['trainGroups'] == sorted({by_id[rid]['sourceGroup'] for rid in meta['trainIds']})
                            and meta['validationGroups'] == [held]
                            and meta['auxiliaryGroups'] == {tier: sorted({by_id[rid]['sourceGroup'] for rid in ids}) for tier, ids in meta['auxiliaryIds'].items()}, 'Fit group metadata differs')
                    expected_counts = {tier: {field: (np.sum([supervision[rid]['counts'][field] for rid in ids], axis=0).tolist() if ids else [0]*4)
                                             for field in ('valid', 'positiveMass')} for tier, ids in ids_by_tier.items()}
                    require(set(meta['supervisedCounts']) == set(expected_counts), 'Supervision tier differs')
                    for tier, row in expected_counts.items():
                        for field, values in row.items():
                            require(np.allclose(meta['supervisedCounts'][tier][field], values, rtol=1e-6, atol=1e-4), 'Four-head supervision differs')
                    valid, positive = (np.asarray(expected_counts['exact'][field]) for field in ('valid', 'positiveMass'))
                    positive_weight = np.minimum(20., np.sqrt((valid-positive)/np.maximum(positive, 1.)))
                    require(np.allclose(meta['positiveWeight'], positive_weight, rtol=1e-7, atol=1e-7), 'Exact-only positive class weights differ')
                    if not reused:
                        require(meta['architecture'] == kind and meta['lossArm'] == arm and meta['model'] == contract['models'][kind], 'Fit representation/loss contract differs')
                        expected_diagnostics = {'mode': arm, 'rows': {tier: [expected_weights[(rid, arm)] for rid in ids] for tier, ids in ids_by_tier.items()}}
                        require(meta['liveLossWeighting'] == expected_diagnostics, 'Per-fit live weighting differs')
                        counts['freshFitWeightDiagnosticsVerified'] += 1
                    epochs = [selections[outer]['epoch']] if fold == 'refit' else contract['checkpointEpochs']
                    require(meta['epochs'] == epochs and all(e in contract['checkpointEpochs'] for e in epochs), 'Selected checkpoint inventory differs')
                    require(meta['parameters'] == architecture[kind]['parameterCount'], 'Parameter count differs')
                    names = {f'{stem}-{epoch}.npz' for epoch in epochs for stem in ('weights', 'predictions')}
                    require(set(meta['artifacts']) == names and {p.name for p in folder.glob('*.npz')} == names, 'Artifact inventory differs')
                    require([h['epoch'] for h in meta['history']] == list(range(1, max(epochs)+1)), 'Incomplete epoch history')
                    require(meta['optimizerSteps'] == meta['history'][-1]['optimizerSteps']
                            and meta['exposureSha256'] == meta['history'][-1]['exposureSha256'], 'Final exposure differs')
                    require(all(np.isfinite(h['loss']) and set(h['exposureSha256']) == set(ids_by_tier) for h in meta['history']), 'Invalid history or sampling stream')
                    scaler_key = tuple(meta['trainIds'])
                    if scaler_key not in scaler_cache:
                        scaler_cache[scaler_key] = verifier.independent_scaler(meta['trainIds'], inputs)
                    mean, scale, valid_ticks = scaler_cache[scaler_key]
                    checkpoints = [audit_checkpoint(folder, epoch, meta, architecture[kind], mean, scale, inputs) for epoch in epochs]
                    require(digest(path) == completed_identity['sha256'], 'Completed metadata changed during audit')
                    fit_key = (cohort, kind, arm, seed, outer_index, fold)
                    metas[fit_key] = meta
                    fits.append({'cohort': cohort, 'kind': kind, 'lossArm': arm, 'seed': seed, 'reused': reused,
                                 'completed': completed_identity, 'scalerTrainIds': meta['trainIds'], 'scalerValidTicks': valid_ticks,
                                 'checkpoints': checkpoints})
                    counts['reusedFits' if reused else 'freshFits'] += 1
                    counts['checkpoints'] += len(checkpoints)
                    counts['NPZArtifacts'] += 2*len(checkpoints)
                    counts['predictionArrays'] += sum(len(c['predictions']) for c in checkpoints)
                except Exception as error:
                    failures.append({'path': str(folder), 'error': str(error)})
    if not failures:
        for cohort in contract['cohorts']:
            for seed in contract['seeds']:
                for outer_index in range(4):
                    for fold in ('inner-0', 'inner-1', 'inner-2', 'refit'):
                        at = lambda kind, arm: metas[(cohort, kind, arm, seed, outer_index, fold)]
                        for kind in contract['kinds']:
                            for arm in ('global_control', 'short_boost'):
                                counts['crossArmEpochPrefixes'] += audit_exposure(at(kind, arm), at(kind, 'baseline'))
                                counts['crossArmFitPairs'] += 1
                        for arm in contract['lossArms']:
                            counts['crossArchitectureEpochPrefixes'] += audit_exposure(at('tcn', arm), at('dino_tcn', arm))
                            counts['crossArchitectureFitPairs'] += 1
                        for kind in contract['kinds']:
                            for arm in contract['lossArms']:
                                if cohort == 'reviewed_export' or (kind == 'tcn' and arm == 'baseline'):
                                    continue
                                counts['historicalEpochPrefixes'] += audit_exposure(at(kind, arm), at('tcn', 'baseline'))
                                counts['historicalFitPairs'] += 1
        require(counts['freshFits'] == 768 and counts['reusedFits'] == 96 and counts['checkpoints'] == 2808
                and counts['NPZArtifacts'] == 5616, 'Final fit/checkpoint inventory differs')
        require(counts['crossArmFitPairs'] == 576 and counts['crossArchitectureFitPairs'] == 432
                and counts['historicalFitPairs'] == 480, 'Exposure comparison inventory differs')
        fresh_paths = {Path(f['completed']['path']).parent.resolve() for f in fits if not f['reused']}
        require({p.parent.resolve() for p in (study/'fits').rglob('completed.json')} == fresh_paths, 'Unexpected fresh fit directories')
        require(len(list((study/'fits').rglob('*.npz'))) == 4992, 'Fresh NPZ inventory differs')
    for name, sha in contract['code'].items():
        require(digest(REPO/'analysis'/name) == sha, 'Frozen source changed during audit')
    for bound in locked_inputs:
        require(digest(Path(bound['path'])) == bound['sha256'], 'Registered input changed during audit')
    return {'kind': 'neural-short-boost-independent-tensor-audit-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
        'passed': not failures, 'studyComplete': True, 'contractSha256': reg['sha256'], 'counts': dict(counts),
        'expected': {'resultCells': 54, 'fits': 864, 'freshFits': 768, 'reusedFits': 96, 'checkpoints': 2808,
                     'NPZArtifacts': 5616, 'freshNPZArtifacts': 4992, 'reusedNPZArtifacts': 624},
        'registeredSourceCount': len(contract['code']), 'registeredSourceHashesVerified': contract['code'],
        'historicalSourceCount': len(old_reg['contract']['code']), 'resultCellsVerified': len(results),
        'fourHeadSupervisionRecordingsReconstructed': len(supervision), 'liveWeightVectorsReconstructed': len(expected_weights),
        'uniqueExactScalerPopulations': len(scaler_cache), 'scalerExactlyEqualCheckpointCount': counts['checkpoints'],
        'maximumMeanAbsoluteError': 0. if not failures else None, 'maximumScaleAbsoluteError': 0. if not failures else None,
        'manifest': identity(manifest_path), 'exactManifest': identity(exact_path), 'registration': identity(study/'preregistration.json'),
        'originalManifest': identity(original_path), 'correctedCoverageCaches': revised_caches,
        'report': identity(study/'report.json'), 'referenceBindings': reference_bindings,
        'architectures': architecture, 'dinoInputs': dino_inputs,
        'auditScript': identity(Path(__file__)), 'independentScalerVerifier': identity(REPO/'scripts/audit-neural-expanded-tensors.py'),
        'fits': fits, 'failures': failures,
        'limits': ['CPU-only audit; no fitting or source-video/protected-label reads.',
                   'DINO frozen extractor cost/quality is outside downstream parameter counts.',
                   'Prediction NPZ index/length/validity checked against AV timelines; this is not complete neural forward replay of2808 checkpoints.',
                   'Metric replay, decoder selection and improvement/recovery screens belong to the separate summary audit.',
                   'Historical exposure comparisons include only unchanged exact/draft cohorts; corrected coverage uses freshly trained compact baselines.',
                   'Per-record positive-mass matching does not imply equal sampled-minibatch mass.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root/'tensor-scaler-audit-v1.json'
    require(not output.exists(), 'Refusing to overwrite immutable tensor audit')
    result = audit(args.manifest, args.study or args.root/'study')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'counts': result['counts'], 'failures': result['failures'],
                      'artifact': identity(output)}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
