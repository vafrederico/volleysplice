"""Explicit-role inputs for expansion research without modifying legacy loaders.

Fitting uses the historical supervision rules. Inference has no access to rally
labels, ignored spans, or game windows: all feature ticks are valid. Evaluation
labels can subsequently be attached without changing that inference mask.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from . import neural_development as base
from . import neural_expanded_development as expanded
from . import mobile_visual_features as mobile
from .config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from .features import ABSOLUTE_FEATURE_NAMES, feature_names, percentile_rank_values
from .schema import mask_for_times

TIERS = ('exact', 'draft', 'coverage')
POLICIES = {'exact': 'exact-core', 'draft': 'draft-reviewed',
            'coverage': 'export-coverage', 'unscored': 'none'}
FAMILIES = ('av', 'dino', 'mobile', 'distilled')
_verified = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verified(reference):
    require(isinstance(reference, Mapping) and 'path' in reference and 'sha256' in reference,
            'A path and SHA-256 are required for every feature reference')
    path = Path(reference['path']).resolve()
    stat = path.stat()
    signature = (reference['sha256'], stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
    if path in _verified:
        require(_verified[path] == signature, 'Previously verified immutable feature changed: ' + str(path))
        return path
    require(sha(path) == reference['sha256'], 'Feature identity changed: ' + str(path))
    after = path.stat()
    require((stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
            == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), 'Feature changed while hashing')
    _verified[path] = signature
    return path


def manifest_rows(manifest_path):
    manifest = read(manifest_path)
    require(manifest.get('kind') == 'neural-generalization-inputs-v1', 'Wrong input manifest kind')
    rows = manifest.get('records')
    require(isinstance(rows, list) and rows, 'Empty or invalid generalized population')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate recording IDs')
    for row in rows:
        require(row.get('labelTier') in POLICIES, 'Unknown label quality tier')
        require(row.get('scoringPolicy') == POLICIES[row['labelTier']], 'Label tier/scoring policy mismatch')
        require(isinstance(row.get('eligibleRoles'), list)
                and set(row['eligibleRoles']) <= {'fit', 'calibrate', 'evaluate', 'infer'}, 'Invalid source roles')
        require(isinstance(row.get('protected'), bool), 'Protected status must be explicit')
        require(row.get('environment') != 'beach' and bool(row.get('sourceGroup')), 'Beach or missing group')
        require(np.isfinite(row['durationSeconds']) and row['durationSeconds'] > 0, 'Invalid source duration')
        require(len(row.get('contentSha256', '')) == 64, 'Missing source content SHA-256')
        require(bool(row.get('featureOrigin')), 'Feature origin must be explicit')
    return manifest, rows


def select_rows(rows, role, recording_ids=None):
    require(role in ('fit', 'calibrate', 'evaluate', 'infer'), 'Unknown loading role')
    requested = None if recording_ids is None else list(recording_ids)
    require(requested is None or len(set(requested)) == len(requested), 'Duplicate requested IDs')
    by_id = {r['id']: r for r in rows}
    require(requested is None or set(requested) <= set(by_id), 'Unknown requested recording')
    chosen = [r for r in rows if role in r['eligibleRoles']] if requested is None else [by_id[k] for k in requested]
    for row in chosen:
        require(role in row['eligibleRoles'], 'Recording is not eligible for role: ' + row['id'])
        if role in ('fit', 'calibrate'):
            require(not row['protected'], 'Protected footage cannot fit or calibrate')
            require(row.get('consent', {}).get('train') is True, 'Training/calibration consent absent')
            require(row['labelTier'] in TIERS, 'Unscored source cannot provide supervision')
    return chosen


def feature_entries(rows, feature_index_path=None):
    if feature_index_path is None:
        return {r['id']: r.get('featureCaches', {}) for r in rows}
    index = read(feature_index_path)
    require(index.get('kind') == 'neural-generalization-features-v1', 'Wrong feature index kind')
    records = index.get('records', [])
    require(len({r['recordingId'] for r in records}) == len(records), 'Duplicate feature entries')
    by_id = {r['recordingId']: r for r in records}
    result = {}
    for row in rows:
        require(row['id'] in by_id, 'Missing feature entry: ' + row['id'])
        entry = by_id[row['id']]
        require(entry['sourceGroup'] == row['sourceGroup']
                and entry['contentSha256'] == row['contentSha256'], 'Feature source association differs')
        result[row['id']] = {**row.get('featureCaches', {}), **entry['features']}
    return result


def load_av(row, features):
    reference = features['audiovisual']
    path = verified(reference)
    with np.load(path, allow_pickle=False) as payload:
        times = payload['times'].astype(np.float64)
        raw = payload['values'].astype(np.float32)
        names = tuple(str(n) for n in payload['names'])
        metadata = json.loads(str(payload['metadata_json'].item())) if 'metadata_json' in payload else None
    expected = feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))
    require(names == expected and raw.shape == (len(times), 104) and np.isfinite(raw).all(), 'Invalid AV104 schema')
    require(times.ndim == 1 and len(times) and np.isfinite(times).all()
            and np.all(np.diff(times) > 0) and np.all(np.abs(np.diff(times) - .25) <= .10), 'Invalid 4Hz AV timeline')
    duration = float(metadata['duration']) if metadata is not None else float(row['durationSeconds'])
    require(abs(duration - row['durationSeconds']) < .11 and times[0] >= 0 and times[-1] <= duration,
            'AV duration/source association differs')
    if metadata is None:
        require(row['featureOrigin'] == 'native-android-dsp-v1', 'Only explicitly native caches may omit media metadata')
    values = percentile_rank_values(raw)
    for i, name in enumerate(names):
        if name in ABSOLUTE_FEATURE_NAMES:
            values[:, i] = raw[:, i]
    return times, values, duration


def attach_visual(example, features, family, precision='fp32', student_features=None):
    require(family in FAMILIES, 'Unknown feature family')
    require(precision in ('fp32', 'fp16', 'int8'), 'Unknown DINO arithmetic')
    if family == 'av':
        return example
    if family == 'dino':
        variants = features['dino']
        reference = variants if 'path' in variants and precision == 'fp32' else variants[precision]
        with np.load(verified(reference), allow_pickle=False) as payload:
            times, tokens = payload['timestamps'], payload['tokens']
        require(times.dtype == np.float64 and times.ndim == 1 and len(times)
                and np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Invalid DINO time grid')
        require(tokens.dtype == np.float16 and tokens.shape == (len(times), 10, 384)
                and np.isfinite(tokens).all(), 'Invalid DINO token geometry/storage')
        right = np.clip(np.searchsorted(times, example.times), 0, len(times)-1)
        left = np.maximum(right-1, 0)
        nearest = np.where(abs(times[left]-example.times) <= abs(times[right]-example.times), left, right)
        require(np.max(abs(times[nearest]-example.times)) <= .125+1e-8, 'DINO timeline does not cover AV ticks')
        extra = tokens[nearest].reshape(len(example.times), -1)
    else:
        if family == 'distilled':
            require(student_features is not None and example.id in student_features, 'Missing fold-specific student features')
            receipt = student_features[example.id]
            require(receipt['id'] == example.id and receipt['sourceGroup'] == example.group, 'Student source association differs')
            reference = receipt['output']
            with np.load(verified(reference), allow_pickle=False) as payload:
                times, tokens, quality, pts = (payload[k] for k in
                    ('timestamps', 'tokens', 'quality', 'selected_presentation_times'))
            require(tokens.dtype == np.float16 and tokens.shape == (len(times), 4, 576)
                    and quality.shape == (len(times), 6) and pts.shape == times.shape
                    and np.isfinite(tokens).all() and np.isfinite(quality).all() and np.isfinite(pts).all(), 'Invalid student cache')
            require(times.dtype == np.float64 and len(times) and np.isfinite(times).all()
                    and np.all(np.diff(times) > 0), 'Invalid student timestamps')
            cache = mobile.MobileVisualCache(Path(reference['path']), times, tokens.astype(np.float32), quality, pts, {})
        else:
            cache = mobile.load_mobile_visual_cache(verified(features['mobile']))
            require(cache.metadata.get('completed') is True and cache.metadata.get('partialVideo') is False
                    and cache.metadata.get('labelsUsed') is False, 'Mobile cache is incomplete or label-dependent')
            require(cache.metadata['recordingId'] == example.id, 'Mobile recording identity differs')
        aligned = mobile.align_mobile_features(cache, example.times)
        require(np.all(aligned['available'] == 1), 'Mobile cache leaves AV ticks uncovered')
        extra = np.concatenate((aligned['tokens'].reshape(len(example.times), -1), aligned['quality'],
                    aligned['feature_age_seconds'][:, None], aligned['available'][:, None]), axis=1)
    values = np.concatenate((example.values, extra), axis=1).astype(np.float32)
    require(values.shape[1] == (3944 if family == 'dino' else 2416) and np.isfinite(values).all(), 'Invalid attached visual inputs')
    return replace(example, values=values)


def example_from_row(row, features, *, inference=False):
    times, values, duration = load_av(row, features)
    truth = () if inference else expanded.intervals(row.get('rallies', []))
    ignored = () if inference else expanded.intervals(row.get('ignoredIntervals', []))
    valid = np.ones(len(times), bool) if inference else mask_for_times(times, ignored)
    targets = np.zeros((len(times), 3), np.float32)
    if not inference and row['labelTier'] == 'exact':
        targets = np.stack((base.labels_for_times(times, truth),
                            base.boundary_targets(times, [r.start for r in truth]),
                            base.boundary_targets(times, [r.end for r in truth])), axis=1)
    if not inference and row['labelTier'] == 'coverage':
        window = row['gameWindow']
        require(0 <= window['start'] < window['end'] <= duration + .001,
                'Invalid reviewed game window (allowing existing millisecond-rounded labels)')
        valid &= (times >= window['start']) & (times < window['end'])
    return base.Example(row['id'], row['sourceGroup'], duration, times, values, targets, valid,
                        truth, ignored, row['environment'])


def load_data(manifest_path, feature_index_path=None, *, role='fit', family='av',
              precision='fp32', student_features=None, recording_ids=None):
    """Load authorized supervision using the unchanged historical tier rules."""
    require(role in ('fit', 'calibrate', 'evaluate'), 'Supervision loader requires explicit supervised role')
    _, population = manifest_rows(manifest_path)
    rows = select_rows(population, role, recording_ids)
    entries = feature_entries(rows, feature_index_path)
    result = {tier: [] for tier in TIERS}
    for row in rows:
        tier = row['labelTier']
        require(tier in TIERS, 'Unscored source has no supervision')
        e = attach_visual(example_from_row(row, entries[row['id']]), entries[row['id']], family,
                          precision, student_features)
        if tier == 'exact':
            item = expanded.exact_supervision(e)
        else:
            require(row.get('annotation', {}).get('continuousVideoReviewed') is True, 'Continuous manual review absent')
            if tier == 'coverage':
                require(row.get('targetContract', {}).get('negativesOutsideKeepAuthorizedByFullManualReview') is True,
                        'Coverage negatives have not been authorized')
            item = expanded.auxiliary_supervision(e, tier, expanded.intervals(row.get('keepTargets', [])))
        result[tier].append(item)
    return result


def load_inference_examples(manifest_path, feature_index_path=None, *, family='av', precision='fp32',
                            student_features=None, recording_ids=None):
    """Sanitize labels before constructing full-timeline model inputs."""
    _, population = manifest_rows(manifest_path)
    rows = select_rows(population, 'infer', recording_ids)
    entries = feature_entries(rows, feature_index_path)
    allowed = ('id', 'sourceGroup', 'durationSeconds', 'environment', 'featureOrigin')
    return [attach_visual(example_from_row({k: row[k] for k in allowed}, entries[row['id']], inference=True),
                          entries[row['id']], family, precision, student_features) for row in rows]


def make_examples_for_evaluation(inference_examples, manifest_path, *, scoring_policy='exact-core'):
    """Attach one declared gold type while preserving full-timeline inputs/masks.

Export coverage is a distinct estimand, not exact rally core. Callers must group
and report scoring policies separately, including their boundary limitations.
"""
    require(scoring_policy in ('exact-core', 'draft-reviewed', 'export-coverage'), 'No scored labels for this policy')
    _, rows = manifest_rows(manifest_path)
    by_id = {r['id']: r for r in rows}
    result = []
    for e in inference_examples:
        require(e.id in by_id and e.valid.all() and not e.truth and not e.ignored, 'Expected sanitized full-timeline inference example')
        row = by_id[e.id]
        require(row['sourceGroup'] == e.group and row['scoringPolicy'] == scoring_policy
                and 'evaluate' in row['eligibleRoles'], 'Gold scope/policy/role differs')
        truth_key = 'keepTargets' if scoring_policy == 'export-coverage' else 'rallies'
        ignored = expanded.intervals(row.get('ignoredIntervals', []))
        if scoring_policy == 'export-coverage':
            window = row['gameWindow']
            outside = []
            if window['start'] > 0:
                outside.append({'start': 0., 'end': window['start']})
            if window['end'] < e.duration:
                outside.append({'start': window['end'], 'end': e.duration})
            ignored += expanded.intervals(outside)
        result.append(replace(e, truth=expanded.intervals(row.get(truth_key, [])), ignored=ignored))
    return result


def load_images_teachers(manifest_path, feature_index_path=None, *, recording_ids=None, for_training=True):
    """Return the old student API's image/teacher receipt maps for explicit roles.

Evaluation-only image inputs support encoder inference, but this entry point
never reads or returns teacher targets for them. Training selection is explicit
and must precede any call to the historical student fitter.
"""
    _, population = manifest_rows(manifest_path)
    rows = select_rows(population, 'fit' if for_training else 'infer', recording_ids)
    entries = feature_entries(rows, feature_index_path)
    images, teachers = {}, {}
    for row in rows:
        entry = entries[row['id']]
        image = read(verified(entry['imageInput']))
        require(image['id'] == row['id'] and image['sourceGroup'] == row['sourceGroup'], 'Image source association differs')
        source = image['contract']['source']
        require(source['contentSha256'] == row['contentSha256'], 'Image source content differs')
        for reference in image['arrays'].values():
            verified(reference)
        images[row['id']] = image
        if for_training:
            require('teaching336' in image['arrays'] and image.get('teachingFrames', 0) > 0,
                    'Training image teaching set is absent')
            teacher = read(verified(entry['teacherTargets']))
            require(teacher['id'] == row['id'] and teacher['sourceGroup'] == row['sourceGroup']
                    and teacher['input'] == image['arrays']['teaching336']
                    and teacher.get('labelsUsed') is False, 'Teacher/image lineage differs')
            verified(teacher['output'])
            verified(teacher['registration'])
            teachers[row['id']] = teacher
    return images, teachers
