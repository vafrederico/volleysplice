#!/usr/bin/env python3
"""Run a pinned research model on native feedback features without human labels.

This writes separate read-only labeling references. Model inference always sees
the full source timeline. Explicit operational ignored spans affect only adviser
work queues and export accounting. No gold, human editing, fitting, or selection.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0057'))
FIT = ROOT/'2026-09-19-short-boost-transfer/study/fits/reviewed_export/tcn/short_boost/3407/outer-0/refit'
WEIGHT_SHA = 'b386bb0834c0a3663d2fbefb89377b1300ca9437aeb0485762444747c2c13311'
DECODER = {'smoothing': 1., 'enter': .8, 'minimum': .25, 'boundary': False}
HEADS = ('live', 'serve', 'end', 'keep')


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), REPO/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(path):
    p = Path(path)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'sizeBytes': p.stat().st_size}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def numeric_array(payload, dtype, shape):
    if (payload.get('encoding') != 'base64' or payload.get('byteOrder') != 'little-endian'
            or payload.get('dataType') != dtype or tuple(payload.get('shape', ())) != shape):
        raise ValueError('Malformed encoded numeric schema')
    raw = base64.b64decode(payload['data'], validate=True)
    values = np.frombuffer(raw, dtype='<f4' if dtype == 'float32' else '<f8').copy()
    if values.size != math.prod(shape) or not np.isfinite(values).all():
        raise ValueError('Malformed numeric data')
    return values.reshape(shape)


def feature_input(feedback):
    """Whitelist native feature/source fields; all human corrections are ignored."""
    from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
    from analysis.features import FeatureSequence, VideoMetadata, feature_names
    source, features = feedback['source'], feedback['features']
    media = source['media']
    duration = float(media['duration'])
    n = int(features['rows'])
    config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
    names = tuple(features['names'])
    if (features['analysisFps'] != 4 or features['columns'] != 104
            or names != feature_names(config) or n <= 0 or not math.isfinite(duration) or duration <= 0):
        raise ValueError('Unsupported native AV104 feature signature')
    if source.get('timelineCoordinates') != 'seconds-from-start-of-source':
        raise ValueError('Feature times are not source-relative')
    times = numeric_array(features['timestamps'], 'float64', (n,))
    values = numeric_array(features['values'], 'float32', (n, 104))
    if (np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > duration
            or np.any(np.abs(np.diff(times)-.25) > .1)):
        raise ValueError('Invalid actual 4 Hz feature timeline')
    # The metadata fps is descriptive only; exact feedback timestamps drive all
    # contextualization, neural chunks, decoding, queues, and visualization.
    metadata = VideoMetadata(duration, int(media['width']), int(media['height']), 4., n, bool(media['hasAudio']))
    return FeatureSequence(times, values, names, metadata)


def validate_ignored(rows, duration):
    result = []
    for row in rows:
        start, end = float(row['start']), float(row['end'])
        if not math.isfinite(start+end) or not 0 <= start < end <= duration:
            raise ValueError('Invalid operational ignored interval')
        result.append({'start': start, 'end': end})
    return result


def run(args):
    import torch
    from analysis.config import FeatureConfig
    from analysis.features import ABSOLUTE_FEATURE_NAMES, contextualize, percentile_rank_values
    from analysis import neural_development as base
    from analysis import neural_expanded_development as expanded
    from analysis import neural_boundary_advisor as adviser
    from analysis import neural_boundary_review as review
    from analysis.transfer_temporal_model import model_for

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    if (output/'inference-registration.json').exists():
        raise FileExistsError('Never overwrite a registered inference run')
    feedback = json.loads(args.feedback.read_text())
    sequence = feature_input(feedback)
    ignored = validate_ignored(json.loads(args.ignored_json), sequence.metadata.duration)
    source = feedback['source']
    production = load_script('prepare-neural-production-comparison.py')
    input_builder = load_script('prepare-neural-rally-review-input.py')
    weights = identity(FIT/'weights-60.npz')
    if weights['sha256'] != WEIGHT_SHA:
        raise ValueError('Frozen compact checkpoint changed')
    completed = json.loads((FIT/'completed.json').read_text())
    train_ids = completed['trainIds'] + sum(completed['auxiliaryIds'].values(), [])
    if args.recording_id in train_ids:
        raise ValueError('Recording was used for fitting this research checkpoint')
    train_groups = completed['trainGroups'] + sum(completed['auxiliaryGroups'].values(), [])
    if args.source_group in train_groups:
        raise ValueError('Source group was used for fitting this research checkpoint')
    training_manifest = ROOT/'2026-09-19-short-boost-transfer/manifest-pts-v1.json'
    training_rows = json.loads(training_manifest.read_text())
    if args.content_sha256 and any(args.content_sha256 == row.get(key)
            for section in ('exactRows', 'draftRows', 'coverageRows') for row in training_rows[section]
            for key in ('contentSha256', 'sourceContentSha256')):
        raise ValueError('Source content lineage appears in original study corpus')
    runtimes = {}
    for key, (filename, expected_sha) in production.ASSETS.items():
        p = REPO/'prod/public/runtime'/filename
        runtimes[key] = identity(p)
        if runtimes[key]['sha256'] != expected_sha:
            raise ValueError('Checked-in runtime identity changed')
    source_paths = [Path(__file__), REPO/'analysis/neural_development.py',
        REPO/'analysis/neural_expanded_development.py', REPO/'analysis/transfer_temporal_model.py',
        REPO/'analysis/features.py', REPO/'analysis/neural_boundary_advisor.py',
        REPO/'analysis/neural_boundary_review.py', REPO/'analysis/neural_split_review.py',
        REPO/'scripts/prepare-neural-production-comparison.py', REPO/'scripts/prepare-neural-rally-review-input.py']
    provenance = {'createdAt': datetime.now(timezone.utc).isoformat(),
        'feedback': identity(args.feedback), 'weights': weights, 'fit': identity(FIT/'completed.json'),
        'trainingManifest': identity(training_manifest),
        'seed': 3407, 'outerFoldIndex': 0, 'epoch': 60, 'decoder': DECODER,
        'modelParameters': 29700, 'inputColumns': 104, 'heads': list(HEADS),
        'checkpointChoice': 'Predetermined existing seed3407/outer0 engineering convention; no new-video selection.',
        'checkpointStatus': 'Research outer-refit; not promoted all-development deployment model.',
        'trainingIds': train_ids, 'trainingGroups': completed['trainGroups'],
        'auxiliaryGroups': completed['auxiliaryGroups'],
        'newRecordingAbsentFromTrainingIds': True,
        'newSourceGroupAbsentFromTraining': True, 'rawContentAbsentFromStudyLineage': bool(args.content_sha256),
        'trainingPerformed': False, 'humanRalliesUsedForInference': False,
        'humanCorrectionsUsedForInference': False, 'humanOracleUsed': False,
        'inferenceValidTimeline': 'Full source; all feature ticks valid.',
        'operationalIgnoredIntervalsForAdviserOnly': ignored,
        'featureInput': 'Existing native feedback raw104 and actual float64 source-relative timestamps.',
        'nativeFeatureRuntime': source.get('runtimeVariant'), 'roi': source.get('featureRoi'),
        'sampledFingerprint': source['file'].get('sampledFingerprint'),
        'rawContentSha256': args.content_sha256,
        'policy': 'head_refined', 'ranker': 'evidence', 'budgetFraction': .1,
        'primaryPaddingSeconds': 2, 'paddingCases': [0, 1, 2, 3], 'joinGapSeconds': 3,
        'browserRuntimes': runtimes, 'sources': [identity(p) for p in source_paths]}
    write_json(output/'inference-registration.json', provenance)
    with (output/'native-features.npz').open('xb') as stream:
        np.savez_compressed(stream, times=sequence.times, values=sequence.values, names=sequence.names)

    config = FeatureConfig.from_dict(json.loads(Path(runtimes['previous']['path']).read_text())['featureConfig'])
    contextual, names = contextualize(sequence, config)
    for runtime in runtimes.values():
        if list(names) != json.loads(Path(runtime['path']).read_text())['featureNames']:
            raise ValueError('Production contextual feature signature mismatch')
    payload = {'repoUrl': production.node_repo_url(REPO), 'id': args.recording_id,
        'duration': sequence.metadata.duration, 'times': sequence.times.tolist(),
        'contextual': contextual.ravel().tolist(), 'ignoredIntervals': ignored}
    print('Replaying current checked-in production default', flush=True)
    result = subprocess.run([str(args.node), '--input-type=module', '-e', production.NODE_REPLAY],
        input=json.dumps(payload, allow_nan=False), capture_output=True, text=True, check=True)
    replay = json.loads(result.stdout)
    write_json(output/'production-replay.json', replay)
    production_row = {'durationSeconds': sequence.metadata.duration,
        'currentDefault': replay['variants']['aggressive']['core'],
        'cores': {'productionDefault': replay['variants']['aggressive']['core']}, 'productReplay': replay}
    parents = input_builder.production_events(production_row)

    raw = sequence.values
    values = percentile_rank_values(raw)
    for i, name in enumerate(sequence.names):
        if name in ABSOLUTE_FEATURE_NAMES:
            values[:, i] = raw[:, i]
    example = base.Example(args.recording_id, args.source_group, sequence.metadata.duration, sequence.times,
        values, np.zeros((len(values), 4), np.float32), np.ones(len(values), bool), (), (), 'unspecified')
    model = model_for('tcn')
    with np.load(weights['path'], allow_pickle=False) as archive:
        mean, scale = archive['mean'], archive['scale']
        state = {key[len('model::'):]: torch.from_numpy(archive[key].copy())
                 for key in archive.files if key.startswith('model::')}
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    print('Running pinned compact short-boost inference', flush=True)
    scores = expanded.predict(model, example, mean, scale, 'tcn', args.device)
    if scores.shape != (len(sequence.times), 4) or not np.isfinite(scores).all():
        raise ValueError('Malformed compact scores')
    neural = [{'id': f'compact:{i+1}', 'start': row.start, 'end': row.end}
              for i, row in enumerate(base.decode(example, scores, DECODER))]
    with (output/'compact-probabilities.npz').open('xb') as stream:
        np.savez_compressed(stream, times=sequence.times, scores=scores)
    write_json(output/'compact-events.json', {'recordingId': args.recording_id, 'events': neural, 'decoder': DECODER})

    record = {'id': args.recording_id, 'durationSeconds': sequence.metadata.duration,
        'sourceGroup': args.source_group, 'ignoredIntervals': ignored, 'rallies': [], 'productionEvents': parents}
    plan = adviser.plan(record, parents, neural, sequence.times, scores, 'head_refined')
    jobs = review.jobs(record, plan)
    queue = review.queues(record, jobs, 'evidence', (.1,))[0]
    write_json(output/'boundary-adviser.json', {'record': record, 'plan': plan, 'jobs': jobs, 'queue': queue})
    chosen = set(queue['selectedParentIds'])
    reasons = {'initial_start': 'Check rally start', 'additional_start': 'Check additional rally / split', 'end': 'Check rally end'}
    regions = [{k: row[k] for k in ('id', 'parentId', 'start', 'end', 'priority')} | {
        'recommended': row['parentId'] in chosen,
        'reasons': sorted({reasons[p['kind']] for p in plan['proposals'] if p['parentId'] == row['parentId']})}
        for row in jobs]
    flags = [{k: flag[k] for k in ('id', 'kind', 'time', 'parentId', 'priority')} for flag in plan['proposals']]
    research = {'recommendation': 'Keep production export. Review recommended whole rallies; use compact boundaries as guidance. Serve signal is a rally-start cue, not serving side or a validated serve contact.',
        'signals': {'times': sequence.times.tolist(), **{head: scores[:, i].tolist() for i, head in enumerate(HEADS)}},
        'boundaryFlags': flags, 'reviewRegions': regions,
        'queue': {'budgetFraction': .1, 'reviewSeconds': queue['reviewSeconds'], 'selectedParentCount': len(chosen)},
        'provenance': provenance}
    def rallies(rows):
        return [{'start': r['start'], 'end': r['end'], 'tags': ['ai-reference', 'research-preview'],
                 'notes': 'Read-only model proposal; not human ground truth.'} for r in rows]
    exported = rallies(parents)
    references = [
        {'modelId': 'production-compact-head-refined-review-3407-outer0',
         'modelLabel': 'Production + compact review',
         'description': 'Current checked-in aggressive whole-rally production baseline, with compact review flags. Export remains production.',
         'rallies': exported, 'exportRallies': exported, 'exportPolicy': 'fixed-production', 'research': research},
        {'modelId': 'compact-head-refined-boundary-preview-3407-outer0',
         'modelLabel': 'Proposed boundary preview',
         'description': 'Unconfirmed compact rally boundary suggestions; provisional clipped starts are not validated serves. Export remains production.',
         'rallies': rallies(plan['events']), 'exportRallies': exported, 'exportPolicy': 'fixed-production'},
    ]
    row = {'recordingId': args.recording_id, 'videoFilename': source['file']['name'],
           'durationSeconds': sequence.metadata.duration, 'references': references}
    # The labeling corpus may identify a feedback source differently from raw
    # media. Raw SHA is preserved in provenance; filename/duration/id bind UI.
    manifest = {'schemaVersion': 1, 'kind': 'volleycut-labeling-research-references', 'recordings': [row]}
    write_json(output/'research-references.json', manifest)
    original = feedback.get('initialInference', {}).get('ranges', [])
    try:
        parity = {'maximumEndpointErrorSeconds': production.verify_endpoints(replay['union'], original), 'equal': True}
    except ValueError as error:
        parity = {'equal': False, 'reason': str(error)}
    write_json(output/'original-feedback-production-context.json', {'recordingId': args.recording_id,
        'originalUnsuppressedRanges': original, 'currentReplayUnsuppressedParity': parity,
        'usedForInferenceOrAdviser': False})
    receipt = {'recordingId': args.recording_id, 'durationSeconds': sequence.metadata.duration,
        'featureRows': len(sequence.times), 'featureColumns': 104, 'compactRallies': len(neural),
        'productionUnionRallies': len(replay['union']), 'productionDefaultRallies': len(parents),
        'proposedTimelineRallies': len(plan['events']), 'boundaryFlags': len(flags),
        'availableReviewParents': len(jobs), 'recommendedReviewParents': len(chosen),
        'reviewSeconds': queue['reviewSeconds'], 'reviewBudgetSeconds': queue['budgetSeconds'],
        'sourceTimelinePreserved': True, 'humanLabelsUsed': False, 'trainingPerformed': False,
        'originalFeedbackParity': parity, 'manifest': identity(output/'research-references.json')}
    for source_identity in provenance['sources']:
        if identity(source_identity['path']) != source_identity:
            raise ValueError('Inference source changed during the run')
    write_json(output/'inference-receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback', type=Path, required=True)
    parser.add_argument('--recording-id', required=True)
    parser.add_argument('--source-group', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--content-sha256')
    parser.add_argument('--ignored-json', default='[]')
    parser.add_argument('--node', type=Path, default=Path('/mnt/c/Program Files/nodejs/node.exe'))
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
