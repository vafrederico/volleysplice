#!/usr/bin/env python3
"""Predetermined first-eight-sample content parity for seven repaired Pixel pairs.

Prepare/pin this script before reading any cached feature values. Run only after
all extraction workers have exited. No labels, model outcomes or training.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(private_value('private-reference-0084'))
REPO = Path(__file__).resolve().parents[1]
TICKS = np.arange(8, dtype=np.float64)/4
GLOBAL_AUDIO = frozenset(('audio_onset_strength', 'audio_contact_like_transient',
                          'audio_onset_cadence', 'audio_cadence_collapse', 'audio_seconds_since_transient'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            sha.update(block)
    return {'path': str(path.resolve()), 'sha256': sha.hexdigest(), 'sizeBytes': path.stat().st_size}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_sources(plan):
    for value in plan['sourceCode'].values():
        require(identity(value['path']) == value, 'Frozen extraction source changed')


def prepare(root, output, repo):
    require(not output.exists(), 'Content audit leaf already exists')
    repair_path = root/'pts-repair-plan-v1.json'
    repair = read(repair_path)
    require(len(repair['recordingIds']) == len(set(repair['recordingIds'])) == 7, 'Expected seven repaired recordings')
    verify_sources(repair)
    script = identity(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/Path(__file__).name
    with snapshot.open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    require(identity(snapshot)['sha256'] == script['sha256'], 'Content auditor snapshot differs')
    plan = {'kind': 'repaired-feature-content-parity-plan-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'script': script, 'sourceSnapshot': identity(snapshot), 'repository': str(repo.resolve()),
            'repairPlan': identity(repair_path), 'originalExtractionPlan': identity(root/'extraction-plan.json'),
            'recordingIds': repair['recordingIds'], 'ticksPerRecord': 8, 'gridTicksSeconds': TICKS.tolist(),
            'selection': 'First eight media grid ticks, fixed before cached feature values are read; independent sequential observed-PTS nearest selection, earlier ties.',
            'dino': 'One frozen FP32 batch of eight, original336 preprocessing and token pooling, then float16; exact stored-token parity.',
            'visual': 'Original frame-feature functions, including previous sampled frame for optical flow; exclude four temporal lookahead features.',
            'audio': 'Decode only first2.25s to mono16k signed PCM, shift by independently verified first decoded audio PTS, compare causal/non-global-rank audio channels.',
            'excludedAudioChannels': sorted(GLOBAL_AUDIO), 'gpuWorkers': 1,
            'inputsReadBeforePin': 'Repair/extraction plans and source code identities only; no cached feature values, labels or model outcomes.'}
    write_new(output/'plan.json', plan)
    print(json.dumps({'prepared': identity(output/'plan.json'), 'script': script}), flush=True)


def nearest_from_sequential(capture, cv2, targets=TICKS):
    """Keep only two decoder frames and eight selected copies, never seek."""
    rows, cursor, ordinal, previous = [], 0, -1, None
    try:
        while cursor < len(targets):
            ok, frame = capture.read()
            ordinal += 1
            require(ok and frame is not None and frame.size, 'Video ended during fixed prefix decode')
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            require(np.isfinite(pts) and (previous is None or pts > previous[1]), 'Decoded prefix PTS not strictly increasing')
            if previous is None:
                require(abs(pts) <= 1e-6, 'Unexpected video origin')
            current = (ordinal, pts, frame)
            while cursor < len(targets) and pts >= targets[cursor]:
                chosen = (previous if previous is not None and targets[cursor]-previous[1] <= pts-targets[cursor] else current)
                require(abs(chosen[1]-targets[cursor]) <= .125+1e-9, 'Nearest prefix frame exceeds half-tick error')
                rows.append((chosen[0], chosen[1], chosen[2].copy()))
                cursor += 1
            previous = current
    finally:
        capture.release()
    return rows, ordinal+1


def compare_arrays(actual, expected, label):
    exact = actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes()
    error = float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))) if actual.shape == expected.shape else None
    require(exact, f'{label} is not bit exact; maximum absolute error={error}')
    return {'bitExact': True, 'shape': list(actual.shape), 'dtype': str(actual.dtype), 'maximumAbsoluteError': error}


def prefix_audio(path, selection, config):
    command = ['ffprobe', '-v', 'error', '-read_intervals', '%+0.2', '-select_streams', 'a:0',
               '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time,nb_samples', '-of', 'json', str(path)]
    frames = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)['frames']
    start = float(frames[0]['best_effort_timestamp_time'])
    audio = selection['audio']
    require(abs(start-audio['startOffsetSeconds']) <= 1/float(audio['stream']['sample_rate']), 'Audio origin differs from extraction')
    shift = int(round(start*config.audio_sample_rate))
    require(shift == audio['alignmentSamples'], 'Audio sample alignment differs')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-vn',
               '-ac', '1', '-ar', str(config.audio_sample_rate), '-t', '2.25', '-f', 's16le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype='<i2').astype(np.float32)
    samples *= np.float32(1/32768)
    samples = np.pad(samples, (shift, 0)) if shift > 0 else samples[-shift:] if shift < 0 else samples
    require(len(samples) >= 2*config.audio_sample_rate, 'Audio prefix too short for fixed pooling windows')
    return samples, {'startPTSseconds': start, 'shiftSamples': shift, 'decodedPrefixLimitSeconds': 2.25,
                     'samplesAfterAlignment': len(samples), 'command': command}


def verify_runtime(expected, cv2):
    actual = {'opencv': cv2.__version__,
              'opencvBuildSha256': hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
              'ffprobe': subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0],
              'ffmpeg': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0],
              'python': platform.python_version(), 'numpy': np.__version__}
    require(all(expected[key] == value for key, value in actual.items()), 'Content audit decoder runtime differs')
    return actual


def run(root, output, repo):
    report_path = output/'content-audit.json'
    require(not report_path.exists(), 'Refusing to replace completed content audit')
    plan_path = output/'plan.json'
    plan = read(plan_path)
    require(identity(Path(__file__)) == plan['script'] and identity(plan['sourceSnapshot']['path']) == plan['sourceSnapshot'], 'Pinned content auditor changed')
    require(plan['ticksPerRecord'] == 8 and plan['gridTicksSeconds'] == TICKS.tolist()
            and plan['repository'] == str(repo.resolve()), 'Predetermined content scope changed')
    require(identity(root/'pts-repair-plan-v1.json') == plan['repairPlan']
            and identity(root/'extraction-plan.json') == plan['originalExtractionPlan'], 'Pinned extraction plans changed')
    repair, original = read(plan['repairPlan']['path']), read(plan['originalExtractionPlan']['path'])
    require(plan['recordingIds'] == repair['recordingIds'], 'Content recording scope changed')
    verify_sources(repair)
    # Fail before loading feature values or initializing CUDA if any worker has
    # not published its final completion record.
    for rid in plan['recordingIds']:
        require((root/'pts-records-v1'/rid/'record-audit.json').is_file(), 'Wait until all seven repaired extractions complete')
    sys.path.insert(0, str(repo))
    import cv2
    import torch
    from analysis import features
    from analysis.config import FeatureConfig
    from analysis.dinov2_embeddings import load_pinned_dinov2
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    report = {'kind': 'repaired-feature-content-parity-v1', 'passed': False,
              'createdAt': datetime.now(timezone.utc).isoformat(), 'script': identity(Path(__file__)),
              'plan': identity(plan_path), 'planSha256': identity(plan_path)['sha256'], 'repairPlan': plan['repairPlan'],
              'ticksPerRecord': 8, 'gridTicksSeconds': TICKS.tolist(), 'records': [],
              'limitations': ['Only first eight grid ticks0..1.75s per recording are reproduced; no whole-record feature-content claim.',
                  'Only the prefix is sequentially decoded. Separate decoder-audit-v1 covers late Pixel2047 ordinal/timing evidence.',
                  'Four temporal visual features are excluded because their lookahead extends beyond the reproduced prefix.',
                  'Five onset/rank-derived audio channels are excluded because they depend on whole-record ranks; no full audio re-extraction.',
                  'Source full-file SHA comes from the completed extraction; this check verifies its unchanged size and modification time, not a second full-video hash.',
                  'No labels, temporal-model predictions, decoder outcomes, selection or training are read or computed.']}
    started = time.perf_counter()
    try:
        report['decoderRuntime'] = verify_runtime(repair['decoderRuntime'], cv2)
        repository = Path(original['modelRepositoryPath'])
        commit = subprocess.run(['git', '-C', str(repository), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True).stdout
        require(commit == original['modelRepositoryCommit'] and not dirty.strip(), 'Frozen DINO repository changed')
        require(importlib.util.find_spec('xformers') is None, 'Frozen DINO fallback backend changed')
        backbone = load_pinned_dinov2(repository, repository_commit=commit,
            checkpoint=original['checkpoint']['path'], checkpoint_sha256=original['checkpoint']['sha256'], device='cuda')
        for rid in plan['recordingIds']:
            began = time.perf_counter()
            original_record = next(row for row in original['records'] if row['recordingId'] == rid)
            leaf = root/'pts-records-v1'/rid
            completed_path = leaf/'record-audit.json'
            completed = read(completed_path)
            record, av = completed['record'], completed['audiovisual']
            require(completed['passed'] is True and record['recordingId'] == rid, 'Invalid repair completion record')
            selection_path = Path(record['decoderSelectionPath'])
            require(identity(selection_path)['sha256'] == record['decoderSelectionSha256'], 'Decoder selection changed')
            selection = read(selection_path)
            source = selection['source']
            require(source['path'] == original_record['sourceVideoPath']
                    and source['sha256'] == original_record['recordingContentSha256'], 'Original source identity differs')
            video = Path(source['path'])
            stat = video.stat()
            require(stat.st_size == source['sizeBytes'] and stat.st_mtime_ns == source['mtimeNs'], 'Video changed after extraction')
            require(selection['recordingId'] == rid and selection['repairPlan'] == plan['repairPlan'], 'Selection provenance differs')
            arrays_path = Path(selection['arrays']['path'])
            require(identity(arrays_path)['sha256'] == selection['arrays']['sha256'], 'Selection arrays changed')
            with np.load(arrays_path, allow_pickle=False) as arrays:
                require(np.array_equal(arrays['times'][:8], TICKS), 'Stored selection grid differs')
                expected_indexes = arrays['selectedOrdinals'][:8].copy()
                expected_pts = arrays['selectedPresentationTimes'][:8].copy()
                expected_hashes = arrays['selectedFrameSha256'][:8].copy()
            capture = cv2.VideoCapture(str(video))
            require(capture.isOpened(), 'Cannot open repaired source video')
            selected, decoded_frames = nearest_from_sequential(capture, cv2)
            indexes, pts, frames = [r[0] for r in selected], [r[1] for r in selected], [r[2] for r in selected]
            hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
            require(indexes == expected_indexes.tolist() and np.max(np.abs(np.asarray(pts)-expected_pts)) <= 2e-6
                    and hashes == expected_hashes.tolist(), 'Independent prefix frame identity differs')
            require(identity(av['path'])['sha256'] == av['sha256']
                    and identity(record['dinoPath'])['sha256'] == record['dinoSha256'], 'Completed feature cache changed')
            with np.load(av['path'], allow_pickle=False) as cache:
                require(np.array_equal(cache['times'][:8], TICKS), 'AV grid differs')
                stored_av = cache['values'][:8].copy()
                names = tuple(str(v) for v in cache['names'])
            with np.load(record['dinoPath'], allow_pickle=False) as cache:
                require(np.array_equal(cache['timestamps'][:8], TICKS), 'DINO grid differs')
                stored_tokens = cache['tokens'][:8].copy()
                metadata = json.loads(str(cache['metadata_json'].item()))
            config = FeatureConfig.from_dict(av['config'])
            require(av['config'] == original_record['audiovisual']['config'], 'Original AV config changed')
            require(names == features.feature_names(config) and len(names) == 104, 'AV formula schema changed')
            roi = tuple(metadata['roi']) if metadata['roi'] is not None else None
            require(metadata['roi'] == original_record['roi'] and metadata['preprocessing'] == original['extractorConfig'], 'Frozen DINO ROI/preprocessing differs')
            require(metadata['backbone'] == backbone.identity(), 'DINO backbone identity differs')
            previous, visual = None, []
            for frame in frames:
                values, previous = features._frame_features(features._crop_roi(frame, roi), previous, config)
                visual.append(values)
            visual_names = features._frame_feature_names(config)
            visual_check = compare_arrays(np.stack(visual).astype(np.float32),
                stored_av[:, [names.index(n) for n in visual_names]], rid+'/core-visual')
            tokens = backbone.embed_frames(frames, (roi,)*8, input_size=336).astype(np.float16)
            dino_check = compare_arrays(tokens, stored_tokens, rid+'/DINO-float16')
            samples, audio_origin = prefix_audio(video, selection, config)
            audio_values = features._audio_features_from_samples(samples, TICKS, config, available=True)
            audio_names = features._audio_feature_names(config)
            checked_audio = [name for name in audio_names if name not in GLOBAL_AUDIO]
            require(GLOBAL_AUDIO <= set(audio_names), 'Excluded audio schema differs')
            audio_check = compare_arrays(audio_values[:, [audio_names.index(name) for name in checked_audio]],
                stored_av[:, [names.index(name) for name in checked_audio]], rid+'/causal-audio')
            require(video.stat().st_size == stat.st_size and video.stat().st_mtime_ns == stat.st_mtime_ns, 'Video changed during content audit')
            report['records'].append({'recordingId': rid, 'passed': True, 'recordAudit': identity(completed_path),
                'source': source, 'selection': identity(selection_path), 'AV': identity(av['path']), 'DINO': identity(record['dinoPath']),
                'decodedPrefixFrames': decoded_frames, 'selectedOrdinals': indexes, 'selectedPTSseconds': pts,
                'selectedFrameBgrSha256': hashes, 'framesExactlyEqual': True,
                'visual': {**visual_check, 'columns': list(visual_names)}, 'dino': dino_check,
                'audio': {**audio_check, 'columns': checked_audio, 'origin': audio_origin},
                'excludedAudioColumns': sorted(GLOBAL_AUDIO),
                'excludedVisualColumns': list(features._temporal_visual_feature_names(config)),
                'wallSeconds': time.perf_counter()-began})
            print(json.dumps({'contentParityPassed': rid, 'wallSeconds': report['records'][-1]['wallSeconds']}), flush=True)
        verify_sources(repair)
        require(identity(Path(__file__)) == plan['script'] and identity(plan_path) == report['plan'], 'Audit source/plan changed during run')
        require(len(report['records']) == 7 and [r['recordingId'] for r in report['records']] == plan['recordingIds'], 'Incomplete seven-record content audit')
        report['passed'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['completedAt'] = datetime.now(timezone.utc).isoformat()
        report['wallSeconds'] = time.perf_counter()-started
        write_new(report_path, report)
        print(json.dumps({'passed': report['passed'], 'artifact': identity(report_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repo', type=Path, default=REPO)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    output = args.output or args.root/'decoder-audit/repair-content-v1'
    action = prepare if args.prepare_only else run
    action(args.root.resolve(), output.resolve(), args.repo.resolve())


if __name__ == '__main__':
    main()
