"""NAS-only shared frame staging for registered generalization inputs.

New videos use one sequential presentation-time decoder for AV104, DINO336,
and Mobile224. Existing legacy DINO inputs retain their registered sampler.
No rally labels enter this module; only an optional training validity mask is
used to choose a bounded teacher-image subset.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from . import features as av
from . import mobile_visual_features as mobile
from . import dinov2_embeddings as dino
from .config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
from .distillation_image_inputs import uint8_from_normalized, teaching_indexes
from .neural_context_development import identity, read, write_immutable
from .neural_generalization_inputs import require, verified
from .schema import mask_for_times, Interval

REPO = Path(__file__).resolve().parents[1]
NAS = Path(private_value('private-reference-0057'))
ROOT = NAS/'2026-09-23-recall-sweep-generalization/features-v1'


def environment(root):
    root = Path(root).resolve()
    require(root.is_relative_to(NAS) and Path('/mnt/freenas').is_mount(), 'Artifacts must use the direct NAS mount')
    root.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        path = root/'runtime'/key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return root


def npz_new(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def array_sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def verify_plan(path):
    plan = read(path)
    require(plan.get('kind') == 'generalization-label-blind-feature-plan-v1'
            and plan.get('labelsUsedForFeatures') is False, 'Wrong feature plan')
    for reference in plan['code'].values():
        verified(reference)
    for reference in plan['parents']:
        verified(reference)
    require(len({r['id'] for r in plan['records']}) == len(plan['records']), 'Duplicate feature source')
    return plan


def pts_selection(pts, duration, frequency):
    return mobile.sample_selection(pts, duration, frequency)


def audio_on_grid(video, metadata, times, config):
    pcm, available = av._decode_audio_samples(video, metadata, config.audio_sample_rate)
    origin, shift = 0., 0
    if available:
        command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                   'stream=start_time,sample_rate', '-of', 'json', str(video)]
        rows = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)['streams']
        require(rows, 'Audio decoder and stream inventory disagree')
        origin = float(rows[0].get('start_time', 0.))
        require(np.isfinite(origin) and abs(origin) <= .25, 'Audio origin needs separate qualification')
        command = ['ffprobe', '-v', 'error', '-read_intervals', '%+0.2', '-select_streams', 'a:0',
                   '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)]
        frames = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)['frames']
        require(frames and abs(float(frames[0]['best_effort_timestamp_time'])-origin)
                <= 1/float(rows[0]['sample_rate']), 'Decoded audio origin differs from stream')
        shift = int(round(origin*config.audio_sample_rate))
        if shift > 0:
            pcm = np.concatenate((np.zeros(shift, np.float32), pcm))
        elif shift < 0:
            pcm = pcm[-shift:]
    values = av._audio_features_from_samples(np.ascontiguousarray(pcm, np.float32), times, config, available=available)
    return values, {'available': available, 'originSeconds': origin, 'alignmentSamples': shift,
                    'sampleRate': config.audio_sample_rate, 'formulaUnchanged': True}


def stage_record(plan_path, recording_id, *, root=ROOT):
    import cv2
    root = environment(root)
    plan = verify_plan(plan_path)
    source = next(r for r in plan['records'] if r['id'] == recording_id)
    require(source['environment'] != 'beach', 'Beach is outside this plan')
    folder = root/'staged'/recording_id
    receipt_path = folder/'receipt.json'
    lineage = {'plan': identity(plan_path), 'source': source}
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['lineage'] == lineage, 'Staging resume differs')
        for reference in receipt['outputs'].values():
            verified(reference)
        for reference in receipt.get('dinoRgbChunks', []):
            verified(reference)
        return receipt
    reuse = source.get('reuse', {})
    dino_needed = not all(arm in reuse.get('dino', {}) for arm in ('fp32', 'fp16', 'int8'))
    mobile_needed = 'imageInput' not in reuse or source.get('engineeringReplay', False)
    av_needed = 'audiovisual' not in reuse or source.get('engineeringReplay', False)
    if not (dino_needed or mobile_needed or av_needed):
        receipt = {'lineage': lineage, 'id': recording_id, 'sourceGroup': source['sourceGroup'],
                   'outputs': {}, 'dinoRgbChunks': [], 'allFeaturesReused': True, 'labelsUsed': False}
        write_immutable(receipt_path, receipt)
        return receipt
    require(not folder.exists(), 'Incomplete staging directory must be audited before resuming: '+str(folder))
    folder.mkdir(parents=True)
    video = verified(source['videoIdentity'])
    before = video.stat()
    cv2.setNumThreads(1)
    metadata = av.probe_video(video)
    roi = mobile.normalize_roi(source.get('roi'))
    sampler = source['sampling']
    if sampler == 'media-pts':
        pts, inventory = mobile.presentation_inventory(video)
        duration = float(inventory['stream']['duration'])
        times, ordinals = pts_selection(pts, duration, 4.)
        require(np.max(abs(pts[ordinals]-times)) <= .125+1e-9, '4Hz media PTS selection error exceeds half tick')
        metadata = av.VideoMetadata(duration, metadata.width, metadata.height, metadata.fps, len(pts), metadata.has_audio)
    else:
        require(sampler == 'historical-ordinal' and not av_needed and not mobile_needed,
                'Historical ordinal mode only reproduces missing DINO precision inputs')
        with np.load(verified(reuse['dino']['fp32']), allow_pickle=False) as payload:
            times = payload['timestamps']
        duration = metadata.duration
        ordinals = np.clip(np.rint(times*metadata.fps).astype(np.int64), 0, metadata.frame_count-1)
        pts = None
    require(abs(duration-source['durationSeconds']) < .11, 'Staging media duration differs from source')
    teach_allowed = source['teaching']['allowed']
    require(not teach_allowed or not source['protected'], 'Protected teacher inputs forbidden')
    if mobile_needed:
        mobile_times, mobile_ordinals = pts_selection(pts, duration, 2.)
        require(np.array_equal(mobile_times, times[::2]) and np.array_equal(mobile_ordinals, ordinals[::2]),
                'The 2Hz image selection is not the even4Hz subgrid')
        image_path = folder/'images224.npy'
        images = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8,
                                           shape=(len(mobile_times), 3, 224, 224))
        boxes = np.empty((len(mobile_times), 4), np.float64)
        quality = np.empty((len(mobile_times), 6), np.float32)
        if teach_allowed:
            teaching = source['teaching']
            ignored = tuple(Interval(r['start'], r['end']) for r in teaching['excludedIntervals'])
            valid = mask_for_times(times, ignored)
            window = teaching.get('gameWindow')
            if window:
                valid &= (times >= window['start']) & (times < window['end'])
            selected_teaching = teaching_indexes(mobile_times, times, valid, maximum=128)
            teachers = np.lib.format.open_memmap(folder/'teaching336.npy', mode='w+', dtype=np.uint8,
                shape=(len(selected_teaching), 3, 336, 336))
            teach_dest = {int(index): i for i, index in enumerate(selected_teaching)}
        else:
            selected_teaching = np.asarray([], np.int64)
            teachers, teach_dest = None, {}
    else:
        images = teachers = None
    config = FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET)
    visual, previous, actual_pts, frame_hashes, rgb_buffer, rgb_refs = [], None, [], [], [], []
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), 'Cannot open staged video')
    started = time.perf_counter()
    cursor, ordinal = 0, -1
    try:
        while capture.grab():
            ordinal += 1
            if cursor >= len(ordinals) or ordinal != ordinals[cursor]:
                continue
            ok, frame = capture.retrieve()
            require(ok and frame is not None, 'Selected frame could not decode')
            actual = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            if pts is not None:
                require(abs(actual-pts[ordinal]) <= 2e-6, 'OpenCV presentation time differs from ffprobe')
            actual_pts.append(actual)
            frame_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            if av_needed:
                values, previous = av._frame_features(av._crop_roi(frame, roi), previous, config)
                visual.append(values)
            rgb = dino.letterbox_frame(dino._crop_frame(frame, roi), 336)
            if dino_needed:
                rgb_buffer.append(rgb)
                if len(rgb_buffer) == 128 or cursor+1 == len(times):
                    chunk = folder/'dino-rgb'/f'{cursor+1-len(rgb_buffer):06d}.npy'
                    chunk.parent.mkdir(exist_ok=True)
                    with chunk.open('xb') as stream:
                        np.save(stream, np.stack(rgb_buffer), allow_pickle=False)
                    rgb_refs.append(identity(chunk))
                    rgb_buffer = []
            if mobile_needed and cursor % 2 == 0:
                mobile_index = cursor//2
                values, box, current_quality = mobile.preprocess_frame(frame, source.get('roi'))
                images[mobile_index] = uint8_from_normalized(values, mobile.RGB_MEAN, mobile.RGB_STD)
                boxes[mobile_index], quality[mobile_index] = box, current_quality
                quality[mobile_index, -1] = actual-mobile_times[mobile_index]
                if mobile_index in teach_dest:
                    teachers[teach_dest[mobile_index]] = rgb.transpose(2, 0, 1)
            cursor += 1
            if cursor % 1024 == 0 or cursor == len(times):
                print(json.dumps({'staged': recording_id, 'frames': cursor, 'total': len(times)}), flush=True)
        require(cursor == len(times), 'Video ended before every selected frame')
        if pts is not None:
            require(ordinal+1 == len(pts), 'Decoded full-video frame count differs')
    finally:
        capture.release()
        if images is not None:
            images.flush()
        if teachers is not None:
            teachers.flush()
    del images, teachers
    outputs = {}
    selection_path = folder/'selection.npz'
    npz_new(selection_path, times=times, ordinals=ordinals,
            selected_pts=np.asarray(actual_pts, np.float64), frame_sha256=np.asarray(frame_hashes))
    outputs['selection'] = identity(selection_path)
    if av_needed:
        matrix = np.vstack(visual).astype(np.float32, copy=False)
        temporal = av._temporal_visual_features(matrix, av._frame_feature_names(config), config)
        audio, audio_info = audio_on_grid(video, metadata, times, config)
        values = np.concatenate((matrix, temporal, audio), axis=1).astype(np.float32)
        require(values.shape == (len(times), 104) and np.isfinite(values).all(), 'Invalid staged AV104 matrix')
        path = folder/'audiovisual.npz'
        npz_new(path, times=times, values=values, names=np.asarray(av.feature_names(config)),
                metadata_json=np.asarray(json.dumps(asdict(metadata), sort_keys=True)),
                video_decoder=np.asarray('sequential-opencv-nearest-media-pts-v1'))
        outputs['audiovisual'] = identity(path)
    else:
        audio_info = None
    if mobile_needed:
        timing = folder/'timing.npz'
        npz_new(timing, times=mobile_times, selected_pts=np.asarray(actual_pts, np.float64)[::2],
                selected_ordinals=ordinals[::2], selected_frame_sha256=np.asarray(frame_hashes)[::2],
                quality=quality, boxes=boxes, teaching_indexes=selected_teaching)
        arrays = {'images224': identity(image_path), 'timing': identity(timing)}
        if teach_allowed:
            arrays['teaching336'] = identity(folder/'teaching336.npy')
        image_receipt = {'id': recording_id, 'sourceGroup': source['sourceGroup'],
            'contract': {'source': {'id': recording_id, 'contentSha256': source['videoIdentity']['sha256']},
                         'teachingIndexes': selected_teaching.tolist(), 'plan': identity(plan_path)},
            'arrays': arrays, 'frames': len(mobile_times), 'teachingFrames': len(selected_teaching),
            'labelsUsed': False, 'ignoredMaskUsedForTeachingSelectionOnly': True,
            'teachingAllowed': teach_allowed}
        write_immutable(folder/'image-input.json', image_receipt)
        outputs['imageInput'] = identity(folder/'image-input.json')
    after = video.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed during staging')
    receipt = {'lineage': lineage, 'id': recording_id, 'sourceGroup': source['sourceGroup'],
        'outputs': outputs, 'dinoRgbChunks': rgb_refs, 'sampling': sampler,
        'samples': len(times), 'audio': audio_info, 'labelsUsed': False,
        'allRequestedFramesDecoded': True, 'wallSeconds': time.perf_counter()-started}
    write_immutable(receipt_path, receipt)
    return receipt
