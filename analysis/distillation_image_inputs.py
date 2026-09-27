"""Immutable NAS image inputs for source-isolated DINO-to-MobileNet research.

Rally targets never enter pixel preprocessing or teacher targets. The validity
mask only excludes ignored time when choosing a fixed unlabeled teaching set.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import mobile_visual_features as mobile
from . import dinov2_embeddings as dino
from .neural_context_development import identity, read, write_immutable

ROOT = Path(private_value('private-reference-0057'))
PREVIOUS = ROOT/'2026-09-22-recognition'
OUTPUT = ROOT/'2026-09-23-recall-distillation'
SOURCE = ROOT/'2026-09-19-short-boost-transfer/manifest-pts-v1.json'
EXTRACTION = PREVIOUS/'inputs/extraction-manifest.json'
MOBILE_INDEX = PREVIOUS/'mobile-features-v1/training-index.json'
COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
ASSETS = Path(private_value('private-reference-0058'))


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(reference):
    path = Path(reference['path'])
    require(mobile.sha256_file(path) == reference['sha256'], f'Artifact changed: {path}')
    return path


def nearest_indexes(source, target):
    source, target = np.asarray(source), np.asarray(target)
    right = np.minimum(np.searchsorted(source, target), len(source)-1)
    left = np.maximum(right-1, 0)
    return np.where(abs(source[left]-target) <= abs(source[right]-target), left, right)


def teaching_indexes(times, valid_times, valid, maximum=128):
    """Evenly spaced valid frame indexes; no rally labels or score selection."""
    indexes = nearest_indexes(valid_times, times)
    require(np.max(abs(np.asarray(valid_times)[indexes]-times)) <= .251, 'Valid grid differs')
    allowed = np.flatnonzero(np.asarray(valid, bool)[indexes])
    require(len(allowed) > 0 and maximum > 0, 'No usable teaching frames')
    return allowed[np.linspace(0, len(allowed)-1, min(maximum, len(allowed)), dtype=np.int64)]


def uint8_from_normalized(values, mean, std):
    canvas = np.rint((values.transpose(1, 2, 0) * std + mean) * 255.)
    require(np.all((canvas >= 0) & (canvas <= 255)), 'Reconstructed pixels outside uint8')
    result = np.ascontiguousarray(canvas.astype(np.uint8).transpose(2, 0, 1))
    replay = (result.transpose(1, 2, 0).astype(np.float32)/255. - mean)/std
    require(np.array_equal(replay.transpose(2, 0, 1), values), 'Pixel roundtrip changes preprocessing')
    return result


def source_contract():
    manifest = read(EXTRACTION)
    require(manifest['sourceManifest'] == identity(SOURCE), 'Source manifest changed')
    require(manifest['labelsUsed'] is False and manifest['beachIncluded'] is False
            and manifest['protectedTestOpened'] is False and len(manifest['records']) == 18,
            'Sanitized development manifest required')
    return manifest


def prepare_record(record, entry, example, directory):
    folder = directory/record['id']
    receipt_path = folder/'receipt.json'
    original_path = Path(entry['cachePath'])
    require(mobile.sha256_file(original_path) == entry['cacheSha256'], 'Prior mobile cache changed')
    cache = mobile.load_mobile_visual_cache(original_path)
    teach = teaching_indexes(cache.timestamps, example.times, example.valid)
    contract = {'source': record, 'originalMobileCache': identity(original_path),
                'teachingIndexes': teach.tolist(), 'validMaskSha256': hashlib.sha256(example.valid.tobytes()).hexdigest(),
                'code': {name: identity(Path(__file__).with_name(name)) for name in
                         ('distillation_image_inputs.py', 'mobile_visual_features.py', 'dinov2_embeddings.py')}}
    if receipt_path.exists():
        old = read(receipt_path)
        require(old['contract'] == contract, 'Image preparation resume contract differs')
        for reference in old['arrays'].values():
            verify(reference)
        return old
    require(not folder.exists(), f'Incomplete image directory: {folder}')
    folder.mkdir(parents=True)
    import cv2
    cv2.setNumThreads(1)
    video = Path(record['video'])
    require(mobile.sha256_file(video) == record['contentSha256'], 'Source video changed')
    before = video.stat()
    pts, inventory = mobile.presentation_inventory(video)
    times, selected = mobile.sample_selection(pts, float(inventory['stream']['duration']))
    require(np.array_equal(times, cache.timestamps), 'Mobile nominal times differ')
    with np.load(original_path, allow_pickle=False) as previous:
        require(np.array_equal(selected, previous['selected_ordinals']), 'Mobile selected frames differ')
        hashes = previous['selected_frame_sha256']
    image_path, teacher_path = folder/'images224.npy', folder/'teaching336.npy'
    images = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8, shape=(len(times), 3, 224, 224))
    teachers = np.lib.format.open_memmap(teacher_path, mode='w+', dtype=np.uint8, shape=(len(teach), 3, 336, 336))
    boxes = np.empty((len(times), 4), np.float64)
    chosen = {int(index): i for i, index in enumerate(teach)}
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), 'Could not open source')
    start = time.perf_counter()
    try:
        for i, (_, frame, actual) in enumerate(mobile.selected_frames(capture, selected, pts, cv2.CAP_PROP_POS_MSEC)):
            require(hashlib.sha256(frame.tobytes()).hexdigest() == str(hashes[i]), 'Decoded pixels differ from original mobile input')
            values, box, quality = mobile.preprocess_frame(frame, record.get('roi'))
            require(np.array_equal(quality[:5], cache.quality[i, :5]) and actual == cache.selected_presentation_times[i],
                    'Mobile quality or actual PTS changed')
            images[i] = uint8_from_normalized(values, mobile.RGB_MEAN, mobile.RGB_STD)
            boxes[i] = box
            if i in chosen:
                roi = mobile.normalize_roi(record.get('roi'))
                teacher = dino.preprocess_frames([frame], [roi], input_size=336)[0]
                teachers[chosen[i]] = uint8_from_normalized(teacher, dino.IMAGENET_RGB_MEAN, dino.IMAGENET_RGB_STD)
        images.flush()
        teachers.flush()
    finally:
        capture.release()
    del images, teachers
    after = video.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed while preparing')
    timing_path = folder/'timing.npz'
    np.savez_compressed(timing_path, times=times, selected_pts=cache.selected_presentation_times,
                        quality=cache.quality, boxes=boxes, teaching_indexes=teach)
    receipt = {'schemaVersion': 1, 'id': record['id'], 'sourceGroup': record['sourceGroup'],
               'contract': contract, 'arrays': {'images224': identity(image_path), 'teaching336': identity(teacher_path),
                                               'timing': identity(timing_path)},
               'frames': len(times), 'teachingFrames': len(teach), 'labelsUsed': False,
               'ignoredMaskUsedForTeachingSelectionOnly': True, 'allDecodedFrameHashesMatch': True,
               'wallSeconds': time.perf_counter()-start}
    write_immutable(receipt_path, receipt)
    print(json.dumps({'prepared': record['id'], 'frames': len(times), 'teaching': len(teach),
                      'wallSeconds': receipt['wallSeconds']}), flush=True)
    return receipt


def prepare(directory):
    from .neural_short_boost_transfer import load_data
    manifest = source_contract()
    original = read(MOBILE_INDEX)
    entries = {r['id']: r for r in original['records']}
    data = load_data(SOURCE, None, with_dino=False)
    examples = {r.example.id: r.example for rows in data.values() for r in rows}
    registration = {'kind': 'distillation-image-inputs-v1', 'manifest': identity(EXTRACTION),
                    'originalMobileIndex': identity(MOBILE_INDEX), 'source': identity(SOURCE),
                    'protocol': identity(OUTPUT/'protocol.md'), 'sourceCode': identity(__file__),
                    'pixels': 'uint8 CHW; normalized with original means/stds at inference',
                    'labelsUsed': False, 'protectedTestOpened': False}
    write_immutable(directory/'registration.json', registration)
    records = [prepare_record(r, entries[r['id']], examples[r['id']], directory) for r in manifest['records']]
    write_immutable(directory/'index.json', {'registration': identity(directory/'registration.json'), 'records': records})


def teacher_targets(directory, output, device):
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = read(directory/'index.json')
    contract = {'kind': 'same-frame-frozen-dino-targets-v1', 'inputIndex': identity(directory/'index.json'),
                'checkpointSha256': WEIGHT_SHA, 'repositoryCommit': COMMIT, 'sourceCode': identity(__file__),
                'device': device, 'precision': 'float32', 'labelsUsed': False}
    write_immutable(output/'registration.json', contract)
    backbone = dino.load_pinned_dinov2(ASSETS/f'dinov2-{COMMIT}', repository_commit=COMMIT,
        checkpoint=ASSETS/'dinov2_vits14_pretrain.pth', checkpoint_sha256=WEIGHT_SHA, device=device)
    results = []
    for record in source['records']:
        path = output/(record['id']+'.npy')
        receipt = output/(record['id']+'.json')
        if receipt.exists():
            old = read(receipt)
            require(old['input'] == record['arrays']['teaching336'], 'Teacher input changed')
            verify(old['output'])
            results.append(old)
            continue
        require(not path.exists(), 'Incomplete teacher targets')
        pixels = np.load(verify(record['arrays']['teaching336']), mmap_mode='r')
        rows = []
        with torch.inference_mode():
            for start in range(0, len(pixels), 8):
                batch = pixels[start:start+8].astype(np.float32)/255.
                batch = (batch-dino.IMAGENET_RGB_MEAN[None, :, None, None])/dino.IMAGENET_RGB_STD[None, :, None, None]
                rows.append(dino._extract_feature_tokens(backbone.model, torch, torch.from_numpy(batch).to(device), 336).cpu().numpy())
        targets = np.concatenate(rows)
        require(targets.shape == (record['teachingFrames'], 10, 384) and np.isfinite(targets).all(), 'Invalid teacher target')
        np.save(path, targets.astype(np.float32), allow_pickle=False)
        row = {'id': record['id'], 'sourceGroup': record['sourceGroup'], 'input': record['arrays']['teaching336'],
               'output': identity(path), 'registration': identity(output/'registration.json'), 'labelsUsed': False}
        write_immutable(receipt, row)
        results.append(row)
        print(json.dumps({'teacher': record['id'], 'frames': len(pixels)}), flush=True)
    write_immutable(output/'index.json', {'registration': identity(output/'registration.json'), 'records': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'teacher'))
    parser.add_argument('--images', type=Path, default=OUTPUT/'distillation-images-v1')
    parser.add_argument('--targets', type=Path, default=OUTPUT/'distillation-teacher-v1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.images)
    else:
        teacher_targets(args.images, args.targets, args.device)


if __name__ == '__main__':
    main()
