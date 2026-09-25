"""Prepare verified, label-blind beach images for frozen student inference.

All locations are resolved through private configuration or an explicit output
argument. No teacher images, targets, labels, training, or GPU work is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import mobile_visual_features as mobile
from analysis.distillation_image_inputs import uint8_from_normalized
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_inputs import verified
from analysis.private_ledger import private_value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def input_row(row):
    """Do not let annotations enter the image preparation/inference contract."""
    result = {key: row[key] for key in
              ('id', 'sourceGroup', 'contentSha256', 'durationSeconds', 'environment', 'video')}
    result['roi'] = list(mobile.normalize_roi(row.get('roi')))
    require(result['environment'] == 'beach', 'Only beach inference inputs are permitted')
    return result


def prepare(row, alias, prior_path, audiovisual, output, plan_reference):
    import cv2
    cv2.setNumThreads(1)
    prior = mobile.load_mobile_visual_cache(prior_path)
    require(prior.metadata['recordingId'] == row['id']
            and prior.metadata['recordingContentSha256'] == row['contentSha256']
            and prior.metadata['completed'] and not prior.metadata['partialVideo']
            and not prior.metadata['labelsUsed'], 'Prior source qualification cache differs')
    require(prior.metadata['identity']['roi'] == row['roi'], 'Prior ROI differs')
    verified(audiovisual)
    folder = output / 'images' / alias
    receipt_path = folder / 'image-input.json'
    contract = dict(source={key: row[key] for key in
                           ('id', 'sourceGroup', 'contentSha256', 'durationSeconds', 'environment', 'roi')},
                    plan=plan_reference, priorMobileCache=identity(prior_path),
                    audiovisual=audiovisual, teachingIndexes=[])
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['contract'] == contract, 'Prepared beach input contract changed')
        for reference in receipt['arrays'].values():
            verified(reference)
        return dict(**{key: row[key] for key in
                       ('id', 'sourceGroup', 'contentSha256', 'durationSeconds', 'environment')},
                    recordingIndex=alias, featureOrigin='opencv-av104-v3',
                    features=dict(audiovisual=audiovisual, imageInput=identity(receipt_path)))
    require(not folder.exists(), 'Incomplete beach image folder; investigate before resuming')
    video = Path(row['video'])
    before = video.stat()
    require(mobile.sha256_file(video) == row['contentSha256'], 'Source video hash differs')
    pts, inventory = mobile.presentation_inventory(video)
    require(abs(float(inventory['stream']['duration']) - row['durationSeconds']) < .11,
            'Source duration differs')
    times, indexes = mobile.sample_selection(pts, row['durationSeconds'])
    require(np.array_equal(times, prior.timestamps), 'Mobile sampling ticks differ')
    with np.load(prior_path, allow_pickle=False) as previous:
        require(np.array_equal(indexes, previous['selected_ordinals']), 'Selected frame ordinals differ')
        frame_hashes = previous['selected_frame_sha256'].copy()
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), 'Source video cannot be decoded')
    folder.mkdir(parents=True)
    image_path = folder / 'images224.npy'
    pixels = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8,
                                      shape=(len(times), 3, 224, 224))
    boxes = np.empty((len(times), 4), np.float64)
    quality = np.empty((len(times), 6), np.float32)
    selected_pts = np.empty(len(times), np.float64)
    started = time.monotonic()
    count = 0
    try:
        for i, (_, frame, actual) in enumerate(mobile.selected_frames(
                capture, indexes, pts, cv2.CAP_PROP_POS_MSEC)):
            require(hashlib.sha256(frame.tobytes()).hexdigest() == str(frame_hashes[i]),
                    'Decoded source pixels differ from qualification cache')
            normalized, box, values = mobile.preprocess_frame(frame, row['roi'])
            require(np.array_equal(values[:5], prior.quality[i, :5])
                    and actual == prior.selected_presentation_times[i],
                    'Selected PTS or image quality differs')
            # This inverse normalization must reproduce the prior normalized
            # tensor bit-for-bit when normalized again; it does not round anew.
            pixels[i] = uint8_from_normalized(normalized, mobile.RGB_MEAN, mobile.RGB_STD)
            values[-1] = actual - times[i]
            boxes[i], quality[i], selected_pts[i] = box, values, actual
            count = i + 1
            if count % 256 == 0 or count == len(times):
                print(json.dumps(dict(phase='beach-images', recordingIndex=alias,
                                      complete=count, total=len(times))), flush=True)
    finally:
        capture.release()
        pixels.flush()
        del pixels
    require(count == len(times), 'Beach image preparation is incomplete')
    after = video.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Source changed during beach image preparation')
    timing_path = folder / 'timing.npz'
    np.savez_compressed(timing_path, times=times, boxes=boxes, quality=quality,
                        selected_pts=selected_pts, selected_ordinals=indexes,
                        selected_frame_sha256=frame_hashes, teaching_indexes=np.asarray([], np.int64))
    receipt = dict(id=row['id'], sourceGroup=row['sourceGroup'], contract=contract,
                   arrays=dict(images224=identity(image_path), timing=identity(timing_path)),
                   frames=len(times), teachingFrames=0, teachingAllowed=False,
                   labelsUsed=False, ignoredIntervalsUsed=False,
                   allFramesPixelsPtsQualityAndNormalizationVerified=True,
                   wallSeconds=time.monotonic() - started)
    write_immutable(receipt_path, receipt)
    return dict(**{key: row[key] for key in
                   ('id', 'sourceGroup', 'contentSha256', 'durationSeconds', 'environment')},
                recordingIndex=alias, featureOrigin='opencv-av104-v3',
                features=dict(audiovisual=audiovisual, imageInput=identity(receipt_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-index', default='private-reference-0061')
    parser.add_argument('--phase1-index', default='private-reference-0129')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), 'Use an explicit absolute external output root')
    study = Path(private_value(args.study_index))
    phase1_path = Path(private_value(args.phase1_index))
    index_path = Path(os.environ['VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH'])
    ledger_path = os.environ.get('VOLLEYCUT_PRIVATE_LEDGER_POSIX') or os.environ['VOLLEYCUT_PRIVATE_LEDGER']
    aliases = read(Path(ledger_path))['originalToAlias']
    inventory_path = study / 'inventory-v1/inventory-v2.json'
    rows = [input_row(row) for row in read(inventory_path)['records'] if row['environment'] == 'beach']
    require(len(rows) == 2 and len({row['id'] for row in rows}) == 2, 'Expected two distinct beach recordings')
    sources = {row['id']: row for row in read(phase1_path)['recordings']}
    plan = dict(kind='label-blind-beach-student-images-v1', inventory=identity(inventory_path),
                phase1=identity(phase1_path), sourceCode=identity(__file__),
                preprocessingCode=identity(mobile.__file__), labelsUsed=False,
                teacherInputsProduced=False, beachUsedForTrainingCalibrationSelection=False,
                recordingIndexes=[aliases[row['id']] for row in rows])
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output / 'plan.json', plan)
    references = []
    for row in rows:
        source = sources[row['id']]
        require(source['contentSha256'] == row['contentSha256']
                and source['sourceGroup'] == row['sourceGroup'] and source['environment'] == 'beach',
                'Phase1 audiovisual source identity differs')
        prior = list((index_path.parent / 'beach-features/mobile').glob(row['id'] + '-*.npz'))
        require(len(prior) == 1, 'Ambiguous or missing prior Mobile frame qualification cache')
        audiovisual = {key: source['featureCaches']['audiovisual'][key] for key in ('path', 'sha256')}
        references.append(prepare(row, aliases[row['id']], prior[0], audiovisual, args.output,
                                  identity(args.output / 'plan.json')))
    write_immutable(args.output / 'catalog.json', dict(kind='label-blind-beach-student-inputs-v1',
        plan=identity(args.output / 'plan.json'), labelsUsed=False, records=references))
    print(json.dumps(dict(phase='complete', recordingCount=len(references))), flush=True)


if __name__ == '__main__':
    main()
