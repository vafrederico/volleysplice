"""Run the frozen, selected Large TCNs on beach footage without label access.

Exact locations come from ignored environment configuration and the private
ledger. The existing Small cache is only a frame/PTS/ROI qualification reference;
no Small or DINO inference, calibration, or training is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import mobile_visual_features as mobile
from analysis import neural_generalization_inputs as inputs
from analysis.neural_development import decode
from analysis.neural_evaluation import evaluate_predictions
from analysis.neural_generalization_experiment import load_checkpoint
from analysis.neural_recognition_fit import predict
from analysis.neural_recall_sweep import SweepExample
from analysis.private_ledger import private_value
from analysis.recognition_temporal_model import RecognitionConfig

CONFIG = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8, token_dimension=960)
MODEL_IDS = ('neural-mobile-large-tcn-fp32', 'neural-mobile-large-tcn-fp32-high-recall')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def sha(path):
    return mobile.sha256_file(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ledger_aliases():
    location = os.environ.get('VOLLEYCUT_PRIVATE_LEDGER_POSIX') or os.environ['VOLLEYCUT_PRIVATE_LEDGER']
    return read(location)['originalToAlias']


def extract(row, alias, prior, encoder, encoder_sha, destination, device):
    """Replay every declared sample, verifying pixels against the prior cache."""
    import cv2
    import torch
    cv2.setNumThreads(1)
    folder = destination / 'features' / alias
    receipt_path, output = folder / 'receipt.json', folder / 'tokens.npz'
    previous = mobile.load_mobile_visual_cache(prior)
    require(previous.metadata['recordingId'] == row['id'] and previous.metadata['completed']
            and not previous.metadata['partialVideo'] and not previous.metadata['labelsUsed'], 'Prior cache identity differs')
    require(previous.metadata['recordingContentSha256'] == row['contentSha256'], 'Prior source hash differs')
    require(previous.metadata['identity']['roi'] == list(mobile.normalize_roi(row.get('roi'))), 'Prior ROI differs')
    contract = dict(recordingIndex=alias, sourceSha256=row['contentSha256'], roi=list(mobile.normalize_roi(row.get('roi'))),
                    priorSmallCacheSha256=sha(prior), encoderSha256=encoder_sha, codeSha256=sha(__file__),
                    preprocessingSourceSha256=sha(mobile.__file__), model='mobilenet_v3_large',
                    weights='MobileNet_V3_Large_Weights.IMAGENET1K_V1', inputSize=224, samplingHz=2,
                    tokenShape=[4, 960], computePrecision='float32', savedPrecision='float16', labelsUsed=False)
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['contract'] == contract and receipt['outputSha256'] == sha(output), 'Extraction resume differs')
        return output, receipt
    video = Path(row['video'])
    before = video.stat()
    require(sha(video) == row['contentSha256'], 'Video differs from inventory')
    pts, inventory = mobile.presentation_inventory(video)
    duration = float(inventory['stream']['duration'])
    require(abs(duration-row['durationSeconds']) < .11, 'Video duration differs')
    times, indexes = mobile.sample_selection(pts, duration)
    require(np.array_equal(times, previous.timestamps), 'Sample ticks differ')
    with np.load(prior, allow_pickle=False) as z:
        require(np.array_equal(indexes, z['selected_ordinals']), 'Selected display frames differ')
        frame_hashes = z['selected_frame_sha256'].copy()
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), 'Cannot decode source')
    chunks, pixels, boxes, qualities, observed = [], [], [], [], []
    started = time.monotonic()
    try:
        for i, (_, frame, actual) in enumerate(mobile.selected_frames(capture, indexes, pts, cv2.CAP_PROP_POS_MSEC)):
            require(hashlib.sha256(frame.tobytes()).hexdigest() == str(frame_hashes[i]), 'Decoded pixels differ')
            normalized, box, quality = mobile.preprocess_frame(frame, row.get('roi'))
            require(np.array_equal(quality[:5], previous.quality[i, :5])
                    and actual == previous.selected_presentation_times[i], 'Preprocessing or selected PTS differs')
            pixels.append(normalized)
            boxes.append(box)
            quality[-1] = actual-times[i]
            qualities.append(quality)
            observed.append(actual)
            if len(pixels) == 32 or i+1 == len(times):
                with torch.inference_mode():
                    spatial = encoder(torch.from_numpy(np.stack(pixels)).to(device))
                    require(spatial.shape[1] == 960, 'Encoder geometry differs')
                    weights = mobile.regional_pool_weights(np.stack(boxes), *spatial.shape[-2:])
                    pooled = torch.einsum('bchw,brhw->brc', spatial, torch.from_numpy(weights).to(device))
                    chunks.append(pooled.cpu().numpy().astype(np.float16))
                pixels, boxes = [], []
                print(json.dumps(dict(phase='extract', recordingIndex=alias, complete=i+1, total=len(times))), flush=True)
    finally:
        capture.release()
    after = video.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed during extraction')
    tokens = np.concatenate(chunks)
    require(tokens.shape == (len(times), 4, 960) and np.isfinite(tokens).all(), 'Invalid embedding tensor')
    folder.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, timestamps=times, tokens=tokens, quality=np.stack(qualities),
                            selected_presentation_times=np.asarray(observed), selected_ordinals=indexes)
    receipt = dict(contract=contract, outputSha256=sha(output), shape=list(tokens.shape),
                   savedBytes=output.stat().st_size, float32TensorBytes=int(tokens.size*4),
                   framePixelsPtsAndQualityMatch=True, wallSeconds=time.monotonic()-started)
    save(receipt_path, receipt)
    return output, receipt


def inference_example(row, av, visual):
    # Deliberately remove annotations before any feature or classifier operation.
    sanitized = {k: row[k] for k in ('id', 'sourceGroup', 'durationSeconds', 'environment')}
    sanitized['featureOrigin'] = 'opencv-av104-v3'
    base = inputs.example_from_row(sanitized, {'audiovisual': av}, inference=True)
    with np.load(visual, allow_pickle=False) as z:
        cache = mobile.MobileVisualCache(visual, z['timestamps'], z['tokens'].astype(np.float32),
                                        z['quality'], z['selected_presentation_times'], {})
    aligned = mobile.align_mobile_features(cache, base.times)
    require(np.all(aligned['available'] == 1), 'Visual features leave AV ticks uncovered')
    extra = np.concatenate((aligned['tokens'].reshape(len(base.times), -1), aligned['quality'],
                            aligned['feature_age_seconds'][:, None], aligned['available'][:, None]), axis=1)
    example = replace(base, values=np.concatenate((base.values, extra), axis=1).astype(np.float32))
    require(example.values.shape[1] == CONFIG.input_dimension == 3952 and np.isfinite(example.values).all(), 'Fused geometry differs')
    require(not example.truth and not example.ignored and example.valid.all(), 'Labels reached inference')
    return example


def reference(row, model, times, scores, receipt, output_index):
    shell = SweepExample(row['id'], row['sourceGroup'], row['durationSeconds'], times, np.ones(len(times), bool), (), ())
    rallies = [dict(start=r.start, end=r.end) for r in decode(shell, scores, model['decoder'])]
    description = (f"Frozen MobileNetV3 Large V1 at 224px/2Hz with FP32 TCN; {model['variant']} draw {model['seed']}, "
                   f"epoch {model['epoch']}. Non-beach calibration target {model['recallTargetPercent']}%; no beach fitting, "
                   "calibration, or selection. Core rally boundaries stay separate; export gaps strictly under 3s join. "
                   "Serve/start is timing only, not serving side.")
    return dict(modelId=model['modelId'], modelLabel=model['modelLabel'], description=description,
                rallies=rallies, exportRallies=rallies, exportPolicy='model-predictions', research=dict(
                    recommendation=description, signals=dict(times=times.tolist(), **{h:scores[:, i].tolist()
                        for i, h in enumerate(('live', 'serve', 'end', 'keep'))}), boundaryFlags=[], reviewRegions=[],
                    queue=dict(budgetFraction=0, reviewSeconds=0, selectedParentCount=0),
                    provenance=dict(**model, labelsUsed=False, ignoredIntervalsUsed=False,
                                    sourceIndex=output_index, sourceSha256=receipt['outputSha256'])))


def publish(index_path, original_index, changes, backup):
    """Preserve non-Large references and roll back the small atomic publication."""
    require(index_path.read_bytes() == original_index, 'Comparison index changed during inference')
    backup.mkdir(parents=True, exist_ok=True)
    if not (backup/'index.json').exists():
        (backup/'index.json').write_bytes(original_index)
    try:
        for path, old, new, alias in changes:
            require(path.read_bytes() == old, 'A recording reference changed during inference')
            if not (backup/(alias+'.json')).exists():
                (backup/(alias+'.json')).write_bytes(old)
            save(path, new)
        index = json.loads(original_index)
        by_id = {doc['recordings'][0]['recordingId']: doc['recordings'][0] for _, _, doc, _ in changes}
        for entry in index['recordings']:
            if entry['id'] in by_id:
                entry['modelIds'] = [r['modelId'] for r in by_id[entry['id']]['references']]
        save(index_path, index)
    except Exception:
        for path, old, _, _ in changes:
            path.write_bytes(old)
        index_path.write_bytes(original_index)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-index', default='private-reference-0211')
    parser.add_argument('--study-index', default='private-reference-0061')
    parser.add_argument('--phase1-index', default='private-reference-0129')
    parser.add_argument('--output-index', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torchvision.models import mobilenet_v3_large
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    experiment = Path(private_value(args.experiment_index))
    study = Path(private_value(args.study_index))
    destination = Path(private_value(args.output_index))
    index_path = Path(os.environ['VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH'])
    original_index = index_path.read_bytes()
    index = json.loads(original_index)
    catalog_path = index_path.parent/'labeling-catalog.json'
    catalog_before = catalog_path.read_bytes()
    catalog = json.loads(catalog_before)
    rows = [r for r in read(study/'inventory-v1/inventory-v2.json')['records'] if r['environment'] == 'beach']
    require(len(rows) == 2, 'Expected two registered beach recordings')
    aliases = ledger_aliases()
    require(all(r['id'] in aliases and r['sourceGroup'] in aliases for r in rows), 'Missing beach recording or source-group ledger aliases')
    phase1_path = Path(private_value(args.phase1_index))
    sources = {r['id']: r for r in read(phase1_path)['recordings']}
    catalog_rows = {r['recordingId']: r for r in catalog['records']}
    by_model = {r['modelId']: r for r in index['models']}
    models = [by_model[key] for key in MODEL_IDS]
    selected = read(experiment/'evaluation.json')['selected']
    for model in models:
        choice = next(s for s in selected if s['mode'] == model['selectionMode'])
        require((model['variant'], model['seed'], model['epoch'], model['decoder'], model['recallTargetPercent']) ==
                (choice['variant'], choice['draw'], choice['setting']['epoch'], choice['setting']['decoder'], choice['floorPercent']),
                'UI selection differs from frozen experiment')
    checkpoint = experiment/'torch-cache/checkpoints/mobilenet_v3_large-8738ca79.pth'
    encoder_sha = sha(checkpoint)
    require(encoder_sha.startswith('8738ca79'), 'Unexpected official ImageNet V1 weights')
    encoder = mobilenet_v3_large(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    encoder = encoder.features.to(args.device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    results, changes, feature_receipts = {key: [] for key in MODEL_IDS}, [], []
    for row in rows:
        alias = aliases[row['id']]
        source = sources[row['id']]
        require(source['contentSha256'] == row['contentSha256'] and source['sourceGroup'] == row['sourceGroup']
                and source['environment'] == 'beach', 'Phase1 source identity differs')
        require(catalog_rows[row['id']]['rallies'] == row['rallies']
                and catalog_rows[row['id']]['ignoredIntervals'] == row['ignoredIntervals'], 'Human beach labels differ')
        prior = list((index_path.parent/'beach-features/mobile').glob(row['id']+'-*.npz'))
        require(len(prior) == 1, 'Ambiguous or missing prior frame qualification cache')
        visual, feature_receipt = extract(row, alias, prior[0], encoder, encoder_sha, destination, args.device)
        feature_receipts.append(dict(recordingIndex=alias, **feature_receipt))
        av = {k: source['featureCaches']['audiovisual'][k] for k in ('path', 'sha256')}
        example = inference_example(row, av, visual)
        entry = next(r for r in index['recordings'] if r['id'] == row['id'])
        path = index_path.parent/entry['file']
        old = path.read_bytes()
        doc = json.loads(old)
        require(len(doc['recordings']) == 1 and doc['recordings'][0]['recordingId'] == row['id'], 'UI source association differs')
        record = doc['recordings'][0]
        refs = [r for r in record['references'] if r['modelId'] not in MODEL_IDS]
        prior_revision = hashlib.sha256(old).hexdigest()
        for ref in refs:
            ref.setdefault('research', {}).setdefault('provenance', {}).setdefault('uiDraftRevision', prior_revision)
        for model in models:
            fit = experiment/'fits'/model['variant']/f"split-{model['seed']}"
            weights = fit/'temporal'/f"weights-{model['epoch']}.npz"
            completed = read(fit/'temporal/completed.json')
            require(sha(weights) == completed['artifacts'][weights.name], 'Frozen checkpoint hash differs')
            lineage = dict(recordingIndex=alias, modelId=model['modelId'], weightsSha256=sha(weights),
                           selectedEvaluationSha256=sha(experiment/'evaluation.json'),
                           visualSha256=sha(visual), audiovisualSha256=av['sha256'], labelsUsed=False,
                           ignoredIntervalsUsed=False, decoder=model['decoder'], featureDimension=3952)
            output = destination/'inference'/model['modelId']/(alias+'.npz')
            receipt_path = output.with_suffix('.json')
            if receipt_path.exists():
                receipt = read(receipt_path)
                require(receipt['lineage'] == lineage and sha(output) == receipt['outputSha256'], 'Inference resume differs')
                with np.load(output, allow_pickle=False) as z:
                    require(np.array_equal(z['times'], example.times), 'Inference times differ')
                    scores = z['scores'].copy()
            else:
                temporal, mean, scale = load_checkpoint(weights, CONFIG, args.device)
                scores = predict(temporal, example, mean, scale, CONFIG, args.device)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('xb') as stream:
                    np.savez_compressed(stream, times=example.times, scores=scores)
                receipt = dict(lineage=lineage, outputSha256=sha(output), samples=len(example.times))
                save(receipt_path, receipt)
            require(scores.shape == (len(example.times), 4) and np.isfinite(scores).all(), 'Invalid four-head inference')
            ref = reference(row, model, example.times, scores, receipt, args.output_index)
            refs.append(ref)
            # Evaluation attaches annotations only after the frozen prediction is saved.
            results[model['modelId']].append(dict(id=alias, sourceGroup=aliases[row['sourceGroup']],
                durationSeconds=row['durationSeconds'], rallies=row['rallies'], ignoredIntervals=row['ignoredIntervals'],
                predictions=ref['rallies']))
            print(json.dumps(dict(phase='inference', recordingIndex=alias, modelId=model['modelId'], rallies=len(ref['rallies']))), flush=True)
        record['references'] = refs
        require(len({r['modelId'] for r in refs}) == len(refs), 'Duplicate model reference')
        changes.append((path, old, doc, alias))
    require(catalog_path.read_bytes() == catalog_before, 'Human labeling catalog changed')
    publish(index_path, original_index, changes, destination/'ui-backup')
    require(catalog_path.read_bytes() == catalog_before, 'Publication changed human labels')
    report = dict(kind='mobile-large-frozen-beach-evaluation-v1', sourceIndex=args.output_index,
                  labelsUsedForInference=False, beachUsedForTrainingCalibrationSelection=False,
                  humanCatalogSha256=sha(catalog_path), modelSelections=models, features=feature_receipts,
                  evaluation={key:evaluate_predictions(value) for key,value in results.items()},
                  predictionCounts={key:{r['id']:len(r['predictions']) for r in value} for key,value in results.items()})
    save(destination/'report.json', report)
    lines = ['# Frozen MobileNetV3 Large TCN beach evaluation', '',
             'Both existing selections were frozen before beach inference. No beach training, calibration, or selection was performed. '
             'All frames were processed without rally labels or ignored intervals. Prior source, ROI, decoded pixel, PTS, and quality checks passed. '
             'Human labels were attached only for evaluation and remain unchanged.', '',
             'Target product padding is 2 seconds before and after; join positive gaps strictly under 3 seconds. '
             'Subtract ignored intervals after joining and never rejoin across them. Metrics pool time across both recordings.', '',
             '| Selection | Recording | Found rallies | Wholly missed human rallies | P_pad | R_core | F1_padP_coreR |',
             '|---|---|---:|---:|---:|---:|---:|']
    for model in models:
        evaluation = report['evaluation'][model['modelId']]
        for recording in evaluation['recordings']:
            m = recording['primary']
            counts = recording['guardrails']['primaryExportCoverage']
            lines.append(f"| {model['selectionMode']} | {recording['id']} | {report['predictionCounts'][model['modelId']][recording['id']]} | "
                         f"{counts['completeRallyLosses']} | {m['P_pad']:.2%} | {m['R_core']:.2%} | {m['F1_padP_coreR']:.2%} |")
    lines += ['', '| Selection | Padding | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in models:
        for m in report['evaluation'][model['modelId']]['padding']:
            lines.append(f"| {model['selectionMode']} | {m['paddingSecondsBeforeAndAfter']:.0f}s | {m['P_pad']:.2%} | {m['R_core']:.2%} | "
                         f"{m['F1_padP_coreR']:.2%} | {m['paddedModelExportSeconds']:.2f} | {m['paddedHumanExportSeconds']:.2f} | {m['exportDurationDifferenceSeconds']:+.2f} |")
    lines += ['', 'Wholly missed means no retained nonignored human core after product padding and gap joining. '
              'Event matching and all coverage diagnostics are included in the JSON. Beach results do not change either selected model.', '']
    (destination/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(phase='complete', sourceIndex=args.output_index, counts=report['predictionCounts'])), flush=True)


if __name__ == '__main__':
    main()
