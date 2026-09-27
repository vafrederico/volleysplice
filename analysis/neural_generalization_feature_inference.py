"""Frozen image encoders on shared registered NAS pixels; no model fitting."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from . import mobile_visual_features as mobile
from . import dinov2_embeddings as dino
from . import neural_generalization_feature_stage as stage
from .neural_context_development import identity, read, write_immutable
from .neural_generalization_inputs import require, verified

REPO = Path(__file__).resolve().parents[1]
PRIOR = stage.NAS/'2026-09-23-recall-distillation/dino-precision-v1'
MOBILE_WEIGHT = stage.NAS/'2026-09-22-recognition/assets/mobilenet_v3_small-047dcff4.pth'


def precision_module():
    spec = importlib.util.spec_from_file_location('frozen_precision', REPO/'scripts/evaluate-dino-precision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register(root=stage.ROOT):
    root = stage.environment(root)
    stage.verify_plan(root/'stage-plan.json')
    require(read(root/'engineering-parity.json')['passed'] is True, 'Shared pixel/AV engineering gate is absent')
    q = read(PRIOR/'qualification.json')
    require(q['passed'] is True, 'Original FP32 encoder qualification failed')
    refs = [identity(root/'stage-plan.json'), identity(root/'engineering-parity.json'),
            identity(PRIOR/'qualification.json'), identity(PRIOR/'publication-v1/manifest.json'),
            identity(PRIOR/'encoder-dynamic-int8.onnx'), identity(MOBILE_WEIGHT),
            identity(REPO/'scripts/evaluate-dino-precision.py'),
            identity(REPO/'scripts/extract-neural-generalization-encoders.py'), identity(__file__)]
    result = {'kind': 'generalization-frozen-image-encoders-v1', 'parents': refs,
        'stagePlan': identity(root/'stage-plan.json'), 'source': identity(__file__),
        'dinoPrecision': {'fp32': 'PyTorch CUDA float32, batch8, no TF32',
                          'fp16': 'Identical original parameters cast to CUDA float16, batch8',
                          'int8': 'Original mixed dynamic perchannelQInt8 weight graph, nativeCPUORT, batch1, two threads'},
        'quantizedGraph': identity(PRIOR/'encoder-dynamic-int8.onnx'),
        'mobileCheckpoint': identity(MOBILE_WEIGHT), 'mobilePrecision': 'FP32 frozen ImageNet MobileNetV3Small',
        'teacherTargets': 'FullFP32 same-frame tokens on existing fixed teaching indexes; fit-eligible sources only.',
        'browserInt8ParityPassed': False, 'physicalPhoneQualified': False,
        'labelsUsed': False, 'trainingPerformed': False}
    write_immutable(root/'encoder-plan.json', result)
    archive = root/'registered-encoder-sources'
    archive.mkdir(exist_ok=True)
    for path in (Path(__file__), REPO/'scripts/extract-neural-generalization-encoders.py'):
        target = archive/path.name
        if target.exists():
            require(target.read_bytes() == path.read_bytes(), 'Encoder source snapshot changed')
        else:
            with target.open('xb') as stream:
                stream.write(path.read_bytes())
    return result


def verify_plan(root):
    plan = read(root/'encoder-plan.json')
    for reference in plan['parents']:
        verified(reference)
    stage.verify_plan(root/'stage-plan.json')
    return plan


def dino_gpu_operation(model, dtype):
    import torch
    def infer(values):
        with torch.inference_mode():
            return np.concatenate([dino._extract_feature_tokens(model, torch,
                torch.from_numpy(values[i:i+8]).to('cuda', dtype=dtype), 336).float().cpu().numpy()
                for i in range(0, len(values), 8)])
    return infer


def gpu_backbones():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = precision_module().backbone('cuda').model.eval()
    half = copy.deepcopy(model).half().eval()
    backbone = mobile.load_mobile_backbone(MOBILE_WEIGHT, device='cuda')
    return {'fp32': dino_gpu_operation(model, torch.float32),
            'fp16': dino_gpu_operation(half, torch.float16)}, backbone


def mobile_pool(backbone, pixels, boxes):
    import torch
    chunks = []
    with torch.inference_mode():
        for left in range(0, len(pixels), 16):
            values = pixels[left:left+16].transpose(0, 2, 3, 1).astype(np.float32)/255.
            values = (values-mobile.RGB_MEAN)/mobile.RGB_STD
            tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(backbone.device)
            spatial = backbone.model.features(tensor)
            weights = torch.from_numpy(mobile.regional_pool_weights(boxes[left:left+16], *spatial.shape[-2:])).to(backbone.device)
            chunks.append(torch.einsum('bchw,brhw->brc', spatial.float(), weights).cpu().numpy().astype(np.float16))
    return np.concatenate(chunks)


def qualify(root=stage.ROOT):
    import torch
    root = stage.environment(root)
    verify_plan(root)
    plan = stage.verify_plan(root/'stage-plan.json')
    source = next(r for r in plan['records'] if r['id'] == plan['engineeringRecord'])
    staged = read(root/'staged'/source['id']/'receipt.json')
    with np.load(verified(source['reuse']['dino']['fp32']), allow_pickle=False) as payload:
        reference = payload['tokens']
    chosen = np.linspace(0, len(reference)-1, 32, dtype=np.int64)
    rgbs = []
    for index in chosen:
        block = np.load(verified(staged['dinoRgbChunks'][int(index)//128]), mmap_mode='r')
        rgbs.append(block[int(index)%128])
    p = precision_module()
    operations, backbone = gpu_backbones()
    pixels = p.normalized(np.stack(rgbs))
    outputs = {arm: operation(pixels) for arm, operation in operations.items()}
    require(np.allclose(outputs['fp32'].astype(np.float16), reference[chosen], atol=.02, rtol=.002),
            'New DINO GPU wrapper differs from original same-frame cache')
    drift = p.errors(outputs['fp16'], outputs['fp32'])
    require(drift['meanCosineSimilarity'] >= .9999 and drift['rootMeanSquareError'] <= .02,
            'New FP16 wrapper unexpectedly differs from FP32')
    image, _ = image_input(source, staged)
    data = np.load(verified(image['arrays']['images224']), mmap_mode='r')
    selected = np.linspace(0, len(data)-1, 32, dtype=np.int64)
    with np.load(verified(image['arrays']['timing']), allow_pickle=False) as timing:
        boxes = timing['boxes'][selected]
    old_mobile = mobile.load_mobile_visual_cache(verified(source['reuse']['mobile']))
    actual = mobile_pool(backbone, data[selected], boxes)
    require(np.allclose(actual, old_mobile.tokens[selected], atol=.01, rtol=.002),
            'New frozen MobileNet wrapper differs from original same-frame cache')
    result = {'kind': 'generalization-image-wrapper-engineering-v1', 'passed': True,
        'encoderPlan': identity(root/'encoder-plan.json'), 'staging': identity(root/'staged'/source['id']/'receipt.json'),
        'recordingId': source['id'], 'dinoFrameIndexes': chosen.tolist(), 'mobileFrameIndexes': selected.tolist(),
        'dinoFp32VsOriginalFloat16Cache': p.errors(outputs['fp32'].astype(np.float16), reference[chosen]),
        'fp16VsFp32': drift, 'mobileVsOriginalFloat16Cache': p.errors(actual, old_mobile.tokens[selected]),
        'labelsUsed': False, 'phoneQualified': False}
    write_immutable(root/'encoder-engineering.json', result)
    return result


def image_input(source, staged):
    reference = staged['outputs'].get('imageInput', source.get('reuse', {}).get('imageInput'))
    require(reference is not None, 'Image input is missing')
    return read(verified(reference)), reference


def dino_record(source, staged, arm, infer, plan, root):
    if arm in source.get('reuse', {}).get('dino', {}):
        return source['reuse']['dino'][arm]
    key = source['id']
    folder = root/'encoders'/arm/key
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder/'receipt.json'
    lineage = {'id': key, 'sourceGroup': source['sourceGroup'], 'arm': arm,
               'encoderPlan': identity(root/'encoder-plan.json'),
               'staged': identity(root/'staged'/key/'receipt.json')}
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(all(receipt[k] == v for k, v in lineage.items()), 'DINO precision resume differs')
        verified(receipt['output'])
        return receipt['output']
    with np.load(verified(staged['outputs']['selection']), allow_pickle=False) as payload:
        times = payload['times']
    teacher_indexes, teacher_rows = {}, {}
    if arm == 'fp32' and source['teaching']['allowed'] and 'teacherTargets' not in source.get('reuse', {}):
        images, _ = image_input(source, staged)
        with np.load(verified(images['arrays']['timing']), allow_pickle=False) as timing:
            selected = timing['teaching_indexes']
            require(np.array_equal(timing['times'], times[::2]), 'Teacher2Hz and DINO4Hz grids differ')
        teacher_indexes = {int(tick)*2: i for i, tick in enumerate(selected)}
    chunks, refs = [], []
    started = time.perf_counter()
    p = precision_module()
    for reference in staged['dinoRgbChunks']:
        left = int(Path(reference['path']).stem)
        rgb = np.load(verified(reference), allow_pickle=False)
        path = folder/f'{left:06d}.npy'
        # Recompute orphaned chunks rather than trusting unbound partial outputs.
        if path.exists():
            require((folder/f'{left:06d}.json').exists(), 'Orphaned DINO chunk requires audit')
            previous = read(folder/f'{left:06d}.json')
            require(previous['input'] == reference and previous['lineage'] == lineage, 'Chunk source binding differs')
            tokens = np.load(verified(previous['output']), allow_pickle=False)
            require(not teacher_indexes, 'Interrupted teacher extraction requires a fresh versioned output')
        else:
            raw = infer(p.normalized(rgb))
            require(raw.shape == (len(rgb), 10, 384) and np.isfinite(raw).all(), 'Invalid DINO output')
            for tick, destination in teacher_indexes.items():
                if left <= tick < left+len(rgb):
                    teacher_rows[destination] = raw[tick-left].copy()
            tokens = raw.astype(np.float16)
            require(np.isfinite(tokens).all(), 'DINO output overflowed float16 storage')
            with path.open('xb') as stream:
                np.save(stream, tokens, allow_pickle=False)
            write_immutable(folder/f'{left:06d}.json', {'lineage': lineage, 'input': reference, 'output': identity(path)})
        require(tokens.dtype == np.float16 and tokens.shape == (len(rgb), 10, 384), 'DINO chunk geometry differs')
        chunks.append(tokens)
        refs.append(identity(path))
        if left % 1024 == 0:
            print(json.dumps({'embedding': key, 'arm': arm, 'frames': left+len(rgb), 'total': len(times)}), flush=True)
    tokens = np.concatenate(chunks)
    require(tokens.shape == (len(times), 10, 384), 'DINO cache coverage differs')
    path = folder/'tokens.npz'
    stage.npz_new(path, timestamps=times, tokens=tokens)
    teacher_ref = None
    if teacher_indexes:
        require(len(teacher_rows) == len(teacher_indexes), 'Teacher subset incomplete')
        targets = np.stack([teacher_rows[i] for i in range(len(teacher_rows))]).astype(np.float32)
        target = folder/'teacher.npy'
        with target.open('xb') as stream:
            np.save(stream, targets, allow_pickle=False)
        teacher = {'id': key, 'sourceGroup': source['sourceGroup'], 'input': images['arrays']['teaching336'],
                   'output': identity(target), 'registration': identity(root/'encoder-plan.json'), 'labelsUsed': False}
        teacher_path = folder/'teacher.json'
        write_immutable(teacher_path, teacher)
        teacher_ref = identity(teacher_path)
    receipt = {**lineage, 'output': identity(path), 'chunks': refs, 'teacherTargets': teacher_ref,
               'samples': len(times), 'wallSeconds': time.perf_counter()-started, 'labelsUsed': False}
    write_immutable(receipt_path, receipt)
    return receipt['output']


def mobile_record(source, staged, backbone, root):
    if 'mobile' in source.get('reuse', {}):
        return source['reuse']['mobile']
    import torch
    images, image_reference = image_input(source, staged)
    folder = root/'encoders/mobile'/source['id']
    folder.mkdir(parents=True, exist_ok=True)
    lineage = {'id': source['id'], 'sourceGroup': source['sourceGroup'],
               'encoderPlan': identity(root/'encoder-plan.json'), 'imageInput': image_reference}
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(all(receipt[k] == v for k, v in lineage.items()), 'Mobile resume differs')
        verified(receipt['output'])
        return receipt['output']
    pixels = np.load(verified(images['arrays']['images224']), mmap_mode='r')
    with np.load(verified(images['arrays']['timing']), allow_pickle=False) as timing:
        arrays = {key: timing[key].copy() for key in ('times', 'selected_pts', 'selected_ordinals',
                  'selected_frame_sha256', 'quality', 'boxes')}
    tokens = mobile_pool(backbone, pixels, arrays['boxes'])
    times = arrays['times']
    require(tokens.shape == (len(times), 4, 576) and np.isfinite(tokens).all(), 'Invalid frozen MobileNet tokens')
    metadata = {'schemaVersion': 1, 'kind': 'volleycut-frozen-mobile-visual-cache',
        'recordingId': source['id'], 'recordingContentSha256': source['videoIdentity']['sha256'],
        'sourceVideoPath': source['videoIdentity']['path'], 'completed': True, 'partialVideo': False,
        'labelsUsed': False, 'qualityNames': list(mobile.QUALITY_NAMES),
        'video': {'duration': source['durationSeconds']}, 'extractedDurationSeconds': source['durationSeconds'],
        'identity': {'recordingId': source['id'], 'recordingContentSha256': source['videoIdentity']['sha256'],
                     'roi': list(mobile.normalize_roi(source.get('roi'))), 'backbone': backbone.identity(),
                     'config': mobile.MobileVisualConfig(batch_size=16).to_dict(),
                     'extractorSourceSha256': identity(__file__)['sha256'], 'maximumExtractionSeconds': None},
        'runtime': {'device': backbone.device, 'batchSize': 16},
        'registeredInputs': lineage}
    path = folder/'tokens.npz'
    stage.npz_new(path, timestamps=times, tokens=tokens, quality=arrays['quality'],
        selected_presentation_times=arrays['selected_pts'], selected_ordinals=arrays['selected_ordinals'],
        selected_frame_sha256=arrays['selected_frame_sha256'], metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    mobile.load_mobile_visual_cache(path)
    write_immutable(receipt_path, {**lineage, 'output': identity(path), 'labelsUsed': False})
    return identity(path)


def run(arm, identifiers=None, root=stage.ROOT):
    root = stage.environment(root)
    plan = verify_plan(root)
    staging = stage.verify_plan(root/'stage-plan.json')
    engineering = read(root/'encoder-engineering.json')
    require(engineering['passed'] is True and engineering['encoderPlan'] == identity(root/'encoder-plan.json'),
            'Image wrapper engineering gate is missing or stale')
    p = precision_module()
    require(arm in ('gpu', 'int8'), 'Unknown encoder phase')
    if arm == 'gpu':
        operations, backbone = gpu_backbones()
    else:
        runtime = p.session(verified(plan['quantizedGraph']), threads=2)
        operations = {'int8': lambda values: np.concatenate([runtime.run(None, {'image': values[i:i+1]})[0]
                                                           for i in range(len(values))])}
    requested = None if identifiers is None else set(identifiers)
    require(requested is None or requested <= {r['id'] for r in staging['records']}, 'Unknown encoder source IDs')
    for source in staging['records']:
        if requested is not None and source['id'] not in requested:
            continue
        staged = read(root/'staged'/source['id']/'receipt.json')
        require(staged['lineage']['plan'] == identity(root/'stage-plan.json')
                and staged['lineage']['source'] == source, 'Staging source/plan binding differs')
        for precision, operation in operations.items():
            dino_record(source, staged, precision, operation, plan, root)
        if arm == 'gpu':
            mobile_record(source, staged, backbone, root)
        print(json.dumps({'encoderComplete': source['id'], 'phase': arm}), flush=True)


def publish_index(mode='complete', root=stage.ROOT, scope='all'):
    root = stage.environment(root)
    require(mode in ('av', 'core', 'complete'), 'Unknown feature publication mode')
    require(scope in ('all', 'fit'), 'Unknown feature publication scope')
    plan = stage.verify_plan(root/'stage-plan.json')
    rows = []
    for source in plan['records']:
        if scope == 'fit' and not source['teaching']['allowed']:
            continue
        key = source['id']
        staging_path = root/'staged'/key/'receipt.json'
        staged = read(staging_path) if staging_path.exists() else None
        if staged is not None:
            require(staged['lineage']['source'] == source
                    and staged['lineage']['plan'] == identity(root/'stage-plan.json'), 'Published stage source differs')
        else:
            require(scope == 'fit' and mode != 'complete' and source.get('reuse'),
                    'New or full-inference source staging is incomplete: '+key)
        features = copy.deepcopy(source.get('reuse', {}))
        for name in ('audiovisual', 'imageInput'):
            if name not in features and staged is not None and name in staged['outputs']:
                features[name] = staged['outputs'][name]
        require('audiovisual' in features, 'AV cache not complete: '+key)
        if mode != 'av':
            for arm in ('fp32', 'fp16', 'int8'):
                path = root/'encoders'/arm/key/'receipt.json'
                if path.exists():
                    receipt = read(path)
                    require(receipt['id'] == key and receipt['sourceGroup'] == source['sourceGroup']
                            and receipt['encoderPlan'] == identity(root/'encoder-plan.json'), 'Encoder receipt differs')
                    features.setdefault('dino', {})[arm] = receipt['output']
                    if receipt.get('teacherTargets'):
                        features['teacherTargets'] = receipt['teacherTargets']
            path = root/'encoders/mobile'/key/'receipt.json'
            if path.exists():
                receipt = read(path)
                require(receipt['encoderPlan'] == identity(root/'encoder-plan.json'), 'Mobile encoder plan differs')
                features['mobile'] = receipt['output']
            require('fp32' in features.get('dino', {}) and 'mobile' in features and 'imageInput' in features,
                    'Core image feature family is incomplete: '+key)
            if source['teaching']['allowed']:
                require('teacherTargets' in features, 'Fit-eligible teacher targets missing: '+key)
            else:
                require('teacherTargets' not in features, 'Evaluation-only source has teacher targets')
            if mode == 'complete':
                require(set(features['dino']) == {'fp32', 'fp16', 'int8'}, 'Precision inventory incomplete: '+key)
        for name, reference in features.items():
            if name == 'dino':
                for subreference in reference.values():
                    verified(subreference)
            else:
                verified(reference)
        rows.append({'recordingId': key, 'sourceGroup': source['sourceGroup'],
                    'contentSha256': source['videoIdentity']['sha256'], 'features': features,
                    'stagingReceipt': identity(staging_path) if staged is not None else None,
                    'reusedOriginalFeatureRegistration': identity(root/'reference-qualification-v1/report.json')
                        if staged is None else None})
    require(len(rows) == (32 if scope == 'fit' else 42), 'Published feature population differs')
    result = {'kind': 'neural-generalization-features-v1', 'stagePlan': identity(root/'stage-plan.json'),
        'encoderPlan': identity(root/'encoder-plan.json') if mode != 'av' else None,
        'publicationMode': mode, 'publicationScope': scope, 'records': rows, 'labelsUsedForFeatures': False,
        'protectedImageryUsedForTraining': False, 'precisionRuntimeLimitationsUnchanged': True}
    path = root/f'features-{"fit-" if scope == "fit" else ""}{mode}.json'
    write_immutable(path, result)
    return identity(path)
