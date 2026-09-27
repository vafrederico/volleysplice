"""DINO-to-MobileNetV3-Large student with the existing Small training recipe.

Callers supply registered, local ImageNet V1 weights and external artifact paths.
This module never downloads weights or chooses training/selection recordings.
Only the 960-channel encoder is needed after training; the DINO projector is not
part of the rally inference pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import mobile_visual_features as mobile
from .neural_context_development import canonical_hash, identity, read, write_immutable
from .neural_mobile_distillation import (
    distillation_loss, normalize_pixels, require, teaching_tokens,
    training_arrays, verify_once,
)


CHANNELS = 960
EXTRACTION_BATCH_SIZE = 32
RECIPE = {
    'kind': 'dino-distilled-mobilenet-v3-large-v1',
    'backbone': 'mobilenet_v3_large',
    'initialWeights': 'MobileNet_V3_Large_Weights.IMAGENET1K_V1',
    'embeddingDimension': CHANNELS,
    'teacherTokenShape': [10, 384],
    'epochs': 8,
    'batchSize': 16,
    'optimizer': 'AdamW',
    'learningRate': 1e-4,
    'weightDecay': 1e-4,
    'batchNormStatisticsFrozen': True,
    'loss': '.5 global cosine + .5 mean9 spatial cosine; source/record-balanced frame weights',
}


def _checkpoint_identity(checkpoint):
    path = Path(checkpoint)
    require(path.is_file(), 'Local ImageNet V1 checkpoint is required; downloads are disabled')
    reference = identity(path)
    require(reference['sha256'].startswith('8738ca79'),
            'Large checkpoint does not match official ImageNet V1 hash prefix')
    return reference


def build_student(seed, device, checkpoint):
    """Initialize from the local pinned checkpoint, with gradients and fixed BN."""
    from torchvision.models import mobilenet_v3_large

    reference = _checkpoint_identity(checkpoint)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = mobilenet_v3_large(weights=None)
    model.load_state_dict(torch.load(reference['path'], map_location='cpu', weights_only=True), strict=True)
    encoder = model.features.to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    encoder.eval()
    projector = nn.Linear(CHANNELS, 384).to(device)
    return encoder, projector


def _fit_identity(rows, excluded, images, teachers, seed, digest, checkpoint):
    ids = [row.example.id for row in rows]
    groups = sorted({row.example.group for row in rows})
    require(rows and len(ids) == len(set(ids)), 'Empty or duplicate student training membership')
    require(not (set(excluded) & set(groups)), 'Held source in student fitting')
    # The enclosing digest pins the experiment; this additional identity prevents
    # a completed student being reused with a different image/teacher mapping.
    inputs = [{'id': key, 'image': images[key], 'teacher': teachers[key]} for key in ids]
    return {'contractSha256': digest, 'seed': seed, 'excludedGroups': sorted(excluded),
            'trainIds': ids, 'trainGroups': groups, 'recipe': RECIPE,
            'initialCheckpoint': _checkpoint_identity(checkpoint),
            'trainingInputSha256': canonical_hash(inputs)}


def fit_student(rows, excluded, images, teachers, seed, directory, digest, device, checkpoint):
    """Fit eight deterministic epochs, or verify an exactly matching completion.

    Incomplete runs are rejected instead of silently starting a different fit in
    the same artifact directory. The orchestrator owns restart/recovery policy.
    """
    directory = Path(directory)
    identity_fields = _fit_identity(rows, excluded, images, teachers, seed, digest, checkpoint)
    done = directory / 'completed.json'
    if done.exists():
        completed = read(done)
        require(all(completed.get(key) == value for key, value in identity_fields.items()),
                'Student resume identity differs')
        verify_once(completed['weights'])
        return completed
    require(not directory.exists(), 'Incomplete Large student fit')
    directory.mkdir(parents=True)
    pixels, targets, weights, membership = training_arrays(rows, images, teachers)
    require(len(pixels) > 0 and targets.shape == (len(pixels), 10, 384)
            and weights.shape == (len(pixels),) and np.isfinite(weights).all()
            and np.all(weights > 0), 'Invalid student training arrays')
    weights = weights / weights.mean()
    encoder, projector = build_student(seed, device, checkpoint)
    trainable = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history, exposure = [], hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, 9):
        order = rng.permutation(len(pixels))
        exposure.update(order.astype('<i8').tobytes())
        total = 0.
        for start in range(0, len(order), 16):
            indexes = order[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            x = normalize_pixels(pixels[indexes], device)
            target = torch.from_numpy(targets[indexes]).to(device)
            output = teaching_tokens(encoder(x), projector)
            loss = (distillation_loss(output, target)
                    * torch.from_numpy(weights[indexes]).to(device)).mean()
            require(torch.isfinite(loss), 'Nonfinite distillation loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable),
                    'Nonfinite student gradient')
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indexes)
        history.append({'epoch': epoch, 'weightedMeanLoss': total / len(pixels),
                        'exposureSha256': exposure.hexdigest()})
        (directory / 'progress.json').write_text(json.dumps({
            **identity_fields, 'history': history, 'wallSeconds': time.perf_counter() - started,
        }, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'student': directory.name, 'seed': seed, **history[-1]}), flush=True)
    state = {**{'encoder::' + key: value.detach().cpu().numpy()
                for key, value in encoder.state_dict().items()},
             **{'projector::' + key: value.detach().cpu().numpy()
                for key, value in projector.state_dict().items()}}
    path = directory / 'weights.npz'
    np.savez_compressed(path, **state)
    result = {**identity_fields, 'weights': identity(path), 'trainingFrames': len(pixels),
              'membership': membership, 'history': history, 'batchNormStatisticsFrozen': True,
              'encoderParameters': sum(p.numel() for p in encoder.parameters()),
              'trainingOnlyProjectorParameters': sum(p.numel() for p in projector.parameters()),
              'wallSeconds': time.perf_counter() - started}
    write_immutable(done, result)
    return result


def load_encoder(student, device, checkpoint):
    require(student.get('recipe') == RECIPE, 'Large student recipe differs')
    require(student.get('initialCheckpoint') == _checkpoint_identity(checkpoint),
            'Large student initial checkpoint differs')
    encoder, _ = build_student(student['seed'], device, checkpoint)
    with np.load(verify_once(student['weights']), allow_pickle=False) as payload:
        state = {key[len('encoder::'):]: torch.from_numpy(payload[key].copy())
                 for key in payload.files if key.startswith('encoder::')}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def extract_student_record(encoder, student, record, folder, device):
    """Cache four content-relative 960-channel tokens per original sampled frame."""
    require(student.get('recipe') == RECIPE, 'Large student recipe differs')
    folder = Path(folder)
    path, receipt = folder / (record['id'] + '.npz'), folder / (record['id'] + '.json')
    lineage = {'id': record['id'], 'sourceGroup': record['sourceGroup'],
               'encoder': student['weights'], 'imageInput': record['arrays'],
               'contractSha256': student['contractSha256'], 'recipe': RECIPE,
               'extractionBatchSize': EXTRACTION_BATCH_SIZE}
    if receipt.exists():
        old = read(receipt)
        require(all(old.get(key) == value for key, value in lineage.items()),
                'Student feature resume differs')
        verify_once(old['output'])
        return old
    require(not path.exists(), 'Incomplete student feature output')
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.load(verify_once(record['arrays']['images224']), mmap_mode='r', allow_pickle=False)
    with np.load(verify_once(record['arrays']['timing']), allow_pickle=False) as timing:
        times, pts, quality, boxes = (timing[key].copy()
                                    for key in ('times', 'selected_pts', 'quality', 'boxes'))
    require(len(pixels) == len(times) > 0 and pixels.ndim == 4 and pixels.shape[1] == 3
            and pixels.dtype == np.uint8 and pts.shape == times.shape
            and quality.shape == (len(times), 6) and boxes.shape == (len(times), 4)
            and np.isfinite(times).all() and np.isfinite(pts).all()
            and np.isfinite(quality).all() and np.all(np.diff(times) > 0),
            'Invalid student feature inputs')
    encoder.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), EXTRACTION_BATCH_SIZE):
            stop = start + EXTRACTION_BATCH_SIZE
            spatial = encoder(normalize_pixels(pixels[start:stop], device))
            require(spatial.ndim == 4 and spatial.shape[1] == CHANNELS,
                    'Large student encoder dimension differs')
            pool = torch.from_numpy(mobile.regional_pool_weights(
                boxes[start:stop], *spatial.shape[-2:])).to(device)
            outputs.append(torch.einsum('bchw,brhw->brc', spatial.float(), pool)
                           .cpu().numpy().astype(np.float16))
    tokens = np.concatenate(outputs)
    require(tokens.shape == (len(times), 4, CHANNELS) and np.isfinite(tokens).all(),
            'Invalid Large student tokens')
    np.savez_compressed(path, timestamps=times, tokens=tokens, quality=quality,
                        selected_presentation_times=pts)
    result = {**lineage, 'output': identity(path), 'shape': list(tokens.shape), 'labelsUsed': False}
    write_immutable(receipt, result)
    return result
