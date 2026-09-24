"""Fold-local DINO feature distillation with an unchanged mobile rally head."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import mobile_visual_features as mobile
from . import neural_expanded_development as expanded
from . import neural_short_boost_transfer as source
from .distillation_image_inputs import OUTPUT, PREVIOUS, SOURCE, verify, require
from .neural_context_development import canonical_hash, identity, read, write_immutable
from .neural_recognition_development import dataset_contract, serializable_rows
from .neural_recognition_fit import fit_model
from .neural_evaluation import evaluate_predictions
from .neural_recall_operating_point import evaluate_candidates, strict_selection
from .recognition_temporal_model import RecognitionConfig, model_metadata

REPO = Path(__file__).resolve().parents[1]
IMAGES = OUTPUT/'distillation-images-v1'
TEACHERS = OUTPUT/'distillation-teacher-v1'
CHECKPOINT = PREVIOUS/'assets/mobilenet_v3_small-047dcff4.pth'
SEEDS = (3407, 1729, 20260918)
EPOCHS = (5, 15, 30, 60)
CONFIG = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8)
_verified = {}


def verify_once(reference):
    path = Path(reference['path'])
    stat = path.stat()
    state = (reference['sha256'], stat.st_size, stat.st_mtime_ns)
    if path in _verified:
        require(_verified[path] == state, f'Artifact changed during process: {path}')
    else:
        verify(reference)
        _verified[path] = state
    return path


def normalize_pixels(pixels, device):
    # Preserve the historical NumPy FP32 normalization order exactly. CUDA
    # division/normalization introduces small differences before the encoder.
    values = np.asarray(pixels).astype(np.float32)/255.
    values = (values-mobile.RGB_MEAN[None, :, None, None])/mobile.RGB_STD[None, :, None, None]
    return torch.from_numpy(np.ascontiguousarray(values)).to(device)


def build_student(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = mobile.load_mobile_backbone(CHECKPOINT, device=device)
    encoder = backbone.model.features
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    encoder.eval()  # Fixed BatchNorm statistics; gradients still train weights.
    projector = nn.Linear(576, 384).to(device)
    return encoder, projector


def teaching_tokens(spatial, projector):
    # Same overlapping bins as adaptive_avg_pool2d, expressed as a fixed linear
    # map to avoid CUDA's nondeterministic adaptive-pool backward kernel.
    height, width = spatial.shape[-2:]
    weights = spatial.new_zeros((9, height, width))
    for y in range(3):
        for x in range(3):
            y0, y1 = y*height//3, ((y+1)*height+2)//3
            x0, x1 = x*width//3, ((x+1)*width+2)//3
            weights[y*3+x, y0:y1, x0:x1] = 1./((y1-y0)*(x1-x0))
    grid = torch.einsum('bchw,rhw->brc', spatial, weights)
    global_token = spatial.mean((2, 3))[:, None]
    return projector(torch.cat((global_token, grid), dim=1))


def distillation_loss(student, teacher):
    require(student.shape == teacher.shape and student.ndim == 3 and student.shape[1:] == (10, 384),
            'Teacher/student token dimensions differ')
    similarity = F.cosine_similarity(student, teacher, dim=-1, eps=1e-6)
    return .5*(1-similarity[:, 0]) + .5*(1-similarity[:, 1:].mean(1))


def allowed_records(data, excluded):
    train = [r for r in data['exact'] if r.example.group not in excluded]
    auxiliary = expanded.auxiliary_for_fold(data, 'reviewed_export', set(excluded))
    rows = train + [r for records in auxiliary.values() for r in records]
    require(train and rows and not (set(excluded) & {r.example.group for r in rows}), 'Distillation source leakage')
    require(len({r.example.id for r in rows}) == len(rows), 'Duplicate distillation record')
    return train, auxiliary, rows


def load_image_indexes():
    inputs, targets = read(IMAGES/'index.json'), read(TEACHERS/'index.json')
    verify(inputs['registration'])
    verify(targets['registration'])
    require(read(TEACHERS/'registration.json')['inputIndex'] == identity(IMAGES/'index.json'), 'Teacher/image lineage differs')
    images = {r['id']: r for r in inputs['records']}
    teachers = {r['id']: r for r in targets['records']}
    require(len(images) == len(teachers) == 18 and set(images) == set(teachers), 'Distillation population differs')
    for key in images:
        require(images[key]['sourceGroup'] == teachers[key]['sourceGroup']
                and teachers[key]['input'] == images[key]['arrays']['teaching336']
                and teachers[key]['registration'] == identity(TEACHERS/'registration.json'), 'Teacher frame association differs')
    return images, teachers


def training_arrays(rows, images, teachers):
    counts = Counter(r.example.group for r in rows)
    pixels, targets, weights, membership = [], [], [], []
    for row in rows:
        e = row.example
        image, teacher = images[e.id], teachers[e.id]
        require(image['sourceGroup'] == e.group, 'Distillation source group mismatch')
        timing = np.load(verify_once(image['arrays']['timing']), allow_pickle=False)
        indexes = timing['teaching_indexes']
        require(indexes.tolist() == image['contract']['teachingIndexes'], 'Teaching membership changed')
        array = np.load(verify_once(image['arrays']['images224']), mmap_mode='r')
        target = np.load(verify_once(teacher['output']), allow_pickle=False)
        require(target.shape == (len(indexes), 10, 384) and np.isfinite(target).all(), 'Bad teacher values')
        pixels.append(array[indexes].copy())
        targets.append(target)
        weights.extend([1./(counts[e.group]*len(indexes))]*len(indexes))
        membership.append({'id': e.id, 'group': e.group, 'frameIndexes': indexes.tolist(),
                           'teacher': teacher['output'], 'image': image['arrays']['images224']})
        timing.close()
    return np.concatenate(pixels), np.concatenate(targets), np.asarray(weights, np.float32), membership


def fit_student(rows, excluded, images, teachers, seed, directory, digest, device):
    identity_fields = {'contractSha256': digest, 'seed': seed, 'excludedGroups': sorted(excluded),
                       'trainIds': [r.example.id for r in rows], 'trainGroups': sorted({r.example.group for r in rows})}
    require(not (set(excluded) & set(identity_fields['trainGroups'])), 'Held source in student fitting')
    done = directory/'completed.json'
    if done.exists():
        completed = read(done)
        require(all(completed[k] == v for k, v in identity_fields.items()), 'Student resume identity differs')
        verify(completed['weights'])
        return completed
    require(not directory.exists(), f'Incomplete student fit: {directory}')
    directory.mkdir(parents=True)
    pixels, targets, weights, membership = training_arrays(rows, images, teachers)
    weights = weights/weights.mean()
    encoder, projector = build_student(seed, device)
    trainable = list(encoder.parameters())+list(projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history, exposure = [], hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, 9):
        order = rng.permutation(len(pixels))
        exposure.update(order.astype('<i8').tobytes())
        total = 0.
        for start in range(0, len(order), 16):
            indexes = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            x = normalize_pixels(pixels[indexes], device)
            target = torch.from_numpy(targets[indexes]).to(device)
            output = teaching_tokens(encoder(x), projector)
            loss = (distillation_loss(output, target)*torch.from_numpy(weights[indexes]).to(device)).mean()
            require(torch.isfinite(loss), 'Nonfinite distillation loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable), 'Nonfinite student gradient')
            optimizer.step()
            total += float(loss.detach().cpu())*len(indexes)
        history.append({'epoch': epoch, 'weightedMeanLoss': total/len(pixels), 'exposureSha256': exposure.hexdigest()})
        expanded.base.write_json(directory/'progress.json', {'history': history, 'wallSeconds': time.perf_counter()-started})
        print(json.dumps({'student': directory.name, 'seed': seed, **history[-1]}), flush=True)
    state = {**{'encoder::'+k: v.detach().cpu().numpy() for k, v in encoder.state_dict().items()},
             **{'projector::'+k: v.detach().cpu().numpy() for k, v in projector.state_dict().items()}}
    path = directory/'weights.npz'
    np.savez_compressed(path, **state)
    result = {**identity_fields, 'weights': identity(path), 'trainingFrames': len(pixels), 'membership': membership,
              'history': history, 'batchNormStatisticsFrozen': True, 'encoderParameters': sum(p.numel() for p in encoder.parameters()),
              'trainingOnlyProjectorParameters': sum(p.numel() for p in projector.parameters()),
              'wallSeconds': time.perf_counter()-started}
    write_immutable(done, result)
    return result


def load_encoder(student, device):
    encoder, _ = build_student(student['seed'], device)
    with np.load(verify_once(student['weights']), allow_pickle=False) as payload:
        state = {key[len('encoder::'):]: torch.from_numpy(payload[key].copy())
                 for key in payload.files if key.startswith('encoder::')}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def extract_student_record(encoder, student, record, folder, device):
    path, receipt = folder/(record['id']+'.npz'), folder/(record['id']+'.json')
    lineage = {'id': record['id'], 'sourceGroup': record['sourceGroup'], 'encoder': student['weights'],
               'imageInput': record['arrays'], 'contractSha256': student['contractSha256']}
    if receipt.exists():
        old = read(receipt)
        require(all(old[k] == v for k, v in lineage.items()), 'Student feature resume differs')
        verify_once(old['output'])
        return old
    require(not path.exists(), 'Incomplete student feature output')
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.load(verify_once(record['arrays']['images224']), mmap_mode='r')
    with np.load(verify_once(record['arrays']['timing']), allow_pickle=False) as timing:
        times, pts, quality, boxes = (timing[key].copy() for key in ('times', 'selected_pts', 'quality', 'boxes'))
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 64):
            spatial = encoder(normalize_pixels(pixels[start:start+64], device))
            pool = torch.from_numpy(mobile.regional_pool_weights(boxes[start:start+64], *spatial.shape[-2:])).to(device)
            outputs.append(torch.einsum('bchw,brhw->brc', spatial.float(), pool).cpu().numpy().astype(np.float16))
    tokens = np.concatenate(outputs)
    require(tokens.shape == (len(times), 4, 576) and np.isfinite(tokens).all(), 'Invalid student tokens')
    np.savez_compressed(path, timestamps=times, tokens=tokens, quality=quality, selected_presentation_times=pts)
    result = {**lineage, 'output': identity(path), 'shape': list(tokens.shape), 'labelsUsed': False}
    write_immutable(receipt, result)
    return result


def attach_rows(rows, features):
    result = []
    for row in rows:
        e = row.example
        reference = features[e.id]
        require(reference['sourceGroup'] == e.group, 'Student feature group differs')
        path = verify_once(reference['output'])
        with np.load(path, allow_pickle=False) as payload:
            cache = mobile.MobileVisualCache(path, payload['timestamps'], payload['tokens'].astype(np.float32),
                payload['quality'], payload['selected_presentation_times'], {})
            aligned = mobile.align_mobile_features(cache, e.times)
        require(np.all(aligned['available'] == 1), 'Student feature gaps')
        values = np.concatenate((e.values, aligned['tokens'].reshape(len(e.times), -1), aligned['quality'],
                    aligned['feature_age_seconds'][:, None], aligned['available'][:, None]), axis=1).astype(np.float32)
        require(e.values.shape[1] == 104 and values.shape[1] == 2416 and np.array_equal(values[:, :104], e.values)
                and np.isfinite(values).all(), 'Student attachment changed AV inputs')
        result.append(replace(row, example=replace(e, values=values)))
    return result


def prepare_fit(data, excluded, images, teachers, seed, folder, digest, device):
    train, auxiliary, permitted = allowed_records(data, excluded)
    validation = [r for r in data['exact'] if r.example.group in excluded]
    student = fit_student(permitted, excluded, images, teachers, seed, folder/'student', digest, device)
    encoder = load_encoder(student, device)
    needed = permitted+validation
    features = {r.example.id: extract_student_record(encoder, student, images[r.example.id], folder/'features', device) for r in needed}
    del encoder
    torch.cuda.empty_cache()
    train = attach_rows(train, features)
    auxiliary = {tier: attach_rows(rows, features) for tier, rows in auxiliary.items()}
    validation = [r.example for r in attach_rows(validation, features)]
    write_immutable(folder/'features.json', {'student': identity(folder/'student/completed.json'), 'records': list(features.values())})
    return train, auxiliary, validation


def register(output):
    previous = read(PREVIOUS/'mobile-tcn-v1/preregistration.json')
    engineering_path = OUTPUT/'distillation-engineering-v1/report.json'
    engineering = read(engineering_path)
    require(engineering.get('passed') is True and engineering['studentSource'] == identity(__file__)
            and engineering['protocol'] == identity(OUTPUT/'protocol.md'), 'Student engineering gate absent or stale')
    code = {name: identity(Path(__file__).with_name(name)) for name in previous['contract']['code']}
    code.update({name: identity(Path(__file__).with_name(name)) for name in
        ('neural_mobile_distillation.py', 'distillation_image_inputs.py', 'neural_recall_operating_point.py', 'mobile_visual_features.py')})
    require(all(code[name]['sha256'] == sha for name, sha in previous['contract']['code'].items()), 'Frozen numerical dependencies changed')
    groups, population = dataset_contract(read(SOURCE))
    contract = {'kind': 'fold-isolated-dino-distilled-mobile-v1', 'protocol': identity(OUTPUT/'protocol.md'),
                'source': identity(SOURCE), 'images': identity(IMAGES/'index.json'), 'teacherTargets': identity(TEACHERS/'index.json'),
                'mobileCheckpoint': identity(CHECKPOINT), 'controlRegistration': identity(PREVIOUS/'mobile-tcn-v1/preregistration.json'),
                'code': code, 'engineering': identity(engineering_path),
                'groups': groups, 'population': population, 'seeds': list(SEEDS), 'checkpointEpochs': list(EPOCHS),
                'model': model_metadata(CONFIG), 'studentEpochs': 8, 'studentBatchSize': 16, 'studentLearningRate': 1e-4,
                'studentWeightDecay': 1e-4, 'batchNormStatisticsFrozen': True, 'teacherCheckpointSha256': read(TEACHERS/'registration.json')['checkpointSha256'],
                'studentLoss': '.5 global cosine + .5 mean9 spatial cosine; source/record-balanced frame weights',
                'selectionFloor': .99, 'decoderCandidates': expanded.decoder_candidates(), 'targetPaddingSeconds': 2,
                'paddingSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3, 'protectedTestOpened': False, 'productionPromotionAllowed': False}
    registration = {'sha256': canonical_hash(contract), 'contract': contract}
    write_immutable(output/'preregistration.json', registration)
    archive = output/'registered-sources'
    archive.mkdir(parents=True, exist_ok=True)
    for name, reference in code.items():
        dest = archive/name
        content = Path(reference['path']).read_bytes()
        if dest.exists():
            require(dest.read_bytes() == content, 'Source archive changed')
        else:
            with dest.open('xb') as handle:
                handle.write(content)
    return registration


def run_seed(data, registration, images, teachers, seed, output, device):
    result_path = output/f'result-{seed}.json'
    digest, contract = registration['sha256'], registration['contract']
    if result_path.exists():
        result = read(result_path)
        require(result['contractSha256'] == digest and result['seed'] == seed, 'Distillation result resume differs')
        return result
    groups, owners = contract['groups'], {}
    examples = {r.example.id: r.example for r in data['exact']}
    for left, right in combinations(range(4), 2):
        excluded = {groups[left], groups[right]}
        folder = output/'fits'/str(seed)/f'inner-{left}-{right}'
        train, aux, held = prepare_fit(data, excluded, images, teachers, seed, folder, digest, device)
        owners[(left, right)] = fit_model(train, aux, held, CONFIG, seed, EPOCHS, folder/'temporal', device, digest)
    rows, selections = [], []
    for outer_index, outer in enumerate(groups):
        selection_examples = [e for e in examples.values() if e.group != outer]
        probabilities = {epoch: {} for epoch in EPOCHS}
        for inner_index, inner in enumerate(groups):
            if inner == outer:
                continue
            owner = owners[tuple(sorted((outer_index, inner_index)))]
            for epoch in EPOCHS:
                probabilities[epoch].update({key: values for key, values in owner[epoch].items() if examples[key].group == inner})
        candidates = evaluate_candidates(selection_examples, probabilities)
        selection = strict_selection(candidates)
        selection['heldSourceGroup'] = outer
        write_immutable(output/'selections'/f'{seed}-{outer_index}.json', {'contractSha256': digest, 'candidates': candidates, **selection})
        selections.append(selection)
        if not selection['feasible']:
            print(json.dumps({'infeasible': outer, 'seed': seed, 'maximumInnerRecall': selection['maximumInnerRecall']}), flush=True)
            continue
        chosen = selection['selected']
        folder = output/'fits'/str(seed)/f'outer-{outer_index}'
        train, aux, held = prepare_fit(data, {outer}, images, teachers, seed, folder, digest, device)
        scores = fit_model(train, aux, held, CONFIG, seed, (chosen['epoch'],), folder/'temporal', device, digest)[chosen['epoch']]
        rows.extend(e.row(expanded.base.decode(e, scores[e.id], chosen['decoder'])) for e in held)
    complete = all(s['feasible'] for s in selections)
    result = {'contractSha256': digest, 'seed': seed, 'selections': selections, 'completeEvaluation': complete,
              'evaluation': evaluate_predictions(rows) if complete else None, 'predictions': serializable_rows(rows),
              'partialScopeNotRankable': not complete, 'protectedTestOpened': False}
    write_immutable(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT/'distilled-mobile-v1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    require(not args.device.startswith('cuda') or os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUBLAS environment required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    registration = register(args.output)
    if args.register_only:
        print(json.dumps({'registration': registration['sha256']}))
        return
    images, teachers = load_image_indexes()
    data = source.load_data(SOURCE, None, with_dino=False)
    for seed in ((args.seed,) if args.seed else SEEDS):
        run_seed(data, registration, images, teachers, seed, args.output, args.device)
    paths = [args.output/f'result-{seed}.json' for seed in SEEDS]
    if all(path.exists() for path in paths):
        write_immutable(args.output/'report.json', {'contractSha256': registration['sha256'],
            'results': [read(path) for path in paths], 'protectedTestOpened': False})


if __name__ == '__main__':
    main()
