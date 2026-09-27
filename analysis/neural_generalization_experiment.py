"""Frozen source-composition fits and label-blind all-video inference.

This runner never chooses a split or an operating point. An immutable task names
the inputs and ordered training/calibration membership before it is executed.
The existing numerical fitting routines remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .neural_context_development import canonical_hash, identity, read, write_immutable
from .neural_development import Example, file_sha256
from .neural_recognition_fit import fit_model, predict
from .recognition_temporal_model import RecognitionConfig, model_for

EPOCHS = (5, 15, 30, 60)
MODELS = {
    'av-tcn': RecognitionConfig(family='av', head='tcn'),
    'dino-tcn': RecognitionConfig(family='dino', head='tcn'),
    'mobile-tcn': RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8),
    'distilled-mobile-tcn': RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8),
    'av-transformer': RecognitionConfig(family='av', head='transformer'),
    'dino-transformer': RecognitionConfig(family='dino', head='transformer'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(reference):
    path = Path(reference['path'])
    require(file_sha256(path) == reference['sha256'], f'Changed input: {path}')
    return path


def family(model):
    return 'distilled' if model == 'distilled-mobile-tcn' else MODELS[model].family


def validate_membership(task, records):
    by_id = {r['id']: r for r in records}
    require(len(by_id) == len(records), 'Duplicate input recording IDs')
    train, calibration = task['trainIds'], task.get('calibrationIds', [])
    require(train and len(set(train)) == len(train), 'Missing/duplicate training members')
    require(len(set(calibration)) == len(calibration), 'Duplicate calibration members')
    require(not (set(train) & set(calibration)), 'Training/calibration IDs overlap')
    require(set(train + calibration) <= set(by_id), 'Unknown task recording')
    train_groups = {by_id[k]['sourceGroup'] for k in train}
    calibration_groups = {by_id[k]['sourceGroup'] for k in calibration}
    require(not (train_groups & calibration_groups), 'Training/calibration source leakage')
    require(not any(by_id[k].get('protected') for k in train + calibration), 'Protected fitting/selection source')
    require(not any(by_id[k].get('environment') == 'beach' for k in train + calibration), 'Beach source')
    require(not (set(task.get('commonEvaluationGroups', [])) & (train_groups | calibration_groups)),
            'Common evaluation panel leaked into fitting/selection')
    require(all(by_id[k]['labelTier'] == 'exact' for k in calibration), 'Calibration requires exact rally gold')
    require(all('calibrate' in by_id[k].get('eligibleRoles', [])
                and by_id[k].get('consent', {}).get('train') is True for k in calibration),
            'Calibration role/consent absent')
    require(any(by_id[k]['labelTier'] == 'exact' for k in train), 'No exact training supervision')
    return by_id


def load_task(path):
    task = read(path)
    require(task['model'] in MODELS and task['epochs'] == list(EPOCHS), 'Unsupported model/checkpoints')
    require(task['seed'] in (3407, 1729, 20260918), 'Unregistered training seed')
    manifest_path, feature_path = verify(task['manifest']), verify(task['features'])
    registration = read(verify(task['registration']))
    require(canonical_hash(registration['contract']) == registration['sha256'], 'Registration digest mismatch')
    require(task['registrationSha256'] == registration['sha256'], 'Task registration mismatch')
    task_without_registration = {k: v for k, v in task.items() if k not in ('registration', 'registrationSha256')}
    require(canonical_hash(task_without_registration) in registration['contract']['taskDigests'], 'Task not registered')
    for reference in registration['contract']['code'].values():
        verify(reference)
    manifest = read(manifest_path)
    by_id = validate_membership(task, manifest['records'])
    return task, manifest_path, feature_path, by_id


def sentinel(config):
    """A label-free one-tick output probe, used only for all-corpus fitting.

    Existing fitters require a nonempty disjoint prediction population. This
    deterministic probe changes no training sample, loss, RNG or selection.
    """
    return Example('__fit_output_probe__', '__reserved_output_probe__', .25,
                   np.asarray([0.], np.float64), np.zeros((1, config.input_dimension), np.float32),
                   np.zeros((1, 4), np.float32), np.ones(1, bool), (), (), 'indoor')


def ordered_rows(data, identifiers):
    by_id = {r.example.id: r for rows in data.values() for r in rows}
    require(set(by_id) == set(identifiers), 'Loaded training population differs')
    grouped = {'exact': [], 'draft': [], 'coverage': []}
    for identifier in identifiers:
        row = by_id[identifier]
        grouped[row.tier].append(row)
    return grouped


def image_indexes(task, identifiers, *, for_training, panel=None):
    from .neural_generalization_inputs import load_images_teachers
    source = task if panel is None else panel
    require(panel is None or not for_training, 'External panel cannot provide teaching targets')
    return load_images_teachers(verify(source['manifest']), verify(source['features']),
                                recording_ids=identifiers, for_training=for_training)


def student_features(task, by_id, folder, identifiers, device, *, fitting=False, panel=None):
    from . import neural_mobile_distillation as student
    from .neural_generalization_inputs import load_data
    require(not fitting or panel is None, 'Panel imagery cannot enter student fitting')
    images, _ = image_indexes(task, identifiers, for_training=False, panel=panel)
    metadata_path = folder/'student/completed.json'
    if fitting:
        training_images, teachers = image_indexes(task, task['trainIds'], for_training=True)
        data = load_data(verify(task['manifest']), verify(task['features']), role='fit',
                         family='av', recording_ids=task['trainIds'])
        ordered = ordered_rows(data, task['trainIds'])
        rows = [r for tier in ('exact', 'draft', 'coverage') for r in ordered[tier]]
        train_groups = {r.example.group for r in rows}
        excluded = {r['sourceGroup'] for r in by_id.values()} - train_groups
        metadata = student.fit_student(rows, excluded, training_images, teachers, task['seed'],
                                      folder/'student', task['registrationSha256'], device)
    else:
        metadata = read(metadata_path)
        require(metadata['contractSha256'] == task['registrationSha256']
                and set(metadata['trainIds']) == set(task['trainIds']), 'Student fitting lineage differs')
    encoder = student.load_encoder(metadata, device)
    result = {identifier: student.extract_student_record(encoder, metadata, images[identifier],
               folder/'student-features', device) for identifier in identifiers}
    del encoder
    return result


def fit_task(task_path, destination, device='cuda'):
    from .neural_generalization_inputs import load_data, load_inference_examples
    task, manifest, features, by_id = load_task(task_path)
    config = MODELS[task['model']]
    needed = list(dict.fromkeys(task['trainIds'] + task.get('calibrationIds', [])))
    adapted = student_features(task, by_id, destination, needed, device, fitting=True) if family(task['model']) == 'distilled' else None
    data = load_data(manifest, features, role='fit', family=family(task['model']),
                     recording_ids=task['trainIds'], student_features=adapted)
    rows = ordered_rows(data, task['trainIds'])
    validation = load_inference_examples(manifest, features, family=family(task['model']),
        recording_ids=task['calibrationIds'], student_features=adapted) if task.get('calibrationIds') else [sentinel(config)]
    scores = fit_model(rows['exact'], {tier: rows[tier] for tier in ('draft', 'coverage')}, validation,
        config, task['seed'], EPOCHS, destination/'temporal', device, task['registrationSha256'], 'short_boost')
    result = {'task': identity(task_path), 'taskId': task['taskId'], 'config': asdict(config),
              'temporal': identity(destination/'temporal/completed.json'),
              'calibrationPredictionIds': [e.id for e in validation],
              'syntheticOutputProbeOnly': not bool(task.get('calibrationIds')),
              'externalLabelsUsed': False, 'selectionPerformed': False}
    if adapted is not None:
        result['student'] = identity(destination/'student/completed.json')
    write_immutable(destination/'fit-result.json', result)
    return result


def load_checkpoint(path, config, device):
    import torch
    model = model_for(config).to(device)
    with np.load(path, allow_pickle=False) as archive:
        state = {key[len('model::'):]: torch.from_numpy(archive[key].copy()).to(device)
                 for key in archive.files if key.startswith('model::')}
        mean, scale = archive['mean'].copy(), archive['scale'].copy()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mean, scale


def infer_task(task_path, destination, identifiers, precision='fp32', device='cuda', panel_path=None):
    from .neural_generalization_inputs import load_inference_examples
    task, manifest, features, by_id = load_task(task_path)
    panel = None
    if panel_path is not None:
        panel = read(panel_path)
        require(panel.get('kind') == 'frozen-independent-inference-panel-v1', 'Unknown independent panel contract')
        manifest, features = verify(panel['manifest']), verify(panel['features'])
        panel_records = read(manifest)['records']
        by_id = {r['id']: r for r in panel_records}
        require(len(by_id) == len(panel_records), 'Duplicate panel recording')
    require(precision in ('fp32', 'fp16', 'int8'), 'Unknown precision')
    require(precision == 'fp32' or family(task['model']) == 'dino', 'Precision transfer is DINO-only')
    require(len(set(identifiers)) == len(identifiers) and set(identifiers) <= set(by_id), 'Invalid inference population')
    fit = read(destination/'fit-result.json')
    require(fit['task'] == identity(task_path), 'Fit task differs')
    completed = read(verify(fit['temporal']))
    adapted = student_features(task, by_id, destination, identifiers, device, panel=panel) if family(task['model']) == 'distilled' else None
    panel_key = 'training-input-population' if panel is None else file_sha256(panel_path)[:16]
    folder = destination/'inference'/panel_key/precision
    folder.mkdir(parents=True, exist_ok=True)
    receipts = []
    # One recording in memory, then all checkpoints. Avoid expanding an entire
    # multi-hour corpus's visual tokens into several gigabytes of float32.
    for identifier in identifiers:
        receipt_path = folder/(identifier+'.json')
        expected = {'task': identity(task_path), 'id': identifier, 'precision': precision,
                    'manifest': identity(manifest), 'features': identity(features), 'labelsUsed': False,
                    'panel': identity(panel_path) if panel is not None else None,
                    'ignoredLabelsUsedForContextOrDecoding': False, 'epochs': list(EPOCHS)}
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(all(receipt.get(k) == v for k, v in expected.items()), 'Inference resume differs')
            verify(receipt['output'])
            require(set(receipt['weights']) == {str(e) for e in EPOCHS}, 'Inference checkpoint inventory differs')
            for epoch in EPOCHS:
                reference = receipt['weights'][str(epoch)]
                expected_path = destination/'temporal'/f'weights-{epoch}.npz'
                require(Path(reference['path']) == expected_path
                        and reference['sha256'] == completed['artifacts'][expected_path.name], 'Inference checkpoint lineage differs')
                verify(reference)
            receipts.append(receipt)
            continue
        example, = load_inference_examples(manifest, features, family=family(task['model']), precision=precision,
                                           recording_ids=[identifier], student_features=adapted)
        require(not example.truth and not example.ignored and example.valid.all(), 'Labels reached raw inference')
        predictions, weight_refs = {}, {}
        for epoch in EPOCHS:
            weight_path = destination/'temporal'/f'weights-{epoch}.npz'
            require(file_sha256(weight_path) == completed['artifacts'][weight_path.name], 'Checkpoint changed')
            model, mean, scale = load_checkpoint(weight_path, MODELS[task['model']], device)
            predictions[f'epoch_{epoch}'] = predict(model, example, mean, scale, MODELS[task['model']], device)
            weight_refs[str(epoch)] = identity(weight_path)
            del model
        path = folder/(identifier+'.npz')
        require(not path.exists(), 'Incomplete inference archive; investigate before resuming')
        np.savez_compressed(path, times=example.times, **predictions)
        receipt = {**expected, 'output': identity(path), 'weights': weight_refs}
        write_immutable(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps({'inferred': identifier, 'taskId': task['taskId'], 'precision': precision}), flush=True)
    return receipts


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit', 'infer'))
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', type=Path, help='JSON array of ordered inference recording IDs')
    parser.add_argument('--panel', type=Path, help='Independent frozen inference inputs; never used for fitting')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUBLAS environment required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.action == 'fit':
        fit_task(args.task, args.output, args.device)
    else:
        require(args.ids is not None, 'Inference requires explicit IDs')
        infer_task(args.task, args.output, json.loads(args.ids.read_text()), args.precision, args.device, args.panel)


if __name__ == '__main__':
    main()
