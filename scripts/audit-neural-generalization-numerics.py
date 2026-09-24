#!/usr/bin/env python3
"""Independent CPU checks of fitting exposure and sampled saved neural outputs.

No optimizer or training code runs. Training scalers need AV and quality scalars,
not complete visual-token matrices. Inference replay loads one recording at a time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis.neural_generalization_experiment import load_task, MODELS, family, sentinel
from analysis.neural_generalization_inputs import load_data, load_inference_examples, load_images_teachers, feature_entries
from analysis.neural_recall_sweep import read, identity, require, write_new, EPOCHS
from analysis.recognition_temporal_model import model_for, model_metadata, RecognitionConfig

PRIOR_STUDENT = Path(private_value('private-reference-0090'))
PRIOR_STUDENT_SHA = '7fa2f5febdbbf15c5f0140adb811729efb3464687dc2b8e8ec4e7a7261beb387'
STUDENT_AUDITOR_SHA = 'd8892b6335569e9622f3f3fa659c4e2f93861084c92cf104535084355683bdff'
STUDENT_HELPER_SHA = '090167d06859469be5ae1d06ee7eddcb7a380c28f2d74d6838a967ac4b09000e'


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), REPO/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    # Existing independent auditor pins its own tensor/interval helper sources.
    path = REPO/'scripts/audit-neural-recognition.py'
    require(identity(path)['sha256'] == STUDENT_AUDITOR_SHA, 'Prior independent numerical auditor changed')
    old = script('audit-neural-recognition.py')
    tensor, _ = old.helpers()
    require(identity(REPO/'scripts/audit-neural-mobile-distillation.py')['sha256'] == STUDENT_HELPER_SHA,
            'Prior independent student auditor changed')
    student = script('audit-neural-mobile-distillation.py')
    return old, tensor, student


def fit_owner(task_path, directory, evidence):
    task, manifest, features, by_id = load_task(task_path)
    fit_ref = evidence.capture(directory/'fit-result.json')
    fit = read(fit_ref['path'])
    require(fit['task'] == evidence.capture(task_path) and fit['taskId'] == task['taskId'], 'Wrong fit owner')
    require(fit['config'] == asdict(MODELS[task['model']]) and fit['externalLabelsUsed'] is False
            and fit['selectionPerformed'] is False, 'Fit recipe or label-use declaration differs')
    require(fit['calibrationPredictionIds'] == (task['calibrationIds'] or ['__fit_output_probe__'])
            and fit['syntheticOutputProbeOnly'] == (not bool(task['calibrationIds'])), 'Fit calibration population differs')
    require(Path(fit['temporal']['path']) == directory/'temporal/completed.json', 'Wrong temporal owner path')
    meta = read(evidence.bind(fit['temporal']))
    for reference in (task['manifest'], task['features'], task['registration']):
        evidence.bind(reference)
    for reference in read(task['registration']['path'])['contract']['code'].values():
        evidence.bind(reference)
    return task, manifest, features, by_id, fit, meta, fit_ref


def bind_helpers(old, evidence):
    evidence.capture(REPO/'scripts/audit-neural-recognition.py')
    evidence.capture(REPO/'scripts/audit-neural-mobile-distillation.py')
    for name, sha in old.HELPERS.items():
        evidence.bind({'path': str(REPO/'scripts'/name), 'sha256': sha})


def bind_feature_entries(entries, model, precision, evidence):
    for entry in entries.values():
        evidence.bind(entry['audiovisual'])
        if family(model) == 'dino':
            evidence.bind(entry['dino'][precision])
        elif family(model) == 'mobile':
            evidence.bind(entry['mobile'])
        elif family(model) == 'distilled':
            image = read(evidence.bind(entry['imageInput']))
            for reference in image['arrays'].values():
                evidence.bind(reference)


def student_receipts(directory, identifiers, evidence):
    result = {}
    for identifier in identifiers:
        receipt = read(evidence.capture(directory/'student-features'/(identifier+'.json'))['path'])
        require(receipt['id'] == identifier and receipt['labelsUsed'] is False, 'Wrong student feature identity')
        evidence.bind(receipt['output'])
        result[identifier] = receipt
    return result


def quality_scalars(example, reference, evidence):
    with np.load(evidence.bind(reference), allow_pickle=False) as archive:
        times, quality = archive['timestamps'], archive['quality']
    indexes = np.searchsorted(times, example.times, side='right')-1
    require(np.all(indexes >= 0), 'Quality scalars have no causal visual input')
    ages = example.times-times[indexes]
    require(np.all((ages >= 0) & (ages < .5+1e-9)), 'Visual age exceeds fixed contract')
    return np.concatenate((quality[indexes], ages.astype(np.float32)[:, None],
                           np.ones((len(ages), 1), np.float32)), axis=1).astype(np.float32)


def independent_scaler(rows, config, entries, adapted, old, evidence):
    mean, scale = old.independent_scaler(rows, RecognitionConfig(family='av', head='tcn'))
    if config.scalar_dimension:
        values = []
        for row in rows:
            reference = adapted[row.example.id]['output'] if adapted is not None else entries[row.example.id]['mobile']
            values.append(quality_scalars(row.example, reference, evidence)[row.example.valid])
        joined = np.concatenate(values)
        mean = np.r_[mean, joined.mean(0, dtype=np.float64).astype(np.float32)]
        scale = np.r_[scale, np.maximum(joined.std(0, dtype=np.float64), 1e-4).astype(np.float32)]
    return mean, scale


def audit_student(task, directory, rows, by_id, images, teachers, prior, student_helper, evidence):
    import torch
    from analysis import mobile_visual_features as mobile
    meta_ref = evidence.capture(directory/'student/completed.json')
    meta = read(meta_ref['path'])
    permitted = [r for tier in ('exact', 'draft', 'coverage') for r in rows[tier]]
    groups = {r.example.group for r in permitted}
    expected = {'contractSha256': task['registrationSha256'], 'seed': task['seed'],
        'trainIds': [r.example.id for r in permitted], 'trainGroups': sorted(groups),
        'excludedGroups': sorted({r['sourceGroup'] for r in by_id.values()}-groups)}
    for key, value in expected.items():
        require(meta[key] == value, 'Student training/source membership differs: '+key)
    membership, frames = [], 0
    for row in permitted:
        image, teacher = images[row.example.id], teachers[row.example.id]
        for reference in image['arrays'].values():
            evidence.bind(reference)
        require(teacher['input'] == image['arrays']['teaching336'] and teacher['sourceGroup'] == row.example.group,
                'Teacher source/frame association differs')
        with np.load(evidence.bind(image['arrays']['timing']), allow_pickle=False) as timing:
            indexes = timing['teaching_indexes']
            expected_indexes = student_helper.independent_teaching_indexes(timing['times'], row.example.times, row.example.valid)
            require(np.array_equal(indexes, expected_indexes) and indexes.tolist() == image['contract']['teachingIndexes'],
                    'Teacher selection used a different valid population')
        target = np.load(evidence.bind(teacher['output']), allow_pickle=False)
        require(target.shape == (len(indexes), 10, 384) and np.isfinite(target).all(), 'Invalid teaching values')
        membership.append({'id': row.example.id, 'group': row.example.group, 'frameIndexes': indexes.tolist(),
            'teacher': teacher['output'], 'image': image['arrays']['images224']})
        frames += len(indexes)
    require(meta['membership'] == membership and meta['trainingFrames'] == frames, 'Student target membership differs')
    require(meta['batchNormStatisticsFrozen'] and meta['encoderParameters'] == 927008
        and meta['trainingOnlyProjectorParameters'] == 221568 and len(meta['history']) == 8, 'Student architecture/epoch contract differs')
    rng, exposure = np.random.default_rng(task['seed']), hashlib.sha256()
    for epoch, history in enumerate(meta['history'], 1):
        exposure.update(rng.permutation(frames).astype('<i8').tobytes())
        require(history['epoch'] == epoch and history['exposureSha256'] == exposure.hexdigest()
            and np.isfinite(history['weightedMeanLoss']) and -.00001 <= history['weightedMeanLoss'] <= 2.00001,
            'Student sample exposure/history differs')
    checkpoint = prior['contract']['mobileCheckpoint']; evidence.bind(checkpoint)
    encoder = mobile.load_mobile_backbone(checkpoint['path'], checkpoint_sha256=checkpoint['sha256'], device='cpu').model.features.eval()
    template = encoder.state_dict(); state = {}; changed = 0; bn_count = 0
    require(Path(meta['weights']['path']) == directory/'student/weights.npz', 'Student weight owner path differs')
    with np.load(evidence.bind(meta['weights']), allow_pickle=False) as archive:
        require(set(archive.files) == {*('encoder::'+k for k in template), 'projector::weight', 'projector::bias'}, 'Student weight inventory differs')
        for name, original in template.items():
            value = archive['encoder::'+name]
            require(value.dtype == original.numpy().dtype and value.shape == tuple(original.shape)
                    and np.isfinite(value).all(), 'Invalid student tensor')
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                require(np.array_equal(value, original.numpy()), 'Student BatchNorm statistics changed')
                bn_count += 1
            elif not np.array_equal(value, original.numpy()):
                changed += 1
            state[name] = torch.from_numpy(value.copy())
        require(archive['projector::weight'].shape == (384, 576) and archive['projector::bias'].shape == (384,)
                and np.isfinite(archive['projector::weight']).all() and np.isfinite(archive['projector::bias']).all(), 'Invalid student projector')
    require(changed > 0, 'Student learned no encoder tensors')
    encoder.load_state_dict(state, strict=True); encoder.eval()
    return encoder, {'completed': meta_ref, 'weights': meta['weights'], 'trainIds': meta['trainIds'],
        'frames': frames, 'exposureSha256': exposure.hexdigest(), 'unchangedBNBuffers': bn_count,
        'changedEncoderTensors': changed, 'initialCheckpoint': checkpoint}


def replay_student(identifier, receipt, image, encoder, weights, contract, evidence):
    import torch
    from analysis import mobile_visual_features as mobile
    require(receipt['encoder'] == weights and receipt['imageInput'] == image['arrays']
            and receipt['sourceGroup'] == image['sourceGroup'] and receipt['contractSha256'] == contract,
            'Student feature lineage differs')
    with np.load(evidence.bind(receipt['output']), allow_pickle=False) as output, \
         np.load(evidence.bind(image['arrays']['timing']), allow_pickle=False) as timing:
        require(set(output.files) == {'timestamps', 'tokens', 'quality', 'selected_presentation_times'}, 'Student feature schema differs')
        tokens, times = output['tokens'], output['timestamps']
        require(tokens.dtype == np.float16 and tokens.shape == (len(times), 4, 576) and np.isfinite(tokens).all(), 'Invalid student tokens')
        require(np.array_equal(times, timing['times']) and np.array_equal(output['quality'], timing['quality'])
                and np.array_equal(output['selected_presentation_times'], timing['selected_pts']), 'Student timing/quality differs')
        selected = sorted({0, len(times)//2, len(times)-1})
        pixels = np.load(evidence.bind(image['arrays']['images224']), mmap_mode='r')
        x = (pixels[selected].astype(np.float32)/255.-mobile.RGB_MEAN[None, :, None, None])/mobile.RGB_STD[None, :, None, None]
        with torch.inference_mode():
            spatial = encoder(torch.from_numpy(x))
            pool = torch.from_numpy(mobile.regional_pool_weights(timing['boxes'][selected], *spatial.shape[-2:]))
            replay = torch.einsum('bchw,brhw->brc', spatial, pool).numpy()
        expected = tokens[selected].astype(np.float32)
        bound = .5*np.spacing(np.abs(tokens[selected])).astype(np.float32)+2e-4+1e-4*np.abs(expected)
        error = np.abs(replay-expected)
        require(np.all(error <= bound), 'Independent student encoder replay differs: '+identifier)
        return {'id': identifier, 'output': receipt['output'], 'sampledFrames': selected,
                'maximumAbsoluteDifference': float(error.max()), 'maximumToleranceRatio': float((error/bound).max())}


def checkpoint(folder, epoch, config, mean, scale, meta, evidence):
    import torch
    model = model_for(config).cpu().eval(); template = model.state_dict()
    name = f'weights-{epoch}.npz'; reference = {'path': str(folder/name), 'sha256': meta['artifacts'][name]}
    with np.load(evidence.bind(reference), allow_pickle=False) as payload:
        require(set(payload.files) == {'mean', 'scale', *('model::'+n for n in template)}, 'Temporal tensor inventory differs')
        for key, expected in (('mean', mean), ('scale', scale)):
            require(payload[key].dtype == np.float32 and np.array_equal(payload[key], expected), 'Independent training scaler differs')
        state = {}
        for key, value in template.items():
            actual = payload['model::'+key]
            require(actual.dtype == np.float32 and actual.shape == tuple(value.shape) and np.isfinite(actual).all(), 'Invalid temporal tensor')
            state[key] = torch.from_numpy(actual.copy())
        model.load_state_dict(state, strict=True)
    return model, reference


def audit_fit(task_path, directory, output):
    old, tensor, student_helper = helpers(); evidence = student_helper.Evidence()
    bind_helpers(old, evidence)
    task, manifest, features, by_id, fit, meta, fit_ref = fit_owner(task_path, directory, evidence)
    config = MODELS[task['model']]
    rows = load_data(manifest, features, family='av', recording_ids=task['trainIds'])
    # Preserve the task's registered sampling order, independently of manifest order.
    row_map = {r.example.id: r for values in rows.values() for r in values}
    rows = {tier: [row_map[k] for k in task['trainIds'] if row_map[k].tier == tier] for tier in rows}
    needed = task['trainIds']+task['calibrationIds']
    entries = feature_entries([by_id[k] for k in needed], features)
    bind_feature_entries(entries, task['model'], 'fp32', evidence)
    adapted = student_receipts(directory, needed, evidence) if task['model'] == 'distilled-mobile-tcn' else None
    student_check, encoder, images = None, None, None
    if adapted is not None:
        prior = read(evidence.bind({'path': str(PRIOR_STUDENT), 'sha256': PRIOR_STUDENT_SHA}))
        images, teachers = load_images_teachers(manifest, features, recording_ids=task['trainIds'], for_training=True)
        for identifier in task['trainIds']:
            evidence.bind(entries[identifier]['teacherTargets'])
            evidence.bind(teachers[identifier]['registration'])
        encoder, student_check = audit_student(task, directory, rows, by_id, images, teachers, prior, student_helper, evidence)
        require(fit['student'] == student_check['completed'], 'Fit/student owner differs')
        all_images, _ = load_images_teachers(manifest, features, recording_ids=needed, for_training=False)
        student_check['featureReplay'] = [replay_student(key, adapted[key], all_images[key], encoder,
            student_check['weights'], task['registrationSha256'], evidence) for key in needed]
        del encoder
    expected = {'contractSha256': task['registrationSha256'], 'kind': f'{config.family}_{config.head}', 'seed': task['seed'],
        'epochs': list(EPOCHS), 'lossArm': 'short_boost', 'model': model_metadata(config),
        'trainIds': [r.example.id for r in rows['exact']], 'auxiliaryIds': {k: [r.example.id for r in rows[k]] for k in ('draft', 'coverage')},
        'trainGroups': sorted({r.example.group for r in rows['exact']}),
        'auxiliaryGroups': {k: sorted({r.example.group for r in rows[k]}) for k in ('draft', 'coverage')},
        'validationIds': task['calibrationIds'] or ['__fit_output_probe__'],
        'validationGroups': sorted({by_id[k]['sourceGroup'] for k in task['calibrationIds']}) if task['calibrationIds'] else ['__reserved_output_probe__']}
    for key, value in expected.items():
        require(meta[key] == value, 'Temporal fit membership/recipe differs: '+key)
    require(meta['scalerTrainIds'] == expected['trainIds'], 'Scaler training scope differs')
    counts = {tier: old.independent_counts(values) for tier, values in rows.items()}
    require(meta['supervisedCounts'] == counts, 'Supervision counts differ')
    positives = np.asarray(counts['exact']['positiveMass'])
    weights = np.minimum(20., np.sqrt((np.asarray(counts['exact']['valid'])-positives)/np.maximum(positives, 1.)))
    require(meta['positiveWeight'] == weights.tolist(), 'Positive class weights differ')
    from analysis.neural_short_boost_weighting import live_event_weights
    require(meta['liveLossWeighting'] == {'mode': 'short_boost', 'rows': {
        tier: [live_event_weights(row, 'short_boost')[1] for row in records] for tier, records in rows.items()}}, 'Loss weighting differs')
    inputs = {r.example.id: {'group': r.example.group, 'valid': r.example.valid} for values in rows.values() for r in values}
    supervision = {r.example.id: {'mask': r.mask} for values in rows.values() for r in values}
    pools = {tier: tensor.chunk_geometry([r.example.id for r in values], inputs, supervision) for tier, values in rows.items()}
    exposure = tensor.independent_exposure(pools, task['seed'], max(EPOCHS)); tensor.validate_history(meta, exposure)
    mean, scale = independent_scaler(rows['exact'], config, entries, adapted, old, evidence)
    replayed = []
    old.verify_artifacts(directory/'temporal', meta)
    for epoch in EPOCHS:
        model, _ = checkpoint(directory/'temporal', epoch, config, mean, scale, meta, evidence)
        score_name = f'predictions-{epoch}.npz'
        with np.load(evidence.bind({'path': str(directory/'temporal'/score_name), 'sha256': meta['artifacts'][score_name]}), allow_pickle=False) as scores:
            require(set(scores.files) == set(expected['validationIds']), 'Calibration score population differs')
            for identifier in expected['validationIds']:
                example = sentinel(config) if identifier == '__fit_output_probe__' else load_inference_examples(manifest, features,
                    family=family(task['model']), recording_ids=[identifier], student_features=adapted)[0]
                require(example.valid.all() and not example.truth and not example.ignored, 'Calibration forward received label masks')
                value = scores[identifier]
                require(value.dtype == np.float32 and value.shape == (len(example.times), 4)
                    and np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)), 'Invalid calibration scores')
                replayed.append({'epoch': epoch, **old.replay_sample(model, example, value, mean, scale, config)})
        del model
    result = {'kind': 'independent-generalization-fit-numerical-audit-v1', 'passed': True,
        'task': identity(task_path), 'taskId': task['taskId'], 'fitResult': fit_ref, 'completed': fit['temporal'],
        'student': student_check, 'checkpointEpochs': list(EPOCHS), 'optimizerSteps': exposure[-1]['optimizerSteps'],
        'scalerTrainIds': expected['trainIds'], 'sourceMembers': expected, 'cpuCalibrationReplay': replayed,
        'initialStudentRegistration': {'path': str(PRIOR_STUDENT), 'sha256': PRIOR_STUDENT_SHA} if adapted is not None else None,
        'evidence': evidence.finish(), 'auditor': identity(__file__), 'trainingPerformed': False, 'gpuUsed': False,
        'scope': 'Independent membership/scaler/count/sampling replay, checkpoint arrays and first/middle/last full-context calibration chunks at all4epochs; sampled student features/BN/teaching membership. No optimizer replay.'}
    write_new(output, result)
    return result


def score_schema(value, times):
    require(value.dtype == np.float32 and value.shape == (len(times), 4)
            and np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)), 'Invalid saved probability array')


def inference_owner(receipt, task_ref, panel_ref, panel, identifier, precision, directory, meta):
    expected = {'task': task_ref, 'id': identifier, 'precision': precision,
        'manifest': panel['manifest'], 'features': panel['features'], 'labelsUsed': False,
        'panel': panel_ref, 'ignoredLabelsUsedForContextOrDecoding': False, 'epochs': list(EPOCHS)}
    require(all(receipt.get(key) == value for key, value in expected.items()), 'Inference owner or label use differs')
    require(set(receipt['weights']) == {str(e) for e in EPOCHS}, 'Inference checkpoint inventory differs')
    for epoch in EPOCHS:
        name = f'weights-{epoch}.npz'
        require(receipt['weights'][str(epoch)] == {'path': str(directory/'temporal'/name), 'sha256': meta['artifacts'][name]},
                'Inference checkpoint owner differs')


def audit_inference(task_path, directory, panel_path, precision, fit_audit_path, output):
    import torch
    old, _, student_helper = helpers(); evidence = student_helper.Evidence()
    bind_helpers(old, evidence)
    task, _, _, _, fit, meta, fit_ref = fit_owner(task_path, directory, evidence)
    fit_audit_ref = evidence.capture(fit_audit_path); fit_audit = read(fit_audit_ref['path'])
    require(fit_audit['kind'] == 'independent-generalization-fit-numerical-audit-v1' and fit_audit['passed']
        and fit_audit['task'] == identity(task_path) and fit_audit['fitResult'] == fit_ref
        and fit_audit['completed'] == fit['temporal'] and fit_audit['auditor'] == identity(__file__), 'Fit numerical audit differs')
    for reference in fit_audit['evidence']:
        evidence.bind(reference)
    panel_ref = evidence.capture(panel_path); panel = read(panel_path)
    require(panel['kind'] == 'frozen-independent-inference-panel-v1' and panel['inferenceUsesLabels'] is False
        and panel['allInferenceTicksValid'] is True and precision in panel['precisionVariants'], 'Inference panel contract differs')
    manifest = evidence.bind(panel['manifest']); features = evidence.bind(panel['features'])
    feature_audit = read(evidence.bind(panel['featureAudit']))
    require(feature_audit['kind'] == 'independent-expansion-feature-audit-v1' and feature_audit['passed']
        and feature_audit['inputs'] == panel['manifest'] and feature_audit['features'] == panel['features']
        and feature_audit['auditedRecordingCount'] == 42 and feature_audit['protectedAndPanelTeacherTargetsAbsent'],
        'Independent feature qualification differs')
    evidence.bind(panel['inventory']); evidence.bind(panel['registrar'])
    records = read(manifest)['records']; by_id = {r['id']: r for r in records}
    identifiers = panel['recordingIds']
    require(len(identifiers) == len(set(identifiers)) == len(by_id) == 42
        and set(identifiers) == set(by_id) and all('infer' in r['eligibleRoles'] for r in records), 'Incomplete all-video population')
    require(precision == 'fp32' or family(task['model']) == 'dino', 'Non-DINO precision transfer')
    entries = feature_entries(records, features)
    bind_feature_entries(entries, task['model'], precision, evidence)
    config = MODELS[task['model']]
    with np.load(evidence.bind({'path': str(directory/'temporal/weights-5.npz'),
                               'sha256': meta['artifacts']['weights-5.npz']}), allow_pickle=False) as weights:
        mean, scale = weights['mean'].copy(), weights['scale'].copy()
    models = {epoch: checkpoint(directory/'temporal', epoch, config, mean, scale, meta, evidence)[0] for epoch in EPOCHS}
    adapted, encoder, images = None, None, None
    if task['model'] == 'distilled-mobile-tcn':
        from analysis import mobile_visual_features as mobile
        require(fit_audit['student'] is not None, 'Distilled fit lacks student audit')
        adapted = student_receipts(directory, identifiers, evidence)
        images, teachers = load_images_teachers(manifest, features, recording_ids=identifiers, for_training=False)
        require(not teachers, 'Inference panel supplied teaching targets')
        initial = fit_audit['student']['initialCheckpoint']; evidence.bind(initial)
        encoder = mobile.load_mobile_backbone(initial['path'], checkpoint_sha256=initial['sha256'], device='cpu').model.features.eval()
        with np.load(evidence.bind(fit_audit['student']['weights']), allow_pickle=False) as archive:
            state = {key[len('encoder::'):]: torch.from_numpy(archive[key].copy())
                     for key in archive.files if key.startswith('encoder::')}
        encoder.load_state_dict(state, strict=True); encoder.eval()
    folder = directory/'inference'/panel_ref['sha256'][:16]/precision
    require({p.stem for p in folder.glob('*.json')} == set(identifiers), 'Inference receipt inventory differs')
    require({p.stem for p in folder.glob('*.npz')} == set(identifiers), 'Inference score inventory differs')
    checks, receipts, student_checks = [], [], []
    for identifier in identifiers:
        reference = evidence.capture(folder/(identifier+'.json')); receipt = read(reference['path'])
        inference_owner(receipt, identity(task_path), panel_ref, panel, identifier, precision, directory, meta)
        require(Path(receipt['output']['path']) == folder/(identifier+'.npz'), 'Inference score owner path differs')
        if adapted is not None:
            student_checks.append(replay_student(identifier, adapted[identifier], images[identifier], encoder,
                fit_audit['student']['weights'], task['registrationSha256'], evidence))
        example, = load_inference_examples(manifest, features, family=family(task['model']), precision=precision,
                                           student_features=adapted, recording_ids=[identifier])
        require(example.valid.all() and not example.truth and not example.ignored
            and np.all(example.targets == 0), 'Labels reached numerical inference replay')
        with np.load(evidence.bind(receipt['output']), allow_pickle=False) as archive:
            require(set(archive.files) == {'times', *(f'epoch_{e}' for e in EPOCHS)}
                and np.array_equal(archive['times'], example.times), 'Inference schema or full timeline differs')
            for epoch in EPOCHS:
                value = archive[f'epoch_{epoch}']; score_schema(value, example.times)
                checks.append({'epoch': epoch, **old.replay_sample(models[epoch], example, value, mean, scale, config)})
        receipts.append(reference)
        print(json.dumps({'cpuAudited': identifier, 'taskId': task['taskId'], 'precision': precision}), flush=True)
        del example
    result = {'kind': 'independent-generalization-inference-numerical-audit-v1', 'passed': True,
        'task': identity(task_path), 'taskId': task['taskId'], 'fitResult': fit_ref, 'completed': fit['temporal'],
        'panel': panel_ref, 'fitAudit': fit_audit_ref, 'precision': precision, 'checkpointEpochs': list(EPOCHS),
        'inferenceReceipts': receipts, 'cpuReplay': checks, 'studentFeatureReplay': student_checks,
        'evidence': evidence.finish(), 'auditor': identity(__file__), 'trainingPerformed': False, 'gpuUsed': False,
        'scope': 'All42 recordings and all4 checkpoint arrays; CPU first/middle/last real-context chunks per recording/checkpoint, sampled student features. Full timeline, no gold masks; not exhaustive neural replay.',
        'cpuReplayTolerance': {'rtol': old.REPLAY_RTOL, 'atol': old.REPLAY_ATOL}}
    write_new(output, result)
    return result


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('fit', 'inference'), default='fit')
    p.add_argument('--panel', type=Path)
    p.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    p.add_argument('--fit-audit', type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    require(str(a.output).startswith(private_value('private-reference-0060')), 'Numerical audit output must be on NAS')
    if a.phase == 'fit':
        audit_fit(a.task, a.fit, a.output)
    else:
        require(a.panel is not None and a.fit_audit is not None, 'Inference audit requires panel and fit audit')
        audit_inference(a.task, a.fit, a.panel, a.precision, a.fit_audit, a.output)
    print(json.dumps({'passed': True, 'output': str(a.output)}))


if __name__ == '__main__':
    main()
