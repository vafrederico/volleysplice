"""Preregistered exact-training reuse; no calibration decision is reusable.

The registered fitter and its numerical auditors remain unchanged. This adapter
materializes ordinary fit/inference artifacts and adds explicit execution lineage.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import shutil
import time

import numpy as np

from .neural_context_development import canonical_hash, identity, read, write_immutable
from .neural_development import file_sha256

REPO = Path(__file__).resolve().parents[1]
TIERS = ('exact', 'draft', 'coverage')
EPOCHS = (5, 15, 30, 60)


def require(value, message):
    if not value:
        raise ValueError(message)


class Bindings:
    """Hash each input once and fail if any changes during materialization."""
    def __init__(self):
        self.entries = {}

    def bind(self, reference):
        path = Path(reference['path'])
        stat = path.stat()
        state = (reference['sha256'], stat.st_size, stat.st_mtime_ns)
        if str(path) in self.entries:
            require(self.entries[str(path)][1] == state, f'Input changed: {path}')
        else:
            require(file_sha256(path) == reference['sha256'], f'Input hash differs: {path}')
            self.entries[str(path)] = (dict(reference), state)
        return path

    def capture(self, path):
        reference = identity(path)
        self.bind(reference)
        return reference

    def finish(self):
        for reference, _ in list(self.entries.values()):
            self.bind(reference)
        return [reference for reference, _ in self.entries.values()]


def training_recipe(task, records, configurations):
    """Deliberately excludes validation membership, variant and split draw."""
    by_id = {r['id']: r for r in records}
    require(len(by_id) == len(records), 'Duplicate manifest IDs')
    ids = task['trainIds']
    require(ids and len(set(ids)) == len(ids) and set(ids) <= set(by_id), 'Bad ordered training IDs')
    tiers = {tier: [key for key in ids if by_id[key]['labelTier'] == tier] for tier in TIERS}
    require(sum(map(len, tiers.values())) == len(ids), 'Unsupported training tier')
    return {'registration': task['registration'], 'registrationSha256': task['registrationSha256'],
        'manifest': task['manifest'], 'features': task['features'], 'model': task['model'],
        'config': configurations[task['model']], 'seed': task['seed'], 'epochs': task['epochs'],
        'lossArm': task['lossArm'], 'trainIds': list(ids), 'tierOrderedIds': tiers,
        'trainGroups': sorted({by_id[key]['sourceGroup'] for key in ids}),
        'excludedStudentGroups': sorted({r['sourceGroup'] for r in records} - {by_id[key]['sourceGroup'] for key in ids})}


def first_owners(tasks, records, configurations):
    seen, result = {}, []
    for task in tasks:
        digest = canonical_hash(training_recipe(task, records, configurations))
        owner = seen.setdefault(digest, task['taskId'])
        result.append((task['taskId'], digest, owner))
    return result


def fit_directory(registration_directory, task):
    if task['variant'] == 'original-corpus':
        return registration_directory/'fits'/task['model']/f'seed-{task["seed"]}'
    return registration_directory/'fits'/task['variant']/task['model']/f'split-{task["splitSeed"]}'


def register(registration_directory, protocol, qualification):
    from .neural_generalization_experiment import MODELS, load_task
    require(registration_directory.resolve().is_relative_to('/mnt/freenas'), 'Reuse registration must be on NAS')
    registration_path = registration_directory/'registration.json'
    registration = read(registration_path)
    require(canonical_hash(registration['contract']) == registration['sha256'], 'Registration changed')
    evidence = Bindings()
    evidence.capture(registration_path)
    qualification_ref = evidence.capture(qualification)
    qualified = read(qualification)
    require(qualified.get('kind') == 'generalization-validation-independence-qualification-v1'
            and qualified.get('passed') is True and qualified.get('device') == 'cpu', 'CPU parity qualification absent')
    for reference in qualified['evidence']:
        evidence.bind(reference)
    configurations = {key: asdict(value) for key, value in MODELS.items()}
    required_configs = {canonical_hash(value) for value in configurations.values()}
    require({canonical_hash(r['config']) for r in qualified['architectures']} == required_configs
            and all(r['checkpointHistoryAndSharedScoresBitExact'] is True for r in qualified['architectures']),
            'CPU qualification does not cover all distinct architectures')
    tasks, paths = [], {}
    for task_id in registration['contract']['taskIds']:
        path = registration_directory/'tasks'/(task_id.replace('/', '__')+'.json')
        task, manifest, _, _ = load_task(path)
        require(task['taskId'] == task_id and task['registration'] == identity(registration_path), 'Task owner differs')
        tasks.append(task); paths[task_id] = path
    require(tasks and len({t['taskId'] for t in tasks}) == len(tasks), 'Duplicate tasks')
    require(len({canonical_hash(t['manifest']) for t in tasks}) == 1, 'Mixed manifests in registration')
    rows = read(manifest)['records']
    by_id = {task['taskId']: task for task in tasks}
    mappings = [{'task': identity(paths[key]), 'taskId': key, 'recipeSha256': digest,
        'physicalOwnerTaskId': owner, 'physicalOwnerTask': identity(paths[owner]),
        'fitDirectory': str(fit_directory(registration_directory, by_id[key])),
        'physicalOwnerFitDirectory': str(fit_directory(registration_directory, by_id[owner]))}
        for key, digest, owner in first_owners(tasks, rows, configurations)]
    owners = [row for row in mappings if row['taskId'] == row['physicalOwnerTaskId']]
    sources = ['analysis/neural_generalization_reuse.py', 'scripts/register-neural-generalization-reuse.py',
        'scripts/reuse-neural-generalization-fit.py', 'scripts/reuse-neural-generalization-inference.py',
        'scripts/audit-neural-generalization-reuse.py',
        'scripts/audit-neural-generalization-numerics.py',
        'scripts/run-neural-generalization-reuse-queue.py',
        'scripts/qualify-neural-generalization-reuse.py', 'analysis/tests/test_generalization_reuse.py']
    plan = {'kind': 'generalization-training-reuse-plan-v1', 'registration': identity(registration_path),
        'registrationSha256': registration['sha256'], 'protocol': evidence.capture(protocol),
        'qualification': qualification_ref, 'code': {name: evidence.capture(REPO/name) for name in sources},
        'tasks': mappings, 'counts': {'logicalHeadFits': len(tasks), 'physicalHeadFits': len(owners),
            'reusedHeadFits': len(tasks)-len(owners),
            'logicalStudentFits': sum(t['model'] == 'distilled-mobile-tcn' for t in tasks),
            'physicalStudentFits': sum(by_id[t['taskId']]['model'] == 'distilled-mobile-tcn' for t in owners)},
        'policy': {'owner': 'first exact ordered recipe in registration order, before outcomes',
            'crossRegistrationReuse': False, 'reuseSelectedEpochOrDecoder': False,
            'freshTargetCalibrationRequired': True, 'sharedCalibrationScoresMustBeBitExact': True,
            'targetNumericalGatesRequired': True, 'allLogicalTasksAndEvaluationsPreserved': True,
            'rawInferenceReuse': 'All42 score archives only, same panel/precision/checkpoint bytes; target numerical audit required',
            'decodedOrMetricReuse': False}}
    evidence.finish()
    write_immutable(registration_directory/'reuse-plan.json', plan)
    return plan


def context(plan_path, task_path, destination):
    from .neural_generalization_experiment import MODELS, load_task
    require(destination.resolve().is_relative_to('/mnt/freenas'), 'Reuse output must be on NAS')
    evidence = Bindings()
    plan_ref = evidence.capture(plan_path); plan = read(plan_path)
    require(plan.get('kind') == 'generalization-training-reuse-plan-v1', 'Unknown reuse plan')
    for reference in plan['code'].values(): evidence.bind(reference)
    for key in ('registration', 'protocol', 'qualification'): evidence.bind(plan[key])
    for reference in read(evidence.bind(plan['qualification']))['evidence']: evidence.bind(reference)
    registration = read(evidence.bind(plan['registration']))
    require(registration['sha256'] == plan['registrationSha256'], 'Reuse registration digest differs')
    configurations = {key: asdict(value) for key, value in MODELS.items()}
    tasks = []
    for row in plan['tasks']:
        task, manifest, _, _ = load_task(evidence.bind(row['task']))
        require(task['taskId'] == row['taskId'] and task['registration'] == plan['registration'], 'Plan task differs')
        tasks.append(task)
    require([t['taskId'] for t in tasks] == registration['contract']['taskIds'], 'Plan task order differs')
    derived = first_owners(tasks, read(manifest)['records'], configurations)
    require([(r['taskId'], r['recipeSha256'], r['physicalOwnerTaskId']) for r in plan['tasks']] == derived,
            'Plan does not use the first exact ordered recipe')
    matches = [r for r in plan['tasks'] if r['task'] == identity(task_path)]
    require(len(matches) == 1, 'Task not mapped once')
    mapping = matches[0]
    require(mapping['taskId'] != mapping['physicalOwnerTaskId'], 'Physical owner must run unchanged fitter')
    require(Path(mapping['fitDirectory']) == destination, 'Reuse output differs from registered destination')
    source = Path(mapping['physicalOwnerFitDirectory'])
    by_task = {t['taskId']: t for t in tasks}
    task, donor = by_task[mapping['taskId']], by_task[mapping['physicalOwnerTaskId']]
    for current, folder in ((task, destination), (donor, source)):
        require(fit_directory(Path(plan['registration']['path']).parent, current) == folder, 'Mapped directory differs')
    return evidence, plan_ref, mapping, task, donor, source


def numerical_gate(folder, task_reference, evidence, *, precision=None, panel=None):
    name = 'fit-numerical-audit.json' if precision is None else f'inference-numerical-audit-{precision}.json'
    reference = evidence.capture(folder/name); gate = read(reference['path'])
    kind = 'independent-generalization-fit-numerical-audit-v1' if precision is None else 'independent-generalization-inference-numerical-audit-v1'
    require(gate.get('kind') == kind and gate.get('passed') is True and gate['task'] == task_reference
            and gate['fitResult'] == evidence.capture(folder/'fit-result.json'), 'Numerical gate missing, stale or wrong owner')
    require(gate['auditor'] == identity(REPO/'scripts/audit-neural-generalization-numerics.py'), 'Numerical auditor changed')
    if precision is not None:
        require(gate['precision'] == precision and gate['panel'] == panel, 'Inference gate population/precision differs')
    for ref in gate['evidence']: evidence.bind(ref)
    return reference, gate


def copy_exact(source_reference, target, evidence):
    source = evidence.bind(source_reference)
    require(not target.exists(), f'Refusing existing output: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024*1024)
    result = evidence.capture(target)
    require(result['sha256'] == source_reference['sha256'], 'Copied bytes differ')
    return result


def reuse_student_features(source, destination, identifiers, images, source_student, target_student, evidence):
    """Share immutable arrays, but bind every receipt to the target encoder."""
    output = {}
    for identifier in identifiers:
        source_path = source/'student-features'/(identifier+'.json')
        if not source_path.exists(): continue
        source_ref = evidence.capture(source_path); old = read(source_path)
        image = images[identifier]
        require(old['id'] == identifier and old['sourceGroup'] == image['sourceGroup']
            and old['encoder'] == source_student['weights'] and old['imageInput'] == image['arrays']
            and old['contractSha256'] == target_student['contractSha256'] and old['labelsUsed'] is False,
            'Source student feature identity differs')
        evidence.bind(old['output'])
        target_path = destination/'student-features'/(identifier+'.json')
        new = {**old, 'encoder': target_student['weights'], 'reusedFeatureOutputFrom': source_ref}
        if target_path.exists():
            current = read(evidence.capture(target_path)['path'])
            require(all(current.get(k) == new[k] for k in ('id', 'sourceGroup', 'encoder', 'imageInput', 'contractSha256', 'shape', 'labelsUsed')),
                    'Existing target feature lineage differs')
            with np.load(evidence.bind(current['output']), allow_pickle=False) as left, np.load(evidence.bind(old['output']), allow_pickle=False) as right:
                require(set(left.files) == set(right.files) and all(left[k].dtype == right[k].dtype
                        and left[k].shape == right[k].shape and np.array_equal(left[k], right[k]) for k in left.files),
                        'Existing target student arrays differ from owner')
            output[identifier] = current
        else:
            write_immutable(target_path, new)
            output[identifier] = new
    return output


def materialize_fit(plan_path, task_path, destination, device):
    from .neural_generalization_experiment import MODELS, family, image_indexes, load_checkpoint
    from .neural_generalization_inputs import load_inference_examples
    from .neural_recognition_fit import predict
    evidence, plan_ref, mapping, task, donor, source = context(plan_path, task_path, destination)
    require(device.startswith('cuda'), 'Real reuse calibration must use the registered CUDA execution mode')
    source_audit_ref, _ = numerical_gate(source, mapping['physicalOwnerTask'], evidence)
    source_fit_ref = evidence.capture(source/'fit-result.json'); source_fit = read(source_fit_ref['path'])
    require('trainingReuse' not in source_fit, 'Reuse chain is prohibited')
    source_meta_ref = source_fit['temporal']; source_meta = read(evidence.bind(source_meta_ref))
    require(not destination.exists(), 'Refusing an existing/incomplete reuse fit directory')
    require(task.get('calibrationIds'), 'Reuse target requires real calibration members')
    started = time.perf_counter(); destination.mkdir(parents=True)
    execution = {'kind': 'exact-ordered-training-reuse-v1', 'plan': plan_ref,
                 'sourceTask': mapping['physicalOwnerTask'], 'sourceFitResult': source_fit_ref,
                 'sourceNumericalAudit': source_audit_ref, 'trainingPerformed': False}
    adapted, source_student_ref, target_student_ref = None, None, None
    feature_copies, new_feature_ids = [], []
    if family(task['model']) == 'distilled':
        from . import neural_mobile_distillation as student
        source_student_ref = source_fit['student']; original = read(evidence.bind(source_student_ref))
        weights = copy_exact(original['weights'], destination/'student/weights.npz', evidence)
        target_student = {**copy.deepcopy(original), 'weights': weights, 'trainingExecution': execution}
        write_immutable(destination/'student/completed.json', target_student)
        target_student_ref = evidence.capture(destination/'student/completed.json')
        needed = list(dict.fromkeys(task['trainIds'] + task['calibrationIds']))
        images, _ = image_indexes(task, needed, for_training=False)
        adapted = reuse_student_features(source, destination, needed, images, original, target_student, evidence)
        feature_copies = [{'id': key,
            'sourceReceipt': evidence.capture(source/'student-features'/(key+'.json')),
            'targetReceipt': evidence.capture(destination/'student-features'/(key+'.json')),
            'sourceOutput': adapted[key]['output'], 'targetOutput': adapted[key]['output']} for key in adapted]
        missing = [key for key in needed if key not in adapted]
        new_feature_ids = missing
        if missing:
            encoder = student.load_encoder(target_student, device)
            for key in missing:
                adapted[key] = student.extract_student_record(encoder, target_student, images[key], destination/'student-features', device)
            del encoder
    validation = load_inference_examples(Path(task['manifest']['path']), Path(task['features']['path']),
        family=family(task['model']), recording_ids=task['calibrationIds'], student_features=adapted)
    require(all(not e.truth and not e.ignored and e.valid.all() for e in validation), 'Calibration is not label blind')
    folder = destination/'temporal'; folder.mkdir()
    copies, predictions, artifacts = [], [], {}
    shared = [key for key in task['calibrationIds'] if key in donor['calibrationIds']]
    for epoch in EPOCHS:
        name = f'weights-{epoch}.npz'
        old_ref = {'path': str(source/'temporal'/name), 'sha256': source_meta['artifacts'][name]}
        new_ref = copy_exact(old_ref, folder/name, evidence)
        copies.append({'epoch': epoch, 'source': old_ref, 'target': new_ref}); artifacts[name] = new_ref['sha256']
        model, mean, scale = load_checkpoint(folder/name, MODELS[task['model']], device)
        scores = {e.id: predict(model, e, mean, scale, MODELS[task['model']], device) for e in validation}
        del model
        old_scores_ref = {'path': str(source/'temporal'/f'predictions-{epoch}.npz'),
                          'sha256': source_meta['artifacts'][f'predictions-{epoch}.npz']}
        with np.load(evidence.bind(old_scores_ref), allow_pickle=False) as original:
            require(set(original.files) == set(donor['calibrationIds']), 'Owner calibration population differs')
            require(all(np.array_equal(scores[key], original[key]) for key in shared), 'Shared calibration scores are not bit exact')
        require(all(value.shape == (len(e.times), 4) and np.isfinite(value).all() and np.all((value >= 0) & (value <= 1))
                    for e in validation for value in (scores[e.id],)), 'Invalid fresh calibration scores')
        score_path = folder/f'predictions-{epoch}.npz'
        np.savez_compressed(score_path, **scores)
        score_ref = evidence.capture(score_path); artifacts[score_path.name] = score_ref['sha256']
        predictions.append({'epoch': epoch, 'output': score_ref, 'sourcePredictions': old_scores_ref,
            'recordingIds': task['calibrationIds'], 'sharedSourceIds': shared,
            'bitExactSharedScores': True if shared else None,
            'sharedControlStatus': 'bit-exact' if shared else 'no-overlapping-calibration-records'})
    meta = {**copy.deepcopy(source_meta), 'validationIds': [e.id for e in validation],
        'validationGroups': sorted({e.group for e in validation}), 'artifacts': artifacts,
        'trainingExecution': execution, 'wallSecondsMeaning': 'Inherited physical-owner optimizer execution; no target optimizer run'}
    write_immutable(folder/'completed.json', meta)
    target_meta_ref = evidence.capture(folder/'completed.json')
    receipt = {'kind': 'generalization-fit-reuse-receipt-v1', 'plan': plan_ref, 'task': identity(task_path),
        'sourceTask': mapping['physicalOwnerTask'], 'recipeSha256': mapping['recipeSha256'],
        'sourceFitResult': source_fit_ref, 'sourceNumericalAudit': source_audit_ref,
        'sourceTemporalCompleted': source_meta_ref, 'targetTemporalCompleted': target_meta_ref,
        'sourceStudentCompleted': source_student_ref, 'targetStudentCompleted': target_student_ref,
        'studentFeatureCopies': feature_copies, 'newStudentFeatureIds': new_feature_ids,
        'checkpointCopies': copies, 'calibrationPredictions': predictions,
        'adapter': identity(__file__), 'trainingPerformed': False, 'freshBlindCalibration': True,
        'externalOutcomesRead': False, 'calibrationDevice': device,
        'materializationWallSeconds': time.perf_counter()-started, 'evidence': evidence.finish()}
    write_immutable(destination/'reuse-receipt.json', receipt)
    result = {'task': identity(task_path), 'taskId': task['taskId'], 'config': asdict(MODELS[task['model']]),
        'temporal': target_meta_ref, 'calibrationPredictionIds': [e.id for e in validation],
        'syntheticOutputProbeOnly': False, 'externalLabelsUsed': False, 'selectionPerformed': False,
        'trainingReuse': identity(destination/'reuse-receipt.json')}
    if target_student_ref is not None: result['student'] = target_student_ref
    write_immutable(destination/'fit-result.json', result)
    return result


def materialize_inference(plan_path, task_path, destination, panel_path, precision):
    """Only copy audited raw probabilities; selections and metrics stay task-local."""
    from .neural_generalization_experiment import family, image_indexes
    evidence, plan_ref, mapping, task, donor, source = context(plan_path, task_path, destination)
    panel_ref = evidence.capture(panel_path); panel = read(panel_path)
    require(panel['kind'] == 'frozen-independent-inference-panel-v1' and panel['inferenceUsesLabels'] is False
            and panel['allInferenceTicksValid'] is True and precision in panel['precisionVariants'], 'Bad independent panel')
    require(precision == 'fp32' or family(task['model']) == 'dino', 'Precision reuse is DINO-only')
    ids = panel['recordingIds']
    require(len(ids) == len(set(ids)) == 42, 'Reuse requires the entire fixed all42 panel')
    for key in ('manifest', 'features', 'featureAudit', 'inventory', 'registrar'): evidence.bind(panel[key])
    source_fit_audit_ref, _ = numerical_gate(source, mapping['physicalOwnerTask'], evidence)
    target_fit_audit_ref, _ = numerical_gate(destination, identity(task_path), evidence)
    source_infer_audit_ref, source_infer_audit = numerical_gate(source, mapping['physicalOwnerTask'], evidence, precision=precision, panel=panel_ref)
    require(source_infer_audit['fitAudit'] == source_fit_audit_ref, 'Owner inference used a different fit gate')
    reuse_fit_gate_ref = evidence.capture(destination/'reuse-audit.json'); reuse_fit_gate = read(reuse_fit_gate_ref['path'])
    require(reuse_fit_gate.get('kind') == 'independent-generalization-fit-reuse-audit-v1'
            and reuse_fit_gate.get('passed') is True
            and reuse_fit_gate['auditor'] == identity(REPO/'scripts/audit-neural-generalization-reuse.py'), 'Independent fit-reuse gate missing')
    require(reuse_fit_gate['plan'] == plan_ref and reuse_fit_gate['task'] == identity(task_path)
            and reuse_fit_gate['sourceTask'] == mapping['physicalOwnerTask']
            and reuse_fit_gate['fitResult'] == identity(destination/'fit-result.json')
            and reuse_fit_gate['targetNumericalAudit'] == target_fit_audit_ref,
            'Fit-reuse gate belongs to another execution')
    for ref in reuse_fit_gate['evidence']: evidence.bind(ref)
    source_fit_ref = evidence.capture(source/'fit-result.json'); source_fit = read(source_fit_ref['path'])
    target_fit_ref = evidence.capture(destination/'fit-result.json'); target_fit = read(target_fit_ref['path'])
    require('trainingReuse' not in source_fit and target_fit['trainingReuse'] == identity(destination/'reuse-receipt.json'), 'Bad reuse ownership')
    source_meta = read(evidence.bind(source_fit['temporal'])); target_meta = read(evidence.bind(target_fit['temporal']))
    weights, weight_copies = {}, []
    for epoch in EPOCHS:
        name = f'weights-{epoch}.npz'
        source_ref = {'path': str(source/'temporal'/name), 'sha256': source_meta['artifacts'][name]}
        target_ref = {'path': str(destination/'temporal'/name), 'sha256': target_meta['artifacts'][name]}
        evidence.bind(source_ref); evidence.bind(target_ref)
        require(source_ref['sha256'] == target_ref['sha256'], 'Inference reuse checkpoint/scaler bytes differ')
        weights[str(epoch)] = target_ref
        weight_copies.append({'epoch': epoch, 'source': source_ref, 'target': target_ref})
    key = panel_ref['sha256'][:16]
    source_folder, folder = source/'inference'/key/precision, destination/'inference'/key/precision
    require(not folder.exists() and not (destination/f'inference-reuse-receipt-{precision}.json').exists(), 'Refusing incomplete/existing inference reuse')
    expected_source_refs = [identity(source_folder/(identifier+'.json')) for identifier in ids]
    require(source_infer_audit['inferenceReceipts'] == expected_source_refs, 'Owner audit did not cover exact ordered panel')
    feature_copies = []
    if family(task['model']) == 'distilled':
        source_student = read(evidence.bind(source_fit['student'])); target_student = read(evidence.bind(target_fit['student']))
        evidence.bind(source_student['weights']); evidence.bind(target_student['weights'])
        require(source_student['weights']['sha256'] == target_student['weights']['sha256'], 'Student encoder bytes differ')
        images, _ = image_indexes(task, ids, for_training=False, panel=panel)
        adapted = reuse_student_features(source, destination, ids, images, source_student, target_student, evidence)
        require(list(adapted) == ids, 'Owner student feature panel is incomplete')
        for identifier in ids:
            source_receipt = evidence.capture(source/'student-features'/(identifier+'.json'))
            feature_copies.append({'id': identifier, 'sourceReceipt': source_receipt,
                'targetReceipt': evidence.capture(destination/'student-features'/(identifier+'.json')),
                'sourceOutput': read(source_receipt['path'])['output'], 'targetOutput': adapted[identifier]['output']})
    folder.mkdir(parents=True)
    copies = []
    for identifier, source_ref in zip(ids, expected_source_refs, strict=True):
        old = read(evidence.bind(source_ref))
        expected = {'task': mapping['physicalOwnerTask'], 'id': identifier, 'precision': precision,
            'manifest': panel['manifest'], 'features': panel['features'], 'labelsUsed': False,
            'panel': panel_ref, 'ignoredLabelsUsedForContextOrDecoding': False, 'epochs': list(EPOCHS)}
        require(all(old.get(k) == v for k, v in expected.items()) and 'rawInferenceReuse' not in old, 'Bad raw inference source lineage')
        require(old['weights'] == {str(r['epoch']): r['source'] for r in weight_copies}, 'Owner raw scores use different checkpoints')
        require(Path(old['output']['path']) == source_folder/(identifier+'.npz'), 'Owner raw archive path differs')
        output_ref = copy_exact(old['output'], folder/(identifier+'.npz'), evidence)
        with np.load(output_ref['path'], allow_pickle=False) as archive:
            require(set(archive.files) == {'times', *(f'epoch_{e}' for e in EPOCHS)}, 'Only raw all-epoch score archives are reusable')
            times = archive['times']
            require(times.dtype == np.float64 and times.ndim == 1 and len(times) > 0
                    and np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Invalid raw timeline')
            for epoch in EPOCHS:
                scores = archive[f'epoch_{epoch}']
                require(scores.dtype == np.float32 and scores.shape == (len(times), 4)
                        and np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid raw probabilities')
        new = {**old, 'task': identity(task_path), 'output': output_ref, 'weights': weights,
            'rawInferenceReuse': {'plan': plan_ref, 'sourceReceipt': source_ref, 'sourceOutput': old['output']}}
        target_path = folder/(identifier+'.json'); write_immutable(target_path, new)
        copies.append({'id': identifier, 'sourceReceipt': source_ref, 'targetReceipt': evidence.capture(target_path),
            'sourceOutput': old['output'], 'targetOutput': output_ref})
    receipt = {'kind': 'generalization-raw-inference-reuse-receipt-v1', 'plan': plan_ref,
        'task': identity(task_path), 'sourceTask': mapping['physicalOwnerTask'], 'recipeSha256': mapping['recipeSha256'],
        'panel': panel_ref, 'precision': precision, 'recordingIds': ids, 'checkpointEpochs': list(EPOCHS),
        'sourceFitResult': source_fit_ref, 'targetFitResult': target_fit_ref,
        'sourceFitNumericalAudit': source_fit_audit_ref, 'targetFitNumericalAudit': target_fit_audit_ref,
        'sourceInferenceNumericalAudit': source_infer_audit_ref, 'fitReuseAudit': reuse_fit_gate_ref,
        'checkpointCopies': weight_copies, 'scoreCopies': copies, 'studentFeatureCopies': feature_copies,
        'rawScoreCopiesBitExact': True, 'decodedPredictionsReused': False, 'selectionOrMetricsReused': False,
        'trainingPerformed': False, 'gpuUsed': False, 'adapter': identity(__file__), 'evidence': evidence.finish()}
    write_immutable(destination/f'inference-reuse-receipt-{precision}.json', receipt)
    return receipt
