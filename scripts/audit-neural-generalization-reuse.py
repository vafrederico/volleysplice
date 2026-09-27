#!/usr/bin/env python3
"""Independently audit exact-training reuse without reading model selections."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_experiment import MODELS, load_task
from analysis.neural_generalization_inputs import load_inference_examples, load_images_teachers

TIERS = ('exact', 'draft', 'coverage')
NUMERICAL = REPO/'scripts/audit-neural-generalization-numerics.py'


def evidence_class():
    path = REPO/'scripts/audit-neural-generalization-selection.py'
    spec = importlib.util.spec_from_file_location('reuse_independent_evidence', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.Evidence


def recipe(task, records, configurations):
    """Derive only the frozen fitter's training inputs, preserving every order."""
    by_id = {r['id']: r for r in records}
    ids = task['trainIds']
    io.require(len(by_id) == len(records) and ids and len(set(ids)) == len(ids)
               and set(ids) <= set(by_id), 'Invalid ordered training population')
    tiers = {tier: [key for key in ids if by_id[key]['labelTier'] == tier] for tier in TIERS}
    io.require(sum(map(len, tiers.values())) == len(ids) and tiers['exact'], 'Invalid training tiers')
    io.require(task['epochs'] == list(io.EPOCHS) and task['lossArm'] == 'short_boost', 'Training recipe changed')
    groups = {by_id[key]['sourceGroup'] for key in ids}
    return {'registration': task['registration'], 'registrationSha256': task['registrationSha256'],
        'manifest': task['manifest'], 'features': task['features'], 'model': task['model'],
        'config': configurations[task['model']], 'seed': task['seed'], 'epochs': list(task['epochs']),
        'lossArm': 'short_boost', 'trainIds': list(ids), 'tierOrderedIds': tiers,
        'trainGroups': sorted(groups), 'excludedStudentGroups': sorted({r['sourceGroup'] for r in records}-groups)}


def owners(tasks, records_by_manifest, configurations):
    first, result = {}, []
    for task in tasks:
        value = recipe(task, records_by_manifest[task['manifest']['sha256']], configurations)
        digest = io.canonical(value)
        first.setdefault(digest, task['taskId'])
        result.append((task['taskId'], digest, first[digest]))
    return result


def fit_directory(parent, task):
    if task['variant'] == 'original-corpus':
        return parent/'fits'/task['model']/f'seed-{task["seed"]}'
    return parent/'fits'/task['variant']/task['model']/f'split-{task["splitSeed"]}'


def check_inherited(source, target, changed):
    io.require({k: v for k, v in source.items() if k not in changed}
               == {k: v for k, v in target.items() if k not in changed},
               'Inherited training history, scaler, membership or recipe changed')


def shared_scores(source, target, source_ids, target_ids):
    io.require(set(source.files) == set(source_ids) and set(target.files) == set(target_ids),
               'Calibration score population differs')
    shared = [key for key in target_ids if key in source_ids]
    for key in shared:
        a, b = source[key], target[key]
        io.require(a.dtype == b.dtype == np.float32 and a.shape == b.shape and np.array_equal(a, b),
                   'Shared calibration scores changed')
    return shared


def numerical_gate(reference, task_ref, fit_ref, completed_ref, evidence):
    gate = evidence.document(reference)
    io.require(gate['kind'] == 'independent-generalization-fit-numerical-audit-v1' and gate['passed'] is True
               and gate['task'] == task_ref and gate['fitResult'] == fit_ref and gate['completed'] == completed_ref
               and gate['taskId'] == evidence.document(task_ref)['taskId']
               and gate['checkpointEpochs'] == list(io.EPOCHS) and gate['auditor'] == io.identity(NUMERICAL),
               'Independent numerical gate is missing, stale or belongs to another fit')
    evidence.closure(gate)
    return gate


def audit_mapping(plan, evidence):
    registration = evidence.document(plan['registration'])
    io.require(plan['kind'] == 'generalization-training-reuse-plan-v1'
               and io.canonical(registration['contract']) == registration['sha256'] == plan['registrationSha256'],
               'Reuse registration differs')
    evidence.closure(plan['code']); evidence.bind(plan['protocol'])
    io.require(plan['code']['scripts/audit-neural-generalization-reuse.py'] == io.identity(__file__), 'Reuse auditor is not frozen in the plan')
    qualification = evidence.document(plan['qualification']); evidence.closure(qualification)
    configurations = {name: asdict(config) for name, config in MODELS.items()}
    io.require(qualification['kind'] == 'generalization-validation-independence-qualification-v1'
               and qualification['passed'] is True and qualification['device'] == 'cpu'
               and {io.canonical(r['config']) for r in qualification['architectures']}
                   == {io.canonical(c) for c in configurations.values()}
               and all(r['checkpointHistoryAndSharedScoresBitExact'] is True for r in qualification['architectures']),
               'Validation independence qualification is incomplete')
    io.require(qualification['source'] == plan['code']['scripts/qualify-neural-generalization-reuse.py']
               and qualification['syntheticOnly'] is True and qualification['realDataRead'] is False,
               'Qualification source/scope differs')
    for architecture in qualification['architectures']:
        io.require(architecture['epochs'] == list(io.EPOCHS) and len(architecture['metadata']) == 2,
                   'Qualification checkpoint grid differs')
        left, right = [evidence.document(ref) for ref in architecture['metadata']]
        check_inherited(left, right, {'validationIds', 'validationGroups', 'artifacts', 'wallSeconds', 'peakAllocatedCudaBytes'})
        io.require([left['validationIds'], right['validationIds']] == architecture['validationPopulations'],
                   'Qualification validation order differs')
        for epoch in io.EPOCHS:
            references = {}
            for index, meta in enumerate((left, right)):
                folder = Path(architecture['metadata'][index]['path']).parent
                for stem in ('weights', 'predictions'):
                    name = f'{stem}-{epoch}.npz'
                    references[index, stem] = evidence.bind({'path': str(folder/name), 'sha256': meta['artifacts'][name]})
            exact_archives(references[0, 'weights'], references[1, 'weights'])
            with np.load(references[0, 'predictions'], allow_pickle=False) as a, np.load(references[1, 'predictions'], allow_pickle=False) as b:
                common = shared_scores(a, b, left['validationIds'], right['validationIds'])
                io.require(common and left['validationIds'] != right['validationIds'], 'Qualification lacks changed validation and shared controls')
    io.require(plan['policy'] == {'owner': 'first exact ordered recipe in registration order, before outcomes',
        'crossRegistrationReuse': False, 'reuseSelectedEpochOrDecoder': False, 'freshTargetCalibrationRequired': True,
        'sharedCalibrationScoresMustBeBitExact': True, 'targetNumericalGatesRequired': True,
        'allLogicalTasksAndEvaluationsPreserved': True,
        'rawInferenceReuse': 'All42 score archives only, same panel/precision/checkpoint bytes; target numerical audit required',
        'decodedOrMetricReuse': False}, 'Reuse policy differs')
    tasks, manifests = [], {}
    for row in plan['tasks']:
        task, _, _, _ = load_task(evidence.bind(row['task']))
        io.require(task['taskId'] == row['taskId'] and task['registration'] == plan['registration']
                   and task['registrationSha256'] == plan['registrationSha256'], 'Mapped task owner differs')
        manifest = evidence.document(task['manifest']); evidence.bind(task['features'])
        evidence.closure(registration['contract']['code'])
        manifests[task['manifest']['sha256']] = manifest['records']; tasks.append(task)
    io.require([t['taskId'] for t in tasks] == registration['contract']['taskIds']
               and len({t['taskId'] for t in tasks}) == len(tasks), 'Registration task order/inventory differs')
    derived = owners(tasks, manifests, configurations)
    io.require([(r['taskId'], r['recipeSha256'], r['physicalOwnerTaskId']) for r in plan['tasks']] == derived,
               'Reuse is not the canonical first identical ordered training recipe')
    by_id = {t['taskId']: t for t in tasks}; mappings = {r['taskId']: r for r in plan['tasks']}
    parent = Path(plan['registration']['path']).parent
    for row in plan['tasks']:
        donor = mappings[row['physicalOwnerTaskId']]
        io.require(row['physicalOwnerTask'] == donor['task']
                   and Path(row['fitDirectory']) == fit_directory(parent, by_id[row['taskId']])
                   and Path(row['physicalOwnerFitDirectory']) == fit_directory(parent, by_id[row['physicalOwnerTaskId']]),
                   'Mapped task path or physical owner differs')
    physical = [r for r in plan['tasks'] if r['taskId'] == r['physicalOwnerTaskId']]
    expected = {'logicalHeadFits': len(tasks), 'physicalHeadFits': len(physical), 'reusedHeadFits': len(tasks)-len(physical),
        'logicalStudentFits': sum(t['model'] == 'distilled-mobile-tcn' for t in tasks),
        'physicalStudentFits': sum(by_id[r['taskId']]['model'] == 'distilled-mobile-tcn' for r in physical)}
    io.require(plan['counts'] == expected, 'Reuse execution counts differ')
    return by_id, mappings


def audit_student(receipt, source_fit, target_fit, source_dir, target_dir, task, execution, source_gate, target_gate, evidence):
    if task['model'] != 'distilled-mobile-tcn':
        io.require(receipt['sourceStudentCompleted'] is None and receipt['targetStudentCompleted'] is None
                   and not receipt['studentFeatureCopies'] and not receipt['newStudentFeatureIds']
                   and 'student' not in source_fit and 'student' not in target_fit, 'Unexpected student lineage')
        return None
    io.require(receipt['sourceStudentCompleted'] == source_fit['student']
               and receipt['targetStudentCompleted'] == target_fit['student'], 'Student ownership differs')
    a = evidence.document(source_fit['student']); b = evidence.document(target_fit['student'])
    check_inherited(a, b, {'weights', 'trainingExecution'})
    io.require(b['trainingExecution'] == execution and b['weights']['sha256'] == a['weights']['sha256']
               and Path(a['weights']['path']) == source_dir/'student/weights.npz'
               and Path(b['weights']['path']) == target_dir/'student/weights.npz', 'Student checkpoint copy differs')
    evidence.bind(a['weights']); evidence.bind(b['weights'])
    io.require(source_gate['student']['completed'] == source_fit['student']
               and target_gate['student']['completed'] == target_fit['student'], 'Student numerical gate differs')
    needed = list(dict.fromkeys(task['trainIds']+task['calibrationIds']))
    images, teachers = load_images_teachers(task['manifest']['path'], task['features']['path'],
                                            recording_ids=needed, for_training=False)
    io.require(not teachers, 'Teacher targets reached feature-reuse validation')
    copies = receipt['studentFeatureCopies']; copied_ids = [r['id'] for r in copies]
    new_ids = receipt['newStudentFeatureIds']
    io.require(len(copied_ids) == len(set(copied_ids)) and len(new_ids) == len(set(new_ids))
               and not set(copied_ids) & set(new_ids) and set(copied_ids+new_ids) == set(needed),
               'Student feature population is incomplete or duplicated')
    for row in copies:
        key = row['id']; old = evidence.document(row['sourceReceipt']); new = evidence.document(row['targetReceipt'])
        io.require(Path(row['sourceReceipt']['path']) == source_dir/'student-features'/(key+'.json')
                   and Path(row['targetReceipt']['path']) == target_dir/'student-features'/(key+'.json')
                   and old['output'] == row['sourceOutput'] and new['output'] == row['targetOutput']
                   and old['encoder'] == a['weights'] and new['encoder'] == b['weights']
                   and new['reusedFeatureOutputFrom'] == row['sourceReceipt'], 'Copied student feature lineage differs')
        check_inherited(old, new, {'encoder', 'output', 'reusedFeatureOutputFrom'})
        io.require(row['sourceOutput']['sha256'] == row['targetOutput']['sha256'], 'Reused student feature bytes differ')
        evidence.bind(row['sourceOutput']); evidence.bind(row['targetOutput'])
    for key in needed:
        ref = evidence.capture(target_dir/'student-features'/(key+'.json')); feature = evidence.document(ref)
        io.require(feature['id'] == key and feature['sourceGroup'] == images[key]['sourceGroup']
                   and feature['imageInput'] == images[key]['arrays'] and feature['encoder'] == b['weights']
                   and feature['contractSha256'] == task['registrationSha256'] and feature['labelsUsed'] is False,
                   'Target student feature source/encoder differs')
        evidence.bind(feature['output']); evidence.closure(feature['imageInput'])
    return {'source': source_fit['student'], 'target': target_fit['student'], 'weightsExactlyEqual': True,
            'inheritedBatchNormAndHistoryExactlyEqual': True, 'reusedFeatureIds': copied_ids, 'freshFeatureIds': new_ids}


def audit(directory, output, plan_path=None, task_path=None):
    evidence = evidence_class()()
    fit_ref = evidence.capture(directory/'fit-result.json'); target_fit = evidence.document(fit_ref)
    receipt_ref = target_fit['trainingReuse']; receipt = evidence.document(receipt_ref)
    io.require(Path(receipt_ref['path']) == directory/'reuse-receipt.json'
               and receipt['kind'] == 'generalization-fit-reuse-receipt-v1'
               and receipt['trainingPerformed'] is False and receipt['freshBlindCalibration'] is True
               and receipt['externalOutcomesRead'] is False and receipt['calibrationDevice'].startswith('cuda'),
               'Reuse receipt scope differs')
    plan = evidence.document(receipt['plan']); tasks, mappings = audit_mapping(plan, evidence)
    io.require(plan_path is None or receipt['plan'] == evidence.capture(plan_path), 'Requested plan differs')
    io.require(task_path is None or receipt['task'] == evidence.capture(task_path), 'Requested task differs')
    mapping = mappings[target_fit['taskId']]; task = tasks[target_fit['taskId']]
    donor = tasks[mapping['physicalOwnerTaskId']]; source_dir = Path(mapping['physicalOwnerFitDirectory'])
    io.require(mapping['taskId'] != mapping['physicalOwnerTaskId'] and Path(mapping['fitDirectory']) == directory
               and receipt['task'] == mapping['task'] == target_fit['task']
               and receipt['sourceTask'] == mapping['physicalOwnerTask']
               and receipt['recipeSha256'] == mapping['recipeSha256'], 'Reuse target/donor differs')
    io.require(receipt['adapter'] == plan['code']['analysis/neural_generalization_reuse.py'], 'Adapter source differs')
    evidence.closure(receipt['evidence'])
    source_fit_ref = evidence.capture(source_dir/'fit-result.json'); source_fit = evidence.document(source_fit_ref)
    io.require(receipt['sourceFitResult'] == source_fit_ref and 'trainingReuse' not in source_fit
               and source_fit['task'] == mapping['physicalOwnerTask'] and source_fit['taskId'] == donor['taskId'],
               'Physical source owner differs or is another reuse chain')
    io.require(receipt['sourceTemporalCompleted'] == source_fit['temporal']
               and receipt['targetTemporalCompleted'] == target_fit['temporal'], 'Temporal completed owner differs')
    for fit, folder in ((source_fit, source_dir), (target_fit, directory)):
        io.require(Path(fit['temporal']['path']) == folder/'temporal/completed.json', 'Temporal checkpoint path differs')
    source_gate = numerical_gate(receipt['sourceNumericalAudit'], mapping['physicalOwnerTask'], source_fit_ref, source_fit['temporal'], evidence)
    io.require(Path(receipt['sourceNumericalAudit']['path']) == source_dir/'fit-numerical-audit.json', 'Source gate path differs')
    target_audit_ref = evidence.capture(directory/'fit-numerical-audit.json')
    target_gate = numerical_gate(target_audit_ref, mapping['task'], fit_ref, target_fit['temporal'], evidence)
    source = evidence.document(source_fit['temporal']); target = evidence.document(target_fit['temporal'])
    execution = {'kind': 'exact-ordered-training-reuse-v1', 'plan': receipt['plan'],
        'sourceTask': receipt['sourceTask'], 'sourceFitResult': source_fit_ref,
        'sourceNumericalAudit': receipt['sourceNumericalAudit'], 'trainingPerformed': False}
    check_inherited(source, target, {'validationIds', 'validationGroups', 'artifacts', 'trainingExecution', 'wallSecondsMeaning'})
    io.require(target['trainingExecution'] == execution
               and target['wallSecondsMeaning'] == 'Inherited physical-owner optimizer execution; no target optimizer run'
               and target['validationIds'] == task['calibrationIds'] and source['validationIds'] == donor['calibrationIds'],
               'Target execution or calibration population differs')
    io.require([r['epoch'] for r in receipt['checkpointCopies']] == list(io.EPOCHS)
               and [r['epoch'] for r in receipt['calibrationPredictions']] == list(io.EPOCHS), 'Incomplete checkpoint/calibration bank')
    sizes = {e.id: len(e.times) for e in load_inference_examples(task['manifest']['path'], task['features']['path'],
              family='av', recording_ids=task['calibrationIds'])}
    checks = []
    for copied, prediction in zip(receipt['checkpointCopies'], receipt['calibrationPredictions'], strict=True):
        epoch = copied['epoch']; weight_name = f'weights-{epoch}.npz'; score_name = f'predictions-{epoch}.npz'
        for owner, folder, meta in (('source', source_dir, source), ('target', directory, target)):
            io.require(copied[owner] == {'path': str(folder/'temporal'/weight_name), 'sha256': meta['artifacts'][weight_name]},
                       'Checkpoint-bank ownership differs'); evidence.bind(copied[owner])
        io.require(copied['source']['sha256'] == copied['target']['sha256'], 'Checkpoint/scaler bytes changed')
        io.require(prediction['output'] == {'path': str(directory/'temporal'/score_name), 'sha256': target['artifacts'][score_name]}
                   and prediction['sourcePredictions'] == {'path': str(source_dir/'temporal'/score_name), 'sha256': source['artifacts'][score_name]}
                   and prediction['recordingIds'] == task['calibrationIds'], 'Fresh calibration archive differs')
        with np.load(evidence.bind(prediction['sourcePredictions']), allow_pickle=False) as a, \
             np.load(evidence.bind(prediction['output']), allow_pickle=False) as b:
            shared = shared_scores(a, b, donor['calibrationIds'], task['calibrationIds'])
            for key, length in sizes.items():
                value = b[key]
                io.require(value.dtype == np.float32 and value.shape == (length, 4) and np.isfinite(value).all()
                           and np.all((value >= 0) & (value <= 1)), 'Invalid target calibration tensor')
        io.require(prediction['sharedSourceIds'] == shared
                   and prediction['bitExactSharedScores'] is (True if shared else None)
                   and prediction['sharedControlStatus'] == ('bit-exact' if shared else 'no-overlapping-calibration-records'),
                   'Shared-calibration control claim differs')
        checks.append({'epoch': epoch, 'weightsExactlyEqual': True, 'sharedCalibrationIds': shared,
                       'sharedScoresExactlyEqual': True if shared else None})
    student = audit_student(receipt, source_fit, target_fit, source_dir, directory, task, execution,
                            source_gate, target_gate, evidence)
    evidence.capture(__file__)
    for ref in list(evidence.references.values()): evidence.bind(ref)
    result = {'kind': 'independent-generalization-fit-reuse-audit-v1', 'passed': True,
        'plan': receipt['plan'], 'receipt': receipt_ref, 'task': mapping['task'], 'sourceTask': mapping['physicalOwnerTask'],
        'recipeSha256': mapping['recipeSha256'], 'sourceFitResult': source_fit_ref, 'fitResult': fit_ref,
        'sourceNumericalAudit': receipt['sourceNumericalAudit'], 'targetNumericalAudit': target_audit_ref,
        'checkpointChecks': checks, 'student': student, 'evidence': list(evidence.references.values()),
        'auditor': io.identity(__file__), 'trainingPerformed': False, 'gpuUsed': False,
        'modelSelectionsRead': False, 'externalOutcomesRead': False,
        'scope': 'Canonical same-registration ordered recipe; exact full checkpoint/scaler/history/student inheritance; fresh target calibration ownership and complete shared-score equality. Separate passed numerical gates independently replay source and target calibration/encoder samples.'}
    io.write_new(output, result)
    return result


def exact_archives(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        io.require(set(a.files) == set(b.files), 'Array inventory differs')
        for key in a.files:
            io.require(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                       and np.array_equal(a[key], b[key]), 'Array bytes or values differ: '+key)


def inference_gate(reference, task_ref, fit_ref, completed_ref, fit_audit_ref, panel_ref, precision, evidence):
    gate = evidence.document(reference)
    io.require(gate['kind'] == 'independent-generalization-inference-numerical-audit-v1' and gate['passed'] is True
               and gate['task'] == task_ref and gate['taskId'] == evidence.document(task_ref)['taskId']
               and gate['fitResult'] == fit_ref and gate['completed'] == completed_ref and gate['fitAudit'] == fit_audit_ref
               and gate['panel'] == panel_ref and gate['precision'] == precision and gate['checkpointEpochs'] == list(io.EPOCHS)
               and gate['auditor'] == io.identity(NUMERICAL), 'Independent inference numerical gate differs')
    evidence.closure(gate)
    return gate


def audit_inference(directory, output, plan_path, task_path, panel_path, precision):
    evidence = evidence_class()()
    receipt_ref = evidence.capture(directory/f'inference-reuse-receipt-{precision}.json')
    receipt = evidence.document(receipt_ref)
    io.require(receipt['kind'] == 'generalization-raw-inference-reuse-receipt-v1'
               and receipt['plan'] == evidence.capture(plan_path) and receipt['task'] == evidence.capture(task_path)
               and receipt['panel'] == evidence.capture(panel_path) and receipt['precision'] == precision
               and receipt['rawScoreCopiesBitExact'] is True and receipt['decodedPredictionsReused'] is False
               and receipt['selectionOrMetricsReused'] is False and receipt['trainingPerformed'] is False and receipt['gpuUsed'] is False,
               'Raw-inference reuse scope differs')
    plan = evidence.document(receipt['plan']); tasks, mappings = audit_mapping(plan, evidence)
    task = evidence.document(receipt['task']); mapping = mappings[task['taskId']]
    io.require(task == tasks[task['taskId']] and mapping['task'] == receipt['task']
               and mapping['physicalOwnerTask'] == receipt['sourceTask']
               and mapping['recipeSha256'] == receipt['recipeSha256'] and mapping['taskId'] != mapping['physicalOwnerTaskId']
               and Path(mapping['fitDirectory']) == directory, 'Raw-inference reuse mapping differs')
    io.require(receipt['adapter'] == plan['code']['analysis/neural_generalization_reuse.py'], 'Raw-inference adapter differs')
    evidence.closure(receipt['evidence'])
    source_dir = Path(mapping['physicalOwnerFitDirectory'])
    source_fit_ref = evidence.capture(source_dir/'fit-result.json'); source_fit = evidence.document(source_fit_ref)
    target_fit_ref = evidence.capture(directory/'fit-result.json'); target_fit = evidence.document(target_fit_ref)
    io.require(receipt['sourceFitResult'] == source_fit_ref and receipt['targetFitResult'] == target_fit_ref
               and source_fit['task'] == receipt['sourceTask'] and target_fit['task'] == receipt['task']
               and 'trainingReuse' not in source_fit, 'Raw-inference fit ownership differs')
    fit_reuse_ref = receipt['fitReuseAudit']; fit_reuse = evidence.document(fit_reuse_ref)
    io.require(Path(fit_reuse_ref['path']) == directory/'reuse-audit.json'
               and fit_reuse['kind'] == 'independent-generalization-fit-reuse-audit-v1' and fit_reuse['passed'] is True
               and fit_reuse['plan'] == receipt['plan'] and fit_reuse['task'] == receipt['task']
               and fit_reuse['sourceTask'] == receipt['sourceTask'] and fit_reuse['fitResult'] == target_fit_ref
               and fit_reuse['sourceFitResult'] == source_fit_ref and fit_reuse['recipeSha256'] == receipt['recipeSha256']
               and fit_reuse['receipt'] == target_fit['trainingReuse'] and fit_reuse['auditor'] == io.identity(__file__),
               'Independent fit-reuse gate is missing or wrong owner')
    evidence.closure(fit_reuse)
    for key, folder, task_ref, fit_ref, fit in (
            ('sourceFitNumericalAudit', source_dir, receipt['sourceTask'], source_fit_ref, source_fit),
            ('targetFitNumericalAudit', directory, receipt['task'], target_fit_ref, target_fit)):
        io.require(Path(receipt[key]['path']) == folder/'fit-numerical-audit.json', 'Numerical fit gate path differs')
        numerical_gate(receipt[key], task_ref, fit_ref, fit['temporal'], evidence)
    panel = evidence.document(receipt['panel']); evidence.closure(panel)
    population = evidence.document(panel['manifest'])['records']; ids = panel['recordingIds']
    io.require(panel['kind'] == 'frozen-independent-inference-panel-v1' and panel['inferenceUsesLabels'] is False
               and panel['allInferenceTicksValid'] is True and precision in panel['precisionVariants']
               and len(ids) == len(set(ids)) == len(population) == 42 and set(ids) == {r['id'] for r in population}
               and receipt['recordingIds'] == ids and receipt['checkpointEpochs'] == list(io.EPOCHS)
               and (precision == 'fp32' or task['model'] in ('dino-tcn', 'dino-transformer')), 'Incomplete/changed blind panel')
    source_gate = inference_gate(receipt['sourceInferenceNumericalAudit'], receipt['sourceTask'], source_fit_ref,
        source_fit['temporal'], receipt['sourceFitNumericalAudit'], receipt['panel'], precision, evidence)
    target_gate_ref = evidence.capture(directory/f'inference-numerical-audit-{precision}.json')
    target_gate = inference_gate(target_gate_ref, receipt['task'], target_fit_ref,
        target_fit['temporal'], receipt['targetFitNumericalAudit'], receipt['panel'], precision, evidence)
    source_meta, target_meta = evidence.document(source_fit['temporal']), evidence.document(target_fit['temporal'])
    io.require([r['epoch'] for r in receipt['checkpointCopies']] == list(io.EPOCHS), 'Raw-score checkpoint bank incomplete')
    for copied in receipt['checkpointCopies']:
        name = f'weights-{copied["epoch"]}.npz'
        for owner, folder, meta in (('source', source_dir, source_meta), ('target', directory, target_meta)):
            io.require(copied[owner] == {'path': str(folder/'temporal'/name), 'sha256': meta['artifacts'][name]}, 'Raw-score checkpoint owner differs')
            evidence.bind(copied[owner])
        io.require(copied['source']['sha256'] == copied['target']['sha256'], 'Raw-score checkpoint/scaler bytes differ')
    student_copies = receipt['studentFeatureCopies']
    if task['model'] == 'distilled-mobile-tcn':
        source_student, target_student = evidence.document(source_fit['student']), evidence.document(target_fit['student'])
        io.require(source_student['weights']['sha256'] == target_student['weights']['sha256'], 'Raw-score student encoders differ')
        evidence.bind(source_student['weights']); evidence.bind(target_student['weights'])
        io.require([r['id'] for r in student_copies] == ids, 'Student raw-inference feature scope/order differs')
        images, teachers = load_images_teachers(panel['manifest']['path'], panel['features']['path'], recording_ids=ids, for_training=False)
        io.require(not teachers, 'Teacher targets reached inference-reuse validation')
        for row in student_copies:
            key = row['id']; a, b = evidence.document(row['sourceReceipt']), evidence.document(row['targetReceipt'])
            io.require(Path(row['sourceReceipt']['path']) == source_dir/'student-features'/(key+'.json')
                       and Path(row['targetReceipt']['path']) == directory/'student-features'/(key+'.json')
                       and a['output'] == row['sourceOutput'] and b['output'] == row['targetOutput']
                       and a['encoder'] == source_student['weights'] and b['encoder'] == target_student['weights']
                       and a['id'] == b['id'] == key and a['imageInput'] == b['imageInput'] == images[key]['arrays']
                       and a['labelsUsed'] is False and b['labelsUsed'] is False, 'Student raw-inference feature owner differs')
            check_inherited(a, b, {'encoder', 'output', 'reusedFeatureOutputFrom'})
            exact_archives(evidence.bind(row['sourceOutput']), evidence.bind(row['targetOutput']))
    else:
        io.require(not student_copies, 'Unexpected student inference payload')
    copies = receipt['scoreCopies']; io.require([r['id'] for r in copies] == ids, 'Raw probability panel/order differs')
    key = receipt['panel']['sha256'][:16]
    for owner, folder, gate in (('source', source_dir, source_gate), ('target', directory, target_gate)):
        references = [r[owner+'Receipt'] for r in copies]
        io.require(gate['inferenceReceipts'] == references, 'Numerical gate raw receipt population differs')
        output_folder = folder/'inference'/key/precision
        io.require({p.stem for p in output_folder.glob('*.json')} == set(ids)
                   and {p.stem for p in output_folder.glob('*.npz')} == set(ids), 'Raw-score artifact inventory differs')
    for row in copies:
        identifier = row['id']; old, new = evidence.document(row['sourceReceipt']), evidence.document(row['targetReceipt'])
        for owner, folder, task_ref, raw in (('source', source_dir, receipt['sourceTask'], old), ('target', directory, receipt['task'], new)):
            io.require(Path(row[owner+'Receipt']['path']) == folder/'inference'/key/precision/(identifier+'.json')
                       and raw['output'] == row[owner+'Output']
                       and Path(raw['output']['path']) == folder/'inference'/key/precision/(identifier+'.npz')
                       and raw['task'] == task_ref and raw['id'] == identifier and raw['panel'] == receipt['panel']
                       and raw['manifest'] == panel['manifest'] and raw['features'] == panel['features']
                       and raw['precision'] == precision and raw['epochs'] == list(io.EPOCHS)
                       and raw['labelsUsed'] is False and raw['ignoredLabelsUsedForContextOrDecoding'] is False
                       and raw['weights'] == {str(r['epoch']): r[owner] for r in receipt['checkpointCopies']}, 'Raw-score receipt ownership differs')
        io.require('rawInferenceReuse' not in old and new['rawInferenceReuse'] == {'plan': receipt['plan'],
            'sourceReceipt': row['sourceReceipt'], 'sourceOutput': row['sourceOutput']}, 'Raw-score reuse chain differs')
        check_inherited(old, new, {'task', 'weights', 'output', 'rawInferenceReuse'})
        io.require(row['sourceOutput']['sha256'] == row['targetOutput']['sha256'], 'Raw probability archive bytes changed')
        evidence.bind(row['sourceOutput']); path = evidence.bind(row['targetOutput'])
        blind, = load_inference_examples(panel['manifest']['path'], panel['features']['path'], family='av', recording_ids=[identifier])
        with np.load(path, allow_pickle=False) as archive:
            io.require(set(archive.files) == {'times', *(f'epoch_{e}' for e in io.EPOCHS)}
                       and archive['times'].dtype == np.float64 and np.array_equal(archive['times'], blind.times), 'Raw archive timeline/schema differs')
            for epoch in io.EPOCHS:
                values = archive[f'epoch_{epoch}']
                io.require(values.dtype == np.float32 and values.shape == (len(blind.times), 4)
                           and np.isfinite(values).all() and np.all((0 <= values) & (values <= 1)), 'Invalid copied raw probabilities')
    evidence.capture(__file__)
    for reference in list(evidence.references.values()): evidence.bind(reference)
    result = {'kind': 'independent-generalization-inference-reuse-audit-v1', 'passed': True,
        'plan': receipt['plan'], 'receipt': receipt_ref, 'task': receipt['task'], 'sourceTask': receipt['sourceTask'],
        'panel': receipt['panel'], 'precision': precision, 'fitResult': target_fit_ref, 'sourceFitResult': source_fit_ref,
        'fitReuseAudit': fit_reuse_ref, 'targetNumericalAudit': target_gate_ref,
        'sourceNumericalAudit': receipt['sourceInferenceNumericalAudit'], 'recordingIds': ids,
        'checkpointEpochs': list(io.EPOCHS), 'allRawScoreBytesExactlyEqual': True,
        'studentFeatureArrayEqualityCount': len(student_copies), 'evidence': list(evidence.references.values()),
        'auditor': io.identity(__file__), 'trainingPerformed': False, 'gpuUsed': False,
        'modelSelectionsRead': False, 'externalOutcomesRead': False,
        'scope': 'All42 immutable raw-score archives and full checkpoint banks; fixed blind panel; student encoder/feature equality; target ownership and separate source/target numerical audits. No decoded outcomes or calibration choices reused.'}
    io.write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fit', 'inference'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    args = parser.parse_args()
    io.require(args.output.resolve().is_relative_to('/mnt/freenas') and not args.output.exists(), 'Audit needs a new NAS output')
    if args.phase == 'fit':
        audit(args.fit, args.output, args.plan, args.task)
    else:
        io.require(args.panel is not None, 'Inference reuse audit requires its frozen panel')
        audit_inference(args.fit, args.output, args.plan, args.task, args.panel, args.precision)
