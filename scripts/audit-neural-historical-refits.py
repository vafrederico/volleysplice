#!/usr/bin/env python3
"""Audit all historical outer refits without fitting or GPU execution.

Reconstructs source membership and optimizer exposure, checks every saved array,
and requires exact original weights/scores/history at all available checkpoints.
Every head receives sampled CPU replay at all four epochs. New distilled
students receive the frozen independent student/feature audit. Existing student
inputs/features are reused only after rehashing their passing audit evidence.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as sweep
from analysis import neural_recall_sweep_historical as study
from analysis.neural_recall_sweep_adapters import bind


def exact_arrays(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        sweep.require(set(a.files) == set(b.files), 'Original tensor inventory differs')
        for key in a.files:
            sweep.require(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                          and np.array_equal(a[key], b[key]), 'Original tensor changed: ' + key)


def audit(output):
    import torch
    from analysis import neural_mobile_distillation as distilled
    from analysis import neural_short_boost_transfer as historical
    from analysis.neural_recognition_inputs import attach_features
    from analysis.recognition_temporal_model import RecognitionConfig
    torch.set_num_threads(2)
    protocol = study.verify(output)
    plan_ref = sweep.identity(output / 'refit-plan.json')
    plan = sweep.read(plan_ref['path'])
    sweep.require(plan['protocol'] == sweep.identity(output / 'protocol.json'), 'Refit plan changed')
    expected_tasks = {(model, seed, outer) for model in study.MODELS for seed in study.SEEDS for outer in range(4)}
    sweep.require({(t['model'], t['seed'], t['outerIndex']) for t in plan['tasks']} == expected_tasks
                  and len(plan['tasks']) == 72, 'Incomplete or duplicate outer refit plan')
    student_audit = study.load_script('audit-neural-mobile-distillation.py')
    evidence = student_audit.Evidence()
    old, tensor, _ = student_audit.legacy_helpers(evidence)
    evidence.bind(plan_ref)
    for ref in [*protocol['code'], *protocol['registrations'], protocol['source'], protocol['records'], protocol['exactManifest']]:
        evidence.bind(ref)
    prior = sweep.read(evidence.bind(protocol['priorStudentAudit']))
    sweep.require(prior['passed'] is True and prior['auditor'] == sweep.identity(REPO / 'scripts/audit-neural-mobile-distillation.py')
                  and prior['counts']['seeds'] == 3 and prior['counts']['physicalStudentFits'] == 28,
                  'Prior full student audit missing or changed')
    # This also binds the exact original image and teacher bytes used by two new
    # students; the raw-frame/teacher gate is not represented as a fresh replay.
    for ref in prior['evidence']:
        evidence.bind(ref)
    exact_examples = {e.id: e for e in sweep.examples_from_rows(sweep.read(protocol['records']['path'])['records'])}
    cached_model, data, config = None, None, None
    checks, parity_epochs = [], 0
    for task in plan['tasks']:
        print(f"AUDIT_OUTER {task['model']} seed={task['seed']} outer={task['outerIndex']}", flush=True)
        directory = study.task_directory(output, task)
        parity_ref = evidence.capture(directory / 'parity.json')
        parity = sweep.read(parity_ref['path'])
        sweep.require(parity['passed'] is True and parity['task'] == task and parity['plan'] == plan_ref,
                      'Refit receipt identity differs')
        meta_ref = evidence.capture(directory / 'temporal/completed.json')
        meta = sweep.read(meta_ref['path'])
        sweep.require(parity['completed'] == meta_ref and meta['epochs'] == list(sweep.EPOCHS)
                      and meta['seed'] == task['seed'] and meta['contractSha256'] == task['originalContractSha256'],
                      'Refit checkpoint grid/identity differs')
        registration = sweep.read(evidence.bind(task['registration']))
        contract = registration['contract']
        sweep.require(registration['sha256'] == task['originalContractSha256']
                      and sweep.canonical(contract) == registration['sha256'], 'Original numerical contract changed')
        student = task['model'] == 'distilled_mobile_tcn'
        legacy = task['model'] in ('av_tcn_short_boost', 'dino_tcn_short_boost')
        if cached_model != task['model']:
            source = contract.get('source') or contract.get('manifest') or protocol['source']
            evidence.bind(source)
            dino = contract.get('dinoManifest')
            if dino:
                evidence.bind(dino)
            data = historical.load_data(Path(source['path']), Path(dino['path']) if dino else None,
                                       with_dino=task['model'] in ('dino_tcn_short_boost', 'dino_transformer'))
            config = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8) if student else RecognitionConfig(**contract['config']) if not legacy else None
            if config and config.family == 'mobile' and not student:
                evidence.bind(contract['featureManifest'])
                data = attach_features(data, Path(contract['featureManifest']['path']), config, Path(source['path']))
            cached_model = task['model']
        excluded = {task['heldSourceGroup']}
        rows, validation, members = old.fold_members(data, excluded)
        sweep.require(all(meta[key] == value for key, value in members.items())
                      and meta['scalerTrainIds'] == members['trainIds'] and meta['lossArm'] == 'short_boost',
                      'Independent source/scaler membership or objective differs')
        expected_artifacts = {f'{stem}-{epoch}.npz' for epoch in sweep.EPOCHS for stem in ('weights', 'predictions')}
        sweep.require(set(meta['artifacts']) == expected_artifacts, 'Refit artifact inventory differs')
        for name, sha in meta['artifacts'].items():
            evidence.bind({'path': str(directory / 'temporal' / name), 'sha256': sha})
        for epoch in sweep.EPOCHS:
            with np.load(directory / f'temporal/predictions-{epoch}.npz', allow_pickle=False) as scores:
                sweep.require(set(scores.files) == set(members['validationIds']), 'Outer score population differs')
                for key in scores.files:
                    e, values = exact_examples[key], scores[key]
                    sweep.require(e.group in excluded and values.dtype == np.float32 and values.shape == (len(e.times), 4)
                                  and np.isfinite(values).all() and np.all((0 <= values) & (values <= 1))
                                  and np.all(values[~e.valid] == 0), 'Invalid held score array')
        expected_parity = []
        for epoch_text, owner in task['parityCheckEpochOwners'].items():
            previous = sweep.read(evidence.bind(owner['completed'])); folder = Path(owner['fitDirectory']); epoch = int(epoch_text)
            for key in ('contractSha256', 'kind', 'seed', 'model', 'trainIds', 'auxiliaryIds', 'validationIds', 'trainGroups',
                        'auxiliaryGroups', 'validationGroups', 'scalerTrainIds', 'supervisedCounts', 'positiveWeight', 'liveLossWeighting'):
                sweep.require(meta[key] == previous[key], 'Original fit field differs: ' + key)
            sweep.require(meta['history'][:len(previous['history'])] == previous['history'], 'Old numerical/exposure history prefix changed')
            for stem in ('weights', 'predictions'):
                name = f'{stem}-{epoch}.npz'
                evidence.bind({'path': str(folder / name), 'sha256': previous['artifacts'][name]})
                exact_arrays(directory / 'temporal' / name, folder / name)
            expected_parity.append({'epoch': epoch, 'owner': owner['completed'], 'weightsExactlyEqual': True,
                                    'scoresExactlyEqual': True, 'historyPrefixExactlyEqual': True})
            parity_epochs += 1
        sweep.require(parity['checks'] == expected_parity and parity['newStudentFit'] == task['requiresStudentFit'], 'Parity receipt claims differ')
        head_check, student_check = None, None
        if legacy:
            # Independent sampler replay covers the new numerical-history tail.
            inputs = {r.example.id: {'group': r.example.group, 'valid': r.example.valid} for tier in rows.values() for r in tier}
            supervision = {r.example.id: {'mask': r.mask} for tier in rows.values() for r in tier}
            pools = {tier: tensor.chunk_geometry([r.example.id for r in records], inputs, supervision) for tier, records in rows.items()}
            tensor.validate_history(meta, tensor.independent_exposure(pools, task['seed'], max(sweep.EPOCHS)))
            original = next(iter(task['parityCheckEpochOwners'].items()))
            kind = 'tcn' if task['model'] == 'av_tcn_short_boost' else 'dino_tcn'
            with torch.random.fork_rng(devices=[]):
                model = historical.model_for(kind).cpu().eval()
            replay_config = RecognitionConfig(family='av' if kind == 'tcn' else 'dino', head='tcn')
            replay_checks = []
            with np.load(Path(original[1]['fitDirectory']) / f'weights-{original[0]}.npz', allow_pickle=False) as template:
                for epoch in sweep.EPOCHS:
                    with np.load(directory / f'temporal/weights-{epoch}.npz', allow_pickle=False) as weights:
                        sweep.require(set(weights.files) == set(template.files), 'New legacy state inventory differs')
                        for key in weights.files:
                            sweep.require(weights[key].dtype == template[key].dtype and weights[key].shape == template[key].shape
                                          and np.isfinite(weights[key]).all(), 'Invalid new legacy tensor')
                        sweep.require(np.array_equal(weights['mean'], template['mean']) and np.array_equal(weights['scale'], template['scale']),
                                      'New legacy scaler changed')
                        model.load_state_dict({key: torch.from_numpy(weights['model::' + key].copy()) for key in model.state_dict()}, strict=True)
                        with np.load(directory / f'temporal/predictions-{epoch}.npz', allow_pickle=False) as saved:
                            replay_checks.extend({'epoch': epoch, **old.replay_sample(model, e, saved[e.id], weights['mean'],
                                weights['scale'], replay_config)} for e in validation)
            head_check = {'cpuReplay': replay_checks, 'allFourEpochs': True}
            sweep.require(not parity['additionalTrainingOwners'], 'Unexpected learned encoder owner')
        elif student:
            sweep.require(prior['registration'] == task['registration'], 'Student prior audit belongs to another registration')
            images = {r['id']: r for r in sweep.read(contract['images']['path'])['records']}
            teachers = {r['id']: r for r in sweep.read(contract['teacherTargets']['path'])['records']}
            if task['requiresStudentFit']:
                _, student_check = student_audit.audit_fit(directory, data, excluded, sweep.EPOCHS, task['seed'], registration,
                                                           images, teachers, config, evidence, old, tensor)
                expected_owner = evidence.capture(directory / 'student/completed.json')
            else:
                folder = Path(task['reuseStudentDirectory']).parent
                expected_owner = evidence.capture(folder / 'student/completed.json')
                prior_seed = next(row for row in prior['results'] if row['seed'] == task['seed'])
                previous_fit = next(row for row in prior_seed['fits'] if row['folder'] == str(folder))
                sweep.require(previous_fit['student']['completed'] == expected_owner, 'Reused student differs from audited fit')
                index_ref = evidence.capture(folder / 'features.json')
                sweep.require(previous_fit['features']['index'] == index_ref, 'Reused feature index differs from audited fit')
                features = {r['id']: r for r in sweep.read(index_ref['path'])['records']}
                student_audit.check_student_membership(sweep.read(expected_owner['path']), data, excluded, task['seed'], registration['sha256'])
                train, auxiliary, permitted = distilled.allowed_records(data, excluded)
                held = [r for r in data['exact'] if r.example.group in excluded]
                attached = {r.example.id: r for r in distilled.attach_rows(permitted + held, features)}
                fold_data = {tier: [attached.get(r.example.id, r) for r in data[tier]] for tier in data}
                adapted = {**registration, 'contract': {**contract, 'lossArm': 'short_boost'}}
                _, head_check = old.audit_fit(directory / 'temporal', fold_data, excluded, sweep.EPOCHS, task['seed'], adapted, config, tensor)
                student_check = {'reusedPriorAudit': protocol['priorStudentAudit'], 'completed': expected_owner,
                                 'features': index_ref, 'evidenceRehashed': True, 'newStudentTraining': False}
            sweep.require(parity['additionalTrainingOwners'] == [expected_owner], 'Wrong distilled encoder training owner')
        else:
            _, head_check = old.audit_fit(directory / 'temporal', data, excluded, sweep.EPOCHS, task['seed'], registration, config, tensor)
            sweep.require(not parity['additionalTrainingOwners'], 'Unexpected learned encoder owner')
        checks.append({'task': task, 'parity': parity_ref, 'completed': meta_ref, 'exactOriginalEpochCount': len(expected_parity),
                       'headReplay': head_check, 'student': student_check, 'allEpochArraysChecked': True})
    sweep.require(parity_epochs == 86 and sum(t['requiresStudentFit'] for t in plan['tasks']) == 2, 'Historical available-epoch population changed')
    study.verify(output)
    return {'kind': 'independent-historical-all-outer-epochs-audit-v1', 'passed': True,
            'protocol': sweep.identity(output / 'protocol.json'), 'plan': plan_ref,
            'counts': {'outerFits': len(checks), 'checkpoints': len(checks) * 4, 'exactOriginalEpochs': parity_epochs, 'newStudents': 2},
            'checks': checks, 'references': evidence.finish(), 'auditor': sweep.identity(Path(__file__)),
            'replayScope': 'Exact old weights/probabilities/history at all86 existing epochs; independent source/scaler/exposure and array integrity for all288. First/middle/last real-context held chunks receive CPU neural replay at every checkpoint for all six families. Two new students also receive sampled encoder replay and independent membership/BN/exposure checks. Original image/teacher gate is reused after evidence rehash.',
            'trainingPerformed': False, 'gpuUsed': False, 'protectedTestOpened': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')) and not args.output.exists(), 'Audit needs new direct-NAS output')
    for name in ('TMPDIR', 'TMP', 'TEMP', 'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH'):
        location = args.study / 'audit-runtime' / name.lower()
        location.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(location)
    sweep.write_new(args.output, audit(args.study))
