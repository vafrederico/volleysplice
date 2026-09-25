"""Run intact registered fits with at most two isolated persistent workers.

This is an execution amendment, not a new experiment. The original plan and
numerical sources stay unchanged. Workers restrict only the orchestrator's task
view and progress destination after validating the complete registered plan.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
EXECUTION_KIND = 'distilled-large-parallel-execution-v2'
ROUNDING_TOLERANCE = 32 * sys.float_info.epsilon


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_key(task):
    return f"{task['variant']}/split-{task.get('splitSeed', task['seed'])}"


def assigned_tasks(total, completed, workers):
    require(isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= 2,
            'Only one or two workers are permitted')
    require(len(set(completed)) == len(completed) and all(0 <= index < total for index in completed),
            'Invalid completed task membership')
    pending = [index for index in range(total) if index not in completed]
    return [pending[worker::workers] for worker in range(workers)]


def filtered_plan(plan, index):
    require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(plan['tasks']),
            'Unknown task index')
    result = copy.deepcopy(plan)
    result['tasks'] = [copy.deepcopy(plan['tasks'][index])]
    return result


def bounded_candidates(candidates):
    """Correct only binary64 excursions outside the metric's exact domain.

    Values already inside [0,1] are preserved bit-for-bit, including immediately
    below one. The calibration floor, scores, intervals, and grid are unchanged.
    """
    bounded, corrections = copy.deepcopy(candidates), []
    for index, row in enumerate(bounded):
        for field in ('innerR_core', 'innerF1_padP_coreR'):
            value = row[field]
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                    'Nonfinite or nonnumeric calibration metric')
            if 0 <= value <= 1:
                continue
            nearest = 0. if value < 0 else 1.
            require(abs(value - nearest) <= ROUNDING_TOLERANCE,
                    'Calibration metric outside binary64 roundoff tolerance')
            row[field] = nearest
            corrections.append(dict(candidateIndex=index, epoch=row['epoch'], decoder=row['decoder'],
                                    field=field, raw=value, bounded=nearest, excursion=abs(value-nearest)))
    return bounded, corrections


def save_immutable(path, value):
    if path.exists():
        require(read(path) == value, 'Calibration audit artifact changed')
    else:
        save(path, value)


def audited_candidates(root, folder, task, candidates):
    bounded, corrections = bounded_candidates(candidates)
    completed = read(folder / 'temporal/completed.json')
    archive = folder / 'calibration-roundoff-v1'
    archive.mkdir(parents=True, exist_ok=True)
    raw_path = archive / 'raw-candidates.json'
    save_immutable(raw_path, candidates)
    evidence = {filename: dict(path=str(folder / 'temporal' / filename), sha256=checksum)
                for filename, checksum in completed['artifacts'].items()}
    receipt = dict(kind='calibration-binary64-domain-normalization-v1', plan=identity(root / 'plan.json'),
        executor=identity(__file__), task=task, contractSha256=completed['contractSha256'],
        rawCandidates=identity(raw_path), temporalArtifacts=evidence,
        studentWeights=read(folder / 'student/completed.json')['weights'],
        tolerance=ROUNDING_TOLERANCE, toleranceDefinition='32 times binary64 machine epsilon',
        reason='Pooled interval arithmetic can exceed exact metric domain by floating-point roundoff',
        corrections=corrections, correctionCount=len(corrections),
        insideDomainValuesUnchanged=True, recallEligibilityFloor=.99,
        scoresChanged=False, decoderGridChanged=False, trainingRepeated=False,
        trainingRecipeChanged=False, calibrationMetricDomainCorrection=True)
    save_immutable(archive / 'receipt.json', receipt)
    return bounded


def require_execution(receipt, root, execution):
    require(receipt['kind'] == EXECUTION_KIND
            and receipt['workerCount'] in (1, 2), 'Unknown execution contract')
    require(receipt['plan'] == identity(root / 'plan.json'), 'Registered plan changed')
    require(receipt['executor'] == identity(__file__), 'Parallel executor source changed')
    require(receipt['validator'] == identity(REPO / 'scripts/evaluate-distilled-mobile-large.py'),
            'Completed-fit validator source changed')
    require(Path(receipt['output']).resolve() == root.resolve()
            and Path(receipt['execution']).resolve() == execution.resolve(), 'Execution output ownership differs')
    planned = [index for group in receipt['assignments'] for index in group]
    completed = receipt['initialCompletedTaskIndexes']
    require(len(receipt['assignments']) == receipt['workerCount'] and len(set(planned)) == len(planned)
            and not set(planned) & set(completed)
            and sorted(planned + completed) == list(range(receipt['registeredTaskCount'])),
            'Execution task ownership is incomplete or overlaps')


def fully_validated_plan(root):
    experiment = import_script('parallel_original_experiment', 'experiment-distilled-mobile-large.py')
    return experiment, experiment.load_plan(root)


def validator():
    return import_script('parallel_original_validator', 'evaluate-distilled-mobile-large.py')


def register(root, execution, workers, device):
    require(root.is_absolute() and execution.is_absolute(), 'Explicit absolute external roots are required')
    require(not execution.exists(), 'Execution directory already exists; no implicit retry')
    experiment, plan = fully_validated_plan(root)
    require(len({task_key(task) for task in plan['tasks']}) == len(plan['tasks']) == 24,
            'Registered task output ownership differs')
    check = validator()
    completed = []
    for index, task in enumerate(plan['tasks']):
        folder = experiment.fit_folder(root, task)
        if (folder / 'selection.json').exists():
            check.validated_fit(root, task)
            completed.append(index)
        else:
            require(not folder.exists(), 'Incomplete fit exists; archive it explicitly before replacing the runner')
    assignments = assigned_tasks(len(plan['tasks']), completed, workers)
    receipt = dict(kind=EXECUTION_KIND, plan=identity(root / 'plan.json'),
        executor=identity(__file__), validator=identity(REPO / 'scripts/evaluate-distilled-mobile-large.py'),
        output=str(root), execution=str(execution), workerCount=workers, device=device,
        registeredTaskCount=len(plan['tasks']), initialCompletedTaskIndexes=completed, assignments=assignments,
        taskKeys=[task_key(task) for task in plan['tasks']],
        trainingRecipeChanged=False, calibrationMetricDomainCorrection=True,
        originalNumericalSourcesChanged=False, originalPlanChanged=False,
        calibrationDomainNormalization=dict(tolerance=ROUNDING_TOLERANCE, outsideDomainOnly=True,
            insideDomainValuesUnchanged=True, recallEligibilityFloor=.99,
            rawAndCorrectionReceiptsSavedPerFit=True),
        executionChanges=['Two persistent isolated processes maximum',
            'Original fit() called once per assigned intact task after full plan validation',
            'Task view and status sink wrapped; original plan hash remains in every fit contract',
            'Candidate metric domain excursions bounded only within32 binary64 epsilons, with full raw receipts',
            'Process-local verification caches retained; unused tensors released between fits'],
        preserved=['Training membership and source groups', 'Student and temporal seeds', 'Data and sample order within each fit',
            'Batch sizes, optimizer, loss, epochs, decoder grid and strict99 calibration',
            'Four Torch CPU threads and deterministic CUDA settings', 'Canonical fit directories and artifact ownership'],
        memoryQualification=dict(maxWorkers=2, priorObservedWorkerHighWaterGiB=5.65,
                                 hostCapacityGiB=16, overlappedTimingsAreNotBenchmarks=True),
        retryPolicy='No retries or partial-output deletion; errors stop both owned workers and preserve diagnosis')
    execution.mkdir(parents=True)
    archived_source = execution / 'executor-source.py'
    archived_source.write_bytes(Path(__file__).read_bytes())
    receipt['archivedExecutor'] = identity(archived_source)
    save(execution / 'registration.json', receipt)
    return receipt


def worker(root, execution, worker_index):
    sys.path.insert(0, str(REPO))
    receipt = read(execution / 'registration.json')
    require_execution(receipt, root, execution)
    require(0 <= worker_index < receipt['workerCount'], 'Unknown worker')
    experiment, original_plan = fully_validated_plan(root)
    original_loader = experiment.load_plan
    original_builder = experiment.sweep.build_candidate_table
    check = validator()
    assigned = receipt['assignments'][worker_index]
    progress_path = execution / 'workers' / f'worker-{worker_index}.json'
    finished = []
    current = None

    def status(_root, phase, **details):
        require(Path(_root).resolve() == root.resolve(), 'Worker attempted another output root')
        save(progress_path, dict(phase=phase, worker=worker_index, pid=os.getpid(),
            taskIndex=current, completedAssignedTaskIndexes=finished, assignedTaskIndexes=assigned, **details))

    try:
        for index in assigned:
            current = index
            require_execution(receipt, root, execution)
            full_plan = original_loader(root)
            require(full_plan == original_plan, 'Registered plan changed between tasks')
            require(task_key(full_plan['tasks'][index]) == receipt['taskKeys'][index], 'Assigned task identity differs')
            folder = experiment.fit_folder(root, full_plan['tasks'][index])
            require(not folder.exists(), 'Assigned task already has output; no overlapping writers or hidden retries')

            def load_one(path, *, _index=index):
                require(Path(path).resolve() == root.resolve(), 'Task view requested another plan')
                verified = original_loader(path)
                require(verified == original_plan, 'Original plan changed inside fit')
                return filtered_plan(verified, _index)

            experiment.load_plan = load_one
            experiment.status = status

            def build_bounded(*args, _task=full_plan['tasks'][index], _folder=folder, **kwargs):
                return audited_candidates(root, _folder, _task, original_builder(*args, **kwargs))

            experiment.sweep.build_candidate_table = build_bounded
            experiment.fit(SimpleNamespace(output=root, device=receipt['device']))
            check.validated_fit(root, full_plan['tasks'][index])
            finished.append(index)
            status(root, 'assigned-fit-validated', taskKey=receipt['taskKeys'][index])
            gc.collect()
            import torch
            if receipt['device'].startswith('cuda'):
                torch.cuda.empty_cache()
        status(root, 'worker-complete')
    except BaseException as error:
        status(root, 'worker-failed', errorType=type(error).__name__, message=str(error))
        raise


def run(root, execution, workers, device):
    receipt = register(root, execution, workers, device)
    children, logs = [], []
    progress_path = execution / 'progress.json'
    started = time.monotonic()
    try:
        for index, tasks in enumerate(receipt['assignments']):
            if not tasks:
                continue
            log = (execution / f'worker-{index}.log').open('wb')
            logs.append(log)
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'worker',
                '--experiment', str(root), '--execution', str(execution), '--worker-index', str(index)],
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
                stdout=log, stderr=subprocess.STDOUT)
            children.append((index, child))
        save(execution / 'launch.json', dict(pid=os.getpid(), registration=identity(execution / 'registration.json'),
             workers=[dict(index=index, pid=child.pid) for index, child in children]))
        while True:
            states = [dict(worker=index, pid=child.pid, returnCode=child.poll()) for index, child in children]
            failures = [state for state in states if state['returnCode'] not in (None, 0)]
            require(not failures, 'A parallel worker failed; inspect its private log')
            snapshots = [read(execution / 'workers' / f'worker-{index}.json') for index, _ in children
                         if (execution / 'workers' / f'worker-{index}.json').exists()]
            completed = len(receipt['initialCompletedTaskIndexes']) + sum(
                len(value['completedAssignedTaskIndexes']) for value in snapshots)
            state = dict(phase='parallel-fits', pid=os.getpid(), completedFits=completed,
                         totalFits=receipt['registeredTaskCount'], workers=states,
                         wallSeconds=time.monotonic() - started)
            save(progress_path, state)
            save(root / 'progress.json', state)
            if all(state['returnCode'] == 0 for state in states):
                break
            time.sleep(3)
        require_execution(receipt, root, execution)
        _, plan = fully_validated_plan(root)
        check = validator()
        for task in plan['tasks']:
            check.validated_fit(root, task)
        result = dict(kind='distilled-large-parallel-completed-v1', status='complete', phase='complete',
                      plan=identity(root / 'plan.json'), completedFits=len(plan['tasks']),
                      registration=identity(execution / 'registration.json'),
                      allRegisteredFitsValidated=True, trainingRecipeChanged=False,
                      calibrationMetricDomainCorrection=True,
                      wallSeconds=time.monotonic() - started)
        save(execution / 'completed.json', result)
        save(progress_path, result)
        save(root / 'progress.json', result)
        return result
    except BaseException as error:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        save(progress_path, dict(phase='failed', errorType=type(error).__name__,
                                message=str(error), outputsPreserved=True))
        raise
    finally:
        for log in logs:
            log.close()


def recover_calibration(root, index):
    """Select from fully saved temporal predictions; never invoke fitting."""
    import numpy as np
    from dataclasses import asdict
    experiment, plan = fully_validated_plan(root)
    from analysis.neural_generalization_results import selection_examples
    task = filtered_plan(plan, index)['tasks'][0]
    folder = experiment.fit_folder(root, task)
    require(not (folder / 'selection.json').exists(), 'Selection already exists; recovery cannot replace it')
    completed, student = read(folder / 'temporal/completed.json'), read(folder / 'student/completed.json')
    plan_reference = identity(root / 'plan.json')
    digest = experiment.sweep.canonical({'plan': plan_reference, 'task': task})
    require(completed['contractSha256'] == student['contractSha256'] == digest
            and completed['seed'] == student['seed'] == task['seed'], 'Recovery model ownership differs')
    require(completed['epochs'] == task['epochs'] == list(experiment.sweep.EPOCHS)
            and completed['kind'] == 'mobile_tcn' and completed['lossArm'] == 'short_boost'
            and all(completed['model'][key] == value for key, value in asdict(experiment.CONFIG).items()),
            'Recovery temporal recipe differs')
    manifest = experiment.inputs.verified(task['manifest'])
    features = experiment.inputs.verified(task['features'])
    records = read(manifest)['records']
    by_id = experiment.validate_membership(task, records)
    tiers = {tier: [key for key in task['trainIds'] if by_id[key]['labelTier'] == tier]
             for tier in ('exact', 'draft', 'coverage')}
    require(completed['trainIds'] == completed['scalerTrainIds'] == tiers['exact']
            and completed['auxiliaryIds'] == {key: tiers[key] for key in ('draft', 'coverage')}
            and completed['validationIds'] == task['calibrationIds']
            and student['trainIds'] == [key for tier in ('exact', 'draft', 'coverage') for key in tiers[tier]],
            'Recovery fitting or calibration membership differs')
    experiment.inputs.verified(student['weights'])
    scores = {}
    for filename, checksum in completed['artifacts'].items():
        experiment.inputs.verified(dict(path=str(folder / 'temporal' / filename), sha256=checksum))
    for epoch in experiment.sweep.EPOCHS:
        with np.load(folder / 'temporal' / f'predictions-{epoch}.npz', allow_pickle=False) as archive:
            require(set(archive.files) == set(task['calibrationIds']), 'Saved calibration prediction membership differs')
            scores[epoch] = {key: archive[key].copy() for key in archive.files}
    examples = selection_examples(task, manifest, features)
    policy = ('export-rally-proxy-selection' if task.get('selectionLabelPolicy') == 'exact-and-export-rally-proxy'
              else 'exact-rallies')
    candidates = audited_candidates(root, folder, task,
        experiment.sweep.build_candidate_table(examples, scores, selection_policy=policy))
    floors = experiment.sweep.select_floors(candidates, floors=(99,))
    selection = dict(task=task, config=asdict(experiment.CONFIG), plan=plan_reference, contractSha256=digest,
                     studentWeights=student['weights'], candidates=candidates, floors=floors)
    save_immutable(folder / 'selection.json', selection)
    validator().validated_fit(root, task)
    receipt = dict(kind='saved-prediction-calibration-recovery-v1', executor=identity(__file__),
                   plan=plan_reference, taskIndex=index, selection=identity(folder / 'selection.json'),
                   audit=identity(folder / 'calibration-roundoff-v1/receipt.json'),
                   trainingRepeated=False, scoresRegenerated=False, recallEligibilityFloor=.99)
    save_immutable(folder / 'calibration-roundoff-v1/recovery.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'worker', 'recover-calibration'))
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args()
    if args.phase == 'run':
        require(args.execution is not None, 'Execution root is required')
        result = run(args.experiment, args.execution, args.workers, args.device)
        print(json.dumps(result), flush=True)
    elif args.phase == 'worker':
        require(args.execution is not None, 'Execution root is required')
        require(args.worker_index is not None, 'Worker index is required')
        worker(args.experiment, args.execution, args.worker_index)
    else:
        require(args.task_index is not None, 'Recovery task index is required')
        print(json.dumps(recover_calibration(args.experiment, args.task_index)), flush=True)


if __name__ == '__main__':
    main()
