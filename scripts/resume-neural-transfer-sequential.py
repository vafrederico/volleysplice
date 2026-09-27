#!/usr/bin/env python3
"""Stdlib-only resource scheduler for eight prospectively named frozen jobs.

--prepare writes the reviewable plan without fitting. --run executes that exact
plan, waiting for each child to exit before the next. Only after every expected
fit and result exists and verifies does the unchanged main finalize with one
validation child. This process never imports Torch, NumPy, or project modules.
"""
from analysis.private_ledger import private_value
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0084'))
CONTRACT = '2f6b6bcc4dd423dfe2dc0365180f1451888b31bc2cec3e1b8f0d8dd58e630467'
THREADS = {'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
PENDING = [('reviewed_export', 'dino_tcn', 'baseline', seed) for seed in (1729, 20260918)] + [
    ('reviewed_export', 'dino_tcn', arm, seed) for arm in ('global_control', 'short_boost') for seed in (3407, 1729, 20260918)]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Not a regular bound file: '+str(path))
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_identity(row):
    require(identity(row['path']) == row, 'Bound file changed: '+row['path'])


def result_path(study, job):
    return study/('result-'+'-'.join(map(str, job))+'.json')


def all_jobs(contract):
    return [(c, k, a, s) for c in contract['cohorts'] for k in contract['kinds']
            for a in contract['lossArms'] for s in contract['seeds']]


def inherited(cohort, kind, arm):
    return cohort in ('exact', 'draft') and kind == 'tcn' and arm == 'baseline'


def code_and_contract(study):
    reg = read(study/'preregistration.json')
    require(reg['sha256'] == CONTRACT == canonical(reg['contract']), 'Registered contract changed')
    c = reg['contract']
    require(c['experiment'] == 'bounded-short-boost-dino-transfer-v1' and len(c['code']) == 16, 'Wrong frozen experiment')
    for name, sha in c['code'].items():
        require(Path(name).name == name and digest(REPO/'analysis'/name) == sha, 'Frozen source changed: '+name)
    return c


def inputs(contract, study):
    return [identity(study/'preregistration.json'), identity(ROOT/'manifest-pts-v1.json'),
            identity(contract['dinoManifest']['path']), identity(contract['preflight']['path']),
            identity(Path(contract['referenceStudy']['path'])/'preregistration.json'),
            identity(Path(contract['referenceStudy']['path'])/'report.json')]


def common_command(contract, study):
    return [sys.executable, '-u', '-m', 'analysis.neural_short_boost_transfer',
            '--manifest', str(ROOT/'manifest-pts-v1.json'), '--dino-manifest', contract['dinoManifest']['path'],
            '--output', str(study), '--reference-study', contract['referenceStudy']['path'],
            '--preflight', contract['preflight']['path'], '--device', 'cuda']


def prepare(leaf):
    study = ROOT/'study'
    require(not leaf.exists(), 'Driver plan leaf already exists')
    c = code_and_contract(study)
    recovery = ROOT/'runtime-recovery-v3/recovery-verification.json'
    r = read(recovery)
    require(r['passed'] is True and r['readyToResume'] is True and r['contractSha256'] == CONTRACT
            and r['completedFitsUnchanged'] == 655 and r['resultCellsUnchanged'] == 46, 'V3 recovery is not verified')
    jobs = all_jobs(c)
    require(len(jobs) == len(set(jobs)) == 54 and set(PENDING) <= set(jobs), 'Unexpected factorial grid')
    original = [identity(result_path(study, job)) for job in jobs if job not in PENDING]
    require(len(original) == 46 and {Path(r['path']) for r in original} == set(study.glob('result-*.json'))
            and all(not result_path(study, job).exists() for job in PENDING), 'Pending set differs from exactly eight absent result cells')
    complete = sorted((study/'fits').rglob('completed.json'))
    require(len(complete) == 655 and not (study/'report.json').exists(), 'Prelaunch study inventory differs')
    bound_inputs = inputs(c, study)
    require(bound_inputs[1]['sha256'] == c['manifestSha256'] and bound_inputs[2]['sha256'] == c['dinoManifest']['sha256']
            and bound_inputs[3]['sha256'] == c['preflight']['sha256']
            and bound_inputs[5]['sha256'] == c['referenceStudy']['reportSha256'], 'Bound source/data identity differs')
    reference_reg = read(bound_inputs[4]['path'])
    require(reference_reg['sha256'] == c['referenceStudy']['contractSha256'] == canonical(reference_reg['contract']), 'Historical registration differs')
    plan = {'kind': 'sequential-frozen-transfer-recovery-plan-v3', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'contractSha256': CONTRACT, 'driver': identity(Path(__file__)), 'recovery': identity(recovery),
            'pythonExecutable': sys.executable, 'workingDirectory': str(REPO), 'threadEnvironment': THREADS,
            'registeredCode': c['code'], 'boundInputs': bound_inputs, 'pendingJobs': [list(j) for j in PENDING],
            'originalResultFiles': original, 'originalResultSetSha256': canonical(original),
            'existingCompletedMetadata': [identity(p) for p in complete],
            'commonCommand': common_command(c, study), 'trainingConcurrency': 1, 'parentImportsTorchOrNumPy': False,
            'jobExecution': 'Blocking subprocess.run; child must exit0 before next starts. No quality-based scheduling or selection.',
            'finalizationCommand': common_command(c, study)+['--workers', '1'],
            'finalizationGate': {'resultCells': 54, 'freshCompletedFits': 768, 'reusedCompletedFits': 96,
                                 'logicalCompletedFits': 864, 'NPZArtifacts': 5616, 'trainingPermitted': False},
            'numericalRecipeChanged': False, 'retainedParityPlan': identity(ROOT/'runtime-recovery-v2/regeneration-parity-plan.json')}
    leaf.mkdir(parents=False, exist_ok=False)
    with (leaf/'driver-source.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    write_new(leaf/'plan.json', plan)
    print(json.dumps({'prepared': True, 'pendingCells': 8, 'trainingConcurrency': 1,
                      'finalizationWorkers': 1, 'plan': identity(leaf/'plan.json')}, indent=2), flush=True)


def validate_plan(leaf):
    plan = read(leaf/'plan.json')
    require(plan['contractSha256'] == CONTRACT and plan['pendingJobs'] == [list(j) for j in PENDING]
            and plan['threadEnvironment'] == THREADS and plan['trainingConcurrency'] == 1
            and plan['pythonExecutable'] == sys.executable and plan['workingDirectory'] == str(REPO)
            and plan['parentImportsTorchOrNumPy'] is False, 'Driver scope/environment changed')
    verify_identity(plan['driver'])
    require(digest(leaf/'driver-source.py') == plan['driver']['sha256'], 'Driver snapshot changed')
    verify_identity(plan['recovery'])
    verify_identity(plan['retainedParityPlan'])
    for row in plan['boundInputs']:
        verify_identity(row)
    c = code_and_contract(ROOT/'study')
    require(c['code'] == plan['registeredCode'] and plan['commonCommand'] == common_command(c, ROOT/'study')
            and plan['finalizationCommand'] == plan['commonCommand']+['--workers', '1'], 'Frozen command/code identity differs')
    for row in plan['originalResultFiles']:
        verify_identity(row)
    for row in plan['existingCompletedMetadata']:
        verify_identity(row)
    return plan, c


def finalization_inventory(contract):
    """All paths already complete; hash NPZ bytes without opening tensor arrays."""
    study, records, fresh_paths = ROOT/'study', [], set()
    fresh = reused = artifacts = 0
    result_files = []
    for job in all_jobs(contract):
        co, kind, arm, seed = job
        result = result_path(study, job)
        meta_result = read(result)
        require(tuple(meta_result[k] for k in ('cohort', 'kind', 'lossArm', 'seed')) == job
                and meta_result['contractSha256'] == CONTRACT, 'Completed result identity differs')
        result_files.append(identity(result))
        old = inherited(co, kind, arm)
        root = Path(contract['referenceStudy']['path'])/'fits'/co/kind/str(seed) if old else study/'fits'/co/kind/arm/str(seed)
        fit_contract = contract['referenceStudy']['contractSha256'] if old else CONTRACT
        for outer in range(4):
            for name in ('inner-0', 'inner-1', 'inner-2', 'refit'):
                path = root/f'outer-{outer}'/name/'completed.json'
                info, meta = identity(path), read(path)
                require(meta['contractSha256'] == fit_contract and meta['kind'] == kind and meta['seed'] == seed
                        and (old or meta['lossArm'] == arm), 'Completed fit identity differs')
                require(meta['epochs'] == contract['checkpointEpochs'] if name != 'refit' else
                        len(meta['epochs']) == 1 and meta['epochs'][0] in contract['checkpointEpochs'], 'Checkpoint epochs differ')
                names = {f'{stem}-{epoch}.npz' for epoch in meta['epochs'] for stem in ('weights', 'predictions')}
                require(set(meta['artifacts']) == names and {p.name for p in path.parent.iterdir()} == names | {'completed.json'}, 'Fit artifact inventory differs')
                hashes = []
                for filename, expected in sorted(meta['artifacts'].items()):
                    item = identity(path.parent/filename)
                    require(item['sha256'] == expected, 'Completed NPZ bytes changed')
                    hashes.append(item)
                records.append({'completed': info, 'reused': old, 'artifactCount': len(hashes), 'artifactSetSha256': canonical(hashes)})
                artifacts += len(hashes)
                if old:
                    reused += 1
                else:
                    fresh += 1
                    fresh_paths.add(path)
    require((fresh, reused, len(records), artifacts) == (768, 96, 864, 5616), 'Finalization fit counts differ')
    require(set((study/'fits').rglob('completed.json')) == fresh_paths, 'Unexpected fresh fit metadata')
    actual_leaves = {p for p in (study/'fits').rglob('*') if p.is_dir() and p.name in ('inner-0', 'inner-1', 'inner-2', 'refit')}
    require(actual_leaves == {p.parent for p in fresh_paths}, 'Incomplete or extra fit destination before finalization')
    require(set(study.glob('result-*.json')) == {Path(r['path']) for r in result_files}, 'Final result-file grid differs')
    return {'resultCells': 54, 'freshCompletedFits': fresh, 'reusedCompletedFits': reused,
            'logicalCompletedFits': len(records), 'NPZArtifacts': artifacts, 'trainingPermitted': False,
            'fits': records, 'resultFiles': result_files, 'completedMetadataSetSha256': canonical([r['completed'] for r in records])}


def run(leaf):
    plan, c = validate_plan(leaf)
    require(not (leaf/'driver-started.json').exists() and not (ROOT/'study/report.json').exists(), 'Driver already started or study complete')
    require(all(not result_path(ROOT/'study', j).exists() for j in PENDING), 'Pending jobs changed before launch')
    env = os.environ.copy()
    env.update(THREADS)
    started = {'kind': 'sequential-frozen-transfer-recovery-execution-v3', 'status': 'running',
               'contractSha256': CONTRACT, 'startedAt': datetime.now(timezone.utc).isoformat(),
               'plan': identity(leaf/'plan.json'), 'pid': os.getpid(), 'trainingConcurrency': 1}
    write_new(leaf/'driver-started.json', started)
    started_clock = time.perf_counter()
    try:
        for index, job in enumerate(PENDING):
            validate_plan(leaf)
            log = leaf/f'job-{index}.log'
            command = plan['commonCommand']+['--job', ':'.join(map(str, job))]
            print(json.dumps({'phase': 'training-job-start', 'index': index, 'job': job,
                              'at': datetime.now(timezone.utc).isoformat()}), flush=True)
            begin = time.perf_counter()
            with log.open('x', encoding='utf-8') as stream:
                subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            validate_plan(leaf)
            result = identity(result_path(ROOT/'study', job))
            write_new(leaf/f'job-{index}-completed.json', {'job': job, 'command': command, 'exitCode': 0,
                      'wallSeconds': time.perf_counter()-begin, 'result': result, 'log': identity(log)})
            print(json.dumps({'phase': 'training-job-completed', 'index': index, 'job': job}), flush=True)
        validate_plan(leaf)
        gate = finalization_inventory(c)
        require({k: gate[k] for k in plan['finalizationGate']} == plan['finalizationGate'], 'Prospective finalization gate differs')
        write_new(leaf/'finalization-gate.json', {'kind': 'all-complete-no-training-finalization-gate-v3',
                  'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
                  'plan': identity(leaf/'plan.json'), **gate})
        before_executions = set((ROOT/'study').glob('execution-*.json'))
        print(json.dumps({'phase': 'frozen-main-validation-only', 'workers': 1, 'freshFitsAlreadyComplete': 768}), flush=True)
        with (leaf/'finalization.log').open('x', encoding='utf-8') as stream:
            subprocess.run(plan['finalizationCommand'], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        validate_plan(leaf)
        for row in gate['fits']:
            verify_identity(row['completed'])
        for row in gate['resultFiles']:
            verify_identity(row)
        require(set((ROOT/'study/fits').rglob('completed.json')) == {Path(r['completed']['path']) for r in gate['fits'] if not r['reused']}, 'Finalization created an unexpected fit')
        new_executions = set((ROOT/'study').glob('execution-*.json'))-before_executions
        require(len(new_executions) == 1, 'Unexpected finalization execution inventory')
        final_execution = next(iter(new_executions))
        final = read(final_execution)
        report = identity(ROOT/'study/report.json')
        require(final['status'] == 'completed' and final['workers'] == 1 and final['contractSha256'] == CONTRACT
                and final['reportSha256'] == report['sha256'], 'Frozen finalization failed')
        value = {**started, 'status': 'completed', 'completedAt': datetime.now(timezone.utc).isoformat(),
                 'wallSeconds': time.perf_counter()-started_clock, 'pendingJobsCompleted': 8,
                 'originalResultFilesUnchanged': 46, 'finalizationPerformedTraining': False,
                 'finalizationGate': identity(leaf/'finalization-gate.json'), 'finalExecution': identity(final_execution),
                 'report': report, 'finalizationLog': identity(leaf/'finalization.log')}
        write_new(leaf/'driver-completed.json', value)
        print(json.dumps({'phase': 'complete', 'execution': identity(leaf/'driver-completed.json'), 'report': report}), flush=True)
    except BaseException as error:
        write_new(leaf/'driver-failed.json', {**started, 'status': 'failed', 'errorType': type(error).__name__, 'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--leaf', type=Path, default=ROOT/'runtime-recovery-v3/sequential-driver')
    args = parser.parse_args()
    require(args.leaf.resolve() == (ROOT/'runtime-recovery-v3/sequential-driver').resolve(), 'Unexpected driver artifact location')
    if args.prepare:
        prepare(args.leaf)
    else:
        run(args.leaf)


if __name__ == '__main__':
    main()
