#!/usr/bin/env python3
"""Finalize an already complete frozen study with no overlapping CUDA owners.

The scheduler imports only the standard library. Each original --job command
revalidates its existing completed fits in a separate blocking subprocess.
Only the explicitly selected --aggregate child imports the frozen analysis
module, validates every input, and writes its exact original report payload.
This is an operational continuation, not completion of the interrupted main.
"""
from analysis.private_ledger import private_value
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0084'))
LEAF = ROOT/'runtime-recovery-v4/finalization-driver'
OLD_LEAF = ROOT/'runtime-recovery-v3/sequential-driver'
OLD_DRIVER_SHA = '6256ae6b7f416ec15d6a4c6e8433a068c45fb5b0292e7698d008f07ebfb9a0a5'
COUNTS = {'resultCells': 54, 'freshCompletedFits': 768, 'reusedCompletedFits': 96,
          'logicalCompletedFits': 864, 'NPZArtifacts': 5616, 'trainingPermitted': False}


def old_driver():
    path = REPO/'scripts/resume-neural-transfer-sequential.py'
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != OLD_DRIVER_SHA:
        raise ValueError('Previously frozen sequential driver changed')
    spec = importlib.util.spec_from_file_location('frozen_transfer_sequential_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = old_driver()
require, read, identity, verify_identity = D.require, D.read, D.identity, D.verify_identity
write_new, canonical = D.write_new, D.canonical


def verify_gate(gate, contract):
    """Rehash all NPZ bytes and compare the complete old gate inventory."""
    require({k: gate[k] for k in COUNTS} == COUNTS, 'Wrong all-complete inventory counts')
    observed = D.finalization_inventory(contract)
    require(observed == {k: gate[k] for k in observed}, 'Completed inventory differs from interrupted finalization gate')
    return observed


def build_report(contract, digest, results):
    return {'schemaVersion': 1, 'contractSha256': digest, 'manifestSha256': contract['manifestSha256'],
            'status': 'completed-short-boost-transfer-development', 'records': contract['evaluationPopulation']['records'],
            'sourceGroups': contract['groups'], 'protectedTestOpened': False,
            'productionPromotionAllowed': False, 'results': results}


def frozen_report_expression(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_study')
    expressions = [n.value for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'report' for target in n.targets)]
    require(len(expressions) == 1 and isinstance(expressions[0], ast.Dict), 'Frozen report assignment is ambiguous')
    return ast.Expression(body=expressions[0])


def faithful_report(contract, digest, results, source):
    report = build_report(contract, digest, results)
    original = eval(compile(ast.fix_missing_locations(frozen_report_expression(source)), '<frozen-report>', 'eval'),
                    {'__builtins__': {}}, {'contract': contract, 'digest': digest, 'results': results})
    require(report == original and list(report) == list(original), 'Operational report differs from frozen report expression')
    return report


def job_command(plan, job):
    return plan['commonCommand']+['--job', ':'.join(map(str, job))]


def result_identity(gate, study, job):
    path = D.result_path(study, job)
    matches = [row for row in gate['resultFiles'] if row['path'] == str(path)]
    require(len(matches) == 1, 'Job result is not uniquely bound')
    return matches[0]


def verify_job_gate(gate, contract, job):
    """Every destination needed by the frozen job already has completed metadata."""
    cohort, kind, arm, seed = job
    reused = D.inherited(cohort, kind, arm)
    base = (Path(contract['referenceStudy']['path'])/'fits'/cohort/kind/str(seed) if reused
            else ROOT/'study/fits'/cohort/kind/arm/str(seed))
    expected = {base/f'outer-{outer}'/name/'completed.json' for outer in range(4)
                for name in ('inner-0', 'inner-1', 'inner-2', 'refit')}
    bound = [row for row in gate['fits'] if Path(row['completed']['path']) in expected]
    require(len(bound) == 16 and {Path(row['completed']['path']) for row in bound} == expected,
            'Missing completed fit destination before validation-only job')
    for row in bound:
        require(row['reused'] is reused, 'Fit reuse binding differs')
        verify_identity(row['completed'])
    verify_identity(result_identity(gate, ROOT/'study', job))


def verify_receipts(leaf, plan, gate):
    expected = {leaf/f'job-{index:02d}-completed.json' for index in range(len(plan['jobs']))}
    require(set(leaf.glob('job-*-completed.json')) == expected, 'Not all54 validation receipts exist')
    receipts = []
    for index, job in enumerate(plan['jobs']):
        path = leaf/f'job-{index:02d}-completed.json'
        row = read(path)
        require(row['index'] == index and row['job'] == job and row['command'] == job_command(plan, job)
                and row['exitCode'] == 0 and row['trainingPermitted'] is False
                and row['result'] == result_identity(gate, ROOT/'study', job), 'Validation receipt differs')
        verify_identity(row['result'])
        require(row['log']['path'] == str(leaf/f'job-{index:02d}.log'), 'Validation log path differs')
        verify_identity(row['log'])
        receipts.append(identity(path))
    return receipts


def prepare(leaf):
    require(not leaf.exists(), 'Operational continuation leaf already exists')
    old_plan, contract = D.validate_plan(OLD_LEAF)
    gate_path = OLD_LEAF/'finalization-gate.json'
    gate = read(gate_path)
    require(gate['contractSha256'] == D.CONTRACT and gate['plan'] == identity(OLD_LEAF/'plan.json'), 'Old finalization gate provenance differs')
    verify_gate(gate, contract)
    require(not (ROOT/'study/report.json').exists() and not (ROOT/'study/report.json.partial').exists(), 'Report or partial report already exists')
    require(not (OLD_LEAF/'driver-completed.json').exists(), 'Old driver unexpectedly completed')
    receipt_files = []
    for index, job in enumerate(D.PENDING):
        path = OLD_LEAF/f'job-{index}-completed.json'
        receipt = read(path)
        require(receipt['job'] == list(job) and receipt['command'] == old_plan['commonCommand']+['--job', ':'.join(map(str, job))]
                and receipt['exitCode'] == 0 and receipt['result'] == result_identity(gate, ROOT/'study', job), 'Old training receipt differs')
        verify_identity(receipt['result'])
        verify_identity(receipt['log'])
        receipt_files += [identity(path), receipt['log']]
    old_execution = ROOT/'study/execution-1789834863339799876.json'
    execution = read(old_execution)
    require(execution['status'] == 'running' and execution['workers'] == 1
            and execution['contractSha256'] == D.CONTRACT, 'Unexpected interrupted frozen execution')
    plan = {'kind': 'sequential-completed-transfer-finalization-plan-v4',
            'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': D.CONTRACT,
            'driver': identity(Path(__file__)), 'originalDriver': identity(REPO/'scripts/resume-neural-transfer-sequential.py'),
            'observation': identity(ROOT/'runtime-recovery-v4/observation.json'),
            'originalPlan': identity(OLD_LEAF/'plan.json'), 'originalGate': identity(gate_path),
            'interruptedExecution': identity(old_execution), 'interruptedLog': identity(OLD_LEAF/'finalization.log'),
            'originalTrainingReceiptsAndLogs': receipt_files,
            'boundInputs': old_plan['boundInputs']+[identity(ROOT/'study/weight-diagnostics.json')],
            'registeredCode': contract['code'], 'jobs': [list(job) for job in D.all_jobs(contract)],
            'commonCommand': old_plan['commonCommand'], 'pythonExecutable': sys.executable,
            'workingDirectory': str(REPO), 'threadEnvironment': D.THREADS, 'gate': COUNTS,
            'parentImportsTorchOrNumPy': False, 'maximumConcurrentAnalysisChildren': 1,
            'trainingPermitted': False, 'numericalRecipeChanged': False,
            'operation': 'All54 unchanged frozen --job commands sequentially; then isolated full initialize and exact frozen report aggregation.',
            'completionSemantics': 'New operational continuation completion; interrupted original execution and driver remain unchanged.'}
    require(len(plan['jobs']) == len({tuple(job) for job in plan['jobs']}) == 54, 'Wrong registered grid')
    leaf.mkdir(parents=False, exist_ok=False)
    (leaf/'driver-source.py').write_bytes(Path(__file__).read_bytes())
    write_new(leaf/'plan.json', plan)
    print(json.dumps({'prepared': True, 'plan': identity(leaf/'plan.json'), 'validationCells': 54,
                      'trainingPermitted': False, 'maximumConcurrentAnalysisChildren': 1}), flush=True)


def validate_plan(leaf):
    plan = read(leaf/'plan.json')
    old_plan, contract = D.validate_plan(OLD_LEAF)
    require(plan['contractSha256'] == D.CONTRACT and plan['registeredCode'] == contract['code']
            and plan['jobs'] == [list(job) for job in D.all_jobs(contract)]
            and plan['commonCommand'] == old_plan['commonCommand']
            and plan['gate'] == COUNTS and plan['trainingPermitted'] is False
            and plan['numericalRecipeChanged'] is False and plan['parentImportsTorchOrNumPy'] is False
            and plan['maximumConcurrentAnalysisChildren'] == 1
            and plan['pythonExecutable'] == sys.executable and plan['workingDirectory'] == str(REPO)
            and plan['threadEnvironment'] == D.THREADS, 'Continuation scope or environment differs')
    for key in ('driver', 'originalDriver', 'observation', 'originalPlan', 'originalGate', 'interruptedExecution', 'interruptedLog'):
        verify_identity(plan[key])
    require(D.digest(leaf/'driver-source.py') == plan['driver']['sha256'], 'Continuation source snapshot changed')
    for row in plan['boundInputs']+plan['originalTrainingReceiptsAndLogs']:
        verify_identity(row)
    gate = read(plan['originalGate']['path'])
    require(gate['contractSha256'] == D.CONTRACT, 'Old finalization gate changed')
    return plan, contract, gate


def aggregate(leaf):
    """Only this mode imports analysis/Torch; the scheduling parent stays stdlib."""
    plan, contract, gate = validate_plan(leaf)
    require((leaf/'driver-started.json').exists() and not (leaf/'driver-completed.json').exists()
            and not (leaf/'driver-failed.json').exists(), 'No active operational continuation')
    receipts = verify_receipts(leaf, plan, gate)
    aggregation_gate = read(leaf/'aggregation-gate.json')
    require(aggregation_gate['plan'] == identity(leaf/'plan.json')
            and aggregation_gate['contractSha256'] == D.CONTRACT
            and aggregation_gate['originalGate'] == plan['originalGate']
            and aggregation_gate['validationReceiptsSha256'] == canonical(receipts)
            and {key: aggregation_gate[key] for key in COUNTS} == COUNTS,
            'Full-revalidation aggregation gate differs')
    require(not (ROOT/'study/report.json').exists() and not (ROOT/'study/report.json.partial').exists(), 'Report already exists')
    sys.path.insert(0, str(REPO))
    from analysis import neural_short_boost_transfer as frozen
    data, actual_contract, digest = frozen.initialize(ROOT/'manifest-pts-v1.json', ROOT/'dino-manifest.json',
            ROOT/'study', Path(contract['referenceStudy']['path']), Path(contract['preflight']['path']), 'cuda')
    require(actual_contract == contract and digest == D.CONTRACT, 'Frozen final input validation changed contract')
    del data
    results = []
    for job in plan['jobs']:
        bound = result_identity(gate, ROOT/'study', job)
        verify_identity(bound)
        row = read(bound['path'])
        require([row[key] for key in ('cohort', 'kind', 'lossArm', 'seed')] == job
                and row['contractSha256'] == digest, 'Completed result identity differs')
        results.append(row)
    report = faithful_report(contract, digest, results, (REPO/'analysis/neural_short_boost_transfer.py').read_text(encoding='utf-8'))
    require(not (ROOT/'study/report.json').exists(), 'Report appeared before exclusive finalization')
    frozen.base.write_json(ROOT/'study/report.json', report)
    write_new(leaf/'aggregation-completed.json', {'kind': 'isolated-frozen-report-aggregation-v4',
              'completedAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': digest,
              'plan': identity(leaf/'plan.json'), 'report': identity(ROOT/'study/report.json'),
              'aggregationGate': identity(leaf/'aggregation-gate.json'),
              'validationReceiptsSha256': canonical(receipts), 'resultCells': 54,
              'fullFrozenInitializePassed': True, 'reportMatchesFrozenExpression': True,
              'trainingPermitted': False, 'protectedTestOpened': False, 'productionPromotionAllowed': False})
    print(json.dumps({'phase': 'aggregation-completed', 'resultCells': 54, 'report': identity(ROOT/'study/report.json')}), flush=True)


def run(leaf):
    plan, contract, gate = validate_plan(leaf)
    require(not (leaf/'driver-started.json').exists() and not (ROOT/'study/report.json').exists(), 'Continuation already started or report exists')
    verify_gate(gate, contract)
    env = os.environ.copy()
    env.update(D.THREADS)
    began = time.perf_counter()
    started = {'kind': 'sequential-completed-transfer-finalization-execution-v4', 'status': 'running',
               'contractSha256': D.CONTRACT, 'startedAt': datetime.now(timezone.utc).isoformat(),
               'plan': identity(leaf/'plan.json'), 'pid': os.getpid(), 'trainingPermitted': False}
    write_new(leaf/'driver-started.json', started)
    try:
        for index, job in enumerate(plan['jobs']):
            validate_plan(leaf)
            verify_job_gate(gate, contract, job)
            command = job_command(plan, job)
            log = leaf/f'job-{index:02d}.log'
            begin = time.perf_counter()
            print(json.dumps({'phase': 'validation-start', 'index': index, 'job': job,
                              'at': datetime.now(timezone.utc).isoformat()}), flush=True)
            with log.open('x', encoding='utf-8') as stream:
                subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            validate_plan(leaf)
            verify_job_gate(gate, contract, job)
            write_new(leaf/f'job-{index:02d}-completed.json', {'index': index, 'job': job, 'command': command,
                      'exitCode': 0, 'wallSeconds': time.perf_counter()-begin, 'trainingPermitted': False,
                      'result': result_identity(gate, ROOT/'study', job), 'log': identity(log)})
            print(json.dumps({'phase': 'validation-completed', 'index': index, 'job': job}), flush=True)
        receipts = verify_receipts(leaf, plan, gate)
        verify_gate(gate, contract)
        write_new(leaf/'aggregation-gate.json', {'kind': 'all54-revalidated-no-training-aggregation-gate-v4',
                  'createdAt': datetime.now(timezone.utc).isoformat(), 'plan': identity(leaf/'plan.json'),
                  'contractSha256': D.CONTRACT, 'originalGate': plan['originalGate'],
                  'validationReceiptsSha256': canonical(receipts), **COUNTS})
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--aggregate', '--leaf', str(leaf)]
        with (leaf/'aggregation.log').open('x', encoding='utf-8') as stream:
            subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        validate_plan(leaf)
        verify_gate(gate, contract)
        receipts = verify_receipts(leaf, plan, gate)
        aggregation = read(leaf/'aggregation-completed.json')
        report = identity(ROOT/'study/report.json')
        require(aggregation['plan'] == identity(leaf/'plan.json') and aggregation['contractSha256'] == D.CONTRACT
                and aggregation['report'] == report and aggregation['validationReceiptsSha256'] == canonical(receipts)
                and aggregation['aggregationGate'] == identity(leaf/'aggregation-gate.json')
                and aggregation['fullFrozenInitializePassed'] is True and aggregation['reportMatchesFrozenExpression'] is True,
                'Isolated aggregation completion differs')
        write_new(leaf/'driver-completed.json', {**started, 'status': 'completed',
                  'completedAt': datetime.now(timezone.utc).isoformat(), 'wallSeconds': time.perf_counter()-began,
                  'validationJobsCompleted': 54, 'originalMetadataResultsAndNPZUnchanged': True,
                  'finalizationPerformedTraining': False, 'interruptedExecutionPreserved': plan['interruptedExecution'],
                  'aggregationGate': identity(leaf/'aggregation-gate.json'),
                  'aggregation': identity(leaf/'aggregation-completed.json'), 'aggregationLog': identity(leaf/'aggregation.log'),
                  'report': report, 'validationReceiptsSha256': canonical(receipts),
                  'completionSemantics': plan['completionSemantics']})
        print(json.dumps({'phase': 'complete', 'completion': identity(leaf/'driver-completed.json'), 'report': report}), flush=True)
    except BaseException as error:
        write_new(leaf/'driver-failed.json', {**started, 'status': 'failed', 'endedAt': datetime.now(timezone.utc).isoformat(),
                  'errorType': type(error).__name__, 'error': str(error), 'wallSeconds': time.perf_counter()-began})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--aggregate', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--leaf', type=Path, default=LEAF)
    args = parser.parse_args()
    require(args.leaf.resolve() == LEAF.resolve(), 'Unexpected operational artifact location')
    if args.prepare:
        prepare(args.leaf)
    elif args.run:
        run(args.leaf)
    else:
        aggregate(args.leaf)


if __name__ == '__main__':
    main()
