"""Explicit recovery and execution using the registered legacy-manifest read view."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
N = Path(private_value('private-reference-0061'))
ROOT = N/'evaluation-manifest-view-v1'
ORIGINAL = N/'evaluation-queue-v1'
STAGE = 'neural-evaluation'


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value):
    require(ref(value['path']) == value, 'Changed reference: '+value['path'])
    return Path(value['path'])


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def view():
    from analysis import neural_original_manifest_view
    return neural_original_manifest_view


def original_helper():
    path = REPO/'scripts/run-neural-nonstudent-evaluation.py'
    require(ref(path)['sha256'] == 'cf067e4da245db1f48c63eb2d83c7b2c75be2d6c751331b167ad839ba7959495', 'Frozen scheduling helper changed')
    return module('frozen_manifest_view_scheduling_helper', path)


def process_identity(pid):
    return original_helper().process_identity(pid)


def companion_path(job):
    return Path(job['fitDirectory'])/('evaluation-'+job['precision']+'-manifest-view-execution.json')


def outputs(job):
    path = Path(job['fitDirectory'])/('evaluation-'+job['precision']+'.json')
    return {'evaluation': ref(path),
            'publicationGate': ref(path.with_name(path.stem+'-publication-gate.json')),
            'queueExecution': ref(path.with_name(path.stem+'-queue-execution.json'))}


def expected_cells(plan, scope):
    require(scope in ('nonstudent', 'all'), 'Unknown execution scope')
    return plan['nonstudentCellIds'] if scope == 'nonstudent' else plan['allCellIds']


def register(args):
    helper = original_helper(); worker = helper.load_worker()
    original = read(ORIGINAL/'plan.json')
    included, excluded = helper.partition(original['jobs'], lambda r: read(worker.verify(r)))
    view_plan = view().verify_plan(args.view_plan)
    view().verify_qualification(args.view_qualification, args.view_plan)
    require(view_plan['originalEvaluationPlan'] == ref(ORIGINAL/'plan.json')
            and view_plan['legacyManifest'] == original['originalManifest'], 'Compatibility input association differs')
    prior = N/'nonstudent-evaluation-v1/execution-v1'
    incident = N/'nonstudent-evaluation-v1/first-cell-failure-v1/incident.json'
    failure = read(incident); actual = read(prior/'observed-exit.json')
    require(actual['exitCode'] == 1 and failure['actualExit'] == ref(prior/'observed-exit.json')
            and failure['cellLedgerPresent'] is False and failure['resultOrPublicationOrExecutionReceiptPresent'] is False,
            'Expected preserved first-cell schema failure')
    first = original['jobs'][0]
    failed_start = Path(first['fitDirectory'])/'evaluation-fp32-evaluation-start.json'
    require(ref(failed_start)['sha256'] == failure['snapshots']['failedStart']['snapshot']['sha256'], 'Failed start changed')
    ROOT.mkdir(exist_ok=True)
    sources = ('scripts/run-neural-manifest-view-evaluation.py', 'analysis/tests/test_manifest_view_evaluation.py',
               'docs/research/neural-original-manifest-evaluation-recovery-2026-09-23.md')
    plan = {'kind': 'registered-manifest-view-evaluation-v1', 'createdAtUTC': now(), 'source': ref(__file__),
        'code': {name: ref(REPO/name) for name in sources}, 'originalPlan': ref(ORIGINAL/'plan.json'),
        'originalWorker': original['source'], 'schedulingHelper': ref(REPO/'scripts/run-neural-nonstudent-evaluation.py'),
        'viewPlan': ref(args.view_plan), 'viewQualification': ref(args.view_qualification),
        'globalSelectionGate': ref(Path(original['globalSelectionGatePath'])),
        'globalIdentityCacheGate': ref(Path(original['globalSelectionGatePath']).with_name('global-selection-gate-identity-cache-gate.json')),
        'failedIncident': ref(incident), 'failedObservedExit': ref(prior/'observed-exit.json'),
        'failedProcess': ref(prior/'process.json'), 'failedStart': ref(failed_start),
        'allCellIds': [j['cellId'] for j in original['jobs']], 'nonstudentCellIds': included, 'studentCellIds': excluded,
        'studentPlan': ref(N/'student-streaming-v3/plan.json'),
        'studentProcess': ref(N/'student-streaming-v3/execution-v1/process-identity.json'),
        'studentLaunch': ref(N/'student-streaming-v3/execution-v1/launch.json'),
        'studentGrant': ref(N/'student-streaming-v3/execution-v1/grant.json'),
        'finalizationV3': ref(N/'finalization-v3/plan.json'),
        'supersedesUnlaunchedObservers': [ref(N/name/'proposal.json') for name in ('original-evaluation-resume-v1', 'original-evaluation-resume-v2')],
        'outputRoots': {scope: str(ROOT/('execution-'+scope+'-v1')) for scope in ('nonstudent', 'all')},
        'recoveryPath': str(ROOT/'recovery.json'), 'archivePath': str(ROOT/'preserved-failed-start.json'),
        'policy': {'unchangedOriginal270PlanAndCallables': True, 'unchangedOriginalArgv': True,
                   'perCellManifestViewCompanionRequired': True, 'studentActualExitAndIndependentColdRequired': True,
                   'prior243ActualExitRequiredBeforeAllScope': True, 'sameOriginalFlock': True,
                   'neverOverwriteFailedBytes': True, 'originalProgressNeverModified': True}}
    write_new(args.output, plan)
    print(json.dumps({'plan': ref(args.output), 'nonstudent': len(included), 'all': 270}), flush=True)


def verify_plan(path):
    plan = read(path)
    require(plan['kind'] == 'registered-manifest-view-evaluation-v1' and plan['source'] == ref(__file__), 'Wrong dispatcher source/plan')
    for reference in [*plan['code'].values(), plan['originalPlan'], plan['originalWorker'], plan['schedulingHelper'],
                      plan['viewPlan'], plan['viewQualification'], plan['globalSelectionGate'], plan['globalIdentityCacheGate'],
                      plan['failedIncident'], plan['failedObservedExit'], plan['failedProcess'], plan['studentPlan'],
                      plan['studentProcess'], plan['studentLaunch'], plan['studentGrant'], plan['finalizationV3'],
                      *plan['supersedesUnlaunchedObservers']]:
        checked(reference)
    view_plan = view().verify_plan(Path(plan['viewPlan']['path']))
    view().verify_qualification(Path(plan['viewQualification']['path']), Path(plan['viewPlan']['path']))
    helper = original_helper(); worker = helper.load_worker(); original = read(plan['originalPlan']['path'])
    require(view_plan['originalEvaluationPlan'] == plan['originalPlan']
            and view_plan['legacyManifest'] == original['originalManifest'], 'Compatibility input association differs')
    included, excluded = helper.partition(original['jobs'], lambda r: read(worker.verify(r)))
    require(included == plan['nonstudentCellIds'] and excluded == plan['studentCellIds']
            and [j['cellId'] for j in original['jobs']] == plan['allCellIds'], 'Changed task partition')
    require(plan['originalWorker'] == original['source']
            and plan['outputRoots'] == {scope: str(ROOT/('execution-'+scope+'-v1')) for scope in ('nonstudent', 'all')}
            and plan['recoveryPath'] == str(ROOT/'recovery.json')
            and plan['archivePath'] == str(ROOT/'preserved-failed-start.json'), 'Fixed execution/archive destination changed')
    failure = read(plan['failedIncident']['path'])
    for snapshot in failure['snapshots'].values(): checked(snapshot['snapshot'])
    expected_start = str(Path(original['jobs'][0]['fitDirectory'])/'evaluation-fp32-evaluation-start.json')
    require(failure['cellId'] == original['jobs'][0]['cellId']
            and failure['actualExit'] == plan['failedObservedExit']
            and plan['failedStart']['path'] == expected_start == failure['snapshots']['failedStart']['originalPath']
            and plan['failedStart']['sha256'] == failure['snapshots']['failedStart']['snapshot']['sha256'],
            'Only the exact interrupted first-start may be archived')
    return plan, helper, worker, original


def grant(path, plan_path, action, scope=None):
    value = read(path)
    require(value['kind'] == 'manifest-view-evaluation-grant-v1' and value['authorized'] is True
            and value['plan'] == ref(plan_path) and value['action'] == action
            and value.get('scope') == scope, 'Explicit reviewed execution grant required')
    return value


def recover(args):
    plan, helper, worker, original = verify_plan(args.plan)
    grant(args.grant, args.plan, 'archive-failed-start')
    prior = read(checked(plan['failedObservedExit'])); identity = read(checked(plan['failedProcess']))['identity']
    require(prior['exitCode'] == 1 and not (Path('/proc')/str(identity['pid'])).exists(), 'Failed worker must be stopped')
    source = checked(plan['failedStart']).resolve(); destination = Path(plan['archivePath']).resolve()
    require(source.is_relative_to(N.resolve()) and destination.is_relative_to(N.resolve())
            and not destination.exists() and not Path(plan['recoveryPath']).exists(), 'Unsafe or duplicate archive target')
    with (ORIGINAL/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in original['jobs']:
            folder = Path(job['fitDirectory']); prefix = 'evaluation-'+job['precision']
            for path in folder.glob(prefix+'*.json'):
                require(path.resolve() == source, 'Unexpected result or partial artifact: '+str(path))
        source.rename(destination)
        require(ref(destination)['sha256'] == plan['failedStart']['sha256'] and not source.exists(), 'Archive byte identity differs')
        helper.zero_outputs(original['jobs'])
        write_new(plan['recoveryPath'], {'kind': 'preserved-manifest-view-first-start-recovery-v1', 'passed': True,
            'plan': ref(args.plan), 'grant': ref(args.grant), 'actualFailureExit': plan['failedObservedExit'],
            'originalPath': str(source), 'originalReference': plan['failedStart'], 'archivedStart': ref(destination),
            'archiveBytesUnchanged': True, 'all270OutputPathsEmpty': True, 'createdAtUTC': now()})
    print(json.dumps({'recovery': ref(plan['recoveryPath'])}), flush=True)


def verify_recovery(plan_path, plan):
    recovery = read(plan['recoveryPath'])
    require(recovery['kind'] == 'preserved-manifest-view-first-start-recovery-v1' and recovery['passed'] is True
            and recovery['plan'] == ref(plan_path) and recovery['originalReference'] == plan['failedStart']
            and recovery['archiveBytesUnchanged'] is True and recovery['all270OutputPathsEmpty'] is True
            and ref(checked(recovery['archivedStart']))['sha256'] == plan['failedStart']['sha256'], 'Explicit failed-start recovery required')
    return ref(plan['recoveryPath'])


def require_complete_scope(document, plan_ref, scope, expected):
    require(document['kind'] == 'manifest-view-evaluation-completed-v1' and document['passed'] is True
            and document['plan'] == plan_ref and document['scope'] == scope
            and document['completedCellIds'] == expected and document['count'] == len(expected), 'Exact prior scope completion required')


def verify_scope_exit(plan_path, plan, scope):
    root = Path(plan['outputRoots'][scope])
    done_ref = ref(root/'completed.json'); done = read(done_ref['path'])
    expected = expected_cells(plan, scope)
    require_complete_scope(done, ref(plan_path), scope, expected)
    actual = read(root/'observed-exit.json')
    require(actual['kind'] == 'manifest-view-evaluation-observed-exit-v1' and actual['exitCode'] == 0
            and actual['scopeComplete'] is True and actual['scope'] == scope
            and actual['completed'] == done_ref and actual['plan'] == ref(plan_path), 'Actual completed scope exit0 required')
    launch = read(checked(actual['launch'])); process = read(checked(actual['process']))
    require(launch['plan'] == ref(plan_path) and launch['scope'] == scope
            and launch['source'] == plan['source'] and process['launch'] == actual['launch']
            and not (Path('/proc')/str(process['identity']['pid'])).exists(), 'Completed worker identity differs or is still present')
    grant(checked(launch['grant']), plan_path, 'run', scope)
    started = read(checked(done['started']))
    require(started['plan'] == ref(plan_path) and started['scope'] == scope
            and started['identity'] == process['identity'] and started['grant'] == launch['grant']
            and done['originalProgressModified'] is False, 'Scope execution association differs')
    rows = [json.loads(line) for line in checked(done['ledger']).read_text().splitlines() if line.strip()]
    require(len(rows) == len(expected) and len({r['cellId'] for r in rows}) == len(expected)
            and {r['cellId'] for r in rows} == set(expected), 'Complete scope ledger required')
    jobs = {job['cellId']: job for job in read(checked(plan['originalPlan']))['jobs']}
    for row in rows:
        job = jobs[row['cellId']]
        require(row['viewExecution'] == ref(companion_path(job)) and row['outputs'] == outputs(job),
                'Scope ledger is not bound to its own canonical cell')
        require(row['resumed'] is (scope == 'all' and row['cellId'] in plan['nonstudentCellIds']),
                'Scope ledger reuse classification differs')
    return {'completed': done_ref, 'observedExit': ref(root/'observed-exit.json'),
            'started': done['started'], 'ledger': done['ledger']}


def student_gate(plan, authorization):
    prior = verify_scope_exit(checked(authorization['plan']), plan, 'nonstudent')
    exit_ref = authorization['studentObservedExit']; actual_student = read(checked(exit_ref))
    student_identity = read(checked(plan['studentProcess']))
    require(actual_student['kind'] == 'student-streaming-cache-observed-exit-v3' and actual_student['exitCode'] == 0
            and actual_student['all27Completed'] is True and actual_student['plan'] == plan['studentPlan']
            and actual_student['grant'] == plan['studentGrant'] and actual_student['launch'] == plan['studentLaunch']
            and actual_student['pid'] == student_identity['pid']
            and actual_student['originalInterruptedWorkerExitStillUnknown'] is True
            and not (Path('/proc')/str(student_identity['pid'])).exists(), 'Actual student27 exit0 required')
    final_source = REPO/'scripts/finalize-neural-generalization-v3.py'
    require(ref(final_source)['sha256'] == '97d2667ec3d85d9830fa4f3a17ba606beff9619a6a0c51528bdcb63adeefc961', 'Frozen student finalizer changed')
    finalizer = module('registered_student_cold_gate_for_manifest_dispatch', final_source)
    final_plan, _ = finalizer.verify_plan(checked(plan['finalizationV3']))
    cold = finalizer.preflight(final_plan, checked(authorization['studentColdAudit']))
    require(cold['queuePlan'] == plan['studentPlan'] and cold['queueCompleted'] == actual_student['completed']
            and cold['all27PublicationGatesPassed'] is True, 'Independent complete student cold gate required')
    return {'prior243Completed': prior['completed'], 'prior243ActualExit': prior['observedExit'],
            'studentActualExit': exit_ref, 'studentColdPublication': cold}


def check_cell(worker, original, global_gate, panel, job, plan):
    if not worker.completed(job, original, global_gate, panel):
        return False
    _, argv, _ = worker.evaluation_call(job, original, global_gate, panel)
    view().verify_execution(companion_path(job), Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
                            stage=STAGE, argv=argv, outputs=outputs(job))
    return True


def run(args):
    plan, helper, worker, original = verify_plan(args.plan)
    authorization = grant(args.grant, args.plan, 'run', args.scope)
    require(str(args.output_root) == plan['outputRoots'][args.scope], 'Unregistered output destination')
    recovery = verify_recovery(args.plan, plan)
    require(sorted(os.sched_getaffinity(0)) == [18, 19] and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Original CPU policy required')
    require(all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('TMPDIR', '').startswith(str(N)+'/') and Path(os.environ['TMPDIR']).is_dir(), 'NAS/two-thread runtime required')
    with (ORIGINAL/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        helper.verify_original(worker, original)
        global_gate = worker.global_ready(original)
        require(global_gate is not None and global_gate['reference'] == plan['globalSelectionGate'], 'All162 global freeze required')
        extra = student_gate(plan, authorization) if args.scope == 'all' else None
        if args.scope == 'nonstudent':
            helper.zero_outputs(original['jobs'])
        expected = expected_cells(plan, args.scope); wanted = set(expected)
        jobs = [j for j in original['jobs'] if j['cellId'] in wanted]
        started = {'kind': 'manifest-view-evaluation-started-v1', 'plan': ref(args.plan), 'grant': ref(args.grant),
            'scope': args.scope, 'source': ref(__file__), 'identity': process_identity(os.getpid()),
            'recovery': recovery, 'globalSelectionGate': global_gate['reference'], 'studentGate': extra, 'startedAtUTC': now()}
        write_new(args.output_root/'started.json', started)
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        done = set(); began = time.monotonic()
        bindings = {name: getattr(worker, name) for name in helper.CALLABLES}; frozen_root = worker.ROOT
        workflow = worker.workflow(); frozen_evaluate = workflow.evaluate
        while len(done) < len(expected) and not stopping and time.monotonic()-began < 12*3600:
            require(not worker.stop_requested() and not (args.output_root/'STOP').exists(), 'Stopped; preserve current artifacts')
            require(worker.ROOT == frozen_root and workflow.evaluate is frozen_evaluate
                    and all(getattr(worker, k) is v for k, v in bindings.items()), 'Frozen worker was replaced')
            require(worker.global_ready(original) == global_gate, 'Global freeze changed')
            worked = False
            for job in jobs:
                if stopping:
                    break
                if job['cellId'] in done:
                    continue
                panel = worker.panel_ready(original, job['precision'])
                if panel is None:
                    continue
                prerequisites = worker.ready(job, original, global_gate, panel)
                if prerequisites is None:
                    continue
                resumed = check_cell(worker, original, global_gate, panel, job, plan)
                if not resumed:
                    if not worker.resources()['passed']:
                        break
                    for reference in original['code'].values():
                        worker.verify(reference)
                    path = Path(job['fitDirectory'])/('evaluation-'+job['precision']+'.json')
                    require(not path.exists() and not companion_path(job).exists(), 'Unpublished/partial output must be preserved')
                    _, argv, _ = worker.evaluation_call(job, original, global_gate, panel)
                    with view().installed(Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
                            stage=STAGE, argv=argv, extra_bindings={'worker.execute_cell': (worker, 'execute_cell'),
                            'workflow.evaluate': (worker.workflow(), 'evaluate')}) as proof:
                        worker.execute_cell(job, original, global_gate, panel, prerequisites)
                    write_new(companion_path(job), {**proof, 'outputs': outputs(job)})
                    require(check_cell(worker, original, global_gate, panel, job, plan), 'Original/view publication gates missing')
                done.add(job['cellId']); worked = True
                row = {'time': now(), 'cellId': job['cellId'], 'resumed': resumed,
                       'viewExecution': ref(companion_path(job)), 'outputs': outputs(job)}
                with (args.output_root/'cells.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                print(json.dumps({'completed': len(done), 'total': len(expected), 'cellId': job['cellId'], 'resumed': resumed}), flush=True)
            if not worked and not stopping:
                time.sleep(30)
        require(len(done) == len(expected), 'Scope stopped incomplete; no completion claim')
        require(worker.ROOT == frozen_root and workflow.evaluate is frozen_evaluate
                and all(getattr(worker, k) is v for k, v in bindings.items()), 'Frozen worker changed')
        write_new(args.output_root/'completed.json', {'kind': 'manifest-view-evaluation-completed-v1', 'passed': True,
            'plan': ref(args.plan), 'scope': args.scope, 'count': len(done), 'completedCellIds': expected,
            'started': ref(args.output_root/'started.json'), 'ledger': ref(args.output_root/'cells.jsonl'),
            'finishedAtUTC': now(), 'originalProgressModified': False})


def supervise(args):
    plan, _, _, _ = verify_plan(args.plan)
    grant(args.grant, args.plan, 'run', args.scope)
    root = Path(plan['outputRoots'][args.scope]); root.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), 'run', '--plan', str(args.plan), '--grant', str(args.grant),
               '--scope', args.scope, '--output-root', str(root)]
    write_new(root/'launch.json', {'kind': 'manifest-view-evaluation-launch-v1', 'plan': ref(args.plan), 'grant': ref(args.grant),
        'source': ref(__file__), 'scope': args.scope, 'command': command, 'identity': process_identity(os.getpid()), 'startedAtUTC': now()})
    with (root/'stdout.log').open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        write_new(root/'process.json', {'launch': ref(root/'launch.json'), 'identity': process_identity(child.pid)})
        print(json.dumps({'pid': child.pid, 'launch': ref(root/'launch.json')}), flush=True)
        code = child.wait()
    completed = ref(root/'completed.json') if (root/'completed.json').exists() else None
    full = False; completion_error = None
    if code == 0 and completed:
        try:
            require_complete_scope(read(completed['path']), ref(args.plan), args.scope, expected_cells(plan, args.scope))
            full = True
        except Exception as error:
            completion_error = type(error).__name__+': '+str(error)
    write_new(root/'observed-exit.json', {'kind': 'manifest-view-evaluation-observed-exit-v1', 'plan': ref(args.plan),
        'scope': args.scope, 'launch': ref(root/'launch.json'), 'process': ref(root/'process.json'),
        'exitCode': code, 'scopeComplete': full, 'completed': completed,
        'completionValidationError': completion_error, 'finishedAtUTC': now()})
    print(json.dumps({'actualExitCode': code, 'scopeComplete': full, 'observedExit': ref(root/'observed-exit.json')}), flush=True)
    if code:
        raise SystemExit(code)
    require(full, 'Observed exit0 is not full scope completion')


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('register')
    for name in ('view-plan', 'view-qualification', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('recover')
    for name in ('plan', 'grant'):
        p.add_argument('--'+name, type=Path, required=True)
    for action in ('run', 'supervise'):
        p = sub.add_parser(action)
        for name in ('plan', 'grant'):
            p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--scope', choices=('nonstudent', 'all'), required=True)
        if action == 'run':
            p.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    require(Path('/mnt/freenas').is_mount(), 'Direct NAS mount required')
    if args.action == 'register':
        require(args.output.resolve().is_relative_to(N.resolve()), 'Direct study NAS output required')
    {'register': register, 'recover': recover, 'run': run, 'supervise': supervise}[args.action](args)


if __name__ == '__main__':
    main()
