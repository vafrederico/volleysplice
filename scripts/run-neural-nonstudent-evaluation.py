"""Schedule only frozen nonstudent cells through the unchanged original evaluator."""
from analysis.private_ledger import private_value
import argparse
import collections
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
N = Path(private_value('private-reference-0061'))
ORIGINAL = N/'evaluation-queue-v1'
RECOVERY = N/'cpu-selection-evaluation-recovery-v1/evaluation-queue-v1'
SOURCE_SHA = '170fdb0ee5cac74b1f50a822d136e39e7509c51820d5dbd9f6fede0279539e7c'
PLAN_SHA = 'd7800668fb9f434901e80410a5922b9e86f684b770d440a938476d65a910f2b5'
CALLABLES = ('global_ready', 'panel_ready', 'ready', 'completed', 'execute_cell',
             'evaluation_call', 'verify', 'sha', 'resources')


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


def checked(reference):
    require(ref(reference['path']) == reference, 'Reference changed: '+reference['path'])
    return Path(reference['path'])


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def process_identity(pid):
    folder = Path('/proc')/str(pid)
    stat = (folder/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'bootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'processStartTimeTicks': int(stat[19]),
            'cmdline': (folder/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
            'parentPid': int(stat[1])}


def load_worker():
    source = ORIGINAL/'evaluation-worker.py'
    require(ref(source)['sha256'] == SOURCE_SHA, 'Frozen evaluator source changed')
    require(ref(ORIGINAL/'plan.json')['sha256'] == PLAN_SHA, 'Frozen evaluator plan changed')
    spec = importlib.util.spec_from_file_location('unchanged_nonstudent_evaluation_worker', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partition(jobs, task_loader):
    require(len(jobs) == 270 and len({j['cellId'] for j in jobs}) == 270, 'Expected original270 cells')
    tasks = {}
    included, excluded = [], []
    for job in jobs:
        key = job['task']['sha256']
        if key not in tasks:
            tasks[key] = task_loader(job['task'])
        task = tasks[key]
        (excluded if task['model'] == 'distilled-mobile-tcn' else included).append(job['cellId'])
    require(len(tasks) == 162 and len(included) == 243 and len(excluded) == 27,
            'Expected exact243 nonstudent/27 student partition of162 tasks')
    counts = collections.Counter(j['precision'] for j in jobs if j['cellId'] in included)
    require(counts == {'fp32': 135, 'fp16': 54, 'int8': 54}, 'Nonstudent precision inventory differs')
    require(all(j['precision'] == 'fp32' for j in jobs if j['cellId'] in excluded), 'Student precision differs')
    return included, excluded


def zero_outputs(jobs):
    by_folder = {}
    for job in jobs:
        stem = 'evaluation-'+job['precision']
        by_folder.setdefault(job['fitDirectory'], set()).update(
            stem+suffix for suffix in ('.json', '-publication-gate.json', '-queue-execution.json', '-evaluation-start.json'))
    for folder, names in by_folder.items():
        if not Path(folder).exists():
            continue
        found = names.intersection(p.name for p in Path(folder).iterdir())
        require(not found, 'Original evaluator has output or partial output: '+str(Path(folder)/next(iter(found), '')))


def verify_original(worker, plan):
    for reference in [plan['source'], plan['routingTests'], plan['selectionQueuePlan'], plan['cachePlan'],
                      plan['executionPlan'], plan['serializationPlan'], plan['correctionPlan'], plan['inventory'],
                      plan['originalManifest'], *plan['registrations'], *plan['reusePlans'], *plan['code'].values()]:
        worker.verify(reference)


def register(args):
    worker = load_worker()
    original = read(ORIGINAL/'plan.json')
    included, excluded = partition(original['jobs'], lambda r: read(checked(r)))
    old = read(RECOVERY/'process.json')
    identity = process_identity(old['pid'])
    require(all(identity[k] == old[k] for k in ('pid', 'bootId', 'parentPid', 'cmdline')), 'Recovered original process identity changed')
    require(not Path(original['globalSelectionGatePath']).exists(), 'Register the schedule before external outcomes open')
    zero_outputs(original['jobs'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, {'kind': 'nonstudent-evaluation-schedule-v1', 'createdAtUTC': now(),
        'source': ref(__file__), 'tests': ref(REPO/'analysis/tests/test_nonstudent_evaluation.py'),
        'protocol': ref(REPO/'docs/research/neural-nonstudent-evaluation-schedule-2026-09-23.md'),
        'originalPlan': ref(ORIGINAL/'plan.json'), 'originalWorker': ref(ORIGINAL/'evaluation-worker.py'),
        'originalLaunch': ref(RECOVERY/'launch.json'), 'originalProcess': ref(RECOVERY/'process.json'),
        'originalIdentity': identity, 'includedCellIds': included, 'excludedStudentCellIds': excluded,
        'policy': {'originalPlanRetains270': True, 'all162GlobalFreezeRequired': True,
                   'unchangedOriginalCallables': list(CALLABLES), 'sameOriginalFlockRequired': True,
                   'originalActualExitZeroAndNoOutputsRequired': True, 'noSourceOrHashReplacement': True,
                   'studentCellsRequireLaterFullStudentColdProof': True, 'originalResumeAfterSubsetActualExit': True,
                   'cpuAffinity': [18, 19], 'niceAtLeast': 10, 'cudaVisibleDevices': '', 'threads': 2}})
    print(json.dumps({'registered': ref(args.output), 'included': len(included), 'excluded': len(excluded)}), flush=True)


def verify_plan(path):
    plan = read(path)
    require(plan['kind'] == 'nonstudent-evaluation-schedule-v1' and plan['source'] == ref(__file__), 'Scheduling source/plan differs')
    for name in ('tests', 'protocol', 'originalPlan', 'originalWorker', 'originalLaunch', 'originalProcess'):
        checked(plan[name])
    worker = load_worker()
    original = read(checked(plan['originalPlan']))
    require(plan['originalWorker'] == original['source'], 'Original source association differs')
    included, excluded = partition(original['jobs'], lambda r: read(worker.verify(r)))
    require(included == plan['includedCellIds'] and excluded == plan['excludedStudentCellIds'], 'Scheduling partition changed')
    return plan, worker, original


def verify_stop(plan_path, plan, stop_path):
    stop = read(stop_path)
    verify_stop_value(plan_path, plan, stop)
    return stop


def verify_stop_value(plan_path, plan, stop):
    require(stop['kind'] == 'observed-idle-evaluator-stop-v1' and stop['plan'] == ref(plan_path)
            and stop['originalIdentity'] == plan['originalIdentity'] and stop['noEvaluationOutputs'] is True,
            'Original idle-stop receipt differs')
    actual = read(checked(stop['observedExit']))
    require(actual['kind'] == 'observed-cpu-worker-exit-v1' and actual['exitCode'] == 0
            and actual['launch'] == plan['originalLaunch'] and actual['process'] == plan['originalProcess'],
            'Actual original recovery exit0 required')
    events = [json.loads(line) for line in checked(stop['eventsSnapshot']).read_text().splitlines() if line.strip()]
    require(events[-1]['stage'] == 'worker-stop' and events[-1]['completed'] == 0,
            'Original stopped after starting evaluation')
    require(not any(row['stage'] in ('evaluation-start', 'cell-complete') for row in events), 'Original outcomes were opened')
    require(not (Path('/proc')/str(plan['originalIdentity']['pid'])).exists(), 'Original evaluator PID is present')


def stop_original(args):
    require(args.permit_stop, 'Root review and explicit --permit-stop required')
    plan, worker, original = verify_plan(args.plan)
    expected = plan['originalIdentity']
    require(process_identity(expected['pid']) == expected, 'Do not signal a recycled or changed PID')
    require(not Path(original['globalSelectionGatePath']).exists(), 'Stop original before publishing global selection gate')
    require(not (Path('/proc')/str(expected['pid'])/'task'/str(expected['pid'])/'children').read_text().strip(), 'Original has active children')
    zero_outputs(original['jobs'])
    require(not args.output.exists(), 'Stop proof already exists')
    was_stopped = 'T' in (Path('/proc')/str(expected['pid'])/'status').read_text().split('State:', 1)[1].splitlines()[0]
    os.kill(expected['pid'], signal.SIGTERM)
    # SIGTERM is handled by the frozen worker; resume only to deliver it if stopped.
    if was_stopped:
        os.kill(expected['pid'], signal.SIGCONT)
    deadline = time.monotonic()+120
    while not (RECOVERY/'exit.json').exists() and time.monotonic() < deadline:
        time.sleep(1)
    require((RECOVERY/'exit.json').exists(), 'No actual wrapper exit receipt; stop proof withheld')
    zero_outputs(original['jobs'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem+'-events.jsonl')
    with snapshot.open('xb') as stream:
        stream.write((ORIGINAL/'events.jsonl').read_bytes())
    proof = {'kind': 'observed-idle-evaluator-stop-v1', 'plan': ref(args.plan), 'createdAtUTC': now(),
             'originalIdentity': expected, 'observedExit': ref(RECOVERY/'exit.json'),
             'eventsSnapshot': ref(snapshot), 'noEvaluationOutputs': True, 'source': ref(__file__)}
    verify_stop_value(args.plan, plan, proof)
    write_new(args.output, proof)
    verify_stop(args.plan, plan, args.output)
    print(json.dumps({'stopped': ref(args.output), 'actualExitCode': 0}), flush=True)


def run(args):
    require(args.permit_evaluation, 'Root review and explicit --permit-evaluation required')
    plan, worker, original = verify_plan(args.plan)
    verify_stop(args.plan, plan, args.original_stop)
    require(sorted(os.sched_getaffinity(0)) == [18, 19] and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Original CPU-only resource policy required')
    require(all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'Two CPU threads required')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and os.environ.get('TMPDIR', '').startswith(str(N)+'/'), 'NAS-only runtime required')
    require(Path(os.environ['TMPDIR']).is_dir(), 'NAS temporary directory must exist')
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (ORIGINAL/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, plan, worker, original)


def run_locked(args, plan, worker, original):
    verify_original(worker, original)
    global_gate = worker.global_ready(original)
    require(global_gate is not None, 'All162 global freeze and original cold closure required')
    zero_outputs(original['jobs'])
    snapshots = {name: getattr(worker, name) for name in CALLABLES}
    frozen_root = worker.ROOT
    workflow = worker.workflow()
    frozen_evaluate = workflow.evaluate
    started = {'kind': 'nonstudent-evaluation-schedule-execution-v1', 'plan': ref(args.plan),
               'source': ref(__file__), 'originalStop': ref(args.original_stop), 'globalSelectionGate': global_gate['reference'],
               'identity': process_identity(os.getpid()), 'startedAtUTC': now(), 'sameOriginalFlockHeld': True,
               'unchangedOriginalPlanAndCallables': True, 'includedCellIds': plan['includedCellIds']}
    write_new(args.output_root/'started.json', started)
    stopping = False
    def interrupt(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    done = []
    jobs = [j for j in original['jobs'] if j['cellId'] in plan['includedCellIds']]
    start_time = time.monotonic()
    while len(done) < 243 and not stopping and time.monotonic()-start_time < 12*3600:
        worked = False
        require(not (args.output_root/'STOP').exists() and not worker.stop_requested(), 'Stop requested; preserve partial outputs')
        require(worker.ROOT == frozen_root and workflow.evaluate is frozen_evaluate
                and all(getattr(worker, k) is v for k, v in snapshots.items()), 'Frozen callable or ROOT replacement')
        current = worker.global_ready(original)
        require(current == global_gate, 'Global gate changed')
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
            require(not worker.completed(job, original, global_gate, panel), 'Unexpected concurrent completed cell')
            if not worker.resources()['passed']:
                break
            for reference in original['code'].values():
                worker.verify(reference)
            output = Path(job['fitDirectory'])/('evaluation-'+job['precision']+'.json')
            require(not output.exists(), 'Unpublished evaluation exists; preserve partial')
            worker.execute_cell(job, original, global_gate, panel, prerequisites)
            require(worker.completed(job, original, global_gate, panel), 'Original complete-cell proof missing')
            companion = output.with_name(output.stem+'-queue-execution.json')
            row = {'time': now(), 'cellId': job['cellId'], 'scheduler': ref(__file__), 'schedulePlan': ref(args.plan),
                   'originalQueueReceipt': ref(companion), 'globalSelectionGate': global_gate['reference'],
                   'directFrozenCallable': 'execute_cell', 'originalWorker': plan['originalWorker'], 'pid': os.getpid()}
            with (args.output_root/'cells.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            done.append(job['cellId'])
            worked = True
            print(json.dumps({'completed': len(done), 'total': 243, 'cellId': job['cellId']}), flush=True)
        if not worked and not stopping:
            time.sleep(30)
    require(len(done) == 243, 'Subset stopped before243; no full-completion claim')
    require(worker.ROOT == frozen_root and workflow.evaluate is frozen_evaluate
            and all(getattr(worker, k) is v for k, v in snapshots.items()), 'Frozen callable or ROOT changed')
    write_new(args.output_root/'completed.json', {'kind': 'nonstudent-evaluation-schedule-completed-v1', 'passed': True,
        'plan': ref(args.plan), 'started': ref(args.output_root/'started.json'), 'ledger': ref(args.output_root/'cells.jsonl'),
        'completedCellIds': [j['cellId'] for j in jobs], 'count': len(done), 'finishedAtUTC': now(),
        'studentCellsEvaluated': 0, 'full270Completed': False, 'originalResumeRequiresStudentColdAudit': True})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('register'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('stop-original'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--permit-stop', action='store_true')
    p = sub.add_parser('run'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--original-stop', type=Path, required=True); p.add_argument('--output-root', type=Path, required=True); p.add_argument('--permit-evaluation', action='store_true')
    args = parser.parse_args()
    {'register': register, 'stop-original': stop_original, 'run': run}[args.action](args)


if __name__ == '__main__':
    main()
