#!/usr/bin/env python3
"""Disclosed original-selection hash reuse with mandatory independent cold closure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_inputs import verified
from analysis.neural_selection_serialization import load_module
from analysis import neural_selection_duration_view as duration
from analysis.neural_selection_identity_cache import BINDINGS, POLICY, IdentityCache, installed, signature

body = load_module('frozen_composite_selection_execution_body', REPO/'scripts/generalization-selection-audit-execution.py')
correction = load_module('frozen_selection_correction_companion', REPO/'scripts/audit-neural-generalization-selection-correction.py')
STAGES = ('select', 'ordinary', 'correction')
OUTPUTS = ('selection.json', 'selection-audit.json', 'selection-correction-audit.json')
JOINT_KIND = 'independent-generalization-result-identity-cache-audit-v1'


def now(): return datetime.now(timezone.utc).isoformat()


def verify_plan(path):
    plan = io.read(path)
    io.require(plan['kind'] == 'generalization-selection-identity-cache-amendment-v1' and plan['policy'] == POLICY,
               'Identity-cache policy differs')
    for name in ('durationPlan', 'correctionPlan', 'registration', 'historicalProtocol', 'protocol'): verified(plan[name])
    inherited = duration.verify_plan(plan['durationPlan']['path'])
    io.require(inherited['correctionPlan'] == plan['correctionPlan'] and inherited['historicalProtocol'] == plan['historicalProtocol'],
               'Cache amendment source association differs')
    for ref in plan['code'].values(): verified(ref)
    io.require(plan['code']['scripts/generalization-selection-cache-execution.py'] == io.identity(__file__), 'Cache workflow changed')
    registration = io.read(plan['registration']['path'])
    io.require([j['taskId'] for j in plan['tasks']] == registration['contract']['taskIds'] and len(plan['tasks']) == 18,
               'Original task population/order differs')
    for job in plan['tasks']:
        task = io.read(verified(job['task']))
        io.require(task['registration'] == plan['registration'] and task['variant'] == 'original-corpus'
            and [s['stage'] for s in job['stages']] == list(STAGES), 'Original task/stage registration differs')
        io.require([s['output'] for s in job['stages']] == [str(Path(job['fitDirectory'])/name) for name in OUTPUTS], 'Stage output paths differ')
    return plan


def register(args):
    inherited = duration.verify_plan(args.duration_plan)
    registration_ref = io.identity(args.registration); registration = io.read(args.registration)
    root = args.registration.parent; tasks = []
    for task_id in registration['contract']['taskIds']:
        task_path = root/'tasks'/(task_id.replace('/', '__')+'.json'); task = io.read(task_path)
        io.require(task['variant'] == 'original-corpus', 'Only original eighteen tasks use this cache')
        folder = root/'fits'/task['model']/f'seed-{task["seed"]}'
        stages = [{'stage': stage, 'output': str(folder/name),
            'preexistingOutput': io.identity(folder/name) if (folder/name).exists() else None}
            for stage, name in zip(STAGES, OUTPUTS, strict=True)]
        existing = [s['preexistingOutput'] is not None for s in stages]
        io.require(all(existing) or not any(existing), 'Partial original task must be resolved before cache registration')
        tasks.append({'task': io.identity(task_path), 'taskId': task_id, 'fitDirectory': str(folder), 'stages': stages})
    io.require(len(tasks) == 18 and sum(s['preexistingOutput'] is not None for j in tasks for s in j['stages']) == 3
        and all(s['preexistingOutput'] for s in tasks[0]['stages']), 'Expected only first original task to be preexisting')
    names = ('analysis/neural_selection_identity_cache.py', 'scripts/generalization-selection-cache-execution.py',
        'scripts/audit-neural-generalization-numerics.py', 'scripts/audit-neural-historical-refits.py',
        'scripts/audit-neural-selection-identity-cache.py', 'analysis/tests/test_selection_identity_cache.py',
        'analysis/tests/test_selection_identity_cache_audit.py')
    io.write_new(args.output, {'kind': 'generalization-selection-identity-cache-amendment-v1', 'policy': POLICY,
        'createdAtUTC': now(), 'durationPlan': io.identity(args.duration_plan), 'correctionPlan': inherited['correctionPlan'],
        'registration': registration_ref, 'historicalProtocol': inherited['historicalProtocol'], 'protocol': io.identity(args.protocol.resolve()),
        'tasks': tasks, 'qualification': {'task': tasks[0]['task'], 'referenceSelection': tasks[0]['stages'][0]['preexistingOutput'],
            'output': str(args.output.parent/'qualification-selection.json')},
        'code': {**inherited['code'], **{name: io.identity(REPO/name) for name in names}},
        'execution': {'cpuAffinity': [18, 19], 'minimumNice': 10, 'threads': 1, 'cudaVisibleDevices': '',
            'minimumLinuxAvailableBytes': 4*1024**3, 'minimumCFreeBytes': 20*1024**3, 'maxHours': 12},
        'requiresIndependentFinalColdAuditBeforeGlobalFreeze': True,
        'mandatoryPublicationWorkflow': 'scripts/generalization-selection-cache-execution.py'})


def resources():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    stat = os.statvfs('/mnt/c'); free = stat.f_bavail * stat.f_frsize
    return {'linuxAvailableBytes': available, 'windowsCFreeBytes': free,
            'passed': available >= 4*1024**3 and free >= 20*1024**3}


def emit(root, event, **values):
    row = {'time': now(), 'event': event, **values}
    with (root/'events.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


def fit_gate(job, plan):
    folder = Path(job['fitDirectory']); fit = io.identity(folder/'fit-result.json')
    ref = io.identity(folder/'fit-numerical-audit.json'); gate = io.read(ref['path'])
    io.require(gate['kind'] == 'independent-generalization-fit-numerical-audit-v1' and gate['passed'] is True
        and gate['task'] == job['task'] and gate['fitResult'] == fit
        and gate['auditor'] == plan['code']['scripts/audit-neural-generalization-numerics.py'], 'Original fit numerical gate differs')
    return fit, ref


def stage_inputs(job, stage, plan):
    folder = Path(job['fitDirectory']); fit, numeric = fit_gate(job, plan)
    refs = [job['task'], fit, numeric, plan['durationPlan'], plan['correctionPlan'], plan['historicalProtocol']]
    if stage in ('ordinary', 'correction'): refs.append(io.identity(folder/'selection.json'))
    if stage == 'correction': refs.append(io.identity(folder/'selection-audit.json'))
    return refs


def execute_stage(plan_path, plan, cache, job, stage, output, receipt_path):
    io.require(not Path(output).exists() and not Path(receipt_path).exists(), 'Unreceipted/duplicate stage output exists')
    started = now(); cache.begin()
    with installed(cache):
        inputs = [io.identity(plan_path), *stage_inputs(job, stage, plan)]
        for ref in inputs: verified(ref)
        folder = Path(job['fitDirectory'])
        if stage in ('select', 'qualification-select'):
            body.old.select(argparse.Namespace(plan=Path(plan['correctionPlan']['path']), task=Path(job['task']['path']),
                fit=folder, output=Path(output), historical=Path(plan['historicalProtocol']['path']).parent))
        elif stage == 'ordinary': duration.audit_selection(plan['durationPlan']['path'], folder/'selection.json', output)
        elif stage == 'correction': correction.audit(folder/'selection.json', folder/'selection-audit.json', output)
        else: raise ValueError('Unknown registered cache stage')
        result = io.identity(output)
    inventory = cache.finish()
    receipt = {'kind': 'explicit-selection-identity-cache-stage-v1', 'passed': True, 'plan': io.identity(plan_path),
        'task': job['task'], 'stage': stage, 'inputs': inputs, 'output': result, 'bindings': BINDINGS,
        **inventory, 'functionBindingsRestored': True, 'numericalFunctionsUnchanged': True,
        'startedAtUTC': started, 'finishedAtUTC': now()}
    io.write_new(receipt_path, receipt)
    return io.identity(receipt_path)


def check_saved_stage(reference, plan_ref, task, stage, output):
    receipt = io.read(verified(reference))
    io.require(receipt['kind'] == 'explicit-selection-identity-cache-stage-v1' and receipt['passed'] is True
        and receipt['plan'] == plan_ref and receipt['task'] == task and receipt['stage'] == stage
        and receipt['output'] == io.identity(output) and receipt['bindings'] == BINDINGS
        and receipt['functionBindingsRestored'] and receipt['numericalFunctionsUnchanged'], 'Saved cache-stage identity differs')
    for ref in receipt['files']:
        io.require(signature(ref['path']) == ref['stat'], 'Saved stage file metadata changed')
    return receipt


def run(args):
    plan = verify_plan(args.plan); root = args.plan.parent; plan_ref = io.identity(args.plan)
    io.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Nice10/hidden GPU required')
    os.sched_setaffinity(0, {18, 19})
    import fcntl
    lock = (root/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cache = IdentityCache(); start = time.monotonic()
    def wait_resource():
        while True:
            io.require(not (root/'STOP').exists() and time.monotonic()-start < 12*3600, 'Stop/12h cache-worker limit')
            values = resources()
            if values['passed']: return
            emit(root, 'resource-wait', resources=values); time.sleep(30)
    emit(root, 'worker-start', plan=plan_ref, pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)))
    qual_path = root/'qualification.json'; qual_execution = root/'qualification-execution.json'
    if not qual_path.exists():
        wait_resource()
        ref = execute_stage(args.plan, plan, cache, plan['tasks'][0], 'qualification-select',
                            plan['qualification']['output'], qual_execution)
        output = io.identity(plan['qualification']['output'])
        io.require(output['sha256'] == plan['qualification']['referenceSelection']['sha256'], 'Cached selection differs from uncached saved bytes')
        io.write_new(qual_path, {'kind': 'selection-identity-cache-qualification-v1', 'passed': True,
            'plan': plan_ref, 'referenceSelection': plan['qualification']['referenceSelection'], 'selection': output,
            'execution': ref, 'byteExactSelectionParity': True})
    qualification = io.read(qual_path)
    io.require(qualification['kind'] == 'selection-identity-cache-qualification-v1' and qualification['passed'] is True
        and qualification['plan'] == plan_ref and qualification['referenceSelection'] == plan['qualification']['referenceSelection']
        and qualification['selection'] == io.identity(plan['qualification']['output'])
        and qualification['selection']['sha256'] == qualification['referenceSelection']['sha256']
        and qualification['byteExactSelectionParity'] is True, 'Cached qualification differs')
    executions = [check_saved_stage(qualification['execution'], plan_ref, plan['tasks'][0]['task'],
                                  'qualification-select', plan['qualification']['output'])]
    emit(root, 'qualification-passed', qualification=io.identity(qual_path))
    stages = []
    for job in plan['tasks']:
        for stage in job['stages']:
            if stage['preexistingOutput']:
                io.require(io.identity(stage['output']) == stage['preexistingOutput'], 'Preexisting selection output changed')
                stages.append({'task': job['task'], 'stage': stage['stage'], 'preexistingOutput': stage['preexistingOutput'],
                    'execution': None, 'output': stage['preexistingOutput']})
                continue
            receipt_path = root/'stages'/(job['taskId'].replace('/', '__')+'__'+stage['stage']+'.json')
            if receipt_path.exists(): ref = io.identity(receipt_path)
            else:
                wait_resource(); emit(root, 'stage-start', taskId=job['taskId'], stage=stage['stage'])
                ref = execute_stage(args.plan, plan, cache, job, stage['stage'], stage['output'], receipt_path)
            receipt = check_saved_stage(ref, plan_ref, job['task'], stage['stage'], stage['output']); executions.append(receipt)
            stages.append({'task': job['task'], 'stage': stage['stage'], 'preexistingOutput': None,
                'execution': ref, 'output': receipt['output']})
            emit(root, 'stage-complete', taskId=job['taskId'], stage=stage['stage'], execution=ref, statistics=receipt['statistics'])
    files = {}
    for receipt in executions:
        for entry in receipt['files']:
            io.require(entry['path'] not in files or files[entry['path']] == entry, 'Cross-stage file identity/stat changed')
            files[entry['path']] = entry
    io.write_new(args.output, {'kind': 'selection-identity-cache-run-inventory-v1', 'plan': plan_ref,
        'qualification': io.identity(qual_path), 'stages': stages, 'files': [files[k] for k in sorted(files)], 'source': io.identity(__file__)})
    emit(root, 'worker-complete', inventory=io.identity(args.output), tasks=18, stages=54)


def cold_gate(path):
    gate = io.read(path)
    io.require(gate['kind'] == 'independent-selection-identity-cache-cold-audit-v1' and gate['passed'] is True
        and gate['allColdHashesPassed'] is True and gate['taskCount'] == 18 and gate['stageCount'] == 54
        and gate['cachedStageCount'] == 51 and gate['preexistingStageCount'] == 3
        and gate['closurePolicy'] == POLICY['coldClosurePolicy'] and gate['allImmutableStatsPassed'] is True
        and gate['completeStageAndDeclaredEvidenceClosureVerified'] is True and gate['byteExactQualificationParity'] is True
        and gate['auditor'] == io.identity(REPO/'scripts/audit-neural-selection-identity-cache.py'), 'Complete independent cold cache gate required')
    verify_plan(verified(gate['plan']))
    cold_files = gate['coldFiles']; known = {r['path']: r for r in cold_files}
    io.require(len(known) == len(cold_files) and cold_files == sorted(cold_files, key=lambda r: r['path']), 'Cold closure inventory differs')
    for reference in gate['evidence']:
        path = str(Path(reference['path']).resolve())
        io.require(path in known and known[path]['sha256'] == reference['sha256'], 'Cold closure omits an evidence identity')
    for reference in gate['files']:
        io.require(known.get(reference['path']) == reference, 'Cold closure/cache inventory differs')
    for reference in cold_files:
        io.require(signature(reference['path']) == reference['stat'], 'Cold-audited file metadata changed')
    return gate


def global_cache_gate(path):
    frozen = body.global_gate(path)
    side = Path(path).with_name(Path(path).stem+'-identity-cache-gate.json'); gate = io.read(side)
    io.require(gate['kind'] == 'global-selection-freeze-identity-cache-gate-v1' and gate['passed'] is True
        and gate['globalSelectionGate'] == io.identity(path) and gate['workflow'] == io.identity(__file__), 'Global freeze lacks required cache publication gate')
    cold = cold_gate(verified(gate['coldAudit'])); plan = verify_plan(verified(cold['plan']))
    io.require(frozen['plan'] == plan['durationPlan'] and gate['plan'] == cold['plan'], 'Global/cache amendment association differs')
    return frozen, io.identity(side)


def freeze_selections(args):
    gate = cold_gate(args.cold_audit); plan = verify_plan(verified(gate['plan']))
    io.require(io.identity(args.plan) == plan['durationPlan'], 'Composite duration plan differs')
    body.freeze_selections(args)
    io.require(cold_gate(args.cold_audit) == gate, 'Cold closure changed during global selection freeze')
    side = args.output.with_name(args.output.stem+'-identity-cache-gate.json')
    io.write_new(side, {'kind': 'global-selection-freeze-identity-cache-gate-v1', 'passed': True,
        'globalSelectionGate': io.identity(args.output), 'coldAudit': io.identity(args.cold_audit),
        'plan': gate['plan'], 'workflow': io.identity(__file__)})


def evaluate(args):
    _, before = global_cache_gate(args.global_selection_gate)
    body.evaluate(args)
    io.require(global_cache_gate(args.global_selection_gate)[1] == before, 'Cache gate changed during evaluation')


def audit_results(args):
    _, cache_ref = global_cache_gate(args.global_selection_gate)
    base = args.output.with_name(args.output.stem+'-execution-base.json')
    body.audit_results(argparse.Namespace(**{**vars(args), 'output': base}))
    io.require(global_cache_gate(args.global_selection_gate)[1] == cache_ref, 'Cache publication gate changed during full audit')
    result = io.read(base)
    io.write_new(args.output, {**result, 'kind': JOINT_KIND, 'cacheExecutionAuditor': io.identity(__file__),
        'delegatedExecutionAudit': io.identity(base), 'globalIdentityCacheGate': cache_ref})


def report(args):
    from analysis.neural_generalization_report import build_report
    audit = io.read(args.audit)
    io.require(audit['kind'] == JOINT_KIND and audit['passed'] is True and audit['cacheExecutionAuditor'] == io.identity(__file__)
        and audit['resultIndex'] == io.identity(args.index), 'Report requires the independent cold-cache publication audit')
    body.old.verify_closure(audit)
    base = io.read(verified(audit['delegatedExecutionAudit']))
    stripped = {k: v for k, v in audit.items() if k not in ('cacheExecutionAuditor', 'delegatedExecutionAudit', 'globalIdentityCacheGate')}
    stripped['kind'] = body.JOINT_KIND
    io.require(io.canonical(stripped) == io.canonical(base), 'Cache wrapper changed delegated numerical audit')
    io.require(base['executionAuditor'] == io.identity(REPO/'scripts/generalization-selection-audit-execution.py')
        and global_cache_gate(verified(base['globalSelectionGate']))[1] == audit['globalIdentityCacheGate']
        and body.publication_preflight(args.index, base['globalSelectionGate']['path']) == base['evaluationPublicationGates'],
        'Final global/execution/cache publication gates differ')
    return build_report(args.index, args.output, args.audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for key in ('duration-plan', 'registration', 'protocol', 'output'): reg.add_argument('--'+key, type=Path, required=True)
    runner = sub.add_parser('run')
    for key in ('plan', 'output'): runner.add_argument('--'+key, type=Path, required=True)
    freeze = sub.add_parser('freeze-selections')
    for key in ('plan', 'cold-audit', 'output'): freeze.add_argument('--'+key, type=Path, required=True)
    freeze.add_argument('--registration', type=Path, action='append', required=True)
    ev = sub.add_parser('evaluate')
    for key in ('task', 'fit', 'selection', 'selection-audit', 'correction-audit', 'panel', 'inventory', 'original-manifest', 'global-selection-gate', 'output'):
        ev.add_argument('--'+key, type=Path, required=True)
    ev.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    aud = sub.add_parser('audit-results')
    for key in ('index', 'report', 'global-selection-gate', 'output'): aud.add_argument('--'+key, type=Path, required=True)
    aud.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    rep = sub.add_parser('report')
    for key in ('index', 'audit', 'output'): rep.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(); io.require(Path('/mnt/freenas').is_mount()
        and args.output.resolve().is_relative_to('/mnt/freenas'), 'Mounted direct-NAS output required')
    globals()[args.action.replace('-', '_')](args)
    print(json.dumps({'action': args.action, 'output': io.identity(args.output)}), flush=True)


if __name__ == '__main__': main()
