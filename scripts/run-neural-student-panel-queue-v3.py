#!/usr/bin/env python3
"""Preserve the student panel experiment while sharing verified numeric-audit identities."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
V2_PATH = REPO/'scripts/run-neural-student-panel-queue-v2.py'
V2_SHA = 'ae394b12d970dbc9691867f9b3ff3902a54d99ac2c011af8dacba9fbbfcc5a5f'
if hashlib.sha256(V2_PATH.read_bytes()).hexdigest() != V2_SHA:
    raise ValueError('Preserved student v2 source changed')
SPEC = importlib.util.spec_from_file_location('preserved_student_streaming_v2', V2_PATH)
V2 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(V2)
ROOT, H = V2.ROOT, V2.H
read, require, verified, identity = V2.read, V2.require, V2.verified, V2.identity
STOP = False


def adapter():
    from analysis import neural_student_inference_identity_cache
    return neural_student_inference_identity_cache


def numeric_command(plan, row):
    folder = Path(row['fitDirectory'])
    return [sys.executable, str(REPO/'scripts/audit-neural-generalization-numerics.py'),
        '--phase', 'inference', '--task', row['task']['path'], '--fit', str(folder),
        '--panel', plan['panel']['path'], '--precision', 'fp32',
        '--fit-audit', str(folder/'fit-numerical-audit.json'),
        '--output', str(folder/'inference-numerical-audit-fp32.json')]


def classified(plan, row):
    matches = [item for item in plan['numericAuditClassification'] if item['taskId'] == row['taskId']]
    require(len(matches) == 1, 'Missing/duplicate numeric execution classification')
    item = matches[0]; folder = Path(row['fitDirectory'])
    require(item['task'] == row['task'] and item['fitDirectory'] == str(folder)
            and item['numericAuditPath'] == str(folder/'inference-numerical-audit-fp32.json'),
            'Numeric execution classification targets another task')
    require(item['mode'] in ('preexisting-uncached', 'cached-required'), 'Unknown numeric execution mode')
    return item


def previous_exit(previous_plan, previous_exit_path, previous_output):
    base = read(verified(previous_plan))
    require(base == V2.contract(verified(base['resourceQualification'])), 'Previous v2 plan changed')
    output = Path(previous_output).resolve(); require(output.is_relative_to(ROOT), 'NAS previous output required')
    result = read(verified(previous_exit_path))
    process_ref = identity(output/'process-identity.json'); process = read(verified(process_ref))
    require(result['kind'] == 'streaming-student-recovery-observed-exit-v2' and result['exitCode'] == 0
            and result['all27Completed'] is False and result['previousWorkerExitStillUnknown'] is True
            and result['grant'] == identity(output/'grant.json'), 'Observed successful boundary stop required')
    require(process['plan'] == previous_plan and process['grant'] == result['grant']
            and process['bootId'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and not (Path('/proc')/str(process['pid'])).exists(), 'Previous worker is live or its identity differs')
    stop = read(output/'stop-request.json')
    require(stop['rootAuthorized'] is True and stop['signalSent'] is False
            and stop['workerIdentity'] == process_ref, 'Root graceful-stop receipt required')
    verified(stop['stopFile'])
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    require(events and events[-1]['stage'] == 'worker-stop'
            and events[-1]['reason'] == 'stop-or-12h-limit'
            and 0 < events[-1]['completed'] < 27, 'Previous queue did not stop at an incomplete-task boundary')
    require(not (output/'completed.json').exists(), 'A full completion cannot be an incomplete recovery')
    return base, process_ref, events[-1]['completed']


def snapshot_classification(base, previous_output, expected_completed):
    helpers = V2.V1.gate_helpers(); cache = {}; rows = []; completed = 0
    previous_output = Path(previous_output)
    for row in base['tasks']:
        folder = Path(row['fitDirectory']); numeric = folder/'inference-numerical-audit-fp32.json'
        companion = folder/'student-inference-cache-execution-fp32.json'
        output = folder/'inference'/base['panel']['sha256'][:16]/'fp32'
        require(not companion.exists(), 'Preexisting cache execution needs a different registered transition')
        completion = previous_output/(row['taskId'].replace('/', '__')+'-completed.json')
        item = {'taskId': row['taskId'], 'task': row['task'], 'fitDirectory': str(folder),
            'numericAuditPath': str(numeric), 'numericAudit': None, 'preexistingCompletion': None,
            'mode': 'cached-required', 'executionCompanion': str(companion)}
        if numeric.exists():
            own = V2.task_preflight(base, row, helpers, cache)
            require(own is not None, 'Preexisting inference has incomplete fit/selection prerequisites')
            checked = V2.inference_complete(base, row, own, helpers, cache)
            require(checked is not None and completion.is_file(), 'Previous full-panel post-gates are incomplete')
            old = read(completion)
            require(old['task'] == row['task'] and old['completion'] == checked
                    and read(verified(old['preflight']))['plan'] == identity(previous_output.parent/'plan.json'),
                    'Previous owner completion is unbound or differs')
            item.update(mode='preexisting-uncached', numericAudit=identity(numeric),
                        executionCompanion=None, preexistingCompletion=identity(completion))
            completed += 1
        else:
            require(not completion.exists() and not (output.exists() and any(output.iterdir()))
                    and not (folder/'reuse-inference-audit-fp32.json').exists()
                    and not (folder/'inference-reuse-receipt-fp32.json').exists(),
                    'Partial unreceipted inference must be preserved and investigated before registration')
        rows.append(item)
    require(len(rows) == 27 and completed == expected_completed, 'Previous completion count differs from exact inventory')
    return rows


def contract(resource_qualification, numeric_plan, numeric_qualification, previous_plan, previous_exit_ref,
             previous_output, classification_ref):
    cache = adapter(); cache.verify_plan(numeric_plan); cache.verify_qualification(numeric_qualification, numeric_plan)
    base, process, count = previous_exit(previous_plan, previous_exit_ref, previous_output)
    snapshot = read(verified(classification_ref))
    require(snapshot['kind'] == 'student-numeric-execution-classification-v3'
            and snapshot['previousV2Plan'] == previous_plan and snapshot['previousV2Exit'] == previous_exit_ref
            and snapshot['previousV2Output'] == str(Path(previous_output).resolve())
            and snapshot['preexistingCount'] == count and snapshot['partialUnreceiptedOutputsPresent'] is False,
            'Wrong numeric execution classification snapshot')
    require(base == V2.contract(verified(resource_qualification)), 'Resource qualification differs from preserved v2')
    value = {**base, 'kind': 'registered-streaming-student-panel-queue-v3', 'source': identity(__file__),
        'tests': identity(REPO/'analysis/tests/test_student_panel_streaming_v3.py'),
        'protocol': identity(REPO/'docs/research/neural-student-streaming-panel-v3-2026-09-23.md'),
        'preservedV2Source': identity(V2_PATH), 'previousV2Plan': previous_plan,
        'previousV2Exit': previous_exit_ref, 'previousV2Output': str(Path(previous_output).resolve()),
        'previousV2ProcessIdentity': process, 'classificationSnapshot': classification_ref,
        'numericAuditClassification': snapshot['numericAuditClassification'],
        'numericExecutionPlan': numeric_plan, 'numericExecutionQualification': numeric_qualification,
        'cacheAdapter': identity(REPO/'analysis/neural_student_inference_identity_cache.py'),
        'numericAuditDispatch': 'One registered parent-local StudentInferenceAuditCache; original inference numeric.main and exact argv.',
        'nonNumericCommandsUnchanged': True, 'newNumericGatesRequireCompanions': True,
        'numericOutcomeAndReplayFunctionsChanged': False,
        'hashOrNumericalFunctionReplacement': 'Only the separately registered identity-cache adapter bindings; no numerical replacement.',
        'stopFile': str(ROOT/'student-streaming-v3/STOP')}
    require([r['taskId'] for r in value['numericAuditClassification']] == [r['taskId'] for r in value['tasks']],
            'Classification does not preserve all27 task order')
    require(sum(r['mode'] == 'preexisting-uncached' for r in value['numericAuditClassification']) == count,
            'Classified preexisting audit count differs from the stopped queue')
    for row in value['tasks']:
        item = classified(value, row)
        if item['mode'] == 'preexisting-uncached':
            require(item['numericAudit'] is not None and item['preexistingCompletion'] is not None
                    and item['executionCompanion'] is None, 'Incomplete preexisting classification')
            verified(item['numericAudit']); verified(item['preexistingCompletion'])
        else:
            require(item['numericAudit'] is None and item['preexistingCompletion'] is None
                    and item['executionCompanion'] == str(Path(row['fitDirectory'])/'student-inference-cache-execution-fp32.json'),
                    'Incomplete companion-required classification')
    return value


def validate_plan(plan):
    expected = contract(plan['resourceQualification'], plan['numericExecutionPlan'], plan['numericExecutionQualification'],
        plan['previousV2Plan'], plan['previousV2Exit'], plan['previousV2Output'], plan['classificationSnapshot'])
    require(plan == expected, 'Registered v3 execution plan/source changed')
    for reference in plan['panelBindings'].values(): verified(reference)


def companion_gate(plan, row):
    item = classified(plan, row); output = Path(item['numericAuditPath'])
    if not output.exists():
        require(item['executionCompanion'] is None or not Path(item['executionCompanion']).exists(),
                'Companion exists without its numerical result')
        return None
    if item['mode'] == 'preexisting-uncached':
        require(identity(output) == item['numericAudit'], 'Preexisting uncached numerical result changed')
        verified(item['preexistingCompletion'])
        require(not (Path(row['fitDirectory'])/'student-inference-cache-execution-fp32.json').exists(),
                'Preexisting uncached audit unexpectedly acquired a companion')
        return None
    path = Path(item['executionCompanion'])
    require(path.is_file(), 'New numerical result has no execution companion; preserve partial output')
    adapter().verify_execution(path, plan_ref=plan['numericExecutionPlan'], task_ref=row['task'],
        panel_ref=plan['panel'], fit_audit_ref=identity(Path(row['fitDirectory'])/'fit-numerical-audit.json'),
        output_path=output, command=numeric_command(plan, row))
    return identity(path)


def inference_complete(plan, row, own, helpers, evidence_cache):
    companion_gate(plan, row)
    return V2.inference_complete(plan, row, own, helpers, evidence_cache)


def install_dispatch(plan, reuse, execution_cache):
    frozen_runner = reuse.run_commands
    rows = {r['taskId']: r for r in plan['tasks']}
    numeric_path = str(REPO/'scripts/audit-neural-generalization-numerics.py')
    def run_commands(commands, log_path, task_id):
        for command in commands:
            is_inference_numeric = (len(command) > 1 and command[1] == numeric_path and '--phase' in command
                and command[command.index('--phase')+1] == 'inference')
            if not is_inference_numeric:
                frozen_runner([command], log_path, task_id)
                continue
            require(task_id in rows, 'Unregistered student numeric task')
            row = rows[task_id]; item = classified(plan, row)
            require(item['mode'] == 'cached-required' and command == numeric_command(plan, row),
                    'Only the exact registered new student inference numeric command may use the cache')
            execution_cache.execute(command, plan_ref=plan['numericExecutionPlan'], task_ref=row['task'],
                panel_ref=plan['panel'], fit_audit_ref=identity(Path(row['fitDirectory'])/'fit-numerical-audit.json'),
                receipt_path=Path(item['executionCompanion']), log_path=log_path)
            companion_gate(plan, row)
    reuse.run_commands = run_commands
    return frozen_runner


def validate_grant(plan, grant, bindings):
    require(grant['kind'] == 'streaming-student-panel-execution-grant-v3'
            and grant['numericExecutionPlan'] == plan['numericExecutionPlan']
            and grant['classificationSnapshot'] == plan['classificationSnapshot']
            and grant['previousV2Exit'] == plan['previousV2Exit']
            and grant['newNumericGatesRequireCompanions'] is True, 'Wrong v3 numeric execution authorization')
    # Reuse the unchanged v2 slot validator through an explicit schema view only.
    slot_view = {**grant, 'kind': 'streaming-student-panel-execution-grant-v2'}
    return V2.validate_slot(plan, slot_view, read(verified(bindings['slotProof'])),
        read(verified(bindings['predecessorIdentity'])), bindings, mode='recovery')


def register(args):
    previous_plan, previous_exit_ref = identity(args.previous_v2_plan), identity(args.previous_v2_exit)
    base, _, count = previous_exit(previous_plan, previous_exit_ref, args.previous_v2_output)
    rows = snapshot_classification(base, args.previous_v2_output, count)
    snapshot_path = args.plan.parent/'preexisting-uncached-audits.json'
    H.write_new(snapshot_path, {'kind': 'student-numeric-execution-classification-v3',
        'previousV2Plan': previous_plan, 'previousV2Exit': previous_exit_ref,
        'previousV2Output': str(args.previous_v2_output.resolve()), 'preexistingCount': count,
        'numericAuditClassification': rows, 'partialUnreceiptedOutputsPresent': False})
    plan = contract(base['resourceQualification'], identity(args.numeric_plan), identity(args.numeric_qualification),
        previous_plan, previous_exit_ref, args.previous_v2_output, identity(snapshot_path))
    validate_plan(plan); H.write_new(args.plan, plan); print(json.dumps(identity(args.plan)), flush=True)


def run(args):
    global STOP
    for path, digest in ((args.plan,args.plan_sha256),(args.grant,args.grant_sha256)):
        require(identity(path)['sha256'] == digest, 'Execution input identity differs')
    plan = read(args.plan); validate_plan(plan); grant = read(args.grant)
    bindings = {key: grant[key] for key in ('plan','slotProof','predecessorIdentity')}
    require(bindings['plan'] == identity(args.plan), 'Grant names another plan')
    accepted = validate_grant(plan, grant, bindings)
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10, 'nice10 required'); os.sched_setaffinity(0,{2,3})
    folder = args.output.resolve(); require(folder.is_relative_to(ROOT), 'NAS output required')
    folder.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'student-streaming-v2/worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH'):
        path=folder/'runtime'/name.lower();path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PATH']=os.environ.get('PATH','')+':/usr/lib/wsl/lib'
    def interrupt(*_):
        global STOP; STOP=True
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    process=Path('/proc')/str(os.getpid());parent,ticks=H.stat_identity((process/'stat').read_text())
    H.write_new(folder/'process-identity.json',{'pid':os.getpid(),'parentPid':parent,'startTimeTicks':ticks,
        'bootId':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'plan':identity(args.plan),'grant':identity(args.grant)})
    H.write_new(folder/'slot-acceptance.json',{'kind':'streaming-student-slot-acceptance-v3','passed':True,
        'plan':identity(args.plan),'grant':identity(args.grant),'accepted':accepted,'processIdentity':identity(folder/'process-identity.json')})
    helpers=V2.V1.gate_helpers(); execution_cache=adapter().StudentInferenceAuditCache()
    frozen_runner=install_dispatch(plan,helpers[1],execution_cache)
    done={};before={};evidence_cache={};first_dispatch=False;started=time.monotonic()
    logs=folder/'task-logs';logs.mkdir(exist_ok=True);V2.log(folder,'worker-start',plan=identity(args.plan),pid=os.getpid())
    try:
        while not STOP and not Path(plan['stopFile']).exists() and time.monotonic()-started < plan['maximumHours']*3600:
            worked=False
            for row in plan['tasks']:
                if row['taskId'] in done:continue
                if STOP or Path(plan['stopFile']).exists():break
                if row['taskId']!=row['physicalOwnerTaskId'] and row['physicalOwnerTaskId'] not in done:continue
                dependencies=V2.dependency_rows(plan,row)
                if not all(V2.ready_documents(Path(r['fitDirectory']),r['physicalOwnerTaskId']!=r['taskId']) for r in dependencies):continue
                own=V2.task_preflight(plan,row,helpers,evidence_cache);require(own is not None,'Ready task became incomplete')
                existing=inference_complete(plan,row,own,helpers,evidence_cache)
                if existing:done[row['taskId']]=existing;before[row['taskId']]=own;continue
                state=H.resources()
                if not V2.resource_passes(state):V2.log(folder,'resource-wait',resources=state);break
                group=[own if r['taskId']==row['taskId'] else V2.task_preflight(plan,r,helpers,evidence_cache) for r in dependencies]
                require(all(g is not None for g in group),'Owner or descendant lost completed gates')
                validate_plan(plan);state=H.resources()
                if not V2.resource_passes(state):V2.log(folder,'resource-wait-after-preflight',resources=state);break
                if STOP or Path(plan['stopFile']).exists():break
                if not first_dispatch: first_validation=validate_grant(plan,grant,bindings)
                launch=folder/(row['taskId'].replace('/','__')+'-preflight.json')
                H.write_new(launch,{'kind':'streaming-student-owner-family-preflight-v3','passed':True,
                    'plan':identity(args.plan),'task':row['task'],'ownerTaskId':row['physicalOwnerTaskId'],
                    'dependencies':group,'inferenceOwnerCompletion':done.get(row['physicalOwnerTaskId']),
                    'resources':state,'numericClassification':classified(plan,row),'noMetricsOrSelectionExecuted':True})
                dispatch=argparse.Namespace(plan=Path(row['reusePlan']['path']) if row['reusePlan'] else None,
                    panel=Path(plan['panel']['path']),precision=['fp32'],device='cuda')
                V2.log(folder,'task-start',taskId=row['taskId'],preflight=identity(launch))
                if not first_dispatch:
                    H.write_new(folder/'first-dispatch.json',{'kind':'streaming-student-first-dispatch-v3','passed':True,
                        'plan':identity(args.plan),'grant':identity(args.grant),'slotValidation':first_validation,
                        'task':row['task'],'preflight':identity(launch),'dispatchStartedAtUTC':datetime.now(timezone.utc).isoformat()})
                    first_dispatch=True
                helpers[1].infer_task(dispatch,row,verified(row['task']),read(row['task']['path']),logs)
                require([V2.task_preflight(plan,r,helpers,evidence_cache) for r in dependencies]==group,'Fit/selection gates changed')
                complete=inference_complete(plan,row,own,helpers,evidence_cache);require(complete is not None,'Full42/all4 gates missing')
                done[row['taskId']]=complete;before[row['taskId']]=own;worked=True
                H.write_new(folder/(row['taskId'].replace('/','__')+'-completed.json'),{'task':row['task'],
                    'preflight':identity(launch),'completion':complete,'numericExecutionCompanion':companion_gate(plan,row)})
                V2.log(folder,'task-complete',taskId=row['taskId'],completed=len(done),total=27)
            if len(done)==27:break
            if not worked:time.sleep(plan['pollSeconds'])
        if len(done)==27:
            initial=[{k:v for k,v in before[r['taskId']].items() if k!='fitReuseAudit'} for r in plan['tasks']]
            checked=V2.V1.completion_gates(plan,initial)
            companions=[{'taskId':row['taskId'],'companion':companion_gate(plan,row)} for row in plan['tasks']
                        if classified(plan,row)['mode']=='cached-required']
            H.write_new(folder/'completed.json',{'kind':'streaming-student-full-panel-completed-v3','passed':True,
                'plan':identity(args.plan),'grant':identity(args.grant),'logicalTaskCount':27,'physicalEncoderCount':20,
                'completionGates':checked,'numericExecutionCompanions':companions,
                'classificationSnapshot':plan['classificationSnapshot'],'newNumericGatesRequireCompanions':True,
                'all27Full42All4NumericAndApplicableReuseGatesPassed':True,'evaluationMetricsProduced':False})
        V2.log(folder,'worker-stop',completed=len(done),total=27,reason='complete' if len(done)==27 else 'stop-or-12h-limit')
    finally:
        helpers[1].run_commands=frozen_runner


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','run'))
    p.add_argument('--plan',type=Path,required=True)
    for name in ('numeric-plan','numeric-qualification','previous-v2-plan','previous-v2-exit','previous-v2-output','grant','output'):
        p.add_argument('--'+name,type=Path)
    for name in ('plan','grant'):p.add_argument('--'+name+'-sha256')
    a=p.parse_args();require(Path('/mnt/freenas').is_mount() and a.plan.resolve().is_relative_to(ROOT),'Mounted NAS plan required')
    if a.action=='register':
        require(all(getattr(a,k) for k in ('numeric_plan','numeric_qualification','previous_v2_plan','previous_v2_exit','previous_v2_output')),
                'Qualified numeric plan and successful previous boundary stop required');register(a)
    else:
        require(all(getattr(a,k) for k in ('plan_sha256','grant','grant_sha256','output')),'Root plan/grant/output required');run(a)


if __name__=='__main__':main()
