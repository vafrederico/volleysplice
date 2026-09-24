#!/usr/bin/env python3
"""Stream blind student inference only after each complete training-owner family freezes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0061'))
GIB = 1024**3
SPEC = importlib.util.spec_from_file_location('frozen_student_panel_v1', REPO/'scripts/run-neural-student-panel-queue.py')
if hashlib.sha256((REPO/'scripts/run-neural-student-panel-queue.py').read_bytes()).hexdigest() != '15c2bd6673776e5ccd94993f0e22040dc7d6fab59c7e7870b2f25d0a3b3b472e':
    raise ValueError('Frozen serial student queue changed')
V1 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(V1)
H = V1.HELPERS
read, require, verified = V1.read, V1.require, V1.verified
STOP = False
SLOTS = {'root-nonstudent-fit': 97282, 'root-raw-inference': 107405}


def identity(path): return H.identity(path)


def resource_passes(state):
    return (state['linuxAvailableBytes'] >= 10*GIB and state['windowsPhysicalFreeBytes'] >= 3*GIB
            and state['windowsCFreeBytes'] >= 20*GIB and state['gpuFreeMiB'] >= 5*1024)


def qualification(reference):
    report = read(verified(reference))
    require(report['kind'] == 'student-batch64-streaming-resource-qualification-result-v1'
            and report['passed'] is True and report['ownerCacheUnchanged'] is True
            and report['teacherTargetsLoaded'] is False and report['scoresOrMetricsProduced'] is False,
            'Passing non-mutating extraction qualification required')
    require([r['array'] for r in report['arrayChecks']] == ['timestamps', 'tokens', 'quality', 'selected_presentation_times']
            and all(r['bitExact'] is True for r in report['arrayChecks']), 'Extractor qualification array parity differs')
    for key in ('plan', 'startup', 'outputReceipt', 'rssSamples'): verified(report[key])
    registered = read(report['plan']['path'])
    require(registered['source'] == identity(REPO/'scripts/qualify-student-streaming-resources.py'), 'Qualification source changed')
    for reference in [registered['resourceHelpers'], *registered['numericSources']]: verified(reference)
    return report


def task_inventory():
    rows = V1.task_inventory(); refs = {r['taskId']: r['task'] for r in rows}
    for row in rows:
        row['physicalOwnerTask'] = refs[row['physicalOwnerTaskId']]
        row['reusePlan'] = None if row['taskId'].startswith('original-corpus/') else identity(
            ROOT/('export-proxy-v1' if row['taskId'].startswith('export-rally-') else 'randomized-variants-v1')/'reuse-plan.json')
    return rows


def contract(qualification_path):
    base = V1.contract(); reference = identity(qualification_path); proof = qualification(reference)
    return {**{k: v for k, v in base.items() if k not in ('commands', 'preflightRequiresAll27FitOrdinaryCorrectionExecutionGates')},
        'kind': 'registered-streaming-student-panel-queue-v2', 'source': identity(__file__),
        'preservedV1Source': identity(REPO/'scripts/run-neural-student-panel-queue.py'),
        'tests': identity(REPO/'analysis/tests/test_student_panel_streaming_v2.py'),
        'protocol': identity(REPO/'docs/research/neural-student-streaming-panel-v2-2026-09-23.md'),
        'resourceQualification': reference, 'resourceQualificationPlan': proof['plan'],
        'resourceQualificationSource': identity(REPO/'scripts/qualify-student-streaming-resources.py'),
        'quiescentSlotHelper': identity(REPO/'scripts/manage-neural-raw-quiescent-slot.py'),
        'qualificationScope': 'Batch64 extraction only; does not bound full temporal inference or audit RAM.',
        'tasks': task_inventory(), 'cpuAffinity': [2, 3], 'threads': 2, 'nice': 10,
        'maxLongLivedGpuWorkers': 3, 'maxImageWorkers': 2, 'imageEncoderConcurrency': 1,
        'preflightRequiresOwnerAndAllFitReuseDescendants': True,
        'dispatchCallable': 'scripts/run-neural-generalization-reuse-queue.py:infer_task',
        'hashOrNumericalFunctionReplacement': False, 'startupMinimumAvailableRamBytes': 10*GIB,
        'evidenceVerificationCachePolicy': 'One worker-local dictionary passed unchanged to frozen correction_gate/verify_closure; their existing expected-SHA/stat validation applies. Final v1 full completion starts fresh evidence dictionaries.',
        'startupMinimumGpuFreeMiB': 5*1024, 'startupMinimumWindowsPhysicalFreeBytes': 3*GIB,
        'minimumWindowsCFreeBytes': 20*GIB, 'allowedPredecessorSlots': SLOTS,
        'slotAuthorizationModes': ['exited', 'quiescent', 'recovery'],
        'recoveryNeverClaimsPriorSuccessfulExit': True,
        'quiescentRecheckBeforeFirstDispatch': True,
        'pollSeconds': 30, 'maximumHours': 12, 'stopFile': str(ROOT/'student-streaming-v2/STOP')}


def validate_plan(plan):
    require(plan == contract(verified(plan['resourceQualification'])), 'Streaming plan/source population changed')
    for reference in plan['panelBindings'].values(): verified(reference)
    correction, reuse, _ = V1.gate_helpers()
    inspect.signature(reuse.infer_task).bind(None, None, None, None, None)
    return correction, reuse, V1.gate_helpers()[2]


def pause_helper(plan):
    spec=importlib.util.spec_from_file_location('explicit_raw_quiescent_slot_helper',verified(plan['quiescentSlotHelper']))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def dependency_rows(plan, row):
    """An owner cannot append caches until every fit-copy descendant is immutable."""
    if row['physicalOwnerTaskId'] != row['taskId']: return [row]
    result = [r for r in plan['tasks'] if r['physicalOwnerTaskId'] == row['taskId']]
    require(result and result[0]['taskId'] == row['taskId'], 'Canonical owner must precede its registered descendants')
    return result


def ready_documents(folder, copied):
    names = ['fit-result.json', 'fit-numerical-audit.json', 'selection.json', 'selection-audit.json', 'selection-correction-audit.json']
    if copied: names.append('reuse-audit.json')
    for name in names:
        path = folder/name
        if not path.exists(): return False
        try: read(path)
        except json.JSONDecodeError: return False  # immutable publication still being written
    return True


def task_preflight(plan, row, helpers, cache=None):
    if cache is None: cache={}
    correction, reuse, verify_execution = helpers
    folder = Path(row['fitDirectory']); copied = row['physicalOwnerTaskId'] != row['taskId']
    if not ready_documents(folder, copied): return None
    task_path = verified(row['task']); task = read(task_path)
    require(task['taskId'] == row['taskId'] and task['model'] == 'distilled-mobile-tcn', 'Wrong student task ownership')
    numeric_path = folder/'fit-numerical-audit.json'
    numeric = reuse.check_gate(numeric_path, task_path, folder, 'independent-generalization-fit-numerical-audit-v1')
    require(numeric['taskId'] == row['taskId'] and numeric['checkpointEpochs'] == [5,15,30,60]
            and numeric['completed'] == read(folder/'fit-result.json')['temporal'] and numeric['student'] is not None,
            'Wrong student numerical scope')
    correction.verify_closure(numeric, cache)
    selection, ordinary, companion = [folder/name for name in ('selection.json','selection-audit.json','selection-correction-audit.json')]
    correction.correction_gate(selection, ordinary, companion, cache)
    require(read(selection)['task'] == row['task'], 'Selection task differs')
    execution = verify_execution(ordinary)
    copied_ref = None
    if copied:
        copied_ref = identity(folder/'reuse-audit.json')
        gate = reuse.check_gate(folder/'reuse-audit.json', task_path, folder, 'independent-generalization-fit-reuse-audit-v1')
        require(gate['plan'] == row['reusePlan'] and gate['sourceTask'] == row['physicalOwnerTask']
                and gate['targetNumericalAudit'] == identity(numeric_path), 'Wrong fit-copy owner/plan/numerical gate')
        correction.verify_closure(gate, cache)
    return {'task': row['task'], 'fitNumericalAudit': identity(numeric_path), 'selection': identity(selection),
        'ordinaryAudit': identity(ordinary), 'correctionAudit': identity(companion), 'execution': execution,
        'fitReuseAudit': copied_ref}


def inference_complete(plan, row, preflight, helpers, cache=None):
    if cache is None: cache={}
    correction, reuse, _ = helpers; folder = Path(row['fitDirectory']); numeric_path = folder/'inference-numerical-audit-fp32.json'
    copied = row['physicalOwnerTaskId'] != row['taskId']
    if not numeric_path.exists() or copied and not (folder/'reuse-inference-audit-fp32.json').exists(): return None
    numeric = reuse.check_gate(numeric_path, verified(row['task']), folder, 'independent-generalization-inference-numerical-audit-v1')
    require(numeric['taskId'] == row['taskId'] and numeric['panel'] == plan['panel'] and numeric['precision'] == 'fp32'
            and numeric['checkpointEpochs'] == [5,15,30,60] and numeric['fitAudit'] == preflight['fitNumericalAudit']
            and numeric['completed'] == read(folder/'fit-result.json')['temporal'], 'Wrong student inference numerical scope')
    output = folder/'inference'/plan['panel']['sha256'][:16]/'fp32'
    require(numeric['inferenceReceipts'] == [identity(output/(key+'.json')) for key in plan['recordingIds']]
            and {p.stem for p in output.glob('*.json')} == set(plan['recordingIds'])
            and {p.stem for p in output.glob('*.npz')} == set(plan['recordingIds']), 'Incomplete full42 raw inference inventory')
    require([(r['recordingId'],r['epoch']) for r in numeric['cpuReplay']]
            == [(key,epoch) for key in plan['recordingIds'] for epoch in (5,15,30,60)]
            and [r['id'] for r in numeric['studentFeatureReplay']] == plan['recordingIds'], 'Incomplete independent replay scope')
    correction.verify_closure(numeric, cache)
    copied_ref = None
    if copied:
        path = folder/'reuse-inference-audit-fp32.json'; gate = reuse.check_gate(path, verified(row['task']), folder, 'independent-generalization-inference-reuse-audit-v1')
        require(gate['plan'] == row['reusePlan'] and gate['sourceTask'] == row['physicalOwnerTask']
                and gate['panel'] == plan['panel'] and gate['precision'] == 'fp32'
                and gate['targetNumericalAudit'] == identity(numeric_path) and gate['recordingIds'] == plan['recordingIds']
                and gate['fitReuseAudit'] == preflight['fitReuseAudit']
                and gate['checkpointEpochs'] == [5,15,30,60] and gate['allRawScoreBytesExactlyEqual'] is True
                and gate['studentFeatureArrayEqualityCount'] == 42, 'Incomplete/mismatched raw inference reuse proof')
        correction.verify_closure(gate, cache); copied_ref = identity(path)
    return {'task':row['task'],'numericalAudit':identity(numeric_path),'reuseAudit':copied_ref}


def validate_slot(plan, grant, slot_receipt, predecessor, bindings, mode='exited', exists=lambda pid: (Path('/proc')/str(pid)).exists(), boot_id=None, resume_proof=None):
    boot_id = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(mode in ('exited','quiescent','recovery'),'Unknown slot authorization')
    if mode != 'recovery':
        slot = slot_receipt['slot'] if mode == 'exited' else 'root-raw-inference'; pid = predecessor['pid']
        require(slot in SLOTS and pid == SLOTS[slot] and predecessor['bootId'] == boot_id, 'Wrong predecessor identity/boot')
    pids = slot_receipt['requiredExitedPids'] if mode == 'exited' else []
    if mode == 'exited':
        require(pids and len(pids) == len(set(pids)) and pid in pids and all(type(p) is int and p > 0 for p in pids)
                and slot_receipt['released'] is True and slot_receipt['allDescendantsExited'] is True
                and slot_receipt['exitCode'] == 0 and slot_receipt['processIdentity'] == bindings['predecessorIdentity']
                and all(not exists(p) for p in pids), 'Predecessor slot or descendants not released successfully')
        validation={'released':True,'requiredExitedPids':pids}
    elif mode == 'quiescent':
        validation = pause_helper(plan).validate_pause_proof(verified(bindings['slotProof']))
        require(validation['proof'] == bindings['slotProof'] and slot_receipt['predecessorExited'] is False, 'Quiescent authorization differs')
        # The helper independently checks same boot/start/command, live T state,
        # complete first stages, absent children and the unlaunched next stage.
        require(all(slot_receipt['processIdentity'][key] == predecessor[key] for key in ('pid','startTimeTicks','bootId','command')),
                'Paused predecessor identity differs')
        require(resume_proof is not None and Path(resume_proof).resolve().is_relative_to(ROOT), 'NAS resume lifecycle proof path required')
    else:
        require(slot_receipt['kind'] == 'generalization-interruption-recovery-observation-v1'
                and slot_receipt['bootId'] == boot_id and slot_receipt['priorObservedExitCode'] is None
                and slot_receipt['allListedOwnedProcessesAbsent'] is True
                and slot_receipt['newEvaluationOutcomesOpened'] is False
                and slot_receipt['frozenNumericalArtifactsModified'] is False
                and slot_receipt['knownProcessIdentityRecord'] == bindings['predecessorIdentity'],
                'Recovery must bind the explicit interruption observation without a successful-exit claim')
        require(predecessor['kind'] == 'precision-raw-prerequisite-process-identities-v1',
                'Recovery predecessor inventory has the wrong schema')
        prior = slot_receipt['ownedProcessObservations']
        absent = [item['pid'] for item in prior]
        require(len(absent) == len(set(absent)) and all(type(pid) is int and pid > 0 for pid in absent)
                and set(SLOTS.values()) | {104269, 107249} <= set(absent)
                and all(item['exists'] is False and not exists(item['pid']) for item in prior),
                'An interrupted predecessor remains live, was reused, or is absent from the observation')
        require(set(SLOTS.values()) <= {item['pid'] for item in predecessor['processes']}
                and all(type(item['startTimeTicks']) is int and item['startTimeTicks'] > 0
                        and item['bootId'] == predecessor['bootId'] and item['command']
                        for item in predecessor['processes']), 'Invalid interrupted predecessor identity collection')
        for key in ('priorProgressSnapshot', 'knownProcessIdentityRecord'):
            verified(slot_receipt[key])
        validation = {'priorObservedExitCode': None, 'requiredAbsentPriorPids': absent,
                      'interruptionObservation': bindings['slotProof'], 'freshAbsenceRechecked': True}
    expected = {'kind':'streaming-student-panel-execution-grant-v2','source':plan['source'],**bindings,
        'resourceQualification':plan['resourceQualification'],'slotMode':mode,'requiredExitedPids':pids,'cpuAffinity':[2,3],
        'threads':2,'nice':10,'maxLongLivedGpuWorkers':3,'maxImageWorkers':2,'imageEncoderConcurrency':1,
        'numericalRecipesChanged':False,'evaluationMetricsAllowed':False,'activeWorkerInventoryComplete':True}
    if mode == 'quiescent': expected['resumeProofPath']=str(Path(resume_proof).resolve())
    if mode == 'recovery':
        expected.update({'priorObservedExitCode': None, 'requiredAbsentPriorPids': validation['requiredAbsentPriorPids'],
                         'priorSuccessfulExitClaimed': False})
    H.validate_grant(grant,expected)
    active = grant['activeWorkerInventory']
    require(len(active) <= 2 and sum(r['usesImageEncoder'] is True for r in active) <= 1,
            'Grant would exceed three GPU or two image workers')
    require(len({r['processIdentity']['path'] for r in active}) == len(active), 'Duplicate active slot inventory')
    for row in active:
        state = read(verified(row['processIdentity']))
        require(state['bootId'] == boot_id and type(row['usesImageEncoder']) is bool, 'Wrong active-worker identity')
        if exists(state['pid']):
            _, ticks = H.stat_identity((Path('/proc')/str(state['pid'])/'stat').read_text())
            require(ticks == state['startTimeTicks'], 'Active worker PID was reused')
    return {'slotMode':mode,'slotProof':bindings['slotProof'],'validation':validation}


def log(folder, stage, **values):
    event = {'time':datetime.now(timezone.utc).isoformat(),'stage':stage,**values}
    with (folder/'events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)


def run(args):
    global STOP
    for path,digest in ((args.plan,args.plan_sha256),(args.grant,args.grant_sha256),
                        (args.slot_proof,args.slot_proof_sha256),(args.predecessor_identity,args.predecessor_identity_sha256)):
        require(identity(path)['sha256'] == digest, 'Execution input identity differs')
    plan=read(args.plan);helpers=validate_plan(plan)
    bindings={'plan':identity(args.plan),'slotProof':identity(args.slot_proof),'predecessorIdentity':identity(args.predecessor_identity)}
    accepted=validate_slot(plan,read(args.grant),read(args.slot_proof),read(args.predecessor_identity),bindings,args.slot_mode,resume_proof=args.resume_proof)
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    os.sched_setaffinity(0,{2,3});folder=args.output.resolve();require(folder.is_relative_to(ROOT),'NAS output required')
    folder.mkdir(parents=True,exist_ok=True);lock=(ROOT/'student-streaming-v2/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH'):
        path=folder/'runtime'/name.lower();path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PATH']=os.environ.get('PATH','')+':/usr/lib/wsl/lib'
    def interrupt(*_):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    process=Path('/proc')/str(os.getpid());parent,ticks=H.stat_identity((process/'stat').read_text())
    H.write_new(folder/'process-identity.json',{'pid':os.getpid(),'parentPid':parent,'startTimeTicks':ticks,
        'bootId':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'plan':identity(args.plan),'grant':identity(args.grant)})
    H.write_new(folder/'slot-acceptance.json',{'kind':'streaming-student-slot-acceptance-v2','passed':True,
        'plan':identity(args.plan),'grant':identity(args.grant),'accepted':accepted,'processIdentity':identity(folder/'process-identity.json')})
    done={};before={};evidence_cache={};first_dispatch=False;started=time.monotonic();logs=folder/'task-logs';logs.mkdir(exist_ok=True)
    log(folder,'worker-start',plan=identity(args.plan),grant=identity(args.grant),pid=os.getpid())
    while not STOP and not Path(plan['stopFile']).exists() and time.monotonic()-started < plan['maximumHours']*3600:
        worked=False
        for row in plan['tasks']:
            if row['taskId'] in done:continue
            if STOP or Path(plan['stopFile']).exists():break
            if row['taskId']!=row['physicalOwnerTaskId'] and row['physicalOwnerTaskId'] not in done:continue
            dependencies=dependency_rows(plan,row)
            if not all(ready_documents(Path(r['fitDirectory']),r['physicalOwnerTaskId']!=r['taskId']) for r in dependencies):continue
            own=task_preflight(plan,row,helpers,evidence_cache)
            require(own is not None,'Ready task became incomplete')
            existing=inference_complete(plan,row,own,helpers,evidence_cache)
            if existing:done[row['taskId']]=existing;before[row['taskId']]=own;continue
            state=H.resources()
            if not resource_passes(state):log(folder,'resource-wait',resources=state);break
            group=[own if r['taskId']==row['taskId'] else task_preflight(plan,r,helpers,evidence_cache) for r in dependencies]
            require(all(g is not None for g in group),'Owner or descendant lost its completed gates')
            validate_plan(plan)
            state=H.resources()
            if not resource_passes(state):log(folder,'resource-wait-after-preflight',resources=state);break
            if not first_dispatch:
                first_validation=validate_slot(plan,read(args.grant),read(args.slot_proof),read(args.predecessor_identity),bindings,args.slot_mode,resume_proof=args.resume_proof)
            launch=folder/(row['taskId'].replace('/','__')+'-preflight.json')
            H.write_new(launch,{'kind':'streaming-student-owner-family-preflight-v2','passed':True,'plan':identity(args.plan),
                'task':row['task'],'ownerTaskId':row['physicalOwnerTaskId'],'dependencies':group,
                'inferenceOwnerCompletion':done.get(row['physicalOwnerTaskId']),'resources':state,
                'callable':plan['dispatchCallable'],'noMetricsOrSelectionExecuted':True})
            dispatch=argparse.Namespace(plan=Path(row['reusePlan']['path']) if row['reusePlan'] else None,
                panel=Path(plan['panel']['path']),precision=['fp32'],device='cuda')
            log(folder,'task-start',taskId=row['taskId'],preflight=identity(launch))
            if not first_dispatch:
                H.write_new(folder/'first-dispatch.json',{'kind':'streaming-student-first-dispatch-v2','passed':True,
                    'plan':identity(args.plan),'grant':identity(args.grant),'slotMode':args.slot_mode,'slotProof':identity(args.slot_proof),
                    'task':row['task'],'preflight':identity(launch),'callable':plan['dispatchCallable'],
                    'sameSlotRecheckedImmediatelyBeforeDispatch':True,'slotValidation':first_validation,
                    'dispatchStartedAtUTC':datetime.now(timezone.utc).isoformat()})
                first_dispatch=True
            helpers[1].infer_task(dispatch,row,verified(row['task']),read(row['task']['path']),logs)
            require([task_preflight(plan,r,helpers,evidence_cache) for r in dependencies]==group,'Fit/selection gates changed during blind inference')
            complete=inference_complete(plan,row,own,helpers,evidence_cache);require(complete is not None,'Full42/all4 inference gates missing')
            done[row['taskId']]=complete;before[row['taskId']]=own;worked=True
            H.write_new(folder/(row['taskId'].replace('/','__')+'-completed.json'),{'task':row['task'],'preflight':identity(launch),'completion':complete})
            log(folder,'task-complete',taskId=row['taskId'],completed=len(done),total=27)
        if len(done)==27:break
        if not worked:time.sleep(plan['pollSeconds'])
    if len(done)==27:
        initial=[{k:v for k,v in before[r['taskId']].items() if k!='fitReuseAudit'} for r in plan['tasks']]
        checked=V1.completion_gates(plan,initial)
        lifecycle={'mode':args.slot_mode,'slotAcceptance':identity(folder/'slot-acceptance.json')}
        if args.slot_mode=='quiescent':
            while not args.resume_proof.exists() and not STOP and not Path(plan['stopFile']).exists() and time.monotonic()-started<plan['maximumHours']*3600:
                log(folder,'awaiting-root-resume-lifecycle',resumeProofPath=str(args.resume_proof));time.sleep(plan['pollSeconds'])
            require(args.resume_proof.exists(),'Full inference is complete but root resume lifecycle remains pending')
            lifecycle['resumeValidation']=pause_helper(plan).validate_resume_proof(args.resume_proof,expected_pause=bindings['slotProof'])
            lifecycle['resumeProof']=identity(args.resume_proof)
        H.write_new(folder/'completed.json',{'kind':'streaming-student-full-panel-completed-v2','passed':True,
            'plan':identity(args.plan),'grant':identity(args.grant),'logicalTaskCount':27,'physicalEncoderCount':20,
            'completionGates':checked,'slotLifecycle':lifecycle,'all27Full42All4NumericAndApplicableReuseGatesPassed':True,'evaluationMetricsProduced':False})
    log(folder,'worker-stop',completed=len(done),total=27,reason='complete' if len(done)==27 else 'stop-or-12h-limit')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','run'));p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--qualification',type=Path)
    p.add_argument('--slot-mode',choices=('exited','quiescent','recovery'))
    for name in ('grant','slot-proof','predecessor-identity','resume-proof','output'):p.add_argument('--'+name,type=Path)
    for name in ('plan','grant','slot-proof','predecessor-identity'):p.add_argument('--'+name+'-sha256')
    a=p.parse_args();require(Path('/mnt/freenas').is_mount() and a.plan.resolve().is_relative_to(ROOT),'Mounted NAS plan required')
    if a.action=='register':
        require(a.qualification is not None,'Passing extraction qualification required');value=contract(a.qualification);validate_plan(value);H.write_new(a.plan,value);print(json.dumps(identity(a.plan)));return
    require(all(getattr(a,k) for k in ('grant','grant_sha256','plan_sha256','slot_mode','slot_proof','slot_proof_sha256','predecessor_identity','predecessor_identity_sha256','output')),'Root slot authorization and execution grant required')
    run(a)


if __name__=='__main__':main()
