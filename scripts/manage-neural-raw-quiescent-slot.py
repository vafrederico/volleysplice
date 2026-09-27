#!/usr/bin/env python3
"""Explicit, reversible scheduling handoff of an owned idle raw-inference parent."""
from analysis.private_ledger import private_value
import argparse
from datetime import datetime, timezone
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
N = Path(private_value('private-reference-0061'))
RAW = N/'cached-fp32-remaining-v1'
PROCESS_RECEIPT = N/'cached-dino-precision-execution-v1/prerequisite-process-identities.json'
RAW_PLAN_SHA = 'ea1a9f772c3260d4f05bf4ef4ca99cb88c66fa139b09da8fd5cec9f932ef783f'
RAW_SOURCE_SHA = 'e86e01fcc35beaa595c4caba5f16070c2b46738c5b7bdf0753641a25aceb65d0'
PREREQUISITES = ('fit-result.json', 'fit-numerical-audit.json', 'selection.json',
                 'selection-audit.json', 'selection-correction-audit.json')
IDENTITY_KEYS = ('pid', 'startTimeTicks', 'bootId', 'command')


def require(value, message):
    if not value:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def checked(reference):
    require(identity(reference['path']) == reference, 'Changed bound file: '+reference['path'])
    return Path(reference['path'])


def write_new(path, document):
    path = Path(path).resolve()
    require(path.is_relative_to(N) and not path.exists(), 'Require a new NAS output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False); stream.write('\n')


def process(pid):
    path = Path('/proc')/str(pid)
    value = (path/'stat').read_text(); fields = value[value.rfind(')')+2:].split()
    return {'pid': pid, 'startTimeTicks': int(fields[19]), 'state': fields[0],
            'parentPid': int(fields[1]), 'bootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'command': (path/'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}


def same_process(actual, expected):
    require(all(actual[key] == expected[key] for key in IDENTITY_KEYS), 'Owned process identity changed')


def descendant_ids(pid, parents):
    result, frontier = set(), {pid}
    while frontier:
        added = {child for child, parent in parents.items() if parent in frontier} - result
        result.update(added); frontier = added
    return sorted(result)


def descendants(pid):
    parents = {}
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            value = (item/'stat').read_text(); fields = value[value.rfind(')')+2:].split()
            parents[int(item.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return descendant_ids(pid, parents)


def validate_stopped(expected, actual, children):
    same_process(actual, expected)
    require(actual['state'] == 'T' and not children, 'Parent must be stopped with zero descendants')


def gpu_observation():
    command = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    return {'command': command, 'returnCode': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr, 'scope': 'Informational: WSL may not expose attributable CUDA PIDs.'}


def known_gpu_pids(observation):
    if observation['returnCode']:
        return set()
    return {int(line.split(',')[0].strip()) for line in observation['stdout'].splitlines()
            if line.split(',')[0].strip().isdigit()}


def load_plan(path):
    plan = read(path)
    require(plan['kind'] == 'registered-raw-quiescent-slot-amendment-v1', 'Wrong plan kind')
    require(plan['source'] == identity(__file__), 'Scheduling helper source changed')
    for reference in plan['code']:
        checked(reference)
    require(plan['rawPlan']['sha256'] == RAW_PLAN_SHA and plan['rawSource']['sha256'] == RAW_SOURCE_SHA,
            'Wrong frozen raw queue')
    for key in ('rawPlan', 'rawSource', 'processIdentities', 'rawLaunch', 'wrapperGrant', 'protocol'):
        checked(plan[key])
    require(plan['parent']['pid'] == 107405 and plan['wrapper']['pid'] == 107249,
            'This amendment only owns the two registered processes')
    return plan


def register(output):
    prior = read(PROCESS_RECEIPT)
    owners = {row['pid']: row for row in prior['processes']}
    for pid in (107405, 107249):
        same_process(process(pid), owners[pid])
    require(read(RAW/'root-launch-grant.json')['parentPid'] == 107249, 'Wrong wrapper')
    require(identity(RAW/'plan.json')['sha256'] == RAW_PLAN_SHA
            and identity(RAW/'worker.py')['sha256'] == RAW_SOURCE_SHA, 'Raw queue bytes differ')
    code = [REPO/path for path in ('scripts/manage-neural-raw-quiescent-slot.py',
        'analysis/tests/test_raw_quiescent_slot.py', 'scripts/run-neural-generalization-reuse-queue.py',
        'scripts/audit-neural-generalization-numerics.py', 'scripts/audit-neural-generalization-reuse.py',
        'scripts/generalization-selection-containment.py')]
    plan = {'kind': 'registered-raw-quiescent-slot-amendment-v1', 'source': identity(__file__),
        'code': [identity(path) for path in code], 'rawPlan': identity(RAW/'plan.json'),
        'rawSource': identity(RAW/'worker.py'), 'processIdentities': identity(PROCESS_RECEIPT),
        'rawLaunch': identity(RAW/'launch.json'), 'wrapperGrant': identity(RAW/'root-launch-grant.json'),
        'protocol': identity(REPO/'docs/research/neural-raw-quiescent-slot-2026-09-23.md'),
        'parent': owners[107405], 'wrapper': owners[107249], 'fitPredecessor': owners[97282],
        'maximumActiveGpuWorkers': 3, 'pauseIsExit': False, 'createdAtUTC': stamp()}
    write_new(output, plan)
    return plan


def helpers():
    sys.path.insert(0, str(REPO))
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
    return (load('quiescent_reuse', REPO/'scripts/run-neural-generalization-reuse-queue.py'),
            load('quiescent_closure', REPO/'scripts/generalization-selection-containment.py'))


def audit_fit(row, reuse, closure, cache):
    folder = Path(row['fitDirectory']); task = checked(row['task']); fit = read(folder/'fit-result.json')
    path = folder/'fit-numerical-audit.json'
    gate = reuse.check_gate(path, task, folder, 'independent-generalization-fit-numerical-audit-v1')
    require(gate['taskId'] == row['taskId'] and gate['completed'] == fit['temporal']
            and gate['checkpointEpochs'] == [5,15,30,60] and gate['student'] is None, 'Fit scope differs')
    closure.verify_closure(gate, cache)
    result = {'task': row['task'], 'fitAudit': identity(path)}
    if row['taskId'] != row['physicalOwnerTaskId']:
        path = folder/'reuse-audit.json'
        copied = reuse.check_gate(path, task, folder, 'independent-generalization-fit-reuse-audit-v1')
        require(copied['plan'] == row['reusePlan'] and copied['sourceTask'] == row['physicalOwnerTask']
                and copied['targetNumericalAudit'] == result['fitAudit'], 'Fit reuse scope differs')
        closure.verify_closure(copied, cache); result['reuseAudit'] = identity(path)
    return result


def audit_raw(row, panel_ref, reuse, closure, cache):
    result = audit_fit(row, reuse, closure, cache)
    folder = Path(row['fitDirectory']); path = folder/'inference-numerical-audit-fp32.json'
    gate = reuse.check_gate(path, checked(row['task']), folder, 'independent-generalization-inference-numerical-audit-v1')
    ids = read(checked(panel_ref))['recordingIds']; out = folder/'inference'/panel_ref['sha256'][:16]/'fp32'
    require(len(ids) == len(set(ids)) == 42 and gate['panel'] == panel_ref and gate['precision'] == 'fp32'
            and gate['checkpointEpochs'] == [5,15,30,60] and gate['fitAudit'] == result['fitAudit']
            and gate['completed'] == read(folder/'fit-result.json')['temporal'], 'Raw audit association differs')
    require(gate['inferenceReceipts'] == [identity(out/(key+'.json')) for key in ids]
            and {p.stem for p in out.glob('*.json')} == {p.stem for p in out.glob('*.npz')} == set(ids), 'Raw population differs')
    require([(r['recordingId'],r['epoch']) for r in gate['cpuReplay']] == [(key,e) for key in ids for e in (5,15,30,60)]
            and gate['studentFeatureReplay'] == [], 'Raw replay population differs')
    closure.verify_closure(gate, cache); result['rawAudit'] = identity(path)
    if row['taskId'] != row['physicalOwnerTaskId']:
        path = folder/'reuse-inference-audit-fp32.json'
        copied = reuse.check_gate(path, checked(row['task']), folder, 'independent-generalization-inference-reuse-audit-v1')
        require(copied['plan'] == row['reusePlan'] and copied['sourceTask'] == row['physicalOwnerTask']
                and copied['targetNumericalAudit'] == result['rawAudit'] and copied['fitReuseAudit'] == result['reuseAudit']
                and copied['panel'] == panel_ref and copied['precision'] == 'fp32' and copied['recordingIds'] == ids
                and copied['checkpointEpochs'] == [5,15,30,60] and copied['allRawScoreBytesExactlyEqual'] is True
                and copied['studentFeatureArrayEqualityCount'] == 0, 'Raw reuse scope differs')
        closure.verify_closure(copied, cache); result['rawReuseAudit'] = identity(path)
    return result


def missing_proxy(stage):
    return [{'task': row['task'], 'missing': [name for name in PREREQUISITES
            if not (Path(row['fitDirectory'])/name).exists()]} for row in stage['jobs']
            if any(not (Path(row['fitDirectory'])/name).exists() for name in PREREQUISITES)]


def pause_preflight(plan):
    raw = read(checked(plan['rawPlan']))
    require([(s['id'],len(s['jobs'])) for s in raw['stages']] ==
            [('proxy-av',16),('original-nonstudent',15),('random-nonstudent',48),('proxy-nonstudent',24)], 'Raw stages changed')
    require(not (RAW/'proxy-nonstudent-launch.json').exists() and not (RAW/'complete.json').exists(), 'Too late to pause')
    missing = missing_proxy(raw['stages'][-1]); require(missing, 'No missing proxy prerequisites; no idle handoff')
    reuse, closure = helpers(); cache = {}; stages = []
    for stage in raw['stages'][:3]:
        path = RAW/(stage['id']+'-complete.json'); receipt = read(path)
        require(receipt['plan'] == plan['rawPlan'] and receipt['all42All4Checkpoints'] is True
                and [r['task'] for r in receipt['checks']] == [r['task'] for r in stage['jobs']], 'Incomplete prior stage')
        closure.verify_closure(receipt, cache)
        stages.append({'stage': stage['id'], 'receipt': identity(path),
                       'checks': [audit_raw(row, raw['panel'], reuse, closure, cache) for row in stage['jobs']]})
    require(not (RAW/'proxy-nonstudent-launch.json').exists(), 'Proxy stage launched during audit')
    missing = missing_proxy(raw['stages'][-1]); require(missing, 'Proxy prerequisites finished during audit')
    same_process(process(plan['parent']['pid']), plan['parent'])
    same_process(process(plan['wrapper']['pid']), plan['wrapper'])
    require(not descendants(plan['parent']['pid']), 'Raw stage still has descendants')
    return stages, missing


def stop_and_verify(owner, get_process=process, get_children=descendants, send_signal=os.kill):
    same_process(get_process(owner['pid']), owner)
    send_signal(owner['pid'], signal.SIGSTOP)
    for _ in range(100):
        actual = get_process(owner['pid'])
        if actual['state'] == 'T':
            children = get_children(owner['pid']); validate_stopped(owner, actual, children)
            return {'process': actual, 'state': actual['state'], 'descendants': children}
        time.sleep(.02)
    raise ValueError('Owned parent did not enter stopped state')


def pause(plan_path, output):
    plan = load_plan(plan_path); stages, missing = pause_preflight(plan)
    attempted = False
    try:
        attempted = True; stopped = stop_and_verify(plan['parent'])
        require(not (RAW/'proxy-nonstudent-launch.json').exists(), 'Proxy stage launch raced the pause')
        remaining = missing_proxy(read(checked(plan['rawPlan']))['stages'][-1])
        require(remaining, 'Proxy prerequisites finished before stop; explicit recovery required')
        same_process(process(plan['wrapper']['pid']), plan['wrapper'])
        gpu = gpu_observation()
        require(plan['parent']['pid'] not in known_gpu_pids(gpu), 'Parent unexpectedly owns CUDA work')
        proof = {'kind': 'cached-raw-quiescent-slot-pause-v1', 'passed': True, 'alivePaused': True,
            'releasedForTemporaryReplacement': True, 'predecessorExited': False, 'plan': identity(plan_path),
            'source': identity(__file__), 'processIdentity': plan['parent'], 'wrapperIdentity': plan['wrapper'],
            'firstThreeStageProofs': stages, 'missingProxyPrerequisitesBefore': missing,
            'missingProxyPrerequisites': remaining, 'proxyLaunchAbsent': True, 'afterStop': stopped,
            'waitingPhase': 'own selection gates: proxy-nonstudent',
            'waitingPhaseEvidence': 'Inferred from frozen control flow, completed prior stages, absent proxy launch, and missing readiness prerequisites.',
            'gpuObservation': gpu, 'pausedAtUTC': stamp(), 'trainingOrMetricChanges': False}
        write_new(output, proof); return proof
    except BaseException as error:
        write_new(output, {'kind': 'cached-raw-quiescent-slot-pause-failure-v1', 'passed': False,
            'studentGrantAllowed': False, 'plan': identity(plan_path), 'source': identity(__file__),
            'stopAttempted': attempted, 'processIdentity': plan['parent'], 'error': repr(error),
            'recovery': 'Do not launch a replacement. Reverify exact PID/start/boot/command and active descendants; root must explicitly recover/resume the owned parent.', 'failedAtUTC': stamp()})
        raise


def validate_pause_header(proof, plan):
    require(proof['kind'] == 'cached-raw-quiescent-slot-pause-v1' and proof['passed'] is True
            and proof['alivePaused'] is True and proof['releasedForTemporaryReplacement'] is True
            and proof['predecessorExited'] is False and proof['proxyLaunchAbsent'] is True, 'Not a valid paused slot')
    require(proof['source'] == identity(__file__) and proof['processIdentity'] == plan['parent']
            and proof['wrapperIdentity'] == plan['wrapper'], 'Pause ownership differs')
    require([r['stage'] for r in proof['firstThreeStageProofs']] == ['proxy-av','original-nonstudent','random-nonstudent']
            and [len(r['checks']) for r in proof['firstThreeStageProofs']] == [16,15,48]
            and proof['missingProxyPrerequisites'], 'Pause prerequisite proof narrowed')
    validate_stopped(plan['parent'], proof['afterStop']['process'], proof['afterStop']['descendants'])


def validate_pause_proof(proof_path, expected_plan=None):
    proof = read(proof_path); plan_path = checked(proof['plan']); plan = load_plan(plan_path)
    require(expected_plan is None or proof['plan'] == expected_plan, 'Wrong pause amendment')
    validate_pause_header(proof, plan)
    validate_stopped(plan['parent'], process(plan['parent']['pid']), descendants(plan['parent']['pid']))
    same_process(process(plan['wrapper']['pid']), plan['wrapper'])
    require(not (RAW/'proxy-nonstudent-launch.json').exists(), 'Paused proxy launch exists')
    _, closure = helpers(); closure.verify_closure(proof)
    return {'proof': identity(proof_path), 'plan': identity(plan_path), 'parent': process(plan['parent']['pid']),
            'wrapper': process(plan['wrapper']['pid']), 'descendants': [], 'validatedAtUTC': stamp()}


def validate_fit_exit(observed, plan, process_exists):
    require(observed['passed'] is True and observed['exitCode'] == 0 and 97282 in observed['exitedPids']
            and observed['bootId'] == plan['fitPredecessor']['bootId']
            and observed['processIdentities'] == plan['processIdentities'] and observed['allDescendantsExited'] is True
            and not process_exists(97282), 'Fit predecessor did not exit successfully')


def validate_resume_proof(proof_path, expected_pause=None):
    proof = read(proof_path); plan = load_plan(checked(proof['plan']))
    require(proof['kind'] == 'cached-raw-quiescent-slot-resume-v1' and proof['passed'] is True
            and proof['source'] == identity(__file__) and proof['fitCount'] == len(proof['fitChecks']) == 72
            and proof['maximumActiveGpuWorkers'] == 3 and proof['trainingOrMetricChanges'] is False,
            'Invalid resume proof')
    require(expected_pause is None or proof['pauseProof'] == expected_pause, 'Wrong resumed pause')
    pause_proof = read(checked(proof['pauseProof'])); validate_pause_header(pause_proof, plan)
    require(pause_proof['plan'] == proof['plan'], 'Resume and pause plans differ')
    same_process(proof['afterContinue'], plan['parent'])
    require(proof['afterContinue']['state'] != 'T', 'Resume never left stopped state')
    validate_fit_exit(read(checked(proof['fitObservedExit'])), plan, lambda _pid: False)
    _, closure = helpers(); closure.verify_closure(proof)
    return {'proof': identity(proof_path), 'plan': proof['plan'], 'pauseProof': proof['pauseProof'],
            'scope': 'Bound historical lifecycle proof; resumed raw parent may now have exited.'}


def resume(plan_path, proof_path, exit_path, output):
    plan = load_plan(plan_path); before = validate_pause_proof(proof_path, identity(plan_path)); observed = read(exit_path)
    validate_fit_exit(observed, plan, lambda pid: Path('/proc',str(pid)).exists())
    raw = read(checked(plan['rawPlan'])); inventory = read(checked(raw['taskInventory']))
    rows = [r for r in inventory['jobs'] if r['reusePlan'] and read(checked(r['task']))['model']
            in ('dino-tcn','mobile-tcn','dino-transformer')]
    require(len(rows) == 72, 'Fit prerequisite population differs')
    reuse, closure = helpers(); cache = {}; fits = [audit_fit(r,reuse,closure,cache) for r in rows]
    validate_pause_proof(proof_path, identity(plan_path))
    require(not Path('/proc/97282').exists(), 'Predecessor PID reappeared')
    os.kill(plan['parent']['pid'], signal.SIGCONT)
    for _ in range(100):
        state = process(plan['parent']['pid']); same_process(state, plan['parent'])
        if state['state'] != 'T':
            break
        time.sleep(.02)
    require(state['state'] != 'T', 'Resume did not complete')
    result = {'kind': 'cached-raw-quiescent-slot-resume-v1', 'passed': True, 'plan': identity(plan_path),
        'source': identity(__file__), 'pauseProof': identity(proof_path), 'before': before,
        'fitObservedExit': identity(exit_path), 'fitChecks': fits, 'fitCount': 72, 'afterContinue': state,
        'resumedAtUTC': stamp(), 'maximumActiveGpuWorkers': 3, 'trainingOrMetricChanges': False}
    write_new(output,result); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('action',choices=['register','pause','validate','resume'])
    p.add_argument('--plan',type=Path); p.add_argument('--proof',type=Path); p.add_argument('--fit-exit',type=Path)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    require(Path('/mnt/freenas').is_mount() and a.output.resolve().is_relative_to(N)
            and not a.output.exists(), 'Mounted NAS/new output required')
    if a.action == 'register': result=register(a.output)
    elif a.action == 'pause': result=pause(a.plan,a.output)
    elif a.action == 'resume': result=resume(a.plan,a.proof,a.fit_exit,a.output)
    else: result=validate_pause_proof(a.proof);write_new(a.output,result)
    print(json.dumps({'result':identity(a.output),'passed':result.get('passed',True)}))


if __name__ == '__main__':
    main()
