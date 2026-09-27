#!/usr/bin/env python3
"""Preserve one interrupted fit, repeat its fixed seed, and verify saved checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
N = Path(private_value('private-reference-0061'))
ROOT = N/'recovery-final-head-fits-v1'
PARTIAL = N/'export-proxy-v1/fits/export-rally-selection/dino-tcn/split-20260923'
PRESERVED = ROOT/'preserved-incomplete-fit'
MODELS = ('dino-tcn', 'mobile-tcn', 'dino-transformer')
QUEUE = REPO/'scripts/run-neural-generalization-reuse-queue.py'
QUEUE_SHA = '1217f8e26bdec387a9d937e177663743fd94840265afb2297118129f79b285a4'

def require(value, message):
    if not value: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text())

def identity(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result

def completed_checks():
    queue = module('unchanged_recovery_queue', QUEUE); completed, pending = [], []
    for name in ('randomized-variants-v1', 'export-proxy-v1'):
        plan_path = N/name/'reuse-plan.json'; plan = read(plan_path)
        for row in plan['tasks']:
            if read(row['task']['path'])['model'] not in MODELS: continue
            folder = Path(row['fitDirectory']); path = folder/'fit-numerical-audit.json'
            if not path.exists(): pending.append(row['taskId']); continue
            numeric = queue.check_gate(path, Path(row['task']['path']), folder, 'independent-generalization-fit-numerical-audit-v1')
            require(numeric['checkpointEpochs'] == [5,15,30,60] and numeric['student'] is None, 'Wrong head fit scope')
            reuse_ref = None
            if row['taskId'] != row['physicalOwnerTaskId']:
                copied = queue.check_gate(folder/'reuse-audit.json', Path(row['task']['path']), folder, 'independent-generalization-fit-reuse-audit-v1')
                require(copied['plan'] == identity(plan_path) and copied['sourceTask'] == row['physicalOwnerTask'], 'Wrong fit reuse association')
                reuse_ref = identity(folder/'reuse-audit.json')
            completed.append({'task': row['task'], 'fitAudit': identity(path), 'reuseAudit': reuse_ref})
    return completed, pending

def prepare(interruption):
    require(Path('/mnt/freenas').is_mount() and not ROOT.exists(), 'Require a new mounted NAS recovery directory')
    require(identity(QUEUE)['sha256'] == QUEUE_SHA, 'Frozen queue changed')
    event = read(interruption)
    require(event['priorObservedExitCode'] is None and event['allListedOwnedProcessesAbsent'], 'Interruption must not claim successful exit')
    require(all(not Path('/proc', str(pid)).exists() for pid in (97282,107249,107405)), 'Old root worker still present')
    completed, pending = completed_checks()
    expected = [f'export-rally-selection/{model}/split-20260923' for model in MODELS]
    require(len(completed) == 69 and pending == expected, 'Recovery population changed')
    require(not (PARTIAL/'fit-result.json').exists() and not (PARTIAL/'temporal/completed.json').exists(), 'Cannot move a completed fit')
    files = [{'relative': str(p.relative_to(PARTIAL)), 'sha256': identity(p)['sha256']} for p in sorted(PARTIAL.rglob('*')) if p.is_file()]
    require({r['relative'] for r in files} == {'temporal/progress.json', *(f'temporal/{kind}-{epoch}.npz' for kind in ('weights','predictions') for epoch in (5,15,30))}, 'Unexpected partial fit files')
    require(read(PARTIAL/'temporal/progress.json')['epoch'] == 43, 'Partial epoch changed')
    ROOT.mkdir()
    commands = [[sys.executable, '-B', str(QUEUE), '--plan', str(N/'export-proxy-v1/reuse-plan.json'), '--action', 'fit', '--variant', 'export-rally-selection', '--model', model, '--device', 'cuda'] for model in MODELS]
    plan = {'kind':'interrupted-final-head-fit-recovery-v1', 'source':identity(__file__), 'queue':identity(QUEUE),
        'interruption':identity(interruption), 'reusePlan':identity(N/'export-proxy-v1/reuse-plan.json'),
        'priorObservedExitCode':None, 'existing69FitChecks':completed, 'remainingTasks':expected,
        'partialOriginalPath':str(PARTIAL), 'preservedPath':str(PRESERVED), 'partialFiles':files,
        'commands':commands, 'seedAndNumericalRecipeChanged':False, 'resumeOptimizerState':False,
        'repeatPolicy':'Repeat interrupted DINO head from its registered seed; compare all six saved epoch5/15/30 archives array-for-array before proceeding.',
        'cpuAffinity':[0,1], 'threads':2, 'maxActiveGpuWorkers':3, 'maxConcurrentImageWorkers':2,
        'createdAtUTC':datetime.now(timezone.utc).isoformat()}
    write(ROOT/'plan.json',plan)
    require(PARTIAL.resolve().is_relative_to(N.resolve()) and PRESERVED.resolve().is_relative_to(ROOT.resolve()), 'Recovery move escaped NAS study')
    PARTIAL.rename(PRESERVED)
    require(not PARTIAL.exists(), 'Partial directory still present')
    preserved = []
    for row in files:
        ref = identity(PRESERVED/row['relative']); require(ref['sha256'] == row['sha256'], 'Preservation changed bytes'); preserved.append(ref)
    old_log = N/'export-proxy-v1/reuse-queue-logs/export-rally-selection__dino-tcn__split-20260923.log'
    shutil.copyfile(old_log, ROOT/'partial-task-log-before-retry.log')
    write(ROOT/'preservation.json', {'plan':identity(ROOT/'plan.json'),'passed':True,'files':preserved,'priorTaskLogSnapshot':identity(ROOT/'partial-task-log-before-retry.log')})
    print(json.dumps({'plan':identity(ROOT/'plan.json'),'preservation':identity(ROOT/'preservation.json')}),flush=True)

def run():
    import numpy as np
    plan = read(ROOT/'plan.json'); require(plan['source'] == identity(__file__) and plan['queue'] == identity(QUEUE), 'Recovery execution source changed')
    preservation = read(ROOT/'preservation.json')
    require(plan['reusePlan'] == identity(N/'export-proxy-v1/reuse-plan.json') and preservation['passed'] and preservation['plan'] == identity(ROOT/'plan.json'), 'Recovery inputs changed')
    require(plan['interruption'] == identity(plan['interruption']['path']), 'Interruption proof changed')
    for ref in preservation['files'] + [preservation['priorTaskLogSnapshot']]:
        require(ref == identity(ref['path']), 'Preserved interruption evidence changed')
    completed, pending = completed_checks()
    require(completed == plan['existing69FitChecks'] and pending == plan['remainingTasks'], 'Completed fit evidence changed before retry')
    require(all(not Path('/proc',str(pid)).exists() for pid in (97282,107249,107405)), 'Old root worker present')
    os.sched_setaffinity(0,{0,1}); os.nice(10); os.environ['PATH'] += ':/usr/lib/wsl/lib'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
    observer = module('recovery_resource_observer', REPO/'scripts/run-neural-student-reuse-profiled.py')
    proc = Path('/proc/self'); parent,ticks = observer.stat_identity((proc/'stat').read_text())
    write(ROOT/'process-identity.json', {'pid':os.getpid(),'parentPid':parent,'startTimeTicks':ticks,'bootId':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'command':sys.argv,'plan':identity(ROOT/'plan.json')})
    results=[]
    for i, command in enumerate(plan['commands']):
        while True:
            state=observer.resources()
            if state['linuxAvailableBytes']>=6*1024**3 and state['windowsPhysicalFreeBytes']>=3*1024**3 and state['gpuFreeMiB']>=5*1024 and state['windowsCFreeBytes']>=20*1024**3: break
            print(json.dumps({'waitingForResources':state}),flush=True);time.sleep(30)
        write(ROOT/f'launch-{i}.json',{'plan':identity(ROOT/'plan.json'),'command':command,'resources':state})
        result=subprocess.run(command,cwd=REPO);results.append({'command':command,'returnCode':result.returncode})
        write(ROOT/f'exit-{i}.json',results[-1]);require(result.returncode==0,'Recovered head queue failed')
        if i==0:
            checks=[]
            for row in plan['partialFiles']:
                if not row['relative'].endswith('.npz'): continue
                prior=PRESERVED/row['relative'];current=PARTIAL/row['relative']
                require(identity(prior)['sha256']==row['sha256'],'Preserved checkpoint changed')
                with np.load(prior,allow_pickle=False) as a,np.load(current,allow_pickle=False) as b:
                    require(set(a.files)==set(b.files) and all(a[k].dtype==b[k].dtype and np.array_equal(a[k],b[k]) for k in a.files),'Repeated seed differs from interrupted checkpoint')
                checks.append({'prior':identity(prior),'repeated':identity(current),'allArraysBitExact':True})
            write(ROOT/'checkpoint-parity.json',{'passed':True,'plan':identity(ROOT/'plan.json'),'checks':checks})
    checks,pending=completed_checks();require(len(checks)==72 and not pending,'Final nonstudent fit population incomplete')
    write(ROOT/'completed.json',{'kind':'recovered-nonstudent-head-fits-complete-v1','passed':True,'plan':identity(ROOT/'plan.json'),'commandResults':results,'fitChecks':checks,'fitCount':72,'checkpointParity':identity(ROOT/'checkpoint-parity.json'),'priorObservedExitCode':None})
    print(json.dumps({'completed':identity(ROOT/'completed.json')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--interruption',type=Path);args=parser.parse_args()
    if args.action=='prepare': require(args.interruption is not None,'Interruption proof required');prepare(args.interruption)
    else: run()
