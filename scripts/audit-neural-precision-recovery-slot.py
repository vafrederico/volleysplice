"""Audit a newly released precision slot after the documented WSL interruption.

All numerical, feature, fit, selection and inference checks preserve v1.
Only process provenance and the additive recovered raw-queue receipt differ.
"""
from analysis.private_ledger import private_value
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
N=Path(private_value('private-reference-0061'))
REPO=Path(__file__).resolve().parents[1]
HERE=N/'recovery-precision-slots-v1'
INTERRUPTION=N/'recovery-20260923T1758-v1/interruption-observation.json'
INTERRUPTION_SHA='1ce7ccc796ba4e2a70c4f924f5ea02e15e4edb28d258e87fc40f3ea1d4239af4'
OLD_SOURCE=N/'cached-dino-precision-execution-v1/preflight.py'
OLD_SOURCE_SHA='de414c75cfd687897defa6643affba48ad3b41c43315a8e8e458000f86068026'
PLAN_SHA='ee9bd18b11ffbc72a5bed350edae72b9b799db54155188da543cd1ab46743fc7'
WORKER_SHA='4ee8692259725253d9039cde3a2eb604f590df2263f505e42ed292d634edb054'
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return {'path':str(Path(path).resolve()),'sha256':h.hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate_release(observed, precision, boot, interruption, absent):
    assert observed['kind']=='recovered-study-slot-observed-exit-v1'
    assert observed['passed'] is True and observed['exitCode']==0 and observed['precisionSlot']==precision
    assert observed['bootId']==boot and observed['interruption']==interruption
    pids=observed['exitedPids']
    assert pids and len(pids)==len(set(pids)) and all(type(pid) is int and pid>0 for pid in pids)
    assert len(observed['processes'])==len(pids)
    assert {v['pid'] for v in observed['processes']}==set(pids)
    assert all(v['bootId']==boot and int(v['startTimeTicks'])>0 and isinstance(v['command'],str)
               and v['command'].strip() for v in observed['processes'])
    assert all(absent(pid) for pid in pids)
    affinity=observed['releasedCpuAffinity']; assert affinity in ([0,1],[10,11],[12,13])
    return pids,affinity

def main():
    p=argparse.ArgumentParser();p.add_argument('--precision',required=True,choices=['fp16','int8']);p.add_argument('--observed-exit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert Path('/mnt/freenas').is_mount() and a.output.resolve().is_relative_to(HERE) and not a.output.exists()
    planpath=N/'cached-dino-precision-v2/plan.json'; worker=N/'cached-dino-precision-v2/worker.py'
    assert digest(planpath)['sha256']==PLAN_SHA and digest(worker)['sha256']==WORKER_SHA
    assert digest(OLD_SOURCE)['sha256']==OLD_SOURCE_SHA
    assert digest(INTERRUPTION)['sha256']==INTERRUPTION_SHA
    plan=read(planpath)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    observed=read(a.observed_exit)
    pids,affinity=validate_release(observed,a.precision,boot,digest(INTERRUPTION),
                                 lambda pid:not Path('/proc',str(pid)).exists())
    os.sched_setaffinity(0,set(affinity));os.nice(10)
    for key in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH']:
        path=HERE/('runtime-preflight-'+a.precision)/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
    mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    win=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        '$o=Get-CimInstance Win32_OperatingSystem; @{physical=[long]$o.FreePhysicalMemory*1024;cFree=[long](Get-PSDrive C).Free}|ConvertTo-Json -Compress'],text=True))
    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert mem>=6*2**30 and win['physical']>=2*2**30 and win['cFree']>=20*2**30 and gpu>=4096
    sys.path.insert(0,str(REPO))
    from analysis.neural_generalization_inputs import verified
    from analysis import neural_recall_sweep as io
    from analysis.neural_selection_duration_view import verify_execution
    queue=module('precision_prereq_reuse',REPO/'scripts/run-neural-generalization-reuse-queue.py')
    containment=module('precision_prereq_containment',REPO/'scripts/generalization-selection-containment.py')
    for ref in plan['code'].values():verified(ref)
    inventory=read(verified(plan['taskInventory'])); jobs=inventory['jobs'];assert len(jobs)==162
    for ref in inventory['registrations']+inventory['reusePlans']:verified(ref)
    panelpath=N/'panels-v1/precision-complete.json';panel=read(panelpath)
    corepath=verified(plan['corePanel']);core=read(corepath)
    assert len(panel['recordingIds'])==len(set(panel['recordingIds']))==42 and panel['precisionVariants']==['fp32','fp16','int8']
    assert panel['kind']=='frozen-independent-inference-panel-v1' and panel['inferenceUsesLabels'] is False and panel['allInferenceTicksValid'] is True
    assert {k:v for k,v in panel.items() if k not in ('features','featureAudit','precisionVariants')}=={k:v for k,v in core.items() if k not in ('features','featureAudit','precisionVariants')}
    for key in ('manifest','features','featureAudit','inventory','registrar'):verified(panel[key])
    fa=read(panel['featureAudit']['path']);assert fa['kind']=='independent-expansion-feature-audit-v1' and fa['passed'] is True
    assert fa['features']==panel['features'] and fa['inputs']==panel['manifest'] and fa['mode']=='complete' and fa['scope']=='all' and fa['auditedRecordingCount']==42
    assert fa['source']==io.identity(REPO/'scripts/audit-neural-generalization-features.py') and fa['inferenceMaskAlwaysAllTrue'] is True and fa['protectedAndPanelTeacherTargetsAbsent'] is True
    continuitypath=N/'features-v1/audit-core-complete-continuity-v1.json';cont=read(continuitypath)
    assert cont['kind']=='feature-index-core-complete-continuity-audit-v1' and cont['passed'] is True
    assert cont['source']==io.identity(REPO/'scripts/audit-neural-feature-index-continuity.py')
    assert cont['completeIndex']==panel['features'] and cont['completeAudit']==panel['featureAudit'] and cont['coreIndex']==core['features'] and cont['coreAudit']==core['featureAudit']
    assert cont['sameGoldInputRevision']==panel['manifest'] and cont['recordingCount']==42 and cont['existingReferenceChanges']==0 and cont['onlyAddedPrecisionReferences'] is True
    shared={};containment.verify_closure(cont,shared)
    selected=[]
    for row in jobs:
        task=read(verified(row['task']))
        keep=task['model']!='distilled-mobile-tcn' if a.precision=='int8' else row['reusePlan'] is not None and task['model'] in ('dino-tcn','mobile-tcn','dino-transformer')
        if keep:selected.append(row)
    assert len(selected)==(72 if a.precision=='fp16' else 135)
    checks=[]
    for row in selected:
        folder=Path(row['fitDirectory']);tp=verified(row['task']);fit=read(folder/'fit-result.json')
        fpath=folder/'fit-numerical-audit.json';f=queue.check_gate(fpath,tp,folder,'independent-generalization-fit-numerical-audit-v1')
        assert f['taskId']==row['taskId'] and f['completed']==fit['temporal'] and f['checkpointEpochs']==[5,15,30,60] and f['student'] is None
        containment.verify_closure(f,shared)
        c={'task':row['task'],'fitNumericalAudit':io.identity(fpath)}
        reused=row['taskId']!=row['physicalOwnerTaskId']
        if reused:
            rpath=folder/'reuse-audit.json';r=queue.check_gate(rpath,tp,folder,'independent-generalization-fit-reuse-audit-v1')
            assert r['plan']==row['reusePlan'] and r['sourceTask']==row['physicalOwnerTask'] and r['targetNumericalAudit']==io.identity(fpath)
            containment.verify_closure(r,shared);c['fitReuseAudit']=io.identity(rpath)
        if a.precision=='int8':
            npath=folder/'inference-numerical-audit-fp32.json';g=queue.check_gate(npath,tp,folder,'independent-generalization-inference-numerical-audit-v1')
            expected=core['recordingIds'];out=folder/'inference'/plan['corePanel']['sha256'][:16]/'fp32'
            assert g['taskId']==row['taskId'] and g['panel']==plan['corePanel'] and g['precision']=='fp32' and g['completed']==fit['temporal'] and g['fitAudit']==io.identity(fpath)
            assert g['checkpointEpochs']==[5,15,30,60] and g['studentFeatureReplay']==[]
            assert g['inferenceReceipts']==[io.identity(out/(key+'.json')) for key in expected]
            assert {p.stem for p in out.glob('*.json')}=={p.stem for p in out.glob('*.npz')}==set(expected)
            assert [(x['recordingId'],x['epoch']) for x in g['cpuReplay']]==[(key,epoch) for key in expected for epoch in (5,15,30,60)]
            containment.verify_closure(g,shared);c['inferenceNumericalAudit']=io.identity(npath)
            if reused:
                rpath=folder/'reuse-inference-audit-fp32.json';r=queue.check_gate(rpath,tp,folder,'independent-generalization-inference-reuse-audit-v1')
                assert r['plan']==row['reusePlan'] and r['sourceTask']==row['physicalOwnerTask'] and r['targetNumericalAudit']==io.identity(npath) and r['fitReuseAudit']==c['fitReuseAudit']
                assert r['panel']==plan['corePanel'] and r['precision']=='fp32' and r['recordingIds']==expected and r['checkpointEpochs']==[5,15,30,60]
                assert r['allRawScoreBytesExactlyEqual'] is True and r['studentFeatureArrayEqualityCount']==0
                containment.verify_closure(r,shared);c['inferenceReuseAudit']=io.identity(rpath)
        checks.append(c);print(json.dumps({'prerequisiteChecked':row['taskId'],'completed':len(checks),'total':len(selected)}),flush=True)
    dino=[r for s in plan['stages'] for r in s['jobs']];assert len(dino)==54
    selectionchecks=[]
    for row in dino:
        folder=Path(row['fitDirectory']);tp=verified(row['task'])
        queue.check_gate(folder/'fit-numerical-audit.json',tp,folder,'independent-generalization-fit-numerical-audit-v1')
        containment.correction_gate(folder/'selection.json',folder/'selection-audit.json',folder/'selection-correction-audit.json',shared)
        assert read(folder/'selection.json')['task']==row['task']
        selectionchecks.append({'task':row['task'],'selection':io.identity(folder/'selection.json'),'correction':io.identity(folder/'selection-correction-audit.json'),'execution':verify_execution(folder/'selection-audit.json')})
    previous=None
    if a.precision=='int8':
        root=N/'recovery-cached-fp32-v1'
        recovery_plan=read(root/'plan.json'); recovery=read(root/'run/complete.json')
        assert recovery['kind']=='interrupted-cached-fp32-recovery-complete-v1'
        assert recovery['plan']==io.identity(root/'plan.json') and recovery['all135All42All4'] is True
        assert recovery['existingChecksUnchanged'] is True and recovery['priorObservedExitCode'] is None
        assert recovery_plan['interruption']==io.identity(INTERRUPTION)
        existing=read(verified(recovery['existingAudit']))
        assert existing['passed'] is True and existing['logicalTasks']==111 and existing['all42All4'] is True
        assert existing['plan']==recovery['plan']
        assert [x['task'] for x in existing['checks']]==[x['task'] for x in recovery_plan['verifiedExistingJobs']]
        assert [x['task'] for x in recovery['newChecks']]==[x['task'] for x in recovery_plan['pendingJobs']]
        assert len(existing['checks'])+len(recovery['newChecks'])==135
        containment.verify_closure(recovery,shared)
        previous=io.identity(root/'run/complete.json')
    assert all(not Path('/proc',str(pid)).exists() for pid in pids) and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
    result={'kind':'precision-raw-released-slot-evidence-v1','released':True,'source':digest(__file__),'plan':io.identity(planpath),'worker':io.identity(worker),'precision':a.precision,
        'observedExit':io.identity(a.observed_exit),'interruption':io.identity(INTERRUPTION),'historicalObservedExitCode':None,'originalPreflightSource':io.identity(OLD_SOURCE),'requiredExitedPids':pids,'bootId':boot,'cpuAffinity':affinity,
        'checkedLogicalTaskCount':len(checks),'taskChecks':checks,'all54SelectionChecks':selectionchecks,'panel':io.identity(panelpath),'featureContinuityAudit':io.identity(continuitypath),
        'priorRawQueueComplete':previous,'resourcesAtStart':{'linuxAvailableBytes':mem,'windows':win,'gpuFreeMiB':gpu},'externalMetricsOpened':False,'gpuUsed':False,'completedAtUTC':datetime.now(timezone.utc).isoformat()}
    io.write_new(a.output,result);print(json.dumps({'released':True,'receipt':io.identity(a.output)}),flush=True)
if __name__=='__main__':main()
