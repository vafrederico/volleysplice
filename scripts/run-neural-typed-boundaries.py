#!/usr/bin/env python3
"""Frozen-export comparison of typed starts and independently bounded rally cores."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.dont_write_bytecode=True
from analysis.private_ledger import private_value
import numpy as np
from analysis import neural_boundary_advisor as advisor
from analysis import neural_boundary_metrics as metrics
from analysis import neural_boundary_human as human
from analysis import neural_boundary_review as review
from analysis import neural_production_combinations as iv
from analysis import neural_split_metrics as coverage
from analysis.neural_rally_identity_metrics import evaluate_rally_identities

ROOT=Path(private_value('private-reference-0079'))
PRIOR=ROOT.parent/'2026-09-19-production-compact-split-advisor'
SEEDS=(3407,1729,20260918)
MODES=('proposal_confirmation','full_parent')
NEW_SOURCES=('analysis/neural_boundary_advisor.py','analysis/neural_boundary_metrics.py',
 'analysis/neural_boundary_human.py','analysis/neural_boundary_review.py',
 'scripts/run-neural-typed-boundaries.py','scripts/audit-neural-boundary-advisor.py',
 'analysis/tests/test_neural_boundary_advisor.py','analysis/tests/test_neural_boundary_metrics.py',
 'analysis/tests/test_neural_boundary_human.py','analysis/tests/test_neural_boundary_review.py',
 'analysis/tests/test_neural_boundary_audit.py','analysis/tests/test_neural_boundary_runner.py')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p): return {'path':str(p),'sha256':sha(p),'sizeBytes':Path(p).stat().st_size}
def canonical(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,REPO/path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result


def configurations():
    return [{'id':f'{p}--{m}','policy':p,'humanMode':m} for p in advisor.POLICIES for m in MODES]


def register(root):
    require(not (root/'registration.json').exists(),'Already registered')
    previous=read(PRIOR/'registration.json');c=previous['contract']
    require(canonical(c)==previous['sha256'],'Prior contract changed')
    for identity in [*c['sources'].values(),c['input'],c['probabilities']]:
        require(sha(identity['path'])==identity['sha256'],'Prior source/input changed: '+identity['path'])
    require(read(root/'qualification.json')['passed'],'Qualification failed')
    names=sorted(set(c['sources'])|set(NEW_SOURCES))
    for name in names:
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists(),'Existing source copy');shutil.copyfile(REPO/name,dest)
    for src,dest in [(c['input']['path'],root/'input.json'),(c['probabilities']['path'],root/'probabilities.npz'),
        (REPO/'docs/research/neural-typed-boundary-protocol-2026-09-19.md',root/'protocol-initial.md')]:
        require(not dest.exists(),'Existing frozen input');shutil.copyfile(src,dest)
    data=read(root/'input.json')
    require(len(data['records'])==8 and sum(len(r['rallies']) for r in data['records'])==322,'Scope mismatch')
    contract={'kind':'compact-typed-boundaries-v1','createdAt':datetime.now(timezone.utc).isoformat(),
      'seeds':list(SEEDS),'policies':list(advisor.POLICIES),'humanModes':list(MODES),
      'configurations':configurations(),'rankers':list(review.RANKERS),'budgetFractions':list(review.BUDGETS),
      'automaticOutcomes':12,'reviewedOutcomes':192,'primaryMetric':'F1_padP_coreR',
      'targetPaddingSeconds':2,'paddingCases':[0,1,2,3],'joinGapSeconds':3,
      'exportPolicy':'fixed original production export; event core losses reported separately',
      'input':ref(root/'input.json'),'probabilities':ref(root/'probabilities.npz'),
      'protocol':ref(root/'protocol-initial.md'),'qualification':ref(root/'qualification.json'),
      'sources':{p:ref(REPO/p) for p in names},'sourceCopies':{p:ref(root/'source'/p) for p in names},
      'priorRegistration':ref(PRIOR/'registration.json'),'priorContractSha256':previous['sha256'],
      'newFits':0,'protectedTestOpened':False,'productionChanged':False,'goldUsedForPlansOrQueues':False,
      'runtime':{'pythonVersion':sys.version,'pythonExecutable':sys.executable,'numpyVersion':np.__version__}}
    write(root/'registration.json',{'sha256':canonical(contract),'contract':contract})
    print(json.dumps({'registered':True,'contractSha256':canonical(contract),'sources':len(names)}),flush=True)


def load(root):
    reg=read(root/'registration.json');c=reg['contract']
    require(canonical(c)==reg['sha256'],'Contract changed')
    for x in [*c['sources'].values(),*c['sourceCopies'].values(),c['input'],c['probabilities'],c['protocol'],c['qualification']]:
        require(sha(x['path'])==x['sha256'],'Frozen bytes changed: '+x['path'])
    return reg,read(c['input']['path'])


def evaluate(records,exports,auditor,old_auditor,identity_auditor):
    durations=iv.duration_rows(records,export_overrides=exports)
    identities=evaluate_rally_identities(records)
    raw=coverage.evaluate_split_proposals([{**r,'splitProposals':[]} for r in records])
    typed=metrics.evaluate_boundary_proposals(records)
    checks={'fixedExportAudit':auditor.audit_fixed_export(records,durations,exports),
            'identityAudit':identity_auditor.audit_identity_metrics(records,identities),
            'rawCoverageAudit':old_auditor.audit_split_metrics([{**r,'splitProposals':[]} for r in records],raw),
            'typedMetricAudit':auditor.audit_typed_metrics(records,typed)}
    require(all(x['passed'] for x in checks.values()),'Evaluation audit failed')
    return {'durationMetrics':durations,'identityMetrics':identities,'rawCoverageMetrics':raw,
            'typedMetrics':typed,**checks}


def execute_seed(args):
    root_text,seed=args;root=Path(root_text);reg,data=load(root)
    with np.load(reg['contract']['probabilities']['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    auditor=module('typed_boundary_audit','scripts/audit-neural-boundary-advisor.py')
    old_auditor=module('typed_old_split_audit','scripts/audit_neural_split_advisor.py')
    identity_auditor=module('typed_identity_audit','scripts/audit-neural-rally-review-proposals.py')
    sources=data['records'];entries={e['recordingId']:e for e in data['entries'] if e['modelId']=='compact_boost' and e['seed']==seed}
    exports={r['id']:{str(p):iv.export(r['productionEvents'],r,p) for p in (0,1,2,3)} for r in sources}
    plans=[]
    for r in sources:
        e=entries[r['id']];times,scores=arrays[e['timesKey']],arrays[e['scoresKey']]
        require(scores.shape==(len(times),4),'Misaligned scores')
        rows={}
        for policy in advisor.POLICIES:
            plan=advisor.plan(r,r['productionEvents'],e['events'],times,scores,policy)
            poisoned={**r,'rallies':[],'serveMarkers':[{'time':-12345}]}
            require(advisor.plan(poisoned,r['productionEvents'],e['events'],times,scores,policy)==plan,'Gold influenced plan')
            plan['audit']=auditor.audit_plan(r,r['productionEvents'],e['events'],times,scores,policy,plan)
            require(plan['audit']['passed'],'Plan audit failed');rows[policy]=plan
        plans.append({'id':r['id'],'policies':rows})
    baseline=evaluate([{**r,'predictions':r['productionEvents'],'boundaryProposals':[]} for r in sources],exports,auditor,old_auditor,identity_auditor)
    automatic=[]
    for policy in advisor.POLICIES:
        records=[{**r,'predictions':p['policies'][policy]['events'],
                  'boundaryProposals':p['policies'][policy]['eventCandidates']} for r,p in zip(sources,plans)]
        automatic.append({'id':'automatic--'+policy,'policy':policy,
          **evaluate(records,exports,auditor,old_auditor,identity_auditor)})
    reviewed=[]
    for policy in advisor.POLICIES:
        for ranker in review.RANKERS:
            qp=[]
            for r,p in zip(sources,plans):
                plan=p['policies'][policy];jobs=review.jobs(r,plan);queues=review.queues(r,jobs,ranker)
                check=old_auditor.audit_jobs_queues(r,r['productionEvents'],plan['proposals'],[],
                                                  'split_only',ranker,jobs,queues)
                require(check['passed'],'Queue audit failed');qp.append({'jobs':jobs,'queues':queues,'audit':check})
            for bi,fraction in enumerate(review.BUDGETS):
                for mode in MODES:
                    records=[];per=[]
                    for r,p,q in zip(sources,plans,qp):
                        plan=p['policies'][policy];queue=q['queues'][bi];selected=queue['selectedParentIds']
                        if mode=='proposal_confirmation':
                            edit=human.human_edit(r,plan['eventCandidates'],selected,correct_ends=policy in ('separate_ends','head_refined'))
                            check=auditor.audit_proposal_human(r,plan,selected,edit)
                        else:
                            edit=review.full_parent_edit(r,selected)
                            check=auditor.audit_full_parent_human(r,selected,edit)
                        require(check['passed'],'Human audit failed')
                        work=review.workload(r,plan,queue,edit)
                        work_check=auditor.audit_workload(r,plan,queue,edit,work)
                        require(work_check['passed'],'Workload audit failed')
                        per.append({'id':r['id'],'sourceGroup':r['sourceGroup'],**queue,**edit,
                                    'selectedParentIds':queue['selectedParentIds'],
                                    'workload':work,'humanAudit':check,'queueAudit':q['audit'],'workloadAudit':work_check})
                        selected_candidates=[c for c in plan['eventCandidates'] if c['parentId'] in set(selected)]
                        records.append({**r,'predictions':edit['events'],'boundaryProposals':selected_candidates})
                    totals={key:sum(p['workload'][key] for p in per) for key in per[0]['workload']}
                    values=evaluate(records,exports,auditor,old_auditor,identity_auditor)
                    if mode=='full_parent':
                        require(values['typedMetrics']['pooled']['rawCoreSecondsLostFromBaseline']<=1e-8,'Full parent human lost physical core')
                        require(values['rawCoverageMetrics']['pooled']['rawCoreSecondsLostFromBaseline']<=1e-8,'Full parent human lost core')
                        require(values['rawCoverageMetrics']['pooled']['additionalCompleteMisses']==0,'Full parent human lost a rally')
                    reviewed.append({'id':f'{policy}--{mode}--{ranker}--budget-{round(fraction*100):02d}',
                      'policy':policy,'humanMode':mode,'ranker':ranker,'budgetFraction':fraction,
                      'workload':totals,'perRecording':per,'queuePlans':qp,**values})
                print(json.dumps({'seed':seed,'policy':policy,'ranker':ranker,'budget':fraction,'completed':len(reviewed)}),flush=True)
    result={'contractSha256':reg['sha256'],'seed':seed,'plans':plans,'baseline':baseline,
            'automatic':automatic,'reviewed':reviewed}
    path=root/'results'/f'{seed}.json';write(path,result)
    return {'seed':seed,'automatic':len(automatic),'reviewed':len(reviewed),'result':ref(path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--register',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.register:register(args.root)
    if args.run:
        reg,_=load(args.root);require(not (args.root/'results').exists(),'Existing results must be preserved')
        started=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=[f.result() for f in as_completed([pool.submit(execute_seed,(str(args.root),s)) for s in SEEDS])]
        require(sum(r['automatic'] for r in rows)==12 and sum(r['reviewed'] for r in rows)==192,'Incomplete matrix')
        write(args.root/'report.json',{'passed':True,'contractSha256':reg['sha256'],'automaticOutcomes':12,
           'reviewedOutcomes':192,'seeds':sorted(rows,key=lambda r:r['seed']),'wallSeconds':time.monotonic()-started})


if __name__=='__main__':main()
