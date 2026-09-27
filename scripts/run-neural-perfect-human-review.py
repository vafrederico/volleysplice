#!/usr/bin/env python3
"""Register and execute fixed hypothetical human-review policies; no fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_human_review as review
from analysis import neural_production_combinations as iv
from analysis.neural_evaluation import evaluate_predictions

ROOT = Path(private_value('private-reference-0091'))
PRIOR = ROOT.parent/'2026-09-19-production-combinations'
MODELS = ('compact_boost','compact_keep','dino_global','dino_boost','dino_keep')
ANCHORS = ('productionDefault','shippedUnion','refitUnion')
SEEDS = (3407,1729,20260918)
WORKLOAD_FIELDS = ('rawDecisionSeconds','reviewSeconds','decisionSeconds','reviewClips',
                   'flaggedCandidates','flaggedPositiveCandidates','flaggedNegativeCandidates',
                   'reviewedTrueRallies','playbackTrueRallies','binaryKeptCandidates','binaryDroppedCandidates',
                   'mixedCandidates','completeRallyLossesBinary','partialRallyLossesBinary',
                   'completeRallyLossesBoundary','partialRallyLossesBoundary')
SOURCES = ('analysis/neural_human_review.py','scripts/run-neural-perfect-human-review.py',
           'scripts/audit-neural-human-review.py','scripts/prepare-neural-human-review-probabilities.py',
           'analysis/tests/test_neural_human_review.py','analysis/tests/test_neural_human_review_audit.py',
           'analysis/tests/test_neural_human_review_probability_input.py',
           'analysis/neural_production_combinations.py','scripts/audit-neural-combination-accounting.py',
           'analysis/neural_evaluation.py','analysis/crop_evaluation.py','analysis/schema.py','analysis/metrics.py')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    p=Path(path)
    return {'path':str(p),'sha256':sha(p),'sizeBytes':p.stat().st_size}


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def bound(ref):
    require(sha(ref['path'])==ref['sha256'],'Bound input changed: '+ref['path'])
    return read(ref['path'])


def configurations():
    combos=[{'id':f'{anchor}--{model}--{policy}','family':'combination',
             'anchor':anchor,'neuralId':model,'policy':policy}
            for anchor in ANCHORS for model in MODELS for policy in review.COMBO_POLICIES]
    solo=[{'id':f'{model}--{policy}','family':'individual','anchor':None,
           'neuralId':model,'policy':policy} for model in MODELS for policy in review.SOLO_POLICIES]
    require(len(combos)==90 and len(solo)==30,'Fixed inventory differs')
    return combos+solo


def register(root):
    require(not (root/'registration.json').exists(),'Already registered')
    prior=read(PRIOR/'registration.json')
    require(canonical(prior['contract'])==prior['sha256'],'Prior registration changed')
    refs={name:identity(PRIOR/name) for name in ('registration.json','report.json','summary.json',
                                               'recipe-audit-v1.json','interpretation-audit-v1.json')}
    report=bound(refs['report.json'])
    require(report['status']=='completed-fixed-combinations' and report['contractSha256']==prior['sha256'],
            'Prior comparison incomplete')
    for name in ('recipe-audit-v1.json','interpretation-audit-v1.json'):
        audit=bound(refs[name])
        require(audit['passed'] and audit['contractSha256']==prior['sha256'],'Prior audit did not pass')
    for ref in prior['contract']['code'].values():
        require(sha(ref['path'])==ref['sha256'],'Prior registered code changed')
    neural=bound(prior['contract']['neuralInput'])
    production=bound(prior['contract']['productionInput'])
    meta=read(root/'probability-input.json')
    require(meta['previousContractSha256']==prior['sha256'] and meta['previousNeuralInput']==prior['contract']['neuralInput'],
            'Probability source scope differs')
    require(meta['roundTripExact'] and not meta['protectedTestOpened'] and not meta['goldUsedForQueueSelection'],
            'Probability adapter qualification differs')
    require(sha(meta['sourceScript']['path'])==meta['sourceScript']['sha256'],'Probability adapter changed')
    require(sha(meta['npz']['path'])==meta['npz']['sha256'],'Probability arrays changed')
    records={r['id']:r for r in neural['records']}
    require(len(records)==8 and len({r['sourceGroup'] for r in records.values()})==4 and
            sum(len(r['rallies']) for r in records.values())==322,'Evaluation scope differs')
    require({r['id'] for r in production['recordings']}==set(records),'Production scope differs')
    require(len(meta['entries'])==120 and meta['models']==list(MODELS) and meta['seeds']==list(SEEDS),
            'Probability inventory differs')
    arrays=np.load(meta['npz']['path'],allow_pickle=False)
    seen=set()
    for entry in meta['entries']:
        key=entry['modelId'],entry['seed'],entry['recordingId']
        require(key not in seen and key[0] in MODELS and key[1] in SEEDS and key[2] in records,
                'Duplicate or unexpected probability entry')
        seen.add(key)
        values=arrays[entry['liveKey']]
        times=arrays[entry['timesKey']]
        require(values.dtype==np.float32 and times.dtype==np.float64 and values.shape==times.shape
                and len(values)==entry['tickCount'] and np.all(np.isfinite(values))
                and np.all((values>=0)&(values<=1)) and np.all(np.diff(times)>0), 'Invalid probability array')
        require(hashlib.sha256(values.tobytes()).hexdigest()==entry['liveBytesSha256'],'Live bytes differ')
        require(records[key[2]]['sourceGroup']==entry['sourceGroup'],'Probability held group differs')
    for row in meta['records']:
        times=arrays[row['timesKey']]
        require(row['durationSeconds']==records[row['id']]['durationSeconds'] and
                hashlib.sha256(times.tobytes()).hexdigest()==row['timelineBytesSha256'],'Probability times differ')
    arrays.close()
    protocol=REPO/'docs/research/neural-perfect-human-review-protocol-2026-09-19.md'
    snapshot=root/'protocol-initial.md'
    with snapshot.open('xb') as f:f.write(protocol.read_bytes())
    contract={'kind':'fixed-perfect-human-review-development-v1','createdAt':datetime.now(timezone.utc).isoformat(),
              'primaryMetric':'F1_padP_coreR','targetPaddingSeconds':2,'joinGapSeconds':3,'paddingCases':[0,1,2,3],
              'seeds':list(SEEDS),'models':list(MODELS),'anchors':list(ANCHORS),
              'configurations':configurations(),'policyCount':120,'resultCells':360,'legacyPolicies':30,'legacyCells':90,
              'recordings':8,'sourceGroups':sorted({r['sourceGroup'] for r in records.values()}),'rallies':322,
              'newFits':0,'protectedTestOpened':False,'productionChanged':False,'productionPromotionAllowed':False,
              'candidateFlagsUseGold':False,'hypotheticalHumanDecisionsUseGold':True,
              'reviewContextSeconds':2,'negativeGridSeconds':5,'tolerantSupportSeconds':2,
              'thresholds':{k:list(v) for k,v in review.THRESHOLDS.items()},
              'priorStudy':refs,'neuralInput':prior['contract']['neuralInput'],
              'productionInput':prior['contract']['productionInput'],
              'probabilityInput':identity(root/'probability-input.json'),'probabilities':meta['npz'],
              'protocol':identity(snapshot),'sources':{p:identity(REPO/p) for p in SOURCES},
              'binaryDefinition':'Keep whole flagged raw candidate iff it positively intersects evaluable human rally core; then ordinary export. Not boundary editing.',
              'boundaryDefinition':'Correct only export of flagged candidates against padded human export; playback context is not edit permission; no second export transform.'}
    write(root/'registration.json',{'contract':contract,'sha256':canonical(contract)})
    print(json.dumps({'registered':True,'contractSha256':canonical(contract),'policyCount':120,'resultCells':360},indent=2))


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,REPO/path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def load(root):
    reg=read(root/'registration.json');c=reg['contract']
    require(canonical(c)==reg['sha256'],'Registration changed')
    for ref in [*c['sources'].values(),*c['priorStudy'].values(),c['protocol'],c['probabilityInput'],c['probabilities'],
                c['neuralInput'],c['productionInput']]:
        require(sha(ref['path'])==ref['sha256'],'Registered dependency changed: '+ref['path'])
    return reg,bound(c['neuralInput']),bound(c['productionInput']),bound(c['probabilityInput'])


def accounting_scopes(auditor,records,rows,overrides=None):
    checks=[auditor.audit_duration_records(records,rows,overrides)]
    for subset in [[r for r in records if r['sourceGroup']==g] for g in sorted({r['sourceGroup'] for r in records})]+[[r] for r in records]:
        scoped=None if overrides is None else {r['id']:overrides[r['id']] for r in subset}
        checks.append(auditor.audit_duration_records(subset,iv.duration_rows(subset,scoped),scoped))
    return {'passed':all(x['passed'] for x in checks),'scopeCount':len(checks),'checks':checks}


def legacy_results(root,c,records):
    report=bound(c['priorStudy']['report.json'])
    by_id={r['id']:r for r in records};refs=[]
    for source in report['reviewResults']:
        value=bound(source);padding=[]
        for row in value['queue']:
            counts={'decisionSegments':0,'reviewedTrueRallies':0,'playbackTrueRallies':0}
            per=[]
            for r in row['perRecording']:
                rec=by_id[r['id']]
                x={'id':r['id'],'sourceGroup':rec['sourceGroup'],
                   'decisionSegments':len(r['disputedIntervals']),
                   'reviewedTrueRallyIds':review.touched_rallies(rec,r['disputedIntervals']),
                   'playbackTrueRallyIds':review.touched_rallies(rec,r['reviewIntervals'])}
                x['reviewedTrueRallies']=len(x['reviewedTrueRallyIds']);x['playbackTrueRallies']=len(x['playbackTrueRallyIds'])
                for key in counts:counts[key]+=x[key]
                per.append(x)
            padding.append({**{k:v for k,v in row.items() if k!='perRecording'},'joinGapSeconds':3,
                            **counts,'perRecording':per})
        output={'kind':'legacy-export-disagreement-reference','source':source,'anchor':value['anchor'],
                'neuralId':value['neuralId'],'mode':value['mode'],'seed':value['seed'],
                'automaticDurationMetrics':value['automaticDurationMetrics'],
                'boundaryDurationMetrics':value['oracleDurationMetrics'],'padding':padding,
                'interpretation':'Prior fine-grained disputed-export editing only; no whole-candidate binary estimate.'}
        path=root/'legacy'/Path(source['path']).name
        write(path,output);refs.append(identity(path))
    require(len(refs)==90,'Missing legacy reference cell')
    return refs


def run(root):
    reg,neural,production,prob=load(root);c=reg['contract']
    require(not (root/'report.json').exists(),'Completed report already exists')
    started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
    human_auditor=module('scripts/audit-neural-human-review.py','independent_human_review')
    accounting=module('scripts/audit-neural-combination-accounting.py','independent_duration_accounting')
    prod={r['id']:r for r in production['recordings']}
    entries={(e['modelId'],e['seed'],e['recordingId']):e for e in prob['entries']}
    with np.load(prob['npz']['path'],allow_pickle=False) as packed:
        arrays={key:packed[key].tolist() for key in packed.files}
    artifacts=[];candidate_checks=record_checks=0
    for index,config in enumerate(c['configurations'],1):
        for seed in SEEDS:
            outputs=[];checks=[];base_records=[];binary_records=[]
            for rec in neural['records']:
                rid=rec['id'];n=neural['predictions'][config['neuralId']][str(seed)][rid]
                if config['family']=='combination':
                    p=prod[rid]['cores'][config['anchor']]
                    prefix='refit' if config['anchor']=='refitUnion' else 'shipped'
                    old,new=prod[rid]['cores'][prefix+'Previous'],prod[rid]['cores'][prefix+'V2']
                    plan=review.combination_plan(rec,p,n,config['policy'],old,new)
                    construction=human_auditor.audit_candidate_plan(rec,p,n,config['policy'],plan,previous=old,v2=new)
                else:
                    p=n;entry=entries[config['neuralId'],seed,rid]
                    times,live=arrays[entry['timesKey']],arrays[entry['liveKey']]
                    plan=review.individual_plan(rec,p,times,live,config['policy'])
                    construction=human_auditor.audit_candidate_plan(rec,p,None,config['policy'],plan,times=times,live=live)
                outcome=review.human_outcomes(rec,p,plan)
                audited=human_auditor.audit_record(rec,p,outcome)
                require(construction['passed'] and audited['passed'],'Independent human reconstruction failed')
                checks.append({'id':rid,'candidatePlan':construction,'outcome':audited})
                candidate_checks+=1;record_checks+=1
                outputs.append({'id':rid,'sourceGroup':rec['sourceGroup'],'baseRaw':p,**outcome})
                base_records.append({**rec,'predictions':p})
                binary_records.append({**rec,'predictions':outcome['binaryRaw']})
            base_rows=iv.duration_rows(base_records)
            binary_rows=iv.duration_rows(binary_records)
            boundary={r['id']:r['boundaryExports'] for r in outputs}
            boundary_rows=iv.duration_rows(base_records,boundary)
            binary_evaluation=evaluate_predictions(binary_records)
            for actual,expected in zip(binary_rows,binary_evaluation['padding']):
                for key in ('P_pad','R_core','F1_padP_coreR','paddedModelExportSeconds','paddedHumanExportSeconds','exportDurationDifferenceSeconds'):
                    require(abs(actual[key]-expected[key])<=1e-7,'Binary canonical metric disagreement')
            for base,edited in zip(base_rows,boundary_rows):
                require(edited['incorrectExportSeconds']<=base['incorrectExportSeconds']+1e-7 and
                        edited['missedCoreSeconds']<=base['missedCoreSeconds']+1e-7,'Boundary editing worsens errors')
            if config['family']=='individual':
                old=neural['sourcePrimaryMetrics'][f"{config['neuralId']}:{seed}"]
                for key in ('P_pad','R_core','F1_padP_coreR','paddedModelExportSeconds'):
                    require(abs(base_rows[2][key]-old[key])<=1e-7,'Individual automatic reference differs')
            workload=[]
            for pad in iv.PADS:
                values={key:sum(r['padding'][pad][key] for r in outputs) for key in WORKLOAD_FIELDS}
                values['reviewFractionOfVideo']=values['reviewSeconds']/base_rows[pad]['evaluableVideoSeconds']
                workload.append({'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3,**values})
            result={'configuration':config,'seed':seed,'contractSha256':reg['sha256'],
                    'baseDurationMetrics':base_rows,'binaryDurationMetrics':binary_rows,'boundaryDurationMetrics':boundary_rows,
                    'binaryEvaluation':binary_evaluation,'workload':workload,'recordings':outputs,
                    'independentHumanReview':{'passed':True,'recordings':checks},
                    'independentBinaryAccounting':accounting_scopes(accounting,binary_records,binary_rows),
                    'independentBoundaryAccounting':accounting_scopes(accounting,base_records,boundary_rows,boundary)}
            path=root/'results'/f"{config['id']}--{seed}.json"
            write(path,result);artifacts.append(identity(path))
        print(f'COMPLETED policy {index}/120',flush=True)
    legacy=legacy_results(root,c,neural['records'])
    require(load(root)[0]==reg,'Registered dependencies changed during execution')
    require(len(artifacts)==360,'Incomplete result matrix')
    report={'kind':c['kind'],'status':'completed-perfect-human-review','contractSha256':reg['sha256'],
            'startedAt':started,'completedAt':datetime.now(timezone.utc).isoformat(),'wallSeconds':time.perf_counter()-clock,
            'protectedTestOpened':False,'productionChanged':False,'newFits':0,
            'policyCount':120,'resultCells':360,'results':artifacts,'legacyResults':legacy,
            'independentCandidatePlans':candidate_checks,'independentRecordingOutcomes':record_checks,
            'durationScopesPerOutcome':13,'paddingCasesPerScope':4,'hypotheticalHumanOutcomes':True}
    write(root/'report.json',report)
    print(json.dumps({'completed':True,'report':identity(root/'report.json'),'cells':len(artifacts)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    register(args.root) if args.register else run(args.root)
