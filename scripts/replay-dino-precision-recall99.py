#!/usr/bin/env python3
"""Replay all precision arms under independently selected strict99% controls.

Infeasible folds remain absent. Partial feasible-scope metrics are explicitly
separate from full-eight-recording metrics; they are never averaged as full-study
results. No checkpoint/decoder is chosen from precision outcomes.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]))
SPEC=importlib.util.spec_from_file_location('precision',HERE.with_name('evaluate-dino-precision.py'))
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


def main():
    import torch
    from analysis import neural_expanded_development as expanded
    from analysis.transfer_temporal_model import model_for
    from analysis.neural_evaluation import evaluate_predictions
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=p.DEFAULT)
    parser.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args();root=args.root
    p.ensure_nas(root);parent=p.verify(root)
    dest=root/'recall99-v1';dest.mkdir(parents=True,exist_ok=True)
    selection=p.read(args.selection)
    def bind(ref):
        if p.sha(ref['path'])!=ref['sha256']:
            raise ValueError('Strict99 binding changed: '+ref['path'])
        return Path(ref['path'])
    final_audit=p.read(args.selection.parent/'audit-refits.json')
    if final_audit.get('passed') is not True or final_audit['report']['sha256']!=p.sha(args.selection):
        raise ValueError('Passing completed strict99 audit required')
    bind(final_audit['report'])
    prior_audit=p.read(bind(final_audit['priorSelectionAudit']))
    if prior_audit.get('passed') is not True:
        raise ValueError('Prior strict99 selection audit did not pass')
    plan_path=bind(selection['refitPlan']);plan=p.read(plan_path)
    input_audit=p.read(root/'audit-inputs.json')
    if input_audit.get('passed') is not True or input_audit['protocol']!=p.ident(root/'protocol.json'):
        raise ValueError('Passing precision cache audit required')
    manifest=p.read(p.OLD/'manifest-pts-v1.json')
    controls=[r for r in selection['results'] if r['model']=='dino_tcn_short_boost']
    if selection['recallFloor']!=.99 or {r['seed'] for r in controls}!=set(p.SEEDS):
        raise ValueError('Unexpected strict99 selection scope')
    if any(d['decisions']['joint']['outerStatus']=='missing-selected-outer-checkpoint' for r in controls for d in r['folds']):
        raise ValueError('Wait for selected outer refits before final precision replay')
    refs=[]
    for r in controls:
        seed=r['seed']
        old=p.read(p.OLD/'study'/f'result-reviewed_export-dino_tcn-short_boost-{seed}.json')
        for outer,f in enumerate(r['folds']):
            decision=f['decisions']['joint'];chosen=decision['selected']
            if chosen is None: continue
            if decision['outerStatus']=='available-from-parity-verified-refit':
                prediction=Path(decision['supplementalCheckpoint']['path'])
                bind(decision['supplementalCheckpoint'])
                folder=prediction.parent
                parity=p.read(folder/'original-parity.json')
                task=next(t for t in plan['tasks'] if t['model']=='dino_tcn_short_boost'
                          and t['seed']==seed and t['outerIndex']==outer)
                if not (parity.get('passed') is True and parity.get('weightsAndPredictionsExactlyEqual') is True
                        and parity.get('samplingHistoryExactlyEqual') is True and parity['task']==task
                        and task['heldSourceGroup']==f['heldSourceGroup'] and task['selected']==chosen
                        and parity['plan']['sha256']==p.sha(plan_path)):
                    raise ValueError('Supplemental checkpoint original parity/task/plan differs')
                bind(parity['completed']);bind(parity['originalCompleted']);bind(parity['plan'])
                refs.append(p.ident(folder/'original-parity.json'))
            else: folder=Path(old['origin']['fitRoot'])/f'outer-{outer}'/'refit'
            weights=folder/f"weights-{chosen['epoch']}.npz"
            completed=p.read(folder/'completed.json')
            if p.sha(weights)!=completed['artifacts'][weights.name]:
                raise ValueError('Checkpoint hash differs from completed fit')
            group=f['heldSourceGroup']
            train=[r['id'] for r in manifest['exactRows'] if r['sourceGroup']!=group]
            held=[r['id'] for r in manifest['exactRows'] if r['sourceGroup']==group]
            auxiliary={tier:[r['id'] for r in manifest[key] if r['sourceGroup']!=group]
                       for tier,key in (('draft','draftRows'),('coverage','coverageRows'))}
            if not (completed['trainIds']==train and completed['scalerTrainIds']==train
                    and completed['validationIds']==held and completed['validationGroups']==[group]
                    and group not in completed['trainGroups'] and completed['auxiliaryIds']==auxiliary
                    and all(group not in groups for groups in completed['auxiliaryGroups'].values())
                    and completed['seed']==seed and completed['kind']=='dino_tcn'
                    and completed['lossArm']=='short_boost'):
                raise ValueError('Checkpoint source memberships/seed/family differ')
            f['precisionCheckpoint']=p.ident(weights)
            refs += [p.ident(weights),p.ident(folder/'completed.json')]
    if not (dest/'protocol.json').exists():
        p.write(dest/'protocol.json',{'kind':'frozen99-controls-encoder-precision-replay',
            'source':p.ident(HERE),'selection':p.ident(args.selection),'parent':p.ident(root/'protocol.json'),
            'selectionAudit':p.ident(args.selection.parent/'audit-refits.json'),
            'precisionInputAudit':p.ident(root/'audit-inputs.json'),
            'checkpoints':refs,'arms':['fp32','fp16','int8'],
            'rule':'Use each independently selected strict99 fold unchanged for every precision; no fallback for infeasible folds.',
            'aggregation':'Full-eight-recording metrics only when every fold feasible; partial scopes reported separately, no cross-seed average with unequal scopes.'})
    protocol=p.read(dest/'protocol.json')
    for ref in [protocol[k] for k in ('source','selection','parent','selectionAudit','precisionInputAudit')]+protocol['checkpoints']:
        if p.sha(ref['path'])!=ref['sha256']: raise ValueError('Frozen99 replay input changed')
    examples=expanded.base.load_examples(Path(manifest['exactManifest']['path']),False)
    entries={r['source']['id']:r['dino'] for r in parent['records']}
    torch.set_num_threads(2);results=[]
    for control in controls:
        seed=control['seed'];all_rows={arm:[] for arm in ('fp32','fp16','int8')};folds=[]
        for fold in control['folds']:
            group=fold['heldSourceGroup'];decision=fold['decisions']['joint'];chosen=decision['selected']
            if chosen is None:
                folds.append({'heldSourceGroup':group,'feasible':False,'reason':decision['outerStatus']});continue
            with np.load(fold['precisionCheckpoint']['path'],allow_pickle=False) as n:
                mean,scale=n['mean'],n['scale']
                state={k[7:]:torch.from_numpy(n[k].copy()) for k in n.files if k.startswith('model::')}
            model=model_for('dino_tcn');model.load_state_dict(state,strict=True)
            result={'heldSourceGroup':group,'feasible':True,'selected':chosen,
                    'checkpoint':fold['precisionCheckpoint'],'arms':{}}
            reference_scores={}
            for arm in ('fp32','fp16','int8'):
                rows=[];drift={}
                for original in [e for e in examples if e.group==group]:
                    key=original.id;entry=entries[key]
                    path=Path(entry['dinoPath']) if arm=='fp32' else root/arm/key/'tokens.npz'
                    expected=entry['dinoSha256'] if arm=='fp32' else p.read(root/arm/key/'report.json')['cache']['sha256']
                    if p.sha(path)!=expected: raise ValueError('Embedding cache changed')
                    with np.load(path,allow_pickle=False) as n: ts,tokens=n['timestamps'],n['tokens']
                    right=np.clip(np.searchsorted(ts,original.times),0,len(ts)-1);left=np.maximum(0,right-1)
                    nearest=np.where(abs(ts[left]-original.times)<=abs(ts[right]-original.times),left,right)
                    if np.max(abs(ts[nearest]-original.times))>.125+1e-8: raise ValueError('Timeline alignment differs')
                    e=replace(original,values=np.concatenate((original.values,tokens[nearest].reshape(len(original.times),-1)),axis=1).astype(np.float32))
                    scores=expanded.predict(model,e,mean,scale,'dino_tcn','cpu')
                    if arm=='fp32': reference_scores[key]=scores
                    drift[key]=p.errors(scores,reference_scores[key])
                    row=e.row(expanded.base.decode(e,scores,chosen['decoder']))
                    rows.append({**row,**{k:[i.to_dict() for i in row[k]] for k in ('rallies','ignoredIntervals','predictions')}})
                evaluation=evaluate_predictions(rows)
                if arm=='fp32':
                    expected=decision['heldEvaluation']['primary']
                    if any(abs(evaluation['primary'][k]-expected[k])>1e-8 for k in ('P_pad','R_core','F1_padP_coreR')):
                        raise ValueError('Strict99 FP32 replay differs from control')
                    canonical={r['id']:r for r in control['modes']['joint']['predictions'] if r['sourceGroup']==group}
                    if {r['id']:r for r in rows}!=canonical or evaluation!=decision['heldEvaluation']:
                        raise ValueError('Strict99 FP32 raw boundaries/coverage differ from canonical control')
                result['arms'][arm]={'evaluation':evaluation,'predictions':rows,'scoreDrift':drift}
                all_rows[arm].extend(rows)
            folds.append(result)
            print(json.dumps({'seed':seed,'group':group,'complete':True}),flush=True)
        complete=all(f['feasible'] for f in folds)
        summary={'seed':seed,'completeEvaluationScope':complete,'folds':folds,'arms':{}}
        for arm,rows in all_rows.items():
            summary['arms'][arm]={'scopeRecordingIds':[r['id'] for r in rows],
                'fullEightRecordingEvaluation':evaluate_predictions(rows) if complete else None,
                'partialFeasibleScopeEvaluation':evaluate_predictions(rows) if rows and not complete else None}
        p.write(dest/f'result-{seed}.json',summary);results.append(p.ident(dest/f'result-{seed}.json'))
    p.write(dest/'report.json',{'protocol':p.ident(dest/'protocol.json'),'results':results,
        'completeSeeds':[r['seed'] for r in controls if all(f['decisions']['joint']['feasible'] for f in r['folds'])],
        'noPrecisionSpecificTuning':True,'protectedTestOpened':False})


if __name__=='__main__': main()
