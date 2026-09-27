"""Evaluate frozen Large fits, select UI variants, and infer the entire catalog."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import neural_generalization_inputs as inputs
from analysis import neural_recall_sweep as sweep
from analysis.neural_generalization_results import manifest_example, COMMON_GROUPS
from analysis.neural_generalization_experiment import load_checkpoint
from analysis.neural_recognition_fit import predict
from analysis.neural_development import decode
from analysis.neural_evaluation import evaluate_predictions
from analysis.neural_context_development import identity, read, write_immutable

spec=importlib.util.spec_from_file_location('large_experiment',Path(__file__).with_name('experiment-mobile-large.py'))
experiment=importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)


def validated_fit(a, task):
    """Bind selection and checkpoint ownership to the frozen fit contract."""
    draw=task.get('splitSeed',task['seed'])
    fit=a.output/'fits'/task['variant']/f'split-{draw}'
    plan_reference=identity(a.output/'plan.json')
    selection=read(fit/'selection.json')
    assert selection['plan']==plan_reference, 'Selection belongs to another plan'
    assert selection['task']==task, 'Selection task differs from registered task'
    assert selection['config']==asdict(experiment.CONFIG), 'Selection model geometry differs'
    assert selection['floors']==sweep.select_floors(selection['candidates'],floors=(98,99)), 'Selection floors changed'
    completed=read(fit/'temporal/completed.json')
    assert completed['contractSha256']==sweep.canonical({'plan':plan_reference,'task':task}), 'Fit contract differs'
    assert completed['seed']==task['seed'] and completed['epochs']==task['epochs'], 'Fit seed or epochs differ'
    assert completed['kind']=='mobile_tcn' and completed['lossArm']=='short_boost', 'Fit recipe differs'
    assert all(completed['model'].get(k)==v for k,v in asdict(experiment.CONFIG).items()), 'Fit model geometry differs'
    assert completed['model']['inputDimension']==experiment.CONFIG.input_dimension, 'Fit input dimension differs'
    records=read(inputs.verified(task['manifest']))['records']
    experiment.validate_membership(task,records)
    by_id={row['id']:row for row in records}
    tiers={tier:[key for key in task['trainIds'] if by_id[key]['labelTier']==tier]
           for tier in ('exact','draft','coverage')}
    assert completed['trainIds']==tiers['exact'], 'Exact fitting population differs'
    assert completed['auxiliaryIds']=={k:tiers[k] for k in ('draft','coverage')}, 'Auxiliary fitting population differs'
    assert completed['scalerTrainIds']==tiers['exact'], 'Scaler fitting population differs'
    assert completed['validationIds']==task['calibrationIds'], 'Calibration population differs'
    expected_artifacts={f'{stem}-{epoch}.npz' for epoch in task['epochs'] for stem in ('weights','predictions')}
    assert set(completed['artifacts'])==expected_artifacts, 'Fit checkpoint inventory differs'
    return fit, selection, completed


def infer(a, plan, task, records, epochs):
    fit,selection,completed=validated_fit(a,task)
    selected_epochs=sorted({r['selected']['epoch'] for r in selection['floors'] if r['feasible']})
    assert epochs==selected_epochs, 'Inference epochs differ from frozen selection'
    weights={}
    for epoch in epochs:
        checkpoint=fit/'temporal'/f'weights-{epoch}.npz'
        reference=dict(path=str(checkpoint),sha256=completed['artifacts'][checkpoint.name])
        inputs.verified(reference)
        weights[str(epoch)]=reference
    predictions={}
    for row in records:
        path=fit/'inference'/(row['id']+'.npz')
        receipt=path.with_suffix('.json')
        if receipt.exists():
            metadata=read(receipt)
            assert metadata['plan']==identity(a.output/'plan.json') and metadata['epochs']==epochs
            assert metadata['weights']==weights and metadata['labelsUsed'] is False, 'Inference lineage differs'
            assert Path(metadata['output']['path']).resolve()==path.resolve(), 'Inference receipt points to another recording'
            inputs.verified(metadata['output'])
            with np.load(path,allow_pickle=False) as z:
                predictions[row['id']]={k:z[k].copy() for k in z.files}
            continue
        e,=inputs.load_inference_examples(inputs.verified(plan['inferenceManifest']),inputs.verified(plan['features']),
                    family='av',recording_ids=[row['id']])
        e=experiment.attach(e,a)
        arrays={'times':e.times}
        for epoch in epochs:
            checkpoint=inputs.verified(weights[str(epoch)])
            model,mean,scale=load_checkpoint(checkpoint,experiment.CONFIG,a.device)
            arrays[f'epoch_{epoch}']=predict(model,e,mean,scale,experiment.CONFIG,a.device)
            del model
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays)
        write_immutable(receipt,dict(plan=identity(a.output/'plan.json'),epochs=epochs,weights=weights,
                         labelsUsed=False,output=identity(path)))
        predictions[row['id']]=arrays
    return predictions


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args(); torch.set_num_threads(4)
    plan=experiment.load_plan(a)
    records=read(inputs.verified(plan['inferenceManifest']))['records']
    common=[r for r in records if r['sourceGroup'] in COMMON_GROUPS and r['scoringPolicy']=='exact-core']
    assert common
    candidates=[]
    for task in plan['tasks']:
        draw=task.get('splitSeed',task['seed'])
        fit,selection,_=validated_fit(a,task)
        epochs=sorted({r['selected']['epoch'] for r in selection['floors'] if r['feasible']})
        if not epochs:
            continue
        scores=infer(a,plan,task,common,epochs)
        for floor in selection['floors']:
            if not floor['feasible']:
                continue
            setting=floor['selected']
            examples=[manifest_example(row,scores[row['id']]['times']) for row in common]
            evaluated=evaluate_predictions([e.row(decode(e,scores[e.id][f"epoch_{setting['epoch']}"],setting['decoder'])) for e in examples])
            candidates.append(dict(variant=task['variant'],draw=draw,seed=task['seed'],
                floorPercent=floor['floorPercent'],setting=setting,evaluation=evaluated))
        print(json.dumps({'evaluated':task['variant'],'draw':draw}),flush=True)
    selected=[]
    for floor in (99,98):
        available=[r for r in candidates if r['floorPercent']==floor]
        if not available:
            continue
        f1=sorted(available,key=lambda r:(-r['evaluation']['primary']['F1_padP_coreR'],r['variant'],r['draw']))[0]
        recall=sorted([r for r in available if r['variant']==f1['variant']],
            key=lambda r:(-r['evaluation']['primary']['R_core'],-r['evaluation']['primary']['F1_padP_coreR'],r['draw']))[0]
        selected=[dict(f1,mode='f1'),dict(recall,mode='recall')]
        break
    assert selected,'Neither target has a feasible fit; no UI model may be published'
    result=dict(kind='mobile-large-common-selection-v1',plan=identity(a.output/'plan.json'),
                commonUnseenIsSelectionData=True,commonExactPanelRecordingCount=len(common),
                commonExactPanelSourceGroupCount=len({r['sourceGroup'] for r in common}),
                candidates=candidates,selected=selected)
    path=a.output/'evaluation.json'
    if path.exists(): assert read(path)==result
    else: write_immutable(path,result)
    for chosen in selected:
        task=next(t for t in plan['tasks'] if t['variant']==chosen['variant'] and t.get('splitSeed',t['seed'])==chosen['draw'])
        _,selection,_=validated_fit(a,task)
        epochs=sorted({r['selected']['epoch'] for r in selection['floors'] if r['feasible']})
        infer(a,plan,task,records,epochs)
    print(json.dumps({'allVideoInferenceComplete':len(records),'selected':[{k:r[k] for k in ('variant','draw','floorPercent','mode')} for r in selected]}),flush=True)


if __name__=='__main__':main()
