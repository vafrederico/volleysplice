"""Controlled frozen MobileNetV3 Large substitution; all artifacts go to --output.

Uses the already registered source splits, native-source image samples and AV
features. No legacy cache or experiment source is modified. Private input paths
are explicit CLI arguments and remain in private receipts, never public reports.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import mobile_visual_features as mobile
from analysis import neural_generalization_inputs as inputs
from analysis import neural_recall_sweep as sweep
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_experiment import validate_membership, load_checkpoint
from analysis.neural_recognition_fit import fit_model, predict
from analysis.recognition_temporal_model import RecognitionConfig

CONFIG = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8, token_dimension=960)
WEIGHTS = 'MobileNet_V3_Large_Weights.IMAGENET1K_V1'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable(path, data)


def register(a):
    tasks = []
    for directory in ('randomized-variants-v1', 'export-proxy-v1'):
        for path in sorted((a.study/directory/'tasks').glob('*mobile-tcn*')):
            old = read(path)
            if old['model'] != 'mobile-tcn':
                continue
            if not old.get('calibrationIds'):
                continue
            task = {k: old[k] for k in ('variant', 'seed', 'trainIds', 'calibrationIds',
                    'commonEvaluationGroups', 'manifest')}
            for key in ('splitSeed', 'selectionLabelPolicy'):
                if key in old:
                    task[key] = old[key]
            task.update(parent=identity(path), features=identity(a.study/'features-v1/features-complete.json'),
                        epochs=list(sweep.EPOCHS), floorsPercent=[98, 99], config=asdict(CONFIG))
            records = read(inputs.verified(task['manifest']))['records']
            validate_membership(task, records)
            tasks.append(task)
    assert tasks
    plan = dict(kind='mobile-large-substitution-v1', weights=WEIGHTS, tasks=tasks,
                source=identity(__file__), targetPaddingSeconds=2, paddingSeconds=[0, 1, 2, 3],
                joinGapSeconds=3, encoderFrozen=True, precision='fp32', samplingHz=2, inputSize=224,
                selection='Prefer feasible99 then98; max common-unseen exact-label F1; max recall draw within winning variant, F1 tie-break',
                commonUnseenIsUiSelectionData=True, excluded='beach and original-corpus OOF refits',
                inferenceManifest=identity(a.study/'features-v1/inputs.json'),
                features=identity(a.study/'features-v1/features-complete.json'))
    save(a.output/'plan.json', plan)
    print(json.dumps({'registeredFits': len(tasks), 'config': asdict(CONFIG)}), flush=True)


def load_plan(a):
    p = read(a.output/'plan.json')
    assert p['source'] == identity(__file__), 'Experiment source changed after registration'
    return p


def extract(a):
    import torch
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    p = load_plan(a)
    torch.hub.set_dir(str(a.output/'torch-cache'))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).features.to(a.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = inputs.manifest_rows(inputs.verified(p['inferenceManifest']))[1]
    entries = inputs.feature_entries(records, inputs.verified(p['features']))
    for number, row in enumerate(records):
        folder = a.output/'features'/row['id']
        receipt = folder/'receipt.json'
        if receipt.exists():
            prior = read(receipt)
            assert prior['plan'] == identity(a.output/'plan.json')
            inputs.verified(prior['output'])
            continue
        image_ref = entries[row['id']]['imageInput']
        image = read(inputs.verified(image_ref))
        assert image['id'] == row['id'] and image['contract']['source']['contentSha256'] == row['contentSha256']
        pixels = np.load(inputs.verified(image['arrays']['images224']), mmap_mode='r')
        with np.load(inputs.verified(image['arrays']['timing']), allow_pickle=False) as z:
            timing = {k:z[k].copy() for k in ('times', 'boxes', 'quality', 'selected_pts')}
        assert len(pixels) == len(timing['times']) and pixels.shape[1:] == (3,224,224)
        chunks=[]
        started=time.monotonic()
        with torch.inference_mode():
            for left in range(0, len(pixels), 32):
                x=np.asarray(pixels[left:left+32], np.float32)/255.
                x=(x-mobile.RGB_MEAN[None,:,None,None])/mobile.RGB_STD[None,:,None,None]
                spatial=model(torch.from_numpy(x).to(a.device))
                assert spatial.shape[1] == 960
                weights=mobile.regional_pool_weights(timing['boxes'][left:left+len(x)], *spatial.shape[-2:])
                pooled=torch.einsum('bchw,brhw->brc', spatial, torch.from_numpy(weights).to(a.device))
                chunks.append(pooled.cpu().numpy().astype(np.float16))
        tokens=np.concatenate(chunks)
        assert tokens.shape == (len(pixels),4,960) and np.isfinite(tokens).all()
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/'tokens.npz'
        np.savez_compressed(path, timestamps=timing['times'], tokens=tokens,
                            quality=timing['quality'], selected_presentation_times=timing['selected_pts'])
        save(receipt, dict(plan=identity(a.output/'plan.json'), imageInput=image_ref,
             output=identity(path), labelsUsed=False, shape=list(tokens.shape),
             float32TensorBytes=int(tokens.size*4), savedBytes=path.stat().st_size,
             wallSeconds=time.monotonic()-started))
        print(json.dumps({'extracted':number+1,'total':len(records)}),flush=True)


def attach(e, a):
    receipt=read(a.output/'features'/e.id/'receipt.json')
    with np.load(inputs.verified(receipt['output']), allow_pickle=False) as z:
        times,tokens,quality,pts=(z[k].copy() for k in ('timestamps','tokens','quality','selected_presentation_times'))
    assert tokens.shape == (len(times),4,960) and quality.shape == (len(times),6)
    cache=mobile.MobileVisualCache(Path(receipt['output']['path']),times,tokens.astype(np.float32),quality,pts,{})
    aligned=mobile.align_mobile_features(cache,e.times)
    assert np.all(aligned['available']==1)
    extra=np.concatenate((aligned['tokens'].reshape(len(e.times),-1),aligned['quality'],
                          aligned['feature_age_seconds'][:,None],aligned['available'][:,None]),axis=1)
    result=replace(e,values=np.concatenate((e.values,extra),axis=1).astype(np.float32))
    assert result.values.shape[1] == CONFIG.input_dimension
    return result


def fit(a):
    import torch
    torch.set_num_threads(4)
    p=load_plan(a)
    for task in p['tasks']:
        draw=task.get('splitSeed',task['seed'])
        destination=a.output/'fits'/task['variant']/f'split-{draw}'
        if (destination/'selection.json').exists():
            continue
        manifest=inputs.verified(task['manifest']); features=inputs.verified(task['features'])
        print(json.dumps({'fitting':task['variant'],'draw':draw}),flush=True)
        data=inputs.load_data(manifest,features,recording_ids=task['trainIds'],family='av')
        ordered={r.example.id:r for rows in data.values() for r in rows}
        data={tier:[] for tier in ('exact','draft','coverage')}
        for key in task['trainIds']:
            r=ordered[key]
            data[r.tier].append(replace(r,example=attach(r.example,a)))
        validation=[attach(e,a) for e in inputs.load_inference_examples(manifest,features,
                    family='av',recording_ids=task['calibrationIds'])]
        contract=sweep.canonical({'plan':identity(a.output/'plan.json'),'task':task})
        fit_model(data['exact'],{k:data[k] for k in ('draft','coverage')},validation,CONFIG,
                  task['seed'],sweep.EPOCHS,destination/'temporal',a.device,contract,'short_boost')
        del data,ordered
        from analysis.neural_generalization_results import selection_examples
        examples=selection_examples(task,manifest,features)
        scores={}
        for epoch in sweep.EPOCHS:
            model,mean,scale=load_checkpoint(destination/'temporal'/f'weights-{epoch}.npz',CONFIG,a.device)
            scores[epoch]={e.id:predict(model,e,mean,scale,CONFIG,a.device) for e in validation}
            del model
        policy='export-rally-proxy-selection' if task.get('selectionLabelPolicy')=='exact-and-export-rally-proxy' else 'exact-rallies'
        candidates=sweep.build_candidate_table(examples,scores,selection_policy=policy)
        save(destination/'selection.json',dict(task=task,config=asdict(CONFIG),
             plan=identity(a.output/'plan.json'),candidates=candidates,
             floors=sweep.select_floors(candidates,floors=(98,99))))
        print(json.dumps({'fitComplete':task['variant'],'draw':draw}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','extract','fit'))
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    globals()[a.phase](a)


if __name__=='__main__':
    main()
