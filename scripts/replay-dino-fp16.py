#!/usr/bin/env python3
"""Early FP16/FP32 paired results, separate from the final three-arm report."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import sys
import numpy as np
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parents[1]))
def module(name,file):
    s=importlib.util.spec_from_file_location(name,HERE.with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=module('precision','evaluate-dino-precision.py')
summary=module('summary','summarize-neural-recognition.py')


def main():
    import torch
    from analysis import neural_expanded_development as expanded
    from analysis.transfer_temporal_model import model_for
    from analysis.neural_evaluation import evaluate_predictions
    root=p.DEFAULT;p.ensure_nas(root);protocol=p.verify(root)
    dest=root/'fp16-early-v1';dest.mkdir(parents=True,exist_ok=True)
    p.write(dest/'protocol.json',{'source':p.ident(HERE),'parent':p.ident(root/'protocol.json'),
        'extraction':p.ident(root/'extraction-fp16.json'),
        'kind':'early-fixed95-fp16-paired-replay','seeds':list(p.SEEDS),'padding':[0,1,2,3],
        'finalThreeWayReportUnchanged':True,'noNewSelections':True})
    examples={r.example.id:r.example for r in expanded.load_data(p.OLD/'manifest-pts-v1.json')['exact']}
    entries={r['source']['id']:r['dino'] for r in protocol['records']}
    torch.set_num_threads(2);arms={arm:[] for arm in ('fp32','fp16')}
    for seed in p.SEEDS:
        control=p.read(p.OLD/'study'/f'result-reviewed_export-dino_tcn-short_boost-{seed}.json')
        rows={arm:[] for arm in arms};scores={arm:{} for arm in arms};refs=[]
        for outer,choice in enumerate(control['selections']):
            epoch,group=choice['epoch'],choice['heldSourceGroup']
            path=Path(control['origin']['fitRoot'])/f'outer-{outer}/refit/weights-{epoch}.npz'
            metadata=p.read(path.parent/'completed.json')
            if p.sha(path)!=metadata['artifacts'][path.name] or group in metadata['trainGroups']:
                raise ValueError('Checkpoint integrity or exclusion failure')
            with np.load(path,allow_pickle=False) as n:
                mean,scale=n['mean'],n['scale'];state={k[7:]:torch.from_numpy(n[k].copy()) for k in n.files if k.startswith('model::')}
            model=model_for('dino_tcn');model.load_state_dict(state,strict=True);refs.append(p.ident(path))
            for arm in arms:
                for original in [e for e in examples.values() if e.group==group]:
                    key=original.id;entry=entries[key]
                    cache=Path(entry['dinoPath']) if arm=='fp32' else root/'fp16'/key/'tokens.npz'
                    expected=entry['dinoSha256'] if arm=='fp32' else p.read(root/'fp16'/key/'report.json')['cache']['sha256']
                    if p.sha(cache)!=expected:raise ValueError('Cache changed')
                    with np.load(cache,allow_pickle=False) as n:ts,tokens=n['timestamps'],n['tokens']
                    right=np.clip(np.searchsorted(ts,original.times),0,len(ts)-1);left=np.maximum(0,right-1)
                    nearest=np.where(abs(ts[left]-original.times)<=abs(ts[right]-original.times),left,right)
                    if np.max(abs(ts[nearest]-original.times))>.125+1e-8:raise ValueError('Timeline mismatch')
                    e=replace(original,values=np.concatenate((original.values,tokens[nearest].reshape(len(original.times),-1)),axis=1).astype(np.float32))
                    values=expanded.predict(model,e,mean,scale,'dino_tcn','cpu');scores[arm][key]=values
                    row=e.row(expanded.base.decode(e,values,choice['decoder']))
                    rows[arm].append({**row,**{k:[i.to_dict() for i in row[k]] for k in ('rallies','ignoredIntervals','predictions')}})
        for arm in arms:
            result={'arm':arm,'seed':seed,'predictions':rows[arm],'evaluation':evaluate_predictions(rows[arm]),
                'checkpoints':refs,'selections':control['selections'],
                'scoreDrift':{key:p.errors(values,scores['fp32'][key]) for key,values in scores[arm].items()}}
            if arm=='fp32' and any(abs(result['evaluation']['primary'][k]-control['evaluation']['primary'][k])>1e-8 for k in ('P_pad','R_core','F1_padP_coreR')):
                raise ValueError('FP32 historical metrics differ')
            p.write(dest/f'result-{arm}-{seed}.json',result);arms[arm].append(result)
            np.savez_compressed(dest/f'probabilities-{arm}-{seed}.npz',**scores[arm])
    report={'kind':'early-fp16-paired-report','protocol':p.ident(dest/'protocol.json'),
        'arms':{arm:summary.group_summary(rows) for arm,rows in arms.items()},
        'onlyFp16Complete':True,'int8EvaluationPending':True,'fixedOriginal95Selection':True,
        'protectedTestOpened':False}
    p.write(dest/'report.json',report)
    print({arm:r['mean'] for arm,r in report['arms'].items()},flush=True)


if __name__=='__main__':main()
