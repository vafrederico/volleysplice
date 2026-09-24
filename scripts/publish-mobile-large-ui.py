"""Append both selected Large variants to existing private comparison references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.neural_development import decode
from analysis.neural_recall_sweep import SweepExample


def read(p):return json.loads(p.read_text())
def encoded(x):return json.dumps(x,separators=(',',':'),allow_nan=False).encode()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    a=p.parse_args()
    evaluation=read(a.experiment/'evaluation.json')
    selected=evaluation['selected']; index=read(a.index)
    models=[]
    for chosen in selected:
        models.append(dict(modelId='neural-mobile-large-tcn-fp32'+('-high-recall' if chosen['mode']=='recall' else ''),
            model='mobile-large-tcn',precision='fp32',modelLabel=f"MobileNetV3 Large-TCN · highest {'recall' if chosen['mode']=='recall' else 'F1'} · target {chosen['floorPercent']}%",
            variant=chosen['variant'],seed=chosen['draw'],trainingSeed=chosen['seed'],recallTargetPercent=chosen['floorPercent'],
            selectionMode=chosen['mode'],commonUnseenUsedForUiSelection=True,**chosen['setting']))
    new_ids={m['modelId'] for m in models}
    backup=a.experiment/'ui-publication-before'; backup.mkdir(exist_ok=True)
    changes=[]
    common=read(a.experiment/'plan.json')['inferenceManifest']['path']
    rows={r['id']:r for r in read(Path(common))['records']}
    counts=[]
    for entry in index['recordings']:
        if entry['id'] not in rows:continue
        row=rows[entry['id']]
        path=a.index.parent/entry['file']; original=path.read_bytes(); doc=json.loads(original)
        assert len(doc['recordings'])==1
        record=doc['recordings'][0]
        assert record['recordingId']==entry['id']
        refs=[r for r in record['references'] if r['modelId'] not in new_ids]
        revision=hashlib.sha256(original).hexdigest()
        for reference in refs:
            reference.setdefault('research',{}).setdefault('provenance',{}).setdefault('uiDraftRevision',revision)
        for model in models:
            chosen=next(s for s in selected if s['mode']==model['selectionMode'])
            path_scores=a.experiment/'fits'/chosen['variant']/f"split-{chosen['draw']}"/'inference'/(entry['id']+'.npz')
            receipt=read(path_scores.with_suffix('.json'))
            assert receipt['labelsUsed'] is False and hashlib.sha256(path_scores.read_bytes()).hexdigest()==receipt['output']['sha256']
            with np.load(path_scores,allow_pickle=False) as z:
                times=z['times']; scores=z[f"epoch_{model['epoch']}"]
            e=SweepExample(entry['id'],row['sourceGroup'],row['durationSeconds'],times,np.ones(len(times),bool),(),())
            rallies=[dict(start=r.start,end=r.end) for r in decode(e,scores,model['decoder'])]
            description=(f"Frozen MobileNetV3 Large V1 224px/2Hz embeddings with FP32 TCN; {model['variant']} draw {model['seed']}, epoch {model['epoch']}. "
                f"Calibration target {model['recallTargetPercent']}% at 2s padding, not guaranteed recall on this video. "
                "Common-unseen is UI selection data. Core rally boundaries remain separate; export gaps strictly under 3s join. "
                "Serve/start is a timing signal, not a serving-side prediction.")
            refs.append(dict(modelId=model['modelId'],modelLabel=model['modelLabel'],description=description,
                rallies=rallies,exportRallies=rallies,exportPolicy='model-predictions',research=dict(
                    recommendation=description,signals=dict(times=times.tolist(),**{h:scores[:,i].tolist() for i,h in enumerate(('live','serve','end','keep'))}),
                    boundaryFlags=[],reviewRegions=[],queue=dict(budgetFraction=0,reviewSeconds=0,selectedParentCount=0),
                    provenance=dict(**model,labelsUsed=False,sourceSha256=receipt['output']['sha256']))))
        record['references']=refs
        entry['modelIds']=[r['modelId'] for r in refs]
        assert len(set(entry['modelIds']))==len(refs)
        target=backup/entry['file']; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(original)
        changes.append((path,encoded(doc),original))
        counts.append(dict(id=entry['id'],counts={r['modelId']:len(r['rallies']) for r in refs if r['modelId'] in new_ids}))
    assert len(changes)==len(rows),'All non-beach inference recordings must exist in UI catalog'
    index['models']=[m for m in index['models'] if m['modelId'] not in new_ids]+models
    old_index=a.index.read_bytes()
    if not (backup/'index.json').exists():(backup/'index.json').write_bytes(old_index)
    # Prepare every output before publishing; restore all touched files on error.
    try:
        for path,data,_ in changes:
            temporary=path.with_suffix('.large.tmp');temporary.write_bytes(data);temporary.replace(path)
        temporary=a.index.with_suffix('.large.tmp');temporary.write_bytes(encoded(index));temporary.replace(a.index)
    except Exception:
        for path,_,original in changes:path.write_bytes(original)
        a.index.write_bytes(old_index)
        raise
    (a.experiment/'ui-publication.json').write_bytes(encoded(dict(models=models,recordings=counts,labelsUsedForInference=False)))
    print(json.dumps({'publishedRecordings':len(changes),'addedModels':list(new_ids)}),flush=True)


if __name__=='__main__':main()
