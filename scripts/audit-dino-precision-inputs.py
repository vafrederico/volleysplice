#!/usr/bin/env python3
"""Independent integrity gate before the final encoder-precision head replay."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parents[1]))
SPEC=importlib.util.spec_from_file_location('precision',HERE.with_name('evaluate-dino-precision.py'))
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


def main():
    import torch
    import cv2
    from analysis.dinov2_embeddings import _load_checkpoint_state
    root=p.DEFAULT;p.ensure_nas(root);parent=p.verify(root)
    seen={}
    def bind(ref):
        path=Path(ref['path']);stat=path.stat()
        key=(stat.st_size,stat.st_mtime_ns,ref['sha256'])
        if path in seen:
            if seen[path]!=key: raise ValueError('Artifact changed during audit')
        else:
            if p.sha(path)!=ref['sha256'] or ('sizeBytes' in ref and stat.st_size!=ref['sizeBytes']):
                raise ValueError('Artifact identity differs: '+str(path))
            seen[path]=key
        return path
    def require(value,message):
        if not value: raise ValueError(message)
    parent_id=p.ident(root/'protocol.json')
    for entry in parent['sources']:
        bind(entry);bind(entry['archive'])
    qpath=root/'qualification.json';q=p.read(qpath)
    require(q['passed'] and q['protocol']==parent_id,'Qualification does not pass/bind parent')
    require(all(r['onnxFp32ParityPassed'] and r['freshCacheParityPassed'] for r in q['rows']), 'FP32 parity failed')
    for ref in q['graphs'].values():bind(ref)
    bind(q['pilot'])
    staged=p.read(root/'sequential-rgb-amendment-v1/report.json');bind(staged['protocol'])
    stage_protocol=p.read(staged['protocol']['path']);bind(stage_protocol['source'])
    require(staged['passed'] and stage_protocol['parent']==parent_id,'Staging lineage differs')
    staged_by_id={r['recordingId']:r for r in staged['records']}
    quant=p.read(root/'parallel-int8-amendment-v1.json')
    for field in ('source','graph','sequentialRgb'):bind(quant[field])
    require(quant['parent']==parent_id and quant['maximumWorkers']==2 and quant['ortThreadsPerWorker']==2
            and quant['batchSize']==1,'Parallel execution contract differs')
    require(quant['graph']==p.ident(root/'encoder-dynamic-int8.onnx'),'Parallel graph differs')
    runtime=p.session(root/'encoder-dynamic-int8.onnx',threads=2)
    extraction={arm:p.read(root/f'extraction-{arm}.json') for arm in ('fp16','int8')}
    expected_ids={r['source']['id'] for r in parent['records']}
    receipts={}
    for arm,e in extraction.items():
        require(e['protocol']==parent_id and e['qualification']==p.ident(qpath) and e['arm']==arm,'Extraction lineage differs')
        require({r['recordingId'] for r in e['records']}==expected_ids and len(e['records'])==8,'Extraction population differs')
        require(sum(r['samples'] for r in e['records'])==e['totalSamples'],'Extraction total differs')
        receipts[arm]={r['recordingId']:r for r in e['records']}
    records=[]
    for item in parent['records']:
        src,entry=item['source'],item['dino'];key=src['id']
        bind({'path':entry['dinoPath'],'sha256':entry['dinoSha256']})
        bind({'path':src['video'],'sha256':src['contentSha256']})
        with np.load(entry['dinoPath'],allow_pickle=False) as n:
            times,reference=n['timestamps'],n['tokens']
            historical_video=json.loads(str(n['metadata_json'].item()))['video']
        capture=cv2.VideoCapture(src['video'])
        require(capture.isOpened(),'Source metadata cannot be opened')
        current_fps,current_count=capture.get(cv2.CAP_PROP_FPS),int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        old_ordinals=[min(max(historical_video['frameCount']-1,0),max(0,int(round(float(t)*historical_video['fps'])))) for t in times]
        current_ordinals=[min(max(current_count-1,0),max(0,int(round(float(t)*current_fps)))) for t in times]
        require(old_ordinals==current_ordinals,'Historical/current frame ordinals differ')
        require(times.dtype==np.float64 and times.ndim==1 and np.isfinite(times).all()
                and np.all(np.diff(times)>0),'Invalid FP32 reference timestamps')
        require(reference.dtype==np.float16 and reference.shape==(len(times),10,384)
                and np.isfinite(reference).all(),'Invalid FP32 reference tokens')
        stage=staged_by_id[key];require(stage['allComparisonsBitExact'] and stage['samples']==len(times),'Staging samples differ')
        stage_refs={Path(r['path']).name:r for r in stage['rgbChunks']}
        for arm in ('fp16','int8'):
            receipt=receipts[arm][key]
            require(receipt==p.read(root/arm/key/'report.json'),'Completed extraction receipt differs')
            require(receipt['samples']==len(times) and receipt['batchSize']==(8 if arm=='fp16' else 1)
                    and receipt['timestampsIdentical'],'Receipt shape/semantics differ')
            bind(receipt['cache']);bind(receipt['source']);bind(receipt['reference'])
            require(receipt['source']['sha256']==src['contentSha256'] and receipt['reference']['sha256']==entry['dinoSha256'],'Receipt source association differs')
            with np.load(receipt['cache']['path'],allow_pickle=False) as n:ts,tokens=n['timestamps'],n['tokens']
            require(ts.dtype==np.float64 and np.array_equal(ts,times),'Precision timestamps differ')
            require(tokens.dtype==np.float16 and tokens.shape==reference.shape and np.isfinite(tokens).all(),'Invalid precision tokens')
            starts=list(range(0,len(times),128));require(len(receipt['chunks'])==len(starts),'Missing chunk receipt')
            for left,chunk in zip(starts,receipt['chunks'],strict=True):
                right=min(left+128,len(times));token_path=bind(chunk['tokens']);rgb_path=bind(chunk['rgb'])
                require(token_path.name==f'{left:06d}.npy' and rgb_path.name==token_path.name,'Chunk ordering differs')
                require(chunk['rgb']==stage_refs[rgb_path.name],'RGB chunk/staging association differs')
                a=np.load(token_path,allow_pickle=False)
                require(a.dtype==np.float16 and np.array_equal(a,tokens[left:right]),'Chunk and final cache differ')
                rgb=np.load(rgb_path,mmap_mode='r',allow_pickle=False)
                require(rgb.dtype==np.uint8 and rgb.shape==(right-left,336,336,3),'Invalid RGB model input')
            drift=p.errors(tokens,reference)
            require(drift==receipt['driftVsFp32Reference'],'Embedding drift summary differs')
            if arm=='int8':
                for index in (0,len(times)//2,len(times)-1):
                    rgb=np.load(root/'rgb'/key/f'{(index//128)*128:06d}.npy',mmap_mode='r',allow_pickle=False)
                    actual=runtime.run(None,{'image':p.normalized(rgb[index%128:index%128+1])})[0].astype(np.float16)
                    require(np.array_equal(actual[0],tokens[index]),'Sampled INT8 numerical replay differs')
        records.append({'recordingId':key,'samples':len(times),'allChunksAndCachesVerified':True,
                        'allFrameOrdinalsMatchHistorical':True,
                        'int8NumericalReplaySamples':[0,len(times)//2,len(times)-1],
                        'int8SampledReplayBitExact':True})
        print(json.dumps(records[-1]),flush=True)
    cuda=p.read(root/'cuda-pilot.json');bind(cuda['fp16Checkpoint'])
    original=_load_checkpoint_state(p.ASSETS/'dinov2_vits14_pretrain.pth',torch)
    half=torch.load(cuda['fp16Checkpoint']['path'],weights_only=True,map_location='cpu')
    require(set(original)==set(half),'FP16 parameter inventory differs')
    require(all(torch.equal(value.half(),half[name]) for name,value in original.items()),'FP16 parameters are not exact casts of original')
    for path,key in seen.items():
        stat=path.stat();require((stat.st_size,stat.st_mtime_ns)==key[:2],'Artifact changed during audit')
    report={'kind':'dino-precision-input-integrity-audit-v1','passed':True,'source':p.ident(HERE),
        'protocol':parent_id,'qualification':p.ident(qpath),'extractions':{arm:p.ident(root/f'extraction-{arm}.json') for arm in extraction},
        'records':records,'verifiedArtifactCount':len(seen),'fp16WeightsExactlyOriginalHalfCasts':True,
        'allChunkHashesVerified':True,'allPrecisionTimestampsExactlyOriginal':True,'allCacheShapesDtypesFinite':True,
        'checked':[{ 'path':str(path),'sha256':key[2],'sizeBytes':key[0]} for path,key in seen.items()]}
    p.write(root/'audit-inputs.json',report)
    print(json.dumps({'passed':True,'artifacts':len(seen)}),flush=True)


if __name__=='__main__':main()
