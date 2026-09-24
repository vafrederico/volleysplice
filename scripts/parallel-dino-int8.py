#!/usr/bin/env python3
"""Two disjoint CPU workers for the already registered DINO INT8 graph."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import importlib.util
import json
import time
import numpy as np

HERE=Path(__file__).resolve()
SPEC=importlib.util.spec_from_file_location('precision',HERE.with_name('evaluate-dino-precision.py'))
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


def worker(row):
    root=p.DEFAULT
    p.ensure_nas(root)
    source,entry=row['source'],row['dino']
    folder=root/'int8'/source['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'report.json').exists(): return p.read(folder/'report.json')
    if p.sha(source['video'])!=source['contentSha256'] or p.sha(entry['dinoPath'])!=entry['dinoSha256']:
        raise ValueError('Source changed')
    qualification=p.read(root/'qualification.json')
    graph=root/'encoder-dynamic-int8.onnx'
    if p.sha(graph)!=qualification['graphs']['dynamicInt8']['sha256']: raise ValueError('Graph changed')
    session=p.session(graph,threads=2)
    staged=p.read(root/'sequential-rgb-amendment-v1'/(source['id']+'.json'))
    rgb_refs={Path(r['path']).name:r for r in staged['rgbChunks']}
    with np.load(entry['dinoPath'],allow_pickle=False) as n: ts,reference=n['timestamps'],n['tokens']
    chunks=[];refs=[];read_seconds=infer_seconds=0.;resumed=0;start=time.perf_counter()
    for left in range(0,len(ts),128):
        right=min(left+128,len(ts));path=folder/f'{left:06d}.npy';rgb_path=root/'rgb'/source['id']/path.name
        if p.sha(rgb_path)!=rgb_refs[path.name]['sha256']: raise ValueError('RGB staging changed')
        if path.exists():
            values=np.load(path,allow_pickle=False);resumed+=right-left
        else:
            tick=time.perf_counter();rgb=np.load(rgb_path,allow_pickle=False);read_seconds+=time.perf_counter()-tick
            tick=time.perf_counter();rows=[]
            for i in range(len(rgb)):
                rows.append(session.run(None,{'image':p.normalized(rgb[i:i+1])})[0])
            values=np.concatenate(rows).astype(np.float16);infer_seconds+=time.perf_counter()-tick
            np.save(path,values)
        if values.dtype!=np.float16 or values.shape!=(right-left,10,384) or not np.isfinite(values).all():
            raise ValueError('Invalid precision cache chunk')
        chunks.append(values);refs.append({'tokens':p.ident(path),'rgb':rgb_refs[path.name]})
        print(json.dumps({'int8':source['id'],'ticks':right,'total':len(ts)}),flush=True)
    tokens=np.concatenate(chunks)
    np.savez_compressed(folder/'tokens.npz',timestamps=ts,tokens=tokens)
    report={'arm':'int8','recordingId':source['id'],'source':p.ident(source['video']),
        'reference':p.ident(entry['dinoPath']),'cache':p.ident(folder/'tokens.npz'),'chunks':refs,
        'driftVsFp32Reference':p.errors(tokens,reference),'timestampsIdentical':True,
        'wallSeconds':time.perf_counter()-start,'decodeOrReadSeconds':read_seconds,
        'embeddingSeconds':infer_seconds,'samples':len(ts),'batchSize':1,'resumedSamples':resumed,
        'graph':p.ident(graph),'executionAmendment':p.ident(root/'parallel-int8-amendment-v1.json')}
    p.write(folder/'report.json',report)
    return report


def main():
    root=p.DEFAULT;p.ensure_nas(root);parent=p.verify(root)
    amendment=root/'parallel-int8-amendment-v1.json'
    if not amendment.exists():
        p.write(amendment,{'kind':'unchanged-dino-int8-two-worker-execution',
            'source':p.ident(HERE),'parent':p.ident(root/'protocol.json'),
            'graph':p.ident(root/'encoder-dynamic-int8.onnx'),'sequentialRgb':p.ident(root/'sequential-rgb-amendment-v1/report.json'),
            'maximumWorkers':2,'ortThreadsPerWorker':2,'batchSize':1,
            'reason':'CPU-only parallel execution on disjoint recordings; original two-thread graph and batch semantics unchanged.',
            'rallyLabelsUsed':False,'timingCaveat':'Desktop concurrent with other experiments; resumed chunks do not contribute to the new embeddingSeconds.'})
    contract=p.read(amendment)
    if p.sha(HERE)!=contract['source']['sha256']: raise ValueError('Worker code changed')
    reports=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(worker,row) for row in parent['records']]
        for future in as_completed(tasks): reports.append(future.result())
    by_id={r['recordingId']:r for r in reports}
    ordered=[by_id[r['source']['id']] for r in parent['records']]
    p.write(root/'extraction-int8.json',{'protocol':p.ident(root/'protocol.json'),
        'qualification':p.ident(root/'qualification.json'),'executionAmendment':p.ident(amendment),
        'arm':'int8','records':ordered,'totalSamples':sum(r['samples'] for r in reports),
        'peakCudaAllocatedBytes':None})


if __name__=='__main__': main()
