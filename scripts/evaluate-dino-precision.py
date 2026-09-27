#!/usr/bin/env python3
"""Bounded encoder precision ablation; immutable NAS artifacts, frozen TCN heads.

The FP32 reference caches and every historic trained checkpoint are read-only.
Dynamic INT8 covers constant-weight MatMul only, with per-channel signed weights
and dynamic unsigned activations; it is deliberately not called full INT8.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis.dinov2_embeddings import (load_pinned_dinov2, _extract_feature_tokens,
    _crop_frame, letterbox_frame, IMAGENET_RGB_MEAN, IMAGENET_RGB_STD)

NAS = Path(private_value('private-reference-0057'))
OLD = NAS/'2026-09-19-short-boost-transfer'
DEFAULT = NAS/'2026-09-23-recall-distillation/dino-precision-v1'
COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
ASSETS = Path(private_value('private-reference-0058'))
SEEDS = (3407, 1729, 20260918)


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            d.update(b)
    return d.hexdigest()


def ident(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'sizeBytes': p.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def ensure_nas(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(NAS) or not Path('/mnt/freenas').is_mount():
        raise ValueError('Outputs must be on the mounted NAS')
    resolved.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        p = resolved/'runtime'/key.lower()
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def normalized(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (336, 336, 3):
        raise ValueError('Expected uint8 batch of 336px RGB letterboxes')
    x = rgb.astype(np.float32)/np.float32(255.)
    return np.ascontiguousarray(((x-IMAGENET_RGB_MEAN)/IMAGENET_RGB_STD).transpose(0,3,1,2))


def errors(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid precision comparison')
    delta = a-b
    flat_a, flat_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cosine = (flat_a*flat_b).sum(-1)/(np.linalg.norm(flat_a, axis=-1)*np.linalg.norm(flat_b, axis=-1)).clip(1e-30)
    return {'maximumAbsoluteError': float(np.abs(delta).max()),
            'meanAbsoluteError': float(np.abs(delta).mean()),
            'rootMeanSquareError': float(np.sqrt(np.square(delta).mean())),
            'meanCosineSimilarity': float(cosine.mean()), 'minimumCosineSimilarity': float(cosine.min())}


def backbone(device):
    return load_pinned_dinov2(ASSETS/f'dinov2-{COMMIT}', repository_commit=COMMIT,
        checkpoint=ASSETS/'dinov2_vits14_pretrain.pth', checkpoint_sha256=WEIGHT_SHA, device=device)


def register(output):
    if (output/'protocol.json').exists():
        return verify(output)
    manifest = read(OLD/'dino-manifest.json')
    sources = {r['id']: r for r in read(NAS/'2026-09-22-recognition/inputs/extraction-manifest.json')['records']}
    records = []
    for item in manifest['records']:
        if item['tier'] != 'exact':
            continue
        src = sources[item['recordingId']]
        records.append({'source': src, 'dino': item})
    if len(records) != 8:
        raise ValueError('Expected eight exact development recordings')
    dependencies = [Path(__file__), REPO/'analysis/dinov2_embeddings.py',
        REPO/'analysis/neural_expanded_development.py', REPO/'analysis/neural_development.py',
        REPO/'analysis/transfer_temporal_model.py', REPO/'analysis/compact_temporal_model.py',
        REPO/'analysis/neural_evaluation.py', REPO/'analysis/crop_evaluation.py']
    source_refs = []
    for source in dependencies:
        target = output/'registered-sources'/source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_refs.append({**ident(source), 'archive': ident(target)})
    references = [ident(OLD/'dino-manifest.json'), ident(OLD/'manifest-pts-v1.json'),
        ident(output.parent/'protocol.md'), ident(ASSETS/'dinov2_vits14_pretrain.pth')]
    references += [ident(OLD/'study'/f'result-reviewed_export-dino_tcn-short_boost-{s}.json') for s in SEEDS]
    for seed in SEEDS:
        control = read(OLD/'study'/f'result-reviewed_export-dino_tcn-short_boost-{seed}.json')
        for index, selected in enumerate(control['selections']):
            folder = Path(control['origin']['fitRoot'])/f'outer-{index}'/'refit'
            references += [ident(folder/f"weights-{selected['epoch']}.npz"), ident(folder/'completed.json')]
    contract = {'kind': 'dino-encoder-precision-v1', 'sources': source_refs, 'references': references,
        'records': records, 'encoder': {'commit': COMMIT, 'checkpointSha256': WEIGHT_SHA,
            'inputSize': 336, 'sampleHz': 4, 'tokens': [10,384]},
        'arms': ['reference-fp32-stored-fp16', 'cuda-fp16-stored-fp16', 'onnx-dynamic-int8-stored-fp16'],
        'quantization': {'opTypes': ['MatMul'], 'MatMulConstBOnly': True, 'perChannel': True,
            'weightType': 'QInt8', 'activationType': 'dynamic-QUInt8', 'reduceRange': False,
            'calibrationData': None, 'allOtherOps': 'FP32'},
        'comparison': 'Three fixed historical seeds, exact source-held folds/checkpoints/decoders. No precision-specific tuning.',
        'additional99Replay': 'Only if an independently selected 99%-recall control is supplied; identical caches, no retuning.',
        'preprocessing': 'Historical exact-source sampling: round(4Hz timestamp * source fps), same OpenCV source frame, same ROI and 336px RGB letterbox and ImageNet normalization.',
        'cache': 'Every arm uses float16 token storage. New RGB chunks retain exact uint8 model inputs for separate INT8 CPU replay.',
        'engineeringGate': 'FP32 ONNX vs fresh FP32 PyTorch allclose atol=1e-4 rtol=1e-4 on 32 stratified source frames; same-frame fresh FP32 vs reference cache allclose atol=0.02 rtol=0.002 after float16 storage.',
        'pilot': 'Four evenly spaced timestamps per exact recording, independent of labels; 32 total.',
        'targetPaddingSeconds': 2, 'paddingSensitivity': [0,1,2,3], 'joinGapStrictlyLessThanSeconds': 3,
        'ignoredPolicy': 'Same ignored ranges removed after padded joining; temporal context resets at ignored gaps.',
        'protectedTestOpened': False, 'beachIncluded': False, 'trainingPerformed': False,
        'deploymentLimits': 'Desktop CUDA and CPU measured; no phone or full app benchmark. Mixed INT8 ONNX is a portability candidate, not confirmed mobile acceleration.'}
    write(output/'protocol.json', contract)
    return contract


def verify(output):
    p = read(output/'protocol.json')
    for ref in p['references'] + p['sources']:
        if sha(ref['path']) != ref['sha256']:
            raise ValueError(f"Frozen input changed: {ref['path']}")
    return p


def decode_rgb(source, indexes, timestamps):
    import cv2
    cap = cv2.VideoCapture(source['video'])
    if not cap.isOpened():
        raise ValueError('Cannot open source')
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = tuple(source['roi'][k] for k in ('x','y','width','height')) if source.get('roi') else None
    rows = []
    try:
        for i in indexes:
            fi = min(max(0, total-1), max(0, int(round(float(timestamps[i])*fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode {source['id']} frame {fi}")
            rows.append(letterbox_frame(_crop_frame(frame, roi), 336))
    finally:
        cap.release()
    return np.stack(rows)


def session(path, threads=2):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])


def qualify(output):
    import torch
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    protocol = verify(output)
    torch.set_num_threads(2)
    model = backbone('cpu').model

    class Wrapper(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.encoder = source

        def forward(self, image):
            return _extract_feature_tokens(self.encoder, torch, image, 336)

    wrapper = Wrapper(model).eval()
    fp_path, q_path = output/'encoder-fp32.onnx', output/'encoder-dynamic-int8.onnx'
    if not fp_path.exists():
        torch.onnx.export(wrapper, torch.zeros(1,3,336,336), str(fp_path),
            input_names=['image'], output_names=['tokens'], opset_version=17,
            dynamic_axes={'image': {0:'batch'}, 'tokens': {0:'batch'}}, dynamo=False)
    if not q_path.exists():
        quantize_dynamic(str(fp_path), str(q_path), op_types_to_quantize=['MatMul'], per_channel=True,
            weight_type=QuantType.QInt8, extra_options={'MatMulConstBOnly': True}, reduce_range=False)
    fp, q = session(fp_path), session(q_path)
    rows, rgbs = [], []
    for row in protocol['records']:
        entry, src = row['dino'], row['source']
        with np.load(entry['dinoPath'], allow_pickle=False) as n:
            ts, old = n['timestamps'], n['tokens']
        indexes = np.linspace(0, len(ts)-1, 4, dtype=int)
        rgb = decode_rgb(src, indexes, ts)
        x = normalized(rgb)
        with torch.inference_mode():
            reference = wrapper(torch.from_numpy(x)).numpy()
        fp_out = np.concatenate([fp.run(None, {'image': x[i:i+1]})[0] for i in range(4)])
        q_out = np.concatenate([q.run(None, {'image': x[i:i+1]})[0] for i in range(4)])
        parity = np.allclose(fp_out, reference, atol=1e-4, rtol=1e-4)
        cache_parity = np.allclose(reference.astype(np.float16), old[indexes], atol=.02, rtol=.002)
        rows.append({'recordingId': src['id'], 'indexes': indexes.tolist(), 'timestamps': ts[indexes].tolist(),
            'onnxFp32VsTorchFp32': errors(fp_out, reference), 'onnxFp32ParityPassed': bool(parity),
            'freshFp32VsCache': errors(reference.astype(np.float16), old[indexes]),
            'freshCacheParityPassed': bool(cache_parity), 'dynamicInt8VsFp32': errors(q_out, fp_out)})
        rgbs.append(rgb)
    pilot = np.concatenate(rgbs)
    np.save(output/'pilot-rgb.npy', pilot)
    timing = {}
    for label, s in [('fp32', fp), ('dynamicInt8', q)]:
        x = normalized(pilot[:1])
        for _ in range(3):
            s.run(None, {'image': x})
        elapsed = []
        for _ in range(12):
            started = time.perf_counter()
            s.run(None, {'image': x})
            elapsed.append(time.perf_counter()-started)
        timing[label] = {'batchSize':1, 'cpuThreads':2, 'medianSeconds': float(np.median(elapsed)),
                         'p90Seconds': float(np.quantile(elapsed,.9))}
    graphs = {}
    for label, path in [('fp32', fp_path), ('dynamicInt8', q_path)]:
        graph = onnx.load(str(path))
        ops = Counter(n.op_type for n in graph.graph.node)
        arrays = [onnx.numpy_helper.to_array(i) for i in graph.graph.initializer]
        graphs[label] = {**ident(path), 'gzipBytes':len(gzip.compress(path.read_bytes())),
            'operators':dict(ops), 'initializerElementsByDtype':dict(Counter({dtype:sum(a.size for a in arrays if str(a.dtype)==dtype) for dtype in {str(a.dtype) for a in arrays}}))}
    passed = all(r['onnxFp32ParityPassed'] and r['freshCacheParityPassed'] for r in rows)
    write(output/'qualification.json', {'passed':passed, 'protocol':ident(output/'protocol.json'),
        'rows':rows, 'timing':timing, 'graphs':graphs, 'pilot':ident(output/'pilot-rgb.npy')})
    print(json.dumps({'qualificationPassed':passed, 'timing':timing, 'graphs':{k:{f:v[f] for f in ('sizeBytes','gzipBytes')} for k,v in graphs.items()}}), flush=True)
    if not passed:
        raise ValueError('FP32 engineering parity gate failed')


def extract(output, arm):
    import torch
    protocol, qualification = verify(output), read(output/'qualification.json')
    if not qualification['passed']:
        raise ValueError('Engineering parity gate is closed')
    torch.set_num_threads(2)
    if arm == 'fp16':
        model = backbone('cuda').model
        torch.cuda.reset_peak_memory_stats()
        timing = {}
        pilot = normalized(np.load(output/'pilot-rgb.npy',allow_pickle=False))
        outputs = {}
        for label in ('fp32','fp16'):
            if label=='fp16':
                model.half()
            dtype = torch.float32 if label=='fp32' else torch.float16
            x = torch.from_numpy(pilot[:1]).to('cuda',dtype=dtype)
            with torch.inference_mode():
                for _ in range(3):
                    _extract_feature_tokens(model,torch,x,336)
                torch.cuda.synchronize()
                times=[]
                for _ in range(12):
                    t=time.perf_counter()
                    _extract_feature_tokens(model,torch,x,336)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter()-t)
                outputs[label]=np.concatenate([_extract_feature_tokens(model,torch,
                    torch.from_numpy(pilot[i:i+8]).to('cuda',dtype=dtype),336).float().cpu().numpy()
                    for i in range(0,len(pilot),8)])
            timing[label]={'batchSize':1,'medianSeconds':float(np.median(times)),
                          'p90Seconds':float(np.quantile(times,.9))}
        half_path=output/'encoder-fp16-state.pth'
        if not half_path.exists():
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},half_path)
        if not (output/'cuda-pilot.json').exists():
            write(output/'cuda-pilot.json',{'timing':timing,'fp16VsFp32':errors(outputs['fp16'],outputs['fp32']),
                'fp16Checkpoint':ident(half_path),'peakCudaAllocatedBytes':torch.cuda.max_memory_allocated()})
        def infer(x):
            with torch.inference_mode():
                return _extract_feature_tokens(model, torch, torch.from_numpy(x).to('cuda', dtype=torch.float16), 336).float().cpu().numpy()
    else:
        qpath = output/'encoder-dynamic-int8.onnx'
        if sha(qpath) != qualification['graphs']['dynamicInt8']['sha256']:
            raise ValueError('Quantized graph changed')
        ort = session(qpath)
        def infer(x):
            # Dynamic scale depends on the batch. Fix B=1 for deployment-like semantics.
            return np.concatenate([ort.run(None, {'image': x[i:i+1]})[0] for i in range(len(x))])
    reports = []
    for row in protocol['records']:
        src, entry = row['source'], row['dino']
        destination = output/arm/src['id']
        destination.mkdir(parents=True, exist_ok=True)
        if (destination/'report.json').exists():
            reports.append(read(destination/'report.json'))
            continue
        if sha(entry['dinoPath']) != entry['dinoSha256'] or sha(src['video']) != src['contentSha256']:
            raise ValueError('Source identity changed')
        with np.load(entry['dinoPath'], allow_pickle=False) as n:
            ts, reference = n['timestamps'], n['tokens']
        chunks, chunk_refs, decode_s, infer_s = [], [], 0., 0.
        started = time.perf_counter()
        for left in range(0, len(ts), 128):
            right = min(left+128,len(ts))
            rgb_path = output/'rgb'/src['id']/f'{left:06d}.npy'
            tokens_path = destination/f'{left:06d}.npy'
            if tokens_path.exists():
                values = np.load(tokens_path, allow_pickle=False)
            else:
                t = time.perf_counter()
                if rgb_path.exists():
                    rgb = np.load(rgb_path, allow_pickle=False)
                elif arm == 'fp16':
                    rgb = decode_rgb(src, range(left,right), ts)
                    rgb_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(rgb_path, rgb)
                else:
                    raise ValueError('FP16/RGB extraction must complete first')
                decode_s += time.perf_counter()-t
                t = time.perf_counter()
                values = np.concatenate([infer(normalized(rgb[i:i+8])) for i in range(0,len(rgb),8)]).astype(np.float16)
                infer_s += time.perf_counter()-t
                if values.shape != (right-left,10,384) or not np.isfinite(values).all():
                    raise ValueError('Invalid generated embeddings')
                np.save(tokens_path, values)
            chunks.append(values)
            chunk_refs.append({'tokens': ident(tokens_path), 'rgb': ident(rgb_path)})
            print(json.dumps({'arm':arm,'recording':src['id'],'ticks':right,'total':len(ts)}), flush=True)
        values = np.concatenate(chunks)
        np.savez_compressed(destination/'tokens.npz', timestamps=ts, tokens=values)
        report = {'arm':arm, 'recordingId':src['id'], 'source':ident(src['video']),
            'reference':ident(entry['dinoPath']), 'cache':ident(destination/'tokens.npz'), 'chunks':chunk_refs,
            'driftVsFp32Reference':errors(values,reference), 'timestampsIdentical':True,
            'wallSeconds':time.perf_counter()-started, 'decodeOrReadSeconds':decode_s,
            'embeddingSeconds':infer_s, 'samples':len(ts), 'batchSize':8 if arm=='fp16' else 1}
        write(destination/'report.json', report)
        reports.append(report)
    payload = {'protocol':ident(output/'protocol.json'), 'qualification':ident(output/'qualification.json'),
        'arm':arm, 'records':reports, 'totalSamples':sum(r['samples'] for r in reports),
        'peakCudaAllocatedBytes':torch.cuda.max_memory_allocated() if arm=='fp16' else None}
    write(output/f'extraction-{arm}.json',payload)
    print(json.dumps({'arm':arm,'complete':True,'samples':payload['totalSamples']}), flush=True)


def evaluate(output, selection_root=None):
    import torch
    from analysis import neural_expanded_development as expanded
    from analysis.transfer_temporal_model import model_for
    from analysis.neural_evaluation import evaluate_predictions
    torch.set_num_threads(2)
    protocol = verify(output)
    examples = {r.example.id:r.example for r in expanded.load_data(OLD/'manifest-pts-v1.json')['exact']}
    tokens = {'fp32':{},'fp16':{},'int8':{}}
    for record in protocol['records']:
        key, entry = record['source']['id'], record['dino']
        e = examples[key]
        for arm in tokens:
            path = Path(entry['dinoPath']) if arm=='fp32' else output/arm/key/'tokens.npz'
            with np.load(path,allow_pickle=False) as n:
                ts,tok = n['timestamps'],n['tokens']
            if arm!='fp32':
                rep = read(output/arm/key/'report.json')
                if sha(path)!=rep['cache']['sha256']:
                    raise ValueError('Precision cache changed')
            right = np.clip(np.searchsorted(ts,e.times),0,len(ts)-1)
            left = np.maximum(0,right-1)
            nearest = np.where(abs(ts[left]-e.times)<=abs(ts[right]-e.times),left,right)
            if np.max(abs(ts[nearest]-e.times))>.125+1e-8:
                raise ValueError('Precision feature alignment differs')
            tokens[arm][key] = replace(e,values=np.concatenate((e.values,tok[nearest].reshape(len(e.times),-1)),axis=1).astype(np.float32))
    results = []
    for seed in SEEDS:
        control = read(OLD/'study'/f'result-reviewed_export-dino_tcn-short_boost-{seed}.json')
        if selection_root:
            raise NotImplementedError('External 99% schema must be explicitly bound before replay')
        rows = {arm:[] for arm in tokens}
        probabilities = {arm:{} for arm in tokens}
        refs = []
        for outer, selected in enumerate(control['selections']):
            group, epoch = selected['heldSourceGroup'],selected['epoch']
            checkpoint = Path(control['origin']['fitRoot'])/f'outer-{outer}'/'refit'/f'weights-{epoch}.npz'
            completed = read(checkpoint.parent/'completed.json')
            if group in completed['trainGroups']:
                raise ValueError('Held group leaked into checkpoint fitting')
            with np.load(checkpoint,allow_pickle=False) as n:
                mean,scale = n['mean'],n['scale']
                state = {k[len('model::'):]:torch.from_numpy(n[k].copy()) for k in n.files if k.startswith('model::')}
            model = model_for('dino_tcn')
            model.load_state_dict(state,strict=True)
            refs.append(ident(checkpoint))
            for arm in tokens:
                for e in tokens[arm].values():
                    if e.group!=group:
                        continue
                    p = expanded.predict(model,e,mean,scale,'dino_tcn','cpu')
                    probabilities[arm][e.id] = p
                    row = e.row(expanded.base.decode(e,p,selected['decoder']))
                    rows[arm].append({**row,**{k:[v.to_dict() for v in row[k]] for k in ('rallies','ignoredIntervals','predictions')}})
        for arm in tokens:
            name = f'result-{arm}-{seed}.json'
            result = {'arm':arm, 'seed':seed, 'control':ident(OLD/'study'/f'result-reviewed_export-dino_tcn-short_boost-{seed}.json'),
                'checkpointReferences':refs,'selections':control['selections'], 'predictions':rows[arm],
                'evaluation':evaluate_predictions(rows[arm]),
                'scoreDrift':{key:errors(value,probabilities['fp32'][key]) for key,value in probabilities[arm].items()}}
            if arm=='fp32':
                old_primary=control['evaluation']['primary']; new_primary=result['evaluation']['primary']
                if any(abs(new_primary[k]-old_primary[k])>1e-8 for k in ('P_pad','R_core','F1_padP_coreR')):
                    raise ValueError('FP32 CPU replay changed historical decoded metrics')
            np.savez_compressed(output/f'probabilities-{arm}-{seed}.npz',**probabilities[arm])
            write(output/name,result)
            results.append(ident(output/name))
            print(json.dumps({'arm':arm,'seed':seed,**result['evaluation']['primary']}),flush=True)
    write(output/'report.json',{'kind':'dino-precision-frozen-head-comparison','protocol':ident(output/'protocol.json'),
        'results':results,'protectedTestOpened':False,'trainingPerformed':False})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('register','qualify','fp16','int8','evaluate'))
    p.add_argument('--output',type=Path,default=DEFAULT)
    args=p.parse_args()
    ensure_nas(args.output)
    if args.phase=='register': register(args.output)
    elif args.phase=='qualify': qualify(args.output)
    elif args.phase in ('fp16','int8'): extract(args.output,args.phase)
    else: evaluate(args.output)


if __name__=='__main__':
    main()
