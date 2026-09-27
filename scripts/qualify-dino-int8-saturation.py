#!/usr/bin/env python3
"""Label-blind, bounded U8S8 saturation diagnostic on fixed DINO pilot inputs.

The original INT8 configuration is retained. This adds two fixed engineering
alternatives, reports both, and selects by embedding RMSE without rally labels.
"""
from pathlib import Path
import argparse
import gzip
import importlib.util
import json
import time
import numpy as np

HERE=Path(__file__).resolve()
SPEC=importlib.util.spec_from_file_location('precision',HERE.with_name('evaluate-dino-precision.py'))
precision=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precision)


def main():
    from onnxruntime.quantization import quantize_dynamic, QuantType
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=precision.DEFAULT)
    args=parser.parse_args()
    root=args.root
    precision.ensure_nas(root)
    source=precision.verify(root)
    dest=root/'int8-saturation-diagnostic-v1'
    dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'protocol.json').exists():
        precision.write(dest/'protocol.json',{'kind':'label-blind-dino-int8-saturation-diagnostic',
            'parent':precision.ident(root/'protocol.json'), 'source':precision.ident(HERE),
            'qualification':precision.ident(root/'qualification.json'),
            'pilot':precision.ident(root/'pilot-rgb.npy'),
            'reference':precision.ident(root/'encoder-fp32.onnx'),
            'candidates':[{'name':'signed-reduced-range','weightType':'QInt8','reduceRange':True},
                          {'name':'unsigned-full-range','weightType':'QUInt8','reduceRange':False}],
            'fixed':'MatMulConstBOnly=True, per_channel=True, batch=1, same 32 fixed pilot inputs.',
            'selection':'Lowest global token RMSE against FP32 ONNX; original signed-full-range retained as reference. No rally labels, predictions or decoder outcomes.',
            'purpose':'Distinguish documented AVX2 U8S8 saturation from intrinsic 8-bit loss; neither alternative is a full INT8 graph.',
            'labelsUsed':False, 'referenceUrl':'https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html'})
    protocol=precision.read(dest/'protocol.json')
    for key in ('source','pilot','reference','qualification'):
        if precision.sha(protocol[key]['path'])!=protocol[key]['sha256']:
            raise ValueError('Frozen diagnostic input changed')
    rgb=np.load(root/'pilot-rgb.npy',allow_pickle=False)
    values=precision.normalized(rgb)
    fp=precision.session(root/'encoder-fp32.onnx')
    reference=np.concatenate([fp.run(None,{'image':values[i:i+1]})[0] for i in range(len(values))])
    rows=[]
    candidates=[{'name':'signed-full-range','path':root/'encoder-dynamic-int8.onnx'},
                *protocol['candidates']]
    for candidate in candidates:
        path=candidate.get('path',dest/(candidate['name']+'.onnx'))
        if not path.exists():
            quantize_dynamic(str(root/'encoder-fp32.onnx'),str(path),op_types_to_quantize=['MatMul'],
                per_channel=True,weight_type=getattr(QuantType,candidate['weightType']),
                reduce_range=candidate['reduceRange'],extra_options={'MatMulConstBOnly':True})
        session=precision.session(path)
        started=time.perf_counter()
        actual=np.concatenate([session.run(None,{'image':values[i:i+1]})[0] for i in range(len(values))])
        seconds=time.perf_counter()-started
        rows.append({'name':candidate['name'],'graph':precision.ident(path),
                     'gzipBytes':len(gzip.compress(path.read_bytes())),
                     'errors':precision.errors(actual,reference),
                     'all32WallSeconds':seconds,'labelsUsed':False})
        print(json.dumps(rows[-1]),flush=True)
    selected=min(rows,key=lambda r:(r['errors']['rootMeanSquareError'],r['name']))
    precision.write(dest/'report.json',{'protocol':precision.ident(dest/'protocol.json'),
        'candidates':rows,'selectedByEmbeddingRmse':selected['name'],'selectedGraph':selected['graph'],
        'labelsUsed':False,'rallyPerformanceEvaluated':False})


if __name__=='__main__':
    main()
