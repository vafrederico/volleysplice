#!/usr/bin/env python3
"""Prepare two label-blind actual image-encoder fixtures for browser parity."""
from pathlib import Path
import importlib.util
import numpy as np

HERE=Path(__file__).resolve()
SPEC=importlib.util.spec_from_file_location('precision',HERE.with_name('evaluate-dino-precision.py'))
p=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)
root=p.DEFAULT
p.ensure_nas(root)
dest=root/'browser-fixtures-v1'; dest.mkdir(parents=True,exist_ok=True)
if not (dest/'protocol.json').exists():
    p.write(dest/'protocol.json',{'kind':'dino-encoder-browser-parity-v1','source':p.ident(HERE),
        'indices':[0,20], 'meaning':'First pilot frame of first grass and first indoor recording; no label selection.',
        'rtol':.0001,'atol':.0001,'runtime':'onnxruntime-web 1.22.0 WASM CPU, one thread',
        'phoneMeasured':False,'parent':p.ident(root/'qualification.json')})
protocol=p.read(dest/'protocol.json')
rgb=np.load(root/'pilot-rgb.npy',allow_pickle=False)
fixtures=[]
for index in protocol['indices']:
    x=p.normalized(rgb[index:index+1])
    input_path=dest/f'input-{index}.bin'; x.tofile(input_path)
    fixtures.append({'index':index,'input':p.ident(input_path),'shape':list(x.shape)})
models=[]
for name,path in [('fp32',root/'encoder-fp32.onnx'),('dynamic-int8',root/'encoder-dynamic-int8.onnx')]:
    s=p.session(path)
    rows=[]
    for f in fixtures:
        x=np.fromfile(f['input']['path'],dtype=np.float32).reshape(f['shape'])
        output=s.run(None,{'image':x})[0]
        expected=dest/f"expected-{name}-{f['index']}.bin"; output.tofile(expected)
        rows.append({**f,'expected':p.ident(expected),'outputShape':list(output.shape)})
    models.append({'name':name,'graph':p.ident(path),'cases':rows})
p.write(dest/'manifest.json',{'protocol':p.ident(dest/'protocol.json'),'models':models})
