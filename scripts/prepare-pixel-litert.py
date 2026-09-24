#!/usr/bin/env python3
"""Qualify a second native GPU runtime on the same frozen image encoders."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
import litert_torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from analysis.mobile_visual_features import load_mobile_backbone
from analysis.dinov2_embeddings import _extract_feature_tokens

class MobileEncoder(torch.nn.Module):
    def __init__(self,features): super().__init__();self.features=features
    def forward(self,image,pool_weights):
        return torch.matmul(pool_weights.flatten(2),self.features(image).flatten(2).transpose(1,2))

class DinoEncoder(torch.nn.Module):
    def __init__(self,model):super().__init__();self.encoder=model
    def forward(self,image):return _extract_feature_tokens(self.encoder,torch,image,336)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--graphs',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    results=[];cases=[]
    for name in ['mobile','dino']:
        try:
            if name=='mobile':
                source=load_mobile_backbone(a.study_root/'2026-09-22-recognition/assets/mobilenet_v3_small-047dcff4.pth',device='cpu')
                model=MobileEncoder(source.model.features).eval()
                descriptions=[('image',(1,3,224,224)),('pool_weights',(1,4,7,7))]
            else:
                spec=importlib.util.spec_from_file_location('pixel_dino_source',REPO/'scripts/evaluate-dino-precision.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                model=DinoEncoder(module.backbone('cpu').model).eval();descriptions=[('image',(1,3,336,336))]
            values=tuple(torch.from_numpy(np.fromfile(a.graphs/f'{name}-encoder-{key}.f32',dtype='<f4').reshape(shape)) for key,shape in descriptions)
            converted=litert_torch.convert(model,values)
            filename=f'{name}-encoder-fp32.tflite';converted.export(str(a.graphs/filename))
            with torch.no_grad():expected=model(*values).numpy()
            actual=np.asarray(converted(*(v.numpy() for v in values)))
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
            results.append({'name':name,'status':'qualified','rmse':float(np.sqrt(np.mean((actual-expected)**2)))})
            for provider,half in [('cpu',False),('gpu',False),('gpu',True)]:
                cases.append({'id':f'{name}-litert-{provider}-'+('allow-fp16' if half else 'fp32'),
                    'model':filename,'runtime':'litert','provider':provider,'allowFp16':half,'warmup':3,'runs':10,
                    'inputs':[{'file':f'{name}-encoder-{key}.f32'} for key,_ in descriptions]})
        except Exception as error:
            import traceback;traceback.print_exc()
            results.append({'name':name,'status':'failed','error':str(error)})
        (a.graphs/'litert-qualification.json').write_text(json.dumps(results,indent=2))
        (a.graphs/'litert-plan.json').write_text(json.dumps({'runId':'pixel-litert-v1','cases':cases},indent=2))
        print(results[-1],flush=True)

if __name__=='__main__':main()
