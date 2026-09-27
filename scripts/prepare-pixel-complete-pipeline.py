"""Export frozen FP32 TCNs with real variable-length context and fold scalers."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.transfer_temporal_model import model_for
from analysis.recognition_temporal_model import RecognitionConfig, model_for as mobile_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    fits=a.study/'randomized-variants-v1/fits'
    for family,relative,epoch in [('dino','original-medium/dino-tcn/split-3407',30),('mobile','expanded-large/mobile-tcn/split-3407',15)]:
        folder=fits/relative
        model=model_for('dino_tcn') if family=='dino' else mobile_model(RecognitionConfig(**json.loads((folder/'fit-result.json').read_text())['config']))
        import hashlib
        weights=folder/f'temporal/weights-{epoch}.npz'
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        assert digest==json.loads((folder/'temporal/completed.json').read_text())['artifacts'][weights.name]
        with np.load(weights,allow_pickle=False) as archive:
            model.load_state_dict({k[7:]:torch.from_numpy(archive[k].copy()) for k in archive.files if k.startswith('model::')},strict=True)
            mean,scale=archive['mean'].tolist(),archive['scale'].tolist()
        model.eval();dim=3944 if family=='dino' else model.config.input_dimension
        path=a.output/f'{family}-tcn-dynamic-fp32.onnx'
        torch.onnx.export(model,torch.zeros(1,252,dim),str(path),input_names=['features'],output_names=['logits'],dynamic_axes={'features':{1:'ticks'},'logits':{1:'ticks'}},opset_version=17,dynamo=False)
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        checks=[]
        for ticks in (1,62,128,190,224,252):
            values=np.random.default_rng(ticks).normal(0,.2,(1,ticks,dim)).astype('float32')
            with torch.no_grad(): expected=model(torch.from_numpy(values)).numpy()
            actual=session.run(None,{'features':values})[0]
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-4)
            checks.append({'ticks':ticks,'maxAbsoluteError':float(np.max(np.abs(actual-expected)))})
        selection=next(x for x in json.loads((folder/'selection.json').read_text())['floors'] if x['floorPercent']==99)['selected']
        assert selection['epoch']==epoch
        (a.output/f'{family}-pipeline.json').write_text(json.dumps({'mean':mean,'scale':scale,'decoder':selection['decoder'],'weightsSha256':digest,'epoch':epoch,'family':family,'dynamicParity':checks},indent=2))
        print(family+' dynamic FP32 qualified',flush=True)
if __name__=='__main__':main()
