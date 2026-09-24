"""Export and qualify the selected Large FP32 encoder/TCN for native timing."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import mobilenet_v3_large
from analysis.neural_generalization_experiment import load_checkpoint
from analysis.recognition_temporal_model import RecognitionConfig
from analysis.neural_context_development import identity


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.features=model.features

    def forward(self,image,pool_weights):
        spatial=self.features(image)
        return torch.einsum('bchw,brhw->brc',spatial,pool_weights)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--baseline-graphs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder-only',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
    checkpoint=a.experiment/'torch-cache/checkpoints/mobilenet_v3_large-8738ca79.pth'
    assert identity(checkpoint)['sha256'].startswith('8738ca79')
    model=mobilenet_v3_large(weights=None)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    encoder=Encoder(model).eval()
    weights=np.fromfile(a.baseline_graphs/'mobile-encoder-pool_weights.f32',dtype='<f4').reshape(1,4,7,7)
    # Identical content geometry to the existing benchmark source and ROI.
    image=np.fromfile(a.baseline_graphs/'mobile-encoder-image.f32',dtype='<f4').reshape(1,3,224,224)
    inputs={'image':image,'pool_weights':weights}
    path=a.output/'mobile-large-encoder-fp32.onnx'
    torch.onnx.export(encoder,tuple(torch.from_numpy(v) for v in inputs.values()),str(path),
        input_names=list(inputs),output_names=['tokens'],opset_version=17,dynamo=False)
    with torch.inference_mode(): expected=encoder(*(torch.from_numpy(v) for v in inputs.values())).numpy()
    actual=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']).run(None,inputs)[0]
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-4)
    shutil.copyfile(a.baseline_graphs/'mobile-encoder-pool_weights.f32',a.output/'mobile-large-encoder-pool_weights.f32')
    report=dict(encoderSha256=identity(path)['sha256'],checkpointSha256=identity(checkpoint)['sha256'],
                encoderBytes=path.stat().st_size,encoderMaxAbsoluteError=float(np.max(abs(actual-expected))))
    if not a.encoder_only:
        result=json.loads((a.experiment/'evaluation.json').read_text())
        chosen=next(r for r in result['selected'] if r['mode']=='recall')
        fit=a.experiment/'fits'/chosen['variant']/f"split-{chosen['draw']}"
        config=RecognitionConfig(**json.loads((fit/'selection.json').read_text())['config'])
        epoch=chosen['setting']['epoch']; checkpoint=fit/'temporal'/f'weights-{epoch}.npz'
        completed=json.loads((fit/'temporal/completed.json').read_text())
        assert identity(checkpoint)['sha256']==completed['artifacts'][checkpoint.name]
        temporal,mean,scale=load_checkpoint(checkpoint,config,'cpu')
        graph=a.output/'mobile-large-tcn-dynamic-fp32.onnx'
        torch.onnx.export(temporal,torch.zeros(1,252,config.input_dimension),str(graph),
            input_names=['features'],output_names=['logits'],dynamic_axes={'features':{1:'ticks'},'logits':{1:'ticks'}},opset_version=17,dynamo=False)
        runtime=ort.InferenceSession(str(graph),providers=['CPUExecutionProvider']); checks=[]
        for ticks in (1,62,128,190,252):
            x=np.random.default_rng(ticks).normal(0,.2,(1,ticks,config.input_dimension)).astype(np.float32)
            with torch.inference_mode(): y=temporal(torch.from_numpy(x)).numpy()
            z=runtime.run(None,{'features':x})[0]
            np.testing.assert_allclose(z,y,atol=2e-5,rtol=2e-4)
            checks.append(dict(ticks=ticks,maxAbsoluteError=float(np.max(abs(z-y)))))
        (a.output/'mobile-large-pipeline.json').write_text(json.dumps(dict(mean=mean.tolist(),scale=scale.tolist(),
            decoder=chosen['setting']['decoder'],weightsSha256=identity(checkpoint)['sha256'],epoch=epoch,
            family='mobile-large',tokenDimension=3840,config=config.__dict__,dynamicParity=checks),indent=2))
        report.update(temporalBytes=graph.stat().st_size,temporalChecks=checks,selected={k:chosen[k] for k in ('variant','draw','floorPercent','mode')})
    (a.output/'mobile-large-qualification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
