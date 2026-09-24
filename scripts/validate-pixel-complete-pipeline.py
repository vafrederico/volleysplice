"""Check real native fused-input temporal outputs and boundaries against Python."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.neural_development import decode

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True)
    a=p.parse_args();report=json.loads((a.results/'result.json').read_text());checks=[]
    cases={r['id']:r for r in json.loads((a.results/'pipeline-plan.json').read_text())['cases']}
    for row in report['results']:
        if 'neural' not in row:continue
        meta=row['neural'];family=cases[row['id']]['family'];n=meta['featureRows'];dim=meta['featureDimension']
        values=np.fromfile(a.results/(row['id']+'-features.f32'),dtype='<f4').reshape(n,dim)
        actual=np.fromfile(a.results/(row['id']+'-probabilities.f32'),dtype='<f4').reshape(n,4)
        assert np.isfinite(values).all() and np.isfinite(actual).all()
        expected=np.zeros_like(actual);session=ort.InferenceSession(str(a.graphs/(family+'-tcn-dynamic-fp32.onnx')),providers=['CPUExecutionProvider'])
        for core in range(0,n,128):
            end=min(n,core+128);left=max(0,core-62);right=min(n,end+62)
            logits=session.run(None,{'features':values[None,left:right]})[0][0]
            expected[core:end]=(1/(1+np.exp(-logits)))[core-left:end-left]
        np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
        example=SimpleNamespace(times=np.asarray(meta['times']),valid=np.ones(n,bool),duration=meta['video']['seconds'])
        decoded=decode(example,actual,meta['decoder']);bounds=np.asarray([(r.start,r.end) for r in decoded]).reshape(-1,2)
        native=np.asarray([(r['start'],r['end']) for r in row['rallies']]).reshape(-1,2)
        np.testing.assert_allclose(native,bounds,atol=1e-8,rtol=0)
        checks.append({'id':row['id'],'rows':n,'rallies':len(decoded),'maxProbabilityError':float(np.max(np.abs(actual-expected))),'decoderParity':True})
    if not checks:raise RuntimeError('No successful neural outputs to validate')
    output={'passed':True,'scope':'Real device fused tensors: native temporal inference and rally decoder vs desktop. Pixel/PTS/AV extraction parity is separate.','checks':checks}
    (a.results/'temporal-decoder-parity.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':main()
