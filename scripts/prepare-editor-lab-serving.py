"""Prepare frozen production serve-head evidence for each comparison recording.

Locations resolve through the private ledger. No labels enter inference.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    index_path = Path(private_value('private-reference-0216'))
    replay = Path(private_value('private-reference-0061')) / 'production-replay-v1'
    work = index_path.parent / 'serving-v1' / 'work'
    work.mkdir(parents=True, exist_ok=True)
    sources = {row['id']: row for row in read(private_value('private-reference-0129'))['recordings']}
    jobs = []
    for entry in read(index_path)['recordings']:
        recording = read(index_path.parent / entry['file'])['recordings'][0]
        receipt_path = replay / (entry['id'] + '.json')
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt['labelsUsed'] is False and receipt['id'] == entry['id']
            assert abs(receipt['durationSeconds'] - recording['durationSeconds']) < .01
            assert sha(receipt['probabilities']['path']) == receipt['probabilities']['sha256']
            with np.load(receipt['probabilities']['path']) as z:
                times = z['times'].tolist()
                heads = {name: {'probabilities': z[key].tolist(), 'detections': []}
                         for name, key in [('allLabelsV2', 'v2_serve'), ('previousProduction', 'previous_serve')]}
            evidence_sha = receipt['probabilities']['sha256']
        else:
            # The beach population has frozen AV caches but no prior production replay.
            from analysis.config import FeatureConfig
            from analysis.features import FeatureSequence, VideoMetadata, contextualize
            source = sources[entry['id']]
            assert source['contentSha256'] == recording['contentSha256']
            ref = source['featureCaches']['audiovisual']
            assert sha(ref['path']) == ref['sha256']
            with np.load(ref['path']) as z:
                sequence = FeatureSequence(z['times'], z['values'], tuple(str(v) for v in z['names']),
                    VideoMetadata(**json.loads(str(z['metadata_json'].item()))))
            config = FeatureConfig.from_dict(read(REPO/'prod/public/runtime/model-9c92b8e9333f.json')['featureConfig'])
            contextual, names = contextualize(sequence, config)
            payload = {'times': sequence.times.tolist(), 'values': contextual.ravel().tolist(),
                       'names': list(names), 'duration': recording['durationSeconds']}
            program = """
import fs from 'node:fs';
import {loadOnDeviceModelBundle,runOnDeviceModel} from './prod/src/lib/on-device/model.ts';
const input=JSON.parse(fs.readFileSync(0,'utf8')); const output={};
for(const [key,file] of [['allLabelsV2','model-1ca43e38eefc.json'],['previousProduction','model-9c92b8e9333f.json']]){
 const model=loadOnDeviceModelBundle(JSON.parse(fs.readFileSync('prod/public/runtime/'+file,'utf8')));
 if(JSON.stringify(model.featureNames)!==JSON.stringify(input.names)) throw Error('Production feature order differs');
 const result=runOnDeviceModel(model,Float64Array.from(input.times),Float32Array.from(input.values),input.duration);
 output[key]={probabilities:Array.from(result.probabilities.serve),detections:result.serves};
} process.stdout.write(JSON.stringify(output));
"""
            result = subprocess.run(['node', '--input-type=module', '-e', program], cwd=REPO,
                input=json.dumps(payload), text=True, capture_output=True, check=True)
            heads, times, evidence_sha = json.loads(result.stdout), sequence.times.tolist(), ref['sha256']
        output = dict(recordingId=entry['id'], contentSha256=recording['contentSha256'],
            duration=recording['durationSeconds'], times=times, serveOutputs=heads,
            evidenceSha256=evidence_sha, labelsUsed=False, roi=dict(x=0,y=0,width=1,height=1,label='Full frame'))
        (work/(entry['id']+'.input.json')).write_text(json.dumps(output, separators=(',', ':'), allow_nan=False))
        jobs.append(entry['id'])
    (work/'jobs.json').write_text(json.dumps(jobs))
    print(json.dumps({'preparedRecordings': len(jobs), 'labelsUsed': False}))

if __name__ == '__main__':
    main()
