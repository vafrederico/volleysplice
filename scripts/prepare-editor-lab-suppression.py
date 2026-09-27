"""Publish existing, label-blind production suppression evidence for editor combinations.

All input/output locations are explicit CLI arguments resolved outside Git. No
training, video decoding, threshold selection, or prediction changes occur here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-index', type=Path, required=True)
    parser.add_argument('--production-replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.production_replay / 'protocol.json'
    protocol = read(protocol_path)
    assert protocol['labelsUsedForInference'] is False and protocol['trainingPerformed'] is False
    assert protocol['assets']['suppression']['sha256'] == 'ef0ad4eb93fa61ce1d403f083d91f7578cf9ff0f31fac797fde9ab8b73f42794'
    records, missing = [], []
    for entry in read(args.comparison_index)['recordings']:
        receipt_path = args.production_replay / (entry['id'] + '.json')
        if not receipt_path.exists():
            missing.append(entry['id'])
            continue
        receipt = read(receipt_path)
        assert receipt['id'] == entry['id'] and receipt['labelsUsed'] is False
        assert receipt['protocol']['sha256'] == sha(protocol_path)
        recording = read(args.comparison_index.parent / entry['file'])['recordings'][0]
        assert recording['recordingId'] == entry['id']
        assert abs(recording['durationSeconds'] - receipt['durationSeconds']) < .01
        probabilities = receipt['probabilities']
        assert sha(probabilities['path']) == probabilities['sha256']
        replay = receipt['productReplay']
        with np.load(probabilities['path'], allow_pickle=False) as cache:
            times, scores = cache['times'], cache['suppression']
            assert times.shape == scores.shape and np.isfinite(scores).all()
            decoded = []
            for region in replay['suppressionDecoded']:
                selected = scores[(times >= region['start']) & (times < region['end'])]
                assert len(selected)
                decoded.append(dict(**region, score=float(selected.mean())))
        records.append(dict(recordingId=entry['id'], contentSha256=recording['contentSha256'],
            durationSeconds=recording['durationSeconds'], revision=sha(receipt_path),
            gated=replay['suggestions'], decoded=decoded,
            productionEvents=replay['unionWithConfidence']))
    output = dict(schemaVersion=1, kind='volleycut-editor-suppression-index', labelsUsed=False,
                  records=records, unavailableRecordingIds=missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, separators=(',', ':'), allow_nan=False)
    print(json.dumps(dict(recordings=len(records), unavailable=len(missing), trainingPerformed=False)))


if __name__ == '__main__':
    main()
