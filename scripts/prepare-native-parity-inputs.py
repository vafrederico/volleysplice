"""Copy frozen mobile graphs and bind pooling to this recording's desktop ROI."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_inputs as inputs
from analysis.mobile_visual_features import regional_pool_weights
from analysis.private_ledger import private_value


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ledger = read(os.environ.get('VOLLEYCUT_PRIVATE_LEDGER_POSIX') or os.environ['VOLLEYCUT_PRIVATE_LEDGER'])
    rid = next(k for k, v in ledger['originalToAlias'].items() if v == 'recording-044')
    large = Path(private_value('private-reference-0211'))
    baseline = Path(private_value('private-reference-0210')).parent / 'graphs'
    destination = Path(private_value('private-reference-0217')) / 'full-frame-graphs'
    plan = read(large / 'plan.json')
    records = inputs.manifest_rows(inputs.verified(plan['inferenceManifest']))[1]
    entries = inputs.feature_entries(records, inputs.verified(plan['features']))
    image = read(inputs.verified(entries[rid]['imageInput']))
    image_plan = read(inputs.verified(image['contract']['plan']))
    source = next(row for row in image_plan['records'] if row['id'] == rid)
    roi = source['roi']
    assert roi == dict(x=0, y=0, width=1, height=1), 'Full-frame diagnostic requires full-frame desktop inputs'
    with np.load(inputs.verified(image['arrays']['timing'])) as timing:
        boxes = timing['boxes']
        np.testing.assert_array_equal(boxes, np.broadcast_to(boxes[0], boxes.shape))
    weights = regional_pool_weights(boxes[:1], 7, 7)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for family, source in [('mobile', baseline), ('mobile-large', large / 'graphs')]:
        for suffix in ('-encoder-fp32.onnx', '-tcn-dynamic-fp32.onnx', '-pipeline.json'):
            name = family + suffix
            shutil.copyfile(source / name, destination / name)
            hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        name = family + '-encoder-pool_weights.f32'
        weights.astype('<f4').tofile(destination / name)
        hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    receipt = dict(recordingIndex='recording-044', roi=roi, contentBox=boxes[0].tolist(), hashes=hashes,
                   scope='Frozen graphs; corrected full-frame pool geometry only. No training or calibration.')
    (destination / 'input-contract.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
