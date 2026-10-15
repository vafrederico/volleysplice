"""Independent FP32 CPU references for the published serving graphs (ORT 1.24.2)."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / 'models/serving-side/web-manifest.json').read_text(encoding='utf-8'))
rows = 3
windows = ((np.arange(rows * 13 * 3843, dtype=np.int32) % 101 - 50) / 100).astype(np.float32).reshape(rows, 13, 3843)
windows[:, :, 3840] = 1
windows[:, :, 3841] = (np.arange(13) - 6) / 6
windows[:, :, 3842] = 0
features = ((np.arange(rows * 249, dtype=np.int32) % 19 - 9) / 9).astype(np.float32).reshape(rows, 249)
result = {
    'schemaVersion': 1, 'rows': rows,
    'inputRecipe': 'windows=(i%101-50)/100; mask=1; offset=(frame-6)/6; residual=0; features=(i%19-9)/9',
    'variants': {},
}
for key, bundle in manifest['variants'].items():
    graph = root / 'prod/public/runtime/serving-models' / bundle['directory'] / bundle['file']['name']
    session = ort.InferenceSession(str(graph), providers=['CPUExecutionProvider'])
    logits = session.run(['logits'], {'windows': windows, 'features': features})[0]
    assert logits.shape == (rows, 3, 2)
    result['variants'][key] = {'modelId': bundle['id'], 'logits': logits.flatten().tolist()}
(root / 'ios/Tests/Fixtures/neural-serving-golden.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
