# Model-feedback bundle

The production editor's **Share / download model feedback** action shares or writes
`<source>.model-feedback.json`. The file is intended to be useful for model iteration without
the raw recording while remaining directly alignable when the recording is available.

## Contents

The top-level format is identified by:

```json
{
  "schema": "volleycut-model-feedback",
  "schemaVersion": 3
}
```

The bundle contains:

- source file metadata, sampled fingerprint, media metadata, game window, feature ROI, and runtime
  variant;
- the base audiovisual feature matrix and source-relative timestamps used to construct the model
  inputs;
- the initial production-ensemble ranges and timestamped rally, serve, and dead-state probability
  traces (the backward-compatible primary traces are produced by `probabilityModelId`);
- both production components' serve-probability traces and decoded serve contacts under
  `initialInference.componentServeOutputs`;
- the frozen serving-side model identity, fingerprint, feature/anchor contracts, row-aligned
  raw `SERVSIDE237-FLIGHT` matrix, candidate verdicts, probabilities, review reasons, and both
  serve heads' evidence under `initialInference.servingSide` (the browser recomputes tied
  within-recording ranks from the complete raw candidate matrix before classification);
- the untouched raw ranges from both production models, plus the held suppression artifact
  identity, probability trace, decoded events, and policy-eligible suggestion spans;
- the full corrected editor ranges and ignored intervals;
- the selected suppression policy, explicit and dormant decisions, touched inferred ranges, the
  default whole-rally suppression scope plus any per-suggestion veto-region overrides, and the
  effective state and scope of every suggestion;
- the complete editable score-tracking state under `corrections.scoreTracking.state`, including
  team names, the enabled flag, surviving model and manual serve markers, original `modelSide`
  versus corrected `side`, replay flags, side switches, and persisted removed-model-marker IDs;
- score-excluded rally IDs and the derived final score/point history, including counted, ignored,
  and unresolved point outcomes; and
- explicit feedback labels, final padded/joined export intervals, and materialization provenance.

The label contract is deliberately simple:

- a disabled model range is a false positive;
- an included manually added range is a false negative;
- an enabled model range is a confirmed model range;
- a manually added range that was later disabled is retained under `discardedManualRanges` and is
  not treated as a training label.

Ignored intervals remain outside the evaluation/training universe. They must not be converted into
dead-time negatives.

## Numeric arrays

Large numeric arrays use base64-encoded little-endian bytes instead of decimal JSON numbers.
`dataType` is `float32` or `float64`, and `shape` describes the decoded row-major array.

Python decoding example:

```python
import base64
import json
from pathlib import Path

import numpy as np

bundle = json.loads(Path("match.model-feedback.json").read_text())

def decode_array(payload):
    dtype = {"float32": "<f4", "float64": "<f8"}[payload["dataType"]]
    return np.frombuffer(base64.b64decode(payload["data"]), dtype=dtype).reshape(payload["shape"])

times = decode_array(bundle["features"]["timestamps"])
features = decode_array(bundle["features"]["values"])
rally_probability = decode_array(bundle["initialInference"]["probabilities"]["rally"])
all_labels_serve_probability = decode_array(
    bundle["initialInference"]["componentServeOutputs"]["allLabelsV2"]["probabilities"]
)
previous_serve_contacts = bundle["initialInference"]["componentServeOutputs"][
    "previousProduction"
]["detections"]
serving_side_features = decode_array(
    bundle["initialInference"]["servingSide"]["features"]["values"]
)
serving_side_candidates = bundle["initialInference"]["servingSide"]["candidates"]
score_tracking = bundle["corrections"]["scoreTracking"]
```

## Pairing with raw video

All times use seconds from the beginning of the original source, even when only a marked game
window was analyzed. A raw video is the same source when its byte length and
`sampled-sha256-v1` fingerprint match `source.file`. The fingerprint is SHA-256 over an
eight-byte little-endian file length followed by the first and last 1 MiB (or the whole file when
it is at most 2 MiB).

The production app uses that fingerprint when reconnecting a source. This permits the same raw
video to be paired after a copy changes its filename or modification date. Consumers outside the
app can verify the same fingerprint and then seek the raw video at any feature or range timestamp;
the JSON intentionally references the source instead of embedding video bytes.

Projects created before schema version 1 feature retention can still export inference and
corrections. Such bundles have `features: null` and an explicit warning; rerunning inference from
the source creates a complete bundle. Older projects without a retained serving-side result export
`initialInference.servingSide: null`; their final score-marker corrections are still retained.

## Importing into the production app

Use **Import model feedback** on the new-project screen to restore a bundle as a separate local
project. The import validates and decodes the bundle, restores its original inference and editor
corrections, and does not rerun either model. Importing the same bundle again creates another
project instead of replacing the first import.

The restored project initially has no video because `videoBytesIncluded` is always false. Use the
editor's source reconnect action to select the original recording. VolleyCut verifies the source
metadata or sampled fingerprint before enabling playback and video export; reconnecting does not
replace the imported inference or corrections.

On Android, a bundle with a complete `features` payload is also restored into the app-private
native feature cache. The importer validates the shared base-feature names and timeline, splits the
frame and audio columns, reconstructs the contextual matrix, and retains the cache under the
imported project's source identity. A later Android re-export can therefore reproduce the feature
and probability payload and retain suppression analysis without decoding the source video again.
Bundles that explicitly contain `features: null` remain valid degraded imports and cannot recreate
data that was already absent from the imported file.
