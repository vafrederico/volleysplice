# Distilled Large production bundles

The default rally model is **Distilled MobileNetV3-Large + TCN, highest F1**,
labeled **Balanced · BETA**. The alternative **highest recall**, labeled
**Maximum coverage · BETA**, is a separately selected, frozen model. **Legacy model**
selects the production ensemble. Each analysis runs only one selected rally
variant. Switching
selection switches the encoder, temporal head, 112-column normalization, and
decoder together. No retraining or threshold search happens in either app.

| Selection | Model ID | Selection draw / TCN epoch | Decoder smoothing / entry / minimum / boundaries |
| --- | --- | --- | --- |
| Highest F1 (default) | `distilled-large-f1-v1` | 20260918 / 60 | 1 s / 0.9 / 0.25 s / disabled |
| Highest recall | `distilled-large-recall-v1` | 3407 / 15 | 0.5 s / 0.2 / 1 s / enabled |

Both come from the existing target-99% calibration experiment. That calibration
target is not a guarantee of 99% recall on a new recording. Their fitting and
selection lineage remains in [MODELS.md](../../MODELS.md) and the
[completed study](../../docs/research/distilled-mobile-large-results.md). This
integration does not fit new weights or use evaluation videos for another selection.

The checked-in [Android manifest](android-manifest.json) and
[web manifest](web-manifest.json) pin every runtime asset's SHA-256 and byte count.
Each platform's two model payloads total approximately 24.2 MB uncompressed,
excluding the inference runtime. A browser needs only the selected bundle,
approximately 12.1 MB, plus ONNX Runtime. Android packages both for offline use.
Android also extracts a verified copy of a selected bundle into app-private
storage for file-backed inference. Temporary video embeddings are removed after
the run; selecting both models can retain approximately another 24.2 MB of
extracted model files. Browser embedding caches are local to the selected model,
source, crop, and game window.

The web encoder substitutes the qualified Reshape/Transpose/MatMul regional pool
for the original graph's Einsum. It preserves all trained initializers. Native
Android keeps the validated original graph. Both temporal heads run FP32. Web
embeddings retain the qualified FP16 cache rounding; native retains FP32
embeddings. These are explicit platform contracts, not interchangeable caches.
Neither DINO nor the training projection is distributed for inference.

## Prepare build inputs

The selected **web** payloads are checked in at
[`prod/public/runtime/rally-models`](../../prod/public/runtime/rally-models),
so `npm ci` and `npm run build` from `prod/` need no private assets directory.
The web build verifies these bytes against `web-manifest.json`, copies them into
the static output, and includes the pinned ONNX Runtime and license notices.
No private ledger, raw video, embeddings, or research receipts are served.

Android bundles and candidate releases are generated outside Git. Set the private
ledger environment as described in
[private research references](../../docs/research/private-research-ledger.md)
and set `VOLLEYCUT_NEURAL_ASSETS_DIR` to an external generated-output directory.
The source resolves through `private-reference-0223`. When its host spelling
differs, an ignored `VOLLEYCUT_DEVICE_ARTIFACT_ROOT` override supplies that same
indexed artifact root; the ledger entry remains mandatory.

```sh
python scripts/prepare-production-neural-assets.py
```

Keep `VOLLEYCUT_NEURAL_ASSETS_DIR` set for Android builds and bundle generation;
the production web build uses its checked-in release assets.
Its generated layout is:

```text
android/rally-models/manifest.json
android/rally-models/{recall,f1}/{encoder.onnx,temporal.onnx,pipeline.json}
web/rally-models/manifest.json
web/rally-models/{recall,f1}/{encoder.onnx,temporal.onnx,pipeline.json}
```

The preparation command validates the immutable selection, strips research-only
configuration fields, and checks the resulting bytes against the public
manifests. Builds reject absent, incomplete, or mismatched bundles. They must
not silently ship an ensemble under a neural model name. The private ledger,
labels, source media, and research receipts are never application assets.

Public model updates require new IDs/manifests and requalification. Replacing
only a graph while keeping an old model/cache identity is unsupported. Same-file
AV features can be reused only with identical ROI, game window, audio contract,
and AV sampling contract. Recall and F1 image embeddings are incompatible because
their distilled encoder weights differ.
