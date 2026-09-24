# Small-TCN native missed-rally investigation

Recording: **recording-044**. Evidence bundle: **private-reference-0217**, Small diagnostics. No training, threshold selection, or further DINO work was performed.

The native miss at **09:22.496–09:28.746** is a feature-generation mismatch, not a different model selection or temporal-decoder error. The web editor shows frozen desktop inference; it does not regenerate this recording's features in the phone/browser.

Both paths use the Small-TCN highest-recall variant, expanded-large draw 3407, epoch 15, target 99%, with identical checkpoint SHA-256 `4f68de01e384c642e02acf1bad860802894854d97950c5b6215ab9b246b29558`. Decoder: 1s smoothing, 0.35 entry threshold, 0.25s minimum, boundary correction enabled. Their 4,245 timestamps match exactly.

Desktop ONNX replay of each saved feature tensor reproduced its corresponding saved probabilities within **4.18e-7 maximum absolute error**. Native inputs reproduce the miss; desktop inputs reproduce the web detection. No inference arithmetic or boundary-decoding discrepancy is needed to explain the result.

| Frozen model input | Highest smoothed live score in human rally | Detected core range nearby |
|---|---:|---|
| All native | 0.169 | None |
| Native, replacing only embeddings with desktop | 0.188 | None |
| Native, replacing AV104 with desktop | 0.759 | 09:23.125–09:27.625 |
| Native, replacing quality/PTS scalars with desktop | 0.550 | 09:22.375–09:24.875 |
| Native, replacing AV104 and quality/PTS; native embeddings retained | 0.877 | 09:22.375–09:28.375 |
| All desktop | 0.899 | 09:22.375–09:28.375 |

The threshold is 0.35. The AV104-plus-quality substitution restores the desktop boundary for this rally even with native embeddings. These are diagnostic counterfactuals, not new calibrated variants or deployment performance estimates.

The most consequential individually tested quality feature was **clipped pixel fraction**. Around the rally, native reported approximately **2.947%**, while the desktop cache averaged **0.0125%**. The training scaler maps the native value to its +10 clipping limit versus roughly -1.10 for desktop. Replacing this scalar alone yields 09:22.375–09:25.375 and 09:26.625–09:27.375; their normal 2s padding and under-3s gap joining cover the entire human rally. Replacing only the embedding vectors does not recover it.

The following input contracts differ:

| Input stage | Desktop training/editor features | Native benchmark | Repository evidence |
|---|---|---|---|
| ROI | Full frame for this recording | Indoor default `(0.03,0.12,0.94,0.86)` | `PipelineBenchmarkActivity.java:95`; `AnalysisEngine.java:172,696`; external frozen manifest |
| Color conversion | OpenCV/FFmpeg decoded BGR; source declares full-range BT.709 | Fixed limited-range BT.601 YUV coefficients; source color metadata not consulted | `VideoEncoderBenchmark.java:222`; `NativeVideoDecoder.java:960` |
| AV image reduction | OpenCV `INTER_AREA`, 192×108 | One source YUV sample per target pixel; no area averaging | `analysis/features.py:389`; `NativeVideoDecoder.java:1040` |
| Embedding image preparation | Native-source ROI, RGB bilinear resize, black letterbox; uint8 rounding | ROI YUV color conversion, float bilinear resize, black letterbox | `analysis/mobile_visual_features.py:100`; `VideoEncoderBenchmark.java:184` |
| Embedding frame choice | Nearest frame to 2Hz tick, earlier frame on ties | First frame at/after the tick, except terminal fallback | `analysis/mobile_visual_features.py:211`; `VideoEncoderBenchmark.java:155` |
| PTS scalar | Signed nearest-frame offset | Nonnegative first-after offset during normal sampling | Same selection code; `NeuralRallyPipeline.java:84` |
| Normalization | AV104 and eight scalars standardized with training mean/scale and clipped ±10 | Same formula and constants, but different source values | `analysis/neural_recognition_fit.py:41`; `NeuralRallyPipeline.java:26` |

Source locations: [native benchmark activity](../../android/app/src/pipelineBenchmark/java/com/volleycut/nativeanalysis/PipelineBenchmarkActivity.java), [analysis engine](../../android/app/src/main/java/com/volleycut/nativeanalysis/AnalysisEngine.java), [video encoder](../../android/neuralbenchmark/src/main/java/com/volleycut/neuralbenchmark/VideoEncoderBenchmark.java), [native AV decoder](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java), [neural input fusion](../../android/app/src/pipelineBenchmark/java/com/volleycut/nativeanalysis/NeuralRallyPipeline.java), [desktop AV features](../../analysis/features.py), [desktop visual features](../../analysis/mobile_visual_features.py), [training normalization](../../analysis/neural_recognition_fit.py). Line references describe the investigated prototype and may move in subsequent fixes.

The feature-time grid is identical; the decoded source frame selected for a tick is not. At the 566s tick, desktop uses PTS 565.994900 and native uses 566.011566. The offset feature's training standard deviation is floored at 0.0001, so these become -10 and +10 respectively. Correcting only this scalar in isolation worsened the local score; it is an interacting mismatch, not a sufficient explanation by itself.

Color-range evidence is strong. At approximately 566s, decoding the original YUV plane on desktop and applying the native limited-range formula reproduces clipped pixel fraction **2.958%**, close to the phone's **2.950%**. Keeping that formula but switching to full-frame ROI yields **3.111%**; ROI alone therefore does not repair this outlier. Full-range conversion with full-frame ROI yields **0.0248%**, close to the desktop cache's **0.0213%** at its nearby selected PTS. The source declares `yuvj420p`, `color_range=pc`, `color_space=bt709`.

This establishes a reproducible range-conversion mismatch. Matrix choice, decoder output metadata, RGB rounding, exact PTS, ROI, and AV downsampling still require a controlled corrected phone run before claiming full extraction parity. OpenCV's cached colors are not proven to follow every source BT.709 tag, so matching a metadata-correct implementation to the actual frozen training pixels must be checked, rather than assumed.

The original Android times remain valid measurements of the tested prototype. Its accuracy numbers should not be interpreted as intrinsic Small-TCN accuracy or as a direct reproduction of the editor's frozen desktop output.

Reproduce the feature substitutions with [the diagnostic script](../../scripts/diagnose-native-neural-features.py), resolving input locations through the external ledger and ignored environment configuration. Evidence includes the broad and narrow feature-substitution reports, a color-range diagnostic, and private diagnostic tensors in the indexed bundle. All output locations must remain on configured research storage.

## Implemented color correction; phone work stopped

The table and accuracy evidence above describe the original prototype. The native AV decoder, embedding decoder, and shared serve/side-switch frame decoder now use the same [YUV conversion helper](../../android/video-common/src/main/java/com/volleycut/video/YuvColorConversion.java). The conversion preserves packed RGB and existing RGBA-to-BGR ordering, handles full and limited ranges, and uses the selected BT.601 or BT.709 matrix. It also implements the BT.2020 non-constant-luminance matrix; it does not perform HDR tone mapping or gamut conversion.

For each sampled image, [the metadata resolver](../../android/video-common/src/main/java/com/volleycut/video/DecodedVideoColor.java) reads the actual decoded buffer's `MediaCodec.getOutputFormat(index)`. Decoder output takes precedence over extractor track metadata independently for range and standard. This matters because a decoder may change the range or matrix. A missing output field falls back to the corresponding extractor field; only when both are unspecified does the implementation assume BT.601 or limited range. Explicit unsupported metadata is rejected. These are the documented output-format and color-aspect APIs. ([Android MediaCodec](https://developer.android.com/reference/android/media/MediaCodec#getOutputFormat(int)), [Android MediaFormat](https://developer.android.com/reference/android/media/MediaFormat#KEY_COLOR_RANGE))

The embedding benchmark now records `decodedColorProfiles`: resolved standard, range, each field's source, and the first sample timestamp for each changed profile. Raw AV/context cache identity now includes `opencv-v2-decoded-yuv-color`; the shared specialist cache identity uses `shared-gap5-mediacodec-decoded-yuv-color-gray-bgr-v3`. Thus old feature caches are not reused for newly requested analysis. Existing saved edits are not rewritten.

The [pure JVM tests](../../android/app/src/test/java/com/volleycut/video/YuvColorConversionTest.java) cover neutral black/white/shadows, full- and limited-range primary color bars, RGB ordering, clipping, decoder-over-track precedence, partial metadata fallback, and unsupported metadata. All nine color tests pass. Both benchmark and normal-app builds pass; the normal app passes 156 unit tests with no failures or skipped tests. Its APK was verified to use the normal debug application ID, not the benchmark ID. Phone work is stopped at the user's request because the device is unavailable. The correction addresses declared color semantics, not proven equality with frozen OpenCV pixels. A controlled replay would still need to measure the missed rallies and remaining ROI, resize, source-frame selection, and rounding differences. No improved phone accuracy or timing is claimed.

Imported model-feedback bundles remain historical artifact replay. Their supplied features and specialist results are preserved for inspection and re-export; import does not prove that their producer used the corrected color pipeline. Imported raw features are isolated under a fresh synthetic source URI, retained as `featureCacheSource` after reconnecting the video. Fresh video analysis uses the actual source URI and the current extractor cache identity. This fix therefore preserves existing import/re-export behavior without treating imported measurements as validation of corrected native extraction.
