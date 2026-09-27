# Distilled MobileNetV3-Large in the browser

Both frozen FP32 selections completed actual-video browser feature generation, TCN inference, and serving-side/side-switch analysis on the full 1061.016489-second recording-044. The Android emulator was stopped before browser work; measured browser runs and CPU replay were serialized.

With the corrected AV option and qualified portable WebGPU export, highest recall: 5m25.9s all ready, 88.05% P_pad / 99.26% R_core / 93.32% F1_padP_coreR, 0 wholly missed human rallies; highest F1: 5m23.9s all ready, 97.57% P_pad / 96.68% R_core / 97.13% F1_padP_coreR, 1 wholly missed human rallies. These complete totals include all required feature generation and both score specialists. Current AV and production comparisons remain separate below; the production default has not changed.

This is a desktop Chrome experiment on one previously investigated recording, not a mobile-browser qualification or new training/selection result. Labels were hidden from inference. The saved 37-rally gold is manually reviewed export coverage with edits, not independently precise serve-contact/dead-ball annotation.

## Readiness times

| Scope / runtime / AV | Rallies ready s | Neural all ready s | Production shared-pass estimate s |
| --- | --- | --- | --- |
| 120s / WASM 1 thread / current AV | 34.304 | 47.957 | 26.522 |
| 120s / WASM 1 thread / area/nearest AV | 36.609 | 50.133 | 29.352 |
| 120s / WASM 4 threads / current AV | 32.885 | 46.466 | 26.536 |
| 120s / WASM 4 threads / area/nearest AV | 35.167 | 48.804 | 29.438 |
| 120s / WebGPU / current AV | 26.825 | 40.504 | 26.333 |
| 120s / WebGPU / area/nearest AV | 29.023 | 42.349 | 29.505 |
| Full / highest recall / current AV | 213.223 | 311.868 | 223.012 |
| Full / highest recall / area/nearest AV | 229.385 | 325.909 | 239.512 |
| Full / highest F1 / current AV | 214.381 | 308.205 | 221.904 |
| Full / highest F1 / area/nearest AV | 229.425 | 323.904 | 238.551 |

Each row includes all required feature generation for its proposal set. Neural all-ready adds production serve/state evidence and the real production score-specialist entry point, including its additional video sampling, serving-side features and side-switch features. Export rendering is excluded. Rows are single observations, with new browser profiles per run and no discarded pipeline warmup.

Totals are reconstructed stage sums: both AV variants reuse one embedding pass, and later passes have warmer models/filesystem caches. Production is a shared-AV, warm-runtime estimate; it is not a separately launched cold production app. Corrected AV follows current AV in each run and reuses loaded score models. Diagnostic artifact writes are excluded from browser readiness but recorded separately in JSON; Android benchmark timing includes diagnostic writes. Do not interpret desktop/emulator totals as a hardware speed comparison.

Measured runs use the actual file input and shipped browser File/Blob media path. The selected file is NAS-backed, without copying it to a local drive. Source I/O and browser decode are included; this is not phone-local-file latency. Earlier HTTP range-server experiments exposed media-fetch retries on the full source (`net::ERR_NO_BUFFER_SPACE` on two open-ended requests). They remain private diagnostics and are excluded from the primary timing matrix. The file path removes that artificial transport while preserving the same video and frozen models.

WASM 1 thread and WebGPU ran without cross-origin isolation, matching current hosting capability. WASM 4 threads required isolation headers. WebGPU used a real NVIDIA Ampere adapter; operator placement was not profiled, so mixed WASM/CPU execution remains possible. The pinned runtime is ONNX Runtime Web 1.22.0, Chrome 153.0.8010.48, Windows x64, Ryzen 9 5900X. The same all-backends runtime bundle was used for all measured profiles. WASM uses the original encoder graph; WebGPU uses the equivalent portable pooling graph described below. Timing compares these qualified deployment configurations, not an isolated execution-provider switch on byte-identical graphs.

## Full-video accuracy

Target padding is 2 seconds before and after. Positive gaps join only when strictly below 3 seconds. Ignored intervals are subtracted from model, human-core and padded-human unions without rejoining. R_core measures retained human-core time; it is not event recall. Wholly missed means zero retained nonignored human core after this padding and joining. Counts can therefore differ from event matching. Both selections retain their original 99% inner-calibration target; that target is not a promise of 99% recall on each recording or runtime.

Found-interval counts are raw model outputs, including intervals later excluded by ignored-time evaluation. Event denominators can be lower after censoring and overlap normalization.

| Model / AV | Found intervals | Wholly missed human rallies | P_pad | R_core | F1_padP_coreR | Export s |
| --- | --- | --- | --- | --- | --- | --- |
| Full / highest recall / current AV / neural | 39 | 0 | 83.74% | 99.18% | 90.81% | 539.241 |
| Full / highest recall / current AV / production | 59 | 1 | 66.88% | 97.45% | 79.32% | 682.544 |
| Full / highest recall / area/nearest AV / neural | 37 | 0 | 88.05% | 99.26% | 93.32% | 505.625 |
| Full / highest recall / area/nearest AV / production | 60 | 1 | 65.63% | 97.38% | 78.41% | 689.141 |
| Full / highest F1 / current AV / neural | 34 | 1 | 97.46% | 97.08% | 97.27% | 412.229 |
| Full / highest F1 / current AV / production | 59 | 1 | 66.88% | 97.45% | 79.32% | 682.544 |
| Full / highest F1 / area/nearest AV / neural | 34 | 1 | 97.57% | 96.68% | 97.13% | 403.500 |
| Full / highest F1 / area/nearest AV / production | 60 | 1 | 65.63% | 97.38% | 78.41% | 689.141 |
| Saved desktop / highest F1 | 34 | 1 | 97.69% | 96.69% | 97.19% | 402.000 |
| Saved desktop / highest recall | 37 | 0 | 88.12% | 99.26% | 93.36% | 506.625 |

| Model / AV | Matched events | Event precision | Event recall | Event F1 |
| --- | --- | --- | --- | --- |
| Full / highest recall / current AV / neural | 28 | 71.79% | 75.68% | 73.68% |
| Full / highest recall / current AV / production | 29 | 55.77% | 78.38% | 65.17% |
| Full / highest recall / area/nearest AV / neural | 31 | 83.78% | 83.78% | 83.78% |
| Full / highest recall / area/nearest AV / production | 28 | 53.85% | 75.68% | 62.92% |
| Full / highest F1 / current AV / neural | 29 | 85.29% | 78.38% | 81.69% |
| Full / highest F1 / current AV / production | 29 | 55.77% | 78.38% | 65.17% |
| Full / highest F1 / area/nearest AV / neural | 28 | 82.35% | 75.68% | 78.87% |
| Full / highest F1 / area/nearest AV / production | 28 | 53.85% | 75.68% | 62.92% |

Event metrics use chronological one-to-one matching of unpadded intervals at IoU >= 0.5. Human rallies touching ignored spans are censored from event matching, with their spans removed from event predictions. These guardrails do not replace the padded-export ranking metric.

Saved desktop rows reuse the same gold revision and selected encoder/TCN/decoder identities from the [visual repair evaluation](android-visual-feature-repair-evaluation.json). They are output references, not timed runs in this matrix. Browser/native/desktop decoding and audio operations can still differ. The existing native encoder also uses float bilinear preparation and unrounded FP32 tokens, while this browser experiment follows training's uint8 INTER_LINEAR and FP16 cache round trip. Cross-platform differences therefore cannot be assigned solely to AV features or video decoding. The JNI optimization preserves native outputs and does not change those embedding contracts.

Current AV is the shipped canvas/preceding-frame contract. Area/nearest AV is the existing opt-in correction. No production default, model threshold or selected checkpoint changed. Full-video accuracy has exact source alignment; the 120-second pilot uses a requested stream-copy offset and only approximate label alignment. The JSON includes all predicted intervals and wholly missed saved-human ranges. The AV comparison bundles resize, selected frames and row timestamps: legacy actual-PTS rows can causally align to an older image token, whereas corrected nominal 4 Hz rows remove that lag. Changes cannot be attributed to resize alone.

## Frozen-input timestamp diagnostic

After timed runs finished, the saved AV104 values and fusion/decoder time grids were crossed in a 2-by-2 CPU replay for each selection. Embeddings, quality values, fitted scalers, model weights and decoder settings remain fixed. Matching complete row grids are required; no interpolation, video decode or training occurs. The two observed diagonals reproduce their saved tensors, probabilities and boundaries.

| Selection | AV values | Fusion/decode grid | Kind | Rallies | Wholly missed | P_pad | R_core | F1_padP_coreR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recall | legacy | legacy | observed replay | 39 | 0 | 83.74% | 99.18% | 90.81% |
| recall | legacy | corrected | crossed | 36 | 0 | 87.68% | 99.26% | 93.11% |
| recall | corrected | legacy | crossed | 39 | 0 | 84.29% | 99.92% | 91.44% |
| recall | corrected | corrected | observed replay | 37 | 0 | 88.05% | 99.26% | 93.32% |
| f1 | legacy | legacy | observed replay | 34 | 1 | 97.46% | 97.08% | 97.27% |
| f1 | legacy | corrected | crossed | 34 | 1 | 97.64% | 96.68% | 97.16% |
| f1 | corrected | legacy | crossed | 34 | 1 | 97.34% | 97.08% | 97.21% |
| f1 | corrected | corrected | observed replay | 34 | 1 | 97.57% | 96.68% | 97.13% |

The recall diagnostic exposes a clock mismatch: at most 16.7ms of AV timestamp displacement causes 2,119/4,245 rows to select an older nominal 2 Hz embedding, with a 0.5s token lag. Keeping current AV values but using the corrected fusion/decode grid recovers most of the recall selection's F1 improvement. The highest-F1 choice does not improve under the bundled correction; the complete mode comparison must be retained rather than claiming a universal accuracy gain.

AV values combine selected-frame, resize and audio-feature-window effects: the original extraction timestamps also drive audio features. The time grid jointly changes causal embedding selection, quality age and decoded boundary timestamps. Crossed inputs are diagnostics, not an implemented or timed product pipeline, and this known recording must not select a new model or default.

## Full-video timing breakdown

| Model / AV | Setup s | Embedding pass s | AV/audio s | Round/fuse/TCN/decode s | Production evidence s | Score specialists s |
| --- | --- | --- | --- | --- | --- | --- |
| Full / highest recall / current AV | 1.220 | 113.668 | 97.153 | 1.182 | 0.216 | 98.428 |
| Full / highest recall / area/nearest AV | 1.220 | 113.668 | 113.617 | 0.879 | 0.170 | 96.354 |
| Full / highest F1 / current AV | 1.099 | 115.243 | 96.879 | 1.160 | 0.214 | 93.609 |
| Full / highest F1 / area/nearest AV | 1.099 | 115.243 | 112.225 | 0.858 | 0.168 | 94.311 |

| Selection | Source RGB readback s | 224 image preparation s | Encoder/readback s | Decode/other s |
| --- | --- | --- | --- | --- |
| Full / highest recall | 82.022 | 7.744 | 11.377 | 12.525 |
| Full / highest F1 | 83.239 | 7.866 | 12.032 | 12.107 |

The second table contains nested embedding counters; do not add them again to complete stage totals. Decode/other is a residual wall-time bucket. Source-resolution image readback is measured separately from the neural encoder, so an encoder-only throughput claim would omit substantial work.

## WebGPU pooling repair and qualification

The original encoder graph failed WebGPU numerical validation. Its terminal `bchw,brhw->brc` Einsum was reduced incorrectly by the pinned runtime. A tiny uniform-input probe returned 1 instead of 2; a signed-input probe matched squared products rather than the required weighted sum. Inspection of the [ORT 1.22.0 shader generator](https://github.com/microsoft/onnxruntime/blob/v1.22.0/js/web/lib/wasm/jsep/webgpu/ops/einsum.ts) shows operand multiplication emitted for both reduced symbols. This is a correctness issue, not a model-precision experiment.

A portable export replaces only that pooling operation with Reshape/Transpose/MatMul. Every trained initializer, TCN, scaler and decoder is unchanged; original graphs remain frozen. CPU comparison with the originals is exact on all eight image fixtures for both selections. Independent evaluation reconstructs the permitted rewrite from canonical bytes and checks the derived hash and preserved-initializer manifest.

The image contract follows training: source-resolution RGB, ties-to-even ROI rounding, uint8 OpenCV INTER_LINEAR 224 letterboxing, ImageNet normalization, four regional pools, then FP16 cache rounding. Inference remains FP32. Two-Hz tokens align causally to AV104 plus eight quality/age/availability values at 4 Hz. The two selections have distinct encoder weights and cannot share image embeddings.

They can still reuse AV/audio inputs and prepared image pixels when source, ROI, sampling window and preprocessing version match. Reusing those stages across selections was not timed here; each selection ran in a fresh browser profile.

Independent checks cover eight synthetic images per selected encoder, twenty decoder cases, two real-scaler fusion fixtures, and every finite FP16 value. For each actual-video run, five saved browser RGB snapshots are reprocessed in Python and replayed through the original CPU encoder; all actual fused rows are replayed through the original CPU TCN. Numerical errors, browser-versus-Python boundaries, and CPU-probability decoded boundaries are recorded in JSON. Snapshot equality validates operations on those pixels, not full source color-conversion parity. Nearest-frame selection additionally relies on the tested sampling helper; there is no independent full-source PTS inventory in this browser report.

All primary runs hash browser sources and decoder dependencies before navigation, disable HMR, and reject any source change afterward. The structured qualification evidence records fixture/probe hashes and maximum errors. Portable GPU FP32 outputs are numerically close to CPU; subsequent FP16 rounding can differ by a half-precision step near a rounding boundary and is not claimed byte-identical to CPU.

Serving-side and side-switch outputs are generated and checked for completeness, but their classification accuracy is not independently scored here. The editor lab still displays precomputed neural predictions; this harness does not add neural inference to the production upload flow.

## Download and feature storage

| Selection | Model/config/pool MB | Served ORT bundle MB |
| --- | --- | --- |
| Full / highest recall | 12.104 | 22.709 |
| Full / highest F1 | 12.104 | 22.709 |

Sizes are decimal, uncompressed bytes. The runtime is shared across selections, while each selection has its own encoder/head. Existing production OpenCV/audio/score assets are separately itemized in JSON and are not all incremental neural downloads. The experiment loads both OpenCV bundles: production prefers the worker, while neural image preparation forces the main bundle. That main OpenCV bundle is another 10.873 MB when it is not already cached or needed by other work. Thus the full observed production-asset total must not be treated as already loaded in every baseline. A WASM-only deployment could use a smaller bundle; that alternative bundle was not the measured configuration.

| Selection / AV | AV rows | AV104 MB | Fused3952 MB | Probabilities MB |
| --- | --- | --- | --- | --- |
| Full / highest recall / current AV | 4245 | 1.766 | 67.105 | 0.068 |
| Full / highest recall / area/nearest AV | 4245 | 1.766 | 67.105 | 0.068 |
| Full / highest F1 / current AV | 4245 | 1.766 | 67.105 | 0.068 |
| Full / highest F1 / area/nearest AV | 4245 | 1.766 | 67.105 | 0.068 |

The full source has 2,123 image samples: 32.609 MB of raw FP32 tokens, another 32.609 MB of FP16-rounded tokens stored in Float32Array, or 16.305 MB if packed as FP16. These are alternative/diagnostic representations; fused inputs duplicate the embeddings. They are not additive minimum memory. Sampled JavaScript heap excludes WASM linear memory, external buffers and GPU allocations, so it is not peak-memory qualification.

## Deployment implications

The portable pooling graph is a candidate for subsequent mobile-browser qualification, with WASM as a separately tested fallback. The original WebGPU graph must not be shipped on the strength of session creation or apparent speed. A physical phone still needs video-to-results timing, memory and output checks; the desktop NVIDIA GPU and NAS source cannot establish phone performance.

Further optimization should target video passes and source-resolution RGB readback while preserving the qualified image operations. A faster resize/color path needs the same golden-image and frozen-model checks before acceptance. Streaming temporal chunks could also avoid retaining duplicated full-video embedding/fused diagnostic arrays. Neither optimization nor a production-default change is part of this experiment.

## Required padding sensitivity

Every compared observation uses the same gold revision, ignored-range revision and strict join threshold. The declared 2-second case remains primary; no per-model best-padding selection is performed.

| Observation / model | Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 120s / WASM 1 thread / current AV / neural | 0s | 97.82% | 77.04% | 86.19% | 52.275 | 66.377 | -14.102 |
| 120s / WASM 1 thread / current AV / neural | 1s | 98.20% | 86.63% | 92.05% | 63.401 | 77.377 | -13.976 |
| 120s / WASM 1 thread / current AV / neural | 2s | 98.47% | 92.84% | 95.57% | 74.401 | 89.876 | -15.475 |
| 120s / WASM 1 thread / current AV / neural | 3s | 98.68% | 97.18% | 97.92% | 85.385 | 98.876 | -13.491 |
| 120s / WASM 1 thread / current AV / production | 0s | 80.98% | 73.84% | 77.25% | 60.524 | 66.377 | -5.853 |
| 120s / WASM 1 thread / current AV / production | 1s | 77.11% | 82.89% | 79.89% | 78.648 | 77.377 | +1.272 |
| 120s / WASM 1 thread / current AV / production | 2s | 81.74% | 87.41% | 84.48% | 87.648 | 89.876 | -2.228 |
| 120s / WASM 1 thread / current AV / production | 3s | 87.28% | 93.41% | 90.24% | 99.256 | 98.876 | +0.380 |
| 120s / WASM 1 thread / area/nearest AV / neural | 0s | 98.56% | 75.91% | 85.76% | 51.125 | 66.377 | -15.252 |
| 120s / WASM 1 thread / area/nearest AV / neural | 1s | 98.81% | 86.45% | 92.22% | 62.250 | 77.377 | -15.127 |
| 120s / WASM 1 thread / area/nearest AV / neural | 2s | 98.99% | 92.28% | 95.52% | 73.250 | 89.876 | -16.626 |
| 120s / WASM 1 thread / area/nearest AV / neural | 3s | 99.12% | 96.03% | 97.55% | 84.250 | 98.876 | -14.626 |
| 120s / WASM 1 thread / area/nearest AV / production | 0s | 77.60% | 75.55% | 76.56% | 64.625 | 66.377 | -1.752 |
| 120s / WASM 1 thread / area/nearest AV / production | 1s | 73.83% | 84.75% | 78.91% | 84.000 | 77.377 | +6.623 |
| 120s / WASM 1 thread / area/nearest AV / production | 2s | 76.84% | 89.28% | 82.59% | 96.000 | 89.876 | +6.124 |
| 120s / WASM 1 thread / area/nearest AV / production | 3s | 84.21% | 94.15% | 88.90% | 106.875 | 98.876 | +7.999 |
| 120s / WASM 4 threads / current AV / neural | 0s | 97.82% | 77.04% | 86.19% | 52.275 | 66.377 | -14.102 |
| 120s / WASM 4 threads / current AV / neural | 1s | 98.20% | 86.63% | 92.05% | 63.401 | 77.377 | -13.976 |
| 120s / WASM 4 threads / current AV / neural | 2s | 98.47% | 92.84% | 95.57% | 74.401 | 89.876 | -15.475 |
| 120s / WASM 4 threads / current AV / neural | 3s | 98.68% | 97.18% | 97.92% | 85.385 | 98.876 | -13.491 |
| 120s / WASM 4 threads / current AV / production | 0s | 80.98% | 73.84% | 77.25% | 60.524 | 66.377 | -5.853 |
| 120s / WASM 4 threads / current AV / production | 1s | 77.11% | 82.89% | 79.89% | 78.648 | 77.377 | +1.272 |
| 120s / WASM 4 threads / current AV / production | 2s | 81.74% | 87.41% | 84.48% | 87.648 | 89.876 | -2.228 |
| 120s / WASM 4 threads / current AV / production | 3s | 87.28% | 93.41% | 90.24% | 99.256 | 98.876 | +0.380 |
| 120s / WASM 4 threads / area/nearest AV / neural | 0s | 98.56% | 75.91% | 85.76% | 51.125 | 66.377 | -15.252 |
| 120s / WASM 4 threads / area/nearest AV / neural | 1s | 98.81% | 86.45% | 92.22% | 62.250 | 77.377 | -15.127 |
| 120s / WASM 4 threads / area/nearest AV / neural | 2s | 98.99% | 92.28% | 95.52% | 73.250 | 89.876 | -16.626 |
| 120s / WASM 4 threads / area/nearest AV / neural | 3s | 99.12% | 96.03% | 97.55% | 84.250 | 98.876 | -14.626 |
| 120s / WASM 4 threads / area/nearest AV / production | 0s | 77.60% | 75.55% | 76.56% | 64.625 | 66.377 | -1.752 |
| 120s / WASM 4 threads / area/nearest AV / production | 1s | 73.83% | 84.75% | 78.91% | 84.000 | 77.377 | +6.623 |
| 120s / WASM 4 threads / area/nearest AV / production | 2s | 76.84% | 89.28% | 82.59% | 96.000 | 89.876 | +6.124 |
| 120s / WASM 4 threads / area/nearest AV / production | 3s | 84.21% | 94.15% | 88.90% | 106.875 | 98.876 | +7.999 |
| 120s / WebGPU / current AV / neural | 0s | 97.82% | 77.04% | 86.19% | 52.275 | 66.377 | -14.102 |
| 120s / WebGPU / current AV / neural | 1s | 98.20% | 86.63% | 92.05% | 63.401 | 77.377 | -13.976 |
| 120s / WebGPU / current AV / neural | 2s | 98.47% | 92.84% | 95.57% | 74.401 | 89.876 | -15.475 |
| 120s / WebGPU / current AV / neural | 3s | 98.68% | 97.18% | 97.92% | 85.385 | 98.876 | -13.491 |
| 120s / WebGPU / current AV / production | 0s | 80.98% | 73.84% | 77.25% | 60.524 | 66.377 | -5.853 |
| 120s / WebGPU / current AV / production | 1s | 77.11% | 82.89% | 79.89% | 78.648 | 77.377 | +1.272 |
| 120s / WebGPU / current AV / production | 2s | 81.74% | 87.41% | 84.48% | 87.648 | 89.876 | -2.228 |
| 120s / WebGPU / current AV / production | 3s | 87.28% | 93.41% | 90.24% | 99.256 | 98.876 | +0.380 |
| 120s / WebGPU / area/nearest AV / neural | 0s | 98.56% | 75.91% | 85.76% | 51.125 | 66.377 | -15.252 |
| 120s / WebGPU / area/nearest AV / neural | 1s | 98.81% | 86.45% | 92.22% | 62.250 | 77.377 | -15.127 |
| 120s / WebGPU / area/nearest AV / neural | 2s | 98.99% | 92.28% | 95.52% | 73.250 | 89.876 | -16.626 |
| 120s / WebGPU / area/nearest AV / neural | 3s | 99.12% | 96.03% | 97.55% | 84.250 | 98.876 | -14.626 |
| 120s / WebGPU / area/nearest AV / production | 0s | 77.60% | 75.55% | 76.56% | 64.625 | 66.377 | -1.752 |
| 120s / WebGPU / area/nearest AV / production | 1s | 73.83% | 84.75% | 78.91% | 84.000 | 77.377 | +6.623 |
| 120s / WebGPU / area/nearest AV / production | 2s | 76.84% | 89.28% | 82.59% | 96.000 | 89.876 | +6.124 |
| 120s / WebGPU / area/nearest AV / production | 3s | 84.21% | 94.15% | 88.90% | 106.875 | 98.876 | +7.999 |
| Full / highest recall / current AV / neural | 0s | 77.54% | 94.28% | 85.09% | 390.492 | 321.146 | +69.346 |
| Full / highest recall / current AV / neural | 1s | 81.68% | 98.33% | 89.23% | 463.742 | 397.146 | +66.596 |
| Full / highest recall / current AV / neural | 2s | 83.74% | 99.18% | 90.81% | 539.241 | 470.645 | +68.596 |
| Full / highest recall / current AV / neural | 3s | 86.87% | 99.49% | 92.76% | 607.239 | 544.644 | +62.595 |
| Full / highest recall / current AV / production | 0s | 62.39% | 97.07% | 75.96% | 499.630 | 321.146 | +178.484 |
| Full / highest recall / current AV / production | 1s | 64.71% | 97.14% | 77.68% | 594.877 | 397.146 | +197.731 |
| Full / highest recall / current AV / production | 2s | 66.88% | 97.45% | 79.32% | 682.544 | 470.645 | +211.899 |
| Full / highest recall / current AV / production | 3s | 71.36% | 97.77% | 82.50% | 747.788 | 544.644 | +203.144 |
| Full / highest recall / area/nearest AV / neural | 0s | 82.18% | 92.54% | 87.05% | 361.625 | 321.146 | +40.479 |
| Full / highest recall / area/nearest AV / neural | 1s | 86.06% | 98.24% | 91.75% | 433.625 | 397.146 | +36.479 |
| Full / highest recall / area/nearest AV / neural | 2s | 88.05% | 99.26% | 93.32% | 505.625 | 470.645 | +34.980 |
| Full / highest recall / area/nearest AV / neural | 3s | 89.40% | 99.57% | 94.21% | 585.500 | 544.644 | +40.856 |
| Full / highest recall / area/nearest AV / production | 0s | 61.08% | 95.46% | 74.50% | 501.891 | 321.146 | +180.745 |
| Full / highest recall / area/nearest AV / production | 1s | 63.02% | 96.83% | 76.35% | 603.516 | 397.146 | +206.371 |
| Full / highest recall / area/nearest AV / production | 2s | 65.63% | 97.38% | 78.41% | 689.141 | 470.645 | +218.497 |
| Full / highest recall / area/nearest AV / production | 3s | 69.74% | 97.69% | 81.38% | 762.766 | 544.644 | +218.123 |
| Full / highest F1 / current AV / neural | 0s | 93.32% | 80.27% | 86.30% | 276.229 | 321.146 | -44.917 |
| Full / highest F1 / current AV / neural | 1s | 96.38% | 93.47% | 94.90% | 344.229 | 397.146 | -52.916 |
| Full / highest F1 / current AV / neural | 2s | 97.46% | 97.08% | 97.27% | 412.229 | 470.645 | -58.415 |
| Full / highest F1 / current AV / neural | 3s | 98.65% | 97.47% | 98.06% | 480.229 | 544.644 | -64.415 |
| Full / highest F1 / current AV / production | 0s | 62.39% | 97.07% | 75.96% | 499.630 | 321.146 | +178.484 |
| Full / highest F1 / current AV / production | 1s | 64.71% | 97.14% | 77.68% | 594.877 | 397.146 | +197.731 |
| Full / highest F1 / current AV / production | 2s | 66.88% | 97.45% | 79.32% | 682.544 | 470.645 | +211.899 |
| Full / highest F1 / current AV / production | 3s | 71.36% | 97.77% | 82.50% | 747.788 | 544.644 | +203.144 |
| Full / highest F1 / area/nearest AV / neural | 0s | 93.35% | 77.76% | 84.84% | 267.500 | 321.146 | -53.646 |
| Full / highest F1 / area/nearest AV / neural | 1s | 96.49% | 91.99% | 94.19% | 335.500 | 397.146 | -61.646 |
| Full / highest F1 / area/nearest AV / neural | 2s | 97.57% | 96.68% | 97.13% | 403.500 | 470.645 | -67.145 |
| Full / highest F1 / area/nearest AV / neural | 3s | 98.77% | 97.32% | 98.04% | 471.500 | 544.644 | -73.144 |
| Full / highest F1 / area/nearest AV / production | 0s | 61.08% | 95.46% | 74.50% | 501.891 | 321.146 | +180.745 |
| Full / highest F1 / area/nearest AV / production | 1s | 63.02% | 96.83% | 76.35% | 603.516 | 397.146 | +206.371 |
| Full / highest F1 / area/nearest AV / production | 2s | 65.63% | 97.38% | 78.41% | 689.141 | 470.645 | +218.497 |
| Full / highest F1 / area/nearest AV / production | 3s | 69.74% | 97.69% | 81.38% | 762.766 | 544.644 | +218.123 |
| Saved desktop / highest F1 | 0s | 93.50% | 77.45% | 84.72% | 266.000 | 321.146 | -55.146 |
| Saved desktop / highest F1 | 1s | 96.62% | 91.84% | 94.17% | 334.000 | 397.146 | -63.146 |
| Saved desktop / highest F1 | 2s | 97.69% | 96.69% | 97.19% | 402.000 | 470.645 | -68.645 |
| Saved desktop / highest F1 | 3s | 98.88% | 97.39% | 98.13% | 470.000 | 544.644 | -74.644 |
| Saved desktop / highest recall | 0s | 82.30% | 92.93% | 87.29% | 362.625 | 321.146 | +41.479 |
| Saved desktop / highest recall | 1s | 86.15% | 98.24% | 91.80% | 434.625 | 397.146 | +37.479 |
| Saved desktop / highest recall | 2s | 88.12% | 99.26% | 93.36% | 506.625 | 470.645 | +35.980 |
| Saved desktop / highest recall | 3s | 89.46% | 99.57% | 94.24% | 586.250 | 544.644 | +41.606 |
| Frozen replay / recall / AV=legacy, time=legacy | 0s | 77.54% | 94.28% | 85.09% | 390.492 | 321.146 | +69.346 |
| Frozen replay / recall / AV=legacy, time=legacy | 1s | 81.68% | 98.33% | 89.23% | 463.742 | 397.146 | +66.596 |
| Frozen replay / recall / AV=legacy, time=legacy | 2s | 83.74% | 99.18% | 90.81% | 539.241 | 470.645 | +68.596 |
| Frozen replay / recall / AV=legacy, time=legacy | 3s | 86.87% | 99.49% | 92.76% | 607.239 | 544.644 | +62.595 |
| Frozen replay / recall / AV=legacy, time=corrected | 0s | 82.15% | 92.92% | 87.21% | 363.250 | 321.146 | +42.104 |
| Frozen replay / recall / AV=legacy, time=corrected | 1s | 86.02% | 98.29% | 91.75% | 435.250 | 397.146 | +38.104 |
| Frozen replay / recall / AV=legacy, time=corrected | 2s | 87.68% | 99.26% | 93.11% | 510.000 | 470.645 | +39.355 |
| Frozen replay / recall / AV=legacy, time=corrected | 3s | 89.40% | 99.57% | 94.21% | 586.375 | 544.644 | +41.731 |
| Frozen replay / recall / AV=corrected, time=legacy | 0s | 78.39% | 94.94% | 85.87% | 388.976 | 321.146 | +67.830 |
| Frozen replay / recall / AV=corrected, time=legacy | 1s | 82.36% | 99.14% | 89.98% | 462.476 | 397.146 | +65.330 |
| Frozen replay / recall / AV=corrected, time=legacy | 2s | 84.29% | 99.92% | 91.44% | 538.224 | 470.645 | +67.580 |
| Frozen replay / recall / AV=corrected, time=legacy | 3s | 87.05% | 100.00% | 93.08% | 610.720 | 544.644 | +66.076 |
| Frozen replay / recall / AV=corrected, time=corrected | 0s | 82.18% | 92.54% | 87.05% | 361.625 | 321.146 | +40.479 |
| Frozen replay / recall / AV=corrected, time=corrected | 1s | 86.06% | 98.24% | 91.75% | 433.625 | 397.146 | +36.479 |
| Frozen replay / recall / AV=corrected, time=corrected | 2s | 88.05% | 99.26% | 93.32% | 505.625 | 470.645 | +34.980 |
| Frozen replay / recall / AV=corrected, time=corrected | 3s | 89.40% | 99.57% | 94.21% | 585.500 | 544.644 | +40.856 |
| Frozen replay / f1 / AV=legacy, time=legacy | 0s | 93.32% | 80.27% | 86.30% | 276.229 | 321.146 | -44.917 |
| Frozen replay / f1 / AV=legacy, time=legacy | 1s | 96.38% | 93.47% | 94.90% | 344.229 | 397.146 | -52.916 |
| Frozen replay / f1 / AV=legacy, time=legacy | 2s | 97.46% | 97.08% | 97.27% | 412.229 | 470.645 | -58.415 |
| Frozen replay / f1 / AV=legacy, time=legacy | 3s | 98.65% | 97.47% | 98.06% | 480.229 | 544.644 | -64.415 |
| Frozen replay / f1 / AV=legacy, time=corrected | 0s | 93.46% | 78.07% | 85.08% | 268.250 | 321.146 | -52.896 |
| Frozen replay / f1 / AV=legacy, time=corrected | 1s | 96.57% | 92.53% | 94.50% | 336.250 | 397.146 | -60.896 |
| Frozen replay / f1 / AV=legacy, time=corrected | 2s | 97.64% | 96.68% | 97.16% | 404.250 | 470.645 | -66.395 |
| Frozen replay / f1 / AV=legacy, time=corrected | 3s | 98.83% | 97.32% | 98.07% | 472.250 | 544.644 | -72.394 |
| Frozen replay / f1 / AV=corrected, time=legacy | 0s | 93.12% | 79.95% | 86.04% | 275.729 | 321.146 | -45.417 |
| Frozen replay / f1 / AV=corrected, time=legacy | 1s | 96.23% | 93.47% | 94.83% | 343.729 | 397.146 | -53.416 |
| Frozen replay / f1 / AV=corrected, time=legacy | 2s | 97.34% | 97.08% | 97.21% | 411.729 | 470.645 | -58.915 |
| Frozen replay / f1 / AV=corrected, time=legacy | 3s | 98.55% | 97.47% | 98.01% | 479.729 | 544.644 | -64.914 |
| Frozen replay / f1 / AV=corrected, time=corrected | 0s | 93.35% | 77.76% | 84.84% | 267.500 | 321.146 | -53.646 |
| Frozen replay / f1 / AV=corrected, time=corrected | 1s | 96.49% | 91.99% | 94.19% | 335.500 | 397.146 | -61.646 |
| Frozen replay / f1 / AV=corrected, time=corrected | 2s | 97.57% | 96.68% | 97.13% | 403.500 | 470.645 | -67.145 |
| Frozen replay / f1 / AV=corrected, time=corrected | 3s | 98.77% | 97.32% | 98.04% | 471.500 | 544.644 | -73.144 |

## Reproduction and artifacts

Private inputs and raw receipts resolve through private-reference-0223. The [harness](../../scripts/benchmark-distilled-large-browser.mjs), [portable export preparer](../../scripts/prepare-distilled-browser-portable-pooling.py), [independent evaluator](../../scripts/evaluate-distilled-large-browser.py) and [report builder](../../scripts/report-distilled-large-browser.py) preserve indexed provenance. See the [structured report](distilled-large-browser-experiment.json), [earlier AV-only experiment](web-visual-preprocessing-validation.md), and [optimized Android full-video validation](android-native-area-full-video.md).
