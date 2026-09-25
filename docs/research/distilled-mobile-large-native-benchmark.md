# Distilled MobileNetV3-Large native benchmark

Both distilled selections completed the native pipeline. Highest-F1 / highest-recall all-ready times were **67.272s / 69.669s** for the two-minute excerpt (three-run medians), and **712.924s / 734.295s** for the complete recording (one run each). These totals include all feature generation and both score specialists, before export rendering.

Physical Pixel 10 Pro, FP32 ONNX Runtime CPU with four threads, hardware MediaCodec decoding. Both frozen distilled selections are measured independently from video through rally, serving-side and side-switch results. DINO and the training projection are absent from inference.

Target export padding is 2s each side; positive gaps strictly under 3s join. Ignored intervals are removed after joining, without rejoining across them. Recall is retained saved human core time.

## Two-minute excerpt

| Model | Measured runs | AV video | Audio | Embedding pass | Rallies ready | Score specialists | All ready | Rallies |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Production ensemble | 3 | 22.579s | 3.620s | 0.000s | 26.557s | 16.692s | 43.142s | 6, 6, 6 |
| Frozen Large (highest recall) | 3 | 22.596s | 3.434s | 26.491s | 52.697s | 17.734s | 70.431s | 7, 7, 7 |
| Distilled Large (highest F1) | 3 | 22.332s | 3.453s | 26.452s | 53.047s | 14.225s | 67.272s | 7, 7, 7 |
| Distilled Large (highest recall) | 3 | 22.211s | 3.700s | 26.266s | 52.520s | 17.105s | 69.669s | 7, 7, 7 |

Pilot values are medians of three measured passes following one full warmup per model. All use the same excerpt and corrected APK; stage medians need not sum to median total.

## Full 17m41s recording

| Model | Measured runs | AV video | Audio | Embedding pass | Rallies ready | Score specialists | All ready | Rallies |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Production ensemble | 1 | 224.588s | 65.247s | 0.000s | 290.228s | 197.923s | 488.151s | 60 |
| Frozen Small (highest recall) | 1 | 223.378s | 80.012s | 252.164s | 556.871s | 167.559s | 724.430s | 40 |
| Frozen Large (highest recall) | 1 | 240.022s | 83.995s | 270.098s | 595.719s | 150.627s | 746.346s | 39 |
| Distilled Large (highest F1) | 1 | 249.653s | 72.565s | 280.779s | 604.768s | 108.156s | 712.924s | 31 |
| Distilled Large (highest recall) | 1 | 256.119s | 73.164s | 271.066s | 602.463s | 131.832s | 734.295s | 35 |

Full-video values are single observations. Production and frozen Small/Large are prior corrected full-frame controls on the identical APK and source, not simultaneous controlled thermal comparisons. The new full runs execute highest F1 then highest recall; OS media cache and run-order effects are not isolated.

## Full-video saved-human comparison

| Model | Found rallies | Wholly missed / human rallies | P_pad | R_core | F1_padP_coreR | Export (s) |
|---|---:|---:|---:|---:|---:|---:|
| Production ensemble | 60 | 1 / 37 | 66.21% | 96.01% | 78.37% | 650.391 |
| Frozen Small (highest recall) | 40 | 0 / 37 | 78.20% | 97.93% | 86.96% | 565.750 |
| Frozen Large (highest recall) | 39 | 0 / 37 | 86.84% | 96.99% | 91.63% | 492.250 |
| Distilled Large (highest F1) | 31 | 6 / 37 | 90.69% | 86.35% | 88.46% | 384.750 |
| Distilled Large (highest recall) | 35 | 2 / 37 | 87.66% | 93.84% | 90.65% | 471.375 |
| Distilled Large (highest F1) — saved desktop | 34 | 1 / 37 | 97.69% | 96.69% | 97.19% | 402.000 |
| Distilled Large (highest recall) — saved desktop | 37 | 0 / 37 | 88.12% | 99.26% | 93.36% | 506.625 |

Wholly missed means no nonignored saved human core retained after export padding and joining. The 37-rally gold snapshot contains manually reviewed export-derived boundaries with later edits, not independently precise serve-contact/dead-ball annotation. It is a different revision from the model-selection panel.

Saved desktop rows use these same gold labels, source bytes, selected encoder/TCN weights and decoder thresholds. They are existing outputs with hash-verified provenance and replayed boundary decoding, with no training or inference rerun and no desktop speed measurement. A native/desktop accuracy gap therefore cannot be attributed solely to a different checkpoint or threshold; extraction/runtime-input differences remain material.

## Bounded native-input diagnosis

The following highest-F1 checks replace selected blocks of already-saved native features with their desktop counterparts. Weights, normalization and decoder remain frozen. These are synthetic CPU replays, not new phone runs or usable corrected predictions.

| Saved inputs / replacement | Rallies | Wholly missed | P_pad | R_core | F1_padP_coreR |
|---|---:|---:|---:|---:|---:|
| Native observed | 31 | 6 | 90.69% | 86.35% | 88.46% |
| Desktop observed | 34 | 1 | 97.69% | 96.69% | 97.19% |
| Replace AV104 only | 34 | 1 | 97.80% | 95.75% | 96.77% |
| Replace embeddings only | 31 | 6 | 90.73% | 86.27% | 88.44% |
| Replace quality scalars only | 33 | 5 | 89.90% | 87.67% | 88.77% |
| Replace AV104 + quality; keep native embeddings | 34 | 1 | 97.69% | 96.69% | 97.19% |
| Replace AV audio only | 36 | 1 | 97.34% | 96.26% | 96.80% |

Replacing handcrafted AV inputs, particularly the audio block, recovers most of the gap; replacing image embeddings alone does not. AV104 plus quality scalars nearly reproduces desktop coverage while retaining native student embeddings. This localizes a material input-contract problem but does not identify its underlying audio decode/feature defect or qualify a fix. No feature change is installed by this benchmark.

## Image and score-stage decomposition

| Scope / model | Embedding decode / other | Image preparation | Encoder / readback | TCN | Specialist decode | Serving features / evaluation | Side-switch features / evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot / Production ensemble | 0.000s | 0.000s | 0.000s | 0.000s | 13.702s | 0.628s | 2.333s |
| pilot / Frozen Large (highest recall) | 18.087s | 5.471s | 2.935s | 0.034s | 14.198s | 0.724s | 2.807s |
| pilot / Distilled Large (highest F1) | 18.103s | 5.431s | 2.924s | 0.034s | 10.641s | 0.728s | 2.724s |
| pilot / Distilled Large (highest recall) | 17.944s | 5.429s | 2.937s | 0.034s | 13.639s | 0.723s | 2.778s |
| full / Production ensemble | 0.000s | 0.000s | 0.000s | 0.000s | 163.404s | 6.256s | 28.165s |
| full / Frozen Small (highest recall) | 187.632s | 50.894s | 13.638s | 0.131s | 143.697s | 4.406s | 19.312s |
| full / Frozen Large (highest recall) | 193.098s | 49.838s | 27.162s | 0.226s | 128.755s | 4.099s | 17.750s |
| full / Distilled Large (highest F1) | 202.423s | 51.714s | 26.642s | 0.239s | 90.824s | 3.412s | 13.842s |
| full / Distilled Large (highest recall) | 193.822s | 50.815s | 26.428s | 0.222s | 111.269s | 3.789s | 16.674s |

Embedding pass includes a separate decode pass in full. Decode/other is a residual wall-time bucket, not isolated decoder time. Specialist workloads use each model's actual rally boundaries; their cost is measured separately. Do not add nested or overlapping profile timings to wall-clock totals.

## Full-video feature storage

| Model | AV104 calculated | AV520 calculated | Tokens measured | Fused features measured | Probabilities measured | Specialist payloads measured |
|---|---:|---:|---:|---:|---:|---:|
| Production ensemble | 1.766 MB | 8.830 MB | n/a | n/a | n/a | 0.130 MB |
| Frozen Small (highest recall) | 1.766 MB | 8.830 MB | 19.566 MB | 41.024 MB | 0.068 MB | 0.088 MB |
| Frozen Large (highest recall) | 1.766 MB | 8.830 MB | 32.609 MB | 67.105 MB | 0.068 MB | 0.085 MB |
| Distilled Large (highest F1) | 1.766 MB | 8.830 MB | 32.609 MB | 67.105 MB | 0.068 MB | 0.067 MB |
| Distilled Large (highest recall) | 1.766 MB | 8.830 MB | 32.609 MB | 67.105 MB | 0.068 MB | 0.076 MB |

Decimal MB. AV arrays were not saved: sizes are calculated from actual row counts. Tokens are duplicated in fused features; these diagnostics are neither additive minimum memory nor measured peak RAM. Timed diagnostic writes add conservative overhead.

## Required full-video padding sensitivities

| Model | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production ensemble | 0s | 61.37% | 84.37% | 71.05% | 441.516 | 321.146 | +120.370 |
| Production ensemble | 1s | 64.04% | 94.07% | 76.20% | 557.016 | 397.146 | +159.871 |
| Production ensemble | 2s | 66.21% | 96.01% | 78.37% | 650.391 | 470.645 | +179.747 |
| Production ensemble | 3s | 70.23% | 96.80% | 81.40% | 725.516 | 544.644 | +180.873 |
| Frozen Small (highest recall) | 0s | 70.73% | 91.32% | 79.72% | 414.625 | 321.146 | +93.479 |
| Frozen Small (highest recall) | 1s | 75.44% | 96.09% | 84.52% | 490.125 | 397.146 | +92.979 |
| Frozen Small (highest recall) | 2s | 78.20% | 97.93% | 86.96% | 565.750 | 470.645 | +95.105 |
| Frozen Small (highest recall) | 3s | 82.34% | 99.18% | 89.98% | 633.125 | 544.644 | +88.481 |
| Frozen Large (highest recall) | 0s | 81.48% | 87.03% | 84.16% | 343.000 | 321.146 | +21.854 |
| Frozen Large (highest recall) | 1s | 85.25% | 94.05% | 89.43% | 417.000 | 397.146 | +19.854 |
| Frozen Large (highest recall) | 2s | 86.84% | 96.99% | 91.63% | 492.250 | 470.645 | +21.605 |
| Frozen Large (highest recall) | 3s | 89.06% | 98.78% | 93.67% | 566.500 | 544.644 | +21.856 |
| Distilled Large (highest F1) | 0s | 85.26% | 69.96% | 76.85% | 263.500 | 321.146 | -57.646 |
| Distilled Large (highest F1) | 1s | 88.69% | 80.97% | 84.66% | 323.500 | 397.146 | -73.646 |
| Distilled Large (highest F1) | 2s | 90.69% | 86.35% | 88.46% | 384.750 | 470.645 | -85.895 |
| Distilled Large (highest F1) | 3s | 93.26% | 88.76% | 90.96% | 442.750 | 544.644 | -101.894 |
| Distilled Large (highest recall) | 0s | 80.50% | 85.07% | 82.73% | 339.375 | 321.146 | +18.229 |
| Distilled Large (highest recall) | 1s | 85.16% | 91.38% | 88.16% | 405.375 | 397.146 | +8.229 |
| Distilled Large (highest recall) | 2s | 87.66% | 93.84% | 90.65% | 471.375 | 470.645 | +0.730 |
| Distilled Large (highest recall) | 3s | 88.94% | 95.36% | 92.04% | 546.000 | 544.644 | +1.356 |
| Distilled Large (highest F1) — saved desktop | 0s | 93.50% | 77.45% | 84.72% | 266.000 | 321.146 | -55.146 |
| Distilled Large (highest F1) — saved desktop | 1s | 96.62% | 91.84% | 94.17% | 334.000 | 397.146 | -63.146 |
| Distilled Large (highest F1) — saved desktop | 2s | 97.69% | 96.69% | 97.19% | 402.000 | 470.645 | -68.645 |
| Distilled Large (highest F1) — saved desktop | 3s | 98.88% | 97.39% | 98.13% | 470.000 | 544.644 | -74.644 |
| Distilled Large (highest recall) — saved desktop | 0s | 82.30% | 92.93% | 87.29% | 362.625 | 321.146 | +41.479 |
| Distilled Large (highest recall) — saved desktop | 1s | 86.15% | 98.24% | 91.80% | 434.625 | 397.146 | +37.479 |
| Distilled Large (highest recall) — saved desktop | 2s | 88.12% | 99.26% | 93.36% | 506.625 | 470.645 | +35.980 |
| Distilled Large (highest recall) — saved desktop | 3s | 89.46% | 99.57% | 94.24% | 586.250 | 544.644 | +41.606 |

## Observed variability and thermal state

| Scope / model | All-ready range | Thermal start/end by measured run |
|---|---:|---|
| pilot / Production ensemble | 42.752–43.627s | 0/0, 0/0, 0/0 |
| pilot / Frozen Large (highest recall) | 69.965–70.633s | 0/0, 0/0, 0/0 |
| pilot / Distilled Large (highest F1) | 66.730–67.606s | 0/0, 0/0, 0/0 |
| pilot / Distilled Large (highest recall) | 68.940–72.501s | 0/0, 0/0, 0/0 |
| full / Production ensemble | 488.151–488.151s | 0/0 |
| full / Frozen Small (highest recall) | 724.430–724.430s | 0/0 |
| full / Frozen Large (highest recall) | 746.346–746.346s | 0/0 |
| full / Distilled Large (highest F1) | 712.924–712.924s | 0/0 |
| full / Distilled Large (highest recall) | 734.295–734.295s | 0/0 |

Android thermal status 0 means none; 1 means light. Values are observations, not a controlled cooling or battery experiment. A locked/dozing initial pilot attempt was stopped before any completed case, preserved separately and excluded. The reported sequence began again with the phone unlocked.

## Limits and evidence

- Every measurement bypasses AV caches. It includes all required features and both score specialists; app launch, transfer, installation, collection and export rendering are excluded.
- Neural rally decisions use AV104 and their own encoder/TCN. Serving-side still requires the two production serve heads; side-switch requires production rally/dead-state evidence. Those auxiliary computations are included.
- Real-device fused-tensor temporal and decoder replay must pass before publication. Native image/PTS/AV extraction is still not qualified against desktop; prior residual AV drift remains a limitation, documented in the [previous native comparison](mobile-tcn-follow-up.md). This is native CPU timing, not GPU or browser timing.
- Pilot labels in the JSON are approximate diagnostics: source labels are shifted by the requested 180s stream-copy offset and clipped to 120s, without frame-accurate remux alignment. They are not the headline accuracy comparison. Full-video labels use exact source time.
- The JSON preserves per-run timings, thermal start/end status, rally boundaries, all four padding cases, wholly missed ranges, event matching, and measured file sizes. No source filenames or runtime locations are included.
- Accuracy on this one previously studied recording is a deployment diagnostic, not a new model selection or an independent generalization estimate.
- Recording `recording-044`; new measurements `private-reference-0223`; previous corrected neural/production controls `private-reference-0218` / `private-reference-0219`; same-gold comparison `private-reference-0221`.
- Gold SHA-256: `97ab7adce70f57ad521add3a18f18765abec61d36b2ba31cf0c739638164d94c`. Corrected benchmark APK SHA-256: `293fc06007a41b9cc58318c259fa90764c5ac93171e5098f284b18ad9ace5061`.

## Frozen model identities

| UI selection | Dataset draw | TCN epoch | Encoder bytes | TCN bytes |
|---|---:|---:|---:|---:|
| `neural-distilled-mobile-large-tcn-fp32` | 20260918 | 60 | 11,877,354 | 217,498 |
| `neural-distilled-mobile-large-tcn-fp32-high-recall` | 3407 | 15 | 11,877,358 | 217,502 |

The exact selected encoder/TCN weights, export qualification and per-run graph identities are frozen. No retraining, decoder adjustment or selection was performed for this benchmark. Raw graph sizes exclude the shared inference runtime and are not Play Store download sizes.

## Benchmark recording exposure

Source group `source-group-001` is identified through the external ledger. Membership is checked against the selected fit manifests and actual scored common panels.

| Model | Recording / source group in training | Recording / source group in calibration | Recording / source group in model selection |
|---|---|---|---|
| Distilled Large (highest F1) | no / no | no / no | no / no |
| Distilled Large (highest recall) | no / no | no / no | no / no |
| Frozen Large (highest recall) | no / no | no / no | no / no |
| Frozen Small (highest recall) | no / no | no / no | no / no |

The production lineage audit records no fit or calibration exposure for either the rally pipeline or the whole product. The benchmark recording and its source group were not used to fit, calibrate, or select either distilled choice or the frozen Small/Large choices. Reserved evaluation-group membership alone does not imply use in the actual exact-label selection panel. Historical diagnostic viewing and repeated pipeline fixes on this video remain separate exposure; these measurements are same-video deployment diagnostics, not a newly protected test.

Generated by [`report-distilled-mobile-large-native.py`](../../scripts/report-distilled-mobile-large-native.py). The [JSON companion](distilled-mobile-large-native-benchmark.json) contains the complete indexed observations.
