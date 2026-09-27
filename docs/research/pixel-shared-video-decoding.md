# Pixel shared-decoding Distilled Large benchmark

Physical Pixel 10 Pro over Tailscale ADB, using the frozen highest-recall Distilled MobileNetV3-Large + TCN at FP32. The native area-conversion optimization and shared AV/embedding decoder are enabled. Both serving-side and side-switch results are included.

| Window | Shared AV + embeddings s | Audio s | Rallies ready s | Score specialists s | All results s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 33.996 | 3.926 | 38.559 | 23.560 | 62.119 |
| full | 363.558 | 81.332 | 449.292 | 193.763 | 643.055 |

For historical same-phone context only:

| Window | Earlier corrected Java + independent decoding, all results s | Optimized, all results s |
| --- | ---: | ---: |
| short | 122.946 | 62.119 |
| full | 1497.437 | 643.055 |

These earlier observations have unmatched device conditions and differ in both area conversion and decode sharing. Their entire timing difference cannot be attributed to shared decoding alone.

## Outputs and accuracy

Both runs match the earlier corrected Pixel outputs exactly: selected frame timestamps, quality features, saved embeddings, fused features, probabilities, rally boundaries and confidences, production proposals, and both score-specialist feature matrices and decisions. Independent CPU temporal replay also reproduces every rally boundary. All 240 prepared-input pixel hashes on the phone match the validated emulator's short run exactly.

The full video produces **37 rallies**, with **0/37 wholly missed saved human rallies**. At the predeclared 2s symmetric padding: P_pad **87.90%**, R_core **99.26%**, F1_padP_coreR **93.23%**.

Positive padded gaps strictly below 3s are joined. Ignored intervals are removed without rejoining. All four padding cases are independently checked.

| Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0s | 82.00% | 92.65% | 87.00% | 362.875 | 321.146 | +41.729 |
| 1.0s | 85.90% | 98.08% | 91.59% | 434.875 | 397.146 | +37.729 |
| 2.0s | 87.90% | 99.26% | 93.23% | 506.875 | 470.645 | +36.230 |
| 3.0s | 89.27% | 99.57% | 94.14% | 586.500 | 544.644 | +41.856 |

## Device conditions

| Window | Observations | Awake/unlocked throughout | Thermal statuses | Battery temperature °C |
| --- | ---: | --- | --- | --- |
| short | 6 | True | [1] | [38.6, 39.2] |
| full | 43 | True | [1] | [39.2, 41.5] |

Android thermal status 0 means none, 1 light, 2 moderate, and 3 severe. No thermal override was used. The full run records whether the benchmark remains foreground.

## Storage and nested timings

4245 AV/fused rows and 2123 embedding rows. The four prepared-image buffers occupy 2,408,448 bytes; this is queue storage, not process memory.

| Saved FP32 diagnostic tensor | Bytes |
| --- | ---: |
| tokens | 32,609,280 |
| features | 67,104,960 |
| probabilities | 67,920 |

Within the shared AV stage, area conversion took 64.221s, neural image preparation 52.612s, encoder/readback 38.578s, and encoder queue backpressure 0.026s. The TCN took 0.551s. Overlapping measurements are not additive.

Calculated AV104 and contextual AV520 float payloads are 1,765,920 and 8,829,600 bytes respectively. Specialist feature sizes are recorded in the JSON report.

## Scope

- One fresh-process observation per duration; caches bypassed, no discarded warmup. Transfers and source hashing are excluded from device timing.
- Device-condition observations cover host orchestration, including transfer intervals; the measured stage and total times come from the app's internal clock.
- The phone was wirelessly charging. Thermal status and battery-temperature observations are recorded; this is not a controlled cool-device comparison.
- Historical Pixel reference times combine a different preprocessing implementation and separate decode passes, with unmatched device conditions; do not attribute their entire timing difference to shared decoding alone.
- This tests the existing FP32 CPU neural path in the native app harness, not a newly introduced GPU or NNAPI delegate.
- The exact comparison is against corrected same-phone outputs. Older Pixel references did not capture prepared-pixel hashes; the new short run does, but no same-phone historical pixel-hash equality is claimed.
- Fixed, repeatedly investigated recording with manually reviewed export-derived labels; this is runtime/output qualification, not model selection or a new generalization result.
- Short excerpt labels use a nominal 180-second offset for a stream-copy excerpt; full-video metrics are the headline accuracy evidence.
- Embedding preparation/inference overlap the AV stage. Their diagnostic timings must not be added again to AV wall time. Score specialists retain their later shared pass.

See the [JSON report](pixel-shared-video-decoding.json), [runner](../../scripts/benchmark-pixel-shared-decoding.py), [report generator](../../scripts/report-pixel-shared-decoding.py), and [emulator qualification](android-shared-video-decoding.md). Exact private inputs resolve through the external ledger and ignored local environment.
