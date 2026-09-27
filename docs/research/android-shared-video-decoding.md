# Android shared video decoding experiment

Frozen highest-recall Distilled MobileNetV3-Large + TCN FP32 on recording-044. AV and embeddings now share one timestamp inventory and asynchronous MediaCodec pass. Each consumer still prepares its own input directly from the original YUV image.

The experimental decoder hook is in the actual Android app. The opt-in neural consumer is wired through the pipeline benchmark source set; the normal production app still runs its existing ensemble.

## Runtime

Seconds, including feature generation and both score specialists. Shared AV includes concurrent embedding preparation/inference; the separate embedding-pass column is zero for shared runs.

| Window/path | AV incl. shared work | Audio | Separate embeddings | Rallies ready | Score specialists | All ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-independent | 109.072 | 6.498 | 107.697 | 223.900 | 55.612 | 279.512 |
| short-shared | 113.792 | 6.774 | 0.000 | 121.140 | 53.826 | 174.966 |
| full-independent | 1003.360 | 75.457 | 1018.399 | 2100.003 | 358.961 | 2458.964 |
| full-shared | 1021.519 | 72.783 | 0.000 | 1096.375 | 352.307 | 1448.682 |

Full-video all-ready savings: 1010.282s (41.09%). Rallies-ready savings: 1003.628s (47.79%).

For context, the earlier optimized production-only run took 1572.423s for all results on 63 rally proposals. Its score workload differs from the neural output; this is a separate observation, not an isolated measure of adding the neural model.

## Output validation

Short A/B requires exact equality of every prepared input SHA256, selected PTS, quality row, saved embedding, fused feature, probability, rally (including confidence), production rally, and both score-specialist inputs and decisions. Full comparison requires the same checks except pixel hashes, which the saved full reference did not capture. Every exact gate passed. Independent CPU temporal replay also reproduces rally boundaries.

Primary accuracy uses symmetric 2s padding, joins positive gaps strictly below 3s, and subtracts ignored intervals without rejoining. The target was fixed before evaluation.

| Full-video output | Rallies | Wholly missed saved human rallies | P_pad | R_core | F1_padP_coreR |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-independent | 37 | 0/37 | 87.90% | 99.26% | 93.23% |
| full-shared | 37 | 0/37 | 87.90% | 99.26% | 93.23% |
| Production from same AV | 63 | 1/37 | 66.33% | 99.01% | 79.44% |

### Required padding sensitivity

Full independent/shared outputs are identical; the following values apply to both.

| Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0s | 82.00% | 92.65% | 87.00% | 362.875 | 321.146 | +41.729 |
| 1.0s | 85.90% | 98.08% | 91.59% | 434.875 | 397.146 | +37.729 |
| 2.0s | 87.90% | 99.26% | 93.23% | 506.875 | 470.645 | +36.230 |
| 3.0s | 89.27% | 99.57% | 94.14% | 586.500 | 544.644 | +41.856 |

## Storage and implementation

4245 AV/fused rows; 2123 embedding rows. The bounded prepared-input pool holds four images (2,408,448 bytes). This is additional queue storage, not total process memory. It retains no decoder Image after the callback.

| Saved diagnostic tensor | Bytes |
| --- | ---: |
| tokens FP32 | 32,609,280 |
| features FP32 | 67,104,960 |
| probabilities FP32 | 67,920 |

AV104 payload size is 1,765,920 bytes; contextual AV520 is 8,829,600 bytes. These are calculated float payloads, not additional saved cache files in this bypassed-cache run. The JSON report also records decoded and persisted specialist-feature payload sizes.

Storage is unchanged by sharing. Separate preparation preserves AV area-resize and MobileNet bilinear-letterbox contracts. Repeated target rows reuse one source image and its embedding; the neural exclusive-end rule stays independent of AV nearest-frame selection. The encoder uses a bounded worker queue while the decoder owns and releases YUV images. Failures propagate and drain workers.

## Scope and reproducibility

- Emulator observation, not physical Pixel performance; API 37 x86_64, about 4 GB guest RAM and 24 GB configured storage.
- The native neural harness retains its existing ONNX Runtime CPU execution with four intra-op threads and one inter-op thread; this is not a phone GPU benchmark.
- One observation per path/window. Short A/B uses the same APK; full independent reference is the previously saved native-area run. Filesystem/JIT warmth and host scheduling are not controlled.
- Shared path currently qualifies fresh, unrotated, zero-start full-frame MobileNet input. Cached AV uses the independent path. Score-specialist decode remains a separate, already shared pass after rally selection.
- AV and embedding work overlap. Shared embedding span, inference and prepare measurements are nested within AV; do not add them to AV wall time.
- Short runs include optional prepared-pixel SHA256 diagnostics. All runs include saved diagnostic tensors and candidate reports.
- Fixed, repeatedly investigated recording and manually reviewed export-derived labels; no new generalization result or selection of model weights/thresholds.
- Short excerpt label alignment uses a nominal 180-second offset for a stream-copy excerpt; full-video metrics are the headline accuracy evidence.

200 JVM tests and 8 emulator instrumentation tests passed, including shared-consumer AV parity, repeated/terminal samples, consumer failure propagation and native color/resize checks. Exact runtime inputs resolve through the external ledger and ignored environment. See the [machine-readable report](android-shared-video-decoding.json), [runner](../../scripts/benchmark-shared-video-decoding.py), [exact comparison](../../scripts/validate-shared-video-decoding.py), and [report generator](../../scripts/report-shared-video-decoding.py).
