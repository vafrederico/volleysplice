# Pixel visual-pipeline retest

The physical Pixel 10 Pro was tested over Tailscale with the same byte-verified two-minute excerpt of `recording-044` used for the emulator comparison. The frozen highest-recall Distilled MobileNetV3-Large + TCN FP32 selection and corrected audio were unchanged. Both versions include serving-side and side-switch analysis. No training, model selection, or performance optimization was performed.

## Controlled comparison

The sequence was baseline warmup, repaired warmup, baseline, repaired, repaired, baseline. The tables use the medians of two measured fresh-input runs per version; warmups are excluded. Each run starts a new benchmark process and bypasses feature caches. The old build restores only the previous visual decoder and encoder frame-selection implementation. The builds share the benchmark harness and identical ARM64 native dependencies.

| Stage | Previous visual pipeline | Repaired visual pipeline | Change |
|---|---:|---:|---:|
| All results ready | 64.847s | 122.325s | +88.6% |
| Rallies ready | 48.370s | 104.522s | +116.1% |
| Video features | 19.779s | 74.178s | +275.0% |
| Audio features | 3.510s | 3.450s | -1.7% |
| Embedding pass | 24.665s | 26.484s | +7.4% |
| Score specialists | 16.477s | 17.802s | +8.0% |

Measured total times:

- Baseline: 64.943s, 64.750s.
- Repaired: 122.946s, 121.703s.

## Detailed counters

| Counter | Previous median | Repaired median |
|---|---:|---:|
| Video resize/color conversion | 2.494s | 54.240s |
| AV timestamp inventory | 1.632s | 1.506s |
| OpenCV feature calls | 10.387s | 14.292s |
| Embedding timestamp inventory | 0.000s | 2.101s |
| Embedding image preparation | 5.165s | 5.176s |
| Encoder/readback | 2.857s | 2.864s |
| TCN inference | 0.033s | 0.035s |

Video feature generation accounts for 94.6% of the total median increase. The video resize/color counter directly measures the added area-conversion work. Detailed counters are nested and overlap with stage totals; they must not be added together. Encoder/readback also changed bookkeeping scope, so it is not a pure model-compute comparison.

The emulator and this phone retest both reproduce the area-resampling regression and added embedding timestamp scan. Compare the embedding-preparation values above separately from the earlier single full-video observations. A short-clip repeat cannot establish the cause of the full-video increase or predict a new full-video total.

## Verification and device conditions

- All six runs completed; each generated 480 fused feature rows and 240 embedding samples.
- Desktop temporal replay and exact rally-boundary comparison passed for all six runs; maximum probability error was 4.77e-07.
- Corrected audio feature values were exactly identical across all runs.
- All 44 periodic device observations showed the phone awake and unlocked, with no monitoring errors. Every observation within the processing windows showed the benchmark in the foreground.
- Maximum observed thermal status: 0; battery temperature range: 30.9 to 32.3 degrees C.
- Foreground windows are bounded by plan save and final result save, including setup/polling margin and excluding later tensor transfers. Periodic sampling does not rule out shorter interruptions.
- Physical ARM64 device, API 37, SELinux Enforcing. APK/model installation and Tailscale transfers are outside on-device processing totals.
- The repaired benchmark APK was restored after the final baseline run.

Every baseline run produced six neural intervals and six production intervals.
Every repaired run produced six neural intervals and eight production intervals.
These are descriptive interval counts, not a new human-label accuracy evaluation.

[Structured results](pixel-visual-pipeline-retest.json). Raw runtime artifacts are indexed under `private-reference-0223`.

Related: [emulator comparison](android-emulator-visual-regression.md) and [full-video visual repair evaluation](android-visual-feature-repair.md).
