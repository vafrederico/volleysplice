# Optimized Android full-video validation

The optimized Android pipeline processed the complete **1061.016-second recording-044** on the emulator. Rallies were ready in **35m00s**; all inferences, including serving side and side switching, were ready in **40m59s**. These timings come from one complete fresh-input x86_64 emulator run.

The frozen highest-recall Distilled MobileNetV3-Large + TCN FP32 selection produced **37 intervals**, wholly missed **0 of 37 saved human rallies**, and achieved **87.90% P_pad / 99.26% R_core / 93.23% F1_padP_coreR** at the declared 2-second target padding.

This validates the optimized shared app preprocessing over the full source. It does not measure the physical Pixel or establish a full-video speedup against Java: only the optimized full pass was run. The [previous controlled two-minute comparison](android-native-area-optimization.md) established the Java/native speedup and exact output equality.

A subsequent [production-only full-video comparison](production-emulator-full-comparison.md)
measures original point, repaired Java area and optimized native area separately,
including both score specialists. It supplies a full-video production control;
the neural timing above still has no paired full-video Java run.

## Complete pipeline timing

| Stage | Seconds |
|---|---:|
| Video decode and AV features | 1003.360 |
| Audio decode and features | 75.457 |
| Contextualization | 0.222 |
| Embedding video pass | 1018.399 |
| TCN inference | 0.295 |
| Fusion, normalization and diagnostic saves | 0.914 |
| Rallies ready | 2100.003 |
| Serving side and side switch | 358.961 |
| All inferences ready | 2458.964 |

| Nested preprocessing counter | Seconds |
|---|---:|
| AV timestamp inventory | 85.588 |
| AV area resize and color conversion | 38.085 |
| AV OpenCV feature calls | 289.102 |
| AV output image acquisition | 4.490 |
| AV codec output release | 6.386 |
| AV output callback wall time | 54.669 |
| Embedding timestamp inventory | 81.783 |
| Embedding image preparation | 27.410 |
| Embedding encoder/readback | 22.752 |
| Embedding decode/other, excluding inventory | 886.453 |
| Shared score decode | 338.540 |

After the optimization, area resize/color takes 38.085 seconds across the full source. The largest remaining embedding bucket is decode/other (886.453 seconds), while image preparation and encoder/readback take 27.410 and 22.752 seconds. The separate AV/embedding timestamp scans add 85.588 and 81.783 seconds to their respective passes.

Nested counters overlap and must not be added to stage totals. Encoder/readback includes bookkeeping; decode/other is a residual wall-time bucket. The run includes model loading, diagnostic tensor writes, production evidence required by score specialists, and all required feature generation. Install, media/model transfer, collection and export rendering are excluded.

## Short-to-full scaling

| Stage | Two-minute median: seconds per video minute | Full video: seconds per video minute | Full / short rate |
|---|---:|---:|---:|
| AV resize/color | 2.084 | 2.154 | 1.034x |
| Complete video AV | 53.670 | 56.740 | 1.057x |
| Audio | 3.405 | 4.267 | 1.253x |
| Embedding pass | 54.567 | 57.590 | 1.055x |
| Score specialists | 27.492 | 20.299 | 0.738x |
| All ready | 139.463 | 139.053 | 0.997x |

Overall processing cost per video minute was essentially stable (0.3% lower). Area resize/color cost rose 3.4% per video minute, and complete AV/embedding passes rose about 5.7%/5.5%; embedding image preparation was 13.6% lower per video minute. This full run does not reproduce a large sustained preprocessing regression.

These rates compare one complete recording with a different two-minute section, not identical workloads. Content, source-file size, score proposals, filesystem/JIT state and duration can change the ratio; it is a scaling diagnostic, not an isolated speedup claim.

## Saved-human comparison

Target padding is 2 seconds before and after, with positive gaps joined only when strictly below 3 seconds. Ignored intervals are removed from every union without rejoining. R_core is retained human-core duration, not event recall. Wholly missed means zero retained nonignored core after padding and joining.

| Inputs / model | Found intervals | Wholly missed human rallies | P_pad | R_core | F1_padP_coreR | Export seconds |
|---|---:|---:|---:|---:|---:|---:|
| Optimized Android emulator | 37 | 0 | 87.90% | 99.26% | 93.23% | 506.875 |
| Production ensemble from same emulator AV pass | 63 | 1 | 66.33% | 99.01% | 79.44% | 694.891 |
| Saved repaired Pixel, Java area | 37 | 0 | 87.90% | 99.26% | 93.23% | 506.875 |
| Saved desktop | 37 | 0 | 88.12% | 99.26% | 93.36% | 506.625 |

Event matching gives the neural output 83.78% precision / 83.78% recall / 83.78% F1 (31 matched events), and production 55.56% / 81.08% / 65.93%. Zero wholly missed human rallies means retained coverage; it does not establish perfect one-to-one rally separation.

All 37 optimized emulator neural boundaries exactly match the saved repaired Pixel boundaries, despite small cross-platform input/probability differences. Saved Pixel/desktop rows reuse the same frozen 37-rally gold, encoder/TCN weights and decoder. They are output references; their runtimes are not compared to emulator timing. Production is evaluated from the same fresh emulator AV pass. These labels are manually reviewed export-derived ranges with later edits, not independently precise serve-contact/dead-ball annotation. This previously investigated recording is a deployment diagnostic, not a new generalization estimate.

## Required padding sensitivity

| Inputs / model | Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Optimized Android emulator | 0s | 82.00% | 92.65% | 87.00% | 362.875 | 321.146 | +41.729 |
| Optimized Android emulator | 1s | 85.90% | 98.08% | 91.59% | 434.875 | 397.146 | +37.729 |
| Optimized Android emulator | 2s | 87.90% | 99.26% | 93.23% | 506.875 | 470.645 | +36.230 |
| Optimized Android emulator | 3s | 89.27% | 99.57% | 94.14% | 586.500 | 544.644 | +41.856 |
| Production ensemble from same emulator AV pass | 0s | 61.02% | 96.72% | 74.83% | 509.016 | 321.146 | +187.870 |
| Production ensemble from same emulator AV pass | 1s | 63.60% | 98.78% | 77.38% | 607.016 | 397.146 | +209.871 |
| Production ensemble from same emulator AV pass | 2s | 66.33% | 99.01% | 79.44% | 694.891 | 470.645 | +224.247 |
| Production ensemble from same emulator AV pass | 3s | 70.90% | 99.01% | 82.63% | 757.516 | 544.644 | +212.873 |
| Saved repaired Pixel, Java area | 0s | 82.00% | 92.65% | 87.00% | 362.875 | 321.146 | +41.729 |
| Saved repaired Pixel, Java area | 1s | 85.90% | 98.08% | 91.59% | 434.875 | 397.146 | +37.729 |
| Saved repaired Pixel, Java area | 2s | 87.90% | 99.26% | 93.23% | 506.875 | 470.645 | +36.230 |
| Saved repaired Pixel, Java area | 3s | 89.27% | 99.57% | 94.14% | 586.500 | 544.644 | +41.856 |
| Saved desktop | 0s | 82.30% | 92.93% | 87.29% | 362.625 | 321.146 | +41.479 |
| Saved desktop | 1s | 86.15% | 98.24% | 91.80% | 434.625 | 397.146 | +37.479 |
| Saved desktop | 2s | 88.12% | 99.26% | 93.36% | 506.625 | 470.645 | +35.980 |
| Saved desktop | 3s | 89.46% | 99.57% | 94.24% | 586.250 | 544.644 | +41.606 |

## Feature storage

| Data | Decimal MB | Evidence |
|---|---:|---|
| AV104 | 1.766 | Calculated from actual row count |
| Contextual AV520 | 8.830 | Calculated from actual row count |
| tokens | 32.609 | Saved file size verified |
| features | 67.105 | Saved file size verified |
| probabilities | 0.068 | Saved file size verified |
| servingSide features | 0.070 | Decoded saved payload |
| sideSwitch features | 0.010 | Decoded saved payload |

Tokens also appear inside fused features. These diagnostic files are not additive minimum memory and do not measure peak RAM. AV arrays are calculated sizes, not persisted outputs.

## Validation and limits

The next performance targets follow from these counters and source review:

- Reuse the actual presentation-timestamp inventory between AV and embeddings.
  They currently scan the same source independently, costing 85.588s and 81.783s.
  A source/track/window-keyed inventory could avoid the second scan; this is an
  opportunity, not a measured speedup. Preserve packet reorder handling, real
  timestamps, previous-sync coverage, clipping and earlier ties.
- Share sequential decoding between the 4 Hz AV and 2 Hz embedding streams, or
  first move embeddings onto the AV decoder's existing asynchronous decode-only
  design. The embedding decoder emits 63,670 frames for 2,123 selected images
  and holds an output image while inference runs. Generate each stream's exact
  pixels from the original selected YUV image, release it promptly, and use
  bounded inference buffers. Resizing the AV thumbnail into the encoder input
  would change the contract.
- Improve specialist output draining while retaining its separate sample times.
  Serve and switch already share one decoder and skip large gaps. Their requests
  depend on predicted rallies, so a 2 Hz image cache cannot replace this pass.
  The native specialist sampler selects the first output at/after a target
  within its 1 ms tolerance and has a terminal fallback; it must not silently
  inherit the AV/embedding nearest-frame policy.

These targets are not implemented in this experiment. The relevant paths are
[AV decoding](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java),
[embedding decoding](../../android/neuralbenchmark/src/main/java/com/volleycut/neuralbenchmark/VideoEncoderBenchmark.java)
and [specialist decoding](../../android/app/src/main/java/com/volleycut/nativeanalysis/SpecialistFrameDecoder.java).
Each change needs actual selected-PTS equality, feature/quality/probability and
score-output parity, decoded boundaries and all four padding checks. Inter-frame
dependencies still require compressed frames to be decoded; suppressing unwanted
output materialization does not imply a speedup proportional to the sampling ratio.

- 4,245 AV/fused rows and 2,123 image embedding samples cover the complete source.
- Frozen temporal replay passed; maximum probability error 4.17e-07, with identical decoded boundaries.
- Independent interval code verified all 16 padding comparisons with maximum error 0.
- 164 wake/memory observations; every observed state awake: True. Minimum host available memory 9757 MB; monitor errors 0.
- The embedding decoder emitted 63,670 source frames for 2,123 selected images. The AV path does not persist a decoded-source frame count. Codec output acquisition/release/callback timings are included above.
- One fresh-process pass, no discarded warmup, feature caches bypassed. Sampling observations cannot exclude all short interruptions or host scheduling effects.
- Same NAS-backed API 37 x86_64 Google APIs revision 6 userdebug emulator as the short comparison, with the documented SELinux workaround, WHPX and host graphics. Its 24 GB configuration refers to storage, not RAM.
- Physical Pixel performance after this optimization remains unmeasured. Cross-platform codec and floating-point differences remain visible in the JSON tensor comparison; this is not an isolated Java/native pixel test.
- SELinux enforcement and non-root ADB were restored, and emulator process exit was confirmed before browser timing began.
- No training, calibration, threshold change or model selection. Raw artifacts resolve through private-reference-0223.

[Machine-readable report](android-native-area-full-video.json).
