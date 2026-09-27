# Highest-F1 distilled Large: remaining native missed rally

**The extra wholly missed rally is human rally 16, 09:22.496-09:28.746. Its remaining discrepancy is visual preprocessing, with strong causal evidence for the handcrafted focus/blur inputs.** The corrected native replay misses rallies 6 and 16; desktop misses only rally 6 (04:33.366-04:41.114).

Recording `recording-044`, external evidence `private-reference-0223`. Same 37-rally gold snapshot, encoder, TCN weights, fitted normalization and decoder throughout. Highest-F1 means the previously selected draw 20260918 / epoch 60 model, not a new selection on this recording. No app changes or training were performed.

Follow-up: the [shared Android visual repair](android-visual-feature-repair.md) is now implemented and passes unit/device operator tests. Full-video confirmation is tracked separately; the diagnostic results below remain frozen historical input replacements.

## Why this rally disappears

The frozen decoder averages live-play probabilities over one second and requires 0.90 to enter a rally. For rally 16, corrected native inputs peak at **0.8833 after smoothing**, while desktop reaches **0.9130**. Native has individual samples up to 0.9902, but those brief peaks do not survive smoothing above the entry threshold. The decoder therefore produces no interval to pad or join.

Desktop emits a short core, 09:23.875-09:25.125. Two-second padding retains 4.629 of the 6.250 human-core seconds; it still omits the last 1.621 seconds. This is partial recovery, not a fully correct boundary.

## Controlled input replacements

Target padding is two seconds before and after; positive gaps strictly under three seconds join. Ignored spans are removed after joining, without rejoining. Wholly missed means zero retained nonignored human core. Rows using mixed native/desktop features are diagnostic counterfactuals, not deployable models or measured phone fixes.

| Inputs | Found | Wholly missed human indexes | P_pad | R_core | F1_padP_coreR | Rally 16 maximum smoothed live |
|---|---:|---|---:|---:|---:|---:|
| Corrected phone audio + saved native visuals | 35 | 6, 16 | 97.24% | 94.55% | 95.88% | 0.8833 |
| Saved desktop | 34 | 6 | 97.69% | 96.69% | 97.19% | 0.9130 |
| Phone + desktop audio | 36 | 16 | 97.34% | 96.26% | 96.80% | 0.8439 |
| Phone + desktop AV visual block | 34 | 6 | 97.62% | 95.99% | 96.80% | 0.9466 |
| Phone + desktop embeddings | 35 | 6, 16 | 96.66% | 94.55% | 95.60% | 0.8597 |
| Phone + desktop focus/blur only | 36 | None | 95.47% | 98.25% | 96.84% | 0.9783 |
| Phone + desktop frame-time offset only | 36 | 6, 22 | 94.60% | 95.39% | 94.99% | 0.9652 |

Equal missed-rally counts concealed different failures: replacing audio alone gives one miss, but that miss is rally 16, whereas desktop misses rally 6. Replacing only the learned embeddings does not recover the extra miss. Replacing only focus and blur recovers both intervals on this recording, but still omits 5.619 seconds of total human core. This does not establish generalization or justify choosing that synthetic mixture.

## Visual feature contract mismatch

- Desktop [frame preprocessing](../../analysis/features.py) downsamples the full source image to 192x108 with OpenCV `INTER_AREA`. Android [shared video decoding](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java) selects one source YUV pixel per output pixel in both its synchronous conversion and `YuvCropSampler` paths.
- The [Android visual extractor](../../android/app/src/main/java/com/volleycut/nativeanalysis/VisualFeatureExtractor.java) and desktop use matching Laplacian/focus/blur formulas. The input pixels differ. Point sampling preserves/aliases fine detail that area averaging removes.
- Focus/blur are absolute features on both platforms; they bypass per-recording percentile ranking. Around rally 16, native mean focus is 0.99227 versus desktop 0.96294. The frozen scaler turns that into +4.67 versus +0.41 standardized units; blur shifts oppositely. This is a substantial input-distribution change.

An independent same-frame check isolates resizing on eight identical decoded frames near this rally. At source time 09:23.495:

| Resize of identical source frame | Laplacian variance | Focus | Blur |
|---|---:|---:|---:|
| Desktop area averaging | 2,658.995 | 0.963755 | 0.036245 |
| Android-coordinate point sampling | 13,207.921 | 0.992486 | 0.007514 |
| Bilinear control | 10,207.050 | 0.990298 | 0.009702 |

The roughly fivefold variance inflation and focus/blur values reproduce the direction and scale of the saved-device discrepancy. This check holds decoded pixels fixed; it does not emulate MediaCodec YUV color conversion or claim exact device parity. Bilinear resizing alone is also not equivalent to the training area-downsampling contract.

## Secondary frame-selection mismatch

Desktop [embedding extraction](../../analysis/mobile_visual_features.py) chooses the nearest source timestamp, preferring the earlier frame on a tie. The [native encoder benchmark](../../android/neuralbenchmark/src/main/java/com/volleycut/neuralbenchmark/VideoEncoderBenchmark.java) takes the first frame at or after the target. Both calculate selected timestamp minus requested timestamp; this is a sampling-policy difference, not reversed subtraction.

At rally 16 the fitted scale is so narrow that native offset inputs saturate at +10 while desktop saturates at -10. Replacing only this scalar recovers rally 16 but creates a different wholly missed rally (22). Thus it is another input-contract problem to qualify, not a safe isolated substitution. Reconstructed clipped offset values are not the actual source PTS offsets.

## Checks and next repair

- Current clean editor-lab output exactly matches the saved desktop 34 rally ranges. Latest saved human labels reproduce the same metrics. Serving-side enrichment leaves all rally boundaries intact; default suppression is off.
- Original native and desktop probabilities reproduce within 1.5e-6 maximum absolute error. Corrected-audio boundaries and metrics reproduce the earlier report exactly.
- Independent interval arithmetic verified six principal cases at all four padding settings, agreeing within 1.14e-13 seconds.
- The next repair is to match shared native AV downsampling to desktop area averaging, then qualify matched-frame focus/blur, remaining AV features and frozen-model output. Align embedding frame selection separately and regenerate the corresponding image, embedding and timestamp scalar together; changing the scalar alone would misrepresent the selected image. Preserve existing checkpoints and thresholds during parity work; then rerun the real app pipeline and production controls.

This investigation localizes a remaining visual input mismatch. It does not claim an implemented fix, fresh end-to-end phone timing, or a new generalization result.

## Required padding sensitivity

All cases use identical model/human padding and the same ignored-range revision. Rank/selection is unchanged; the target case remains 2s.

| Inputs | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected phone audio + saved native visuals | 0s | 92.79% | 75.63% | 83.34% | 261.750 | 321.146 | -59.396 |
| Corrected phone audio + saved native visuals | 1s | 96.07% | 88.98% | 92.39% | 327.750 | 397.146 | -69.396 |
| Corrected phone audio + saved native visuals | 2s | 97.24% | 94.55% | 95.88% | 393.750 | 470.645 | -76.895 |
| Corrected phone audio + saved native visuals | 3s | 98.51% | 95.64% | 97.05% | 459.750 | 544.644 | -84.894 |
| Saved desktop | 0s | 93.50% | 77.45% | 84.72% | 266.000 | 321.146 | -55.146 |
| Saved desktop | 1s | 96.62% | 91.84% | 94.17% | 334.000 | 397.146 | -63.146 |
| Saved desktop | 2s | 97.69% | 96.69% | 97.19% | 402.000 | 470.645 | -68.645 |
| Saved desktop | 3s | 98.88% | 97.39% | 98.13% | 470.000 | 544.644 | -74.644 |
| Phone + desktop audio | 0s | 92.93% | 76.25% | 83.77% | 263.500 | 321.146 | -57.646 |
| Phone + desktop audio | 1s | 96.19% | 90.31% | 93.16% | 331.500 | 397.146 | -65.646 |
| Phone + desktop audio | 2s | 97.34% | 96.26% | 96.80% | 399.500 | 470.645 | -71.145 |
| Phone + desktop audio | 3s | 98.58% | 97.43% | 98.00% | 467.500 | 544.644 | -77.144 |
| Phone + desktop AV visual block | 0s | 93.35% | 76.37% | 84.01% | 262.750 | 321.146 | -58.396 |
| Phone + desktop AV visual block | 1s | 96.53% | 90.71% | 93.53% | 330.750 | 397.146 | -66.396 |
| Phone + desktop AV visual block | 2s | 97.62% | 95.99% | 96.80% | 398.750 | 470.645 | -71.895 |
| Phone + desktop AV visual block | 3s | 98.82% | 97.08% | 97.95% | 466.750 | 544.644 | -77.894 |
| Phone + desktop embeddings | 0s | 91.98% | 75.82% | 83.12% | 264.750 | 321.146 | -56.396 |
| Phone + desktop embeddings | 1s | 95.39% | 88.90% | 92.03% | 330.750 | 397.146 | -66.396 |
| Phone + desktop embeddings | 2s | 96.66% | 94.55% | 95.60% | 396.750 | 470.645 | -73.895 |
| Phone + desktop embeddings | 3s | 98.00% | 95.64% | 96.81% | 462.750 | 544.644 | -81.894 |
| Phone + desktop focus/blur only | 0s | 90.49% | 81.22% | 85.61% | 288.250 | 321.146 | -32.896 |
| Phone + desktop focus/blur only | 1s | 94.03% | 93.30% | 93.66% | 358.250 | 397.146 | -38.896 |
| Phone + desktop focus/blur only | 2s | 95.47% | 98.25% | 96.84% | 428.250 | 470.645 | -42.395 |
| Phone + desktop focus/blur only | 3s | 96.97% | 99.26% | 98.10% | 500.000 | 544.644 | -44.644 |
| Phone + desktop frame-time offset only | 0s | 90.67% | 78.35% | 84.06% | 277.500 | 321.146 | -43.646 |
| Phone + desktop frame-time offset only | 1s | 92.99% | 90.91% | 91.93% | 349.500 | 397.146 | -47.646 |
| Phone + desktop frame-time offset only | 2s | 94.60% | 95.39% | 94.99% | 416.750 | 470.645 | -53.895 |
| Phone + desktop frame-time offset only | 3s | 96.57% | 96.25% | 96.41% | 480.750 | 544.644 | -63.894 |

The [structured evidence](distilled-mobile-large-remaining-miss.json) preserves all block and individual-feature substitutions, all four padding settings, event metrics, missed ranges, frozen identities, input hashes and same-frame resize observations. See the [prior audio repair](android-web-audio-fix.md) and [original native comparison](distilled-mobile-large-native-benchmark.md) for historical measurements.
