# Same-source Android/iPad diagnostic — 2026-09-11

The recovered Android cache and full iPad run use the **same source bytes, game
window and ROI**. All 4,424 analysis timestamps match. Both production ensembles
have 59 ranges: 55 have identical boundaries, and the other four differ by one
0.25-second tick. Unpadded time IoU is 0.997523. Audio features disagree substantially
more than visual features; exact audio or probability parity is not established.

This is a frozen implementation diagnostic, not model selection or release approval.
`indoor-source-05` / `spu-match1-20260526` is historically protected test footage.
No model, threshold, padding, decoder setting, seed or review policy was selected
using it. Product padding remains **2 seconds before/after**; the other required
padding cases are sensitivity results, never a best-case choice.

## Provenance

The original is `20260527 - Spu Rev4s - 2026-05-26 - Match 1 - Set 1 [Tds6i9umsku](1).mp4`,
named `VolleyCut-full-match.mp4` on Android and `tds6-reference.mp4` on iPad:
3,579,653,863 bytes, 3840×2160, rotation 0°, audio present, duration 1105.817 s.
Full-file SHA-256 read from the existing Android emulator copy:

`19c88cb2cbcd42c0ad65b778689a613ea648a4f03f89849ae20953c80d66b6e4`

- Android cache `c490bc66537f0510ec3359439fb79f9ee88b533ee6767bc843b19f9a26add6a3`
  was retained from September 3 under `files/native-features-v1`. Its key was
  reproduced from `NativeFeatureCache.buildKey`: URI
  `content://media/external/video/media/21`, source name/size above, modified time
  `1786727888`, geometry/codecs, ROI `(0.03,0.12,0.94,0.86)` and source-frame cap
  1,000,000. The cap was not reached: 66,349 source frames, 4,424 rows, software
  decoder `c2.android.avc.decoder`.
- iPad artifact `${VOLLEYCUT_IOS_LAB_ROOT}/artifacts/analysis-25c03a1835e4.json`, 29,543,721 bytes,
  SHA-256 `57b928665a143c6d701c886bc3497a771341b0926ac83418464e85205888b506`.
  Hashing the source SHA's ASCII text followed by
  `[0,1105.817,0.03,0.12,0.94,0.86,0]` reproduces its full cache identity
  `25c03a1835e4f457910cc0f931ce3c30c584391040a4208b4f9766e5862a9573`.
  This binds the artifact to the source hash despite its missing explicit source
  hash field. Its recorded decoded PTS all equal its requested timestamps.
- The 720p30 cache `ebe803716a32…` was separately identified as MediaStore 23,
  `VolleyCut-full-match-720p30-camera-feed.mp4`, 709,667,116 bytes. It is excluded.
- Android visual/audio/context binaries were copied unchanged through binary
  `adb exec-out run-as`. Base features and all six probability heads were replayed
  offline with unchanged compiled Android JVM code. **All 2,300,480 reconstructed
  contextual Float values match the stored cache bit-for-bit.** This is replay of
  retained device features, not a fresh decode or on-device inference measurement.
  Android actual decoded PTS are not retained; only requested times can be compared.
- Frozen all-labels model `model-1ca43e38eefc` SHA-256:
  `d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f`;
  previous production `model-9c92b8e9333f`:
  `d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d`.
  Both use `overlap-union-disagreement-v1` ensemble composition.

Artifacts are in `${VOLLEYCUT_IOS_LAB_ROOT}/artifacts/android-reference/`: `provenance.json`, original
cache binaries and identity preimages, `CacheInference.java`, `compare_cached.py`,
`android-ipad-native-comparison.json`, and `same-source-padding-report.json`.
`android-analysis.json` is inside the original cache directory. These retain model,
source and artifact checksums. No media was decoded on the unaccelerated Mac.
The JVM utilities and reproducible input/command instructions are checked in under
[scripts/android-reference](scripts/android-reference/README.md).

## Native feature and interval agreement

Errors compare Float32 values at identical times. Aggregate errors mix channel
scales; the JSON report also provides all 104 channels separately.

| Feature group | Values | Exactly equal | Mean absolute error | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| Visual, 73 columns | 322,952 | 224,088 | 6.84e-9 | 0.000350446 |
| Temporal, 4 columns | 17,696 | 11,989 | 2.53e-9 | 0.000005915 |
| Audio, 27 columns | 119,448 | 12,353 | 0.016447 | 4.500000 |
| Contextual, 520 columns | 2,300,480 | 1,738,524 | 0.001800 | 0.842754 |

Mean absolute rally/serve/dead probability errors are
`0.004305 / 0.001746 / 0.001992` for all-labels and
`0.004469 / 0.003687 / 0.001746` for previous production. The largest individual
head error is 0.175033 in the all-labels serve head.

An offline feature-block experiment isolates the difference: Android JVM inference
on iPad's exact contextual matrix reproduces **all 26,544 probabilities bit-for-bit**
and every interval. Android visual/temporal columns combined with iPad's 27 audio
columns reproduce every iPad interval; iPad visual/temporal columns combined with
Android audio reproduce every Android interval. Context was regenerated normally
for both combinations. Thus all observed interval differences follow the audio
inputs, not inference arithmetic or visual extraction. This is attribution only,
not a proposed mixed-runtime pipeline or model selection. Evidence is retained in
`audio-attribution.json` with the offline `AudioAttribution.java` utility.

| Output | Android / iPad ranges | Identical bounds | Unpadded time IoU | Symmetric difference s |
| --- | ---: | ---: | ---: | ---: |
| Ensemble | 59 / 59 | 55 | 0.997523 | 1.250 |
| Previous production | 59 / 59 | 56 | 0.993522 | 2.750 |
| All-labels v2 | 54 / 53 | 47 | 0.993045 | 3.000 |

The four ensemble differences are explicit:

| Android interval s | iPad interval s |
| --- | --- |
| 111.375–116.375 | 111.625–116.375 |
| 257.875–261.125 | 258.125–261.125 |
| 655.375–662.375 | 655.625–662.375 |
| 1006.500–1010.000 | 1006.750–1010.250 |

All-labels additionally splits `455.875–464.625` and `466.375–476.625` on Android;
iPad merges `455.875–476.625`. Previous production's difference around 1007 s is
larger than one tick; ensemble composition reduces the final boundary difference.

## Required padding sensitivity against reviewed human labels

Both runtimes use the same complete reviewed gold:
`labeling-v1-2026-08-09/completed/full-v1/indoor-source-05.labels.json`,
SHA-256 `48fd12d57a2b0265609421a65f108c3e3b39c2c67a76fd28622371f25248d1df`.
It has 39 rallies / 322.419 s core, annotator Vini, reviewed
`2026-08-11T05:50:11.261480+00:00`, with no ignored intervals. Its timeout hard
negatives remain in the evaluation universe. The mutable unreviewed `labels/full/`
revision is not substituted.

Common duration is the gold proxy's 1105.800 s. Apply identical padding to model
and human, clip, merge touching/overlapping intervals, and join positive gaps
**strictly below 3 seconds**. Subtract ignored spans afterwards and never rejoin
across them. Exactly 3 seconds remains a cut. Pool intersection numerators and
duration denominators before calculating `P_pad`, `R_core` and `F1_padP_coreR`;
this scope has one recording. Outputs are unreviewed, before suppression or user
corrections. Human accuracy and Android/iPad similarity are separate measurements.

| Output | Runtime | Before/after s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ensemble | Android | 0 | 0.605129 | 0.962115 | 0.742964 | 512.625 | 322.419 | +190.206 |
| Ensemble | Android | 1 | 0.614093 | 0.994417 | 0.759292 | 633.000 | 400.419 | +232.581 |
| Ensemble | Android | **2 (target)** | 0.628570 | 0.998983 | 0.771625 | 745.375 | 478.419 | +266.956 |
| Ensemble | Android | 3 | 0.654784 | 1.000000 | 0.791383 | 836.375 | 556.419 | +279.956 |
| Ensemble | iPad | 0 | 0.605719 | 0.962115 | 0.743410 | 512.125 | 322.419 | +189.706 |
| Ensemble | iPad | 1 | 0.617262 | 0.994417 | 0.761710 | 629.750 | 400.419 | +229.331 |
| Ensemble | iPad | **2 (target)** | 0.628780 | 0.998983 | 0.771784 | 745.125 | 478.419 | +266.706 |
| Ensemble | iPad | 3 | 0.654784 | 1.000000 | 0.791383 | 836.375 | 556.419 | +279.956 |
| Previous | Android | 0 | 0.700983 | 0.933248 | 0.800610 | 429.250 | 322.419 | +106.831 |
| Previous | Android | 1 | 0.681586 | 0.987315 | 0.806447 | 557.250 | 400.419 | +156.831 |
| Previous | Android | 2 | 0.689766 | 0.998753 | 0.815988 | 667.000 | 478.419 | +188.581 |
| Previous | Android | 3 | 0.715286 | 1.000000 | 0.834014 | 752.250 | 556.419 | +195.831 |
| Previous | iPad | 0 | 0.701298 | 0.933667 | 0.800970 | 429.250 | 322.419 | +106.831 |
| Previous | iPad | 1 | 0.681829 | 0.987315 | 0.806617 | 557.250 | 400.419 | +156.831 |
| Previous | iPad | 2 | 0.689710 | 0.998753 | 0.815949 | 667.250 | 478.419 | +188.831 |
| Previous | iPad | 3 | 0.715228 | 1.000000 | 0.833974 | 752.500 | 556.419 | +196.081 |
| All-labels | Android | 0 | 0.659459 | 0.895862 | 0.759694 | 438.000 | 322.419 | +115.581 |
| All-labels | Android | 1 | 0.681073 | 0.950285 | 0.793466 | 538.625 | 400.419 | +138.206 |
| All-labels | Android | 2 | 0.698808 | 0.969292 | 0.812120 | 639.250 | 478.419 | +160.831 |
| All-labels | Android | 3 | 0.720220 | 0.981149 | 0.830676 | 740.375 | 556.419 | +183.956 |
| All-labels | iPad | 0 | 0.660071 | 0.896182 | 0.760215 | 437.750 | 322.419 | +115.331 |
| All-labels | iPad | 1 | 0.681581 | 0.951061 | 0.794081 | 538.375 | 400.419 | +137.956 |
| All-labels | iPad | 2 | 0.699243 | 0.970067 | 0.812686 | 639.000 | 478.419 | +160.581 |
| All-labels | iPad | 3 | 0.720359 | 0.981924 | 0.831046 | 740.375 | 556.419 | +183.956 |

At target padding, Android/iPad ensemble export durations differ by 0.250 s.
At 1-second padding the difference is 3.250 s: a one-tick boundary change can cross
the strict 3-second joining threshold. Matching range counts hide that retained-gap
effect. Both runtimes retain roughly 267 s more than padded gold.

Unpadded event guardrails use maximum-cardinality chronological matching, then
total IoU, at IoU ≥ 0.5. They do not replace `F1_padP_coreR`.

| Output | Runtime | Matched / 39 gold | Event F1 | Median absolute start/end error s |
| --- | --- | ---: | ---: | ---: |
| Ensemble | Android | 23 | 0.469388 | 0.432 / 0.951 |
| Ensemble | iPad | 23 | 0.469388 | 0.432 / 0.951 |
| Previous | Android | 32 | 0.653061 | 0.534 / 0.528 |
| Previous | iPad | 32 | 0.653061 | 0.447 / 0.557 |
| All-labels | Android | 21 | 0.451613 | 0.459 / 0.972 |
| All-labels | iPad | 22 | 0.478261 | 0.470 / 1.022 |

## Audio diagnosis and remaining acceptance

The core implementations agree on source-frame timestamp rounding, negative
priming removal, overlap/gap handling, 16 kHz interpolation/quantization, 800-sample
FFT frames and inclusive ±0.125-second pooling. No clear core arithmetic defect
was found. Android defaults to signed 16-bit decoded PCM and supports float output;
iOS requests Float32. Actual decoded PCM and per-buffer formats/PTS were not
retained, so AAC implementation, output precision and fine timing remain hypotheses.

The median iPad/Android RMS ratio is 0.999915, inconsistent with a large global gain
error. The largest standardized channel differences are peak-to-RMS, elapsed time
since a transient, and high-band SNR/flux. Small transient differences can change
elapsed time by seconds and move a decoder threshold by a tick. No speculative
gain, offset or PCM-format adjustment was applied to fit protected footage. The next
isolating check is a short shared PCM capture with buffer PTS, sample rate, channel
count and encoding, passed through the same DSP. Any correction should have an
independent regression fixture.

The cold iPad artifact records video 111.969 s, audio 9.007 s, context 0.949 s and
inference 0.016 s. This is one device measurement, not a hardware speed comparison
with the Android emulator cache or interrupted historical runs.

`compare-device.py` supplies the metric functions and five contract tests. Its
earlier iPad padding/event results independently matched the canonical repository
evaluators within `1e-9`; this report uses those same functions and verified gold.
UI orientations, scoring specialists, edit/export interoperability, rotated-video
pixels and export colors remain separate acceptance checks in [Testing](TESTING.md).
A current-device decode repeat, including actual Android decoded PTS, remains
separate from this source-proven historical feature comparison.

The old [Android/browser report](../android/FULL_INFERENCE_PARITY.md) compares the
single previous-production model against a 960×540 browser proxy. It is historical
context, not the Android cache measurement above; browser results must not be
labeled Android output.
