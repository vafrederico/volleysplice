# Android visual feature repair

The shared Android app now uses area downsampling for AV images and nearest actual
presentation timestamps for frame selection. The normal app and the separate neural
benchmark build successfully. All **194 JVM tests and four Pixel instrumentation
tests pass**. Full-video rally accuracy and updated processing time are **pending
device unlock**; no new missed-rally count is claimed yet.

This follows the [remaining highest-F1 miss investigation](distilled-mobile-large-remaining-miss.md).
The additional native miss was human rally 16, 09:22.496–09:28.746 on `recording-044`.
Evidence and prepared validation jobs use external artifact `private-reference-0223`.
No weights, normalization, decoder thresholds or selected models changed.

## Shared application changes

- [YuvAreaResampler](../../android/video-common/src/main/java/com/volleycut/video/YuvAreaResampler.java)
  converts and clips each source YUV pixel to RGB before area averaging, following
  the desktop RGB resize order. It supports fractional footprints, ROI boundaries,
  rotation and plane strides, with an integer-shrink fast path and no full-resolution
  RGB allocation. Bilinear interpolation and averaging YUV before clipping are not
  substituted for this operation.
- Both [native AV decoder paths](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java)
  use area sampling at 192×108. Explicit-size serving-side and side-switch samplers
  preserve their existing behavior. Production AV heads consume the repaired images
  too, so production output needs reevaluation alongside the neural selections.
- [NearestFrameSelection](../../android/video-common/src/main/java/com/volleycut/video/NearestFrameSelection.java)
  selects from actual source PTS, prefers the earlier frame on ties, handles repeated
  target rows, and retains terminal grid targets inside the permitted source window.
  Resume includes the prior selected image as temporal warmup. The old synthetic CFR
  timestamp shortcut is removed; source-inventory time remains in profiling.
- The [neural encoder prototype](../../android/neuralbenchmark/src/main/java/com/volleycut/neuralbenchmark/VideoEncoderBenchmark.java)
  uses the same nearest-frame policy. Embeddings and quality offsets refer to the
  image actually sampled. It retains its separate bilinear letterbox operation.
- `opencv-v3-area-nearest-frame` is included in visual cache and saved-project
  identity. Fresh analyses regenerate features instead of reusing older extraction.
  Existing projects and manual edits remain editable with their original provenance.

## Validation completed

| Check | Result |
|---|---|
| Normal debug app and instrumentation APK | Build passed |
| Separate pipeline benchmark APK | Build passed |
| JVM suite | 194 passed, zero failures or skips |
| Pixel area-sampler versus native OpenCV | Three tests passed |
| Pixel normal-app synchronous versus asynchronous decoding | One test passed |

The area tests include a checkerboard aliasing fixture, 24 combinations of color
standard/range/rotation with fractional crops, and 3840×2160 to 192×108 shrinking.
All output channels remain within one uint8 level of native OpenCV `INTER_AREA`,
allowing accumulation/tie-rounding differences. The actual-app video fixture checks
repeated targets and a requested timestamp after the final presentation frame but
before media end. Both decoder paths produce the full requested grid, finite
features, and matching feature values within 1e-5.

These checks validate the changed operators and decoder paths. They do not establish
complete real-video parity: hardware decoding, color conversion, audio resampling,
and numerical reductions can still differ from the desktop inputs.

## Full-video device confirmation

The repaired two-minute highest-F1 run completed in 139.857 seconds, including
feature generation and both score specialists, and produced seven neural rallies.
Desktop replay of its saved native fused tensor matched all decoded boundaries;
the maximum temporal probability difference was 7.45e-7. This verifies temporal
inference and decoding, not parity of native features with desktop extraction.
This is one timing observation, not a repeated speed qualification.

The completed highest-F1 run recovers human rally 16 (09:22.496–09:28.746), the
extra native-only miss. Only human rally 6 (04:33.366–04:41.114) remains wholly
missed, matching the saved desktop miss set. Model weights and decoder are unchanged.
The full run generated 2,123 embedding samples and 4,245 fused feature rows.
Desktop replay reproduced the phone's boundaries exactly, with a maximum probability
difference of 1.133e-6. An independent interval implementation matched all 16 metric
cases with zero numerical difference. This is one recording, not a generalization test.

| Highest-F1 inputs | Found intervals | Wholly missed human rallies | P_pad | R_core | F1_padP_coreR |
|---|---:|---:|---:|---:|---:|
| Previous visual pipeline, corrected audio replay | 35 | 2 | 97.24% | 94.55% | 95.88% |
| Repaired actual Pixel | 34 | 1 | 97.54% | 96.65% | 97.09% |
| Saved desktop | 34 | 1 | 97.69% | 96.69% | 97.19% |

The repaired Pixel's padding sensitivity is below. Export difference is model minus
human. A wholly missed rally has zero retained core overlap; R_core measures duration,
not the fraction of rally events found.

| Padding each side | P_pad | R_core | F1_padP_coreR | Model export seconds | Human export seconds | Difference seconds |
|---|---:|---:|---:|---:|---:|---:|
| 0 seconds | 93.28% | 77.41% | 84.61% | 266.500 | 321.146 | -54.646 |
| 1 second | 96.44% | 91.75% | 94.04% | 334.500 | 397.146 | -62.646 |
| 2 seconds (target) | 97.54% | 96.65% | 97.09% | 402.500 | 470.645 | -68.145 |
| 3 seconds | 98.75% | 97.47% | 98.10% | 470.500 | 544.644 | -74.144 |

The run was interrupted by device locking and later resumed through Tailscale.
Its elapsed time is invalid for speed comparison. The interruption receipt and
original outputs are preserved under the external artifact index.

Highest-recall also completed: 37 predicted intervals, zero wholly missed saved
human rallies, and 87.90% P_pad / 99.26% R_core / 93.23% F1_padP_coreR at target
padding. It omits 2.384 seconds of human core across partial boundary losses.
Its saved desktop counterpart has 88.12% / 99.26% / 93.36%, also zero whole misses.
The older visual pipeline with corrected-audio replay had 92.85% / 98.75% / 95.71%
and zero whole misses: restoring the input contract improves recall and desktop
agreement, but does not improve every metric for this selection on this recording.
Temporal replay error is at most 4.768e-7, with identical decoded boundaries.
Independent validation now covers 32 metric cases across both selections. The two
repaired feature passes also produce identical production ensemble boundaries.

| Highest-recall padding each side | P_pad | R_core | F1_padP_coreR | Model export seconds | Human export seconds | Difference seconds |
|---|---:|---:|---:|---:|---:|---:|
| 0 seconds | 82.00% | 92.65% | 87.00% | 362.875 | 321.146 | +41.729 |
| 1 second | 85.90% | 98.08% | 91.59% | 434.875 | 397.146 | +37.729 |
| 2 seconds (target) | 87.90% | 99.26% | 93.23% | 506.875 | 470.645 | +36.230 |
| 3 seconds | 89.27% | 99.57% | 94.14% | 586.500 | 544.644 | +41.856 |

## Runtime regression

The highest-recall run measured 1,497.437 seconds (24m57s), with rallies ready at
1,362.619 seconds (22m43s), then 134.818 seconds for the score specialists.
No lock interruption was recorded for this run; continuous foreground/lock/thermal
monitoring was not collected. Thermal status was zero at start and finish. These
are single observations, not controlled or repeated throughput measurements.

| Highest-recall stage | Previous pipeline | Repaired pipeline | Increase |
|---|---:|---:|---:|
| Video features | 256.119s | 805.417s | 549.298s |
| Audio features | 73.164s | 82.948s | 9.784s |
| Neural pass, including embeddings | 272.962s | 474.011s | 201.049s |
| Score specialists | 131.832s | 134.818s | 2.986s |
| All ready | 734.295s | 1497.437s | 763.142s |

The correctness repair introduced a major preprocessing regression. Area reduction
converts and averages all 1920×1080 source pixels instead of sampling only 192×108
points: 100 times as many source pixel conversions per AV sample. The measured
resize/color bucket rose from 23.814s to 511.880s. The AV stage accounts for about
72% of the total increase; its profile buckets overlap and must not be summed.

The embedding pass added a 63.915s actual-PTS scan. Image preparation also rose
from 50.815s to 165.781s, and encoder/readback from 26.428s to 43.570s. Its pixel
preparation algorithm and weights are unchanged, and only selected frames are
prepared; extra candidate-frame processing does not explain this increase. The
remaining runtime/device-state contributions require profiling. The newer
encoder/readback timer also includes quality JSON and timestamp bookkeeping that
previously sat outside it, so that bucket is not a pure model-compute comparison.
The TCN itself
took 0.218s, essentially unchanged from 0.222s. Tailscale transfers are outside
the measured processing total.

The next performance work should optimize area conversion in native/vectorized
code while retaining the verified resize and color contract, and reuse the actual
PTS inventory across video and embedding passes. The current implementation is
accuracy-validated on this recording, but its runtime is not product-ready.

The repaired benchmark APK is installed. The prepared sequence first runs the
two-minute excerpt, verifies frozen TCN/decoder replay, then runs the full source
for the highest-F1 and highest-recall distilled Large selections. Each run bypasses
feature caches and includes both score specialists. The benchmark now also saves
the production ensemble ranges already computed from the same new AV features.

Full-video evaluation uses the unchanged 37-rally gold snapshot and reports
wholly missed rallies, P_pad, R_core, F1_padP_coreR and export durations for all four
0/1/2/3-second symmetric padding cases, with 2 seconds as the target, strict
less-than-3-second gap joining, and ignored intervals subtracted without rejoining.
The timing comparison must include the extra area-conversion and timestamp-scan
work; no speed claim follows from the synthetic tests.

[Structured build/test receipt and source hashes](android-visual-feature-repair.json).
[Full evaluation, all comparison padding cases, and independent verification](android-visual-feature-repair-evaluation.json).
