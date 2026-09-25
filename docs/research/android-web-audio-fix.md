# Android and browser audio correction

The shared Android app decoder now reads AAC-LC frame length from codec configuration instead of inferring it from the first packet timestamp gap. Supported 1,024- and 960-sample layouts can use batching. Unknown profiles, extensions, or channel configurations use synchronous decoding in AUTO mode. Inconsistent output format, missing timestamps, unexplained leftover input timestamps, or excessive discarded overlap cause AUTO to retry synchronously; explicitly requested batching fails rather than returning suspect features.

This code is shared by production and neural analysis. Model weights and rally thresholds are unchanged. The rolling-percentile interpolation also now preserves integer quantiles, fixing false zero noise floors and SNR spikes during startup.

## Production web assessment

The production web app does **not** have Android's first-packet frame-size inference defect. `AudioAccumulator.push` uses each decoded sample's actual `numberOfFrames`, and the planar path uses the actual plane length. A regression sends 2,000 actual 1,024-frame decoded samples with an irregular 1,106-sample first timestamp gap: all 2,048,000 PCM frames survive, with only the genuine 82-frame silence gap inserted. Both the production and lab implementations pass.

The browser **did** share the startup percentile bug. Both web implementations are corrected and tested across the growing startup window and the steady 200-frame window. This changes feature generation; production rally accuracy after this smaller correction has not been remeasured. Android still uses linear resampling while production web uses libswresample; this change does not claim complete desktop/browser/Android feature equivalence.

## Cache and saved-project behavior

- Android feature-cache identity is now `native-dsp-v2-codec-framing-percentile`. Because raw visual and audio data share that cache identity, a fresh analysis regenerates both.
- Android projects and editor recovery records retain audio-extractor provenance. Older saved projects stay editable, but cannot satisfy a new analysis request. Newly analyzed copies have distinct IDs, preserving old manual edits.
- Production web audio-cache schema is now 2. Visual checkpoints remain reusable. Existing saved browser analyses are not automatically rewritten; rerun analysis to obtain corrected audio features.

## Verification

All 172 Android unit tests pass, including irregular initial timestamps, codec-layout rejection, gapless/carry accounting, startup percentiles, and saved-project reuse. Normal debug and instrumentation APKs build. The 26 targeted web tests, production TypeScript check, static build, and build integrity checks pass. No production web server or deployment was started.

The device test invokes the actual app's `NativeAudioDecoder` and `AudioFeatureExtractor`, using media supplied through runtime arguments. The optional media permission is confined to the debug manifest; release media access is unchanged. Generated builds, logs, tensors, and receipts are kept under external ledger artifact `private-reference-0223`.

On the Pixel 10 Pro, the first 120 seconds of `recording-044` produce byte-identical 480-by-27 pooled feature arrays in single-unit and AUTO/batched modes. Both produce 1,920,001 resampled samples; extraction takes 40.665s and 16.157s respectively. These are single audio-only diagnostic observations, not full-pipeline performance benchmarks or controlled thermal comparisons. The full 1,061.016489-second source also passes: both decoders produce byte-identical 4,245-by-27 arrays, 16,976,025 resampled samples (1,061.001563 seconds), and 21,221 audio frames. Full-file extraction measures 460.890s single-unit and 87.799s batched. Both little-endian feature files have SHA-256 `276a2952be5899245be43935b2d4e71d4912979041ebacd7f3dc3b90c7c7ff68`.

See the [root-cause investigation](android-audio-timeline-root-cause.md) for the original failure and frozen evidence. Earlier native benchmark accuracy/timing tables describe the pre-fix pipeline and remain historical measurements.

## Frozen-model replay with corrected phone audio

Only the 27 audio columns were replaced in each saved native feature matrix, then the same frozen scaler, weights, thresholds, and boundary decoder were replayed on CPU. All other native columns are unchanged. These are corrected-audio replay results, not newly measured complete video-to-score runs. The gold revision contains 37 manually reviewed saved human rallies; it is the same revision used by the earlier benchmark.

Target padding is two seconds before and after, with positive gaps strictly under three seconds joined. Ignored spans are subtracted after joining, without rejoining across them. This is one recording, with no model selection or generalization claim.

| Distilled Large selection | Found rallies | Wholly missed human rallies | P_pad | R_core | F1_padP_coreR | Export (s) |
|---|---:|---:|---:|---:|---:|---:|
| Highest f1 | 35 | 2 / 37 | 97.24% | 94.55% | 95.88% | 393.750 |
| Highest recall | 39 | 0 / 37 | 92.85% | 98.75% | 95.71% | 466.125 |

The highest-recall choice improves from **two wholly missed rallies to zero** (retained-core recall 93.84% to 98.75%). Zero wholly missed rallies does not imply perfect boundaries: some human core remains omitted. Highest F1 improves from six wholly missed rallies to two, human rally indexes 6 and 16. Remaining non-audio differences and boundary errors are not claimed fixed.

### Required padding sensitivity

| Selection | Padding each side (s) | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1 | 0 | 92.79% | 75.63% | 83.34% | 261.750 | 321.146 | -59.396 |
| f1 | 1 | 96.07% | 88.98% | 92.39% | 327.750 | 397.146 | -69.396 |
| f1 | 2 | 97.24% | 94.55% | 95.88% | 393.750 | 470.645 | -76.895 |
| f1 | 3 | 98.51% | 95.64% | 97.05% | 459.750 | 544.644 | -84.894 |
| recall | 0 | 88.29% | 88.46% | 88.37% | 321.750 | 321.146 | +0.604 |
| recall | 1 | 91.09% | 96.80% | 93.85% | 396.125 | 397.146 | -1.021 |
| recall | 2 | 92.85% | 98.75% | 95.71% | 466.125 | 470.645 | -4.520 |
| recall | 3 | 94.63% | 99.38% | 96.94% | 536.125 | 544.644 | -8.519 |

Audio RMS correlation with the saved desktop reference rises from 0.178 to 0.996; noise-floor correlation rises from 0.474 to 0.998. Individual transient/spectral features still differ, so the correction establishes decoder timing parity rather than complete desktop equivalence.

[Structured validation, identities, correlations and all padding cases](android-web-audio-fix.json).
