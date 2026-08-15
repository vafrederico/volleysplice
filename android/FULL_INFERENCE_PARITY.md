# Historical single-model native A/V inference validation

Validation date: 2026-08-14

This report predates the production two-model ensemble. It remains the validation
record for the previous-production model and native feature/cache path; current app
inference also runs all-labels v2, unions overlapping detections, and flags
single-model candidates with reduced review confidence.

## Scope

- Native source: `20260527 - Spu Rev4s - 2026-05-26 - Match 1 - Set 1 [Tds6i9umsku](1).mp4`
- Native duration: 1,105.817 seconds
- Native analysis rows: 4,424 at 4 Hz
- Model: `model-9c92b8e9333f`
- ROI: `(0.03, 0.12, 0.94, 0.86)`
- Primary browser reference: `model-browser-on-device-libswresample-wasm-9c92b8e9333f--indoor-source-05`
- Secondary browser reference: `model-browser-on-device-9c92b8e9333f--indoor-source-05`

The primary reference is the closest browser comparison because it uses the WASM
`libswresample` experiment. Android still uses the app's linear 16 kHz resampler,
so the audio front ends are not byte-identical. The native YUV conversion and OpenCV
reductions are also expected sources of feature-distribution differences.

This is an implementation-parity comparison of the same frozen model, not a model
iteration ranking or protected-test selection.

## Execution and cache validation

The first full run was intentionally stopped after 18 seconds, after which 208 visual
rows were present in 13 atomic chunks. The resumed run reported
`resumedVisualRows = 208`, completed all 4,424 visual rows, decoded the full audio
track, and produced 59 ranges without a cache failure.

| Measurement | Result |
| --- | ---: |
| Resumed completion pass | 481.465 s |
| Video stage | 359.937 s |
| Audio stage | 121.230 s |
| Context stage | 0.147 s |
| Inference stage | 0.107 s |
| Completion-pass real-time ratio | 2.297x |
| Thermal status, start/end | 0 / 0 |
| Completed feature cache | 11,007,398 bytes |
| Warm full rerun | 0.980 s |
| Warm video cache read | 0.010 s |
| Warm audio cache read | 0.001 s |
| Warm contextual-feature cache read | 0.016 s |

The completion timing excludes the interrupted 18-second prefix. The resume path
must still feed the recording through MediaCodec from the beginning to restore codec
state, but it suppresses cached analysis outputs and skips their OpenCV work.

The warm full rerun reported visual, audio, and contextual cache hits and reproduced
59 ranges. The 1,000-source-frame cold/warm cache check reproduced the range
`4.375-9.125` and confidence `0.8420774341` exactly.

## Unpadded range parity

All duration metrics below operate on the union of the unpadded ranges. The intervals
within each result are already non-overlapping.

| Metric | Browser libswresample/WASM | Browser default audio |
| --- | ---: | ---: |
| Native/browser range count | 59 / 55 | 59 / 56 |
| Native/browser range duration | 423.000 / 404.800 s | 423.000 / 418.217 s |
| Intersection duration | 382.000 s | 386.000 s |
| Union duration | 445.800 s | 455.217 s |
| Symmetric-difference duration | 63.800 s | 69.217 s |
| Time precision | 0.9031 | 0.9125 |
| Time recall | 0.9437 | 0.9230 |
| Time F1 | 0.9229 | 0.9177 |
| Time IoU | 0.8569 | 0.8479 |
| Native ranges with any overlap | 55 / 59 | 55 / 59 |
| Browser ranges with any overlap | 54 / 55 | 53 / 56 |

For the primary libswresample/WASM reference, greedy chronological one-to-one
matching produced 54 overlapping pairs, 50 pairs at IoU >= 0.5, median absolute
start/end deltas of 0.250/0.275 seconds, and median pair IoU of 0.8518.

Native ranges with no overlap in the primary browser result:

- `179.750-182.250`
- `723.875-728.875`
- `791.500-794.000`
- `849.625-853.125`

Primary browser range with no overlap in the native result:

- `700.983-703.483`

The largest structural difference is the browser's continuous
`146.117-174.350` range. Native splits this into `146.375-160.375` and
`162.375-174.375`, leaving a two-second gap. This is why simple one-to-one event
counts make parity look slightly worse than the duration-union metrics.

## Conclusion

The Android port runs the complete audiovisual model successfully and preserves most
browser range time, but its result is not range-identical. The next parity work should
capture and compare native/browser feature matrices channel-by-channel, starting with
audio resampling and the YUV-derived visual channels. The completed cache makes those
model and decoder iterations roughly one second each without repeating media decode.
