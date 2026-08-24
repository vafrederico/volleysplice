# Android serving-side feature benchmark — 2026-08-23

> Implementation follow-up: the Android team-switch port now uses the recommended
> shared gap-aware decoder with a fixed five-second maximum forward gap. See
> [Android team-switch production port](./android-team-switch-production-port-2026-08-23.md)
> for the completed Pixel 10 combined-pipeline result. The sequential measurements
> below remain the pre-change serving-side baseline.

> A post-port sync-frame audit confirmed five seconds as the actual minimum-work
> threshold for the combined 1,210-frame schedule, not merely the static-gap choice.
> The final exact-parity combined run took 463.954 seconds, down from 882.845 seconds
> before the decoder/conversion fixes. Its phase breakdown is in the linked port note.

## Decision

The Android serving-side decoder is already a full sequential pass. It opens one
`MediaExtractor` and one `MediaCodec`, seeks once to the first requested timestamp,
and decodes every source frame through the last requested timestamp. Replacing a
seek-heavy implementation with that policy therefore cannot improve the current
Android path: that replacement has already happened.

On the Pixel 10 Pro workload below, the full sequential traversal is the dominant
bottleneck. Its two-run midpoint was 442.89 seconds for decode and 8.81 seconds for
all serving-side feature math. A deliberately pessimistic control that reopened the
extractor and codec once per rally made 52 seeks but retained byte-identical sampled
frames and raw features. It reduced the two-run end-to-end midpoint from 451.70 to
346.99 seconds, a 104.71-second or 23.2% improvement.

Do not add a second full-video pass for team-side switching, and do not preserve the
serving-side full traversal merely because it is sequential. Build the union of the
serving-side and side-switch timestamp schedules, then use a gap-aware segmented
decoder: decode forward inside a dense segment and seek/flush to the previous sync
sample across a sufficiently large positive gap. Reuse one codec if device parity
allows it. Select the gap threshold from device measurements with exact sampled-frame
and feature parity gates.

## Device and workload

| Property | Value |
| --- | --- |
| Device | Pixel 10 Pro (`blazer`) |
| OS | Android 17 / API 37 |
| Source | `1080p60.mp4` from the existing native project `project-mqmfyy` |
| Media | H.264, 1,920×1,080, 60 fps, 1,105.817 seconds, 1,964,677,385 bytes |
| ROI | Existing project indoor-camera ROI |
| Serving candidates | 52 merged production ranges |
| Per-candidate schedule | 17 model references, 16 distinct timestamps because both banks share `+0.55 s` |
| Distinct requested frames | 832 |
| Requested span | 110.50–1,078.75 seconds, or 968.25 seconds |

The source was local to the phone. The display was kept awake. The first run began
with the device at 27.8 °C; completed-run temperatures were 35.8–36.6 °C. Later runs
were on AC power. This is a representative device benchmark, not a thermally controlled
laboratory result, so the paired improvement range is reported as well as the midpoint.

## Methods

The instrumentation benchmark is
[`ServingSidePipelineBenchmarkInstrumentedTest.kt`](../../android/app/src/androidTest/java/com/volleycut/nativeanalysis/ServingSidePipelineBenchmarkInstrumentedTest.kt).
It keeps OpenCV initialization and a static first-use warm-up outside the measured
sections, then times decode and the unchanged 82+155 feature computation separately.

Two strategies use the same `ServingSideFrameDecoder` frame-selection and
pixel-conversion code:

- `sequential`: pass all 832 sorted timestamps in one call with an unbounded forward
  gap. This reproduces the production behavior at the time of the benchmark: one
  extractor, one codec, one seek, and a 968.25-second forward traversal.
- `candidate-seeks`: call the same decoder once for each rally's 16 distinct
  timestamps. This is an intentionally high-overhead control with 52 extractor
  instances, 52 codec instances, and 52 seeks. Each local requested window spans
  three seconds before sync-sample preroll.

Every run hashes timestamps plus sampled grayscale bytes and hashes all 12,324 raw
feature doubles (`52 × 237`). A speed result is accepted only when both hashes match.

## Results

| Strategy | Run | Decoder sessions / seeks | Decode | Features | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current sequential | 1 | 1 / 1 | 459.256 s | 8.950 s | 468.205 s |
| Current sequential | 2 | 1 / 1 | 426.530 s | 8.660 s | 435.190 s |
| Current sequential midpoint | — | 1 / 1 | **442.893 s** | **8.805 s** | **451.698 s** |
| Per-candidate seek control | 1 | 52 / 52 | 339.709 s | 8.741 s | 348.450 s |
| Per-candidate seek control | 2 | 52 / 52 | 336.725 s | 8.805 s | 345.530 s |
| Per-candidate seek midpoint | — | 52 / 52 | **338.217 s** | **8.773 s** | **346.990 s** |

The seek-heavy control improved end-to-end time by 25.6% in the first pair and 20.6%
in the second pair. The two-run midpoint gain is 23.2%. The current sequential decoder
accounts for 98.1% of its midpoint total, while all OpenCV serving-side feature math
accounts for only 1.9%.

All four runs produced the same identities:

```text
frameSha256      dee9ba322dd149bc9e6c9a681f3c0c4762ad21441eba33448b9ab54a0513b0aa
rawFeatureSha256 5fa9cfd5ad69e8753c6ae9458c07386e9fb1807e41a85487a8956f46d8506178
```

The control's gain is conservative with respect to a sensible segmented decoder
because it pays codec/extractor construction 52 times. It is nevertheless conclusive
for the question tested: seeking is not the current problem, and a full sequential
pass is not a speed improvement for this sparse serving-side schedule.

## Schedule-gap diagnostic

The 832 timestamps contain 51 inter-candidate gaps greater than three seconds. The
following static calculation groups the same timestamp schedule when a gap greater
than the stated threshold starts a new decoder segment. It excludes sync-sample
preroll and codec setup, so it is a planning diagnostic rather than a runtime estimate.

| Maximum forward gap | Segments | Planned forward span | Avoided versus current |
| ---: | ---: | ---: | ---: |
| 5 s | 49 | 170.000 s | 798.250 s |
| 8 s | 46 | 189.750 s | 778.500 s |
| 10 s | 41 | 233.375 s | 734.875 s |
| 12 s | 32 | 332.000 s | 636.250 s |
| 15 s | 27 | 400.375 s | 567.875 s |
| 20 s | 15 | 615.250 s | 353.000 s |

The next implementation experiment should benchmark at least 5, 10, 15, and 20
seconds using the unioned serving-side plus side-switch schedule. For each threshold,
record extractor seeks, codec flushes/restarts, compressed inputs, decoder outputs,
converted frames, decode wall time, feature wall time, and both parity hashes. A gap
threshold must be fixed for the shared schedule; it must not vary by model outcome.

## Post-port sync-frame audit

`SpecialistDecodePlanInstrumentedTest` sought the real Pixel source to the previous
sync sample at each proposed segment start. This includes GOP preroll omitted by the
static diagnostic and predicts the measured 43,058 decoder outputs within 53 frames.

| Maximum forward gap | Segments | Sync-aware decoded span | Estimated decoded frames |
| ---: | ---: | ---: | ---: |
| 2 s | 86 | 820.617 s | 49,238 |
| 3 s | 58 | 748.954 s | 44,938 |
| **5 s** | **39** | **716.733 s** | **43,005** |
| 8 s | 30 | 746.792 s | 44,808 |
| 10 s | 19 | 792.983 s | 47,580 |
| 15 s | 7 | 886.158 s | 53,170 |
| Full sequential | 1 | 978.767 s | 58,727 |

The result explains why sparse score sampling remains comparable to the main video
pass: previous-sync preroll makes the five-second plan decode about 717 seconds of
1080p60 footage even though only 1,210 output timestamps are materialized. Shorter
thresholds repeat too much GOP preroll; longer thresholds decode too much intervening
video.

## Reproduction

Build and install the debug and instrumentation APKs, retain a ready native project on
the device, and run either strategy:

```powershell
adb -s <pixel-10-serial> shell am instrument -w -r `
  -e class com.volleycut.nativeanalysis.ServingSidePipelineBenchmarkInstrumentedTest `
  -e serving_side_benchmark true `
  -e serving_side_strategy sequential `
  -e serving_side_project_id project-mqmfyy `
  com.volleycut.nativeanalysis.debug.test/androidx.test.runner.AndroidJUnitRunner
```

Use `gap-aware` for the current production policy, `sequential` for the historical
baseline, and `candidate-seeks` for the control. Results are emitted under the
`VolleyCutServingBench` log tag. The benchmark is test-only and does not alter
persisted project output.
