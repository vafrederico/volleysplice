# Android emulator visual-pipeline regression

The Pixel 10 Pro emulator configuration with a 24 GB data partition reproduces
the expensive area-conversion operator. This measures x86 CPU code in an emulator,
not Tensor hardware throughput. Observed guest RAM was about 4 GB; 24 GB refers
to storage. Writable emulator data, build artifacts and private receipts use the NAS.

## Direct conversion measurement

The test uses one deterministic 1920×1080 YUV420 fixture with row padding and
interleaved chroma strides, reduced to 192×108. It compares the preserved old
point loop with the deployed area resampler, using identical color conversion.
After warmup, three timed batches alternate order. Allocation, MediaCodec,
OpenCV features, embeddings and models are excluded. Both outputs are consumed
and checked for deterministic checksums. Their pixels are deliberately different.

| Conversion | Median wall time/frame | Measured range | Median thread CPU/frame |
|---|---:|---:|---:|
| Previous point sampling | 1.433 ms | 1.344–1.517 ms | 1.410 ms |
| Repaired area sampling | 142.669 ms | 141.959–150.312 ms | 141.456 ms |

The area operator costs 99.53× as much wall time and 100.31× as much thread CPU
in this fixture. This supports a computational regression independently of video
decoder waits. It is not an end-to-end slowdown ratio: the device pipeline includes
other work, and the emulator differs from the physical Pixel in CPU, clocks and
memory behavior. All four instrumented tests passed: this microbenchmark and the
three existing comparisons against native OpenCV area resizing.

## Repeated full-pipeline comparison

Two x86_64 APKs are built with identical corrected audio, benchmark harness,
native dependencies and frozen highest-recall distilled Large weights. The
baseline restores only the earlier video decoder and encoder frame-selection
code; the repaired build retains the current implementation. The same hash-verified
two-minute excerpt of `recording-044` is used. The completed order was one warmup
per version, then baseline, repaired, repaired, baseline. Warmups are excluded
from the following medians and ranges.

| Stage | Old visual pipeline | Repaired visual pipeline | Change |
|---|---:|---:|---:|
| All results ready | 276.079s | 352.335s | +27.6% |
| Rallies ready | 223.773s | 295.250s | +31.9% |
| Video features | 108.947s | 175.515s | +61.1% |
| Audio features | 6.655s | 6.689s | +0.5% |
| Embedding pass | 107.512s | 112.423s | +4.6% |
| Score specialists | 52.306s | 57.085s | +9.1% |

The two measured baseline totals were 275.804s and 276.354s; repaired totals were
353.829s and 350.840s. All six complete passes generated 480 AV/fused rows,
240 embedding samples, six neural rally intervals, and both score outputs. These
interval counts are descriptive, not a human-label accuracy evaluation.

The detailed counters isolate the cause:

| Detail | Old median | Repaired median | Interpretation |
|---|---:|---:|---|
| Video resize/color conversion | 1.935s | 68.003s | Large regression reproduced, about 35x in the complete pipeline |
| AV timestamp inventory | 7.407s | 7.348s | No increase on this source; the old AV path already scanned it |
| OpenCV feature calls | 34.872s | 34.289s | No observed regression |
| Embedding timestamp inventory | absent | 6.594s | New scan cost reproduced |
| Embedding image preparation | 3.755s | 3.712s | Phone preparation slowdown not reproduced |
| Encoder/readback | 2.530s | 2.491s | No observed increase; bucket also includes bookkeeping |
| TCN inference | 0.036s | 0.034s | Negligible relative to preprocessing |

These are nested counters and must not be summed with stage totals. Video
preprocessing accounts for about 87% of the 76.256-second total median increase.
The encoder/readback timer also changed bookkeeping scope in the repair, so it
does not isolate pure model compute. No optimization or model selection was
performed during these measurements.

The initial full-pipeline warmup failed before timing: the API 37 Google Play
emulator image rejected MediaCodec shared buffers with a SELinux `memfd_file`
denial. Its user build kept SELinux enforcing even when launched with the emulator's
permissive flag. The completed pipeline runs instead used Google APIs revision 6,
API 37, an x86_64 userdebug image, host graphics and permissive SELinux. This is an
altered emulator environment, not production-device qualification. The startup
flag is documented by [Android](https://developer.android.com/studio/run/emulator-commandline).
Normal enforcement and non-root ADB were restored afterward, and the repaired
benchmark APK was left installed.

All saved tensors passed desktop temporal-inference and exact rally-boundary
replay. Frozen graph contracts, native x86 dependencies, source identity, corrected
audio, sampling counts and decoder settings were checked. Audio feature values
were identical across all runs. Every recorded wake-state observation was awake;
host available memory remained above 3.6 GB. Periodic observations do not establish
that every possible background scheduling effect was absent.

The emulator is suitable for testing area-resampler optimizations while checking
output correctness. The unexplained embedding-preparation increase on the phone
did not appear in the repeated emulator runs and still needs device-specific
profiling. Faster area conversion and shared timestamp planning can be developed
here; optimized complete-pipeline latency still needs confirmation on the phone.

The subsequent [controlled Pixel retest](pixel-visual-pipeline-retest.md) reproduces
the resizing regression on the same two-minute clip. Embedding image preparation
also remained stable on the phone in these repeats; the earlier full-video increase
remains unexplained by the short-clip results.

[Structured observations](android-emulator-visual-regression.json).
