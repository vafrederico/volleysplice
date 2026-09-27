# Android exact-area preprocessing optimization

The shared Android video resizer now uses a native C++ kernel. On the two-minute
recording-044 excerpt, median video-feature time fell from **168.1s to 107.3s**;
the complete Distilled MobileNetV3-Large + TCN highest-recall FP32 pipeline,
including both score specialists, fell from **339.3s to 278.9s (17.8%)**.

This preserves the repaired pixel contract. All four runs produced byte-identical
embeddings, fused features and probabilities, and identical neural rallies,
production rallies, serving-side results and side-switch results. There was no
training, threshold change, accuracy selection, or new full-video evaluation.

The subsequent [full-video validation](android-native-area-full-video.md) is now
complete: 40m59s for all inference on the emulator, 37 rally intervals and zero
wholly missed saved human rallies. It extends the runtime/accuracy check; there
is no paired full-video Java control.

| Stage, median seconds | Java area reference | Native area | Change |
| --- | ---: | ---: | ---: |
| AV resize/color conversion | 63.046 | 4.168 | -93.4% |
| Complete video-feature extraction | 168.062 | 107.341 | -36.1% |
| Audio features | 6.846 | 6.811 | -0.5% |
| Embedding pipeline | 109.119 | 109.133 | +0.01% |
| Rallies ready | 284.621 | 223.943 | -21.3% |
| Both score specialists | 54.709 | 54.984 | +0.5% |
| All inferences ready | 339.330 | 278.927 | -17.8% |

Nested counters overlap and must not be added. Individual all-ready runs were
339.933s and 338.727s for Java, versus 278.788s and 279.065s for native.

## Why it was slow and what changed

Exact 1080p-to-192x108 area reduction needs roughly one billion source-pixel color
conversions across this excerpt. The repaired Java implementation read direct YUV
buffers, converted and clipped each pixel, and accumulated each output bin. The
native kernel removes the per-pixel Java/direct-buffer overhead. It retains the
same rounded color tables, RGB clipping before area averaging, ties-to-even output
rounding, fractional area weights, strides, crops and rotations. It never allocates
a full-resolution RGB frame. Java remains the portable fallback and test reference.

The production app and benchmark share this implementation. ARM64 debug and
unsigned release builds passed; R8 preserves the JNI bridge. The new library is
about 213 KB uncompressed and supports 16 KB page alignment. Adding NDK compilation
also strips debug symbols from existing packaged inference libraries: stripping
the reference copies produces identical files. Bundled model assets are unchanged.

## Validation and limits

- 194 JVM tests and seven emulator instrumentation tests passed.
- Native pixels exactly matched Java for 98 cases, including fractional/integer
  scaling, high-resolution inputs, rotations, color matrices/ranges, cropped and
  strided planes, sliced direct buffers, and truncated-plane rejection.
- Existing OpenCV INTER_AREA reference tests passed against the native path.
- The warmed operator microbenchmark improved from 133.6 to 6.31 ms/frame (21.2x).
- Four fresh-process, cache-bypassed pipeline runs used Java/native/native/Java
  order. No pipeline warmups were discarded. Every temporal/decoder replay passed.
- The API 37 x86_64 emulator used the previously documented debug-image SELinux
  workaround and was shut down afterward. These are emulator results, not physical
  Pixel timing claims. Physical-device validation remains outstanding.

Raw inputs, APKs, logs and tensors resolve through private-reference-0223. The
[machine-readable report](android-native-area-optimization.json) includes timings,
hashes, equality checks, build receipts and environment details.
