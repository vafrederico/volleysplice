# VolleyCut native Android analysis benchmark

This folder contains a native Android port of the on-device analysis path. It is intentionally not an editor. Its only workflow is:

```text
video URI
  -> MediaExtractor presentation-order sample plan
  -> asynchronous MediaCodec decode with unsampled inputs marked decode-only
  -> 192x108 ROI samples at 4 Hz
  -> bounded FIFO worker for native OpenCV visual features
  -> zeroed audio-unavailable features (audio decode temporarily skipped)
  -> 104 base features
  -> whole-recording percentile ranks and +/-2 s context (520 columns)
  -> model-9c92b8e9333f rally, serve, and dead-state heads
  -> unpadded candidate ranges + stage timings
```

No media is uploaded. The app has no network permission. It does not use a WebView, WebCodecs, JavaScript, or WASM.

## Target device and SDK

- Target device: Pixel 10 Pro, arm64-v8a
- `compileSdk`: 36
- `targetSdk`: 36
- `minSdk`: 29
- Android Gradle Plugin: 9.1.1
- Gradle: 9.3.1
- Java: 17 bytecode
- OpenCV Android AAR: 4.12.0

The Pixel 10 Pro can run this API 36-targeted app normally on Android 17. The analysis path does not call Android 17-specific APIs, and targeting API 37 would not accelerate `MediaCodec`, OpenCV, or the Java DSP. Keeping API 36 as the default also avoids opting into unrelated Android 17 target-behavior changes during this performance experiment. The project intentionally packages only `arm64-v8a`, since its immediate purpose is measuring the Pixel 10 Pro rather than running an x86 emulator.

## Build and install

Open `android/` as an Android Studio project, select the Pixel 10 Pro, and run the `app` configuration. From a terminal with `JAVA_HOME` and `ANDROID_HOME` configured:

```powershell
cd android
.\gradlew.bat testDebugUnitTest assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

SDK 36 is the default, so no Gradle property overrides are needed. The app still executes on the Pixel's Android 17 runtime and uses the device's installed codec implementations.

## Run a benchmark

1. Tap **Choose video** and select a local proxy or original master through Android's system picker.
2. Leave **Use full frame** off for the canonical recordings. The app recognizes the same nine filename profiles as the web POC and otherwise uses the same indoor default ROI.
3. Tap **Run native analysis**. The benchmark stops after 1,000 presentation-order source video frames (about 16.7 seconds at 60 FPS), and bounds inference and ranges to the same measured media window. It decodes reference frames normally but uses Android's decode-only flag to avoid materializing unsampled output images. Audio decode/DSP is temporarily skipped so this experiment isolates the video pipeline. Shorter clips run to completion. The app keeps the display awake and requests sustained-performance mode for the duration of the run.
4. During video analysis, watch the live sampled frames/second, feature real-time ratio, elapsed time, ETA, decoded source-frame count, and Java heap use.
5. Record the final feature speed, overall real-time multiplier, audio speed, decoder names, stage times, and range count. **Copy result JSON** copies those values and exact unpadded ranges.

### Automated adb benchmark

The debug build accepts an explicit MediaStore URI and can start analysis immediately. The runner builds, installs, keeps the connected phone awake, finds the named video in MediaStore, launches it with a temporary read grant, and polls the app's private result file through `run-as`:

```powershell
cd android
.\benchmark.bat -VideoName "1080p60.mp4" -FrameLimit 1000 -Runs 3
```

Use `-SkipBuild` or `-SkipInstall` while iterating, and `-SummaryOnly` to suppress the full JSON lines. Every successful run contributes to the median summary. The app writes `benchmarkStatus`, a unique `benchmarkRunId`, and either the normal result payload or structured failure details, so a stale result cannot be mistaken for the current run. `-FrameLimit` can extend a measurement when a 1,000-frame sample is too noisy. The production configuration requests a 240 FPS codec operating rate at best-effort priority; pass `-OperatingRate -1 -CodecPriority -1` for the unhinted control.

The 240 FPS operating-rate request was retained after a 5,000-frame 1080p60 A/B reduced median video-stage time from 29,102.5 ms to 19,749.0 ms (32.1%) with identical sampled timestamps and candidate ranges. Android uses this value for codec resource planning; it does not change source timestamps or the 4 Hz sampling schedule.

Asynchronous decode-only output was then retained after reducing the same 5,000-frame median from 19,749.0 ms to 14,683.5 ms (25.6% further, 49.5% versus the unhinted baseline). It produced only 334 output images for 5,000 decoded source frames. The 1080p and 4K checks retained identical analyzed durations, sample timestamps, ranges, and confidences; three warm 4K60 runs had a 5,884 ms median for 1,000 frames.

The feature speed matches the web POC definition: `generatedFrames / 4 Hz / videoFeatureWallSeconds` for the real-time ratio, and `generatedFrames / videoFeatureWallSeconds` for frames/second. The overall ratio additionally includes audio extraction, contextualization, and inference.

The completed report includes a hierarchical profiling breakdown. Parent and child rows intentionally overlap and should not be added together. Asynchronous video callbacks/YUV conversion and ordered OpenCV extraction run concurrently, connected by a two-sample bounded queue. The report records decode-only inputs, actual decoder outputs, queue backpressure, and final worker-drain time. It also covers the presentation-order planning scan, codec input/output callbacks, demux reads, YUV crop/scale/color conversion, OpenCV filters/readback/phase correlation/Farneback flow/reductions, percentile ranking/context gathering, and every inference/decoder phase. Per-unit time, wall-time percentage, total analysis CPU, GC time, Java/native/PSS memory, sampling timestamp error, and thermal status are included. The copied JSON retains all raw millisecond totals under `profileMilliseconds`.

For a useful WebCodecs-versus-native comparison, use the same physical source file, ROI, model bundle, charging state, and cold/warm-run policy. Run at least three times after one warm-up at a similar starting temperature. Android may select different codec implementations after thermal throttling, so keep the reported decoder name with every result.

## What is parity-tested

The JVM golden test reads the repository's frozen 3,474 x 104 Y9 base-feature fixture, performs the Android port's ranking/contextualization and inference, and reproduces all 37 canonical ranges (including confidence tolerance). This isolates and validates the code after feature extraction.

The media front end is deliberately a native-distribution experiment, not a claim of feature parity:

- Android supplies decoder YUV planes; the app converts those directly into the 192x108 analysis image. That color conversion and resize are not byte-identical to browser canvas or FFmpeg/OpenCV `INTER_AREA`.
- Video is decoded in one pass and the first presentation-order frame at or after each 4 Hz target is sampled. Web and offline frame-selection boundaries can differ by one source frame.
- Audio uses Android's decoded PCM and the existing linear 16 kHz resampling/DSP math. It does not embed FFmpeg `libswresample`.
- Native OpenCV implements phase correlation and Farneback flow. The algorithm settings and 73-channel schema match the web path, but native SIMD and float reductions can produce small numerical differences.

Those differences are why the app reports both performance and final ranges. If the native path is materially faster, the next step is to capture its base features for channel-by-channel comparison and then calibrate/retrain against the Android feature distribution rather than assuming browser/offline thresholds transfer perfectly.

## Important files

- `app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java`: asynchronous decode-only planning and 4 Hz YUV sampling
- `app/src/main/java/com/volleycut/nativeanalysis/VisualFeatureExtractor.java`: native OpenCV visual features
- `app/src/main/java/com/volleycut/nativeanalysis/NativeAudioDecoder.java`: platform audio decode
- `app/src/main/java/com/volleycut/nativeanalysis/AudioFeatureExtractor.java`: resampling, FFT, and audio feature schema
- `app/src/main/java/com/volleycut/nativeanalysis/ModelRunner.java`: exact three-head inference and range decoding
- `app/src/test/java/com/volleycut/nativeanalysis/GoldenModelTest.java`: frozen-feature model parity test
