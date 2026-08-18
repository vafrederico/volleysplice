# VolleyCut native Android analysis and editor

This folder contains the native Android analysis path plus a first-party cut editor and exporter. The analysis workflow is:

```text
video URI
  -> MediaExtractor presentation-order sample plan
  -> asynchronous MediaCodec decode with unsampled inputs marked decode-only
  -> 192x108 ROI samples at 4 Hz
  -> bounded FIFO worker for native OpenCV visual features
  -> MediaCodec audio decode + native resampling/FFT features
  -> 104 base features
  -> game-window percentile ranks and +/-2 s context (520 columns)
  -> all-labels-v2 and previous-production rally/serve/dead-state stacks
  -> overlap-union-disagreement-v1 production ensemble
  -> unpadded candidates with model-agreement provenance + stage timings
```

No media is uploaded. The app has no network permission. It does not use a WebView, WebCodecs, JavaScript, or WASM.

In the production project flow, choose a recording and use the local preview to mark the game start and game end before queueing inference. Only globally aligned 4 Hz samples inside that window generate visual, audio, or contextual features. The bounds are part of the project and feature-cache identity, and the editor overview, playback, padding, manual marks, edit list, and export are constrained to the same window. Existing full-video projects and caches keep their legacy identity.

## Target device and SDK

- Target device: Pixel 10 Pro, arm64-v8a
- `compileSdk`: 37
- `targetSdk`: 37
- `minSdk`: 29
- Android Gradle Plugin: 9.1.1
- Gradle: 9.3.1
- Java: 17 bytecode
- OpenCV Android AAR: 4.12.0

API 37 does not make the analysis kernels faster by itself, but it is now the default so the app is tested against the Pixel 10 Pro's Android 17 target behavior. The project intentionally packages only `arm64-v8a`, since its immediate purpose is the Pixel 10 Pro rather than an x86 emulator.

## Build and install

Open `android/` as an Android Studio project, select the Pixel 10 Pro, and run the `app` configuration. From a terminal with `JAVA_HOME` and `ANDROID_HOME` configured:

```powershell
cd android
.\gradlew.bat testDebugUnitTest assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

SDK 37 is the default, so no Gradle property overrides are needed. Install the Android 17 SDK Platform 37.0 and Build-Tools 37.0.0 before building from the command line.

## Native editor and export

Run full inference, then tap **Open native cut editor**. The editor uses the unpadded inferred ranges as its immutable cores and provides:

- global before/after output padding for inferred ranges;
- configurable joining of positive gaps shorter than 0-10 seconds (3 seconds by default), retained in preview and export;
- a game-window timeline and a focused range timeline with draggable handles;
- exact source seeking, 1x/2x/4x/8x playback, and final-cut-only preview;
- keep/remove review, mandatory single-model disagreement review, confidence review, 0.1/1 second nudges, and per-range reset;
- manual missed cuts and ignored source sections;
- atomic, versioned draft persistence and a **Resume native cut editor** entry after process restart;
- source-availability detection and validated video re-linking that preserves inference, feature
  cache identity, and saved editor corrections after a recording is moved or its document grant expires;
- edit-list JSON output and an exact-boundary MP4 export with progress, cancellation, and encoder telemetry.
- document-saved model-feedback JSON containing source-aligned base features,
  probability traces, original inference, corrections, ignored intervals, and final export ranges,
  without embedding raw video bytes.

The player is Media3 ExoPlayer. Export builds a Media3 `Composition` from the final merged ranges after ignored sections are subtracted, then uses Transformer to encode AVC video and AAC audio into MP4. Export runs as an Android `mediaProcessing` foreground service. A custom Media3 muxer factory writes directly to the seekable file descriptor returned for the document selected with Android's system picker, so normal local exports do not need duplicate temporary storage. Streaming-only document providers automatically retain the app-cache-and-copy fallback. Failed, cancelled, or abandoned exports remove their incomplete destination document. The service logs a `VolleyCutExport` JSON record containing the selected `outputWriteMode`, wall time, real-time ratio, frame rate, encoder names, bitrates, output geometry, file size, conversion modes, and failures.

All new UI/media dependencies are open source: Jetpack Compose, AndroidX Activity/Core, and Media3 are Apache-2.0. OpenCV remains Apache-2.0. No commercial editor SDK is embedded.

## Run a benchmark

1. Tap **Choose video** and select a local proxy or original master through Android's system picker.
2. Leave **Use full frame** off for the canonical recordings. The app recognizes the same nine filename profiles as the web POC and otherwise uses the same indoor default ROI. Leave the 1,000-frame limit checked for performance experiments; uncheck it for full-recording inference.
3. Leave **Reuse saved video + audio features** checked for normal work, or uncheck it to bypass cache reads and writes. **Clear cache** removes all saved matrices. Tap **Run native analysis**. The default benchmark stops after 1,000 presentation-order source video frames (about 16.7 seconds at 60 FPS), and bounds video, audio, inference, and ranges to the same measured media window. It decodes reference frames normally but uses Android's decode-only flag to avoid materializing unsampled output images. Shorter clips run to completion. The app keeps the display awake and requests sustained-performance mode for the duration of the run.
4. During video analysis, watch the live sampled frames/second, feature real-time ratio, elapsed time, ETA, decoded source-frame count, and Java heap use.
5. Record the final feature speed, overall real-time multiplier, audio speed, decoder names, stage times, and range count. **Copy result JSON** copies those values and exact unpadded ranges.

### Automated adb benchmark

The debug build accepts an explicit MediaStore URI and can start analysis immediately. The runner builds, installs, keeps the connected phone awake, finds the named video in MediaStore, launches it with a temporary read grant, and polls the app's private result file through `run-as`:

```powershell
cd android
.\benchmark.bat -VideoName "1080p60.mp4" -FrameLimit 1000 -Runs 3
```

Select individual pipeline stages or combinations with `-Stages`. For example, this runs only
fresh audio decode and feature generation over the first 60 seconds on a specific ADB transport:

```powershell
.\benchmark.ps1 -DeviceSerial "192.0.2.1:40461" -VideoName "1080p60.mp4" `
  -Stages Audio -DurationSeconds 60 -CacheMode Bypass -AudioDecoderMode Batched -Runs 5
```

Accepted stage plans are `All`, `Video`, `Audio`, `Inference`, `VideoAudio`,
`VideoInference`, and `AudioInference`. `-DurationSeconds 0` uses the full selected scope.
The source-frame limit applies only when video generation is selected. Inference without one or
both feature stages loads those exact prerequisites from the matching feature cache and fails
clearly when they are unavailable, keeping an inference-only measurement honest.
Use `-AudioDecoderMode Single` for the one-access-unit control, `Batched` for Android's
asynchronous multi-access-unit path, or `Auto` for the production selection. Normal app analysis
selects batching automatically on API 35+ when the decoder advertises `FEATURE_MultipleFrames`,
and otherwise falls back to the control path.

Use `-SkipBuild` or `-SkipInstall` while iterating, and `-SummaryOnly` to suppress the full JSON lines. Every successful run contributes to the median summary. The app writes `benchmarkStatus`, a unique `benchmarkRunId`, and either the normal result payload or structured failure details, so a stale result cannot be mistaken for the current run. `-FrameLimit` can extend a measurement when a 1,000-frame sample is too noisy. Use `-CacheMode Use`, `Bypass`, or `Refresh` to reuse, ignore, or replace the matching feature entry. The production configuration requests a 240 FPS codec operating rate at best-effort priority; pass `-OperatingRate -1 -CodecPriority -1` for the unhinted control.

The 240 FPS operating-rate request was retained after a 5,000-frame 1080p60 A/B reduced median video-stage time from 29,102.5 ms to 19,749.0 ms (32.1%) with identical sampled timestamps and candidate ranges. Android uses this value for codec resource planning; it does not change source timestamps or the 4 Hz sampling schedule.

The audio-only stage selector exposed a similar codec scheduling opportunity. On the Pixel 10 Pro,
five fresh 60-second runs of `PXL_20260816_160023210.mp4` improved from a 6,276 ms baseline median
to 5,277 ms (15.9%) after requesting a 4x-source-rate audio operating rate with real-time codec
priority. An 8x request regressed to 6,354 ms, while 4x with best-effort priority measured 5,877 ms.
The retained request is reported in benchmark JSON. It affects scheduling only; decoded timestamps,
PCM, and feature math are unchanged.

A full 885.1-second single-access-unit audio-only control took 160,381 ms (5.52x real time). The
accepted Android 17 asynchronous multi-access-unit path reduced that to 61,062 ms (14.50x), a 61.9%
reduction, while collapsing 41,489 codec input submissions and outputs into 2,680 batches. Five fresh
60-second runs improved from a 6,213 ms control median to 1,970 ms (68.3%). Exact SHA-256 fingerprints
of all pooled audio features matched at 10 seconds, 60 seconds, and the full recording. The batching
path retains original access-unit timestamps and reconstructs PCM boundaries before the unchanged
resampler/FFT pipeline; per-unit end-window clipping is also preserved exactly.

After batching, full-file `MediaExtractor.advance()` remains the scaling limit at 39.3 seconds of the
61.1-second run. Codec input queueing fell to 3.8 seconds, output release to 1.4 seconds, and final
whole-recording reductions plus pooling remained about 300 ms. A bounded codec-to-DSP worker and
reusable FFT/source buffers were rejected because CPU contention or buffer clearing erased their
theoretical overlap. Further gains should therefore target extractor traversal or a different demux
path rather than requiring the complete recording to be decoded before DSP begins; codec decode and
feature extraction already stream concurrently in the asynchronous path.

Asynchronous decode-only output was then retained after reducing the same 5,000-frame median from 19,749.0 ms to 14,683.5 ms (25.6% further, 49.5% versus the unhinted baseline). It produced only 334 output images for 5,000 decoded source frames. The 1080p and 4K checks retained identical analyzed durations, sample timestamps, ranges, and confidences; three warm 4K60 runs had a 5,884 ms median for 1,000 frames.

The YUV crop/scale/color sampler now precomputes source-plane offsets and exact integer conversion lookup tables. A 5,000-frame 1080p60 check fell from 14,683.5 ms to 13,429.5 ms (8.5%) with identical ranges and confidences. The 1,000-frame median fell from 3,026 ms to 2,837 ms (6.2%). The decoder-bound 4K60 case improved more modestly, from 5,884 ms to 5,797 ms (1.5%). Batched compressed input was rejected: although it reduced codec calls, the Pixel decoder produced different visual features and ranges.

Raw visual features are checkpointed to app-private storage every 16 analysis rows. An interrupted run resumes by decoding one cached sample as temporal warm-up and then appending new rows. Completed visual chunks are compacted into one file, and the audio and contextual matrices are saved as well. Later model runs therefore skip both decoders and game-window contextualization. Cache identity includes source metadata, media geometry/codecs, ROI, feature schemas, analysis rate, source-frame limit, and any non-default game window.

Strict constant-frame-rate inputs now avoid the full timestamp planning scan after a 128-access-unit probe verifies that the declared frame rate predicts presentation timestamps exactly and measures the stream's reorder depth. VFR and inconsistent inputs retain the scan-based path. This reduced the 5,000-frame 1080p60 median from 13,429.5 ms to 13,002 ms (3.2%) with identical analyzed duration, ranges, and confidences. The 1,000-frame 4K60 median improved from 5,797 ms to 5,673 ms (2.1%).

The feature speed matches the web POC definition: `generatedFrames / 4 Hz / videoFeatureWallSeconds` for the real-time ratio, and `generatedFrames / videoFeatureWallSeconds` for frames/second. The overall ratio additionally includes audio extraction, contextualization, and inference. Cache-hit runs report cache state explicitly; their decoder throughput fields describe saved work rather than a new decode benchmark.

The completed report includes a hierarchical profiling breakdown. Parent and child rows intentionally overlap and should not be added together. Asynchronous video callbacks/YUV conversion and ordered OpenCV extraction run concurrently, connected by a two-sample bounded queue. The report records decode-only inputs, actual decoder outputs, queue backpressure, and final worker-drain time. It also covers the presentation-order planning scan, codec input/output callbacks, demux reads, YUV crop/scale/color conversion, OpenCV filters/readback/phase correlation/Farneback flow/reductions, percentile ranking/context gathering, and every inference/decoder phase. Per-unit time, wall-time percentage, total analysis CPU, GC time, Java/native/PSS memory, sampling timestamp error, and thermal status are included. The copied JSON retains all raw millisecond totals under `profileMilliseconds`.

For a useful WebCodecs-versus-native comparison, use the same physical source file, ROI, model bundle, charging state, and cold/warm-run policy. Run at least three times after one warm-up at a similar starting temperature. Android may select different codec implementations after thermal throttling, so keep the reported decoder name with every result.

## What is parity-tested

The JVM golden test reads the repository's frozen 3,474 x 104 Y9 base-feature fixture, performs the Android port's ranking/contextualization and all-labels-v2 inference, and reproduces all 42 canonical ranges (including confidence tolerance). Separate tests verify both bundled model hashes and the exact production overlap/union/confidence contract. This isolates and validates the code after feature extraction.

The media front end is deliberately a native-distribution experiment, not a claim of feature parity:

- Android supplies decoder YUV planes; the app converts those directly into the 192x108 analysis image. That color conversion and resize are not byte-identical to browser canvas or FFmpeg/OpenCV `INTER_AREA`.
- Video is decoded in one pass and the first presentation-order frame at or after each 4 Hz target is sampled. Web and offline frame-selection boundaries can differ by one source frame.
- Audio uses Android's decoded PCM and the existing linear 16 kHz resampling/DSP math. It does not embed FFmpeg `libswresample`.
- Native OpenCV implements phase correlation and Farneback flow. The algorithm settings and 73-channel schema match the web path, but native SIMD and float reductions can produce small numerical differences.

Those differences are why the app reports both performance and final ranges. If the native path is materially faster, the next step is to capture its base features for channel-by-channel comparison and then calibrate/retrain against the Android feature distribution rather than assuming browser/offline thresholds transfer perfectly.

The first full audiovisual run and its interval-level comparison against the stored browser results are recorded in [`FULL_INFERENCE_PARITY.md`](FULL_INFERENCE_PARITY.md).

## Important files

- `app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java`: asynchronous decode-only planning and 4 Hz YUV sampling
- `app/src/main/java/com/volleycut/nativeanalysis/VisualFeatureExtractor.java`: native OpenCV visual features
- `app/src/main/java/com/volleycut/nativeanalysis/NativeAudioDecoder.java`: platform audio decode
- `app/src/main/java/com/volleycut/nativeanalysis/AudioFeatureExtractor.java`: resampling, FFT, and audio feature schema
- `app/src/main/java/com/volleycut/nativeanalysis/NativeFeatureCache.java`: restart-safe visual checkpoints and completed audio matrices
- `app/src/main/java/com/volleycut/nativeanalysis/ModelRunner.java`: exact three-head inference and range decoding for either bundled model
- `app/src/main/java/com/volleycut/nativeanalysis/ProductionEnsemble.java`: production overlap union and disagreement-confidence policy
- `app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt`: native player, recording/focused timelines, and editing controls
- `app/src/main/java/com/volleycut/nativeanalysis/EditorModels.kt`: final-range interval algebra and editor state
- `app/src/main/java/com/volleycut/nativeanalysis/EditorDraftStore.kt`: restart-safe edit persistence
- `app/src/main/java/com/volleycut/nativeanalysis/EditorProjectStore.kt`: last-project metadata used by editor resume
- `app/src/main/java/com/volleycut/nativeanalysis/DirectUriMp4MuxerFactory.kt`: seekable document-descriptor MP4 output
- `app/src/main/java/com/volleycut/nativeanalysis/ExportService.kt`: foreground Media3 composition export and telemetry
- `app/src/test/java/com/volleycut/nativeanalysis/EditorMathTest.kt`: merge, ignored-time, padding, and seed identity tests
- `app/src/test/java/com/volleycut/nativeanalysis/GoldenModelTest.java`: frozen-feature model parity test
