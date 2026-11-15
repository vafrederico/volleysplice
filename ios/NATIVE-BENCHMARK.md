# Native macOS analysis benchmark

`VolleySpliceBenchmark` is a headless macOS XCTest host for the production
analysis pipeline. It compiles the same media readers, feature extractors,
OpenCV bridges, ONNX sessions and project serializer as iOS. It shares the
`NeuralPipelineIntegrationTests` and frozen ONNX golden tests. It is not a macOS
product build and has no editor or export UI.

This target lets AVFoundation select macOS video decoders independently of the
iOS Simulator runtime. Hardware availability is not inferred from a passing
benchmark or a GPU name: verify actual decoder activity externally. ONNX remains
on the existing four-thread CPU provider; this target does not enable Core ML.

## Dependencies and build

Stage the current checkout using `scripts/prepare-source.py`, as for iOS. Keep
recordings, generated Fixtures, dependencies, build products and reports outside
Git. macOS 14 or later and Xcode are required. The existing pinned ONNX Runtime
1.24.2 Swift package supplies its native macOS slice.

Build native OpenCV 4.12.0 from the upstream `opencv/opencv` source at commit
`49486f61fb25722cbcf586b7f4320921d46fb38e`. The iOS framework cannot be reused.
With `OPENCV_SOURCE`, `OPENCV_BUILD` and `VOLLEYCUT_MACOS_OPENCV_ROOT` pointing to
external source, build and installation directories, and CMake/Ninja on PATH:

```sh
cmake -S "$OPENCV_SOURCE" -B "$OPENCV_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_INSTALL_PREFIX="$VOLLEYCUT_MACOS_OPENCV_ROOT" \
  -DBUILD_LIST=core,imgproc,video -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_dnn=OFF \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF \
  -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DWITH_PROTOBUF=OFF \
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF
cmake --build "$OPENCV_BUILD" -j 4
cmake --install "$OPENCV_BUILD"
bash scripts/build-macos-benchmark.sh
```

The build helper also accepts extra `xcodebuild` options, such as
`-clonedSourcePackagesDirPath "$PACKAGE_CACHE"` to reuse an existing package cache.
Builds run serially. Both Debug targets use Swift `-O`, matching the simulator
benchmark's optimization level. Production iOS target membership is unchanged.

## Correctness before timing

```sh
xcodebuild -project VolleySplice.xcodeproj -scheme VolleySpliceBenchmark \
  -configuration Debug -destination "platform=macOS,arch=$(uname -m)" \
  -derivedDataPath build/macos-benchmark -parallel-testing-enabled NO \
  test-without-building
```

Without an external source environment variable, the optional benchmark skips;
the synthetic pipeline, cache, cancellation and ONNX golden tests still run.
A skip is not an executed video benchmark. The same assertions validate model
identity, finite features/scores, frame selection, cache behavior and project
serialization on both platforms.

## Timed run

Set `BENCHMARK_SOURCE` using the external private ledger, and use its stable
`RECORDING_INDEX` in the report. `XCTESTRUN` is the native `.xctestrun` emitted
under `build/macos-benchmark/Build/Products/`.

```sh
python3 scripts/run-neural-benchmark.py --platform macos \
  --xctestrun "$XCTESTRUN" --source "$BENCHMARK_SOURCE" \
  --source-index "$RECORDING_INDEX" --seconds 120 --output "$BENCHMARK_REPORT"
```

The runner uses a temporary test plan next to the original, preserves the
original, runs only the optional benchmark, and requires three successful model
runs. It retains the raw Xcode log beside the report with `.private.log`; do not
publish that log. The JSON report contains an index rather than the source path.
The simulator invocation remains supported using `--destination "$SIMULATOR_UDID"`
and the default `--platform simulator`.

Each model gets a fresh feature cache. The workload includes the selected rally
model, serving-side and side-switch preparation, and project roundtrip checks;
final video export is separate. Compare identical source windows, model assets,
optimization settings and VM resources. A macOS build may use different CPU
library implementations from the simulator, so the full timing difference is not
a pure measure of the GPU decoder. Do not extrapolate to iPhone throughput or
model accuracy from these lab timings.

## Verified run

On 2026-11-14, macOS 26.7.3 x86-64, GTX 970M / NVIDIA 580.196.07,
4 vCPUs and 8 GiB guest RAM, using the analysis sources and models from
`20eb7ac5`: the 120-second window of `recording-2026111401` passed all three
model runs with fresh caches and scoring specialists enabled.

| Model | iOS Simulator total (s) | Native macOS total (s) |
|---|---:|---:|
| Legacy | 150.95 | 78.08 |
| Maximum coverage | 131.08 | 70.46 |
| Balanced | 119.41 | 61.44 |

This is one full run per environment, not a distribution or an isolated decoder
microbenchmark. Native video-stage times were 38.50 / 41.14 / 41.12 seconds,
versus 78.66 / 77.99 / 79.88 in the simulator. The NVIDIA backend logged 19 decoder
sessions, 34,205 decoded/output frames and zero failures during the native run;
peak sampled decoder utilization was 59%. ONNX inference stayed on the CPU.
Every model produced 480 analysis rows, and rally counts remained 7 / 3 / 3.
Counts matching does not establish identical boundaries, full numerical parity,
or model accuracy. No models or thresholds were changed.

Five native correctness tests passed before timing; the optional benchmark then
passed separately with all three model assertions. The tooling suite passed
26 tests. Existing iOS build settings and source/resource membership were checked
unchanged; the iOS simulator suite was not rerun for these benchmark-only changes.
