# Complete native FP32 pipeline benchmark

Build with `-PpipelineBenchmark :app:assembleDebug`. This installs as
`com.volleycut.nativeanalysis.pipelinebenchmark`, independently of the regular
debug and production apps. Without that property, the benchmark sources and
neural runtime dependencies are excluded. Set `VOLLEYCUT_BENCH_BUILD_DIR`,
`GRADLE_USER_HOME`, the Gradle project cache and temporary directories to NAS.

`scripts/prepare-pixel-complete-pipeline.py` exports the frozen high-recall FP32
DINO-TCN and Mobile-TCN checkpoints, fold scalers, and selected 99% target
decoders. This does not train or choose a new operating point. Dynamic temporal
exports are checked against PyTorch at several sequence lengths.

`scripts/benchmark-pixel-complete-pipeline.py` transfers those artifacts and
runs fresh video-to-results measurements. The app generates production AV104
features, image embeddings (neural cases), Mobile quality/age/availability
features, normalized fused inputs, real-context temporal predictions and rally
boundaries. Each case then runs the existing serving-side and side-switch
pipeline on its own selected boundaries, including specialist frame decoding
and feature extraction. It saves actual specialist outputs, not timing proxies.

Neural rally decisions do not consume production rally predictions. The frozen
score specialists nevertheless require both production serve heads and
production rally/dead-state evidence. Those signals are computed and charged
to every complete neural run; the production suppression/merge work is also
currently retained as small conservative overhead. Shared AV features are
generated once per case. No warm feature cache is used.

The initial implementation performs an additional sequential image-embedding
decode pass. Its real cost is included, not subtracted as a hypothetical
optimization. The pipeline timer includes model loading and diagnostics written
during neural inference. Report serialization/ADB collection after results are
ready is outside that timer. Export rendering is outside this benchmark.

`scripts/validate-pixel-complete-pipeline.py` checks saved real-device fused
inputs, temporal probabilities and decoded boundaries against desktop ONNX and
the canonical Python decoder. That does not establish pixel/PTS extraction
parity or rally accuracy. Native RGB conversion and forward-frame selection
still require separate qualification against training-time OpenCV/nearest-PTS
extraction before this prototype can be deployed.
