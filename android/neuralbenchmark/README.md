# Native neural benchmark

Research-only app (`com.volleycut.neuralbenchmark`), separate from the installed
VolleySplice app. No network permission, embedded browser, or training.

Build with the repository Gradle wrapper, the documented Android SDK/JDK, and
`:neuralbenchmark:assembleDebug`. Set `GRADLE_USER_HOME`, `TEMP`, `TMP`,
`-Djava.io.tmpdir`, `--project-cache-dir`, and `VOLLEYCUT_BENCH_BUILD_DIR` to NAS
locations. The last variable redirects this module and root build reports.
If Windows refuses to execute the downloaded AAPT2 from a network share, use
`-Pandroid.aapt2FromMavenOverride=<installed SDK>/build-tools/37.0.0/aapt2.exe`.
This reads the existing local executable while keeping generated output on NAS.

`scripts/prepare-pixel-neural-benchmark.py` prepares frozen graph conversions and
reference tensors. It binds the existing highest-recall DINO-TCN (seed 3407,
epoch 30) and Mobile-TCN (seed 3407, epoch 15). No fitting or model selection is
performed. FP32 graph outputs are checked against the original PyTorch head.
The initial temporal probes use synthetic standardized inputs, **not** video
predictions. Encoder probes use a real video frame. Mobile INT8 uses static QDQ
calibration on 32 unlabelled pilot frames; DINO uses the existing mixed dynamic
INT8 encoder. TCN INT8 conversions quantize MatMul/Gemm, leaving convolutions
floating point. These are not interchangeable definitions of “all INT8.”

Transfer model/input files into `files/benchmark` with ADB `run-as`; verify their
hashes. `scripts/benchmark-pixel-neural.py` transfers a plan, launches the app,
waits for its matching run ID, and collects outputs and execution profiles.
The app records failures per case, so unsupported backends cannot silently
become successful timing rows. NNAPI can still leave ONNX nodes on CPU; inspect
the separate profiles. `webgpu` selects ONNX Runtime's **native** graphics
provider. It does not start Chrome or a WebView.

Tensor cases specify `id`, `model`, `provider`, `inputs` (name/file/shape/dtype),
`warmup`, and `runs`. Initialization and input loading are timed separately.
Outputs are materialized before stopping the clock. `profile: true` is for a
separate placement audit, never for performance ranking.

Video cases instead specify `video`, `imageSize`, `sampleFps`, `seconds`,
`roi: [x,y,width,height]`, and optional `poolWeights`. They decode sequentially
with MediaCodec, prepare normalized ROI letterboxes, generate embeddings and
save all tokens/timestamps. DINO uses 336px at 4 Hz; Mobile uses 224px at 2 Hz.
This prototype does not share the production decoder or AV generation pass,
does not compute Mobile's quality scalars, and does not yet feed the live tokens
through the temporal head. Its total is **encoder-stage video processing**,
not the complete rally detector. Pixel conversion/frame-selection parity still
requires validation. Report its extra decode cost explicitly.

`scripts/benchmark-pixel-production.py` runs the installed production debug app
on the same MediaStore clip with fresh feature generation. Retain its installed
version/model hashes, raw run results, battery and thermal state. Use the first
run as warm-up and at least three measured runs. Separate rally processing from
score/serving-side specialists when comparing rally-only research models.

Report graph precision, actual provider placement, numerical drift, cold and
warm speed, video-stage costs and unsupported combinations independently.
Neither a successful graph conversion nor a tensor benchmark establishes
retained-play recall, browser compatibility, or complete product readiness.
