# iOS selected neural models: implementation and validation

Date: 2026-09-27. This ports the existing frozen selections; no training,
calibration, seed selection or human-label accuracy evaluation was performed.

## Implemented behavior

The setup selector offers **Maximum coverage · BETA** (default), **Balanced ·
BETA**, and **Legacy model**. Only the selected encoder/TCN bundle runs. Existing
projects and interrupted jobs preserve their selection; older jobs retain Legacy.
Reanalysis creates a separate project and preserves existing edits.

The neural path uses ONNX Runtime 1.24.2, FP32 CPU execution and native FP32
embeddings. Both matched bundles are hash/size pinned by the
[iOS manifest](../../models/distilled-large/ios-manifest.json). The portable
regional-pooling graphs are the same public graph bytes as web; browser FP16 token
rounding is not enabled on this native path. GPU/Core ML/Neural Engine execution
has not been qualified and is not silently selected.

The actual application pipeline now includes:

- Full-frame defaults, source-aligned nearest-frame sampling, correct rotation,
  clean aperture and YUV matrix/range handling.
- A shared sequential decoder for 4 Hz audiovisual features and 2 Hz image
  embeddings. Each encoder has its own cache; compatible audiovisual features
  survive a model change. Temporal inference uses bounded overlapping chunks.
- The corrected rolling audio-percentile behavior and cache-version changes
  for modified audiovisual and specialist preprocessing.
- Serve-side evaluation for every neural rally, including visible review
  recovery when the auxiliary serve heads do not cross their threshold. Optional
  side-switch features retain their existing legacy evidence contract. These
  auxiliary heads do not suppress or replace neural rally boundaries.
- Live preparation, scanning/image-feature, audio, temporal-inference and score
  progress. Model names remain in the selector/legend instead of selected-rally
  bars.
- Shared project exports with selected-model provenance, audiovisual features
  and all four neural probability traces. Embeddings and source media remain
  outside exports. Both model bundles and their required notices are packaged
  offline.

## Executed checks

Validation uses the Intel CPU-only Mac simulator and independent numerical
references. It does not estimate iPad throughput or GPU behavior.

- Portable Swift suite: **158 tests passed**.
- Focused simulator suite: **16 tests passed**, with the optional external-video
  test skipped until supplied its private input by the dedicated runner.
- Python release/CI checks: **14 tests passed**.
- All three Swift feedback variants passed the unchanged production web parser.
- The optional real-video integration test passed separately with its configured
  input, completing all three model runs below.
- Simulator Debug and unsigned ARM64 Release builds passed. The release bundle
  audit verified all six neural graph/config hashes and sizes, the pinned
  manifest, five notice payloads and the app privacy declaration. No test
  resources were present. This was an app-bundle audit, not a signed IPA or an
  Xcode-generated privacy/API report.

The native runtime tests compare every value against independently generated
ONNX Runtime Web 1.22 WASM outputs on public synthetic inputs. The encoders emit
3,840 values each; the temporal cases contain 301 ticks, crossing chunk seams.

| Maximum absolute difference | Maximum coverage | Balanced |
| --- | ---: | ---: |
| Encoder tokens, native versus WASM | 0.0000066162 | 0.0000040532 |
| Temporal logits, native versus WASM | 0.0000007153 | 0.0000022650 |
| Native full-sequence versus chunked temporal output | 0 | 0 |

All comparisons pass the declared `1e-4 absolute + 1e-4 relative` tolerance;
decoded boundaries also match. These tests isolate model execution, rather than
asserting that Apple and Android video/audio decoders are bit-identical.

Media checks cover Android-derived area-resize goldens for all right-angle
rotations, YUV conversion, fractional bilinear pixels, black padding, clean
aperture and saturated full-HD inputs. The real-video check exposed a bilinear
rounding overflow of `255.000015`; the sampler now clamps to the mathematical
pixel range while retaining fractional values. The strict model-input validator
remains in place.

Integration tests exercise cold shared decoding, cached audiovisual features,
reuse of matching embeddings, cancellation, fractional source windows, old
rounded-up queued endpoints and stale specialist caches. Fractional video ends
are normalized to representable milliseconds within the source duration before
analysis and project creation. Analysis-result files use each job's identity,
so switching models cannot overwrite an earlier job's result.

The unchanged production web importer accepts Swift-generated Legacy, Balanced
and Maximum coverage feedback. Each fixture retains its two rows of 104 base
features, initial/corrected rally and final export. Both neural fixtures retain
their model ID and two rows of `live/serve/end/keep` probabilities. This is a
focused contract check, not a claim that unrelated project preferences have
become losslessly interchangeable.

## Short real-video smoke benchmark

The optional external-video test passed using a 30-second 1920×1080 H.264/AAC
clip from ledger `recording-044`. Each model started with an empty application
feature cache, used the full frame, and enabled both score specialists. There
were 120 audiovisual ticks and 60 image samples for each neural variant.

| Model | All results ready | Shared video/features | Audio | Rally inference | Score specialists | Rallies / serve candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 53.97 s | 35.71 s | 0.304 s | 0.041 s | 16.53 s | 3 / 3 |
| Maximum coverage · BETA | 28.33 s | 18.55 s | 0.173 s | 0.023 s | 8.73 s | 3 / 3 |
| Balanced · BETA | 24.32 s | 16.63 s | 0.158 s | 0.018 s | 7.19 s | 3 / 3 |

These are **single sequential Debug simulator observations, not a comparative
speed ranking**. Legacy ran first; OS/codec warmup and VM scheduling were not
controlled. Its preceding diagnostic run took 31.18 seconds, demonstrating the
variability. Do not infer that a neural model is faster than Legacy on an iPad.
All-ready time includes source hashing, feature generation, model inference,
serve-side and side-switch processing. It excludes final project serialization.
This clip has not been scored against human labels; three returned rallies do
not establish recall, precision, boundary accuracy or equal rally boundaries.

Within the shared video stage, Maximum coverage used 0.276 seconds to load its
encoder, 0.088 seconds for normalization/quality/pooling/tensor preparation and
0.263 seconds for encoder execution. Balanced used 0.138, 0.084 and 0.217 seconds
respectively. Bilinear pixel sampling belongs to the outer video stage. TCN
execution, including loading/fusion, took 0.013 and 0.011 seconds respectively.
These nested measurements must not be added again to all-ready time.

| Stored or numeric payload | Legacy | Maximum coverage | Balanced |
| --- | ---: | ---: | ---: |
| Base features, raw bytes | 49,920 | 49,920 | 49,920 |
| Contextual features, raw bytes | 249,600 | 249,600 | 249,600 |
| Source and decoded timestamps, raw bytes | 1,920 | 1,920 | 1,920 |
| Serving-side features, raw bytes | 5,688 | 5,688 | 5,688 |
| Side-switch features, raw bytes | 544 | 544 | 544 |
| Four neural probability heads, raw bytes | 0 | 1,920 | 1,920 |
| Local embedding cache, actual file bytes | 0 | 925,514 | 925,514 |
| Serialized local project, actual bytes | 98,558 | 98,677 | 98,623 |

Raw array sizes exclude container metadata; neural score timestamps are retained
separately in exports. Local project size is the encoded `ProjectDocument`, not
a media-containing package. Embeddings remain local and are excluded from both
project formats. Side-switch output contained one candidate for Legacy and zero
for each neural selection, with the specialist feature/inference pass executed
in all three cases.

## Limits and follow-up

The unsigned ARM64 app bundle occupies 70,628,235 bytes (67.36 MiB); the two
neural bundles account for 24,203,053 bytes (23.08 MiB). These are unpacked build
sizes, not App Store download sizes. No app signing, installation or publishing
was performed. Source publication auditing inspected all 62 changed/new files
and the branch history since the shared base, with no private-reference findings.

Physical-device work remains: Apple hardware decoder parity, sustained full-match
throughput, peak memory, thermal behavior and any Core ML/GPU/Neural Engine
provider qualification. Embedding caches currently load a variant's FP32 tokens
into memory; long-recording memory behavior needs device measurement. Seeking a
late partial window currently decodes its preceding frames to preserve nearest
sampling, which also needs performance measurement.

The release workflow stages the new public assets and exact runtime dependency,
runs portable Swift tests and audits the release bundle. It does not automatically
run these simulator runtime/video integration tests. The reproducible entry points
are [the kernel check](../../ios/scripts/check-video-preprocessing.py),
[the independent runtime fixture generator](../../ios/scripts/generate-neural-runtime-golden.mjs),
[the bounded video benchmark](../../ios/scripts/run-neural-benchmark.py), and the
test targets in [the iOS project](../../ios/VolleySplice.xcodeproj/project.pbxproj).
Private source mappings and raw Xcode logs stay outside Git.

## Publication follow-up — 2026-09-28

The full iOS subtree was reviewed in addition to the changed-file audit. Legacy
reports and optional reference tests still contained source names, project/cache
identities and a personal contact address. The contact was removed; historical
provenance was preserved outside Git under ledger `private-reference-0224`, and
published reports now use indexes. Optional reference tests require
`VOLLEYCUT_IOS_REFERENCE_VIDEO_NAME` instead of embedding a recording name.
The extended ledger-based scan passed across all 167 iOS files. This cleanup
changes current files; it does not rewrite the older commits inherited from main.
