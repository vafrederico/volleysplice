# Distilled Large production integration

Both production apps now default new analyses to **Distilled MobileNetV3-Large +
TCN, highest recall**. The model selector also offers **highest F1** and the
existing **production ensemble**. Existing saved projects retain their model
identity. This change integrates the frozen selections from the
[completed study](distilled-mobile-large-results.md); it does not retrain,
recalibrate, or select models again.

## Switching models

Choosing highest F1 before analysis switches the complete matched bundle:
distilled image encoder, temporal head, normalization, and interval decoder.
The selected model is recorded with the project and feedback. Each selection
has its own project/cache identity, so choosing another variant does not
silently replace an existing edit.
Saved-project pickers display the variant alongside the source name. Stale
neural provenance requeues the selected bundle; unavailable model versions
require an explicit new selection instead of switching to the ensemble.

Recall and F1 use different distilled encoder weights. Their image embeddings
must be recomputed when switching variants. Compatible audiovisual features
can be reused when source, crop, game window, and preprocessing contracts
match. Neither DINO nor its training projection runs in the shipped pipeline.

Serving-side and optional side-switch inference use each variant's resulting
rally boundaries. Neural rallies bypass ensemble suppression and its cleanup
controls. The ensemble remains an explicit alternative and retains its own
preprocessing and cleanup behavior. Neural download or integrity errors are
reported rather than silently returning ensemble predictions.

## Runtime and packaging

- Android uses the shared video decoding implementation in the normal app,
  including the corrected audio and display-geometry handling. Independent
  encoding remains available when audiovisual features are cached.
- Web runs encoding and temporal inference in a dedicated worker, with WebGPU
  where a usable adapter exists and WASM CPU fallback otherwise. Cancellation
  releases the worker and its inference resources.
- Both preserve the qualified 2 Hz image / 4 Hz audiovisual alignment. A
  fractional game-window start may require one preceding image context sample;
  exported predictions stay inside the requested window.
- Temporal inference is chunked with the required context halo. Web retains
  the qualified FP16 embedding-cache rounding; Android retains FP32 embeddings.
  These caches are not interchangeable across platforms.
- Each model bundle is approximately **12.1 MB** uncompressed. Android bundles
  both offline; web fetches the selected bundle. Both bundles total approximately
  **24.2 MB**, excluding ONNX Runtime. Android may retain another approximately
  24.2 MB of verified, extracted model files after using both variants.
- Public manifests pin hashes and sizes. Generated model assets remain outside
  Git and are required explicitly for builds. Build preparation rejects missing
  or changed assets. Source licensing and notices accompany the assets.

The [bundle documentation](../../models/distilled-large/README.md),
[feature contract](../../FEATURE_PIPELINE.md), and
[model inventory](../../MODELS.md) describe the selection and build inputs.

## Integration validation

The real-video checks use a 120-second segment of **recording-044**, resolved
through the private ledger. Detailed receipts remain under artifact index
**private-reference-0223**. Browser and emulator workloads ran sequentially.

| Runtime and selection | Rallies | Reference comparison | Score specialists |
| --- | ---: | --- | --- |
| Android normal app, recall | 6 | Exact boundaries and confidence against the qualified emulator reference; quality rows and selected frame timestamps also exact | Both ready |
| Android normal app, F1 | 7 | Exact boundaries against the corrected physical-device reference; maximum confidence difference approximately 0.000020; quality rows and selected frame timestamps exact | Both ready |
| Desktop Chrome, recall | 6 | Exact boundaries, audiovisual features, timestamps, and rally probabilities against the qualified browser reference | Both ready |
| Desktop Chrome, F1 | 7 | Complete production-module run; correct model identity and separate embeddings verified | Both ready |

These counts test pipeline integration, not a new accuracy ranking. The F1
selection's additional interval in this segment does not establish better
dataset recall. No new human-label precision, recall, or F1 estimate is claimed
by this integration check.

The browser integration harness imports the actual production modules and
workers in an isolated test page; it does not render the production React app.
Fresh recall, cached recall, F1, a fractional game window, and cancellation
passed. Cached recall reproduced the same features, probabilities, and rallies;
F1 did not reuse recall embeddings. A separate 12-second WASM run passed the
same five cases with the production content-security policy, a verified absent
WebGPU adapter, and observed CPU execution. Score specialists were disabled
for that short CPU smoke test.

The normal Android APK passed synthetic checks for shared/independent encoder
agreement, cached-feature processing, fractional windows, cancellation,
temporary-file cleanup, and saved model identity. An independent asymmetric
pixel fixture verified rotation mapping at 90, 180, and 270 degrees. This
caught and repaired an independent-path quality-crop coordinate error before
the final real-video checks. Real-video receipts compare quality and frame
timestamps; exact embedding hashes were checked in synthetic tests, not
captured in the real-video integration receipts.

Observed Android emulator stage totals were **189.4 seconds for recall** and
**171.7 seconds for F1**, including both score specialists. The desktop browser
observations were **49.9 seconds** and **30.2 seconds**, respectively. These
are functional integration runs with different cache and initialization
conditions, not controlled performance comparisons. They do not replace the
earlier full-video or physical-device benchmarks.

Automated checks passed: **341 web tests** with two skips, **four asset
preparation tests**, and **206 Android JVM tests**. Five synthetic/regression
instrumentation executions and two real-video instrumentation executions
passed; one optional external-fixture case was skipped in the synthetic suite.
The production static build verified asset hashes, portable paths, and the
hosting per-file size limit.
The normal ARM64 debug and unsigned release APKs also built successfully,
including release lint and shrinking.
Their complete sizes are 121,513,475 and 86,400,199 bytes respectively; these
are total APK sizes, not the increase attributable to the neural models.
Both contain the six pinned model/configuration assets with matching hashes,
ARM64-only native libraries, and seven license files. The release mapping
preserves ONNX Runtime and native preprocessing JNI names. The release APK
remains unsigned.

Publication checks also found and removed an older suppression manifest's
private training location. Public-field allowlisting and asset-hash validation
now prevent that metadata from entering future builds. The rebuilt website
scan covered 65 filenames and 54 text assets with no private findings after
narrow classification of existing public contact details and unchanged
upstream virtual-filesystem code. Separately, the eight generated ONNX graphs
had no private findings in 6,476 protobuf string fields and no external tensor
references. These checks complement the branch working-file/history audit;
they are not a universal credential-detection claim.

## Release boundary

The real Pixel was unavailable for this integration. The normal-app physical
device run remains pending; emulator or desktop-browser results are not phone
speed measurements. This change does not establish beach-domain accuracy.
No application was signed, published, or deployed, and no production app
server was started.
