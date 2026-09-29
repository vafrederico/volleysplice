# Android to iOS neural port: remaining gaps

Audit date: 2026-09-29. Source baseline: `cd6a185e`, including the pending iOS
project-subtitle and model-specific cleanup-settings fixes. This is an
implementation review, not a new model selection or accuracy evaluation.

The selected model bundles and main inference behavior are present. The most
important remaining differences concern long-video memory, imported-feature
reuse, interrupted work, and project-specific measurements. Source differences
below are confirmed by inspection; performance consequences are hypotheses until
measured. No inference code was changed during this audit.

## Confirmed remaining implementation gaps

### 1. Full-recording embeddings remain in memory on iOS

Android writes encoder output to a token file and maps it for chunked temporal
inference. iOS retains every token in a Swift array, then creates a `Data` copy
and an encoded property-list payload when saving. Cache reads also decode the
whole token payload into an array. Chunked TCN inference is already implemented
on both platforms, but does not bound this embedding storage.

At 2 samples/s, 3,840 float32 values/sample, tokens alone grow by 110,592,000 bytes
per hour (105.47 MiB), before temporary serialization copies, audiovisual arrays,
runtime allocations and decoded frames. This is a calculated payload size, not
a measured peak-memory result or an observed out-of-memory failure.

**Priority:** high for long-video qualification. Use an append-only binary token
store with bounded/mapped reads and an atomic completion manifest; retain the
current model, source, geometry and preprocessing identity checks.

Evidence: [iOS runtime](../../ios/App/NeuralRallyRuntime.swift), especially
`tokens`, cache loading and `finishEmbeddings`;
[Android token mapping](../../android/app/src/main/java/com/volleycut/nativeanalysis/NeuralRallyPipeline.java),
`readFloats`;
[Android encoder writer](../../android/app/src/main/java/com/volleycut/nativeanalysis/SharedEmbeddingConsumer.java).

### 2. Shared decoding is present, but Android's worker pipeline is not

Android uses a dedicated encoder worker with four reusable image buffers and a
bounded job queue. Its video decoder also has asynchronous callbacks and a
feature worker. iOS synchronously invokes the encoder and visual extractor in
the sequential frame-consumption callback. This bounds frame-buffer memory, but
the application does not overlap those operations in the same way as Android.
AVFoundation may perform internal asynchronous work; this review does not claim
that the entire Apple decoder runs serially.

**Priority:** measure after the memory fix. Prototype bounded producer/consumer
scheduling with ordered output, cancellation/draining and identical selected PTS.
Keep the existing implementation as the numerical control. No percentage speed
regression or expected speedup is established by this source comparison.

Evidence: [iOS shared callback](../../ios/App/MediaDecoder.swift), `video`;
[Android encoder queues](../../android/app/src/main/java/com/volleycut/nativeanalysis/SharedEmbeddingConsumer.java);
[Android decoder](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeVideoDecoder.java).

### 3. Imported-feature reuse is incomplete; full reanalysis is a shared limitation

Android's feedback importer validates imported features, reconstructs the native
feature matrices and hydrates its cache. iOS retains the imported features in
the project; editing/exporting does not require fresh inference, and score-only
preparation explicitly reads those retained features. However, a new full
analysis uses only its own source-hash-based cache file. Import/relink does not
seed that cache from the feedback document.

Consequently, changing model on a freshly imported iOS project can recompute
motion and audio features that are already available, unless a matching local
cache independently exists. Android is not a proven fully working reference for
this use case either: relinking preserves `featureCacheSource` for export and
auxiliary evidence, but its full `ProjectAnalysisService` passes the new source
URI into `AnalysisEngine`, without passing that old cache identity. Treat safe
full-reanalysis reuse as a shared product gap, not an established Android
capability omitted exclusively from iOS. The new variant's embeddings still need
to be generated; excluding embeddings from exports remains intentional.

**Priority:** medium. Add a validated import-to-cache adapter, checking source
identity, window, ROI, timestamp grid, feature schema and preprocessing provenance.
Do not reuse features merely because their matrix dimensions match, especially
for exports predating the audio/preprocessing fixes.

Evidence: [Android import/cache hydration](../../android/app/src/main/java/com/volleycut/nativeanalysis/ModelFeedbackImporter.kt),
`hydrateFeatureCache`;
[Android relink identity](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeProjectStore.kt),
`relink`; [Android full-analysis entry](../../android/app/src/main/java/com/volleycut/nativeanalysis/ProjectAnalysisService.kt),
`analyze`;
[iOS import](../../ios/App/VolleySpliceApp.swift), `openProject`;
[iOS full analysis](../../ios/App/AnalysisPipeline.swift);
[iOS score-only feature reuse](../../ios/App/ScoreAnalysis.swift), `prepare`.

### 4. Interrupted neural runs discard partial audiovisual progress

iOS explicitly resets a partial AV cache when the neural encoder has no complete
matching embedding cache. This prevents mixing feature histories, but means
resuming an interrupted neural run repeats already-computed visual work. Android
keeps partial visual rows, finishes that pass, and falls back to a separate
embedding pass when the shared pass cannot be resumed.

**Priority:** medium, particularly for background-expired analyses. Either keep
the compatible AV checkpoint and use the existing embeddings-only path, or add
a jointly validated AV/token checkpoint. Verify restart boundaries, temporal
features and cancellation before enabling partial token reuse.

Evidence: [iOS partial-cache reset](../../ios/App/AnalysisPipeline.swift), near
the `A partial AV checkpoint` comment;
[Android partial-cache handling](../../android/app/src/main/java/com/volleycut/nativeanalysis/AnalysisEngine.java),
`cachedVisual.rows()` and `visualWriter.checkpoint()`.

### 5. Completed measurements are not reliably tied to the selected project

Android stores analysis measurements on each project and displays them from the
editor when enabled. iOS stores step/stage measurements on processing jobs and
writes analysis-result files, but the editor has no equivalent measurements
card. Setup settings reads the transient `WorkspaceModel.result` instead.
Opening another saved project neither loads its result nor clears the previous
one. Returning to setup/settings can therefore show another project's last-run
timings; after relaunch there may be no result to display at all.

**Priority:** medium. Resolve measurements by project/job/model identity, show
them in the editor, and explicitly show unavailable for imported/older projects
without local measurements. Do not silently substitute the previous run.

Evidence: [iOS setup measurements](../../ios/App/NewProjectWorkspace.swift),
`SetupSettings`; [iOS project opening](../../ios/App/VolleySpliceApp.swift),
`openProject`; [iOS result publication](../../ios/App/WorkspaceProcessing.swift);
[Android project measurements](../../android/app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt),
`AnalysisMeasurementsCard(project.analysisMeasurements)`.

### 6. Cleanup help and attention summary still describe a nonexistent capability

The pending fix correctly hides the cleanup settings section for neural projects.
The compact iOS final-video panel still tells those users to fine-tune automatic
cleanup, and includes `0 cleanup` in an attention summary when another review
category is pending. Android uses model-specific help and omits zero-count
categories in that summary.

**Priority:** small UI follow-up. Apply the same capability condition to this
copy and summary. Both platforms still show a disabled zero-count cleanup queue
button, so that button by itself is not an iOS-only regression.

Evidence: [iOS final-video panel](../../ios/App/EditorWorkspace.swift),
`compactFinalVideo`; [Android equivalent](../../android/app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt),
`reviewAttentionBreakdown` and the `RallyModels.isNeural` help text.

### 7. Cold-run embedding progress can imply that completed work has not started

On a cold neural analysis, embeddings are generated during the shared video pass.
The video detail includes the image-feature count, but the separate
`Understanding play` row remains waiting until after audio, when it immediately
becomes ready. Its stage elapsed time therefore does not represent encoder cost.
Encoder timings do exist under the nested `neural/` measurements.

**Priority:** small clarity fix. Mark the work as part of scanning on shared
runs, or support concurrent stage presentation. Keep the separate progressing
row for an embeddings-only pass after an AV cache hit. Avoid adding nested
encoder time to the parent video time when displaying totals.

Evidence: [iOS event ordering](../../ios/App/AnalysisPipeline.swift), `video`
through `embedding` events; [step ordering](../../ios/Sources/VolleyCore/AnalysisProgress.swift);
[progress panel](../../ios/App/SetupAnalysisProgress.swift).

## Already covered and intentional differences

- Both manifest variants have the same selection ID, encoder-weight hash and
  temporal-weight hash as Android. Runtime graph files need not be byte-identical
  across native and portable graph packaging.
- Maximum coverage is the new-analysis default; saved selections and old Legacy
  jobs remain stable. Only the selected encoder/TCN runs.
- Full-frame defaults, fractional image sampling, the rolling audio-percentile
  correction, shared AV/embedding decoding and bounded temporal chunks are present.
- Every neural rally receives serve-side evaluation. Weak auxiliary serve evidence
  becomes review rather than silently removing its serve candidate. Optional
  team-switch inference uses the selected rallies.
- Model provenance, four-head neural scores and base features are retained in
  project interchange; embeddings and source media are excluded.
- iOS already reuses a complete matching local embedding cache. Android's current
  production wrapper uses temporary token workspaces. This is an iOS capability,
  not a missing Android feature.
- Both current distilled native implementations configure ONNX Runtime CPU sessions.
  iOS CPU inference is not a lost Android GPU implementation. Apple accelerator
  qualification is separate future work.
- Background execution differs by platform: the iOS continued-processing lease
  is guarded by iOS 26 availability; older supported systems rely on foreground
  operation and resumable jobs. Android uses its foreground service. No claim of
  equivalent unlimited background execution is made.
- Older shared-project preference/interchange limitations remain outside the
  neural port. The historical [interoperability audit](../../ios/PROJECT-INTEROPERABILITY.md)
  documents those separately; this review did not requalify every exchange path.

## Validation and remaining evidence

Executed in this audit:

- Full portable Swift suite: **159 tests passed**, including the pending
  project-model metadata test.
- Python packaging/release-script suite: **23 tests passed**. These include
  synthetic packaging cases; this is not a new signed archive or App Store upload.
- Both variant IDs and training-weight hashes match the Android manifest.
- Branch publication privacy audit: **76 files inspected, zero working-tree
  findings and zero branch historical objects with findings**. This does not
  claim a rewrite or fresh audit of inherited main-branch history.

No physical device was available. The previous numerical/runtime and short-video
checks remain documented in [the port validation report](ios-neural-port-validation.md).
They do not establish full-video iOS recall, wholly missed rallies, peak memory,
thermal behavior, or iPhone/iPad speed. Those remain validation gaps, not observed
model regressions.

Suggested order: correct measurements and cleanup/progress copy; add safe imported
feature reuse and resume; bound token memory; then compare the serial and worker
pipelines on the same short and full recordings with both variants. Verify all
scores/boundaries before interpreting timings. A same-source Android/iOS
full-video comparison against saved human rallies is still required, using the
existing padding, ignored-time and ranking contract without reselection.
