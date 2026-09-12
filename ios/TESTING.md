# iOS testing

Use portable tests for analysis and project contracts, hosted tests for Apple
framework behavior, and UI/device tests for the rendered workflow. Matching
interval counts or passing a self-roundtrip does not prove cross-platform parity.

## Automated suites

| Area | Tests |
| --- | --- |
| DSP and frozen inference | `AudioFeatureTests`, `FeatureMathTests`, `ModelRunnerTests`, `ServingSideInferenceTests`, `SideSwitchInferenceTests`, `SuppressionTests` |
| Editing, scoring and export timing | `ProjectContractTests`, `RallyStartMarkerTests`, `EditorReviewTests`, `ScoreTrackingTests`, `ScoreOverlayTests`, `ChaptersTests`, `ExportTimelineTests` |
| Storage, queue and progress | `ProjectPersistenceTests`, `AppStorageTests`, `ProcessingJobTests`, `RecordingImportTests`, `AnalysisProgressTests` |
| Layout and tutorial state | `EditorLayoutTests`, `InterfaceScaleTests`, `GuidedTourTests` |
| Hosted media and persistence | `FeatureExtractionIntegrationTests`, `RenderingIntegrationTests`, `SourceVideoAccessTests`, `PhotoKitVideoSourceTests`, `QueueSnapshotTests`, `EditorProjectBindingTests` |
| Orientation declaration | `AppOrientationDeclarationTests` reads the processed app plist, including device-qualified orientation keys |
| SwiftUI behavior | `WorkspaceUITests`, `EditorParityUITests`, `EditorTutorialUITests`, `TimelineScoreVisibilityUITests`, `ReferenceProjectUITests` |

Run portable tests with `scripts/test-core.ps1` on the configured lab, or
`swift test` from this directory after staging the required canonical fixtures.
Missing fixture skips are not a successful inference-parity result.

After normal source upload, run hosted and UI tests on the Mac:

```sh
xcrun simctl list devices available
bash scripts/build-simulator.sh <SIMULATOR-UDID> tests
xcodebuild -project VolleySplice.xcodeproj -scheme VolleySplice \
  -configuration Debug -destination id=<SIMULATOR-UDID> \
  -derivedDataPath build/simulator -parallel-testing-enabled NO \
  -resultBundlePath artifacts/<UNIQUE-RUN>.xcresult test-without-building
```

The lab framework uses x86_64 for Simulator; its arm64 slice targets physical
devices. Grant simulator Photos permissions before Photos integration checks;
those tests explicitly skip when permission is absent. Synthetic UI tests use
`--parity-editor` only in Debug Simulator builds. See
[fixture provenance](Tests/Fixtures/README.md).

`ReferenceProjectImportTests` is opt-in: stage the real full iPad feedback export
as `Reference-iPad-full.model-feedback.json` and its matching original recording
as `tds6-reference.mp4` in simulator Documents. It imports without analysis.
Reference UI tests require that imported project. Do not commit personal videos,
project snapshots or generated receipts as test fixtures.

## Physical device checks

The reusable scripts use the existing lab WDA connection and fresh UI state.
Keep build/install/test operations serial. They return to iOS Lab Parking and
verify its disabled idle timer in teardown, including after a failure. Use the
lab lifecycle procedure for an explicit end-of-session shutdown.

```powershell
python ios/scripts/device-smoke.py --media-check
python ios/scripts/device-smoke.py --recording <RECORDING-NAME> --export-check
python ios/scripts/physical-parity-smoke.py --project <OBSERVED-PROJECT-BUTTON-ID>
python ios/scripts/physical-settings-smoke.py --project <OBSERVED-PROJECT-BUTTON-ID> --orientation-only
```

The settings harness also tests 60/125%, shared setup/editor scaling, relaunch
persistence and unchanged project settings. Add `--layout --udid <PAIRED-UDID>`
with pymobiledevice3 Python to verify saved player/sidebar sizes via read-only
AFC. It restores original display preferences; use its `--recover` journal if
restoration was interrupted. Persistent cleanup tests require a disposable
project and explicit expected results; inspect each script's `--help`.

`queue-smoke.py` covers analysis lifecycle; `export-queue-smoke.py` covers video
interruption/resume. `physical-icon-smoke.py` captures the launcher with bounded
MJPEG reads. `editor-smoke.py` supplies shared harness utilities; its standalone
Undo-based scenario is historical and is not current editor acceptance.

Debug Developer checks expose frozen inference, the four canonical Android
rotation clips and paired overlay-on/off exports. The export fixture creates
13.5 seconds from a recording at least 20 seconds long, covering padding,
retained gaps, ignored spans, score timing and chapters. For independent checks:

- `scripts/verify-chapter-container.py` and `Tests/Container/MP4ChapterWriterProbe.swift`
  verify chapter metadata, unchanged packets and large-file offsets.
- `scripts/validate-feedback.mts` invokes the production browser validator.
- `swift run VolleyProjectCheck <PROJECT-OR-FEEDBACK>` verifies Swift imports and
  immutable-payload roundtrips.
- `scripts/compare-device.py` and [Android cache replay](scripts/android-reference/README.md)
  support source-bound numerical comparisons.

## Validation limits

The Mac VM has no GPU acceleration. Its iPhone landscape framebuffer can be
clipped or black despite valid accessibility geometry. Simulator checks cover
both rotations, four marker rows, visible timeline/legend and scale persistence;
they do not certify hardware pixels, emoji, video playback or touch latency.

Physical iPad checks exercised landscape retention (including upright launch),
60/125% scaling and restored display preferences. Remaining hardware coverage
includes smaller iPhones, maximum player resizing and persistence, notch-side
sidebar dragging, Dynamic Type and preview sharpness. Compact cleanup callout
organization and selected-serve inline side/ignore controls still need comparison
against Android. Measure responsiveness during real saves and queue staging;
WDA request duration is not an app latency measurement.

Use full-match hardware decode/export to compare rallies, padding, ignored spans,
cleanup fragments, retained gaps, score fades and chapters. Excerpt exports and
frozen-feature tests do not establish this full result. Retain coverage for all
encoded orientations, fade phases, long names and multiple output resolutions.
Compressed-audio decoder PTS/priming/window behavior needs independent PCM fixtures;
existing DSP goldens start after decoding.

Exercise real cloud Files providers, expired access, insufficient space, iCloud
eviction/download cancellation, Photos denial/recovery and interrupted delivery.
Local-provider and simulator Photos checks do not cover those conditions.
For cross-platform import/edit/reexport gaps, see
[Project interoperability](PROJECT-INTEROPERABILITY.md).
