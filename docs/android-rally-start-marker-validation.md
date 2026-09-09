# Android rally-start / serve-marker validation

Validated on 2026-09-08 against baseline `ee7a339` (0.10.13), using the
`Pixel_10_Pro_API_37_24G` Android 17 emulator and the debug application.

## Reproduction

`RallyStartMarkerInstrumentedTest` opens the real `EditorActivity` with a
one-second video fixture, one inferred rally (0.500–0.900 seconds), and its
linked model serve at 0.500 seconds. It seeks through the game timeline and
presses **Set rally start here**. Each direction starts from a fresh draft.

| Action | Saved rally start | Baseline saved serve | Fixed saved serve |
| --- | ---: | ---: | ---: |
| Move start forward to playhead | 0.650 s | 0.500 s | 0.650 s |
| Move start backward to playhead | 0.125 s | 0.500 s | 0.125 s |

The baseline failed both timestamp assertions. Screenshots also showed the
volleyball icon at the old timestamp while the green rally core moved.
This was a persisted annotation error, not just a rendering problem.
Seeking without pressing the start-edit button leaves annotations unchanged.

## Cause and history

- `c4d8092` (2026-08-18, Android 0.10.2) introduced core trimming. Its start
  and range setters updated cut boundaries only.
- `dd90646` (2026-08-22) added Android serving-side score tracking, with
  separate timestamps and a `rallyId` link. It did not connect boundary edits
  to those markers.
- `64a5356` (2026-08-23), **Fix serving indicators across merged rallies**,
  fixed which score/serving state appears through padding and joined gaps.
  It did not synchronize persisted marker timestamps after a start edit.
- Draft loading and feedback import reseeded model markers from the original
  serving-side candidate anchor, providing another way to restore stale times.

The available Android history indicates missing edit synchronization, rather
than a failed earlier implementation of that synchronization.

## Fix

`EditorMath.alignRallyServeMarkers` resolves each existing marker's explicit
`rallyId` to the cut's corrected `coreStartMs`. Start/range setters apply it,
and draft reconciliation applies it to all editor mutations and restored drafts.
That also covers manual-cut handle edits and asynchronously reseeded markers.
Feedback import aligns markers after reconstructing corrected cuts.

Only the timestamp changes. Marker identity, side corrections, replay flags,
tombstones, and model inference evidence are preserved. Unlinked manual serves
and team-switch markers keep independent timestamps. Padding and end-only edits
do not move an already aligned serve, and splitting does not invent another serve.
Older drafts with linked stale markers are repaired when loaded.

## Verification and artifacts

- `testDebugUnitTest`: 144 tests passed, including five new regression tests
  covering bidirectional edits, clamping, range edits, padding, splits, disabled
  scoring, independent markers, manual linked rallies, tombstones, and repair.
- `assembleDebug` and `assembleDebugAndroidTest` succeeded.
- Emulator: both real-editor start-edit tests and all three existing
  `ScoreTrackingUiInstrumentedTest` tests passed.
- The start-edit tests also check activity recreation, feedback export/import,
  the serialized MP4-export score snapshot, and loading a deliberately stale
  legacy draft. Corrected range and serve times agree; the original model anchor
  remains unchanged as inference evidence.

Local evidence is retained in ignored `android/app/build/` output:

- `rally-marker-before.log` and `rally-marker-before-evidence/`
- `rally-marker-after.log` and `rally-marker-after-evidence/`
- `rally-marker-roundtrip.log`
- `reports/tests/testDebugUnitTest/index.html`

Feedback labels remain on the original source timeline, as required by the
feedback schema. This validation covers their consistency and the score state
passed to MP4 export; it does not test a new MP4 encode or conversion of labels
to the concatenated exported-video timeline. No Play release was published.

## Production web and training importer follow-up

The actual Android feedback files were also passed through the production web
`importModelFeedbackProject` function and the model refresh used when its editor
opens. Parsing preserved 0.650 / 0.125 seconds, but that refresh originally
replaced both with the 0.500-second model anchor. `scoreTrackingWithServingSideOutput`
now preserves existing linked marker timestamps. Regression tests also verify
that the score reducer activates the serve at the corrected time.

A separate compatibility issue rejected exports made without a retained feature
cache: zero-length numeric arrays correctly encode as empty base64 strings. The
web validator now accepts that encoding only when the declared array is empty;
nonempty arrays still require matching data. Both complete and cache-missing
Android feedback fixtures import successfully after these fixes.

The 23 focused web tests and production TypeScript/static build checks pass.
The Python training normalizer preserves the corrected timestamp as both
`rawTime` and `time`, retains the rally ID, and reports no timing adjustment.
Its five-test suite passes with default settings and realistic rally durations.
The one-second emulator fixtures were also checked with only the micro-range
filter disabled, since the forward edit produces a 0.250-second test rally
that the normal 0.500-second filter deliberately excludes.

Evidence: `android/app/build/moved-marker-web-before.json`,
`moved-marker-web-after.json`, `moved-marker-training-parse.json`,
`moved-marker-web-tests.log`, and `moved-marker-prod-build.log`.

The release AAB was built as version **0.10.14**, version code **28**, with
`testDebugUnitTest assembleDebug bundleRelease`. It remains unsigned in
`android/app/build/outputs/bundle/release/app-release.aab`; signing is performed
by the key owner. The web fixes are local and have not been deployed.
