# VolleyCut Android Play Store release audit

Audit date: 2026-08-19  
Package: `com.volleycut.nativeanalysis`  
Audited version: `0.10.3` (`versionCode` 17)

## Verdict

**Not ready for Play submission yet.** The privacy architecture is unusually strong, the unit tests pass, and a release app bundle can be produced. Submission should wait until the Android compatibility errors, foreground-service timeout handling, legal link, release signing, and final release validation are resolved.

## Release blockers

### 1. Lint fails with 10 errors across the supported Android range

`minSdk` is 29, but production code calls APIs introduced in Android 13–15. Android lint fails both debug and release checks.

Evidence:

- `AnalysisEngine.java:363` and seven `SuppressionPolicyEngine.java` sites call `Stream.toList()`, available from API 34 without core-library desugaring.
- `SuppressionModelRunner.java:70` calls `ByteArrayOutputStream.toString(Charset)`, available from API 33.
- `NativeAudioDecoder.java:584` calls API-35 `MediaCodec.queueInputBuffers()`. Runtime routing appears to guard this batched path to API 35+, but lint cannot prove it; isolate or annotate the API-specific implementation.
- `NativeAudioDecoder.java:559` forwards `MediaExtractor` sample flags where `MediaCodec` buffer flags are required.

Impact: analysis or suppression preparation can fail on Android versions the Play listing would mark compatible, and the required lint gate is red.

Recommendation: replace Java stream terminal calls with `Collectors.toList()` or enable verified core-library desugaring; use `new String(output.toByteArray(), UTF_8)`; isolate API-35 batching behind a versioned implementation; and explicitly map extractor flags to codec flags. Run lint and device tests on API 29, 33, 34, 35, 36, and 37.

### 2. Media-processing foreground services do not implement timeout cleanup

Both `ProjectAnalysisService` and `ExportService` use `mediaProcessing`. Android 15+ limits this foreground-service type to six hours in a rolling 24-hour window and calls `Service.onTimeout(...)` when the budget is exhausted. Neither service overrides the callback. Both wake locks allow up to 12 hours.

Impact: a long analysis/export queue can be terminated with a foreground-service timeout exception instead of cancelling cleanly. The two services also share the same six-hour app-wide media-processing budget.

Recommendation: implement `onTimeout(int, int)` in both services, atomically cancel active work, remove incomplete exports, release wake locks, update project state, and call `stopSelf()` within the callback window. Cap or explain long queues and test with Android's shortened timeout configuration.

### 3. The release AAB is unsigned

`bundleRelease` produced `android/app/build/outputs/bundle/release/app-release.aab`, but `jarsigner -verify` reports `jar is unsigned`.

Impact: Play Console will not accept it as an upload artifact.

Recommendation: use Play App Signing and sign the AAB with the private upload key. Do not commit the private key or passwords. Rebuild after all fixes, then have the key owner run:

```powershell
& "$env:JAVA_HOME\bin\jarsigner.exe" -verbose -sigalg SHA256withRSA -digestalg SHA-256 `
  -keystore "C:\path\to\upload-key.jks" `
  "app\build\outputs\bundle\release\app-release.aab" "UPLOAD_KEY_ALIAS"
```

Verify without exposing a password:

```powershell
& "$env:JAVA_HOME\bin\jarsigner.exe" -verify -verbose -certs `
  "app\build\outputs\bundle\release\app-release.aab"
```

### 4. A privacy policy is not linked or shown inside the Android app

Current Play policy requires a privacy-policy URL in Play Console and a privacy-policy link or text in the app. The Android UI has neither.

Impact: policy rejection even though no data is collected.

Recommendation: deploy `prod/public/privacy.html`, add an in-app **Privacy** entry that opens the HTTPS policy or displays the same text locally, and keep the Play Console Data safety answers consistent with it.

### 5. Foreground-service Play declaration and demonstration video are still required

The manifest correctly declares the `mediaProcessing` type and permission, but Play Console requires the use case, deferral/interruption impact, and a demonstration-video link.

Impact: the App content section cannot be completed without it.

Recommendation: use the prepared text in `play-console-answers.md` and record the rights-cleared demonstration described there.

## High-priority hardening

### Unnecessary exported benchmark activity

`MainActivity` is exported even though it has no public intent filter and exists for benchmark/automation tools. It accepts a content URI and automation extras. Android security guidance is to set internal components `exported="false"`.

Recommendation: make `MainActivity` non-exported. Keep only launcher `EditorActivity` exported, and continue validating all launcher inputs.

### Backup behavior conflicts with the simplest privacy promise

The manifest sets `allowBackup="true"`. Project JSON, selected-document URIs, filenames, edit decisions, and preferences are stored in app-private files. Depending on OS/device settings, some may be included in cloud backup or device-to-device transfer.

Recommendation: either intentionally support backup with explicit `dataExtractionRules` and disclose it, or exclude project metadata, feature caches, temporary exports, and document URIs. The provided policy discloses current Android-controlled backup/transfer behavior.

### Only ARM64 devices are supported

The app bundle filters to `arm64-v8a`. This satisfies Play's 64-bit requirement but excludes x86_64 Chromebooks/emulators and all 32-bit devices.

Recommendation: keep ARM64-only only if that is an intentional market decision. Otherwise test and add x86_64 and/or other supported ABIs through the app bundle.

## Medium and low findings

- `MainActivity` displays a hard-coded `target API 36` while the build currently targets API 37.
- Lint reports missing monochrome adaptive-icon layers for themed icons.
- User-facing strings are mostly hard-coded, so the app is effectively English-only and lint reports localization warnings.
- The release bundle contains dependency version markers and some AndroidX licenses, but there is no user-facing open-source notices page. Add one before wider distribution to make attribution maintenance explicit.
- Media3 1.11.0 is available while the project uses 1.10.1. Upgrade only after regression testing export and playback.
- The app uses no custom network security configuration. This is acceptable because it does not request `INTERNET`.

## Positive findings

- No `INTERNET`, broad storage, photo/video library, camera, microphone, location, contacts, advertising ID, SMS, Call Log, or all-files permission.
- Selected media is accessed with Android's system document picker and persisted read grants.
- No ads, analytics, accounts, payment SDKs, remote services, or WebView.
- Export destinations are user-selected through Android's system picker.
- Foreground services are non-exported, use immutable pending intents, show user-visible progress, and provide cancellation for export.
- Dynamic progress receivers are registered as not exported.
- Failed/cancelled exports remove or truncate incomplete destinations.
- R8 minification is enabled for release.
- The build targets API 37, exceeding Play's API 36 requirement that begins 2026-08-31.
- The app package is 64-bit compliant and uses modern uncompressed native libraries.
- Unit tests completed successfully, including model, editor math, persistence, guided tour, and inference tests.
- A real Pixel 10 Pro successfully installed and launched the signed `v0.10.3` release package on Android 17 while preserving the existing saved project.
- Eight Play-compatible release screenshots were captured from a source-unavailable saved inference; they contain no video frames or thumbnails.

## Commands and results

- `testDebugUnitTest`: passed
- `assembleDebug`: passed
- `bundleRelease`: produced an unsigned AAB
- `lintDebug` / `lintRelease`: failed with 10 errors and 42 warnings
- Debug APK size: approximately 63.8 MB
- Merged permissions: foreground service, media-processing foreground service, wake lock, notifications, and dependency-added network-state access; no internet permission

## Required final verification

1. Resolve every lint error and rerun unit tests and lint.
2. Test clean install, first run, analysis, interruption, cancellation, relinking, editing, and export on the supported API matrix.
3. Test the six-hour foreground-service timeout callback with a shortened device-config duration.
4. Build the final release AAB and sign it with the upload key.
5. Upload to Play internal testing and review the automated pre-launch report, device catalog, permissions, native-code warnings, and app size.
6. Verify the privacy-policy URL is public, non-editable to visitors, and linked from the app.
7. Complete Data safety, target audience, content rating, ads, app access, and foreground-service declarations.
8. Record the foreground-service demonstration using only a rights-cleared non-personal clip.
9. Recheck the prepared release screenshots if the UI changes before submission.
10. If applicable, complete the 12-tester/14-day closed test and apply for production access.
