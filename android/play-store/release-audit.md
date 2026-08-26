# VolleyCut Android Play Store release audit

Audit date: 2026-08-19  
Package: `com.volleycut.nativeanalysis`  
Audited version: `0.10.3` (`versionCode` 17)

## Verdict

The source-level release blockers found in the first audit are addressed. The app is still not an upload artifact until the final release build is signed, the foreground-service declaration/demo is completed, and the remaining supported-device validation is completed against this exact build.

## Resolved findings

### Android compatibility and lint

The release `minSdk` is now 34 (Android 14), the lowest platform required by the current implementation. This makes the Java 34 stream APIs and Java 33 byte-array conversion safe for every supported device. The Android 15 batched `MediaCodec.queueInputBuffers()` path remains explicitly API-gated and annotated. `MediaExtractor` sample flags are mapped to the separate `MediaCodec` input-flag bit field instead of being forwarded directly.

Checks after the fix:

- `testDebugUnitTest`: passed
- `lintDebug`: passed with 0 errors
- `lintRelease`: passed with 0 errors
- `assembleDebug`: passed
- `assembleRelease`: passed and produced an unsigned APK
- `bundleRelease`: passed and produced an unsigned AAB

Lint still reports non-blocking warnings (mostly localization/`UseKtx` suggestions, dependency freshness, ARM64-only ChromeOS support, and two adaptive-icon monochrome suggestions). They do not indicate unsafe API calls or a failed release gate.

### Foreground-service timeout handling

`ProjectAnalysisService` and `ExportService` now override `onTimeout(int, int)`. On timeout they cancel active work, clear queued work, mark affected inference projects as errors or export jobs as failed, remove incomplete export files, release foreground state and wake locks, and call `stopSelf()`. This follows Android's [foreground-service timeout guidance](https://developer.android.com/develop/background-work/services/fgs/timeout).

`ProcessingTimeoutTracker` stores a bounded local history and total count. The editor refreshes from its package-scoped broadcast, shows an acknowledgement dialog with operation/source/time/detail, and posts a notification that links back to the editor. No video bytes or network service are involved.

### Privacy, backup, and attribution

- The app footer now links to `https://volleycut.vafrederico.com/privacy.html`.
- The footer's **Open source** action presents the required dependency/license attribution and links to the Apache 2.0 license.
- `android:allowBackup="false"` is set, and `data_extraction_rules.xml` excludes the app root from both cloud backup and device transfer.
- The privacy and terms pages cover both the Android app and production web app and use `volleycut@vafrederico.com`.

### Benchmark surface

`MainActivity` is no longer in the release manifest. It is overlaid only by the debug manifest (where it remains exported for the existing ADB benchmark harness), and benchmark buttons are compiled behind `BuildConfig.DEBUG`. Release users therefore have only the launcher editor surface.

### Media3 1.11.0 review

The project remains on Media3 `1.10.1`. The official [Media3 release notes](https://developer.android.com/jetpack/androidx/releases/media3) describe 1.11.0 additions and fixes primarily for HLS/DASH, sessions, Cast, Ktor/network data sources, and broader extractor/Transformer behavior. VolleyCut uses local progressive MP4 playback and Transformer/MP4 export, with no HLS, DASH, Cast, Ktor, MediaSession, or network data source. The release notes contain no CVE/security-fix entry. There is no feature or security requirement in this app that justifies taking the dependency upgrade without a dedicated export/playback regression pass, so 1.10.1 is intentionally retained.

## Remaining submission work

### 1. Sign the final release artifact

The AAB produced by `bundleRelease` is unsigned. Use Play App Signing and have the key owner run `android/sign-aab.ps1` locally before uploading it; do not commit a private key or password. The same final source build must be used for the signed APK/AAB and any checked-in or production download artifact.

### 2. Complete Play declarations and demonstration

The manifest declares `mediaProcessing`. Play Console still requires the use-case, deferral/interruption impact, and a public or unlisted demonstration video for Android 14+ foreground services ([official requirements](https://support.google.com/googleplay/android-developer/answer/13392821)). Use the prepared answers in `play-console-answers.md` and record the demonstration with a rights-cleared, non-personal clip. The screenshot workflow intentionally did not open or relink a personal video.

### 3. Repeat device validation

Run a clean install and exercise choose/relink, analysis, interruption, editing, export, cancellation, project deletion, policy link, and open-source dialog on Android 14 through 17. Test the timeout path on Android 15 through 17 with Android's shortened foreground-service timeout configuration before release; Android 14 does not provide the `onTimeout` callback. Verify the app's supported ARM64 device catalog and the final signed artifact.

Pixel 10 Pro validation completed on 2026-08-19 using the current debug build (`0.10.3-debug`, Android 17/API 37): the editor launched, the footer links rendered, the open-source dialog opened, the privacy link resolved to the browser, and the debug-only benchmark activity opened with no video selected. A short local video inference completed successfully. A larger local HEVC clip reached Android's shortened `mediaProcessing` timeout; after decoder/worker teardown fixes, the service stopped without a native crash, the project was marked `ERROR`, the timeout dialog reported the interruption count, and the timeout notification was posted. The test clip stayed on the device and was not used for submission screenshots or other release assets. The temporary device timeout override was removed after testing.

### 4. Refresh release assets

If the UI changes after the captured screenshots, recheck the four featured Play screenshots and their eight source captures. Build/sign the new version before replacing the existing signed APK in `android/releases/` or the production download copy.

## Known non-blocking trade-offs

- The app intentionally ships arm64-v8a only; x86_64 ChromeOS/emulators are not supported.
- The UI is English-only and still has localization lint warnings.
- Adaptive icons do not yet include monochrome layers.
- Media3, OpenCV, and `org.json` have newer versions available, but none is required for this release's supported feature set.
- The release bundle is minified with R8.

## Positive findings

- No `INTERNET`, broad storage, camera, microphone, location, contacts, advertising ID, SMS, Call Log, or all-files permission.
- Selected media is accessed with Android's system document picker and persisted read grants.
- No ads, analytics, accounts, payment SDKs, remote services, or WebView.
- Export destinations are user-selected through Android's system picker.
- Foreground services are non-exported, use immutable pending intents, show progress, and provide cancellation for export.
- Dynamic progress receivers are registered as not exported.
- Failed/cancelled exports remove or truncate incomplete destinations.
- The build targets API 37, exceeding the [Play target API 36 requirement](https://developer.android.com/google/play/requirements/target-sdk) for new apps and updates beginning August 31, 2026.
- Unit tests and both debug/release lint checks pass after the fixes.
