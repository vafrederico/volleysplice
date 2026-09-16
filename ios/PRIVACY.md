# iOS privacy manifest and Release resources

`App/PrivacyInfo.xcprivacy` declares no tracking, no tracking domains, and no
collected data types. The native app processes selected recordings locally and
has no analytics/upload client. User-directed Files/Photos exports are local app
functionality; the declarations must be revisited if remote collection is added.

The reviewed required-reason uses are:

| Category | Reason | Current use |
| --- | --- | --- |
| User Defaults | `CA92.1` | App-owned preferences and tour progress through `@AppStorage` and `UserDefaults.standard`. |
| System boot time | `35F9.1` | `AnalysisProgress` and `ProcessingQueue` use `systemUptime` for elapsed-time/rate calculations and progress throttling, not device identification. |
| File timestamps | `C617.1` | Project/checkpoint refresh, persistence validation, and storage management inside the app container. |
| File timestamps | `3B52.1` | Metadata for user-selected source recordings, including source matching and reconnect validation. |

These map to Apple's [required-reason API policy](https://developer.apple.com/documentation/bundleresources/describing-use-of-required-reason-api)
and [approved reasons](https://developer.apple.com/documentation/bundleresources/app-privacy-configuration/nsprivacyaccessedapitypes/nsprivacyaccessedapitypereasons).
Local-only functionality still requires the declarations. Neither the manifest
nor a generated report establishes that every linked SDK's behavior is compliant.

## Packaging

The project generator includes the manifest in Copy Bundle Resources and the
source packager retains `.xcprivacy` files. Release uses
`EXCLUDED_SOURCE_FILE_NAMES` to omit `golden.json`, `base.bin`, and fixture/golden
resources. Debug retains them. The canonical-inference diagnostic entry points
are also compiled only in Debug.

The five production model JSON files and branding are required app resources,
even though the staging directory is named `Fixtures`. They remain in Release.
Tests and their input files remain available on the build runner for `swift test`;
they must not be copied into the shipping app.

## Final archive checks

Before saving or uploading a Release, `scripts/audit-release.py` checks both the
archived `.app` and the exported IPA. It requires the reviewed app manifest and
production models, rejects test fixtures/test bundles and unexpected JSON/media
resources, and inventories all bundled privacy manifests. It fails on unexpected
tracking/data-collection declarations. CI saves `privacy-resource-audit.json`
alongside `VolleySplice.xcarchive.zip` from that exact build.

The JSON is a resource/declaration inventory, **not Xcode's generated privacy
report**, and it does not scan API usage in the linked executable. OpenCV is
statically linked, so its compiled behavior also needs to be considered when
reviewing Apple's validation feedback; absence of a separate SDK bundle is not
proof that no SDK uses required-reason APIs.

For Apple's report, download and expand `VolleySplice.xcarchive.zip` on a Mac,
open it in Xcode Organizer, Control-click the archive, and choose **Generate
Privacy Report**. Compare its data-collection/tracking declarations with this
manifest and App Store Connect's privacy answers. Review the manifest's required
API reasons and Apple's upload validation separately; Xcode's report aggregates
declarations rather than proving API coverage. Apple's documented report workflow
is described in [Describing data use in privacy manifests](https://developer.apple.com/documentation/bundleresources/describing-data-use-in-privacy-manifests).

A new credentialed GitHub Release build is required to validate the actual archive
and IPA. Xcode Organizer report generation has not been performed on this Windows
host; the workflow retains the archive so that check can be done on a Mac later.
