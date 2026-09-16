# Export-compliance review

The app declares `ITSAppUsesNonExemptEncryption = false` in `App/Info.plist`.
The corresponding App Store Connect declaration is that the app does not use
non-exempt encryption. Apple permits this value for apps, including linked
libraries, that use no encryption or only exempt encryption. See Apple's
[key documentation](https://developer.apple.com/documentation/bundleresources/information-property-list/itsappusesnonexemptencryption)
and [export-compliance guidance](https://developer.apple.com/help/app-store-connect/manage-app-information/overview-of-export-compliance).

## Source review

CryptoKit calls in `AnalysisPipeline`, `ScoreAnalysis`, `SourceVideoAccess`,
`ProjectArchive`, `ServingSideInference`, `SideSwitchInference`, and
`SuppressionPolicyEngine` use SHA-256 for model integrity, source matching, cache
keys, and stable local identifiers. The review found no app encryption/decryption,
key agreement, custom encrypted communications, or network-upload client.
Photos/Files downloads and platform backup are Apple/provider functionality.

## Exact uploaded build inspected

The user-provided GitHub artifact `VolleySplice-1.0.0-1.zip` was inspected locally.
Its ZIP hash matches the GitHub Actions upload screenshot; the IPA hash matches
its accompanying `release.json`.

| Field | Value |
| --- | --- |
| App | `com.volleysplice.VolleySplice`, version `1.0.0`, build `1` |
| Source commit | `92c5fae3546173fd468498761736a79a3344b903` |
| Xcode | `26.3`, build `17C529` |
| Artifact ZIP SHA-256 | `120ffef4f044ef71f92962f86181cafdc52694c2db96b3a4c76e0bca03f312e0` |
| IPA SHA-256 | `7694fcbb21a400ccc3975e5beae2bb3d5fd78ffe72b0532a7a9120fbc577bfec` |
| Executable SHA-256 | `175b1969aa61f42a4c2b081cc2762932b12585d460a93b2fda9e559a613ec65a` |

The arm64 Mach-O executable's load commands and 1,854 symbol-table entries were
examined. Its CryptoKit references were `HashFunction` update/finalize/init,
`SHA256`, and `SHA256Digest`. No encryption primitives or separate third-party
crypto library were identified in that review. The source review and these binary
observations support the declaration above. This is not a proof from exhaustive
disassembly: stripped or statically linked code can limit symbol inspection.
Reassess if dependencies or cryptographic functionality change.

That uploaded build has **no** `ITSAppUsesNonExemptEncryption` key and **no**
`PrivacyInfo.xcprivacy`. It predates the privacy-manifest commit and this flag.
Changes cannot alter an already uploaded IPA: upload a newly built version
`1.0.0` with a fresh build number (for example `2` if unused).

## Subsequent Release checks

The Release audit verifies the Boolean `false` value in both the archive and
exported IPA, and records app identity, version/build, executable hashes, and the
IPA hash in `privacy-resource-audit.json`. It checks declaration propagation,
not cryptographic behavior. Keep that evidence with `release.json` for the exact
IPA uploaded. The changed binary still requires a new GitHub macOS build.

## Store URLs

- Privacy Policy: `https://www.volleysplice.com/privacy.html`
- Support: `https://www.volleysplice.com/support.html`
- Terms: `https://www.volleysplice.com/terms.html`
- Contact: `volleysplice@vafrederico.com`

Deploy the updated static site before using the new Support URL for submission.
The repository now supplies a standalone support document rather than relying
on the site's fallback app shell. Mailbox delivery is not validated by a website
build; the developer must keep the published contact address monitored.
