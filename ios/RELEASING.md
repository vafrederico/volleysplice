# iOS releases from Windows with GitHub Actions

The **iOS release** workflow builds on GitHub's `macos-26` runner with Xcode
26.3, runs the portable Swift tests, archives Release, and exports an App Store
Connect signed IPA. No local Mac or lab VM is required. It uses the existing
source manifest and checksum-pinned OpenCV 4.12.0 framework.

## One-time Apple setup

1. Enroll in the paid [Apple Developer Program](https://developer.apple.com/programs/).
   Complete any pending agreements. A free personal development team cannot
   distribute through TestFlight or the App Store.
2. In [Certificates, Identifiers & Profiles](https://developer.apple.com/account/resources/),
   register an explicit App ID for **`com.volleysplice.VolleySplice`** under the
   intended team. Keep this identifier: it also appears in the app's background
   task configuration. Find your 10-character **Team ID** in Membership details.
3. Create an **Apple Distribution** certificate using the Windows instructions
   below (or use an existing distribution `.p12` containing its private key).
4. Create an **App Store Connect** distribution provisioning profile for that
   App ID and that exact distribution certificate, then download the
   `.mobileprovision` file. Do not choose Development, Ad Hoc, or Enterprise.
5. In [App Store Connect](https://appstoreconnect.apple.com/), create the iOS app
   record using that Bundle ID, your app name, primary language, and a unique SKU
   of your choosing. The app record is required before uploading.

Apple documents [distribution profiles](https://developer.apple.com/help/account/provisioning-profiles/create-an-app-store-provisioning-profile)
and [uploading builds](https://developer.apple.com/help/app-store-connect/manage-builds/upload-builds).

## Create a distribution certificate without a Mac

Use OpenSSL on Windows, for example the executable bundled with Git for Windows
at `C:\Program Files\Git\mingw64\bin\openssl.exe`. Run these commands yourself in
PowerShell, in a private folder **outside the checkout and cloud-synced folders**.
Replace the example name/email with yours. OpenSSL prompts for passwords; keep
them in your password manager, never in commands or chat.

```powershell
$openssl = 'C:\Program Files\Git\mingw64\bin\openssl.exe'
& $openssl genrsa -aes256 -out apple-distribution.key.pem 2048
& $openssl req -new -sha256 -key apple-distribution.key.pem `
  -out apple-distribution.csr -subj '/emailAddress=you@example.com/CN=Your Name/C=US'
```

In the Apple developer portal, add an **Apple Distribution** certificate, upload
`apple-distribution.csr`, and download the issued certificate into this folder as
`distribution.cer`. Then run:

```powershell
& $openssl x509 -inform DER -in distribution.cer -out distribution.pem
& $openssl pkcs12 -export -legacy `
  -provider-path 'C:\Program Files\Git\mingw64\lib\ossl-modules' `
  -inkey apple-distribution.key.pem `
  -in distribution.pem -out distribution.p12 -name 'VolleySplice Distribution'
```

The export password is the value for `IOS_DISTRIBUTION_P12_PASSWORD`. `-legacy`
uses a PKCS#12 format compatible with macOS Keychain (these commands assume
OpenSSL 3). Preserve the encrypted private key and P12 securely: the downloaded
`.cer` alone is insufficient to sign. Do not revoke existing certificates just
to make room without checking what uses them.

If OpenSSL reports `unable to load provider legacy`, use the `mingw64\bin`
executable and explicit provider path above. Git's `usr\bin` executable may
look in `/usr/lib/openssl/ossl-modules`, where this installation has no legacy
module. Existing keys, CSRs, and certificates can be reused; only rerun the
P12 export. The executable and provider DLL must come from the matching build.

## Configure the repository

Push/merge the workflow onto the repository's default branch so GitHub shows
**Actions → iOS release → Run workflow**. Under **Settings → Environments**, create
an environment named **`ios-release`**. Restrict its deployment branches/tags to
trusted release refs; anyone able to modify a dispatched workflow on an allowed
ref could access its secrets. GitHub-hosted macOS runner quota/billing must be
available for this repository.

Add this **environment variable**:

| Name | Value |
| --- | --- |
| `IOS_TEAM_ID` | Your 10-character Apple Developer Team ID |

Add these **environment secrets**:

| Name | Value |
| --- | --- |
| `IOS_DISTRIBUTION_P12_BASE64` | Base64 of the distribution certificate **and private key** P12 |
| `IOS_DISTRIBUTION_P12_PASSWORD` | The P12 export password |
| `IOS_APP_STORE_PROFILE_BASE64` | Base64 of the downloaded App Store Connect `.mobileprovision` |

To upload directly from GitHub, create a **team API key** in App Store Connect →
Users and Access → Integrations → App Store Connect API. The account holder may
need to request API access first. Assign **Developer** access for build uploads,
download the `.p8` private key (available to download only once), and note the
Key ID and Issuer ID. See Apple's [API key instructions](https://developer.apple.com/help/app-store-connect/get-started/app-store-connect-api).

Add three more environment secrets for the optional upload:

| Name | Value |
| --- | --- |
| `ASC_KEY_ID` | Key ID of that team API key |
| `ASC_ISSUER_ID` | Issuer ID from the team API page (not the Developer Team ID) |
| `ASC_PRIVATE_KEY_P8` | Entire `.p8` text, including BEGIN/END lines; **not base64** |

### Provide files from PowerShell

You do not need to send any credentials to the coding agent. With GitHub CLI
authenticated to your repository, these commands send the file contents directly
to GitHub secrets without printing them. Replace `OWNER/REPO` and file paths:

```powershell
$repo = 'OWNER/REPO'
gh variable set IOS_TEAM_ID --env ios-release --repo $repo --body 'YOURTEAMID'
[Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\private\distribution.p12')) |
  gh secret set IOS_DISTRIBUTION_P12_BASE64 --env ios-release --repo $repo
gh secret set IOS_DISTRIBUTION_P12_PASSWORD --env ios-release --repo $repo
[Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\private\VolleySplice.mobileprovision')) |
  gh secret set IOS_APP_STORE_PROFILE_BASE64 --env ios-release --repo $repo

# Optional App Store Connect upload:
gh secret set ASC_KEY_ID --env ios-release --repo $repo
gh secret set ASC_ISSUER_ID --env ios-release --repo $repo
Get-Content -Raw 'C:\private\AuthKey_YOURKEYID.p8' |
  gh secret set ASC_PRIVATE_KEY_P8 --env ios-release --repo $repo
```

Commands without piped input prompt for the value. Alternatively use the
environment's **Add environment secret** form; generate base64 locally and paste
it there. Never commit credentials, paste them in an issue/chat, or pass them as
workflow inputs. No Apple ID password, two-factor code, or personal GitHub token
is needed by the workflow. The runner generates its own temporary keychain
password and removes the imported credentials in an `always()` cleanup step,
following [GitHub's signing pattern](https://docs.github.com/en/actions/how-tos/deploy/deploy-to-third-party-platforms/sign-xcode-applications).

## Run a release

1. Open **Actions → iOS release → Run workflow** and select a trusted branch/tag.
2. Enter a three-part version such as `1.0.0` and an increasing build number
   from `1` through `9999`. Use a new number for each upload, including reruns of
   an already uploaded build. These override Xcode settings for this run only;
   source version defaults are not edited.
3. Leave **Upload to App Store Connect** unchecked for an artifact-only build,
   or check it to send the IPA to Apple. Only the latter needs API key secrets.
4. Download `VolleySplice-VERSION-BUILD` from the run's Artifacts. It contains
   `VolleySplice.ipa`, `dSYMs.zip`, `VolleySplice.xcarchive.zip`,
   `privacy-resource-audit.json`, and `release.json` with the commit, version,
   Xcode version, and IPA SHA-256. Artifacts expire after 14 days; retain symbols
   securely for released builds. The artifact is saved before upload so it remains
   available if Apple rejects the upload.
5. After upload, wait for Apple processing, complete export-compliance questions,
   and configure TestFlight testers in App Store Connect. External testing may
   require Beta App Review. For public release, complete screenshots, privacy
   declarations, metadata, and App Review submission separately.

An App Store IPA cannot be directly sideloaded onto an iPhone/iPad; use TestFlight
for installation. This workflow does not submit for App Review or publish a GitHub
release. It does not certify physical-device behavior or App Store acceptance.

## Maintenance and validation

Renew expiring distribution certificates/profiles and update their GitHub secrets
together. The workflow rejects expired, wrong-team, wrong-app, and non-store
profiles and checks the imported identity against the profile's certificate.
Signing is manual; it does not create or modify signing assets in your Apple account.

The Xcode path is deliberately pinned. If GitHub removes it or Apple's SDK
submission requirements advance, update `DEVELOPER_DIR` in the workflow after
checking the [runner image inventory](https://github.com/actions/runner-images/blob/main/images/macos/macos-26-arm64-Readme.md).
Local helper checks: `python -m unittest discover -s ios/scripts/tests -p 'test_*.py'`.
See [privacy and Release resource checks](PRIVACY.md) for the manifest reasons,
archive/IPA checks, and Xcode Organizer privacy-report review.
See [export compliance](EXPORT-COMPLIANCE.md) for the encryption declaration,
the inspected uploaded build, and App Store contact URLs.
The first credentialed GitHub run must validate the actual macOS archive/export
and optional upload; those cannot be executed on Windows.
