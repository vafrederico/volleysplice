# Android releases with GitHub Actions

The **Android release** workflow builds VolleySplice on GitHub's Ubuntu runner,
runs the JVM tests and debug/release lint, builds the required debug APK and
release Android App Bundle, signs the bundle with a dedicated Play upload key,
and retains the bundle plus R8 mapping and native symbols as workflow artifacts.
It can then publish to internal testing and, after a separate protected-environment
approval, promote that same version to a staged production rollout.

The workflow reads `versionCode` and `versionName` from
`app/build.gradle.kts`. Increment and commit both values before every release.
It does not change source versions during a run.

## Signing-key design

Do not place the permanent `volleycut-release.jks` direct-install key in GitHub.
Directly installed APK updates cannot recover from loss or disclosure of that
key. Instead, create a separate RSA Play **upload key**, register its public
certificate under Play Console **Test and release > Setup > App signing**, and
put only that resettable upload key in the `android-release` environment.
Google Play App Signing continues to hold the key that signs APKs delivered by
Google Play.

Generate the upload key yourself in a private directory outside the checkout and
cloud-synced folders. `keytool` prompts locally for passwords; never place them
in a command, source file, issue, or chat:

```powershell
& 'C:\Program Files\Android\Android Studio\jbr\bin\keytool.exe' -genkeypair `
  -keystore C:\private\volleysplice-play-upload.jks `
  -alias volleysplice-play-upload -keyalg RSA -keysize 4096 -validity 10000

& 'C:\Program Files\Android\Android Studio\jbr\bin\keytool.exe' -exportcert -rfc `
  -keystore C:\private\volleysplice-play-upload.jks `
  -alias volleysplice-play-upload `
  -file C:\private\volleysplice-play-upload-cert.pem
```

If Play currently treats the permanent direct-install key as its upload key,
use Play Console's upload-key reset/change flow to register the new certificate.
The Play **app signing certificate** and **upload certificate** are expected to
be different afterward. Preserve an encrypted backup of the upload keystore and
its passwords outside GitHub.

## Google Play API identities

1. Create or select a Google Cloud project and enable the **Google Play Android
   Developer API** (`androidpublisher.googleapis.com`).
2. Create two service accounts. Invite each service-account email under Play
   Console **Users and permissions**, scoped only to
   `com.volleycut.nativeanalysis`:
   - internal publisher: **View app information** and
     **Release apps to testing tracks**;
   - production publisher: **View app information** and
     **Release to production, exclude devices, and use Play App Signing**.
3. Configure a Google Cloud Workload Identity Pool/provider that trusts GitHub's
   `https://token.actions.githubusercontent.com/` issuer. Restrict its attribute
   condition to this repository. Permit the `android-release` environment
   subject to impersonate only the internal publisher and the
   `android-production` environment subject to impersonate only the production
   publisher. No service-account JSON key is needed.

GitHub environment jobs include the environment name in their OIDC subject. Use
the exact owner/repository and environment claims shown by GitHub when creating
the two `roles/iam.workloadIdentityUser` bindings. Prefer stable numeric owner
and repository claims in the provider condition where available.

## GitHub environments

Create **`android-release`** under repository **Settings > Environments**. Limit
it to trusted release branches or tags and add:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `ANDROID_UPLOAD_KEYSTORE_BASE64` | Base64 of the dedicated Play upload JKS |
| Secret | `ANDROID_UPLOAD_STORE_PASSWORD` | Upload keystore password |
| Secret | `ANDROID_UPLOAD_KEY_PASSWORD` | Upload private-key password |
| Variable | `ANDROID_UPLOAD_KEY_ALIAS` | For example `volleysplice-play-upload` |
| Variable | `ANDROID_UPLOAD_CERT_SHA256` | SHA-256 of the registered Play upload certificate |
| Variable | `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full Workload Identity provider resource name |
| Variable | `GCP_PLAY_PUBLISHER_SERVICE_ACCOUNT` | Internal publisher service-account email |

Create **`android-production`**, restrict it to trusted refs, require a reviewer,
prevent self-review when appropriate, and add only these variables:

| Kind | Name | Value |
| --- | --- | --- |
| Variable | `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full Workload Identity provider resource name |
| Variable | `GCP_PLAY_PUBLISHER_SERVICE_ACCOUNT` | Production publisher service-account email |

With GitHub CLI authenticated to the repository, the key owner can populate the
signing values without printing them. Commands without piped input prompt
privately for their value:

```powershell
$repo = 'OWNER/REPO'
[Convert]::ToBase64String(
  [IO.File]::ReadAllBytes('C:\private\volleysplice-play-upload.jks')
) | gh secret set ANDROID_UPLOAD_KEYSTORE_BASE64 --env android-release --repo $repo

gh secret set ANDROID_UPLOAD_STORE_PASSWORD --env android-release --repo $repo
gh secret set ANDROID_UPLOAD_KEY_PASSWORD --env android-release --repo $repo
gh variable set ANDROID_UPLOAD_KEY_ALIAS --env android-release --repo $repo `
  --body 'volleysplice-play-upload'
gh variable set ANDROID_UPLOAD_CERT_SHA256 --env android-release --repo $repo `
  --body 'REGISTERED_UPLOAD_CERTIFICATE_SHA256'
```

Add the Workload Identity provider and service-account variables through the
GitHub environment UI or `gh variable set`. Do not create or store a Google
service-account JSON key for this workflow.

## Run a release

1. Increment and commit both Android version fields. Every Google Play upload
   requires a new, higher `versionCode`.
2. Open **Actions > Android release > Run workflow** on a trusted ref.
3. Choose a destination:
   - **artifact-only** builds, signs, verifies, and stores the AAB without
     contacting Google Play;
   - **internal** also commits the version to internal testing;
   - **production** first publishes to internal testing, then pauses at the
     protected `android-production` environment. Test the Play-installed build
     before approving the staged production rollout.
4. For production, enter a rollout fraction greater than `0` and less than `1`.
   `0.10` releases to ten percent of eligible users. Complete or change the
   rollout later in Play Console after checking crashes, ANRs, and feedback.

The API commit submits the release to the selected track. Google Play review,
policy declarations, or Managed Publishing can delay when it becomes available.
New permission or foreground-service declarations may still require a Play
Console visit before an API release can proceed.

The GitHub artifact is retained for 30 days and contains the signed AAB,
`release.json`, `mapping.txt`, and native debug symbols when generated. Internal
publishing attaches the R8 mapping and any generated native-symbol ZIP to that
exact Play version for crash deobfuscation. The workflow removes the temporary
keystore even when an earlier step fails.

## Local credential-free validation

```powershell
python -m unittest discover -s android/scripts/tests -p 'test_*.py'
python android/scripts/ci-release.py metadata `
  --gradle-file android/app/build.gradle.kts
```

The first credentialed run must validate the actual keystore fingerprint,
Workload Identity trust, Play permissions, and upload. An artifact-only run is a
safe first check; follow it with internal testing before enabling production.
