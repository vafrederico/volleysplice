# Automatic app regression gates

`.github/workflows/app-regression.yml` runs on every branch push and on PR open,
reopen, and synchronization. It deliberately has no path filters, so changes to
shared runtime code, fixtures, build helpers, or workflows receive the same gates.
New runs cancel older runs for the same event and PR/branch; unrelated branches
and the existing manual release workflows have separate concurrency groups.

The web job installs both npm lockfiles and runs:

```sh
npm ci --no-audit --no-fund
npm --prefix prod ci --no-audit --no-fund
npm run test:web
node --test scripts/browser-distilled-large-contract.test.mjs
npm --prefix prod run test:review-persistence
npm --prefix prod run test:inference-ui
npm --prefix prod run test:youtube-ui
npm --prefix prod run build
```

These cover unit and numerical contracts, IndexedDB/localStorage recovery, and
real React persistence, inference-progress, and chapter-export UI through JSDOM
and Vite middleware without a listening server. The production build runs asset
preparation, TypeScript checks, static export, and the existing release hash,
encoder/head pairing, legal asset, relative URL, and asset-size checks.
Two existing optional editor-lab tests require a private frozen fixture and skip
when it is absent; they are outside this public gate. No ledger or media is needed.

The Android job uses JDK 17, the checked-in Gradle wrapper, Android platform 37.0,
and build tools 37.0.0. Python 3 with venv support and pip is required on the Linux
runner. It installs `onnx==1.23.0` in a job-local environment, runs bootstrap rejection
tests, and calls `scripts/prepare-android-neural-assets.py --output <job-temp-dir>`.
That existing helper reverses only the qualified web pooling rewrite and verifies
every output byte against the Android manifest. All inputs are checked-in public
runtime/model assets; it downloads no weights and accesses no private inputs.

With `VOLLEYCUT_NEURAL_ASSETS_DIR` set to that output, the gate runs:

```sh
cd android
./gradlew --no-daemon --stacktrace --max-workers=1 \
  :app:prepareNeuralAssets :app:prepareServingAssets :app:testDebugUnitTest
```

Asset preparation is mandatory. Gradle rechecks manifests, hashes, sizes, and
encoder/head pairing, plus the existing frozen JSON checks wired through
`preBuild`. Missing or changed bundles fail the gate; there is no task exclusion
or fallback. Reports remain in ignored `android/app/build` directories. This gate
does not build APKs, run devices, sign, publish, or deploy anything.

## Forgejo qualification and security

The workflow uses the same Linux runner label, checkout, Java, and Android action
pins as the repository's existing workflows, plus a commit-pinned setup-node with
Node 24. It avoids GitHub-only artifact wrappers, shared CI caches, release
environments, and credentials. Checkout does not persist its automatic token;
shell steps unset automatic token variables before executing repository code.
Forks use `pull_request`, subject to the host's trust approval, and never
`pull_request_target`. The existing standalone Gitleaks workflow remains separate
and unchanged; release/promotion workflows remain manual.

The declared permission is `contents: read`. Forgejo's permission semantics differ
from GitHub's, so that declaration alone is not proof of a read-only token on every
host/event. Runner isolation and the host's fork-trust policy remain required.
See the official [Forgejo 16 workflow reference](https://forgejo.org/docs/v16.0/user/actions/reference/),
[compatibility notes](https://forgejo.org/docs/v16.0/user/actions/github-actions/),
and [fork trust policy](https://forgejo.org/docs/v16.0/user/actions/security-pull-request/).

Local command success and YAML parsing do not qualify host execution. Before
making these jobs required branch checks, observe successful PR and push runs on
the actual Forgejo runner, verify checkout resolves the intended PR revision,
confirm pinned action/Node support and SDK availability, and exercise cancellation
and an approved fork PR without release secrets. The PR records author validation
separately from outstanding host qualification.

## Pending iOS follow-up

An automatic iOS unit gate remains pending the parent iOS phase, including its
macOS runner and public fixture/asset prerequisites. This Android-phase change
does not edit or execute iOS, macOS, or Swift sources and does not count the manual
iOS release workflow as an automatic regression gate.
