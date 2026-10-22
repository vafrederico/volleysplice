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
python3 -m unittest discover -s scripts -p test_android_regression_gate.py -v
python3 scripts/android-regression-gate.py
```

The runner calls the checked-in wrapper with `--no-daemon --stacktrace
--console=plain --max-workers=1 --no-build-cache --no-configuration-cache` and the
gate init script. Task-local `--rerun` forces `:app:prepareNeuralAssets`,
`:app:prepareServingAssets`, and `:app:testDebugUnitTest` to execute while other
compilation tasks can remain up to date. Asset preparation rechecks manifests,
hashes, sizes, and encoder/head pairing. The three existing frozen JSON checks
wired through `preBuild` are also mandatory.

Each invocation allocates a new directory under ignored
`android/build/regression-gate`. The init script routes JUnit XML there and records
the six mandatory task outcomes with a unique run ID. The runner rejects missing,
duplicate, skipped, `NO-SOURCE`, cached, up-to-date, failed, or no-work outcomes for
those tasks. It also requires fresh, nonempty XML, a positive executed-case count,
matching summary/case counts, and zero skips, errors, or failures in both summaries
and case results. Stale file times and suite timestamps fail. Earlier reports in
the default test directory cannot satisfy this run. Gradle exit code zero alone
never passes the gate.

The negative tests include four small projects executed by the pinned wrapper:
empty test inputs, a disabled test task, disabled asset preparation, and excluded
asset preparation. Each project exits Gradle successfully; the gate must still
reject it, even with a prior passing report present. Separate report fixtures
exercise missing/empty/stale reports, inaccurate counts, and failed/skipped cases.
These probes use no Android plugins, test dependencies, media, or native builds.
Missing or changed bundles fail without a task exclusion or fallback. This gate
does not build APKs, run devices, sign, publish, or deploy anything.

## Forgejo qualification and security

The workflow uses the same Linux runner label, checkout, Java, and Android action
pins as the repository's existing workflows, plus a commit-pinned setup-node with
Node 24. All action origins are fully qualified: a short reference would depend
on the host's action mirror, which does not contain every third-party action.
See [Forgejo action resolution](https://forgejo.org/docs/v16.0/user/actions/actions/#using-actions).
GitHub-specific actionlint needs these URL prefixes normalized in a temporary
copy for its checks; validate the actual Forgejo YAML and pinned origins too.
The workflow avoids GitHub-only artifact wrappers, shared CI caches, release
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
