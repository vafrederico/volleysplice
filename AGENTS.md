<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

## Local production-app proxy

When the production app is served on port 3000, its optional reverse-proxy URL
is configured outside Git through `VOLLEYCUT_APP_BASE_URL`.

## Private research inputs and artifacts

Never embed private dataset files, original media names, identifying recording or
project IDs, machine/NAS paths, private URLs, or private artifact locations directly
in checked-in code, documentation, tests, generated reports, or client bundles. Use
stable recording, source-group, and artifact indexes from the external private
ledger. Resolve exact runtime values through `VOLLEYCUT_PRIVATE_LEDGER` (or its
`_WINDOWS`/`_POSIX` override) and other ignored local environment configuration.
Repository-relative links to checked-in code and public artifacts are allowed.

Keep the ledger and its mappings outside Git. A missing ledger entry must fail
closed; do not substitute a default private path or copy an identifying value into
a fallback. Before committing research changes, run the branch publication audit
described in [`docs/research/private-research-ledger.md`](docs/research/private-research-ledger.md)
against the changed files and branch history.

## Android release signing

Do not open PowerShell, a terminal window, or any other interactive process to sign
an APK or request the release-keystore password. Build the unsigned release APK,
provide the signing command, and ask the user to run it themselves. Never request,
read, handle, store, or pass the keystore password.

### Android version bumps and Google Play distribution

The production Android app is distributed through
`https://play.google.com/store/apps/details?id=com.volleycut.nativeanalysis`.
Keep website install links pointed at that listing. Do not check APKs into the
repository or copy them into the production website.

When publishing a new Android version:

1. Increment both `versionCode` and `versionName` in
   `android/app/build.gradle.kts`. Keep the debug `versionNameSuffix` so debug
   installs report the corresponding `-debug` version.
2. Build `assembleDebug` and `bundleRelease` for Google Play. If a local APK is
   needed, also build `assembleRelease`. Release artifacts must be signed by the
   user with the helpers documented in `android/README.md`; never handle the
   keystore password.
3. For a user-signed local APK, verify it with `apksigner`, confirm its package
   version with `aapt`, confirm that its signer certificate matches the established release
   certificate, and confirm that its non-signature ZIP payload matches the
   unsigned APK that was just built.
4. Keep generated APKs and release bundles in ignored build output directories.
5. Run the Android build/tests and, for website changes, the production static
   build checks. Do not start or serve the production app unless the user
   explicitly asks.

## Rally-model iteration ranking

When comparing or selecting rally-model, decoder, threshold, padding, epoch, or seed
iterations, use **Padded P/Core R F1** (`F1_padP_coreR`) as the primary descending
ranking metric:

```text
P_pad  = duration(padded_model intersection padded_human) / duration(padded_model)
R_core = duration(padded_model intersection core_human)   / duration(core_human)
F1_padP_coreR = 2 * P_pad * R_core / (P_pad + R_core)
```

Apply identical before/after padding to model and padded-human ranges, clip to video
bounds, and merge overlapping or touching ranges. Then join consecutive padded ranges
when the positive gap between them is **strictly less than 3 seconds**; the retained gap
becomes part of the export union. A gap of exactly 3 seconds remains a cut. Apply this
same short-gap rule to padded model and padded-human ranges before measuring union
duration, and record the configured join threshold in every evaluation artifact.
For a dataset ranking, pool intersection numerators and duration denominators across
recordings before calculating F1; do not average per-video F1 values. Compare only the
same recording/source-group scope, gold-label revision, and padding configuration.

Always evaluate and report the four symmetric padding cases `(before, after) =
(0, 0), (1, 1), (2, 2), (3, 3)` seconds. For every case, report pooled `P_pad`,
pooled `R_core`, `F1_padP_coreR`, padded model export duration, padded human export
duration, and their duration difference. Declare the target product padding before
the comparison and rank by that case. Never choose each model's best padding case;
the remaining three cases are required sensitivity results.

Treat label-document `ignoredIntervals` as outside the evaluation universe. Subtract
them from padded model, core human, and padded human interval unions before computing
metric numerators, denominators, or export-duration comparisons. Model predictions in
ignored time are neither true nor false positives. Never convert ignored spans into
dead-time negatives; `hardNegatives` are the separate construct for valid confusing
dead time. Do not rejoin ranges across an ignored interval after subtraction. All
compared models must use the same ignored-range revision and short-gap threshold.

Rank iterations on the declared development/validation scope. Never use the protected
test split to select an iteration. Continue to report event F1 and all predeclared
guardrails, but do not substitute event F1 for this ranking metric. Use the unambiguous
name `F1_padP_coreR` in artifacts and reports. See
[`docs/model-ranking-metric.md`](docs/model-ranking-metric.md) for the canonical contract.
