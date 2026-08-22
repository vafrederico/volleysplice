# VolleyCut

VolleyCut is a local-first volleyball video editor that finds likely rallies, lets a
human correct the proposed cuts, and exports the retained footage. Video analysis and
editing run on the user's device; production inference does not upload source media.

The repository contains three distinct surfaces:

| Surface | Purpose | Status |
| --- | --- | --- |
| [`prod/`](prod/) | Static browser application for local analysis, review, and MP4 export | Production web client |
| [`android/`](android/) | Native on-device analysis, editor, and MP4 exporter | Production Android client |
| Root Next.js/Python workspace | Labeling, comparison, diagnostics, model feedback, training, and evaluation | Internal model lab |

Model suggestions and confidence scores are review aids, not semantic truth or calibrated
probabilities. Inspect every proposed boundary, especially single-model disagreements and
optional suppression suggestions, before exporting.

## Production clients

### Browser

The production web app is a standalone Vite application with no API routes, media
catalog, server database, or upload path. A user selects a local video, marks the game
window, runs the two-model production ensemble, reviews and edits the inferred ranges,
and exports MP4 or JSON output. Optional suppression suggestions are visible,
configurable, and reversible; they are off by default.

Features, inference results, projects, and edit drafts persist in browser storage. The
source video itself is not copied into that storage, so playback or export after a
restart requires reconnecting the same file. Model-feedback JSON can be exported for
future training without embedding video bytes.

Requirements are Node.js 24 and a current Chrome or Edge browser, or Safari 26 on
iOS/macOS:

```bash
cd prod
npm install
npm run dev
```

For a verified static production build:

```bash
cd prod
npm ci
npm run build
npm run preview
```

See [`prod/README.md`](prod/README.md) for browser support, HTTPS/LAN setup, persistent
project behavior, export details, runtime assets, and deployment.

### Android

The native app performs video/audio feature extraction, production inference, editing,
draft persistence, source relinking, model-feedback export, and exact-boundary MP4 export
without a network permission. It targets the Pixel 10 Pro and packages `arm64-v8a`.

The current checked-in release is
[`VolleyCut v0.10.3`](android/releases/VolleyCut-v0.10.3-arm64-release-signed.apk). See
[`android/README.md`](android/README.md) for the SDK requirements, debug build/install
workflow, editor behavior, benchmarks, cache behavior, and native-versus-browser parity
caveats.

## Production model contract

The deployed system samples the selected game window at 4 Hz, generates 104 base
audiovisual features, gathers five temporal contexts into 520 model inputs, and runs two
independent rally/serve/dead-state bundles. Their overlapping intervals are unioned, and
one-model-only regions remain visible as review-priority disagreements. A separate held
suppression model can propose reversible removals from eligible disagreement regions.

The following documents are the maintained contracts:

- [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) — inference heads, decoders,
  production ensemble, and suppression policy;
- [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md) — exact production feature formulas,
  historical profiles, research-only features, and rebuild/parity requirements;
- [`MODELS.md`](MODELS.md) — trained artifacts, feature profiles, exact fitting videos,
  comparison predecessors, changes, and disposition;
- [`docs/model-ranking-metric.md`](docs/model-ranking-metric.md) — canonical
  `F1_padP_coreR` selection and reporting contract.

## Internal model lab

The repository root is an internal NAS-backed development application, not the public
production client. It hosts the gold-label workstation, model/source comparison views,
model-feedback import, suppression and side-switch review, on-device experiments, and
the Python training/evaluation toolchain. The retained side-switch v2 specialist is a
research/review-ranking artifact and is not part of either production client; its exact
features and lineage are registered in
[`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md) and [`MODELS.md`](MODELS.md). The implemented
side-switch v3 cadence/low-resolution ranker and the v1 no-blur counterfactual both failed
their promotion gates and are tracked there as research only; neither is a shipped model.
The v4 multi-frame, court-normalized side-identity study improves on v3 but also remains
research-only because its exact precision is still 16.22%.
The v5 player-isolation study improves exact precision to 25.00%; persistent orientation
was a validation-selected no-op, and v5 also remains research-only.
The v6 quantized-person/adaptive-prototype study improves raw-phone row AP to 45.47% but
regresses candidate-window exact F1 to 24.10%; it is not promoted and v5 remains the
best event-level side-switch artifact. The 61-gap V5/V5-state/V6 proposal union is
available as three separate recording timelines in `/side-switch-review` because the
earlier heuristic marker seeds are not exhaustive. That review page also exposes
explicit/candidate/full-video human marker rails, a per-recording continuous-review
flag, production model range labels, and the corrected production-editor/final-export
timelines. New full-video markers are stored separately from the frozen candidate
decision artifact.
The completed 50-event continuous review now provides exhaustive raw-phone truth: only
33/50 switches are covered by any modern candidate gap with four-second boundary
allowance. No-cadence V5-state leads the comparable macro-recall ranking at 56.36%,
while local peak plus soft count has the best pooled F1 at 44.64%; neither is promoted.
See the
[`full-video marker audit`](docs/research/side-switch-full-video-marker-audit-2026-08-21.md).
The production-state follow-up reuses the on-device ensemble as soft V5 evidence and
raises retrospective exact F1 to 30.14% while reducing proposals 44→38, but count
accuracy and label completeness prevent promotion. Serve-grounded V6 regresses, hard
gates fail, and suppression is excluded for target/data overlap.
The controlled no-cadence V5-state refit confirms that the re-anchored seven-point path
can cascade errors: exact recall rises 31.43%→80.00% and F1 30.14%→44.44%, retaining
all cadence true positives, but proposals rise 38→91. It is tracked as a research-only
decoder diagnostic; the next iteration needs appearance-local cluster suppression or a
soft count prior before any on-device promotion.
The follow-up confirms adjacent-gap peak suppression as the stable cleanup: its locked
raw-phone ablation raises exact F1 to 48.15%. Peak suppression plus a soft post-six
penalty reaches 49.48% exact/57.73% ±2 F1 at 62 proposals, but needs confirmation on
new labels. A separate production rally/dead/serve context head wins historical
validation but fails to improve retrospective exact F1, so production outputs remain
soft features rather than eligibility gates.

### Setup

Requirements are Node.js and npm, Python 3 with `venv`, and FFmpeg/FFprobe with H.264
support.

```bash
npm install
npm run analysis:setup
cp .env.example .env.local
```

Set `VOLLEYCUT_DATA_ROOT` in the ignored `.env.local`. Configure
`VOLLEYCUT_LABELING_WORKSPACE`, `VOLLEYCUT_INTAKE_WORKSPACE`, or
`VOLLEYCUT_MODEL_FEEDBACK_ROOT` when those stores are not under the default data root.
Large source media, feature caches, labels, and trained artifacts remain outside Git.
`.env.example` documents the optional no-beach workspaces, proxy acceleration, and LAN
development origins.

Start the internal application with:

```bash
npm run dev -- --hostname 0.0.0.0
```

Use the printed LAN address. Important routes include:

- `/label` — prepared-task gold labeling with production and blind-AI reference tracks;
- `/model-feedback` — inspect production web/Android feedback bundles and optionally
  link their source video;
- `/suppression-review` — visually audit learned suppression behavior;
- `/side-switch-review` — review side-switch proposals, place exhaustive full-video
  switch markers, and compare them with the saved production editor timeline;
- `/serving-side-review` — review the current serving-side diagnostics;
- `/on-device`, `/on-device-batch`, `/audio-benchmark`, and `/video-benchmark` —
  internal parity and performance tools.

The exact labeling boundary policy, optional hard negatives, ignored intervals, court
geometry, side switches, sparse cues, and save workflow are documented in
[`docs/labeling-guide.md`](docs/labeling-guide.md).

## Model-development workflow

1. Prepare immutable proxy/tasks and label the complete recordings in `/label`. Keep
   human labels distinct from production or blind-AI prelabels.
2. Mark ambiguous time as `ignoredIntervals`; record valid confusing dead time as
   `hardNegatives`. Ignored time is outside both fitting and evaluation.
3. Freeze source-group-aware fitting and development/validation scopes. Do not use the
   protected test split to select features, models, seeds, thresholds, decoders, or
   padding.
4. Generate features under an immutable profile from
   [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md), then train and evaluate with the exact
   label/manifest revision.
5. Rank comparable iterations by the predeclared product-padding
   `F1_padP_coreR` case. Always report symmetric 0, 1, 2, and 3 second padding
   sensitivity using the canonical short-gap and ignored-interval rules.
6. Register every durable artifact or report-only training study in
   [`MODELS.md`](MODELS.md), including exact fitting videos and what changed from the
   predecessor it was intended to improve or compare against.
7. Before promotion, update both production clients and prove model/schema/cache and
   interval parity with the checked-in golden tests.

[`analysis/README.md`](analysis/README.md) contains the full Python command reference for
feature extraction, labeling manifests, training, inference, and research experiments.
Retained experiment decisions live under [`docs/research/`](docs/research/); they are
historical evidence, not the production specification.

## Verification

Run the root model-lab checks with:

```bash
npm test
npm run lint
npm run build
```

Verify the production browser independently because it has its own dependencies and
build contract:

```bash
cd prod
npm ci
npm run build
```

Android unit/build commands and device prerequisites are maintained in
[`android/README.md`](android/README.md). Release signing is intentionally separate from
the normal automated build workflow.

## Repository layout

- `prod/` — standalone static production browser client and checked-in runtime assets;
- `android/` — native Android client, tests, and signed release artifact;
- `app/`, `components/`, `lib/` — internal Next.js model lab and local NAS routes;
- `analysis/` — Python/FFmpeg feature, model, inference, and evaluation implementation;
- `scripts/` — dataset, experiment, parity, export, and reporting entry points;
- `tests/` — web/model golden fixtures and integration tests;
- `docs/` — maintained contracts, labeling guide, implementation plans, and research;
- `public/on-device/` and `android/app/src/main/assets/` — internal/native model and
  parity assets;
- `data/` — small checked-in reports/fixtures plus ignored local fallback outputs;
- external `$VOLLEYCUT_*` roots — source videos, proxies, labels, feature caches,
  analyses, models, and feedback bundles.

## Known constraints

- The current statistical feature models do not explicitly detect the ball, players,
  net, score, touches, actions, or team identity.
- Stationary, landscape, full-court footage remains the strongest input. Camera motion,
  cropped courts, neighboring play, walking, setup, retrieval, and celebrations are
  important confusing cases and require review.
- Quiet or visually subtle rallies can still be missed. The two-model union favors
  retaining extra footage over silently deleting live play.
- Browser and Android media decoders can produce small feature-distribution differences;
  use the parity tests and platform reports rather than assuming byte-identical front
  ends.
- The browser client intentionally keeps source video local and therefore cannot reopen
  playback/export until the original file is reconnected. Android similarly depends on
  a valid document grant or explicit relinking.
