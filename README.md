# VolleySplice

**Bump. Set. Splice.**

[Open the web app](https://www.volleysplice.com/) ·
[Get the Android app on Google Play](https://play.google.com/store/apps/details?id=com.volleycut.nativeanalysis) ·
[Download for iPhone and iPad on the App Store](https://apps.apple.com/us/app/volleysplice/id6812621538)

VolleySplice is a local-first volleyball video editor that finds likely rallies,
lets a human correct the proposed cuts, and exports the retained footage. Video
analysis and editing run on the user's device; production inference does not upload
source media.

Formerly VolleyCut. See [brand assets and compatibility notes](docs/brand.md).

## Applications

| Surface | Purpose | Availability |
| --- | --- | --- |
| [`prod/`](prod/) | Static browser app for local analysis, review, and MP4 export | Published at [volleysplice.com](https://www.volleysplice.com/) |
| [`android/`](android/) | Native on-device analysis, editor, and MP4 exporter | Published on [Google Play](https://play.google.com/store/apps/details?id=com.volleycut.nativeanalysis) |
| [`ios/`](ios/) | Native Swift/SwiftUI port for iPhone and iPad | Published on [the App Store](https://apps.apple.com/us/app/volleysplice/id6812621538) |
| Root Next.js workspace | V2 labeling and model-feedback import | Local development tool |

Model suggestions and confidence scores are review aids, not semantic truth or
calibrated probabilities. Inspect proposed boundaries before exporting.

### Web

The production web app is a standalone Vite application with no API routes, media
catalog, server database, or upload path. A user selects a local video, runs the
production ensemble, reviews the inferred ranges, and exports MP4 or JSON output.
Projects and edit drafts persist in browser storage, while source video remains local.

```bash
cd prod
npm ci
npm run build
npm run preview
```

See [`prod/README.md`](prod/README.md) for browser support, local development,
project persistence, export behavior, runtime assets, and deployment.

### Android

The native Android app performs on-device feature extraction, inference, editing,
draft persistence, source relinking, model-feedback export, score tracking, and
exact-boundary MP4 export without requesting network access. It packages `arm64-v8a`.

See [`android/README.md`](android/README.md) for SDK requirements, debug builds,
testing, editor behavior, performance, and release procedures.

### iOS and iPadOS

The [`ios/`](ios/) directory contains the native Swift/SwiftUI implementation and
portable `VolleyCore` code. It supports the local analysis, project, editing, scoring,
and export architecture used by the other clients.

The iOS and iPadOS app is published on [the Apple App Store](https://apps.apple.com/us/app/volleysplice/id6812621538).
Use the native app on iPhone and iPad; the web editor is unavailable on iOS. See [`ios/README.md`](ios/README.md) for supported platforms,
source layout, build preparation, tests, and current limitations.

## Labeling workspace

The root Next.js app is a focused local labeling tool. `/` redirects to `/labelv2`,
the maintained V2 labeling UI. `/model-feedback` imports model-feedback JSON exported
by the web or Android app, optionally links its source video, and makes the imported
project available to the labeling workflow.

Requirements are Node.js 24, Python 3 with `venv`, and FFmpeg/FFprobe with H.264
support.

```bash
npm install
npm run analysis:setup
cp .env.example .env.local
npm run dev -- --hostname 0.0.0.0
```

Set `VOLLEYCUT_DATA_ROOT` in the ignored `.env.local`. Configure
`VOLLEYCUT_LABELING_WORKSPACE`, `VOLLEYCUT_INTAKE_WORKSPACE`, or
`VOLLEYCUT_MODEL_FEEDBACK_ROOT` when those stores are outside the default data root.
Large source media, labels, caches, and generated training artifacts remain outside
Git.

The labeling boundary policy, ignored intervals, hard negatives, court geometry,
anonymous player tracks, sparse cues, and save workflow are documented in
[`docs/labeling-guide.md`](docs/labeling-guide.md).

## Production model contract

The deployed system samples the selected game window at 4 Hz, generates 104 base
audiovisual features, gathers five temporal contexts into 520 inputs, and runs two
independent rally/serve/dead-state bundles. Overlapping intervals are unioned, while
one-model-only regions remain visible for review. Separate serving-side and team-side
models seed editable score markers where enabled.

Maintained contracts:

- [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) — inference heads, decoders,
  ensemble behavior, and suppression policy;
- [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md) — feature formulas, profiles, and
  rebuild/parity requirements;
- [`MODELS.md`](MODELS.md) — trained artifacts, lineage, comparisons, and disposition;
- [`docs/model-ranking-metric.md`](docs/model-ranking-metric.md) — canonical
  `F1_padP_coreR` selection and reporting contract.

The Python feature, training, inference, and evaluation commands are documented in
[`analysis/README.md`](analysis/README.md). Retained experiment decisions live under
[`docs/research/`](docs/research/).

## Verification

```bash
npm test
npm run lint
npm run build
```

Verify the production browser independently:

```bash
cd prod
npm ci
npm run build
```

Android and iOS build prerequisites and tests are maintained in their respective
README files. Release signing remains separate from normal automated builds.

## Repository layout

- `prod/` — published static browser client and runtime assets;
- `android/` — published native Android client;
- `ios/` — published native iOS/iPadOS client;
- `app/`, `components/`, `lib/` — V2 labeling and model-feedback import;
- `analysis/`, `scripts/` — feature extraction, training, evaluation, and reporting;
- `tests/` — web, model, and cross-runtime tests;
- `docs/` — maintained contracts, guides, and research records;
- `public/on-device/` and `android/app/src/main/assets/` — runtime model assets;
- `data/` — small checked-in reports and fixtures;
- external `$VOLLEYCUT_*` roots — private media, labels, caches, and feedback bundles.

## License

VolleySplice project code and project-owned assets are available under the
[MIT License](LICENSE). Third-party components and brand assets retain their original
terms; see [third-party notices](THIRD_PARTY_NOTICES.md) and
[asset provenance](ASSET_PROVENANCE.md).
