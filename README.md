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
strongest base appearance specialist. The former decoder winner is the V5-state-based
local-peak+soft-count cleanup described below; the current research winner is the
full-union hard-negative model. The 61-gap V5/V5-state/V6 proposal union is
available as three separate recording timelines in `/side-switch-review` because the
earlier heuristic marker seeds are not exhaustive. That review page also exposes
explicit/candidate/full-video human marker rails, a per-recording continuous-review
flag, production model range labels, and the corrected production-editor/final-export
timelines. New full-video markers are stored separately from the frozen candidate
decision artifact.
The completed 50-event continuous review now provides exhaustive raw-phone truth: only
33/50 switches are covered by any modern candidate gap with four-second boundary
allowance. No-cadence V5-state led the original comparable macro-recall ranking at
56.36%, while local peak plus soft count led pooled F1 at 44.64%. A later full-union
hard-negative model is now the user-designated research winner: it produces 52
proposals with 29 TP/23 FP/21 FN and 56.86% pooled F1. It is not promoted to automatic
inference. Its exact identity, per-video results, and source bindings are frozen in
[`data/side-switch-current-research-winner-v1.json`](data/side-switch-current-research-winner-v1.json).
See the
[`winner promotion`](docs/research/side-switch-hard-negative-winner-promotion-2026-08-23.md).
The first rare-event follow-up preserves one V5 orientation observation per detected
rally. Manual parity proves stable rally states can bracket all 50 switches, but the
existing orientation sign recognizes only 21.32% of swapped-state observations and its
persistence-2 decoder finds 5/50 events. The parity architecture is retained for a
better emission; this diagnostic is not a model promotion. See the
[`parity feasibility decision`](docs/research/side-switch-parity-feasibility-2026-08-23.md).
The next follow-up uses extreme same-side continuity only as a conservative veto. In
leave-one-recording-out evaluation it removes five false proposals without removing a
true proposal, improving pooled F1 from 44.64% to 46.73%. A hard same-versus-swapped
gate and add-only candidate use both fail, so the frozen verifier remains a research
layer for the next union experiment rather than a new current winner. See the
[`continuity verifier decision`](docs/research/side-switch-continuity-verifier-2026-08-23.md).
The full-trace candidate follow-up then expands the internal universe from 352 V5 gaps
to all 624 production boundaries plus 80 strong dead-state peaks inside overlong rally
ranges. Candidate recall rises from 66% to 92% with the same configuration selected in
every held-out fold and no new decode/inference. Only 352/704 windows currently have
valid frozen V5 features in that experiment. See the
[`candidate-union decision`](docs/research/side-switch-candidate-union-2026-08-23.md).
The next loop extracts V5+production-state features for every union candidate and
reproduces all 42 legacy values exactly. A class-balanced nested-LOO ranker improves the
current winner's ±4-second end-to-end result from 25 TP/37 FP/25 FN and 44.64% F1 to
27 TP/29 FP/23 FN and 50.94% F1 at 56 proposals. Strict F1 does not improve, internal
peaks remain weakly ranked, and an expanded-distribution continuity veto regresses, so
the new head is retained as an on-device-compatible research contender without
changing the current-winner pointer or production inference. See the
[`full-union ranker decision`](docs/research/side-switch-full-union-ranker-2026-08-23.md).
An imbalance/calibration loop then holds the full-union architecture fixed. Nested
selection of natural, square-root-balanced, fully balanced, robust-logit, and percentile
variants removes two false positives at unchanged recall, moving F1 from 50.94% to
51.92%. Square-root weighting is retained as a candidate training objective; recording
calibration is rejected. Training an otherwise identical no-switch head produces only
the numerical complement of the switch score, so future negative modeling must use a
distinct continuity or hard-negative target. See the
[`imbalance/calibration decision`](docs/research/side-switch-imbalance-calibration-2026-08-23.md).
A candidate-type penalty loop then tests whether weak internal dead-state peaks should
receive a soft negative prior. Nested ±4 F1 improves from a matched 53.47% control to
54.90%, but the penalty is zero in 6/11 folds and removes the control's only correct
internal proposal. It is rejected as a general rule; the next loop should improve
internal-candidate representation or localization instead. See the
[`internal-peak penalty decision`](docs/research/side-switch-internal-peak-penalty-2026-08-23.md).
The next loop lowers the internal dead-state threshold and reaches 100% candidate
coverage with 852 windows. Ranking regresses: nested ±4 F1 falls from 50.94% to 42.74%,
and only one of 13 emitted internal proposals is correct. The 704-candidate union is
retained; future internal candidates need better evidence or a separate head rather
than a lower global threshold. See the
[`expanded internal-candidate decision`](docs/research/side-switch-expanded-internal-candidates-2026-08-23.md).
Recording-balanced hard-negative mining then improves the retained 704-candidate head
without adding inference work. The nested selector removes one FP at unchanged recall
(53.47%→54.00% F1), while the fixed 34-input top-2/2× candidate reaches 56.86% F1.
By explicit user decision, that fixed variant is promoted to the current research
winner, but it is not installed in production or either client. See the
[`hard-negative mining decision`](docs/research/side-switch-hard-negative-mining-2026-08-23.md).
A within-recording pairwise/AUC-surrogate follow-up keeps that winner's inputs, hard
mining, and decoder fixed. Its weakest weight slightly raises row AP but adds seven FP
without another TP (56.86%→53.21% F1); nested selection reaches 54.72%. The pairwise
loss is rejected and the research-winner pointer remains unchanged. See the
[`pairwise ranking decision`](docs/research/side-switch-pairwise-ranking-2026-08-23.md).
A recording-reliability ridge head then predicts per-video threshold offsets from
quality and score summaries. Nested selection rejects every nonzero offset; the best
fixed head gains one TP but adds six FP (56.86%→55.05% F1). Direct blur is unavailable
in the retained artifact, and the winner remains unchanged. See the
[`recording-reliability decision`](docs/research/side-switch-recording-reliability-2026-08-23.md).
Matched focal and effective-number objectives also fail. Focal marginally improves row
AP but lowers event F1 to 53.85%; the best effective-number variant reaches 54.90%, and
nested objective selection reaches 53.47%. Square-root BCE remains the winner. See the
[`rare-event loss decision`](docs/research/side-switch-rare-event-losses-2026-08-23.md).
The production-state follow-up reuses the on-device ensemble as soft V5 evidence and
raises retrospective exact F1 to 30.14% while reducing proposals 44→38, but count
accuracy and label completeness prevent promotion. Serve-grounded V6 regresses, hard
gates fail, and suppression is excluded for target/data overlap.
The controlled no-cadence V5-state refit confirms that the re-anchored seven-point path
can cascade errors: exact recall rises 31.43%→80.00% and F1 30.14%→44.44%, retaining
all cadence true positives, but proposals rise 38→91. It is tracked as a research-only
decoder diagnostic; the next iteration needs appearance-local cluster suppression or a
soft count prior before any on-device promotion.
The follow-up confirmed adjacent-gap peak suppression as the stable cleanup: its locked
raw-phone ablation raises exact F1 to 48.15%. Peak suppression plus a soft post-six
penalty reaches 49.48% exact/57.73% ±2 candidate-conditioned F1 at 62 proposals and was
the research winner after the exhaustive audit before being superseded by the
full-union hard-negative model. A separate production rally/dead/serve context head wins
historical validation but fails to improve retrospective exact F1, so production
outputs remain soft features rather than eligibility gates.

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
