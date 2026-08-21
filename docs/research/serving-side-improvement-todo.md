# Serving-side model improvement TODO

Last updated: 2026-08-21

This file is the durable resume point for the serving-side roadmap. Update the
checkboxes, artifact paths, metrics, and exact next command whenever work advances.
Do not rely on conversation history as the only record.

## Frozen current state

- Last committed research checkpoint: `279a49c` on
  `t3code/serving-review-shortcuts`.
- Final human correction overlay: 20 corrections (4 near, 8 far, 8 not-serve),
  SHA-256 `82fd8a7d46472cccb88561ca5eb03ef6fb54cf97eed89e9efa31f416636b5253`.
- Completed review sample: 61 current errors plus 119 valid stratified controls.
- Weighted review report:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json`.
- Correction-clean fixed-anchor flight baseline:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-development.json`.
- Baseline primary metric: 94.12% source-group macro balanced accuracy.
- Baseline pooled balanced accuracy: 94.05%; worst-group balanced accuracy: 85.83%.
- Key weighted slices: visible 96.41%, partial 94.15%, offscreen 55.34%,
  on-anchor 97.08%, before-anchor 81.31%, after-anchor 76.62% accuracy.
- Offscreen reviewed examples contain only human-near labels. Do not claim an
  offscreen far estimate until that support exists.
- The unconditional -1s/0s/+1s temporal concatenation experiment was rejected:
  it improved pooled BA to 94.35% but reduced source-group macro BA to 93.82%
  and worst-group BA to 81.42%.
- The protected test split has already been opened historically. Do not use it for
  any further feature, threshold, routing, or model selection.

## Remaining roadmap

- [x] **Explicit residual-component trajectory model**
  - Link small residual-motion components across consecutive frame pairs.
  - Measure direction/displacement, expansion or contraction, path straightness,
    velocity, acceleration, link error, motion onset, persistence, and track confidence.
  - Combine trajectory recording-rank features with the current v2 and fixed-flight
    features.
  - Select only with leave-one-source-group-out development predictions.
- [x] **Selective high-resolution inspection**
  - Keep the correction-selected 192x108 full-frame 4x6 representation as baseline.
  - Use trajectory candidates to request small higher-resolution source regions.
  - Compare crop sizes and resolutions without running full-frame 640x360 everywhere.
  - [x] Implement and unit-test source-resolution patches around the strongest
    persistent and compact/fast low-resolution tracks.
  - [x] Smoke-test one rally from all 28 development recordings: 52 finite features
    per row, no human visibility input, and no protected data.
  - [x] Freeze and evaluate the first 20%-of-short-edge / 96x96 patch candidate.
  - [x] Run the predeclared patch-size ablation. The effect was size-sensitive, so
    regional high-resolution features are not production-promoted.
- [x] **Visibility-aware mixture**
  - Derive an inference-time visibility/contact-quality signal; never use human
    visibility directly as an inference input.
  - Train and cross-fit visible and flight/offscreen specialists without source-group
    leakage.
  - Acquire offscreen-far labels before trusting a two-sided offscreen specialist.
  - [x] Train and group-cross-fit inference-time visibility and contact-quality
    classifiers on the weighted reviewed sample.
  - [x] Run a nested side-mixture evaluation in which the held source group's human
    quality labels are excluded from both gating and side-specialist training.
- [x] **Slice guardrails and blending**
  - Keep source-group macro BA as the primary ranking metric.
  - Report pooled BA, worst-group BA, every environment/source group, and weighted
    visibility/contact slices for every selected comparison.
  - Test group-balanced fitting or a conservative baseline/new-model blend only on
    development data.
- [ ] **Confidence-based review routing**
  - [x] Refit calibration/abstention for the selected current model; do not reuse the
    stale base-v2 operating point.
  - Add an uncertain-case queue to the existing results/review UI only after the
    operating point is frozen on development data.

Confidence-routing predeclaration (recorded before evaluation):

- Calibrate the correction-clean fixed-flight cross-fit probabilities only; do not
  change or reselect the side classifier.
- Compare identity calibration with nested leave-one-source-group-out Platt
  calibration at L2 0.01, 0.1, 1, and 10. Select by lower pooled Brier score, then
  lower log loss, lower worst-source-group Brier score, and the simpler candidate.
- Fit the deployable calibrator on all development cross-fit predictions only.
- Select raw-score abstention bands at 94%, 95%, 96%, 97%, and 98% pooled precision
  for both emitted sides, with at least 50 emitted rows per side. Existing near/far
  decisions may only become abstentions; they may never flip sides.
- Use the predeclared 95% target for the initial UI uncertainty queue. Report its
  coverage, review fraction, all-row per-side recall, selective accuracy, and every
  source-group slice. Do not change this choice after seeing protected-test results.

## Required evaluation contract

1. Apply the current correction overlay and source-quality exclusions.
2. Exclude all protected source groups from development fitting and candidate choice.
3. Rank by mean source-group balanced accuracy, then pooled balanced accuracy,
   macro-F1, worst-source-group balanced accuracy, and fewer features.
4. Rescore the frozen two-phase review sample after selection for interpretable
   visibility and contact-timing estimates; those slices are not the primary selector.
5. Version NAS outputs and refuse overwrite. Record source and implementation hashes.
6. A higher pooled score alone is insufficient if the primary or weak-source
   guardrails regress.

## Current execution checkpoint

- [x] Implement `analysis/serving_side_trajectory.py` and synthetic unit tests.
- [x] Add and smoke-test a correction-clean development extractor at 192x108 using
  the existing nine fixed-anchor frames. The smoke covered one rally from each of
  the 28 development recordings and produced all 43 finite features per row.
- [x] Add a development evaluator that compares trajectory-only, fixed-flight,
  v2-plus-trajectory, and v2-plus-fixed-flight-plus-trajectory families.
- [x] Freeze the selected and best-trajectory candidates plus weighted review slices.

Trajectory result (development only):

- Feature artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-trajectory-v1/development.json`,
  SHA-256 `dde02b10cf70ccb2f6fd99c0d23d44232eec881d243a81983fa9a5f763239d13`.
- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-trajectory-v1-development.json`,
  SHA-256 `e6b256d2a7551389aa3e5247108326adf29d370effe464223946d753d5a29135`.
- Internal fixed-flight baseline exactly reproduced v3: 94.1186% source-group macro
  BA, 94.0483% pooled BA, and 85.8289% worst-group BA.
- Best trajectory candidate was v2 + fixed-flight + 43 trajectory recording-rank
  features at L2 0.1. It was rejected: 93.9905% macro BA, 93.9398% pooled BA, and
  84.3583% worst-group BA.
- The trajectory candidate did improve the weighted reviewed visible slice from
  96.41% to 96.79% accuracy and on-anchor from 97.08% to 97.55%, but did not change
  offscreen accuracy and regressed the before-anchor slice. These post-selection
  slices do not override the primary development guardrails.

Selective high-resolution result (development only):

- Feature artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-selective-highres-v1/development.json`,
  SHA-256 `4d2f67adf4478bea8b9791b2448c15603636dd86371d84511a73d14332474141`.
- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-selective-highres-v1-development.json`,
  SHA-256 `422eb37b6dd35d8ab7eb86a6912c47e4ed646307b8d8f3452ebf044e4e3a31dd`.
- Development-selected candidate: v2 + fixed-flight ranks + 52 absolute regional
  high-resolution features at L2 0.1.
- Candidate metrics: 94.2328% source-group macro BA, 94.5365% pooled BA, 87.2995%
  worst-group BA, and 94.5472% accuracy. The fixed-flight baseline reproduced
  94.1186%, 94.0483%, 85.8289%, and 94.0604%, respectively.
- Paired result: 11 candidate-only corrections, 6 baseline-only regressions, 960 both
  correct, and 50 both wrong. This is a five-error net development improvement.
- Weighted review slices also improved: visible 97.05%, partial 94.73%, offscreen
  near-only 63.11%, on-anchor 97.78%, before-anchor 84.82%, and after-anchor 77.79%
  accuracy. Offscreen still has no human-far support.
- Treat this as provisional: the source-group macro gain is only 0.114 percentage
  points and one regional configuration was tried. It is not production-promoted.

Patch-size ablation predeclaration and result:

- Configurations: 12%, 20%, and 30% of the source short edge, each normalized to
  96x96, with identical persistent/compact track selection and 52 descriptors.
- Fixed side model: v2 recording ranks + 192x108-r4c6 flight ranks + absolute patch
  descriptors, L2 0.1. Only source crop coverage changes.
- Robustness rule: every size must improve source-group macro BA and must not regress
  worst-source-group BA relative to the fixed-flight baseline.
- The v2 20% features must exactly reproduce every v1 20% row before evaluation.
- Feature artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-selective-highres-v2/development.json`,
  SHA-256 `163dc85f832e35abcb6d95abd5daf3ffd5a84290366453306bdfb4951f8b14e7`.
- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-selective-highres-v2-ablation-development.json`,
  SHA-256 `6cf819a69288e802be128342e5510aaa9c2d0fabf64df0be460dab36ff2df9ff`.
- Exact parity check passed: every 20% v2 feature equals its v1 feature.
- 12%: 94.3503% macro BA, 94.8274% pooled BA, 87.2995% worst-group BA,
  94.8393% accuracy; 11 baseline errors fixed and 3 regressions introduced.
- 20%: 94.2328% macro BA, 94.5365% pooled BA, 87.2995% worst-group BA,
  94.5472% accuracy; 11 baseline errors fixed and 6 regressions introduced.
- 30%: 94.0467% macro BA, 94.3565% pooled BA, 85.8289% worst-group BA,
  94.3525% accuracy; 9 baseline errors fixed and 6 regressions introduced.
- Robustness rule failed because the 30% crop reduced macro BA by 0.0719 percentage
  points. Do not production-promote regional detail. Preserve 12% as the best research
  candidate and use the established 20% reference for the next quality-signal study
  to avoid post-hoc cherry-picking.

Quality-signal result (development reviewed sample):

- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-quality-v1-development.json`,
  SHA-256 `457698f1da4e7429eb16be691c579077559533dfa7bbf02fe670a8b4dcf48c82`.
- Visibility target: 119 observed visible vs 58 partial/offscreen reviews (3 unclear
  excluded), representing weighted populations 779.60 vs 222.40. Selected 52 absolute
  20%-patch features at L2 1.0: 71.4056% supported-group macro BA, 56.6480% pooled
  BA, 42.6730% worst-supported-group BA, and weighted Brier 0.22369.
- Contact target: 123 observed on-anchor vs 57 early/late reviews, representing
  weighted populations 855.83 vs 171.17. Selected 82 v2 ranks at L2 0.1: 69.1101%
  macro BA, 67.2159% pooled BA, 33.1081% worst-group BA, Brier 0.20346.
- These signals are too weak and uneven to trust as standalone decisions. Continue
  only with soft/nested gating and require side-model guardrails to improve.
- Offscreen-far human support remains zero. No two-sided offscreen specialist may be
  claimed or production-promoted.

Exact next action: implement and evaluate a nested visibility mixture. For every held
source group, exclude that group's quality annotations from its gate and from all
inner gate predictions used to train side specialists. Compare the fixed baseline,
quality probabilities appended as features, and conservative 25/50/75/100% blends
of a v2 visible specialist with a flight-motion degraded-visibility specialist.

Nested quality-mixture result (development only):

- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-quality-mixture-v1-development.json`,
  SHA-256 `a8482616299f1ffbcf8f4daeebf38569df3be79f7a64332ee5ba75d05158cef6`.
- All gate predictions used by an outer fold excluded that outer source group's
  annotations. Inner training-group predictions excluded both the outer group and
  their own group. The artifact records every exclusion/fold audit.
- Best candidate appended only the nested visibility probability to the fixed
  baseline: 94.1325% macro BA, 94.1395% pooled BA, 85.8289% worst-group BA,
  and 94.1577% accuracy versus baseline 94.1186%, 94.0483%, 85.8289%, and
  94.0604%. It fixed 3 baseline errors and introduced 2 regressions.
- Visibility-specialist blends at 25%, 50%, 75%, and 100% regressed macro BA to
  93.9084%, 93.4099%, 93.0368%, and 92.3403%. Reject the specialist mixture.
- Contact-only and combined appended-quality candidates tied the visibility append
  at the thresholded prediction level. The gain is too small and the gates too weak
  for production promotion. Keep the fixed-flight baseline as the operational model.
- Weighted slices: visible improved from 96.41% to 96.79%; offscreen did not change;
  before-anchor did not change; after-anchor improved from 76.62% to 77.79%.

Exact next action: commit the rejected mixture iteration, then execute slice guardrails:
compare ordinary class-balanced fitting with source-group-balanced fitting and
predeclared conservative probability blends. Keep the frozen fixed-flight baseline
unless macro and worst-group guardrails improve meaningfully.

Source-group-balanced fitting result (development only):

- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-group-balance-v1-development.json`,
  SHA-256 `3806af957e1b62462f6142a89a07d869a243802f8ef80c0e04f073290c0c5bf0`.
- Tested equal total weight per training source group at L2 0.01, 0.1, 1, and 10,
  plus fixed-baseline blends of 25%, 50%, and 75% for each weighted model (100% is
  the pure weighted candidate).
- The exact fixed-flight baseline remained selected at 94.1186% macro BA, 94.0483%
  pooled BA, and 85.8289% worst-group BA.
- Best weighted blend was L2 0.1 / 75% at 93.8639% macro BA and 93.5528% pooled BA.
  Best pure weighted candidate was L2 0.1 at 93.3845% macro BA, 93.1879% pooled BA,
  and 80.7487% worst-group BA. Reject group balancing and all blends.

Exact next action: commit the group-balance rejection, recalibrate the current
fixed-flight cross-fit probabilities, predeclare useful review-coverage targets, and
freeze confidence/abstention operating points before changing the review UI.

Fixed-flight calibration and abstention result (development only):

- Evaluation artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-calibration-abstention-development.json`,
  SHA-256 `68e6429e104f1cd76464eb2c786ceebc88406301b6743d6e1236d26227393c77`.
- Identity calibration won with Brier 0.05438, log loss 0.20347, and 10-bin ECE
  0.03863. Every nested Platt candidate had worse pooled Brier and log loss, so raw
  fixed-flight probabilities remain the calibrated deployment representation.
- The predeclared 95% policy freezes raw thresholds far `< 0.3121748737`, review
  `[0.3121748737, 0.5028396704)`, and near `>= 0.5028396704` around the existing
  0.4783744762 decision cutoff. It never changes a near decision to far or vice versa.
- It sends 40 of 1,027 rows (3.895%) to review, retains 96.105% coverage, and reaches
  95.137% selective accuracy. Emitted near precision is 95.132%; emitted far
  precision is 95.142%. All-row near recall is 92.505% and far recall is 90.385%.
- Sensitivity: the 94%, 96%, 97%, and 98% precision targets review 1.558%, 11.295%,
  17.624%, and 43.817% of development rows, respectively. The 95% target remains
  frozen for the initial UI queue.

Exact next action: bind the UI queue only to a matching model fingerprint. The current
all-video UI artifact uses the older court-flow v4 model, while this policy is bound to
the selected fixed-flight fingerprint `85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06`.
Generate matching all-video fixed-flight inference or a separately audited v4 policy;
never silently apply these thresholds to the mismatched v4 probabilities.

Matching fixed-flight all-video inference result:

- Development/unprotected feature bank: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-flight-v4/all-reviewed-inference.json`,
  SHA-256 `8bdf003a2fbb1295609e6051e14767614c4915006fcb2cfcac764d25bfdcb895`.
  It contains 1,075 candidates across 29 recordings, retains all 8 current not-serve
  corrections for UI inspection, and excludes 15 source-quality-affected candidates.
- Post-selection protected feature bank: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-flight-v4/protected-test.json`,
  SHA-256 `58fdb6b0aa7c57fc77ec4393334daf1637f5175b8d11b86e5b6c7e417308ce2e`.
  It contains 39 candidates from the one protected recording. It was extracted only
  after calibration and the 95% review policy were frozen and was never used to revise
  either. The first extraction with ambiguous aggregate count metadata is preserved as
  `protected-test-metadata-v1.json` and was superseded before inference.
- All-video inference artifact: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-dual-serve-gate-all-video-inference-v1.json`,
  SHA-256 `47dea63a37bc66253d92296ca3255538360e5a61d9b0a8b6b822489075cb1401`.
  It covers the same 1,114 candidates and 30 recordings as the prior UI artifact,
  reuses the unchanged dual production serve-head evidence, and binds the review band
  to the matching fixed-flight fingerprint. It marks 24 full-model predictions for
  review.
- Mixed-scope side diagnostics are 96.140% accuracy and 96.138% balanced accuracy
  (near precision 96.364%, near recall 95.841%, far precision 95.922%, far recall
  96.435%). These combine in-sample development and post-selection protected results;
  they are descriptive only and must not replace cross-fit development ranking.
- An intermediate all-video inference with the superseded protected metadata is saved
  as `serving-side-flight-v3-dual-serve-gate-all-video-inference-metadata-v1.json`.

Exact next action: switch the serving-side results loader to the matching all-video
artifact, expose its frozen review policy in the typed server payload, add an
`uncertain` outcome queue and confidence-band explanation, then exercise correction
persistence and URL navigation in browser/tests.
