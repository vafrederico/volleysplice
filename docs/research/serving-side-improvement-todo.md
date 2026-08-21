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
- [ ] **Visibility-aware mixture**
  - Derive an inference-time visibility/contact-quality signal; never use human
    visibility directly as an inference input.
  - Train and cross-fit visible and flight/offscreen specialists without source-group
    leakage.
  - Acquire offscreen-far labels before trusting a two-sided offscreen specialist.
- [ ] **Slice guardrails and blending**
  - Keep source-group macro BA as the primary ranking metric.
  - Report pooled BA, worst-group BA, every environment/source group, and weighted
    visibility/contact slices for every selected comparison.
  - Test group-balanced fitting or a conservative baseline/new-model blend only on
    development data.
- [ ] **Confidence-based review routing**
  - Refit calibration/abstention for the selected current model; do not reuse the
    stale base-v2 operating point.
  - Add an uncertain-case queue to the existing results/review UI only after the
    operating point is frozen on development data.

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

Exact next action: commit the completed crop ablation and weighted-quality tooling,
run `npm run evaluate:serving-side-quality-signals`, record both quality classifiers,
then execute the nested visibility/contact mixture without waiting for user input.
