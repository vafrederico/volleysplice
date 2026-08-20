# Side-switch specialist v6 — 2026-08-20

## Decision

Do not promote v6. The requested person-localization and adaptive-team-prototype
changes are implemented, trained, and evaluated, but the frozen raw-phone result is
worse than v5 at the event level. V5 remains the best side-switch research artifact;
neither model is suitable for automatic scoring or included in browser/Android
production inference.

V6 does produce the best raw-phone row ranking so far: AP rises from v5's 43.40% to
45.47%. That gain does not survive the validation-selected threshold and cadence
decoder. V6 exact-gap F1 is 24.10% versus v5's 27.85%, and ±2-rally-tolerant F1 is
40.96% versus 45.57%.

## Frozen scope and leakage policy

V6 retains the v3–v5 contract:

- one recording is one game/set and every recording begins at score zero;
- grass and beach switch cadence is seven scored points;
- rally order remains a noisy point-count proxy because re-dos and marker mistakes can
  hold or shift the true score;
- the frozen split remains 6 train, 4 validation, and 11 retrospective raw-phone
  evaluation recordings;
- `beach-source-02` is excluded from fitting, preprocessing, threshold selection,
  decoder selection, and every feature row;
- L2 is selected only from recording-held-out training AP; threshold and decoder are
  selected only from validation exact-gap F1; raw-phone labels are opened once after
  both the full model and ablation are frozen.

The resulting artifact contains 729 reviewed gaps: 234 train with 29 positives, 143
validation with 17 positives, and 352 evaluation with 35 positives. Feature extraction
uses every one of the 1,024 rallies across the 21 included sets so team prototypes have
whole-set context, but the classifier still scores only the frozen reviewed gaps.

## Change 3: quantized player localization

V5's temporal-difference connected components are replaced by the pinned OpenCV Zoo
MediaPipe person detector:

- upstream revision `47534e27c9851bb1128ccc0102f1145e27f23f98`;
- block-quantized int8 ONNX, 3,482,053 bytes;
- Apache-2.0 license;
- 224×224 RGB input through the existing OpenCV DNN CPU runtime;
- four 62%-coverage overlapping ownership tiles per sampled frame;
- three frames at 15%, 50%, and 85% of every rally, for 12 detector invocations per
  rally;
- pose-detector hip/shoulder landmarks define a torso crop; implausible landmark
  geometry and duplicate hips are suppressed;
- at most two detections per canonical court side are retained, matching the 2v2
  beach/grass corpus;
- canonical hip height uses v4's per-recording net calibration, so camera distance and
  net position do not depend on one fixed pixel row;
- same-side hip matches across the three samples weight temporal consistency.

This is an actual general-person localizer, not a volleyball-player-trained detector.
Spectators and adjacent-court players inside a broad ROI remain a known failure mode.

### Localization audit

| Role | Sets / rallies | Selected players/frame | Mean confidence | Temporal consistency | Zero-player rallies | Detector inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 6 / 242 | 1.959 | 0.624 | 0.698 | 0 | 78.93 s |
| Validation | 4 / 147 | 2.193 | 0.567 | 0.641 | 0 | 46.03 s |
| Evaluation | 11 / 635 | 2.037 | 0.553 | 0.629 | 1 | 204.83 s |

These are detector-forward times measured on the research CPU, excluding video decode
and frame seeking. The 3.48 MB model size and fixed 12 calls/rally are compatible with
offline on-device execution, but no mobile runtime port was made because the accuracy
gate failed.

## Change 4: adaptive team prototypes

The first three score-zero rallies still initialize canonical near/far team palettes.
Unlike v5, those prototypes are then updated online throughout the set. Each rally is
assigned to the same or swapped team ordering by minimum palette cost. Updates are
confidence-gated by player support, detection confidence, cross-frame consistency,
side separation, and assignment margin; accepted updates use at most a 0.12 exponential
rate. No score or switch label enters an update.

The extractor also materializes a frozen-first-three prototype context. A separately
fitted 26-input ablation uses that context, while the full 29-input model adds adaptive
flip magnitude, flip agreement, and orientation quality. This makes the adaptive change
measurable without altering the split.

The gate is conservative: 26 updates were accepted across all 242 training rallies, 17
across 147 validation rallies, and 26 across 635 evaluation rallies. Mean final
prototype drift was 0.0193, 0.0137, and 0.0077 Hellinger distance by role. Adaptive
features improve row AP over the fixed-prototype ablation by 1.09 points on validation
and 0.80 points on evaluation, but both variants select exactly the same event gaps.

## Feature and model contract

`DETECTED-ADAPTIVE29` has 29 ordered scalar inputs:

- six retained v4 appearance/alignment values;
- detected-player same/swapped palette cost, swap/flip evidence, side separation,
  before/after instability, and global appearance change;
- detection coverage/count, confidence, near/far support, and temporal-consistency
  quality/change values;
- three adaptive whole-set orientation values.

The selected classifier is a class-balanced linear logistic ranker. L2 grid
`[0.01, 0.1, 1.0, 10.0]` is ranked by leave-one-recording-out training AP. Validation
then searches candidate margins ±1 through ±4, distance penalties
`[0, 0.1, 0.25, 0.5]`, orientation weights `[0, 0.25, 0.5, 1, 2, 4]`, and every
observed score threshold. The selected full model is:

- L2 `10.0`;
- threshold `0.43222614272563986`;
- candidate margin `±1`;
- distance penalty `0.25`;
- orientation weight `0.0`;
- maximum six opportunities with re-anchoring after selection.

The fixed-prototype ablation also selects L2 `10.0`, margin `±1`, distance `0.25`, and
orientation weight `0.0`, with threshold `0.43328154932136753`.

## Results

### Development selection

| Candidate | Row AP | Exact P | Exact R | Exact F1 | ±1 F1 | ±2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V5 selected | 53.46% | 94.44% | 100.00% | 97.14% | 97.14% | 97.14% |
| V6 fixed-prototype ablation | 57.41% | 84.21% | 94.12% | 88.89% | 94.44% | 94.44% |
| V6 detected + adaptive selected | 58.51% | 84.21% | 94.12% | 88.89% | 94.44% | 94.44% |

### Retrospective raw-phone confirmation

| Candidate | Row AP | Predictions | Exact P | Exact R | Exact F1 | ±1 F1 | ±2 F1 | Exact-count sets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 | 26.92% | 37 | 16.22% | 17.14% | 16.67% | 27.78% | 41.67% | not recorded here |
| V5 selected | 43.40% | 44 | 25.00% | 31.43% | 27.85% | 37.97% | 45.57% | 3/11 |
| V6 fixed-prototype ablation | 44.67% | 48 | 20.83% | 28.57% | 24.10% | 33.73% | 40.96% | 4/11 |
| V6 detected + adaptive selected | 45.47% | 48 | 20.83% | 28.57% | 24.10% | 33.73% | 40.96% | 4/11 |

The full v6 model has 10 exact true positives, 38 false positives, and 25 false
negatives. At ±2 tolerance it has 17 matches, 31 false positives, and 18 false
negatives. Better row AP with worse decoded F1 indicates that the validation-selected
threshold/count policy does not transfer to the raw-phone source group.

### Candidate-margin sensitivity

This is confirmation-only sensitivity; it does not change the validation-selected ±1
margin.

| Candidate margin | Exact F1 | ±1-tolerant F1 | ±2-tolerant F1 |
| ---: | ---: | ---: | ---: |
| ±1 selected | 24.10% | 33.73% | 40.96% |
| ±2 | 21.69% | 33.73% | 40.96% |
| ±3 | 21.69% | 33.73% | 40.96% |
| ±4 | 21.69% | 33.73% | 40.96% |

### Orientation-weight sensitivity

Validation selected weight zero. The raw-phone values below are retrospective
diagnostics and must not be used to revise the model.

| Weight | Predictions | Exact F1 | ±2-tolerant F1 |
| ---: | ---: | ---: | ---: |
| 0 selected | 48 | 24.10% | 40.96% |
| 0.25 | 48 | 24.10% | 40.96% |
| 0.5 | 45 | 27.50% | 45.00% |
| 1.0 | 38 | 24.66% | 46.58% |
| 2.0 | 33 | 20.59% | 38.24% |
| 4.0 | 30 | 18.46% | 36.92% |

The post-hoc gains at weights 0.5–1.0 are not promotable evidence. They do show that
adaptive orientation contains some useful count/abstention signal, but the four-set
validation scope did not select it reliably.

## Interpretation

The compact detector solves V5's semantic problem—boxes now come from a person model
rather than motion blobs—and it improves row ranking. It does not solve the dominant
system problem: rally index is not scored-point state. The selected decoder still
over-predicts opportunities in sets with re-dos, missing markers, or fewer labeled
switches. Adaptive prototypes are also too weak and too conservatively updated to
stabilize the final event path across source groups.

The next high-value experiment remains a point-advance/redo state model or oracle. It
should determine whether a rally advanced the score before applying the seven-point
cadence, then use V5/V6 visual evidence only to place or abstain near the predicted
crossing.

## Immutable artifacts

- Person detector model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar/model.onnx`,
  SHA-256 `c5ed8c00c028b98e5d2c55b920a6e975af6c4cd538cfeea7c054f4fbbd8b9075`.
- Detector metadata: same directory `model.json`, SHA-256
  `a2a2c39f9a76571b0e5a9de95d2b7162a935f2f82e1dbdba8eacfb5b4733b0df`.
- Detector license: same directory `LICENSE`, SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
- Feature artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v6-detected-adaptive-features.json`,
  SHA-256 `364d9f050b4cc581059e6ad944737a9bfa2c43bdf845fae6bd57394247abeb4a`.
- Model: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v6-detected-adaptive/model.json`,
  SHA-256 `adce9bc3a8a4a84e97cb22fc34a9939c1b8ff9b0fa8da097ae8f0ec3e6d25d07`.
- Development dataset: same model directory `dataset-development.json`, SHA-256
  `9d203ec41fd1027c674107e4f7d052642bb998d92e25f535fc2d07d18a1b122e`.
- Evaluation:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v6-detected-adaptive-evaluation.json`,
  SHA-256 `7159051bffe430f673148e12b65a119d728f6abcdadf1b7959f84588dd50aa58`.
- Selected deployable fingerprint:
  `3abf18c01bff4c68bcbc805d45d302407a7495ad6dfaefb0a78ba856b56a41d4`.
- Fixed-prototype diagnostic fingerprint:
  `df1f9498c21c27b10602baa0632a967fde59bde54afa8d5a2dc0fcf2653e0e33`.
- Implementation revision: `IMPLEMENTATION_REVISION_PENDING`.
- Provenance:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v6-provenance.json`,
  SHA-256 `PROVENANCE_SHA256_PENDING`.
