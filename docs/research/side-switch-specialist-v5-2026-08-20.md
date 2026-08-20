# Side-switch specialist v5 — 2026-08-20

## Decision

V5 validates motion-component player isolation as the strongest visual improvement so
far, but it is **not promoted** to browser or Android inference. On the retrospective
raw-phone evaluation, exact-gap F1 improves from v4's 16.67% to 27.85%, raw row AP from
26.92% to 43.40%, and exact precision from 16.22% to 25.00%. Absolute precision remains
too low for automatic score tracking.

Persistent whole-set orientation parity was implemented and tested, but validation
selected orientation weight `0.0`. It is therefore a controlled negative result and has
no effect on the selected v5 predictions. The retained gain comes from player-isolated
visual features.

## Frozen data and scope

V5 inherits v4's immutable reviewed labels, 6/4/11 recording split, court geometry, and
seven-frame sampling. It preserves seven-point cadence, margins ±1 through ±4,
re-anchoring, the six-opportunity cap, and one-set/start-zero assumptions. The blurry
`beach-source-02` recording remains excluded from fitting and selection.

| Role | Recordings | Reviewed gaps | Switch gaps |
| --- | ---: | ---: | ---: |
| Train | 6 | 234 | 29 |
| Validation | 4 | 143 | 17 |
| Retrospective evaluation | 11 | 352 | 35 |

Unlike v4, V5 extracts all 1,024 rallies across the 21 recordings so orientation context
spans the whole set rather than only reviewed gaps. All 729 reviewed rows contain 22
finite model features and finite orientation context; extraction had zero frame errors.

## `PLAYER-ORIENTATION22`

For each court-normalized frame, V5 thresholds temporal deviation at the larger of 12
luma levels or the frame's 94th percentile. It closes small gaps, filters components by
area, aspect ratio, dimensions, and lower-court position, expands the retained boxes,
applies overlap suppression, and keeps at most six player-like proposals. This is a
detector-free motion heuristic, not a verified person detector.

Proposal palettes are weighted by motion, temporal deviation, and a small saturation
term. Proposal foot position softly assigns each palette to near or far around normalized
`y=0.63`. Per-rally summaries retain near, far, and global player palettes plus support,
coverage, count, and instability measures.

The 22 ordered classifier inputs contain six retained v4 quality/change features and 16
player-proposal assignment, separation, instability, appearance-change, coverage, count,
and side-support features. The linear model selected L2 1.0 by recording-held-out
training AP.

### Whole-set orientation state

Because each recording begins at score zero, the first three rallies pool initial near
and far team palettes before a cadence switch is possible. Every later rally receives a
same-versus-swapped coordinate relative to those anchors. The decoder carries parity:
each selected switch flips the expected direction for all later opportunities.

Validation searched orientation weights `0`, `0.25`, `0.5`, `1`, `2`, and `4` jointly
with threshold, margin, and distance penalty. All weights produced the same selected
validation events, so the deterministic tie-break retained weight zero. On raw-phone
evaluation, weights through 2 also leave predictions unchanged; weight 4 removes one
prediction without increasing exact true positives. Persistent orientation is retained
as research code but is not part of the selected effective model.

The selected decoder uses threshold `0.3607203775661717`, margin ±1 rally, distance
penalty 0.25, orientation weight 0, re-anchoring, and the six-opportunity cap. Its stable
classifier/decoder fingerprint is
`c5bc3c86b9b2d35ea05a204929c2d496fdf85e01402c461a634a07c8b4441dff`.

## Results

### Development validation

| Metric | V4 | V5 |
| --- | ---: | ---: |
| Visual row AP | 34.69% | **53.46%** |
| Exact precision | 82.35% | **94.44%** |
| Exact recall | 82.35% | **100.00%** |
| Exact F1 | 82.35% | **97.14%** |
| Exact switch-count recordings | 2/4 | **3/4** |

### Retrospective raw-phone evaluation

| Scoring tolerance | Model | Predictions | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Exact | V4 | 37 | 16.22% | 17.14% | 16.67% |
| Exact | V5 selected | 44 | **25.00%** | **31.43%** | **27.85%** |
| ±1 rally | V4 | 37 | 27.03% | 28.57% | 27.78% |
| ±1 rally | V5 selected | 44 | **34.09%** | **42.86%** | **37.97%** |
| ±2 rallies | V4 | 37 | 40.54% | 42.86% | 41.67% |
| ±2 rallies | V5 selected | 44 | **40.91%** | **51.43%** | **45.57%** |

V5 predicts the exact switch count in 3 of 11 evaluation recordings, compared with 2 of
11 for v4. It exactly matches 11 of 35 labeled switches, 15 within ±1 rally, and 18
within ±2 rallies.

### Candidate-margin sensitivity

The model and threshold stay fixed; only candidate margin changes. These raw-phone
results are retrospective sensitivity, not selection evidence.

| Margin | Exact F1 | ±1-rally F1 | ±2-rally F1 |
| --- | ---: | ---: | ---: |
| ±1 selected | 27.85% | 37.97% | 45.57% |
| ±2 | 32.94% | 37.65% | 47.06% |
| ±3 | 32.56% | **39.53%** | **48.84%** |
| ±4 | **34.48%** | 39.08% | 48.28% |

The wider margins find additional events retrospectively but also increase predictions
from 44 to 50–52. They cannot replace the validation-selected ±1 margin.

## Pre-fit diagnostic correction

The first V5 extraction allowed ten proposals and used an unlabeled PCA orientation
axis. Proposal counts saturated near 9–10 per frame and the PCA coordinate often tracked
formation or lighting instead of team identity. It was preserved without model fitting:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v5-player-orientation-features-diagnostic-proposal10-pca.json`
(`dbd3d9510ed8cde919692915051dd6d1fb8d00cfe8f5103756f2478239975d67`).

The selected extraction reduced proposal counts to roughly five per frame and replaced
PCA with direct score-zero team anchors before its SHA was frozen and fitting began.

## Selected artifacts

- Feature artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v5-player-orientation-features.json`
  (`4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db`)
- Model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v5-player-orientation/model.json`
  (`5de46585bf5f7fc04a3f19838b448b2198c24faaf77fe6ed6a6a373ad84adb76`)
- Development dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v5-player-orientation/dataset-development.json`
  (`2761db13f9061cf9350e71c5ad4f6810c44f001795bcd5adc90022166e861e2c`)
- Evaluation:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v5-player-orientation-evaluation.json`
  (`9905ff138f6aba82a299fe7b1665e92a32c91ca24737928c2f034e2e9bbf92d8`)

The selected feature artifact was hashed and frozen before model fitting. A separate
provenance artifact binds the implementation revision, inherited full-file hashes for all
21 videos, v4 label/geometry identity, the rejected pre-fit diagnostic, and every
selected v5 artifact.

## Next dependency

Player isolation materially improves ranking, but 33 of 44 exact predictions remain
false positives and 24 of 35 switches remain exact misses. The next independent change
should address scored-point progression and re-dos rather than adding more visual model
capacity or choosing the retrospectively better wider margin.
