# Serving-side specialist v2 — 2026-08-20

## Decision

Keep serving-side specialist v2 as the new research candidate. It replaces the
saturated HOG evidence with camera-motion-cancelled optical flow and connected
motion-component occupancy, preserves pre-serve/contact/post-serve structure, and
normalizes features within each recording. On the protected indoor test recording it
correctly classified 37 of 39 reviewed serves.

This is still a candidate-conditioned serving-side classifier. It does not measure
serve-anchor or rally recall and is not yet wired into production inference.

## Feature pipeline

The immutable feature signature is `serving-side-court-flow-v2`:

- eight frames at `[-1.25, -0.75, -0.35, -0.10, 0.10, 0.30, 0.55, 0.85]`
  seconds relative to the reviewed serve anchor;
- 192×108 ROI-normalized frames;
- explicit pre-serve, contact, and post-serve phase pairs;
- dense Farneback optical flow with median global flow removed;
- near/far service-zone flow mean, P90, active fraction, connected-component
  occupancy/count/centroid, and flow direction;
- phase deltas and near-minus-far contrasts; and
- tied within-recording percentile ranks for camera/crop normalization.

The extractor supports normalized `recording.courtGeometry.serviceZoneAnchors`.
None of the 30 source records currently contains those anchors, so every v2 row uses
the explicitly recorded `roi-relative-end-bands` fallback: the top and bottom 32% of
the recording ROI. This is camera-relative normalization, not metric court
calibration. Adding reviewed service-zone anchors remains the next geometry upgrade.

Implementation:

- [`analysis/serving_side_v2.py`](../../analysis/serving_side_v2.py)
- [`scripts/extract-serving-side-v2-features.py`](../../scripts/extract-serving-side-v2-features.py)
- [`scripts/train-serving-side-v2.py`](../../scripts/train-serving-side-v2.py)
- [`scripts/evaluate-serving-side-v2-protected.py`](../../scripts/evaluate-serving-side-v2-protected.py)

The model bundle records the feature version, ordered 82-feature signature, offsets,
resize, development dataset hash, and SHA-256 of the extractor/model implementation.
The protected evaluation separately binds the frozen model and protected feature
dataset hashes.

## Data policy

`near` is positive and `far` is negative. The 295 `unclear` decisions remain excluded.

- Development: all 1,090 clear reviewed rows outside `split=test`, spanning 29
  recordings and 9 source groups.
- Protected test: 39 clear reviewed rows from
  `indoor-source-05`, source group `spu-match1-20260526`.
- The challenge and raw non-training labels were already opened for v1. They are
  therefore development data for v2 and are not described as unbiased v2 evaluation.
- Candidate, hyperparameters, and operating threshold were frozen before the
  protected feature artifact was extracted or loaded.

Selection uses leave-one-source-group-out predictions. The primary metric is mean
source-group balanced accuracy, followed by pooled balanced accuracy, pooled macro-F1,
worst-source-group balanced accuracy, and fewer features. This prevents the 389-row
phone source from winning by row count alone.

## Model comparison

The grid compared five feature families, logistic L2 values
`[0.01, 0.1, 1.0, 10.0]`, and 25/50-estimator class-balanced AdaBoost decision-stump
models.

| Candidate | Source-group macro BA | Pooled BA | Pooled macro-F1 |
| --- | ---: | ---: | ---: |
| Existing v1 features, best logistic | 78.85% | 84.78% | 84.76% |
| Court-flow recording ranks, 50 boosted stumps | 87.13% | 87.81% | 87.78% |
| **Court-flow recording ranks, logistic L2=0.1** | **87.45%** | 87.27% | 87.22% |

The boosted model had slightly better pooled and worst-group results, but the logistic
candidate won the predeclared primary source-group macro metric. Its frozen threshold
is `0.5408551369287771`; model fingerprint:
`0299c82e48af6bc725d721cc0cf3cb23018db9c9551009eb103a41ed376d974b`.

## Protected result

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Accuracy | 61.54% | **94.87%** |
| Balanced accuracy | 62.30% | **94.84%** |
| Macro-F1 | 61.44% | **94.84%** |
| Near precision | 68.75% | **95.24%** |
| Near recall | 52.38% | **95.24%** |
| Far precision | 56.52% | **94.44%** |
| Far recall | 72.22% | **94.44%** |

The v2 confusion matrix is:

```text
human near: 20 near, 1 far
human far:   1 near, 17 far
```

This is a single 39-row protected recording, so it is strong evidence for the feature
direction, not a precise estimate of all-camera production performance.

## NAS artifacts

| Artifact | SHA-256 |
| --- | --- |
| `features/serving-side-v2/development.json` | `e873880d7633e4786b962583016609e76c48e3b01f0ada16fa1eee2032a6a79e` |
| `features/serving-side-v2/protected-test.json` | `aec3c97be68e1f178ca74d46828e2f477529b5d539651eb59376755330d97dde` |
| `models/serving-side-specialist-v2/model.json` | `76a924e8d5c1d8a035e54a2606be30be861b7bc8690c59ddcce77b114bf6f6e9` |
| `reports/serving-side/serving-side-specialist-v2-development.json` | `8401f9b7e45a88b583be9d8ed246f13f62b82404313d850c0256d1ecedce2c6a` |
| `reports/serving-side/serving-side-specialist-v2-protected-test.json` | `7ce17b0ae146607c3c1cc75c4aa2850aa1aa5ddc4a6e31f4f652345b9fffe887` |

All paths are relative to
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

## Reproduction commands

For a new immutable artifact version, change the default destinations before rerunning;
the commands refuse to overwrite existing artifacts.

```bash
npm run extract:serving-side-v2
npm run train:serving-side-v2

# Run only after selection is frozen. Do not repeat for model selection.
npm run extract:serving-side-v2-protected
npm run evaluate:serving-side-v2-protected
```

## Next step

Review the two protected errors in the existing per-video results UI, then add court
service-zone anchors to a development subset and run a development-only anchor-versus-
fallback ablation. Do not use the protected recording to tune that follow-up.
