# Serving-side concentrated flight-motion experiment — 2026-08-20

## Completed review, final-label retrain, and temporal follow-up — 2026-08-21

The failure-mode review is complete. All 61 current flight-model errors and all
119 still-valid stratified correct controls have annotations. The frozen weighted
report reconstructs the full 1,027-row development population from those 180 rows;
13 older annotations remain available but are outside this sampling design.

The final correction overlay contains 20 decisions: four near, eight far, and eight
not-serve. Base v4 and flight v3 were rebuilt from those labels. The eight non-serves
are excluded from fitting, while the separate all-reviewed inference bank retains
them for the production serve gate and results UI. The camera-impact exclusion for
`raw-no-backup-PXL_20260816_164327879` remains in force.

| Final-label model | Rows | Source-group macro BA | Pooled BA | Worst-group BA |
| --- | ---: | ---: | ---: | ---: |
| Base v4 | 1,067 | 89.91% | 89.16% | 78.88% |
| **Flight v3 fixed anchor** | **1,027** | **94.12%** | **94.05%** | **85.83%** |

The completed two-phase review estimates 94.06% overall accuracy for flight v3.
Visibility is the largest identified slice gap: visible 96.41%, partial 94.15%, and
offscreen 55.34%. The offscreen sample contains only human-near examples, so it does
not estimate offscreen far performance. Contact timing is similarly diagnostic:
on-anchor 97.08%, before-anchor 81.31%, and after-anchor 76.62%.

An inference-safe temporal experiment extracted the same 192×108 4×6 flight features
at fixed shifts of -1s, 0s, and +1s for every development row. The best temporal
candidate concatenated all three windows. It improved pooled BA from 94.05% to
94.35%, before-anchor accuracy from 81.31% to 87.15%, and after-anchor accuracy from
76.62% to 82.47%. It nevertheless reduced the primary mean source-group BA from
94.12% to 93.82%, reduced worst-group BA from 85.83% to 81.42%, and reduced on-anchor
accuracy from 97.08% to 96.03%. The fixed-center candidate therefore remains
selected. The useful follow-up is an inference-time contact-quality or anchor-search
head that routes only likely misaligned examples, not unconditional concatenation.

The all-video v4 inference artifact contains 1,114 rows across all 30 videos: 1,075
development candidates and the unchanged 39-row protected bank. The results UI now
defaults to this artifact. Selection did not reopen or use the protected split.

### Final-review NAS artifacts

All paths are relative to
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | SHA-256 |
| --- | --- |
| `reports/serving-side/serving-side-result-label-corrections-v1.json` | `82fd8a7d46472cccb88561ca5eb03ef6fb54cf97eed89e9efa31f416636b5253` |
| `reports/serving-side/serving-side-flight-error-annotations-v2.json` | `bb8fba35457aea1fba0d00520cefe0a75f4a877b006f4625010bf7868cca3bf5` |
| `reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json` | `4c96624991cc757b41bced5bcda9633e0b97660410edc7555d4d000fe032787b` |
| `features/serving-side-v4/development.json` | `e9712a956e50111940a37e04c8ebc69e8514138714a1e192b0fd1030a055979a` |
| `models/serving-side-specialist-v4/model.json` | `85b8d55fc544b9afd36e73c406baa1bed64ad74c8455a0e18745318c5ffa5453` |
| `reports/serving-side/serving-side-specialist-v4-development.json` | `925e7c8f6f1f288365265c99595f803c197d1ade57b5b3956cd8449be7a8eb30` |
| `features/serving-side-flight-v3/development.json` | `c4ddf9f94bec00ba0fa62f8267dc166fdeeb6f9419180acad2e7eadf4910af7d` |
| `reports/serving-side/serving-side-flight-v3-development.json` | `c867e8a2a141a5231fb1ceb5a0ba289457ac353fab6a9115dc672b84046f2414` |
| `features/serving-side-temporal-flight-v1/development.json` | `7fce86c2ab48c3660a706a7d1bbf04bcdac8ff2b7a10565dac9e69eae2218018` |
| `reports/serving-side/serving-side-temporal-flight-v2-development.json` | `ed0d118dc127429090a91f8b2974607badf0b70bc75c4e7c9b629cd367bd9fc2` |
| `features/serving-side-v4/all-reviewed-inference.json` | `f6a8f23fdc55c703e62095f5afdec2e571f841dd949cb6a7af1cebb7af321421` |
| `reports/serving-side/serving-side-specialist-v4-dual-serve-gate-all-video-inference-v3.json` | `3ef7449f1f97f0b7f1dcebea4718311e61bed2495d33682cd4135d669dcc3623` |

## Corrected-label retrain — 2026-08-21

The v1 result below is preserved as the original experiment record. This section
documents the intermediate 19-correction retrain; the completed 20-correction
retrain above supersedes it. Both exclude source-quality failures before fitting or
scoring.

`raw-no-backup-PXL_20260816_164327879` was hit by a ball at 6:57. The canonical
feedback bundle now marks `[417.0, 978.7967]` as
`camera-hit-rotated-partial-court-view`; its final export is clipped at 417s.
General temporal training masks that exact interval. Serving-side extraction
rejects a row if any sampled frame touches it. This removes 15 reviewed
serving-side rows while retaining the valid 410.125s serve anchor, whose last
flight sample is 411.875s.

The base specialist was retrained because at least one corrected row
(`grass-source-01:rally:37`) was in the original training split, and the
flight model's final refit used all non-test development rows. Seven corrected
non-serves are absent from both training banks. A separate, explicitly
non-training-eligible inference bank retains those seven rows so the production
serve gate remains reviewable in `/serving-side-results`.

| Model | Rows | Source-group macro BA | Pooled BA | Worst-group BA |
| --- | ---: | ---: | ---: | ---: |
| Base v2 before corrections | 1,090 | 87.45% | 87.27% | 72.46% |
| **Base v3 corrected retrain** | **1,068** | **90.10%** | **89.26%** | **78.88%** |
| Flight v1 before latest corrections | 1,046 | 92.56% | 92.73% | 81.71% |
| **Flight v2 corrected retrain** | **1,028** | **94.12%** | **94.05%** | **85.83%** |

The corrected flight model selected v2 plus flight recording ranks at
192×108, grid 4×6, logistic L2 0.1. Its confusion matrix is 472/35 for human
near and 26/495 for human far, or 61 errors total. The report now stores the
reusable final model parameters and fingerprint
`1e9b53a1023249126d0ecca9804fc2d68d63ed8d8c932018033b19caf8a7f9a7`.
The previously opened protected bank was not used for selection or fitting; it
is attached only to the all-video inference artifact.

The review UI now defaults to the v2 evaluation. Sixty-four saved failure-mode
annotations were migrated by rally ID. The two omitted annotations were for
post-impact rallies 42 and 46 in the damaged recording.

After all 61 v2 mistakes were reviewed, a separate correct-control cohort was
frozen for visibility-slice estimation. It originally sampled 120 of the 967 correct
out-of-source-group predictions across all 51 non-empty combinations of
environment, source group, human side, and model-confidence band. Every stratum
receives at least one row; the remaining allocation is proportional by largest
remainder. The artifact stores each stratum's population, sample count, and
sampling weight so later visible/partial/offscreen metrics can recover the
development population mixture. Selection within a stratum is deterministic
from the evaluation SHA-256 and rally ID. The review UI exposes this exact
cohort as **Correct controls** and saves its labels in the existing
evaluation-bound annotation artifact. The final not-serve correction removes one
control, leaving 119 valid controls and 966 current correct predictions; the final
weighted report recalculates the affected stratum weight.

### Corrected NAS artifacts

All paths except the feedback bundle are relative to
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | SHA-256 |
| --- | --- |
| `reports/full-nas-video-corpus-v2.json` | `bf3e1bd2601da1d742379f2b5897c6293a6b84c2b8c4f26550c8045cdde2f04e` |
| `reports/serving-side/serving-side-source-quality-exclusions-v1.json` | `e4782c598ea23245d7303238511a5b5e2c65e69e9ea74a00279efdba5a707c4e` |
| `features/serving-side-v3/development.json` | `2b84f0ae4bb406033e86939eaa5f2efe450f3fc475b6cdbcb5b130ed46dad38f` |
| `models/serving-side-specialist-v3/model.json` | `91fe5e1a0fd9d755fba5afc34724332a36027b81af972878ed60a6527c80c735` |
| `reports/serving-side/serving-side-specialist-v3-development.json` | `6e0a3ba8ce7cc7fdda911550a681f554749676e781fe79da05d2e30e8b5c0726` |
| `features/serving-side-flight-v2/development.json` | `53f0e0f69aa0b3b0ce1d1d4abe824e73edb0ea98b7dad273305860014906438c` |
| `reports/serving-side/serving-side-flight-v2-development.json` | `14c8f464422761a55007777f4490cafdf159c0ceca79042e04329549cb0308c4` |
| `reports/serving-side/serving-side-flight-error-annotations-v2.json` | `9e80a4b5ff667f959cb7dda6bac2e1bcf00809b42df950e0268578f7e6ede285` |
| `reports/serving-side/serving-side-flight-correct-control-cohort-v1.json` | `dddd58d8f89c5f3cdda3e906a31ee52e56224443d168beb48b892fd0a9a302a5` |
| `features/serving-side-v3/all-reviewed-inference.json` | `4a6fd7c7eb410aabd3f9ceeb3c779864782589b2815a4b8cef4d0da9ffd94e07` |
| `reports/serving-side/serving-side-specialist-v3-dual-serve-gate-all-video-inference-v2.json` | `30bbb4b465dd5b30b3568b439d3e4958d2731f8ee3d042f6ffbbd78440e6d298` |
| `/mnt/freenas/volleycut/model-feedback/project-15ljci6-cfd050cdf42eee16/bundle.json` | `7f2117c2859168c01b2b30e5730b4a40998a7ebe42f1fc6736aca85bc56d57c5` |

## Decision

Keep concentrated post-contact flight motion as the leading serving-side v3
feature direction. The development-selected candidate combines the existing v2
recording-rank features with 384×216 residual-motion features on a 4×6 grid. It
improves source-group macro balanced accuracy from 87.88% to 92.56% on the exact
same correction-clean development rows.

Do not open the protected test recording or promote this candidate yet. These are
development selection results across multiple candidates, not an unbiased final
estimate.

## Hypothesis and feature contract

The authoritative reviewed `rally.start` remains the serve-contact anchor. Nine
frames are sampled at
`[-0.15, 0.05, 0.20, 0.35, 0.55, 0.80, 1.10, 1.40, 1.75]`
seconds around contact. The extractor then:

- estimates dense optical flow;
- robustly fits and subtracts an affine camera-flow field, covering translation,
  rotation, and zoom;
- retains the strongest residual-motion energy without morphologically deleting a
  small ball-sized component;
- summarizes launch, early-flight, and late-flight phases;
- records grid energy, vertical flow by grid row, motion centroid/spread, entropy,
  connected-component concentration, flow direction, divergence, top-versus-bottom
  energy, and small-component motion; and
- records phase-to-phase trajectory deltas.

This is an interpretable concentrated-motion proxy, not a claim of direct ball
detection. The immutable feature version is
`serving-side-concentrated-flight-grid-v1`.

Implementation:

- [`analysis/serving_side_flight.py`](../../analysis/serving_side_flight.py)
- [`scripts/extract-serving-side-flight-features.py`](../../scripts/extract-serving-side-flight-features.py)
- [`scripts/evaluate-serving-side-flight.py`](../../scripts/evaluate-serving-side-flight.py)

## Data policy

The feature bank contains 1,046 rows: 522 near and 524 far across 28 recordings and
nine source groups.

- Four current human `not-serve` corrections are excluded.
- Two current near/far corrections are applied.
- All unclear decisions are excluded.
- The protected test recording is never opened.
- The entire protected source group, `spu-match1-20260526`, is excluded. This also
  removes the development recording `indoor-source-03`, fixing a same-match
  scope issue inherited from the earlier v2 development artifact.
- Feature extraction aborts rather than publish if the correction overlay changes
  during the run.

The comparison therefore applies only inside this report. Do not compare its values
directly to a v2 report with a different row/source-group scope.

## Selection protocol

Selection uses leave-one-source-group-out predictions. The primary descending metric
is mean source-group balanced accuracy, followed by pooled balanced accuracy, pooled
macro-F1, worst-source-group balanced accuracy, and fewer features.

The first stage screens all nine resolution/grid combinations for flight recording
ranks alone and v2 plus flight ranks with logistic L2 fixed at 0.1. The second stage
tunes `[0.01, 0.1, 1.0, 10.0]` for the selected configuration across v2-only,
flight-absolute, flight-rank, and combined feature families. Protected data is not
loaded, scored, or used at either stage.

## Result

| Candidate | Features | Source-group macro BA | Pooled BA | Pooled macro-F1 | Worst-group BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correction-clean v2 recording ranks, L2=0.1 | 82 | 87.88% | 87.56% | 87.54% | 68.72% |
| 384×216 4×6 flight ranks, L2=0.1 | 155 | 91.33% | 91.77% | 91.77% | 81.07% |
| **v2 + 384×216 4×6 flight ranks, L2=0.1** | **237** | **92.56%** | **92.73%** | **92.73%** | **81.71%** |

The selected confusion matrix is:

```text
human near: 478 near, 44 far
human far:   32 near, 492 far
```

Against correction-clean v2 on the paired out-of-source-group predictions, the
selected model fixes 64 rows that v2 misses and regresses 10 rows that v2 gets right.
Both models get 906 rows right and 66 rows wrong.

### Resolution and grid screening

The following table holds the family at v2 plus flight recording ranks and L2 at
0.1, isolating resolution/grid choice:

| Resolution | Grid | Source-group macro BA | Pooled BA |
| --- | --- | ---: | ---: |
| **384×216** | **4×6** | **92.56%** | **92.73%** |
| 192×108 | 4×6 | 91.85% | 92.54% |
| 384×216 | 4×4 | 91.78% | 91.68% |
| 640×360 | 4×6 | 91.67% | 92.25% |
| 384×216 | 3×3 | 91.42% | 91.58% |
| 640×360 | 3×3 | 91.34% | 91.39% |
| 192×108 | 3×3 | 91.15% | 91.87% |
| 640×360 | 4×4 | 91.04% | 91.97% |
| 192×108 | 4×4 | 90.36% | 91.68% |

Medium resolution is sufficient for this residual-motion representation; 640×360
adds cost without improving the primary metric. The denser, roughly square-cell 4×6
grid is consistently useful at 192×108 and 384×216.

### Slice behavior

Compared with correction-clean v2, selected balanced accuracy changes from 83.85% to
91.49% on grass, 86.34% to 91.93% indoors, and 90.35% to 93.99% on the raw/unknown
slice. Beach falls from 98.44% to 96.88% on 59 rows and remains a guardrail for the
next iteration.

The largest fitted coefficients include early/late vertical motion spread, energy in
specific grid rows and columns, component concentration, and launch divergence. This
supports the flight-motion hypothesis, but coefficient magnitude is not causal
evidence because features are correlated and standardized.

## NAS artifacts

All paths are relative to
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | SHA-256 |
| --- | --- |
| `features/serving-side-flight-v1/development.json` | `dcfc6431aad6b94452895b0c26393d4670f766e069162e7639d6e9ba67d69e0f` |
| `reports/serving-side/serving-side-flight-v1-development.json` | `0b03cf85cf5b170417ac7be6e1e1e30a3eb6a25ce8c9a1eef5838cb13c1d4fe4` |

The feature artifact records the source video and label hashes for every recording,
the correction overlay hash, all nine full feature vectors per row, and the exact
implementation hashes. The evaluation artifact records every candidate, fold and
slice metric, paired selected predictions, and the fitted candidate's largest
weights.

## Reproduction

The commands refuse to overwrite immutable artifacts:

```bash
npm run extract:serving-side-flight
npm run evaluate:serving-side-flight
```

## Next step

Annotate server visibility on a development subset and report visible versus
offscreen performance. Then freeze the 384×216 4×6 candidate and run one protected
evaluation only if the visibility slices and beach guardrail are acceptable. A later
experiment can compare this concentrated-motion proxy with an explicit high-
resolution ball track, without reopening the protected result for selection.

### Failure-mode annotation contract

The `/serving-side-flight-review` UI reviews the exact 1,046 out-of-source-group
predictions in this report and starts with its 76 mistakes. It writes the separate,
atomic NAS artifact
`reports/serving-side/serving-side-flight-error-annotations-v1.json`. The artifact is
bound to this evaluation's SHA-256 and selected prediction digest, so it cannot be
silently reused with a different experiment.

Each reviewed rally records server visibility, whether actual contact is before, at,
or after the original anchor, an optional exact corrected contact time, ball-flight
visibility, whether visible direction agrees with the human side, notes, and the
server-generated review time. Camera pan and zoom are intentionally not human labels;
the residual-flow pipeline estimates camera motion directly.

The review loader overlays the current, identity-bound human-label correction file on
the frozen out-of-source-group predictions. Corrected near/far decisions immediately
update the review outcome and displayed metrics; corrected non-serves leave the side
evaluation universe. The frozen decision remains visible for audit, and saved
failure-mode annotations remain accessible through **Saved labels** even when a label
correction changes an example from mistake to correct. This is a corrected review of
frozen predictions, not a claim that the model has been retrained.

`correctedServeAnchorSeconds` is a supervision target for a future anchor-quality or
contact-localization head. Side-model selection must not substitute it only on the
outcome-dependent reviewed subset, because production inference does not have that
human correction and the review sample was selected from prior predictions.
Visibility and direction annotations are evaluation slices and possible
mixture-of-experts targets, not input features at inference time. The immutable v1
feature artifact above remains unchanged.
