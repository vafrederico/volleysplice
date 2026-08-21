# Serving-side concentrated flight-motion experiment — 2026-08-20

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

Successor feature extractors must prefer `correctedServeAnchorSeconds` when present.
Visibility and direction annotations are evaluation slices and mixture-of-experts
targets, not input features at inference time. The immutable v1 feature artifact above
remains unchanged.
