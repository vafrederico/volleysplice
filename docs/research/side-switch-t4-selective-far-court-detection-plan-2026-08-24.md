# Side-switch T4 selective far-court high-resolution detection — 2026-08-24

## Decision and isolation

T4 tests the first remediation for T3's engineering failure: increase the effective
resolution of distant far-side players without changing the pinned `224 x 224`
detector, video source, jersey descriptor, tracklet requirement, team pooling, or
transport formula.

The only representation change is detection ownership:

- near-side observations come from the existing four-tile detector over the complete
  native-resolution manifest ROI;
- far-side observations come from a second detector pass over a smaller, court-owned
  far band cut from that same native frame; and
- full-frame far detections and far-band near detections are discarded, preventing
  duplicate candidates or a mixed-resolution side.

No larger proxy, interpolation, super-resolution, new detector weights, score-threshold
change, label, model result, candidate source, or temporal tracker is part of T4.
Court-constrained temporal tracking remains a separate next experiment if selective
resolution alone is insufficient.

## Frozen far-court crop

Reuse T3's five endpoint fractions `(0.10, 0.30, 0.50, 0.70, 0.90)`. Let `n` be the
existing net y ratio in the already cropped manifest ROI. The far-band crop spans the
full ROI width and canonical y `[0.10, 0.62]`. In raw ROI coordinates its limits are:

```text
top_ratio    = 0.20 * n
bottom_ratio = n + 0.24 * (1 - n)
```

Round to pixel edges with the same nearest-integer convention as the manifest ROI.
Run the unchanged pinned detector and its unchanged four internal ownership tiles on
this band. Map every detection box, hip center, and shoulder center back into the full
ROI by adding the crop's top pixel. Retain only detections with canonical hip y below
`0.56` and within the existing court bounds.

The band is roughly half the ROI height, so its internal `224 x 224` resize gives far
players about twice the vertical model pixels. The source pixels and jersey crops
remain native; detector input resolution and anchor geometry remain exactly pinned.

Every frame also receives the unchanged full-ROI detector pass, but only its near-side
detections (canonical y at least `0.56`) enter T4. Total work is therefore 635 endpoint
windows, 3,175 frame requests, 25,400 detector tile calls, and no additional decoded
frames beyond T3.

## Frozen jersey, tracklet, team, and feature contract

Reuse T3 exactly for:

- shoulder-to-hip jersey crop and fallback torso box;
- skin and background rejection;
- 64-value HSV/Lab descriptor and support weighting;
- maximum three observations per side per frame;
- five-frame link cost `0.60 appearance + 0.25 court position + 0.15 scale`;
- link threshold `0.50`, minimum two observed frames, reliability, and maximum three
  tracklets per team;
- anonymous team pooling, cohesion, separation, and reliability; and
- base/swap/continuity gates and transport reductions.

The three T4 model values are separately named to prevent accidental T3/T4 mixing:

1. `selectiveFarJerseyTeamTransportSwapMargin`;
2. `selectiveFarJerseyReliableSwapEvidence`; and
3. `selectiveFarJerseyReliableContinuityEvidence`.

All raw diagnostics receive the same `selectiveFarJersey` prefix. No T3 value enters
the T4 head.

## Immutable artifact and label-free gates

Inputs are pinned to the same full-union, Visual Summary V2, manifest, detector, and
source videos as T3, plus immutable T3 artifact SHA-256
`b8f3a6ea8fa7acc8ca7962179f43e186ff988087d94f7bfc3fa05beb1c5793c5`
for label-free observability comparison only.

Output path:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t4-selective-far-court-detection-features-v1.json`.

T4 reaches modeling only if every gate passes:

- exact preservation of 704 IDs/order and all existing values;
- 624 eligible boundaries and 80 explicitly ineligible internal rows;
- at least 90% pooled endpoint any-team coverage;
- at least 60% pooled endpoint both-team coverage;
- no recording below 20% endpoint both-team coverage;
- at least 50% of boundaries with all four teams and nonzero cross-side similarity;
- at least +10 percentage points both-team endpoint coverage versus T3;
- at least +15 percentage points four-team boundary coverage versus T3;
- all three core values finite and nonconstant;
- reliable swap evidence nonzero on at least 5% of boundaries;
- maximum absolute core/core and core/existing Spearman below `0.98`;
- exactly 3,175 frame requests and 25,400 detector tile calls, no frame/detector
  errors, wall time at most 60 minutes, and peak RSS at most 768 MiB; and
- no audit, feedback, label, T2 model result, or T3 label result loaded.

Failure stops T4 before labels. Do not expand the crop, combine full/far detections,
lower the detector threshold, admit one-frame tracklets, or add temporal propagation
after observing this artifact.

## Frozen opened-development evaluation

If engineering passes, compare the exact 34-input matched boundary T0 with the exact
`34 + T4 three-value core` under the established nested recording-held-out square-root
logistic, top-2/2x hard-negative, threshold, and decoder protocol.

Require at least +2.0 percentage points pooled +/-4-second F1, no more than 2.0 points
lost in precision or recall, no more than 1.0 point lost in strict F1, recovery of at
least one covered T0 miss, reduction of T0-selected false boundaries with zero
`selectiveFarJerseyReliableSwapEvidence`, and the existing recording-robustness rule.

Report row AP/Brier, event counts, all recording deltas, standardized coefficients,
absolute coefficient ranks, and 11-fold sign stability. Do not prune the three-value
bundle unless it passes every gate. The full-E0-internal merge remains diagnostic only.
Opened-development success cannot authorize promotion or browser/Android work.

## Execution ledger

### Preregistration — frozen 2026-08-24

No T4 module, extractor, feature artifact, feature value, or model result existed when
this contract was committed.
