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

### Extraction — engineering pass 2026-08-24

Implementation checkpoint: `9b7b22f`.

The full selective-crop pass succeeds on every frozen engineering gate:

| Engineering metric | T3 | T4 | Requirement | Result |
| --- | ---: | ---: | ---: | --- |
| Endpoint any qualified team | 99.69% | 99.69% | >=90% | Pass |
| Endpoint both qualified teams | 44.57% | 74.17% | >=60%, gain >=10 pp | Pass (+29.61 pp) |
| Boundary four-team observability | 26.12% | 58.33% | >=50%, gain >=15 pp | Pass (+32.21 pp) |
| Weakest recording both-team coverage | 3.39% | 32.69% | >=20% | Pass |
| Reliable swap evidence nonzero | 7.69% | 17.79% | >=5% | Pass |
| Maximum core/core absolute Spearman | — | 0.6124 | <0.98 | Pass |
| Maximum core/existing absolute Spearman | — | 0.2635 | <0.98 | Pass |

All 704 rows, prior values, IDs, and order are exact. The run completes 3,175 frame
requests and 25,400 detector tile calls without errors in 1,554.240 seconds
(25m54.240s) at 233.742 MiB peak RSS.

The improvement transfers across all recordings. Notably, both-team coverage moves
from 3.39% to 57.63% in `183701800`, 12.12% to 65.15% in conflict recording
`193307688`, 30.16% to 93.65% in `203801418`, and 37.04% to 70.37% in `190429172`.
The weakest result is now `161923155` at 32.69%, above the predeclared floor.

Decision: **T4 passes engineering and enters the frozen opened-development model
comparison**. This validates selective-crop observability and feature novelty, not
precision, recall, or promotion.

Immutable feature artifact:

- path: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t4-selective-far-court-detection-features-v1.json`
- SHA-256: `443c0ded15cfddbb5e156f670c895375c8c43caede032a9b3ef5ae445ca4113a`
- module SHA-256: `606168dbd6ff13b399a2c952faa6f497cf2772237ccee33070b6561fba0613f8`
- extractor SHA-256: `441317c732a7fee864b6e0437633512d61e3af6c7db14d12f826a77432e0af8a`

Validation after extraction: all 151 focused `test_side_switch*.py` tests pass.

### Opened-development model — near miss, rejected 2026-08-24

Model-runner checkpoint: `78cf55e`. The run used the exact frozen 624-boundary
comparison: the same labels, recording-held-out outer folds, inner threshold selection,
hard-negative policy, decoder, and 34 baseline inputs. T4 appends only the three named
core values, producing an exact 37-input head.

| Held-out event metric | Matched T0 | T4 | Delta |
| --- | ---: | ---: | ---: |
| Proposals | 52 | 56 | +4 |
| True positives | 26 | 28 | +2 |
| False positives | 26 | 28 | +2 |
| False negatives | 24 | 22 | -2 |
| Precision | 50.00% | 50.00% | 0.00 pp |
| Recall | 52.00% | 56.00% | +4.00 pp |
| F1 | 50.98% | 52.83% | +1.85 pp |
| Strict F1 | 41.18% | 41.51% | +0.33 pp |
| Row average precision | 0.4356 | 0.4446 | +0.0090 |
| Row Brier score | 0.05453 | 0.05431 | -0.00022 |

T4 passes six of seven model checks. It recovers one of the three T0 misses with
nonzero reliable swap evidence, reduces selected zero-reliable-swap false boundaries
from 21 to 19, preserves precision, raises recall, preserves strict F1, and passes the
recording-robustness rule. It fails the primary decision gate because the required F1
gain is at least 2.00 percentage points and the observed gain is 1.8498 points. The
0.1502-point shortfall is not rounded into a pass.

The net gain is concentrated but does not trigger the frozen robustness failure:
`161923155` gains two TP and one FP, `210449857` gains one TP and one FP, and conflict
recording `193307688` loses one TP. The other eight recordings have unchanged event
counts. Across individual selections, T4 adds three TP and four FP while removing one
TP and two FP.

#### Feature direction learned from T4

The raw transport direction is credible: `selectiveFarJerseyTeamTransportSwapMargin`
has coefficient `+0.1053`, rank 17/37 by absolute standardized coefficient, and is
positive in all 11 outer fits. Reliable continuity is stronger and consistently acts
as a veto: coefficient `-0.1577`, rank 13/37, negative in all 11 fits. In contrast,
`selectiveFarJerseyReliableSwapEvidence` is effectively unused: coefficient `+0.0059`,
rank 34/37, with only seven positive outer-fold signs.

This resolves the resolution question empirically. Giving distant players about twice
the effective detector pixels materially improves observability and supplies stable
transport/continuity signal, so `224 x 224` full-ROI detection was a real bottleneck.
Resolution alone is not the complete model bottleneck: reliable swap evidence remains
nonzero on only 17.79% of boundaries, and four new FP arrive with the added recall.

The next isolated feature experiment should therefore preserve the selective far crop
and add court-constrained temporal tracking between its five samples. It should target
track continuity and reliability, especially on the far side, rather than enlarge the
proxy or tune the model threshold. Candidate outputs should remain the same three
semantic reductions, under a new versioned prefix, so the comparison measures whether
tracking turns unstable or zero-gated transport into reliable direction evidence.
Freeze and test that bundle as a whole; do not prune T4 after this failed gate, do not
combine tracking with a detector-resolution change, and do not port T4.

Decision: **T4 is rejected for promotion and runtime work, but selective far-court
detection is retained as the engineering foundation for a separately preregistered
tracking experiment.**

Immutable model artifact:

- path: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t4-selective-far-opened-v1.json`
- SHA-256: `16d617fb8427517b45c31196877d7482cb019f8902b989e5b2db19cfcef0020b`
- runner SHA-256: `65ac8e9dee004b4dd130ad9173ac568c5993192f31168de5674a53477b8f2a41`
- model module SHA-256: `63be32333891e2e0e01521e6ef99e6d1281ba844c74aff40e4ac548145b80c42`

Validation after modeling: all 152 focused `test_side_switch*.py` tests pass.
