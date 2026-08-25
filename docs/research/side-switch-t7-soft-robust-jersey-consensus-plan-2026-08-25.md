# Side-switch T7 soft robust jersey consensus — 2026-08-25

## Decision and isolation

T7 tests the representation between the two measured failures: T5 averages every
court-side jersey and preserves coverage but blurs identity; T6 hard-rejects jersey
outliers and restores identity but destroys temporal coverage.

Preserve T5/T6 source frames, detector, selective far crop, court ownership, five
fractions, jersey descriptor, selected observations, two-frame availability rule,
candidate rows, transport formula, labels, and model protocol. Change only the weight
assigned to each retained jersey observation during temporal team aggregation.

No hard rejection, new detector call, frame, crop, threshold, candidate, optical flow,
label, or decoder change is part of T7.

## Frozen soft-consensus tracker

Select the reference medoid with T6's exact radius-`0.38` consensus scoring and
deterministic tie-break. The medoid is only a reference; every T5 observation remains.
For observation-to-medoid Hellinger distance `d`, define:

```text
robust_weight = 1 / (1 + (d / 0.38)^2)
```

Within each non-empty frame:

- descriptor weight is `confidence * support * robust_weight`;
- frame descriptor is the normalized weighted jersey mean;
- frame reliability is T5 frame reliability multiplied by
  `sqrt(mean robust_weight in that frame)`.

Use T5's temporal descriptor, reliability coverage factor, cohesion, and requirement
of observations in at least two distinct frames. Because robust weights are strictly
positive, availability must exactly match T5. Report mean/effective robust support,
distance quantiles, medoid identity, and the number of observations below weights
`0.5` and `0.25`; never discard them.

The model core is:

1. `softConsensusJerseyTeamTransportSwapMargin`;
2. `softConsensusJerseyReliableSwapEvidence`; and
3. `softConsensusJerseyReliableContinuityEvidence`.

No T4/T5/T6 value enters the T7 head.

## Label-free engineering gates

Pin T6 artifact SHA-256
`e53dafc72f1645936c6c4618bb201d33413216ad203550f2a05b6e14b0db89e4`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t7-soft-robust-jersey-consensus-features-v1.json`.

T7 reaches labels only if all gates pass:

- exact preservation of 704 IDs/order and every T6 prior value;
- exact T5 both-team endpoint availability (`81.1024%`) and no recording below T5's
  `42.3077%` minimum;
- four-team visibility no more than one point below T5's `68.75%`;
- mean minimum team separation at least `0.3796`, +0.06 over T5;
- positive raw swap margins at least `23.56%`, above T4 and at least +3 points over
  T5;
- nonzero reliable swap evidence at least T4's `17.79%`;
- mean effective robust support at least 35%, zero discarded observations, and at
  least one observation below weight 0.5 so T7 is not an identity transform;
- all core values finite/nonconstant and maximum absolute core/core, core/T4,
  core/T5, core/T6, and core/existing Spearman below `0.98`;
- exactly 635 endpoints, 3,175 frames, and 25,400 tile calls, no errors, <=60 minutes,
  and <=768 MiB RSS; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not tune the scale, formula, or reliability factor.

## Conditional model comparison

If engineering passes, run exact boundary T0 versus exact `34 + T7 core`, with
immutable T4 as comparator. Require +2 F1 points over T0, +0.5 point over T4, prior
precision/recall/strict/target-slice/recording guardrails, and a positive reliable-swap
coefficient in the full fit and at least nine of 11 outer fits. Report all event,
row-calibration, coefficient, recording, and diagnostic-merge results. Opened-scope
success cannot authorize promotion; new recording-held gold remains required.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T7 code, artifact, feature value, model profile, or model result existed when this
contract was committed.
