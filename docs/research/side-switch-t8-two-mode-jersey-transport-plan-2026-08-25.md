# Side-switch T8 explicit two-mode jersey transport — 2026-08-25

## Decision and isolation

T5/T6/T7 establish a representation tradeoff. A single full mean retains visibility
but blurs identity, a hard dominant mode restores identity but loses visibility, and
a soft dominant-mode mean restores directional coverage but not enough separation.
T8 keeps a bounded appearance set instead of collapsing it to one descriptor.

Preserve T7's exact 704 rows, source videos, ROI, five endpoint fractions, full-frame
near detector, selective far-court crop and detector, jersey descriptor, three-player
selection, court-side ownership, candidate source, labels, model protocol, and
decoder. Change only temporal appearance aggregation and the cost between team
representations. No threshold search, radius search, extra frame, detector call,
resolution change, optical flow, label, or decoder change is part of T8.

## Frozen two-mode representation

For every court side and endpoint, collect the same T5 observations with base weight
`confidence * support`. If none exist, return no modes. If one exists, return it as
one mode. Otherwise choose exactly two observation medoids by exhaustive deterministic
weighted two-medoids minimization under Hellinger distance:

```text
objective = sum(base_weight_i * min(distance(i, medoid_1),
                                    distance(i, medoid_2)))
            / sum(base_weight_i)
```

Break ties lexicographically by medoid frame and within-frame observation index.
Assign every observation to its closest medoid, breaking equal-distance ties toward
the first medoid. A mode descriptor is the normalized base-weighted mean of its
assigned observations; its support is assigned base weight divided by total base
weight. No observation is discarded. Team availability remains T5's observations in
at least two distinct frames, even when an individual mode occupies one frame.

Between two available teams, define symmetric weighted Chamfer cost:

```text
0.5 * (sum(left_support * nearest_right_distance)
       + sum(right_support * nearest_left_distance))
```

Unavailable-team cost remains 1.0. Use this cost in T3's same-versus-swapped transport
margin and reliability gates. Team reliability is T5 reliability unchanged. Team
cohesion is one minus the weighted two-medoid objective. Report mode supports, frame
coverage, dispersion, support entropy, and assignments.

The first model head is exactly:

1. `twoModeJerseyTeamTransportSwapMargin`;
2. `twoModeJerseyReliableSwapEvidence`; and
3. `twoModeJerseyReliableContinuityEvidence`.

T4/T5/T6/T7 values remain comparators and do not enter the T8 head.

## Label-free engineering gates

Pin T7 artifact SHA-256
`d592961f184122afe6e88c95e6405c2bdc57bd71986733480021d46037cba57a`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t8-two-mode-jersey-transport-features-v1.json`.

T8 reaches labels only if all gates pass:

- exact preservation of 704 IDs/order and every T7 prior value;
- exact T5 both-team endpoint availability (`81.1024%`), no recording below T5's
  `42.3077%` minimum, and exact T5 four-team visibility (`68.75%`);
- mean minimum mode-set team separation at least T4's `0.3739`;
- positive raw swap margins at least T7's `25.4808%`;
- nonzero reliable swap evidence at least T7's `19.3910%`;
- every nonempty multi-observation track has two nonempty modes, no observation is
  discarded, and at least 20% of available tracks have secondary support >=20%, so
  the representation is materially multimodal;
- all core values finite/nonconstant and maximum absolute core/core, core/T4,
  core/T5, core/T6, core/T7, and core/existing Spearman below `0.98`;
- exactly 635 endpoints, 3,175 frames, and 25,400 tile calls, no errors, <=60 minutes,
  and <=768 MiB RSS; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not tune the number of modes, objective, set cost, or
support rule on this opened scope.

## Conditional model comparison

If engineering passes, run exact boundary T0 versus exact `34 + T8 core`, with the
immutable T4 result as comparator. Require +2 F1 points over T0, +0.5 point over T4,
prior precision/recall/strict/target-slice/recording guardrails, and a positive
reliable-swap coefficient in the full fit and at least nine of 11 outer fits. Report
all event, row-calibration, coefficient, recording, and diagnostic-merge results.
Opened-scope success cannot authorize promotion; new recording-held gold remains
required.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T8 code, artifact, feature value, model profile, or model result existed when this
contract was committed.
