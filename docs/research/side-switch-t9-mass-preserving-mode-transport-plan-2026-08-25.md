# Side-switch T9 mass-preserving mode transport — 2026-08-25

## Decision and isolation

T8 proves that two explicit appearance modes recover most team separation while
retaining T5 visibility, but symmetric nearest-mode distance loses direction because
several source modes may select the same destination mode. T9 retains T8's modes and
changes only their pairwise cost to exact support-mass optimal transport.

Preserve T8's exact videos, ROI, frames, detector calls, crops, jersey observations,
weighted two-medoid selection, assignments, descriptors, supports, availability,
reliability, cohesion, candidate source, labels, model protocol, and decoder. No mode,
threshold, radius, frame, resolution, detector, crop, label, or decoder change is part
of T9.

## Frozen mass-preserving cost

For available mode sets with normalized supports `a_i` and `b_j`, use the exact
minimum transport cost:

```text
minimize  sum(flow_ij * Hellinger(mode_i, mode_j))
subject to sum_j flow_ij = a_i
           sum_i flow_ij = b_j
           flow_ij >= 0
```

There are at most two modes per side, so solve the one-by-one, one-by-two, two-by-one,
and two-by-two cases analytically with deterministic endpoint choice for a flat
two-by-two objective. Unavailable-team cost remains 1.0. Substitute this cost into
the unchanged T3 same-versus-swapped formula and reliability gates.

The model core is exactly:

1. `massTransportJerseyTeamTransportSwapMargin`;
2. `massTransportJerseyReliableSwapEvidence`; and
3. `massTransportJerseyReliableContinuityEvidence`.

T4–T8 values remain comparators and do not enter the T9 head.

## Label-free engineering gates

Pin T8 artifact SHA-256
`4da68234930052d3669e7b40b003a6c6633068bd88f18e7629ae56536ff5d302`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t9-mass-preserving-mode-transport-features-v1.json`.

T9 reaches labels only if all gates pass:

- exact preservation of 704 IDs/order and every T8 prior value;
- exact T5 both-team endpoint availability (`81.1024%`), weakest-recording
  availability (`42.3077%`), and four-team visibility (`68.75%`);
- exact preservation of T8 mode counts, supports, assignments, and availability;
- mean minimum team separation at least T4's `0.3739`;
- positive raw swap margins at least T7's `25.4808%`;
- nonzero reliable swap evidence at least T7's `19.3910%`;
- mean available-pair mass-transport cost exceeds its paired T8 Chamfer cost by at
  least `0.01`, demonstrating that mass preservation is active;
- all core values finite/nonconstant and maximum absolute core/core, core/T4,
  core/T5, core/T6, core/T7, core/T8, and core/existing Spearman below `0.98`;
- exactly 635 endpoints, 3,175 frames, and 25,400 tile calls, no errors, <=60 minutes,
  and <=768 MiB RSS; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not tune support, modes, transport cost, or gates on
this opened scope.

## Conditional model comparison

If engineering passes, run exact boundary T0 versus exact `34 + T9 core`, with the
immutable T4 result as comparator. Require +2 F1 points over T0, +0.5 point over T4,
prior precision/recall/strict/target-slice/recording guardrails, and a positive
reliable-swap coefficient in the full fit and at least nine of 11 outer fits. Report
all event, row-calibration, coefficient, recording, and diagnostic-merge results.
Opened-scope success cannot authorize promotion; new recording-held gold remains
required.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T9 code, artifact, feature value, model profile, or model result existed when this
contract was committed.

### Implementation and interrupted extraction — 2026-08-25

The analytic transport module, focused tests, and label-free extractor are committed
at `46d850b`. The first full extraction reached the beginning of recording 6 of 11
and was intentionally interrupted for a machine handoff. The process is no longer
running. Atomic output means no T9 artifact or partial feature result exists. Restart
the full extraction from recording 1; do not interpret the interrupted attempt as an
engineering result or change any frozen gate.

Continuation details are in
[the side-switch feature loop handoff](./side-switch-feature-loop-handoff-2026-08-25.md).
