# Side-switch T11 empty-side fallback tracklet transport — 2026-08-25

## Decision and isolation

T10 proves that equal stable-player units deliver strong team separation and slightly
more positive direction than T4 on the same visible scope, but inherits T4's missing
tracklets. T11 retains every T10 stable unit and changes only an otherwise empty side:
if the exact T5 court-team pool is available, represent it as one explicitly typed
pooled fallback unit.

Preserve T10's exact videos, ROI, five frames, detector ownership, jersey observations,
T3 link formula/threshold, two-frame stable-track qualification, maximum stable units,
equal-unit assignment, cost, feature aggregates, candidates, labels, model protocol,
and decoder. Do not add a fallback when any stable tracklet exists. Do not mix pooled
observations into stable descriptors or assign the fallback fractional mass.

## Frozen fallback construction

For each endpoint court side:

1. build the exact T10/T4 qualifying player tracklets;
2. if at least one exists, emit exactly those equal stable units;
3. otherwise build the exact T5 court-team pool from the same per-frame observations;
4. if that pool is available, emit its descriptor/reliability as one `pooled-fallback`
   unit; otherwise emit no unit.

Apply T10's exact one-to-one unit assignment, maximum unmatched-unit cost, coverage,
conditional similarity, reliability gates, and same-versus-swapped formula. Prefix the
five unchanged core semantics with `fallbackTrackletUnitJersey`.

## Immutable input and label-free gates

Pin T10 artifact SHA-256
`10ab41ebab069f20edf7b720152dc0ec94ef982f12a88200ec564b6c63849cce`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t11-empty-side-fallback-tracklet-transport-features-v1.json`.

T11 reaches labels only if all gates pass:

- exact preservation of 704 IDs/order and every T10 prior value;
- exact preservation of every T10 stable-tracklet count and descriptor construction;
- exact T5 availability: 81.1024% both-team endpoints, 42.3077% weakest recording,
  and 68.75% four-team boundaries;
- fallback used on at least one side and never used when a stable unit exists;
- mean minimum team separation at least T4's `0.37386011658617885`;
- positive raw swap margins at least T7's `25.48076923076923%`;
- nonzero reliable swap evidence at least T7's `19.391025641025642%`;
- nonzero bidirectional cross-match coverage on at least 60% of boundaries;
- absolute conditional-similarity/coverage Spearman below `0.90`;
- every core value finite/nonconstant and maximum absolute core/core, core/T4 through
  core/T10, and core/existing Spearman below `0.98`;
- exactly 635 endpoints, 3,175 frames, and 25,400 tile calls, no errors, <=60 minutes,
  and <=2,048 MiB RSS using four deterministic recording workers; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not tune when fallback activates, its unit weight,
the pooled descriptor, stable qualification, assignment, aggregates, or gates after
observing T11.

## Conditional model comparison

If engineering passes, first implement and commit exact boundary T0 (`34`) versus
`34 + T11 five-value core`, with immutable T4 as comparator. Require +2 F1 points over
T0, +0.5 point over T4, every existing guardrail, and positive reliable-swap
coefficient in the full fit and at least nine of 11 outer fits. Opened-scope success
still cannot authorize promotion without new recording-held gold.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T11 code, artifact, feature value, model profile, or model result existed when this
contract was committed.
