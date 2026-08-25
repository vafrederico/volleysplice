# Side-switch T13 source-symmetric routing — 2026-08-25

## Decision and isolation

T12 passes direction but can route a before-side's same and swapped comparisons through
different representation types. T13 freezes one route per before-side source across
both after-side destinations, so route availability cannot itself create the transport
margin.

## Frozen routing

For before-near, use exact T10 stable-player reductions for both `nearNear` and
`nearFar` only when before-near, after-near, and after-far all have stable units;
otherwise use exact T5 pooled reductions for both. Apply the same rule to before-far
for `farFar` and `farNear`. Within-endpoint `beforeNearFar` and `afterNearFar`
separation continues to use stable reduction when both sides are stable and pooled
reduction otherwise.

Recompute the unchanged five T12 semantics with prefix `sourceSymmetricTrackletJersey`.
No route blend, threshold, cost, descriptor, reliability, or aggregate changes.

## Immutable input and label-free gates

Pin T12 artifact SHA-256
`8941f5f369d98c3e25f8070f6fc13fb206858729750d930e8c5b55648cf4275d`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t13-source-symmetric-routing-features-v1.json`.

T13 reaches labels only if all gates pass:

- exact 704-row ID/order and prior-feature preservation;
- exact source-route symmetry for same/swapped comparisons and both routes used;
- exact T5 availability and frozen T10/T5 reduction reuse;
- mean minimum separation >=`0.37386011658617885`;
- positive margins >=`25.48076923076923%` and reliable swap >=`19.391025641025642%`;
- all five core values finite/nonconstant and maximum absolute core/core, core/T4
  through core/T12, and core/existing Spearman below `0.98`;
- no video, detector, manifest, audit, feedback, label, or model loaded;
- atomic output, <=60 seconds, and <=512 MiB RSS.

Failure stops before labels. Do not tune routing granularity, costs, core composition,
or gates after observing T13.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T13 five-value
core` wiring before labels and apply the unchanged +2-over-T0, +0.5-over-T4, guardrail,
and coefficient-sign gates. Opened-scope success remains non-promotable without new
recording-held gold.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T13 code, artifact, feature value, model profile, or model result existed when this
contract was committed.

### Engineering result — reject before labels

Implementation is committed at `9df4233`. Immutable artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t13-source-symmetric-routing-features-v1.json`

Size: 34,852,049 bytes. SHA-256:
`232a26201edf16fe104dec965e73bb9f1864162baa13749187b0d3ad4970972e`.

T13 uses 824 stable and 424 pooled before-side routes with exact route symmetry.
Separation remains 0.534842 and reliable swap passes at 19.5513%, but positive margins
fall to 23.7179% (148/624). Conditional similarity is also redundant with T11 at
0.983110 and T12 at 0.997307 Spearman. The transformation took 0.594 seconds at
127.711 MiB RSS.

Decision: **reject T13 and stop before labels**. No label or model source was loaded.
The T12 gain does not survive source-route symmetry, so no further routing variant is
authorized on this scope.

The next representation should address player-role appearance directly. T14 will
select the unweighted Hellinger medoid of each side's stable player tracklets as its
dominant jersey identity, with deterministic stored-order ties. Only an empty stable
side receives the exact T5 pooled fallback. This tests tracklet-level resistance to a
libero or other secondary role without changing track qualification or tuning a
distance threshold.
