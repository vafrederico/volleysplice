# Side-switch T12 type-consistent hierarchical transport — 2026-08-25

## Decision and isolation

T11 restores coverage but sometimes compares a pooled team fallback directly with an
individual stable player. T12 changes only pair-reduction selection: compare like with
like. It reuses frozen T10 player-unit reductions and frozen T5 pooled-team reductions
from the exact T11 rows; no video, detector, descriptor, tracklet, or label is loaded.

## Frozen hierarchy

For each of `nearNear`, `farFar`, `nearFar`, `farNear`, `beforeNearFar`, and
`afterNearFar` independently:

- if both side endpoints have at least one T10 stable tracklet, use the exact stored
  T10 unit reduction (`cost`, `coverage`, and conditional similarity);
- otherwise use the exact stored T5 pooled-team cost; pooled coverage is `1` and
  conditional similarity is `1 - cost` when both pooled teams are available, or both
  are `0` when either is unavailable.

Recompute T11's unchanged same-versus-swapped margin and reliability gates from those
six selected reductions. Per-side reliability is the mean reliability of its exact
T11 units. Prefix the unchanged five core semantics with `hierarchicalTrackletJersey`.
No blend, confidence threshold, selection margin, or learned router is permitted.

## Immutable input and label-free gates

Pin T11 artifact SHA-256
`aa0b6abfa6324350b7944728757ee2867e7ab6c35eadfc1e085f73339e74a6d5`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t12-type-consistent-hierarchical-transport-features-v1.json`.

T12 reaches labels only if all gates pass:

- exact 704-row ID/order and prior-feature preservation;
- exact T5 availability and exact T10 stable-tracklet flags;
- both stable and pooled reduction routes selected at least once, with every selection
  matching the frozen rule;
- mean minimum team separation at least T4's `0.37386011658617885`;
- positive raw margins at least T7's `25.48076923076923%`;
- nonzero reliable swap evidence at least T7's `19.391025641025642%`;
- all core values finite/nonconstant and maximum absolute core/core, core/T4 through
  core/T11, and core/existing Spearman below `0.98`;
- no video, detector, manifest, audit, feedback, label, or model artifact loaded;
- atomic refusal to overwrite, <=60 seconds, and <=512 MiB RSS.

Failure stops before labels. Do not tune routing, blend the stored costs, change unit
reliability, or adjust gates after observing T12.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T12 five-value
core` wiring before labels. Require +2 F1 points over T0, +0.5 point over T4, all
existing guardrails, and a positive reliable-swap coefficient in the full fit and at
least nine of 11 outer fits. Opened-scope success remains non-promotable without new
recording-held gold.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T12 code, artifact, feature value, model profile, or model result existed when this
contract was committed.

### Engineering result — reject before labels

Implementation is committed at `e683bab`. The immutable 33,267,385-byte artifact is:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t12-type-consistent-hierarchical-transport-features-v1.json`

SHA-256: `8941f5f369d98c3e25f8070f6fc13fb206858729750d930e8c5b55648cf4275d`.

T12 uses 2,833 stable-player and 911 pooled-team reductions. Direction, reliability,
separation, parity, resource, and every other novelty gate pass:

| Engineering measurement | Result | Frozen requirement | Check |
| --- | ---: | ---: | --- |
| Mean minimum team separation | 0.534842 | >=0.373860 | Pass |
| Positive raw margin | 26.9231% (168/624) | >=25.4808% | Pass |
| Nonzero reliable swap | 19.8718% (124/624) | >=19.3910% | Pass |
| Strongest T11 correlation | 0.986633 | <0.98 | **Fail** |

The failed pair is T12 versus T11 conditional cross similarity. Transformation took
0.560 seconds at 124.590 MiB RSS and loaded no video, detector, labels, or model.

Decision: **reject T12 and stop before labels**. Do not prune the observed T12 bundle.
The next representation should instead make route selection symmetric across the two
assignment hypotheses. For each before-side source, select stable-player transport for
both possible after-side destinations only when all three involved sides are stable;
otherwise use pooled-team transport for both. This prevents a same-versus-swapped
margin from comparing unlike route types.
