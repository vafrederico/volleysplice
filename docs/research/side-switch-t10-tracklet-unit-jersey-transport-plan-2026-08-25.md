# Side-switch T10 tracklet-unit jersey transport — 2026-08-25

## Decision and isolation

T9 showed that exact transport over endpoint-local appearance-mode mass restores team
separation but not enough same-versus-swapped direction. T10 changes the representation
upstream of that cost: retain T4's persistent selective-far player tracklets as the
appearance units instead of pooling or reclustering their per-frame observations.

Preserve the exact T9 rows and prior values, T4 videos, ROI, five frame fractions,
full/near and selective-far detector ownership, jersey descriptor, observation cap,
five-frame link formula, `0.50` link threshold, two-frame qualification, maximum three
tracklets per side, candidate source, labels, model protocol, and decoder. T10 does not
alter a detector, crop, descriptor, tracker threshold, label, or model setting.

The only representation change from T4 is that its qualifying player tracklets remain
separate equal identity units; they are not collapsed into one reliability-weighted
team descriptor. Unlike T8/T9, detector-observation frequency is not transport mass.

## Frozen unit matching

For two nonempty side tracklet sets `A` and `B`, create the rectangular matrix of
Hellinger distances between their frozen 64-value jersey descriptors. Solve the exact
one-to-one assignment of `min(|A|, |B|)` pairs, choosing the lexicographically first
assignment when total cost ties. Define:

```text
coverage              = min(|A|, |B|) / max(|A|, |B|)
conditionalSimilarity = 1 - mean(Hellinger distance of assigned pairs)
cost                  = 1 - coverage * conditionalSimilarity
```

An unavailable set has coverage and conditional similarity `0` and cost `1`. Thus an
unmatched stable player unit receives the fixed maximum cost without a tunable penalty.
Reliability is the minimum mean frozen T3 tracklet reliability of the two sets.

Apply the unchanged same-versus-swapped construction to the four before/after court
side pairings. Within-endpoint near/far unit cost supplies team separation. Define the
base reliability gate as the minimum of all four side-set mean reliabilities and both
endpoint team separations. Reliable swap evidence additionally gates positive margin
by cross-side conditional similarity and coverage; reliable continuity evidence gates
negative margin by same-side conditional similarity and coverage.

The exact T10 model core is:

1. `trackletUnitJerseyTeamTransportSwapMargin`;
2. `trackletUnitJerseyConditionalCrossSimilarityMinimum`;
3. `trackletUnitJerseyCrossMatchCoverageMinimum`;
4. `trackletUnitJerseyReliableSwapEvidence`; and
5. `trackletUnitJerseyReliableContinuityEvidence`.

No post-extraction pruning or alternate mean/minimum aggregate is authorized.

## Immutable input and label-free gates

Pin T9 artifact SHA-256
`c21f4f5360185ae37904e622ab5caf3c0edeb13dfd58506a6ab814aa57b9c5e2`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t10-tracklet-unit-jersey-transport-features-v1.json`.

T10 reaches labels only if all gates pass:

- exact preservation of 704 IDs/order and every T9 prior value;
- exactly 624 eligible boundaries and 80 ineligible internal rows;
- exact T4 player-tracklet observability: 74.1732% both-team endpoints, 32.6923%
  weakest-recording availability, and 58.3333% four-team boundaries;
- exact agreement with stored T4 qualifying-tracklet counts at every endpoint and
  exact T4 selected-player counts for every extracted window;
- mean minimum team separation at least T4's `0.37386011658617885`;
- positive raw swap margins at least T7's `25.48076923076923%`;
- nonzero reliable swap evidence at least T7's `19.391025641025642%`;
- nonzero bidirectional cross-match coverage on at least 50% of boundaries;
- absolute Spearman between conditional cross similarity and cross-match coverage
  below `0.90`, matching the earlier T2 decomposition gate;
- all five core values finite/nonconstant and maximum absolute core/core, core/T4,
  core/T5, core/T6, core/T7, core/T8, core/T9, and core/existing Spearman below `0.98`;
- exactly 635 endpoints, 3,175 frames, and 25,400 tile calls, no errors, <=60 minutes,
  and <=768 MiB RSS; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not tune link qualification, unit cost, assignment,
coverage, gates, or feature aggregates on this opened scope.

## Conditional model comparison

If engineering passes, implement and commit the matched runner before loading labels.
Compare exact boundary T0 (`34` inputs) with exact `34 + T10 core`, retaining immutable
T4 as comparator. Require +2 F1 points over T0, +0.5 point over T4, all prior
precision/recall/strict/target-slice/recording guardrails, and a positive reliable-swap
coefficient in the full fit and at least nine of 11 outer fits. Report all event,
row-calibration, fold, coefficient, recording, and diagnostic-merge results. An
opened-scope success still requires new recording-held gold before promotion.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T10 code, artifact, feature value, model profile, or model result existed when this
contract was committed.

### Implementation and extraction — 2026-08-25

The module, focused tests, and initial serial extractor are committed at `478fa6e`.
That first run was intentionally stopped during recording 3 before any atomic output
existed, after the user requested recording-level parallelism. Commit `9c84303` adds
two deterministic recording workers with independent decoders/detectors and ordered
parent merge; it changes execution only, not the frozen representation or gates.

The restarted extraction completed and wrote the immutable 27,329,902-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t10-tracklet-unit-jersey-transport-features-v1.json`

SHA-256: `10ab41ebab069f20edf7b720152dc0ec94ef982f12a88200ec564b6c63849cce`.

### Engineering result — reject before labels

T10 preserves all 704 rows/prior values and exactly reproduces every stored T4
tracklet count, selected-player count, and availability measurement. The equal-unit
representation is strongly separated and nonredundant, but misses both frozen
direction-coverage gates:

| Engineering measurement | Result | Frozen requirement | Check |
| --- | ---: | ---: | --- |
| Both-team endpoint availability | 74.1732% | exact T4 | Pass |
| Weakest-recording availability | 32.6923% | exact T4 | Pass |
| Four-team boundary visibility | 58.3333% | exact T4 | Pass |
| Mean minimum team separation | 0.584519 | >=0.373860 | Pass |
| Positive raw transport margin | 23.7179% (148/624) | >=25.4808% | **Fail** |
| Nonzero reliable swap evidence | 17.1474% (107/624) | >=19.3910% | **Fail** |
| Nonzero cross-match coverage | 58.3333% | >=50% | Pass |
| Conditional-similarity/coverage Spearman | 0.776695 | <0.90 | Pass |

Every core and novelty check passes. The two-worker run completed exactly 635
endpoints, 3,175 frames, and 25,400 detector tile calls without errors in 611.610
seconds (10m11.610s) at 631.914 MiB peak RSS.

Decision: **reject T10 and stop before labels**. No label, audit, feedback, or model
artifact was loaded. There is no T10 precision, recall, F1, or coefficient result.

The 364 four-team-visible boundaries contain 148 positive T10 margins, compared with
147 for T4 on the same visibility scope. The representation improves conditional
direction and separation but cannot meet an all-boundary gate while retaining T4's
tracklet availability. T11 should preserve every T10 stable player unit and introduce
one explicitly typed T5 pooled fallback unit only for a side with no qualifying
tracklet. This restores observation coverage without changing T10's link threshold,
unit cost, or stable-track representation.
