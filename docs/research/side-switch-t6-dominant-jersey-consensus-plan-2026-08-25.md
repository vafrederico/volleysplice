# Side-switch T6 robust dominant-jersey consensus — 2026-08-25

## Decision and isolation

T6 addresses T5's measured failure: court-side temporal ownership improved visibility,
but averaging every retained jersey reduced team separation and made reliable swap
evidence negatively associated with true switches.

Preserve T4/T5 exactly for source frames, selective far crop, pinned `224 x 224`
detector and thresholds, court bounds, five endpoint fractions, jersey descriptor,
skin/background rejection, maximum three observations per side/frame, candidate rows,
and transport equations. Change only the temporal team descriptor: select a robust
dominant jersey mode and reject appearance outliers before aggregation.

No label, candidate, model, threshold, decoder, crop, detector, extra frame, optical
flow, or proxy-resolution change is part of T6.

## Frozen dominant-mode tracker

For every retained observation on one court side, keep its frame index and T5 weight
`max(confidence * support, 1e-6)`. Treat each observation descriptor as a possible
medoid. Its consensus members are all observations with Hellinger distance at most
`0.38` from that medoid.

Rank medoid candidates lexicographically by:

1. more distinct represented frames;
2. greater summed member weight;
3. lower member-weighted mean distance to the medoid; and
4. earlier frame/index for deterministic ties.

The winning mode must span at least two distinct frames. Otherwise the team is
unavailable. Aggregate only winning-mode members, first into per-frame descriptors and
then across frames using T5's frame reliability and temporal reliability equations.
Multiply final team reliability by `sqrt(mode_weight / all_observation_weight)`.
Cohesion is computed only over the selected per-frame mode descriptors. Diagnostics
must report mode observations, total observations, represented frames, support
fraction, rejected observations, and medoid score.

The three model values are:

1. `dominantJerseyTeamTransportSwapMargin`;
2. `dominantJerseyReliableSwapEvidence`; and
3. `dominantJerseyReliableContinuityEvidence`.

No T4 or T5 value enters the T6 head.

## Label-free engineering gates

Pin T5 features to SHA-256
`25d8a23563f803058f23ba2da23df0e82283ab893afd143b5f39200d96fcb460`.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t6-dominant-jersey-consensus-features-v1.json`.

T6 reaches labels only if every gate passes:

- exact preservation of 704 IDs/order and every T5 prior value;
- 624 boundaries and 80 ineligible internal rows;
- pooled both-team endpoint coverage at least T4's 74.17%, with no recording below
  35%;
- four-team boundary visibility at least T4's 58.33%;
- mean minimum team separation at least `0.3496`, a gain of at least 0.03 over T5;
- positive raw transport margins on at least 22.87% of boundaries, a gain of at least
  three percentage points over T5;
- nonzero reliable swap evidence on at least 20.79% of boundaries, a gain of at least
  three points over T5 and above T4;
- pooled mean dominant-mode support fraction at least 60%, with rejected observations
  nonzero so the representation is not an identity transform;
- all three core values finite/nonconstant and maximum absolute core/core,
  core/T4-core, core/T5-core, and core/existing correlation below `0.98`;
- exactly 635 endpoints, 3,175 frames, and 25,400 detector tile calls, no errors,
  at most 60 minutes, and at most 768 MiB RSS; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not change the radius, scoring order, minimum frames,
or reliability penalty after observing extraction.

## Conditional opened-development model

If engineering passes, compare exact boundary T0 with exact `34 + T6 core` under the
same nested recording-held-out protocol and immutable T4 benchmark. Require all prior
model guardrails, at least +2 F1 points over T0, at least +0.5 F1 point over T4,
recovery of a covered T0 miss, reduction of zero-reliable-swap T0 false boundaries,
and recording robustness. Additionally require the full reliable-swap coefficient to
be positive and positive in at least nine of 11 outer fits; T5 showed why semantic sign
stability is a necessary guardrail.

Report counts, P/R/F1, strict F1, row AP/Brier, recording deltas, coefficient ranks and
signs, and the diagnostic internal merge. Do not prune unless the full bundle passes.
Opened labels cannot authorize promotion or runtime work.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T6 code, feature artifact, feature values, model profile, or model result existed
when this contract was committed.
