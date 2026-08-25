# Side-switch T20 compact medoid/disagreement candidate — 2026-08-25

## Purpose and adaptive provenance

After T19 closed the representation-search loop, the user explicitly requested that
the most promising T14/T19 compact candidate be preregistered and executed now. T20
is therefore an intentionally adaptive assembly for future held-recording validation,
not a new independent hypothesis test on the 11 opened recordings.

T14 showed that the stable-medoid raw signed margin is positive in all 11 outer fits
and produces the strongest opened boundary F1. T19 showed that cross-representation
disagreement has the intended negative sign in all 11 fits. T20 freezes exactly those
two physically interpretable values and excludes T14's contradicted reliability
products and T19's weaker equal-weight consensus.

This does not revise the T14 or T19 decisions. The feature choice itself used their
opened results, so current-data metrics can diagnose implementation and whether the
assembly merits future validation but cannot establish generalization.

## Exact feature contract

Use only immutable T19 feature artifact SHA-256
`ab4b474fdbaee20bf884093a37eba25e79b6e93172f0dbfe619c3b0546c421e5`.
Copy, without recomputation:

```text
compactMedoidJerseySwapMargin = dominantTrackletJerseyTeamTransportSwapMargin
compactCrossRepresentationJerseyDisagreement = crossRepresentationJerseySwapDisagreement
```

The model signature is exactly the production 34 inputs followed by those two values.
Do not include T14 reliable swap/continuity, T19 consensus, T4 raw margin, interactions,
quality gates, source-specific values, or any other T3–T19 diagnostic. Internal rows
receive exact zero. No video decode or detector work is required.

## Engineering gates

Before T20 model wiring, require:

- exact 704-row order and all prior-feature values from T19; exact 624 boundaries and
  80 zero-filled internal rows;
- bit-exact copy of both source values on every eligible row;
- finite/nonconstant cores;
- positive raw-margin fraction exactly equal to T14's `206/624`, and nonzero
  disagreement fraction exactly equal to T19's `617/624`;
- <=60 seconds, <=2,048 MiB RSS, and no video, detector, label, audit, feedback, or
  model artifact loaded by the transformation.

Failure stops before the adaptive model diagnostic. Do not add, remove, rescale, or
interact features after observing T20.

## Opened-development diagnostic

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T20 core` wiring
before loading labels. Keep the existing nested recording-LOO objective, threshold
selection, and decoder. Report immutable T14 and T19 comparisons and all prior event,
slice, and recording diagnostics.

The declared development screen requires:

- boundary F1 at least +2 points over T0, at least +0.5 point over T14, and no lower
  than T19;
- precision and recall decline no more than 2 points and strict F1 decline no more
  than 1 point;
- recovery of at least one T0 miss with positive compact medoid margin, reduction of
  T0-selected false boundaries whose T4/T14 directions disagree or include zero, and
  the existing recording-robustness rule;
- compact medoid coefficient positive in the full fit and at least 9/11 outer fits;
  and compact disagreement coefficient negative in the full fit and at least 9/11.

Passing this screen only freezes a candidate for evaluation on new recording-held
gold. It cannot promote the model, retroactively validate feature selection, or
authorize runtime changes.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T20-specific feature names, transformation, profile, artifact, model result, or
metric existed when this contract was committed. All T14/T19 source values and opened
results were already known and are explicit adaptive provenance.
