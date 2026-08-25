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

### Engineering result — pass; adaptive model diagnostic authorized

Implementation is committed at `a9ac52d`. The no-video transformation wrote the
40,795,701-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t20-compact-medoid-disagreement-features-v1.json`

SHA-256: `f770eed53f85dd79f398a23e37d0706e80925971a300538082088ad59e0d5c1e`.

All 704 rows and prior values match exactly. Both source copies are bit-exact, all 80
internal rows are zero, positive medoid margins equal T14 at 206/624, and nonzero
disagreement equals T19 at 617/624. Transformation time is 0.319 seconds at 143.563
MiB RSS.

Decision: **T20 passes engineering**. Commit the exact 36-input runner before loading
labels for the declared adaptive diagnostic.

### Adaptive opened-development diagnostic — screen fail

The exact 36-input runner was committed at `38e006a` before the one-shot diagnostic.
It wrote the 943,228-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t20-compact-medoid-disagreement-opened-v1.json`

SHA-256: `b8f4bdb45d8f3c7fc679efbb372c6145a1e00e4b32c3783cecd5e901adaa0e9b`.

| Boundary-only metric | T0 | T20 | Change |
| --- | ---: | ---: | ---: |
| Row AP | 43.5623% | 43.5318% | −0.0306 pp |
| ±4 proposals / TP / FP / FN | 52 / 26 / 26 / 24 | 54 / 28 / 26 / 22 | +2 / +2 / 0 / −2 |
| ±4 precision | 50.00% | 51.8519% | +1.8519 pp |
| ±4 recall | 52.00% | 56.00% | +4.00 pp |
| ±4 F1 | 50.9804% | 53.8462% | +2.8658 pp |
| Strict F1 | 41.1765% | 42.3077% | +1.1312 pp |

T20 is 0.3808 point above T19 but 1.3919 points below T14. It recovers one covered
T0 miss, but retains all seven frozen T4/T14 direction-disagreement false boundaries.
The compact medoid coefficient is positive (`+0.096130`) in the full fit and 11/11
outer fits; compact disagreement is negative (`-0.075071`) in the full fit and 11/11.

Decision: **the adaptive development screen fails** on the T14 comparator and false
slice. The candidate is fully implemented and semantically stable, but it does not
advance to promotion or receive priority for new-recording validation under the
declared gate. Do not retune the feature set, disagreement weight, threshold, or
decoder on these 11 recordings.
