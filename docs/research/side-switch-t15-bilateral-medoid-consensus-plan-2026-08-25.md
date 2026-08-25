# Side-switch T15 bilateral medoid consensus — 2026-08-25

## Decision and isolation

T14's single-medoid raw transport margin is positive in all 11 model fits, but its
inherited reliability products are negative in all 11. T15 tests a different failure
mode: T14 averages the near-source and far-source assignment advantages, allowing one
strong side to hide disagreement from the other.

Use the immutable T14 feature artifact SHA-256
`d68e0d9a2aa72e1fe3cb06f413fdb43dacd52b9090dd4241863596459f8b9753`
as the sole row and value source. Do not decode video, rerun detection, change the
medoid/fallback representation, load labels during transformation, or reuse T14's
reliable-evidence products.

For each eligible boundary, read the exact T14 one-unit reduction costs and define:

```text
near advantage = cost(near→near) - cost(near→far)
far advantage  = cost(far→far)   - cost(far→near)
bilateral swap evidence       = max(min(near advantage, far advantage), 0)
bilateral continuity evidence = max(min(-near advantage, -far advantage), 0)
```

The two model inputs are exactly:

- `bilateralMedoidJerseySwapEvidence`
- `bilateralMedoidJerseyContinuityEvidence`

Store both signed source advantages as diagnostics only. Internal candidates receive
exact zeros for every T15 value.

## Engineering gates

Before any T15 label/model access, require:

- exact 704-row order and prior-feature parity with T14; exact 624 eligible boundaries
  and 80 zero-filled internal rows;
- formula reconstruction exact within `1e-12` for all eligible rows;
- both core values finite and nonconstant;
- nonzero bilateral swap and bilateral continuity on at least 10% of eligible rows;
- maximum absolute Spearman correlation between the two T15 core values, and between
  either core value and every T4–T14/existing input, below `0.98`; and
- transformation wall time <=60 seconds and peak RSS <=2,048 MiB.

Failure stops before a T15 model run. Do not change the bilateral minimum to a mean,
quantile, soft minimum, or one-side fallback after observing the values.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T15 two-value
core` wiring before loading labels. Use the unchanged nested recording-LOO objective,
threshold selection, decoder, and immutable T4/T14 comparators.

Require all of the following:

- boundary F1 at least +2 points over T0 and +0.5 point over T14;
- precision and recall decline no more than 2 points, strict F1 decline no more than
  1 point, at least one covered T0 miss with nonzero bilateral swap evidence
  recovered, T0 false boundaries with nonzero bilateral continuity evidence reduced,
  and the existing recording-robustness rule;
- bilateral swap coefficient positive in the full fit and at least 9/11 outer fits;
  and
- bilateral continuity coefficient negative in the full fit and at least 9/11 outer
  fits.

This is an adaptive opened-development follow-up motivated by T14. Even a pass cannot
authorize promotion, runtime work, or further within-T15 formula selection.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T15 code, feature value, artifact, or model result existed when this contract was
committed. T14's opened result was used only to choose the bilateral-consensus
hypothesis and is explicitly part of this experiment's adaptive provenance.
