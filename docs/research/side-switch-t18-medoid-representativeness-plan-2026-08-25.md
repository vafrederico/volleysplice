# Side-switch T18 medoid representativeness transport — 2026-08-25

## Decision and isolation

T14 proves that a single medoid direction is useful, while T16/T17 rule out treating
the camera-relative sources as two equally transferable raw or scale-normalized
inputs. T18 tests a different explanation: the selected tracklet can be a stable
player yet poorly represent the pooled team appearance because it captures a distinct
role or jersey.

Use only T17 artifact SHA-256
`200ac7445e6e8b8b64b97aa07b25164761da3526a59a2eebca2e0b7dae5c35dd`.
For each of the four T14 endpoint/side selections define representativeness as
`1 - medoidToPooledDistance` for a stable medoid, exact `1` for a pooled fallback, and
exact `0` for unavailable. Clip only numerical distance spill to `[0,1]`. Define the
boundary gate as the minimum of the four values, then:

```text
representative medoid swap margin = T14 raw signed swap margin × boundary gate
```

The sole model input is `representativeMedoidJerseySwapMargin`. Store the gate and
maximum medoid-to-pool distance as diagnostics. Internal rows receive exact zero. Do
not change medoid selection, use labels, decode video, add a radius, or add the raw
T14 margin as a second input.

## Engineering gates

Require exact 704-row/prior parity, 624 eligible and 80 zero-filled internal rows,
formula reconstruction within `1e-12`, finite/nonconstant output, gate within `[0,1]`,
gate strictly below `0.99` on at least 20% and positive on at least 50% of eligible
rows, nonzero transformed margin on at least 15%, exact sign agreement with nonzero
T14 raw margins whenever the gate is positive, absolute correlation with T14 raw
margin and all other prior inputs below `0.98`, <=60 seconds, and <=2,048 MiB RSS.

Failure stops before labels. Do not replace minimum with mean, adjust fallback values,
or tune the gate after observing T18.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T18 one-value
core` before loading labels. Require +2 F1 points over T0, +0.5 point over immutable
T14, all prior precision/recall/strict/covered-miss/recording guardrails, reduction of
T0 false boundaries in the frozen bottom-quartile representativeness slice, and a
positive T18 coefficient in the full fit and at least 9/11 outer fits.

This remains adaptive opened-development work and cannot authorize promotion or
runtime changes.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T18 code, values, artifact, profile, or result existed at this commit.

### Engineering result — pass; model comparison authorized

Implementation is committed at `040a2eb`. The transformation wrote the 40,357,628-byte
artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t18-medoid-representativeness-features-v1.json`

SHA-256: `0efab13db25849a74413fc595e2753c735d1c9eb00ae4afaf8b73b9007ef8949`.

The gate is positive on 68.75% of boundaries, below `0.99` on 100%, and has mean
`0.469657`. The transformed margin is nonzero on 68.75%; its strongest T14
relationship has magnitude `0.917324`. Every frozen parity, activity, sign, novelty,
runtime, and memory gate passes.

Decision: **T18 passes engineering**. Only the exact 35-input model wiring is now
authorized before label access.

### Matched model result — reject

The exact 35-input wiring was committed at `3447dd8`. The one-shot model wrote the
942,641-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t18-medoid-representativeness-opened-v1.json`

SHA-256: `488b20863cca787d54e973e4f106ad98e6314418a151edffeaf768213492b7dd`.

The coefficient is positive in the full fit (`+0.087936`) and 10/11 folds, but event
behavior fails: boundary F1 is effectively unchanged at 50.94% (−0.04 point), strict
F1 falls 1.55 points, T18 recovers none of five covered misses, and retains all nine
bottom-quartile representativeness false boundaries. It trails T14 by 4.29 points.

Decision: **reject T18**. Medoid-to-pooled distance is associated with the fitted
score but does not provide the intended event-level quality control. Do not tune the
gate, quartile, or aggregation.
