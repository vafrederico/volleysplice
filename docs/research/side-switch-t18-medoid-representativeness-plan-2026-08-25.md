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
