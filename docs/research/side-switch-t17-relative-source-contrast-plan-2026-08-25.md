# Side-switch T17 relative source assignment contrast — 2026-08-25

## Decision and isolation

T16 finds a stable near-source coefficient but an unstable, nearly null far-source
coefficient. T17 tests whether absolute Hellinger cost scale differs enough across
camera-relative sources to make the raw far advantage incomparable.

Use only the immutable T16 artifact SHA-256
`11bcedafe98284bd306398f44946720767a5323275aea38a7948ae43d7d01692`.
For each T14 reduction pair define:

```text
relative source contrast = (same cost - swapped cost) / (same cost + swapped cost)
```

Use exact zero only when the denominator is zero. Apply independently to `nearNear`
versus `nearFar` and `farFar` versus `farNear`. The model inputs are exactly:

- `relativeMedoidJerseyNearSwapContrast`
- `relativeMedoidJerseyFarSwapContrast`

Internal candidates receive zero. Do not clip, center, weight, combine, or drop a
source; do not decode video or load labels during transformation.

## Engineering gates

Require exact 704-row/prior parity, 624 eligible and 80 zero-filled internal rows,
formula reconstruction within `1e-12`, finite/nonconstant cores, exact sign agreement
with the corresponding T16 raw advantage, at least 10% positive and 10% negative per
source, mutual absolute Spearman below `0.95`, corresponding T16 absolute Spearman
below `0.98` (proving normalization changes ordering), all other prior-family
correlations below `0.98`, <=60 seconds, and <=2,048 MiB RSS.

Failure stops before labels. Do not replace the denominator or relax novelty after
observing T17.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T17 core` wiring
before loading labels. Require +2 F1 points over T0, +0.5 point over immutable T14,
all prior event/slice/recording guardrails, and both normalized-source coefficients
positive in the full fit and at least 9/11 outer fits.

This is adaptive opened-development work. It cannot authorize promotion, runtime
changes, or source pruning even if it passes.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T17 code, values, artifact, profile, or model result existed at this commit.
