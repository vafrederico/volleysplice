# Side-switch T16 source-resolved medoid transport — 2026-08-25

## Decision and isolation

T15 shows that 451/624 boundaries have the near-source and far-source one-medoid
advantages disagree or tie. T14 averages those values and T15 discards their
disagreement. T16 instead preserves the two signed source advantages as separate
model inputs, testing whether the fixed camera-relative sources have independently
transferable direction.

Use only the immutable T15 artifact SHA-256
`a5ce28158177b33d9e1864c6b94a03762fa79b394e344253815474eef039b497`.
Rename, without recomputation, its exact diagnostic source advantages to:

- `sourceResolvedMedoidJerseyNearSwapAdvantage`
- `sourceResolvedMedoidJerseyFarSwapAdvantage`

Internal candidates receive exact zeros. Do not average, clip, gate, normalize,
interact, swap camera roles, decode video, or load labels during transformation.

## Engineering gates

Require exact 704-row/prior parity, 624 eligible and 80 zero-filled internal rows,
bit-exact equality to the T15 diagnostic values, finite/nonconstant cores, at least
10% positive and 10% negative values for each source, at least 25% source-sign
disagreement/ties, absolute near/far Spearman below `0.95`, and maximum absolute
core/T4–T15/existing Spearman below `0.98`. Require <=60 seconds and <=2,048 MiB RSS.

Failure stops before model access. Do not exchange near/far definitions or add a
learned source weight after observing engineering values.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T16 two-value
core` wiring before loading labels. Keep the frozen nested recording-LOO model,
threshold, and decoder. Require:

- boundary F1 at least +2 points over T0 and +0.5 point over T14;
- all prior precision, recall, strict-F1, covered-miss, false-boundary-slice, and
  recording-robustness guardrails; and
- both source coefficients positive in the full fit and in at least 9/11 outer fits.

The false-boundary slice is the T0-selected false set whose two source advantages
disagree in sign or include a zero; it must shrink. Even a pass is adaptive
opened-development evidence only and cannot authorize promotion or runtime work.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T16 code, feature artifact, model profile, or result existed when this contract was
committed. T14/T15 opened-development outcomes are explicit hypothesis provenance.
