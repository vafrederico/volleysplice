# Model iteration ranking metric

VolleyCut ranks rally-model iterations by **Padded P/Core R F1**, written
`F1_padP_coreR`. This is the hybrid F1 shown in the review timeline. It rewards an
export that covers the core rally action while measuring the footage cost against
the equally padded human export.

This metric is distinct from chronological event F1 at an IoU threshold and from
an F1 whose precision and recall both use the same reference ranges. Reports must
use the full name or the `F1_padP_coreR` key; an unlabeled `F1` is ambiguous.

## Definition

For each recording, let:

- `M` be the model's core predicted rally intervals.
- `H` be the core human-label intervals.
- `M+` be `M` after applying the configured before/after activity padding,
  clipping to the video bounds, merging overlapping or touching ranges, and joining
  consecutive ranges whose positive gap is strictly less than the configured short-gap
  threshold `G`.
- `H+` be `H` after applying the **same** padding, clipping, and merging rules.
- `|X|` be the union duration of interval set `X`, in seconds.

Before measuring any duration or intersection, subtract the label document's
`ignoredIntervals` from `M+`, `H`, and `H+`. Ignored time is outside evaluation:
model output there is neither a true positive nor a false positive, and ignored
seconds do not contribute to export-duration comparisons. Do not reinterpret
ignored time as dead-time negatives.

`G` defaults to **3 seconds**. The gap itself becomes retained export time when joined.
A gap of exactly `G` is not joined. Apply `G` identically to `M+` and `H+`, record it in
every evaluation artifact, and hold it fixed across a model comparison. Subtract ignored
time only after padding and short-gap joining; never rejoin the fragments produced by an
ignored interval.

The score uses padded-label precision and core-label recall:

```text
P_pad  = |M+ intersection H+| / |M+|
R_core = |M+ intersection H |  / |H|

F1_padP_coreR = 2 * P_pad * R_core / (P_pad + R_core)
```

If the model exports no time for a recording that contains labeled rally time,
precision, recall, and F1 are zero. Recordings without any core human-label time
are not valid inputs to a model-ranking comparison.

## Dataset-level aggregation

Rank model iterations on pooled live time, without allowing intervals to cross
recording boundaries. Sum the intersection numerators and duration denominators
over the exact same recordings before calculating the two ratios:

```text
P_pad  = sum_i |M_i+ intersection H_i+| / sum_i |M_i+|
R_core = sum_i |M_i+ intersection H_i|  / sum_i |H_i|
```

Then calculate `F1_padP_coreR` from those pooled `P_pad` and `R_core` values. The
per-video values in the review UI are diagnostics; do not average their F1 values
to produce the canonical dataset ranking.

## Required padding sweep

Every model-iteration evaluation must measure and report all four symmetric
before/after padding cases:

| Case | Before padding | After padding |
| --- | ---: | ---: |
| `pad-0s` | 0 seconds | 0 seconds |
| `pad-1s` | 1 second | 1 second |
| `pad-2s` | 2 seconds | 2 seconds |
| `pad-3s` | 3 seconds | 3 seconds |

For each case, apply that case's padding to both `M` and `H` to form `M+` and
`H+`, then independently report pooled `P_pad`, pooled `R_core`,
`F1_padP_coreR`, padded model export duration, padded human export duration, and
the model export-duration difference from padded human. The zero-second case is
still evaluated through the same clipping, union, and merge pipeline.

These four results are a required sensitivity profile, not four opportunities to
pick the most favorable score. Declare the target product padding before comparing
iterations and use that case for the primary descending rank. Report the other
three beside it. Changing the target padding is a separate, validation-only product
decision and must not be selected per model or from protected-test performance.

## Ranking contract

- Rank comparable model iterations by `F1_padP_coreR` descending.
- During iteration, rank on the declared development or validation scope only.
  Keep the protected held-out test split closed until the predeclared final gate.
- Use identical recording/source-group scope, gold-label revision, before/after
  padding, video-bound clipping, and interval-merging semantics for every model in
  a comparison, including the same short-gap join threshold (canonical default: 3
  seconds, with only gaps strictly below the threshold joined).
- Use the exact same `ignoredIntervals` revision for every model and subtract those
  ranges before calculating metric numerators, denominators, or export duration.
- Report `P_pad`, `R_core`, `F1_padP_coreR`, padded model export duration, and
  padded human export duration together, along with the short-gap join threshold.
  The component values explain whether a rank change came from footage cost or core
  coverage.
- Always report those values for `pad-0s`, `pad-1s`, `pad-2s`, and `pad-3s`, where
  the named duration is applied equally before and after every model and human
  interval. Rank by the predeclared target-padding case; never cherry-pick the best
  padding case for each model.
- Continue reporting event F1, boundary errors, outcome slices, and other
  predeclared guardrails. They can block promotion, but they do not replace
  `F1_padP_coreR` as the primary ordering metric for rally-model iterations.
- Never select a model, decoder, threshold, padding value, epoch, or seed using a
  protected-test `F1_padP_coreR` result.
