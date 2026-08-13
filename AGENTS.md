<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

## Rally-model iteration ranking

When comparing or selecting rally-model, decoder, threshold, padding, epoch, or seed
iterations, use **Padded P/Core R F1** (`F1_padP_coreR`) as the primary descending
ranking metric:

```text
P_pad  = duration(padded_model intersection padded_human) / duration(padded_model)
R_core = duration(padded_model intersection core_human)   / duration(core_human)
F1_padP_coreR = 2 * P_pad * R_core / (P_pad + R_core)
```

Apply identical before/after padding to model and padded-human ranges, clip to video
bounds, and merge overlapping or touching ranges before measuring union duration.
For a dataset ranking, pool intersection numerators and duration denominators across
recordings before calculating F1; do not average per-video F1 values. Compare only the
same recording/source-group scope, gold-label revision, and padding configuration.

Always evaluate and report the four symmetric padding cases `(before, after) =
(0, 0), (1, 1), (2, 2), (3, 3)` seconds. For every case, report pooled `P_pad`,
pooled `R_core`, `F1_padP_coreR`, padded model export duration, padded human export
duration, and their duration difference. Declare the target product padding before
the comparison and rank by that case. Never choose each model's best padding case;
the remaining three cases are required sensitivity results.

Treat label-document `ignoredIntervals` as outside the evaluation universe. Subtract
them from padded model, core human, and padded human interval unions before computing
metric numerators, denominators, or export-duration comparisons. Model predictions in
ignored time are neither true nor false positives. Never convert ignored spans into
dead-time negatives; `hardNegatives` are the separate construct for valid confusing
dead time. All compared models must use the same ignored-range revision.

Rank iterations on the declared development/validation scope. Never use the protected
test split to select an iteration. Continue to report event F1 and all predeclared
guardrails, but do not substitute event F1 for this ranking metric. Use the unambiguous
name `F1_padP_coreR` in artifacts and reports. See
[`docs/model-ranking-metric.md`](docs/model-ranking-metric.md) for the canonical contract.
