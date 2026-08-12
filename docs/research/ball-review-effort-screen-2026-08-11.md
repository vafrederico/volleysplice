# Ball-review model and effort screen — 2026-08-11

## Decision

Lower Sol reasoning effort clearly reduced latency in this bounded screen.
Use **Sol high** for the next temporally contextual confirmation pass: it was
1.73x faster than fresh Sol xhigh and had the strongest conservative mix of
state, presence, and box agreement. **Sol low** is the aggressive proposal-only
candidate at 5.22x the xhigh throughput, but is not yet a safe replacement for
final labels.

Do not use Luna xhigh/max for this job: both were slower than Sol high without a
corresponding agreement advantage. Terra xhigh was fast and reasonably
consistent on state/presence, but its box localization was unstable. No max
configuration justified its latency.

This is a screening decision, not an accuracy claim. The comparison target is
an earlier blind Sol-xhigh review rather than human truth, and fresh Sol xhigh
itself reproduced that target poorly. Official OpenAI guidance likewise
recommends benchmarking representative workloads instead of assuming the
highest reasoning effort is best: <https://developers.openai.com/api/docs/guides/latest-model>.

## Protocol

- Twelve previously reviewed frames were selected across all eight development
  recordings and all three environments (beach, grass, indoor). The bounded
  sample deliberately contains all four prior state labels and difficult tiny,
  blurred, occluded, truncated, and multiple-ball examples; it is not a random
  prevalence sample.
- Every new run received the same 12 opaque, SHA-pinned 960x540 images, prompt,
  JSON schema, order, and default service tier in an isolated ephemeral Codex
  session. It could not see the prior annotations, detector suggestions,
  recording identity, or source metadata.
- Configurations were Sol low/medium/high/xhigh/max and Terra/Luna xhigh/max.
  Runs were sequential and each configuration ran once. Wall time covers the
  complete Codex subprocess through valid result creation.
- Prior Sol annotations were loaded only after all requested configurations had
  completed. Similarity to that fixed pseudo-reference excludes its
  `indeterminate` frames from presence and box metrics. The separate pairwise
  matrix is symmetric, uses every frame for presence/box comparison, and
  averages state F1 only over states present in either run.
- The new runs saw independent still images, not adjacent temporal frames. This
  materially changes whether a non-visible ball can be called occluded or out
  of frame.

## Timing result

| Configuration | Wall time | Seconds/frame | Speed vs Sol xhigh | Reasoning tokens |
|---|---:|---:|---:|---:|
| Sol low | 64.8 s | 5.40 | 5.22x | 1,355 |
| Sol medium | 98.7 s | 8.23 | 3.43x | 3,186 |
| Terra xhigh | 159.9 s | 13.32 | 2.12x | 7,318 |
| Sol high | 196.0 s | 16.33 | 1.73x | 6,035 |
| Luna xhigh | 264.4 s | 22.03 | 1.28x | 13,338 |
| Terra max | 313.4 s | 26.12 | 1.08x | 14,890 |
| Sol xhigh | 338.3 s | 28.19 | 1.00x | 15,010 |
| Luna max | 456.4 s | 38.03 | 0.74x | 23,546 |
| Sol max | 553.0 s | 46.09 | 0.61x | 25,898 |

All nine jobs exited zero, produced complete schema-valid artifacts, and had no
recorded transport error. Luna max had already completed before cancellation
was requested, so its valid result is retained. Wall time and reasoning-token
count had Pearson correlation 0.989 in this run, supporting a real effort-cost
effect rather than image handling alone.

These are single observations, not stable latency estimates. Scheduler
variance is unmeasured, and the full-corpus Sol review was also active during
the sequential screen.

## Similarity to the prior blind Sol-xhigh pseudo-reference

| Configuration | State accuracy | State macro-F1 | Presence F1 | Box F1 @ .25 | Box F1 @ .50 |
|---|---:|---:|---:|---:|---:|
| Sol low | .583 | .206 | .875 | .375 | .250 |
| Sol medium | .583 | .300 | .800 | .133 | .000 |
| Sol high | .667 | .344 | .875 | .375 | .125 |
| Sol xhigh (fresh) | .417 | .225 | .615 | .462 | .154 |
| Sol max | .500 | .262 | .714 | .571 | .143 |
| Terra xhigh | .667 | .373 | .824 | .118 | .118 |
| Terra max | .667 | .373 | .824 | .118 | .000 |
| Luna xhigh | .500 | .267 | .667 | .000 | .000 |
| Luna max | .417 | .225 | .615 | .154 | .000 |

Higher effort was not monotonically more similar to the prior output. Most
importantly, fresh Sol xhigh matched only 5/12 prior states. Every fresh run used
only `localizable` or `indeterminate`, while the temporally informed prior
review also used `fully_occluded` and `out_of_frame`. The pseudo-reference is
therefore useful for finding gross disagreement, but cannot rank correctness.

## Run-to-run patterns

Against fresh Sol xhigh using corrected symmetric metrics:

| Candidate | State agreement | Active-state macro-F1 | Presence F1 | Box F1 @ .25 | Box F1 @ .50 |
|---|---:|---:|---:|---:|---:|
| Sol low | .667 | .625 | .750 | .625 | .500 |
| Sol medium | .833 | .829 | .857 | .571 | .571 |
| Sol high | .750 | .733 | .800 | .800 | .667 |
| Sol max | .917 | .916 | .923 | .769 | .615 |
| Terra xhigh | .667 | .625 | .750 | .250 | .250 |
| Terra max | .667 | .625 | .750 | .500 | .375 |
| Luna xhigh | .833 | .829 | .857 | .286 | .286 |
| Luna max | .833 | .833 | .833 | .500 | .500 |

Sol low, medium, and high form a strong lower-effort state cluster: low/high
and medium/high agree on 11/12 states. Sol high has the best lower-effort box
agreement with fresh xhigh. Terra xhigh and max agree on every state but only
reach box F1 .20 with each other at IoU .25/.50, exposing substantial
localization instability.

## Next confirmation

Before changing the exact-frame blind-labeling default, repeat Sol high and Sol
low on complete 3-second windows with ordered neighboring frames and compare
against human-adjudicated truth. Use whole-window paired timing, at least two
source groups/environments, repeated runs, and report schema failures/timeouts.
Sol low can meanwhile be used as a fast proposal layer provided a human still
finalizes the label; Sol high is the preferred candidate for a smaller
validation-quality pass.

## Artifacts

- Reusable harness: `scripts/benchmark-ball-review-effort.py`
- Visual comparison UI: `/label/ball/benchmark` in the local VolleyCut app.
- External report:
  `/mnt/freenas/volleycut/ball-presence-v1/reports/ball-review-effort-screen12-v1/report.json`
- Corrected report SHA-256:
  `818641939aa3f8e50d901933c69ad4229ea4146aefc1bce480740c93b3062638`
- Sealed pseudo-reference SHA-256:
  `3db94339ae2e5062dd310e530dd6108aa0748b33da722910a2e04c80809cdb51`

The harness refuses to reuse future cached runs unless result, prompt, schema,
ordered images, model/effort, service tier, Codex binary, and CLI version match.
These nine completed receipts predate that full execution-provenance field, so
the corrected report instead verifies their declared identities and pins every
result, receipt, event log, and stderr artifact hash. `--score-existing` exists
only to recompute metrics over such a completed, separately audited artifact
set without paying for model execution again; it is not equivalent to the
future fail-closed cache check.

The comparison UI reads only the frozen benchmark pack. It validates the
compiled report SHA, every report-pinned result and receipt, exact frame
coverage, normalized annotation semantics, and every displayed PNG hash before
serving a sanitized bundle. It exposes no detector output, human-review draft,
event log, stderr, source path, or generic filesystem route. The default 9-up
view places all configurations on the same frame; A/B, multi-layer overlay,
metrics/scatter, and numeric pairwise heatmap views make localization and state
differences inspectable without treating the pseudo-reference as truth.
