# All-model `F1_padP_coreR` ranking and rally-score export sweep — 2026-08-13

## Outcome

On the two-recording validation source group, with the already-declared product target of one
second before and after every rally, the best unfiltered model is
`full-audiovisual-serve-prob-stack-v1-exploratory [418babe628f9]`:

- `P_pad = 71.69%`
- `R_core = 90.09%`
- `F1_padP_coreR = 79.85%`
- padded model export = 697.0 seconds
- padded human export = 602.1 seconds
- export-duration difference = +94.9 seconds
- unpadded event F1 guardrail = 65.84%

The runner-up, `full-audiovisual-audio-normalized-v3`, scores 79.36%, 0.48 percentage points
behind. No tested common score cutoff beats the winning unfiltered model at the fixed 1-second
product padding.

The complete immutable JSON report is
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/all-models-f1-padp-corer-score-thresholds-v6.json`
(SHA-256 `323ab16cdc5a7f76d8dd71f9c7e309b59c722e52b271716da4b7b5d1bc8f3e40`). It contains all 32
models, all eight score-filter states, all four padding cases, pooled metric numerators and
denominators, per-recording components, duration comparisons, event/boundary/outcome guardrails,
input hashes, model hashes, analysis hashes, and rankings for every fixed threshold/padding pair.

## Scope and label revision

Selection uses only the frozen manifest's two validation recordings, both from
`shoreline-kob-20250616`. The protected test prediction was not evaluated or used for any choice.
The manifest supplies rally gold and split assignments. The evaluator overlays `ignoredIntervals`
from the current human label documents and records each document's SHA-256.

The updated drafts do contain ignored spans in the indoor recordings: 30.341474 seconds and
13.576348 seconds at the beginnings of the two indoor training videos, and 114.134 seconds at the
beginning of the protected indoor test video. Neither validation document has an ignored span, and
both validation rally lists still exactly match the frozen manifest. The new ignored revision
therefore has zero numerical effect on this validation ranking, but it is now part of the evaluator
contract and provenance.

Rally `confidence` is treated as a model-specific, uncalibrated score. A cutoff of 0.80 is not an
80% correctness probability, and a common numeric cutoff is not guaranteed to represent the same
operating point across model families.

## Primary all-model ranking

This is the required descending ranking at the fixed 1-second product padding with no additional
score filter. `vs human` compares padded model export duration with the equally padded human export
of 602.1 seconds. Event F1 is the unpadded IoU-0.5 guardrail, not the ranking metric.

| Rank | Unique model/output variant | `P_pad` | `R_core` | `F1_padP_coreR` | Model export | vs human | Event F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `full-audiovisual-serve-prob-stack-v1-exploratory [418babe628f9]` | 71.69% | 90.09% | **79.85%** | 697.0s | +94.9s | 65.84% |
| 2 | `full-audiovisual-audio-normalized-v3` | 71.44% | 89.26% | **79.36%** | 690.2s | +88.1s | 65.41% |
| 3 | `pair-serve-audio-v7-new-only` | 70.83% | 89.34% | **79.01%** | 698.0s | +95.9s | 65.00% |
| 4 | `full-audiovisual-serve-peak-window-v1-exploratory [e4567657d7bf]` | 66.88% | 95.21% | **78.57%** | 803.8s | +201.7s | 65.88% |
| 5 | `pair-serve-audio-v6-no-legacy` | 64.50% | 94.30% | **76.60%** | 838.3s | +236.2s | 61.29% |
| 6 | `dead-state-transition-audio-normalized-v1-full [8a3690a9134e]` | 64.25% | 93.45% | **76.15%** | 831.8s | +229.6s | 61.29% |
| 7 | `dead-state-transition-audio-normalized-v4-full-final [cbb1125014c1]` | 64.25% | 93.45% | **76.15%** | 831.8s | +229.6s | 61.29% |
| 8 | `dead-ball-specialist-audio-normalized-v3 [f2d3908c2d28]` | 63.95% | 93.60% | **75.98%** | 836.8s | +234.7s | 61.29% |
| 9 | `dead-state-global-audio-normalized-v1-full-final [9d451bd6ddb5]` | 63.67% | 94.17% | **75.97%** | 847.2s | +245.0s | 61.29% |
| 10 | `dead-ball-specialist-audio-normalized-v2-legacy-only [735df75dd1e1]` | 63.88% | 93.71% | **75.97%** | 839.0s | +236.8s | 61.62% |
| 11 | `dead-ball-specialist-audio-normalized-v4-no-legacy [53aee0feac37]` | 63.92% | 93.60% | **75.97%** | 839.3s | +237.2s | 61.29% |
| 12 | `dead-state-transition-audio-normalized-v0-legacy-only [23e961e96bcd]` | 63.85% | 93.66% | **75.94%** | 838.8s | +236.7s | 63.44% |
| 13 | `dead-state-transition-audio-normalized-v3-legacy-only-final [a53ca5089099]` | 63.85% | 93.66% | **75.94%** | 838.8s | +236.7s | 63.44% |
| 14 | `full-audiovisual-serve-prob-stack-v1-control` | 62.68% | 96.29% | **75.93%** | 882.3s | +280.2s | 63.41% |
| 15 | `full-audiovisual-v2` | 62.68% | 96.29% | **75.93%** | 882.3s | +280.2s | 63.41% |
| 16 | `full-audiovisual-v2-final` | 62.68% | 96.29% | **75.93%** | 882.3s | +280.2s | 63.41% |
| 17 | `dead-ball-specialist-audio-normalized-v5-new-only [09d68be8f17a]` | 63.82% | 93.71% | **75.93%** | 839.7s | +237.5s | 61.29% |
| 18 | `pair-serve-audio-v5` | 63.82% | 93.71% | **75.93%** | 839.7s | +237.5s | 61.29% |
| 19 | `dead-state-transition-audio-normalized-v2-no-legacy [5a2a47ee603c]` | 63.55% | 94.13% | **75.88%** | 852.6s | +250.4s | 62.37% |
| 20 | `dead-state-transition-audio-normalized-v5-no-legacy-final [f70f934a2775]` | 63.55% | 94.13% | **75.88%** | 852.6s | +250.4s | 62.37% |
| 21 | `audiovisual-v2-targeted-pruned-diagnostic` | 64.34% | 92.25% | **75.81%** | 818.5s | +216.4s | 62.96% |
| 22 | `dead-state-global-audio-normalized-v2-full-final [bba9eec38729]` | 66.07% | 88.83% | **75.78%** | 749.2s | +147.0s | 58.23% |
| 23 | `dual-serve-v4-v5-fusion-v1` | 60.60% | 96.67% | **74.50%** | 928.6s | +326.5s | 62.77% |
| 24 | `dead-ball-specialist-audiovisual-v2 [de0ee020eb1b]` | 58.36% | 98.59% | **73.32%** | 987.4s | +385.3s | 59.57% |
| 25 | `dead-ball-specialist-audiovisual-v1 [8a27d4f7ca39]` | 58.33% | 98.59% | **73.30%** | 987.9s | +385.8s | 59.57% |
| 26 | `pair-serve-audiovisual-v3` | 58.33% | 98.59% | **73.30%** | 987.9s | +385.8s | 59.57% |
| 27 | `pair-serve-audiovisual-v4` | 58.33% | 98.59% | **73.30%** | 987.9s | +385.8s | 59.57% |
| 28 | `full-percentile-v1` | 58.26% | 94.51% | **72.08%** | 931.5s | +329.4s | 56.80% |
| 29 | `pair-serve-visual-v1` | 57.37% | 96.39% | **71.92%** | 974.4s | +372.3s | 59.09% |
| 30 | `pair-serve-visual-v2` | 57.37% | 96.39% | **71.92%** | 974.4s | +372.3s | 59.09% |
| 31 | `pilot-percentile-v1` | 53.89% | 72.24% | **61.73%** | 749.8s | +147.7s | 37.74% |
| 32 | `pilot-baseline-v0` | 47.45% | 87.97% | **61.65%** | 1076.2s | +474.1s | 34.02% |

Several tied rows are genuinely identical at the unfiltered operating point. They remain separate
because their model identities or score semantics differ, which can make their filtered results
diverge.

## Padding sensitivity

The following table reports unfiltered `F1_padP_coreR` for all required symmetric padding cases.
It is a sensitivity profile, not permission to choose a different best padding per model.

| Rank @ 1s | Unique model/output variant | pad-0s | pad-1s target | pad-2s | pad-3s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `full-audiovisual-serve-prob-stack-v1-exploratory [418babe628f9]` | 72.34% | **79.85%** | 83.72% | 86.42% |
| 2 | `full-audiovisual-audio-normalized-v3` | 72.42% | **79.36%** | 83.08% | 85.98% |
| 3 | `pair-serve-audio-v7-new-only` | 72.26% | **79.01%** | 82.64% | 85.67% |
| 4 | `full-audiovisual-serve-peak-window-v1-exploratory [e4567657d7bf]` | 72.64% | **78.57%** | 81.75% | 84.66% |
| 5 | `pair-serve-audio-v6-no-legacy` | 71.57% | **76.60%** | 79.17% | 81.79% |
| 6 | `dead-state-transition-audio-normalized-v1-full [8a3690a9134e]` | 71.26% | **76.15%** | 78.66% | 81.27% |
| 7 | `dead-state-transition-audio-normalized-v4-full-final [cbb1125014c1]` | 71.26% | **76.15%** | 78.66% | 81.27% |
| 8 | `dead-ball-specialist-audio-normalized-v3 [f2d3908c2d28]` | 71.15% | **75.98%** | 78.55% | 81.20% |
| 9 | `dead-state-global-audio-normalized-v1-full-final [9d451bd6ddb5]` | 71.16% | **75.97%** | 78.51% | 81.10% |
| 10 | `dead-ball-specialist-audio-normalized-v2-legacy-only [735df75dd1e1]` | 71.17% | **75.97%** | 78.50% | 81.09% |
| 11 | `dead-ball-specialist-audio-normalized-v4-no-legacy [53aee0feac37]` | 71.08% | **75.97%** | 78.55% | 81.20% |
| 12 | `dead-state-transition-audio-normalized-v0-legacy-only [23e961e96bcd]` | 70.99% | **75.94%** | 78.61% | 81.14% |
| 13 | `dead-state-transition-audio-normalized-v3-legacy-only-final [a53ca5089099]` | 70.99% | **75.94%** | 78.61% | 81.14% |
| 14 | `full-audiovisual-serve-prob-stack-v1-control` | 70.70% | **75.93%** | 79.20% | 82.15% |
| 15 | `full-audiovisual-v2` | 70.70% | **75.93%** | 79.20% | 82.15% |
| 16 | `full-audiovisual-v2-final` | 70.70% | **75.93%** | 79.20% | 82.15% |
| 17 | `dead-ball-specialist-audio-normalized-v5-new-only [09d68be8f17a]` | 71.01% | **75.93%** | 78.47% | 81.13% |
| 18 | `pair-serve-audio-v5` | 71.01% | **75.93%** | 78.47% | 81.13% |
| 19 | `dead-state-transition-audio-normalized-v2-no-legacy [5a2a47ee603c]` | 71.14% | **75.88%** | 78.47% | 81.10% |
| 20 | `dead-state-transition-audio-normalized-v5-no-legacy-final [f70f934a2775]` | 71.14% | **75.88%** | 78.47% | 81.10% |
| 21 | `audiovisual-v2-targeted-pruned-diagnostic` | 70.30% | **75.81%** | 79.23% | 82.11% |
| 22 | `dead-state-global-audio-normalized-v2-full-final [bba9eec38729]` | 68.56% | **75.78%** | 80.16% | 83.13% |
| 23 | `dual-serve-v4-v5-fusion-v1` | 70.76% | **74.50%** | 77.08% | 79.47% |
| 24 | `dead-ball-specialist-audiovisual-v2 [de0ee020eb1b]` | 69.51% | **73.32%** | 75.94% | 78.41% |
| 25 | `dead-ball-specialist-audiovisual-v1 [8a27d4f7ca39]` | 69.48% | **73.30%** | 75.92% | 78.42% |
| 26 | `pair-serve-audiovisual-v3` | 69.48% | **73.30%** | 75.92% | 78.42% |
| 27 | `pair-serve-audiovisual-v4` | 69.48% | **73.30%** | 75.92% | 78.42% |
| 28 | `full-percentile-v1` | 66.89% | **72.08%** | 75.67% | 79.14% |
| 29 | `pair-serve-visual-v1` | 67.01% | **71.92%** | 75.28% | 78.63% |
| 30 | `pair-serve-visual-v2` | 67.01% | **71.92%** | 75.28% | 78.63% |
| 31 | `pilot-percentile-v1` | 55.08% | **61.73%** | 65.78% | 70.19% |
| 32 | `pilot-baseline-v0` | 55.58% | **61.65%** | 66.52% | 71.49% |

For the winner, the complete unfiltered padding components are:

| Padding each side | `P_pad` | `R_core` | `F1_padP_coreR` | Model export | Human export | Difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0s | 67.00% | 78.59% | 72.34% | 528.0s | 450.1s | +77.9s |
| **1s target** | **71.69%** | **90.09%** | **79.85%** | **697.0s** | **602.1s** | **+94.9s** |
| 2s | 75.33% | 94.21% | 83.72% | 856.6s | 754.1s | +102.5s |
| 3s | 78.71% | 95.80% | 86.42% | 1002.7s | 906.1s | +96.6s |

Moving the winner from 1s to 2s raises F1 by 3.87 points and core recall by 4.12 points, but adds
159.6 seconds to its validation export. Moving from 1s to 3s raises F1 by 6.57 points and recall by
5.70 points, but adds 305.7 seconds. Because human padding also expands by almost exactly the same
amount, the duration difference stays near +100 seconds. These results explain the sensitivity but
do not revise the predeclared 1-second product setting.

## Rally-score threshold sweep

At the fixed 1-second product padding, the winning model's 0.50, 0.60, and 0.70 filters are no-ops:
all 85 validation candidates already score at least 0.70. A 0.75 cutoff removes one candidate and
lowers F1 by 0.59 points. A 0.80 cutoff keeps 72 and lowers F1 by 3.49 points; 0.85 keeps 52 and
lowers it by 9.35 points; and 0.90 keeps only 24 and lowers it by 28.50 points.

| Minimum score | Kept | pad-0s F1 | pad-1s F1 | pad-2s F1 | pad-3s F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| none | 85/85 | 72.34% | **79.85%** | 83.72% | 86.42% |
| 0.50 | 85/85 | 72.34% | **79.85%** | 83.72% | 86.42% |
| 0.60 | 85/85 | 72.34% | **79.85%** | 83.72% | 86.42% |
| 0.70 | 85/85 | 72.34% | **79.85%** | 83.72% | 86.42% |
| 0.75 | 84/85 | 71.85% | **79.25%** | 83.04% | 85.64% |
| 0.80 | 72/85 | 69.18% | **76.36%** | 79.85% | 82.07% |
| 0.85 | 52/85 | 64.31% | **70.50%** | 73.45% | 75.44% |
| 0.90 | 24/85 | 46.95% | **51.35%** | 53.08% | 53.98% |

Applying the same cutoff to every model produces the following best model at the target padding:

| Minimum score | Best model/output at that fixed cutoff | `P_pad` | `R_core` | `F1_padP_coreR` |
| ---: | --- | ---: | ---: | ---: |
| none | `full-audiovisual-serve-prob-stack-v1-exploratory [418babe628f9]` | 71.69% | 90.09% | **79.85%** |
| 0.50 | same | 71.69% | 90.09% | **79.85%** |
| 0.60 | same | 71.69% | 90.09% | **79.85%** |
| 0.70 | same | 71.69% | 90.09% | **79.85%** |
| 0.75 | `pair-serve-audio-v7-new-only` | 72.70% | 87.59% | **79.45%** |
| 0.80 | `full-audiovisual-audio-normalized-v3` | 73.57% | 82.43% | **77.75%** |
| 0.85 | `dead-ball-specialist-audio-normalized-v2-legacy-only [735df75dd1e1]` | 66.42% | 90.01% | **76.44%** |
| 0.90 | `dead-state-transition-audio-normalized-v1-full [8a3690a9134e]` | 66.92% | 76.12% | **71.22%** |

The filter is useful for some lower-ranked variants, but it does not change the winner. The largest
per-model gains at pad-1s are +3.43 points for each audiovisual serve-pair output at 0.60, +3.32
points for dual-serve fusion at 0.60, +2.87 points for normalized-audio serve v6 at 0.50, and +2.80
points for normalized-audio serve v5 at 0.70. The strongest filtered challenger is
`pair-serve-audio-v6-no-legacy` at 0.50 with 79.47%, still 0.38 points below the unfiltered winner.

Across all 32 models at pad-1s, cutoff behavior is unstable in exactly the way expected for
uncalibrated scores:

| Cutoff | Improved | Unchanged | Worse | Mean F1 change |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 8 | 22 | 2 | +0.44 pp |
| 0.60 | 14 | 17 | 1 | +0.26 pp |
| 0.70 | 5 | 14 | 13 | -1.61 pp |
| 0.75 | 15 | 1 | 16 | -3.32 pp |
| 0.80 | 11 | 0 | 21 | -6.37 pp |
| 0.85 | 1 | 0 | 31 | -13.41 pp |
| 0.90 | 0 | 0 | 32 | -31.45 pp |

## Decision

Keep `full-audiovisual-serve-prob-stack-v1-exploratory [418babe628f9]` first in the validation
ranking under the new metric. Keep the product comparison fixed at one second before/after. Do not
add a rally-score export cutoff to this winner: 0.50–0.70 remove nothing, 0.75 makes a small net
regression, and 0.80–0.90 discard too much core play. If a different model family is promoted later, tune its score threshold on
validation as part of that model's frozen configuration rather than transferring a numeric cutoff
from another family.

The protected test remains closed for this selection. Opening it would be a separate, predeclared
final regression gate after the model, threshold policy, and product padding are frozen.
