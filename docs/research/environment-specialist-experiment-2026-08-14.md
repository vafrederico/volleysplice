# Environment-specialist rally-model experiment — 2026-08-14

## Decision

The **all-label refit** (`model-942b67f0d3ab`) is the only candidate that improves the
primary pooled metric on both requested product paddings. On the two newest indoor
evaluation recordings it raises `F1_padP_coreR` from **83.92% to 88.61% at 2 seconds**
and from **85.87% to 90.20% at 3 seconds**. The gain comes from substantially higher
precision with a small recall reduction. It also cuts excess 2-second export duration
from 296.73 seconds to 103.02 seconds.

The grass-only and indoor-only models do not beat the previous offline production model
on this indoor scope. The indoor-only model is especially underdetermined: it has two
training recordings from one source group and predicts nearly the entire evaluation
footage as live. It is useful as a negative expert-model result, not a promotion
candidate.

This is not yet a production-promotion decision. Both new indoor label documents still
say `in-progress` with no `reviewedAt` timestamp, and one retains the annotator string
`GPT-5.6 Sol high (unvalidated)` despite being marked continuously reviewed. Complete
and freeze those labels before treating these numbers as final. The three reserved grass
videos also remain unopened for the requested grass evaluation.

## Experimental contract

The declared target product padding is **2 seconds before and after**. Three seconds is
the requested alternative, while 0 and 1 seconds are required sensitivity cases. All
ranking uses pooled `F1_padP_coreR`, never a mean of per-video F1:

```text
P_pad  = duration(padded_model ∩ padded_human) / duration(padded_model)
R_core = duration(padded_model ∩ core_human)   / duration(core_human)
F1_padP_coreR = 2 × P_pad × R_core / (P_pad + R_core)
```

Model and human intervals receive identical symmetric padding, are clipped to video
bounds, and overlapping or touching ranges are merged. `ignoredIntervals` are removed
from model, core-human, and padded-human unions before any numerator or denominator is
measured.

The evaluation scope contains only:

- `indoor-source-04` — Forest Ridge source group
- `indoor-source-03` — SPU Match 1 source group

Both are `challenge`, `consent.train=false`, and were excluded from all three fits. The
historical protected test recording was not used for model selection or this comparison.

## Training variants

All variants reuse the production three-head feature architecture and freeze the
production decoder, serve composition, dead-state refinement, optimizer settings, and
production-selected epoch caps.

| Variant | Model ID | Training scope | Key limitation |
| --- | --- | --- | --- |
| Previous offline production | `model-9c92b8e9333f` | Existing no-beach production corpus | Baseline |
| All labels + latest grass | `model-942b67f0d3ab` | 6 historical development recordings + 2 latest grass recordings | New grass labels remain `in-progress` |
| Grass specialist | `model-04dc7d97e693` | 4 historical grass + 2 latest grass recordings | Two grass source groups |
| Indoor specialist | `model-69a90313927f` | 2 historical indoor recordings | One source group; no new indoor training labels |

The latest grass hard negatives were applied only to the rally-live and serve-contact
heads. Each marked negative sample appears four times total. Ignored intervals remain
excluded and are never converted into negatives.

## Pooled indoor results

Percentages are pooled over both recordings. Durations are seconds after ignored-time
subtraction. `Δ export` is padded model export minus padded human export.

| Model | Pad | `P_pad` | `R_core` | `F1_padP_coreR` | Model export | Human export | Δ export |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous production | 0s | 69.34% | 90.20% | 78.41% | 835.00 | 641.84 | +193.16 |
| Previous production | 1s | 71.40% | 95.98% | 81.88% | 1,056.97 | 805.46 | +251.51 |
| Previous production | **2s** | **73.24%** | **98.26%** | **83.92%** | **1,264.19** | **967.46** | **+296.73** |
| Previous production | 3s | 75.85% | 98.95% | 85.87% | 1,441.84 | 1,129.46 | +312.38 |
| All labels + latest grass | 0s | 78.51% | 82.37% | 80.40% | 673.36 | 641.84 | +31.52 |
| All labels + latest grass | 1s | 80.72% | 91.46% | 85.76% | 879.63 | 805.46 | +74.17 |
| All labels + latest grass | **2s** | **82.49%** | **95.70%** | **88.61%** | **1,070.48** | **967.46** | **+103.02** |
| All labels + latest grass | 3s | 83.91% | 97.51% | 90.20% | 1,248.26 | 1,129.46 | +118.79 |
| Grass specialist | 0s | 69.36% | 75.41% | 72.26% | 697.89 | 641.84 | +56.06 |
| Grass specialist | 1s | 72.07% | 87.85% | 79.18% | 935.51 | 805.46 | +130.05 |
| Grass specialist | **2s** | **74.31%** | **93.35%** | **82.75%** | **1,150.88** | **967.46** | **+183.42** |
| Grass specialist | 3s | 76.82% | 96.27% | 85.45% | 1,334.21 | 1,129.46 | +204.75 |
| Indoor specialist | 0s | 37.61% | 98.46% | 54.43% | 1,680.31 | 641.84 | +1,038.48 |
| Indoor specialist | 1s | 45.14% | 99.92% | 62.19% | 1,767.20 | 805.46 | +961.74 |
| Indoor specialist | **2s** | **52.18%** | **100.00%** | **68.58%** | **1,841.84** | **967.46** | **+874.38** |
| Indoor specialist | 3s | 59.05% | 100.00% | 74.26% | 1,903.03 | 1,129.46 | +773.57 |

At the declared 2-second target, the all-label model changes precision by **+9.25
percentage points**, recall by **−2.55 points**, `F1_padP_coreR` by **+4.68 points**, and
model export duration by **−193.71 seconds** versus production. At 3 seconds the changes
are **+8.06**, **−1.44**, **+4.33 points**, and **−193.59 seconds**, respectively.

The unpadded event-F1 guardrail also moves slightly upward for the all-label refit:
57.44% to 58.51%. Grass-only reaches 41.58%; indoor-only reaches 20.31%. Event F1 is a
guardrail and was not used for ranking.

The all-label gain is present on each evaluation recording rather than coming from only
one: at 2 seconds its `F1_padP_coreR` is 86.26% versus 82.47% on Forest Ridge and 91.01%
versus 85.39% on SPU Match 1. At 3 seconds those comparisons are 88.20% versus 85.03%
and 92.23% versus 86.71%.

## Walking / ball-retrieval diagnostic

The two latest grass training documents contain 301.53 seconds of marked walking /
ball-retrieval hard negatives. Prediction overlap on those same training recordings is:

| Model | Overlap | Share of marked walking / retrieval time |
| --- | ---: | ---: |
| Previous production | 85.61s | 28.39% |
| All labels + latest grass | 12.77s | 4.24% |
| Grass specialist | 18.87s | 6.26% |
| Indoor specialist | 181.67s | 60.25% |

This strongly supports the observed failure mechanism and shows that explicit hard
negatives taught the new grass-trained heads to suppress it. It is deliberately labeled
an **in-sample diagnostic**, not generalization evidence. Generalization must be measured
after the three reserved grass recordings receive evaluation-only labels.

## UI and inference coverage

All three new models now have analysis artifacts for all 19 videos in the review
catalog: 57 model/video runs total. The main review UI defaults to the all-label model and
enables exactly four model families: the previous offline production model plus the three
new variants. All older experiment models and all browser-inference models begin disabled
but remain available through the track toggles.

The `/label` timeline loads the previous production baseline plus 2-second and 3-second
tracks for each new model when its analysis exists. Each track reports live `P_pad`,
`R_core`, `F1_padP_coreR`, and export duration against the editable labels.

## Artifacts

- Experiment root: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/environment-specialists-v1`
- Frozen split policy: `split-policy.json`
- Model bundles: `models/{all-labels,grass-only,indoor-only}/bundle.json`
- Indoor evaluation: `reports/indoor-evaluation-pad2-target.json`
- Analysis catalog: `/mnt/freenas/volleycut/intake-2026-08-13/analyses`
- Reproducible runner: `scripts/train-environment-specialists.py`

The remaining grass evaluation recordings are `grass-source-02`,
`grass-source-06`, and `grass-source-08`. They overlap historical grass
source groups, so that future result will be within-source generalization, not an unseen
source-group claim.
