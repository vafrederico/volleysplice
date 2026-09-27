# Distilled MobileNet versus frozen MobileNet at strict 99% inner recall

All 28 registered student fits and 28 matching temporal fits are complete, and the full independent audit passed. The study has **10/12 feasible folds**; only seed **1729** has all four source groups. This is the only full-scope pair available against frozen MobileNet. Neither model has an eligible three-seed mean under this strict99 comparison.

On that one prespecified matched seed, distilled MobileNet retains **39.1 more seconds of human core play**: recall rises **96.93% → 98.56%**. It also exports **851.7 more seconds** (14:11.7), including **732.7 more seconds of incorrect footage** (12:12.7). Precision falls **83.33% → 71.31%**, primary F1 falls **89.62% → 82.75%**, and event F1 falls **72.65% → 62.81%**. Thus improved retained-time recall is accompanied by worse export precision and event detection on this pair, not a demonstrated overall improvement.

The guardrail matters: complete losses fall only **12 → 11**, and four previously retained rallies become completely lost. All four last **3.073–5.216 seconds**, classified as long only because the registered cutoff is **strictly greater than 3 seconds**. The frozen baseline fully retained three and partially retained one. Their exact identities are listed below.

The evaluation contains eight exact recordings, four held source groups, 322 original rallies, and 8,270.032 nonignored video seconds. No protected test or beach data was opened. [CPU/browser qualification](distilled-mobile-qualification-2026-09-23.md) passed on the actual trained graph; physical-phone behavior and the complete pixel-to-rally application remain unmeasured. These results do not support replacing the current model with this distilled recipe.

Target padding is 2 seconds before and after; positive gaps strictly below 3 seconds are retained. Ignored time is outside the evaluation universe. Inner eligibility does not guarantee 99% recall on an unseen source.

Each full seed pools recording durations before computing precision, recall, and F1. A model mean requires all three seeds to have all four held source groups. Infeasible seeds have no relaxed fallback or partial-scope rank.

| Model | Feasible folds | Complete seeds | Mean P / R / F1 across all three seeds |
|---|---:|---|---|
| Distilled MobileNet + TCN | 10/12 | 1729 | Unavailable: incomplete three-seed scope |
| Frozen MobileNet + TCN | 11/12 | 3407, 1729 | Unavailable: incomplete three-seed scope |

## Complete seed results at the target padding

These are full-scope descriptive seed results, including any complete seeds from a model whose three-seed result is unavailable. Times are seconds. Correctly removed time is nonignored time outside the wanted human export that was removed. Wanted human-export time omitted is an error.

| Model | Seed | P_pad | R_core | F1_padP_coreR | Export | Correctly removed | Incorrect export | Wanted export omitted | Missed core |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Distilled MobileNet + TCN | 3407 | Infeasible | — | — | — | — | — | — | — |
| Distilled MobileNet + TCN | 1729 | 71.31% | 98.56% | 82.75% | 4912.9 | 3185.2 | 1409.7 | 172.0 | 34.3 |
| Distilled MobileNet + TCN | 20260918 | Infeasible | — | — | — | — | — | — | — |
| Frozen MobileNet + TCN | 3407 | 85.79% | 96.00% | 90.61% | 3903.3 | 4040.4 | 554.5 | 326.3 | 95.5 |
| Frozen MobileNet + TCN | 1729 | 83.33% | 96.93% | 89.62% | 4061.2 | 3917.9 | 677.0 | 290.9 | 73.4 |
| Frozen MobileNet + TCN | 20260918 | Infeasible | — | — | — | — | — | — | — |

## Original-rally and event guardrails

Complete loss retains no evaluable core; partial loss retains some but not all. Short/long uses the original rally duration at most/over 3 seconds, not fragments after ignored-time subtraction. Event P/R/F1 uses unpadded original, uncensored events at IoU ≥ 0.5; it is distinct from retained play recall and does not establish accurate serve boundaries.

| Model | Seed | Fully retained / partial / complete loss | Short partial / complete loss | Long partial / complete loss | Long core recall | Event P / R / F1 |
|---|---:|---|---|---|---:|---|
| Distilled MobileNet + TCN | 1729 | 302 / 9 / 11 | 2 / 7 | 7 / 4 | 99.12% | 58.81% / 67.39% / 62.81% |
| Frozen MobileNet + TCN | 3407 | 275 / 33 / 14 | 3 / 12 | 30 / 2 | 96.74% | 74.71% / 79.81% / 77.18% |
| Frozen MobileNet + TCN | 1729 | 278 / 32 / 12 | 0 / 12 | 32 / 0 | 97.69% | 70.03% / 75.47% / 72.65% |

## All four padding cases

| Model | Seed | Padding per side | P_pad | R_core | F1_padP_coreR | Model export | Human export | Difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Distilled MobileNet + TCN | 1729 | 0s | 64.14% | 93.91% | 76.22% | 3494.8 | 2387.2 | 1107.6 |
| Distilled MobileNet + TCN | 1729 | 1s | 68.15% | 97.56% | 80.24% | 4209.3 | 3031.2 | 1178.2 |
| Distilled MobileNet + TCN | 1729 | 2s | 71.31% | 98.56% | 82.75% | 4912.9 | 3675.2 | 1237.7 |
| Distilled MobileNet + TCN | 1729 | 3s | 74.21% | 98.75% | 84.73% | 5576.6 | 4323.3 | 1253.3 |
| Frozen MobileNet + TCN | 3407 | 0s | 81.96% | 88.07% | 84.91% | 2565.2 | 2387.2 | 178.1 |
| Frozen MobileNet + TCN | 3407 | 1s | 84.11% | 93.87% | 88.72% | 3237.1 | 3031.2 | 205.9 |
| Frozen MobileNet + TCN | 3407 | 2s | 85.79% | 96.00% | 90.61% | 3903.3 | 3675.2 | 228.1 |
| Frozen MobileNet + TCN | 3407 | 3s | 87.16% | 97.11% | 91.87% | 4567.5 | 4323.3 | 244.2 |
| Frozen MobileNet + TCN | 1729 | 0s | 78.25% | 89.09% | 83.32% | 2717.8 | 2387.2 | 330.7 |
| Frozen MobileNet + TCN | 1729 | 1s | 81.15% | 95.01% | 87.53% | 3402.9 | 3031.2 | 371.7 |
| Frozen MobileNet + TCN | 1729 | 2s | 83.33% | 96.93% | 89.62% | 4061.2 | 3675.2 | 386.0 |
| Frozen MobileNet + TCN | 1729 | 3s | 84.69% | 97.90% | 90.82% | 4747.6 | 4323.3 | 424.4 |

## Fold eligibility

| Model | Seed | Held source | Eligible candidates / 192 | Maximum inner recall | Selected epoch |
|---|---:|---|---:|---:|---:|
| Distilled MobileNet + TCN | 3407 | source-group-005 | 32 | 99.61% | 60 |
| Distilled MobileNet + TCN | 3407 | source-group-007 | 20 | 99.62% | 30 |
| Distilled MobileNet + TCN | 3407 | source-group-009 | 8 | 99.32% | 30 |
| Distilled MobileNet + TCN | 3407 | source-group-012 | 0 | 98.78% | Infeasible |
| Distilled MobileNet + TCN | 1729 | source-group-005 | 4 | 99.38% | 5 |
| Distilled MobileNet + TCN | 1729 | source-group-007 | 8 | 99.70% | 5 |
| Distilled MobileNet + TCN | 1729 | source-group-009 | 16 | 99.68% | 30 |
| Distilled MobileNet + TCN | 1729 | source-group-012 | 9 | 99.57% | 15 |
| Distilled MobileNet + TCN | 20260918 | source-group-005 | 24 | 99.48% | 60 |
| Distilled MobileNet + TCN | 20260918 | source-group-007 | 8 | 99.57% | 5 |
| Distilled MobileNet + TCN | 20260918 | source-group-009 | 8 | 99.64% | 5 |
| Distilled MobileNet + TCN | 20260918 | source-group-012 | 0 | 98.61% | Infeasible |
| Frozen MobileNet + TCN | 3407 | source-group-005 | 58 | 99.84% | 60 |
| Frozen MobileNet + TCN | 3407 | source-group-007 | 48 | 99.93% | 30 |
| Frozen MobileNet + TCN | 3407 | source-group-009 | 16 | 99.89% | 30 |
| Frozen MobileNet + TCN | 3407 | source-group-012 | 2 | 99.15% | 15 |
| Frozen MobileNet + TCN | 1729 | source-group-005 | 49 | 99.80% | 30 |
| Frozen MobileNet + TCN | 1729 | source-group-007 | 16 | 99.52% | 15 |
| Frozen MobileNet + TCN | 1729 | source-group-009 | 4 | 99.29% | 15 |
| Frozen MobileNet + TCN | 1729 | source-group-012 | 9 | 99.58% | 15 |
| Frozen MobileNet + TCN | 20260918 | source-group-005 | 30 | 99.90% | 60 |
| Frozen MobileNet + TCN | 20260918 | source-group-007 | 18 | 99.63% | 60 |
| Frozen MobileNet + TCN | 20260918 | source-group-009 | 10 | 99.69% | 5 |
| Frozen MobileNet + TCN | 20260918 | source-group-012 | 0 | 98.27% | Infeasible |

NAS `distilled-mobile-v1/summary.json` includes hashes of both independent audits and reports, all duration errors at every padding, raw-core coverage, original short/long counts, and per-seed paired differences. No phone runtime or accuracy claim follows from these desktop training results.

## Newly completely lost rallies longer than 3 seconds

All four belong to `source-group-007`. Three were fully retained by the frozen baseline; one was partially retained. The distilled export retains zero core seconds for each even after target padding and gap joining. Times below are source-video seconds; these are descriptive diagnostics, not another model-selection pass.

| Recording | Original start–end | Duration | Frozen retained core | Distilled retained core |
|---|---:|---:|---:|---:|
| grass-source-04 | 73.279–78.495 | 5.216s | 4.079s | 0s |
| grass-source-04 | 90.500–93.573 | 3.073s | 3.073s | 0s |
| grass-source-04 | 350.402–353.517 | 3.115s | 3.115s | 0s |
| grass-source-10 | 874.042–877.703 | 3.661s | 3.661s | 0s |

The original interval identities and baseline coverage are bound in NAS `distilled-mobile-v1/newly-lost-over3s-seed1729.json`.

## Audit and artifact bindings

The final independent audit checks all 28 student fits, 28 temporal fits, and 36 logical inner views; source exclusions, eight-epoch student exposures, frozen BatchNorm buffers, input/teacher/feature identities, exact-only scalers, checkpoint/exposure inventories, exhaustive strict99 selections, infeasible outer omissions, and all four padding metrics. Forward replay is sampled: three teacher/encoder images per recording where applicable and first/middle/last real-context temporal chunks, not every training step or video tick.

| Artifact relative to NAS study folder | SHA-256 |
|---|---|
| `report.json` | `0620fe81eee6a852112b4884b1b2f1c9c449d5fc8589ef5f5d9745dda60ac00a` |
| `audit.json` | `09a23abbc24a25befb8fc93e8af8563596d2b36fe3e75bac1712ff8f2df7a12d` |
| `summary.json` | `565f4ff478a84ec5d718a86a67990680b7ab6480158a5d3a45e67a96502f5c07` |
| `newly-lost-over3s-seed1729.json` | `f85a25ffa5ea91a295396ff91817ff1aab7f2104501930cecd6f42b9e7c7580e` |

The fixed strict99 baseline report and both independent audit identities are recorded in `summary.json`. All model and paired means are null when any prespecified seed lacks the full source scope.
