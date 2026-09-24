# Compact typed rally boundaries: results

<!-- EXECUTIVE FINDINGS START -->
## Findings and product implications

**Separating first-start correction from adding rallies is useful; independently correcting ends produces a larger improvement. Automatic edits still lose too much real play from rally records to enable them for score tracking.** Human review of complete flagged production regions is the promising workflow. All results below use the existing compact short-boost TCN and its start/end heads, three seeds, eight development recordings and 322 gold rallies. This experiment did not retrain the network or test DINO transfer.

The fixed run completed 12 automatic and 192 ideal-human outcomes after 88 qualification tests and independent reconstruction of 96 real-input plans. Every outcome passed the inline plan, queue, human-action, workload, identity, coverage and export audits. No production settings or weights changed.

### Automatic boundary edits

| Event timeline | Rally precision % | Rally recall % | Rally F1 % | Raw core recall % | Additional completely missed rallies | Core play removed from rally records (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Production | 62.64 | 70.81 | 66.47 | 95.37 | 0.00 | 0.00 |
| Correct first start only | 63.83 | 72.15 | 67.74 | 91.61 | 1.33 | 89.76 |
| Correct first start and add subsequent starts | 60.17 | 72.26 | 65.66 | 91.61 | 1.33 | 89.76 |
| Give each proposed rally its own end | 63.36 | 76.09 | 69.14 | 83.23 | 2.33 | 289.64 |
| Refine those starts and ends with neural heads | 65.26 | 78.36 | 71.22 | 84.07 | 0.67 | 269.61 |

These are seed means, not fractional events in a single video. The head-refined event F1 ranges from 68.73% to 74.86% across seeds. Its observed start recall within one second rises from production's 71.74% to **87.89%**, and observed end recall from 37.89% to **54.76%**. This supports using the heads as review guidance. It does not offset the 269.61 seconds of actual play omitted from the revised event cores, even though that footage remains exported.

First-start candidates are much more reliable than extra-rally candidates on this scope. At one-second tolerance, head refinement gives initial-start candidates **88.28% precision / 79.66% recall**, versus **14.77% / 47.62%** for additional starts. Additional starts average 22.67 observed candidates but only 3.33 matches. There are only **seven true additional-start targets**, compared with 318 initial targets. These candidate metrics include already-aligned initial starts, not only actionable corrections; they must not be presented as the success rate of edits. Without head refinement, the corresponding initial precision is 82.46% and additional precision 11.87%. Joint typed start-and-end F1 improves from 36.93% with compact event ends to 44.64% with head refinement, still insufficient for unattended score events.

### What ideal human review adds

The table uses the head-refined queue in evidence-per-playback-cost order. A job is one whole original production parent; it can contain multiple candidate events or real rallies. Both human modes receive the same playback and selected parents. Review time is playback at 1x with context, not measured labor.

| Per-video cap | Playback min | Parent jobs | Real rallies reviewed | Proposal-confirmation P / R / F1 % | Full-parent correction P / R / F1 % |
| --- | --- | --- | --- | --- | --- |
| 5% | 6.37 | 45.67 | 43.67 | 67.49 / 76.29 / 71.62 | 68.14 / 76.60 / 72.12 |
| 10% | 13.09 | 84.67 | 82.00 | 68.68 / 77.64 / 72.89 | 69.56 / 78.05 / 73.56 |
| 20% | 26.60 | 151.00 | 145.33 | 71.89 / 81.26 / 76.29 | 73.95 / 82.30 / 77.90 |
| 40% | 53.55 | 245.67 | 239.33 | 77.19 / 87.58 / 82.06 | 81.02 / 90.58 / 85.53 |

Proposal confirmation accepts only model candidates whose observed starts match within one second and materially overlap a true rally. The simulated human may correct the type and, in paired policies, the complete end within the parent even if the predicted end is inaccurate. It cannot discover an unproposed rally. At the 5/10/20% evidence-order head-refined caps, this produces zero new complete misses or core loss on these videos. At 40%, however, it omits **two additional rallies and 10.37 seconds of their play** on average: correcting an earlier proposed rally's end exposes an unproposed later rally. A larger review budget is therefore not automatically safer under proposal-only actions.

Full-parent review explicitly checks for all rallies inside the selected parent, including ones with no model suggestion, and can remove wholly false event records. It has **zero additional complete misses and zero loss of previously covered core time in every tested configuration**. The original three completely missed rallies remain outside production coverage. At the 10% evidence cap, first-start-only proposal confirmation reaches 68.42% event F1, separate-end confirmation 72.11%, and head-refined confirmation 72.89%, at approximately 13 minutes each. Full-parent review is about 73.55-73.56% for all four proposal policies at that cap: most of that ceiling comes from the stronger human operation, not from superior model boundaries.

There is an important serve-start accounting limitation. At the 10% head-refined evidence cap, ordinary start recall within one second improves from **71.74% to 76.09%** with proposal confirmation and **76.92%** with full-parent review. Observed-only start recall instead becomes **69.88% / 69.57%**. All excluded markers are true starts clipped to the original parent boundary: 20.00 / 23.67 otherwise-correct timestamp matches lose observed status. Only 0.33 / 1.33 of those are within one millisecond, so timestamp rounding is a small part of the effect. The baseline's model guesses are marked observed; a confirmed permission edge is explicitly marked unobserved. This drop is not evidence that human review moved that many serves farther away. Playback context or export padding may contain the earlier true serve, but the frozen experiment did not permit edits outside the raw parent. A follow-up should allow confirmed boundaries within the actually reviewed context and preserve their provenance; the current results do not establish complete serve localization.

**Recommendation:** retain production export coverage and show typed, provisional suggestions for (1) correcting an existing first start, (2) adding an additional rally, and (3) correcting each rally's end independently. When a parent is flagged, let the reviewer inspect and repair the entire parent, including adding rallies the model missed. Do not automatically apply the tested start/end edits or treat every neural start as another point. Neural scores currently provide heuristic ordering, not calibrated correctness probabilities. A future confidence model should estimate first-start correctness, additional-rally existence and end correctness separately, with more independently reviewed merged-rally examples before claiming reliable extra-rally detection.

### The three recall measures

Every arm retains exactly production's export: at target symmetric two-second padding, **P_pad 72.45%, retained-play R_core 99.27%, F1_padP_coreR 83.76%**, with **82.38 minutes exported**, **53.88 minutes correctly removed**, **1.57 minutes of wanted export incorrectly omitted**, and **22.70 minutes incorrectly exported**. Positive gaps strictly below three seconds join. All arms tie on this primary metric; event F1 is a separate diagnostic.

Raw core recall measures play covered by unpadded, unjoined event records. Production's 95.37% raw core recall differs from its 95.75% zero-padding export recall because the export still joins short gaps even at zero padding. Rally-event recall counts individual gold rallies matched one-to-one at IoU >= 0.5. Neither percentage can substitute for retained-play recall. Human boundary correction also remains constrained to the original production parent: the 325 parent-specific targets include 73 inaccessible starts and 58 inaccessible ends, so review cannot recover every true endpoint within that permission.

The complete tables below retain all queue orders, budgets, source groups, seed ranges and padding cases. These are development results with ideal human actions, not measured reviewer performance, unseen-video generalization, serving-side accuracy, winner prediction, reconstructed-score accuracy or phone/browser latency.
The [automatic comparison](./assets/compact-typed-boundaries-automatic.png), [review workload comparison](./assets/compact-typed-boundaries-human-review.png), and [typed boundary quality figure](./assets/compact-typed-boundaries-start-and-end-quality.png) visualize these distinctions. The [frozen protocol](./neural-typed-boundary-protocol-2026-09-19.md) defines the policies and permitted human actions.

<!-- EXECUTIVE FINDINGS END -->

This development experiment distinguishes initial rally-start correction, additional rallies, and separate end boundaries. Original production export footage remains fixed. Every arm ties on `F1_padP_coreR`; event-timeline quality and losses are reported separately. No production change, new model fit, or protected test.

Scope: 8 recordings, 4 source groups, 322 eligible gold rallies; 12 automatic outcomes and 192 reviewed outcomes across three seeds. There are 69 aggregate arms including production. Values are seed means; brackets give seed minimum and maximum, not confidence intervals. Counts/rates pool recordings within each seed before averaging.

Raw core recall measures the separate event timeline before export padding. A fixed export can preserve video while an event candidate loses a rally or its endpoints. Review minutes include whole-parent playback context at 1x and are not measured human labor.

## Fixed export accounting

This table applies exactly to every arm. Padding is symmetric; positive gaps strictly below three seconds join. Ignored intervals are removed without rejoining. Correct removed time is omitted footage outside wanted human export; incorrect removed time is wanted human export omitted. Duration columns are minutes.

| Pad each side s | P_pad % | R_core % | F1_padP_coreR % | Model export min | Human export min | Difference min | Correct removed min | Incorrect removed min | Incorrect export min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 65.50 | 95.75 | 77.79 | 58.16 | 39.79 | 18.37 | 77.98 | 1.69 | 20.07 |
| 1 | 69.40 | 98.30 | 81.36 | 70.39 | 50.52 | 19.87 | 65.77 | 1.67 | 21.54 |
| 2 | 72.45 | 99.27 | 83.76 | 82.38 | 61.25 | 21.13 | 53.88 | 1.57 | 22.70 |
| 3 | 75.02 | 99.49 | 85.54 | 93.95 | 72.05 | 21.90 | 42.31 | 1.57 | 23.47 |

## Automatic event timelines

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 62.64 | 70.81 | 66.47 | 71.74 | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 |
| automatic--first_start | 63.83 [63.46, 64.29] | 72.15 [71.74, 72.67] | 67.74 [67.35, 68.22] | 73.08 [70.50, 76.09] | 37.89 | 91.61 [90.74, 92.16] | 89.76 [76.58, 110.47] | 4.33 [3.00, 6.00] | 1.33 [0.00, 3.00] |
| automatic--head_refined | 65.26 [62.89, 68.65] | 78.36 [75.78, 82.30] | 71.22 [68.73, 74.86] | 87.89 [87.58, 88.51] | 54.76 [52.17, 59.94] | 84.07 [83.41, 84.48] | 269.61 [259.98, 285.35] | 3.67 [3.00, 4.00] | 0.67 [0.00, 1.00] |
| automatic--separate_ends | 63.36 [61.92, 64.77] | 76.09 [74.22, 77.64] | 69.14 [67.51, 70.62] | 74.33 [71.74, 77.64] | 50.52 [49.38, 51.24] | 83.23 [82.39, 83.69] | 289.64 [278.76, 309.78] | 5.33 [4.00, 7.00] | 2.33 [1.00, 4.00] |
| automatic--typed_starts | 60.17 [59.02, 61.40] | 72.26 [71.12, 73.60] | 65.66 [64.51, 66.95] | 74.33 [71.74, 77.64] | 37.89 | 91.61 [90.74, 92.16] | 89.76 [76.58, 110.47] | 4.33 [3.00, 6.00] | 1.33 [0.00, 3.00] |

## Original candidate type and paired-boundary quality

These metrics score model candidates before any human correction. Targets are the first and subsequent materially overlapping gold rallies within each original production parent and valid component. Inaccessible gold endpoints remain in denominators. An observed proposed start must be within tolerance and the proposed interval must materially overlap the same gold rally. Wrong-type counts use a separate untyped matching. Joint start/end quality uses the same typed start-matched identity and requires an observed end; no unrelated end may be substituted.

| Policy | Type | Tolerance s | True targets | Candidates with observed start | Matched | Typed P % | Typed R % | Typed F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_start | initial_start | 0.5 | 318.00 | 255.00 [247.00, 266.00] | 170.33 [163.00, 178.00] | 66.79 [65.99, 67.46] | 53.56 [51.26, 55.97] | 59.44 [57.70, 60.96] |
| first_start | initial_start | 1 | 318.00 | 255.00 [247.00, 266.00] | 210.33 [202.00, 222.00] | 82.46 [81.78, 83.46] | 66.14 [63.52, 69.81] | 73.39 [71.50, 76.03] |
| first_start | initial_start | 2 | 318.00 | 255.00 [247.00, 266.00] | 235.00 [225.00, 244.00] | 92.16 [91.09, 93.65] | 73.90 [70.75, 76.73] | 82.00 [79.65, 83.56] |
| first_start | additional_start | 0.5 | 7.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| first_start | additional_start | 1 | 7.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| first_start | additional_start | 2 | 7.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| head_refined | initial_start | 0.5 | 318.00 | 287.00 [283.00, 289.00] | 202.33 [200.00, 206.00] | 70.50 [69.20, 71.28] | 63.63 [62.89, 64.78] | 66.89 [65.90, 67.87] |
| head_refined | initial_start | 1 | 318.00 | 287.00 [283.00, 289.00] | 253.33 [252.00, 254.00] | 88.28 [87.20, 89.75] | 79.66 [79.25, 79.87] | 83.75 [83.03, 84.53] |
| head_refined | initial_start | 2 | 318.00 | 287.00 [283.00, 289.00] | 269.33 [269.00, 270.00] | 93.85 [93.08, 95.05] | 84.70 [84.59, 84.91] | 89.04 [88.63, 89.52] |
| head_refined | additional_start | 0.5 | 7.00 | 22.67 [22.00, 24.00] | 1.00 [0.00, 2.00] | 4.42 [0.00, 9.09] | 14.29 [0.00, 28.57] | 6.75 [0.00, 13.79] |
| head_refined | additional_start | 1 | 7.00 | 22.67 [22.00, 24.00] | 3.33 [3.00, 4.00] | 14.77 [12.50, 18.18] | 47.62 [42.86, 57.14] | 22.54 [19.35, 27.59] |
| head_refined | additional_start | 2 | 7.00 | 22.67 [22.00, 24.00] | 3.67 [3.00, 5.00] | 16.29 [12.50, 22.73] | 52.38 [42.86, 71.43] | 24.84 [19.35, 34.48] |
| separate_ends | initial_start | 0.5 | 318.00 | 255.00 [247.00, 266.00] | 170.33 [163.00, 178.00] | 66.79 [65.99, 67.46] | 53.56 [51.26, 55.97] | 59.44 [57.70, 60.96] |
| separate_ends | initial_start | 1 | 318.00 | 255.00 [247.00, 266.00] | 210.33 [202.00, 222.00] | 82.46 [81.78, 83.46] | 66.14 [63.52, 69.81] | 73.39 [71.50, 76.03] |
| separate_ends | initial_start | 2 | 318.00 | 255.00 [247.00, 266.00] | 235.00 [225.00, 244.00] | 92.16 [91.09, 93.65] | 73.90 [70.75, 76.73] | 82.00 [79.65, 83.56] |
| separate_ends | additional_start | 0.5 | 7.00 | 22.67 [22.00, 24.00] | 1.67 [0.00, 4.00] | 7.45 [0.00, 18.18] | 23.81 [0.00, 57.14] | 11.35 [0.00, 27.59] |
| separate_ends | additional_start | 1 | 7.00 | 22.67 [22.00, 24.00] | 2.67 [2.00, 4.00] | 11.87 [8.33, 18.18] | 38.10 [28.57, 57.14] | 18.09 [12.90, 27.59] |
| separate_ends | additional_start | 2 | 7.00 | 22.67 [22.00, 24.00] | 3.00 [2.00, 5.00] | 13.38 [8.33, 22.73] | 42.86 [28.57, 71.43] | 20.39 [12.90, 34.48] |
| typed_starts | initial_start | 0.5 | 318.00 | 255.00 [247.00, 266.00] | 170.33 [163.00, 178.00] | 66.79 [65.99, 67.46] | 53.56 [51.26, 55.97] | 59.44 [57.70, 60.96] |
| typed_starts | initial_start | 1 | 318.00 | 255.00 [247.00, 266.00] | 210.33 [202.00, 222.00] | 82.46 [81.78, 83.46] | 66.14 [63.52, 69.81] | 73.39 [71.50, 76.03] |
| typed_starts | initial_start | 2 | 318.00 | 255.00 [247.00, 266.00] | 235.00 [225.00, 244.00] | 92.16 [91.09, 93.65] | 73.90 [70.75, 76.73] | 82.00 [79.65, 83.56] |
| typed_starts | additional_start | 0.5 | 7.00 | 22.67 [22.00, 24.00] | 1.67 [0.00, 4.00] | 7.45 [0.00, 18.18] | 23.81 [0.00, 57.14] | 11.35 [0.00, 27.59] |
| typed_starts | additional_start | 1 | 7.00 | 22.67 [22.00, 24.00] | 2.67 [2.00, 4.00] | 11.87 [8.33, 18.18] | 38.10 [28.57, 57.14] | 18.09 [12.90, 27.59] |
| typed_starts | additional_start | 2 | 7.00 | 22.67 [22.00, 24.00] | 3.00 [2.00, 5.00] | 13.38 [8.33, 22.73] | 42.86 [28.57, 71.43] | 20.39 [12.90, 34.48] |

| Policy | Wrong type @1s | Initial proposed for additional | Additional proposed for initial | Joint start/end F1 @1s % | Unobserved starts | Unobserved ends |
| --- | --- | --- | --- | --- | --- | --- |
| first_start | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 0.00 | 27.47 [27.04, 27.75] | 41.67 [33.00, 50.00] | 1.00 |
| head_refined | 1.67 [1.00, 2.00] | 0.33 [0.00, 1.00] | 1.33 [1.00, 2.00] | 44.64 [41.38, 50.31] | 9.67 [8.00, 11.00] | 13.00 [11.00, 16.00] |
| separate_ends | 1.67 [1.00, 2.00] | 0.33 [0.00, 1.00] | 1.33 [1.00, 2.00] | 36.93 [34.06, 38.83] | 41.67 [33.00, 50.00] | 32.33 [29.00, 35.00] |
| typed_starts | 1.67 [1.00, 2.00] | 0.33 [0.00, 1.00] | 1.33 [1.00, 2.00] | 25.55 [24.71, 26.10] | 41.67 [33.00, 50.00] | 23.67 [23.00, 25.00] |

The production parent scope contains 325 parent-specific candidate targets, with 73 inaccessible starts and 58 inaccessible ends. Target counts can differ from unique gold rallies because targets are parent-specific and require material overlap.

## Restricted proposal confirmation

The human reviews every model event candidate in selected parents, accepts one-to-one observed starts within one second plus material interval overlap, and may correct a wrong type. Starts-only arms keep inherited/partitioned ends; paired arms correct accepted end boundaries inside parent permission. Unproposed rallies cannot be invented. Paired corrections may remove an unproposed rally from the event timeline, so raw-core loss and additional misses are reported rather than assumed zero.

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses | Review min | Parent jobs | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_start--proposal_confirmation--chronological--budget-05 | 63.19 [62.91, 63.46] | 71.43 [71.12, 71.74] | 67.06 [66.76, 67.35] | 73.71 [73.29, 74.22] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 6.54 [6.49, 6.59] | 29.00 [28.00, 30.00] | 27.67 [26.00, 29.00] |
| first_start--proposal_confirmation--chronological--budget-10 | 63.49 [62.91, 63.84] | 71.84 [71.12, 72.36] | 67.41 [66.76, 67.83] | 76.92 [76.71, 77.02] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 13.24 [13.09, 13.31] | 54.33 [53.00, 55.00] | 54.00 [53.00, 56.00] |
| first_start--proposal_confirmation--chronological--budget-20 | 65.05 [64.56, 65.48] | 73.60 [72.98, 74.22] | 69.06 [68.51, 69.58] | 80.43 [80.12, 80.75] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 27.27 [27.15, 27.49] | 106.00 [105.00, 107.00] | 107.67 [106.00, 109.00] |
| first_start--proposal_confirmation--chronological--budget-40 | 66.33 [65.38, 66.85] | 75.05 [73.91, 75.78] | 70.42 [69.39, 71.03] | 86.75 [85.71, 87.58] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 53.01 [52.30, 54.17] | 208.00 [205.00, 212.00] | 207.67 [202.00, 214.00] |
| first_start--proposal_confirmation--evidence--budget-05 | 64.47 [64.29, 64.56] | 72.88 [72.67, 72.98] | 68.42 [68.22, 68.51] | 74.22 [73.91, 74.84] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 6.29 [6.26, 6.34] | 42.67 [42.00, 43.00] | 41.00 |
| first_start--proposal_confirmation--evidence--budget-10 | 64.47 [64.29, 64.56] | 72.88 [72.67, 72.98] | 68.42 [68.22, 68.51] | 74.43 [73.60, 75.16] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 13.05 [13.03, 13.06] | 79.33 [79.00, 80.00] | 75.00 [74.00, 76.00] |
| first_start--proposal_confirmation--evidence--budget-20 | 65.29 [64.56, 65.93] | 73.81 [72.98, 74.53] | 69.29 [68.51, 69.97] | 79.09 [78.26, 80.43] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 26.44 [26.23, 26.79] | 137.67 [136.00, 141.00] | 133.00 [130.00, 136.00] |
| first_start--proposal_confirmation--evidence--budget-40 | 66.24 [65.38, 66.85] | 74.95 [73.91, 75.78] | 70.33 [69.39, 71.03] | 85.40 [84.78, 86.02] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 51.82 [51.36, 52.10] | 211.67 [208.00, 218.00] | 209.33 [204.00, 216.00] |
| head_refined--proposal_confirmation--chronological--budget-05 | 63.92 [63.74, 64.01] | 72.26 [72.05, 72.36] | 67.83 [67.64, 67.93] | 71.74 [71.43, 72.05] | 41.20 [40.99, 41.30] | 95.37 | 0.00 | 3.00 | 0.00 | 6.68 [6.60, 6.76] | 32.67 [32.00, 34.00] | 31.33 [30.00, 33.00] |
| head_refined--proposal_confirmation--chronological--budget-10 | 66.48 [66.39, 66.67] | 75.57 [75.47, 75.78] | 70.74 [70.64, 70.93] | 73.29 [72.98, 73.91] | 45.13 [44.72, 45.34] | 95.37 | 0.00 | 3.00 | 0.00 | 13.36 [13.33, 13.39] | 60.00 [59.00, 61.00] | 58.67 [57.00, 60.00] |
| head_refined--proposal_confirmation--chronological--budget-20 | 69.66 [69.21, 70.03] | 79.40 [78.88, 79.81] | 74.21 [73.73, 74.60] | 75.88 [75.16, 76.71] | 51.04 [50.00, 51.86] | 95.13 [95.02, 95.37] | 5.56 [0.00, 8.34] | 3.67 [3.00, 4.00] | 0.67 [0.00, 1.00] | 27.09 [26.98, 27.15] | 110.67 [109.00, 112.00] | 110.33 [110.00, 111.00] |
| head_refined--proposal_confirmation--chronological--budget-40 | 77.06 [76.84, 77.45] | 87.99 [87.58, 88.51] | 82.17 [81.86, 82.61] | 78.88 [77.95, 79.50] | 66.87 [66.46, 67.39] | 94.87 [94.83, 94.88] | 11.95 [11.50, 12.85] | 6.00 | 3.00 | 54.48 [54.37, 54.57] | 222.33 [221.00, 223.00] | 220.00 |
| head_refined--proposal_confirmation--evidence--budget-05 | 67.49 [67.31, 67.58] | 76.29 [76.09, 76.40] | 71.62 [71.43, 71.72] | 72.05 [71.74, 72.36] | 44.00 [43.79, 44.41] | 95.37 | 0.00 | 3.00 | 0.00 | 6.37 [6.18, 6.49] | 45.67 [44.00, 47.00] | 43.67 [43.00, 44.00] |
| head_refined--proposal_confirmation--evidence--budget-10 | 68.68 [67.86, 69.23] | 77.64 [76.71, 78.26] | 72.89 [72.01, 73.47] | 69.88 [69.57, 70.50] | 46.58 [45.96, 47.20] | 95.37 | 0.00 | 3.00 | 0.00 | 13.09 [12.96, 13.25] | 84.67 [84.00, 85.00] | 82.00 |
| head_refined--proposal_confirmation--evidence--budget-20 | 71.89 [71.43, 72.25] | 81.26 [80.75, 81.68] | 76.29 [75.80, 76.68] | 71.64 [71.12, 72.36] | 56.21 [55.59, 56.52] | 95.37 | 0.00 | 3.00 | 0.00 | 26.60 [26.57, 26.65] | 151.00 [150.00, 152.00] | 145.33 [145.00, 146.00] |
| head_refined--proposal_confirmation--evidence--budget-40 | 77.19 [77.05, 77.26] | 87.58 | 82.06 [81.98, 82.10] | 75.57 [74.84, 76.40] | 69.98 [69.57, 70.50] | 94.93 [94.89, 94.95] | 10.37 [9.92, 11.27] | 5.00 | 2.00 | 53.55 [53.17, 54.04] | 245.67 [243.00, 247.00] | 239.33 [237.00, 242.00] |
| separate_ends--proposal_confirmation--chronological--budget-05 | 63.55 [63.46, 63.74] | 71.84 [71.74, 72.05] | 67.44 [67.35, 67.64] | 73.40 [73.29, 73.60] | 40.68 [40.37, 40.99] | 95.37 | 0.00 | 3.00 | 0.00 | 6.62 [6.60, 6.66] | 32.00 [31.00, 33.00] | 30.67 [29.00, 32.00] |
| separate_ends--proposal_confirmation--chronological--budget-10 | 65.60 [65.48, 65.85] | 74.43 [74.22, 74.84] | 69.74 [69.58, 70.06] | 75.88 [75.47, 76.40] | 42.96 [42.55, 43.48] | 95.22 [95.15, 95.37] | 3.48 [0.00, 5.23] | 3.67 [3.00, 4.00] | 0.67 [0.00, 1.00] | 13.42 [13.35, 13.45] | 60.00 [59.00, 61.00] | 58.67 [57.00, 60.00] |
| separate_ends--proposal_confirmation--chronological--budget-20 | 68.24 [67.49, 68.94] | 77.64 [76.71, 78.57] | 72.64 [71.80, 73.44] | 79.30 [77.95, 80.43] | 47.93 [47.20, 48.45] | 95.10 [95.02, 95.15] | 6.26 [5.23, 8.34] | 4.00 | 1.00 | 27.15 [27.10, 27.23] | 109.67 [108.00, 111.00] | 109.67 [109.00, 111.00] |
| separate_ends--proposal_confirmation--chronological--budget-40 | 74.84 [74.32, 75.27] | 85.30 [84.47, 86.02] | 79.73 [79.07, 80.29] | 84.47 [82.61, 85.40] | 62.63 [62.42, 63.04] | 94.84 [94.61, 95.01] | 12.65 [8.39, 18.08] | 6.33 [6.00, 7.00] | 3.33 [3.00, 4.00] | 54.39 [54.18, 54.69] | 221.00 [220.00, 222.00] | 219.00 [217.00, 220.00] |
| separate_ends--proposal_confirmation--evidence--budget-05 | 66.94 [66.76, 67.31] | 75.67 [75.47, 76.09] | 71.04 [70.85, 71.43] | 73.60 [73.29, 73.91] | 43.37 [42.86, 44.10] | 95.37 | 0.00 | 3.00 | 0.00 | 6.30 [6.18, 6.36] | 44.67 [44.00, 45.00] | 43.00 |
| separate_ends--proposal_confirmation--evidence--budget-10 | 67.95 [67.31, 68.41] | 76.81 [76.09, 77.33] | 72.11 [71.43, 72.59] | 73.29 [72.67, 74.53] | 45.55 [44.72, 45.96] | 95.37 | 0.00 | 3.00 | 0.00 | 13.10 [12.92, 13.25] | 84.00 [83.00, 85.00] | 81.33 [81.00, 82.00] |
| separate_ends--proposal_confirmation--evidence--budget-20 | 70.24 [69.51, 70.60] | 79.40 [78.57, 79.81] | 74.54 [73.76, 74.93] | 75.36 [74.53, 77.02] | 53.93 [52.48, 54.97] | 95.37 | 0.00 | 3.00 | 0.00 | 26.62 [26.52, 26.72] | 150.00 [149.00, 151.00] | 144.67 [143.00, 146.00] |
| separate_ends--proposal_confirmation--evidence--budget-40 | 74.59 [73.97, 75.34] | 84.47 [83.85, 85.40] | 79.22 [78.60, 80.06] | 81.47 [80.12, 82.92] | 64.70 [63.66, 65.53] | 94.90 [94.67, 95.08] | 11.08 [6.81, 16.50] | 5.33 [5.00, 6.00] | 2.33 [2.00, 3.00] | 53.70 [53.51, 54.04] | 244.33 [243.00, 246.00] | 239.00 [237.00, 240.00] |
| typed_starts--proposal_confirmation--chronological--budget-05 | 63.19 [62.91, 63.46] | 71.43 [71.12, 71.74] | 67.06 [66.76, 67.35] | 73.71 [73.29, 74.22] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 6.53 [6.49, 6.56] | 29.00 [28.00, 30.00] | 27.67 [26.00, 29.00] |
| typed_starts--proposal_confirmation--chronological--budget-10 | 63.59 [63.01, 63.93] | 72.15 [71.43, 72.67] | 67.60 [66.96, 68.02] | 77.23 [76.71, 77.64] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 13.19 [13.10, 13.23] | 54.67 [54.00, 55.00] | 54.33 [53.00, 56.00] |
| typed_starts--proposal_confirmation--chronological--budget-20 | 64.97 [64.75, 65.12] | 73.91 [73.60, 74.22] | 69.15 [68.90, 69.38] | 80.95 [80.12, 81.37] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 27.25 [27.14, 27.42] | 106.33 [105.00, 108.00] | 108.00 [107.00, 109.00] |
| typed_starts--proposal_confirmation--chronological--budget-40 | 66.48 [65.57, 67.12] | 75.78 [74.53, 76.71] | 70.83 [69.77, 71.59] | 87.78 [86.34, 89.44] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 53.43 [52.99, 54.22] | 207.67 [205.00, 211.00] | 208.00 [203.00, 213.00] |
| typed_starts--proposal_confirmation--evidence--budget-05 | 64.47 [64.29, 64.56] | 72.88 [72.67, 72.98] | 68.42 [68.22, 68.51] | 74.22 [73.91, 74.84] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 6.27 [6.22, 6.34] | 42.67 [42.00, 43.00] | 40.67 [40.00, 41.00] |
| typed_starts--proposal_confirmation--evidence--budget-10 | 64.47 [64.29, 64.56] | 72.88 [72.67, 72.98] | 68.42 [68.22, 68.51] | 74.43 [73.60, 75.16] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 13.04 [13.02, 13.05] | 79.33 [79.00, 80.00] | 75.00 [74.00, 76.00] |
| typed_starts--proposal_confirmation--evidence--budget-20 | 65.29 [64.56, 65.93] | 73.81 [72.98, 74.53] | 69.29 [68.51, 69.97] | 79.19 [78.26, 80.75] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 26.51 [26.33, 26.79] | 138.00 [136.00, 141.00] | 133.67 [131.00, 136.00] |
| typed_starts--proposal_confirmation--evidence--budget-40 | 66.42 [65.57, 66.94] | 75.57 [74.53, 76.09] | 70.70 [69.77, 71.22] | 85.82 [85.09, 86.65] | 37.89 | 95.37 | 0.00 | 3.00 | 0.00 | 52.17 [51.98, 52.47] | 213.33 [210.00, 219.00] | 211.33 [206.00, 217.00] |

## Full-parent human correction ceiling

The human replaces selected parent portions with every gold-core intersection, including rallies absent from model proposals, and corrects both endpoints wherever visible. This is an optimistic full annotation operation, not proposal-only confirmation. It preserves all production-covered core within selected parents but cannot add video outside production. Unselected parents remain original.

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses | Review min | Parent jobs | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_start--full_parent--chronological--budget-05 | 64.25 [64.09, 64.46] | 72.36 [72.05, 72.67] | 68.06 [67.84, 68.32] | 72.98 [72.05, 73.91] | 41.61 [41.30, 41.93] | 95.37 | 0.00 | 3.00 | 0.00 | 6.54 [6.49, 6.59] | 29.00 [28.00, 30.00] | 27.67 [26.00, 29.00] |
| first_start--full_parent--chronological--budget-10 | 66.64 [66.30, 67.03] | 75.26 [74.84, 75.78] | 70.69 [70.45, 71.14] | 76.92 [76.71, 77.02] | 45.55 [44.72, 45.96] | 95.37 | 0.00 | 3.00 | 0.00 | 13.24 [13.09, 13.31] | 54.33 [53.00, 55.00] | 54.00 [53.00, 56.00] |
| first_start--full_parent--chronological--budget-20 | 71.56 [70.92, 71.90] | 81.26 [81.06, 81.68] | 76.10 [75.65, 76.45] | 80.02 [79.81, 80.12] | 54.14 [53.73, 54.66] | 95.37 | 0.00 | 3.00 | 0.00 | 27.27 [27.15, 27.49] | 106.00 [105.00, 107.00] | 107.67 [106.00, 109.00] |
| first_start--full_parent--chronological--budget-40 | 79.56 [79.22, 79.95] | 89.86 [88.82, 90.37] | 84.39 [83.75, 84.84] | 87.16 [86.96, 87.27] | 69.98 [68.32, 70.81] | 95.37 | 0.00 | 3.00 | 0.00 | 53.01 [52.30, 54.17] | 208.00 [205.00, 212.00] | 207.67 [202.00, 214.00] |
| first_start--full_parent--evidence--budget-05 | 67.62 [67.22, 67.96] | 76.09 [75.78, 76.40] | 71.60 [71.24, 71.93] | 73.40 [73.29, 73.60] | 43.58 [43.17, 44.10] | 95.37 | 0.00 | 3.00 | 0.00 | 6.29 [6.26, 6.34] | 42.67 [42.00, 43.00] | 41.00 |
| first_start--full_parent--evidence--budget-10 | 69.69 [68.89, 70.28] | 77.85 [77.02, 78.57] | 73.55 [72.73, 74.19] | 73.29 [72.98, 73.60] | 48.14 [47.52, 49.38] | 95.37 | 0.00 | 3.00 | 0.00 | 13.05 [13.03, 13.06] | 79.33 [79.00, 80.00] | 75.00 [74.00, 76.00] |
| first_start--full_parent--evidence--budget-20 | 73.47 [72.70, 74.52] | 81.99 [81.06, 83.54] | 77.49 [76.65, 78.77] | 77.54 [77.02, 77.95] | 57.14 [55.90, 58.39] | 95.37 | 0.00 | 3.00 | 0.00 | 26.44 [26.23, 26.79] | 137.67 [136.00, 141.00] | 133.00 [130.00, 136.00] |
| first_start--full_parent--evidence--budget-40 | 79.45 [79.17, 79.89] | 89.23 [88.51, 90.06] | 84.06 [83.58, 84.67] | 85.30 [83.54, 86.34] | 69.98 [68.01, 71.12] | 95.37 | 0.00 | 3.00 | 0.00 | 51.82 [51.36, 52.10] | 211.67 [208.00, 218.00] | 209.33 [204.00, 216.00] |
| head_refined--full_parent--chronological--budget-05 | 64.34 [64.19, 64.46] | 72.46 [72.36, 72.67] | 68.16 [68.03, 68.32] | 71.22 [70.81, 72.05] | 41.41 [40.99, 41.93] | 95.37 | 0.00 | 3.00 | 0.00 | 6.68 [6.60, 6.76] | 32.67 [32.00, 34.00] | 31.33 [30.00, 33.00] |
| head_refined--full_parent--chronological--budget-10 | 67.37 [67.31, 67.49] | 76.09 | 71.46 [71.43, 71.53] | 72.77 [72.36, 72.98] | 45.34 [44.72, 45.96] | 95.37 | 0.00 | 3.00 | 0.00 | 13.36 [13.33, 13.39] | 60.00 [59.00, 61.00] | 58.67 [57.00, 60.00] |
| head_refined--full_parent--chronological--budget-20 | 71.39 [71.04, 71.90] | 80.85 [80.75, 81.06] | 75.83 [75.58, 76.20] | 75.26 [74.84, 75.78] | 53.11 [52.48, 54.04] | 95.37 | 0.00 | 3.00 | 0.00 | 27.09 [26.98, 27.15] | 110.67 [109.00, 112.00] | 110.33 [110.00, 111.00] |
| head_refined--full_parent--chronological--budget-40 | 80.75 [80.55, 80.99] | 91.20 [90.99, 91.30] | 85.66 [85.55, 85.84] | 79.40 [78.88, 79.81] | 71.22 [70.81, 71.74] | 95.37 | 0.00 | 3.00 | 0.00 | 54.48 [54.37, 54.57] | 222.33 [221.00, 223.00] | 220.00 |
| head_refined--full_parent--evidence--budget-05 | 68.14 [67.96, 68.42] | 76.60 [76.40, 76.71] | 72.12 [71.93, 72.33] | 71.84 [71.43, 72.05] | 44.10 | 95.37 | 0.00 | 3.00 | 0.00 | 6.37 [6.18, 6.49] | 45.67 [44.00, 47.00] | 43.67 [43.00, 44.00] |
| head_refined--full_parent--evidence--budget-10 | 69.56 [68.70, 70.08] | 78.05 [77.02, 78.57] | 73.56 [72.62, 74.08] | 69.57 [68.94, 70.50] | 46.89 [45.96, 47.52] | 95.37 | 0.00 | 3.00 | 0.00 | 13.09 [12.96, 13.25] | 84.67 [84.00, 85.00] | 82.00 |
| head_refined--full_parent--evidence--budget-20 | 73.95 [73.46, 74.58] | 82.30 [81.68, 82.92] | 77.90 [77.35, 78.53] | 70.60 [70.19, 71.12] | 57.35 [57.14, 57.45] | 95.37 | 0.00 | 3.00 | 0.00 | 26.60 [26.57, 26.65] | 151.00 [150.00, 152.00] | 145.33 [145.00, 146.00] |
| head_refined--full_parent--evidence--budget-40 | 81.02 [80.83, 81.16] | 90.58 [90.37, 90.99] | 85.53 [85.34, 85.80] | 75.98 [75.16, 77.02] | 73.81 [73.29, 74.84] | 95.37 | 0.00 | 3.00 | 0.00 | 53.55 [53.17, 54.04] | 245.67 [243.00, 247.00] | 239.33 [237.00, 242.00] |
| separate_ends--full_parent--chronological--budget-05 | 64.34 [64.19, 64.46] | 72.46 [72.36, 72.67] | 68.16 [68.03, 68.32] | 71.43 [71.12, 72.05] | 41.41 [40.99, 41.93] | 95.37 | 0.00 | 3.00 | 0.00 | 6.62 [6.60, 6.66] | 32.00 [31.00, 33.00] | 30.67 [29.00, 32.00] |
| separate_ends--full_parent--chronological--budget-10 | 67.46 [67.31, 67.58] | 76.19 [76.09, 76.40] | 71.56 [71.43, 71.72] | 73.29 [72.67, 73.60] | 45.45 [44.72, 45.96] | 95.37 | 0.00 | 3.00 | 0.00 | 13.42 [13.35, 13.45] | 60.00 [59.00, 61.00] | 58.67 [57.00, 60.00] |
| separate_ends--full_parent--chronological--budget-20 | 71.42 [71.04, 71.90] | 80.95 [80.75, 81.06] | 75.89 [75.58, 76.20] | 75.78 [75.16, 76.40] | 53.11 [52.80, 53.42] | 95.37 | 0.00 | 3.00 | 0.00 | 27.15 [27.10, 27.23] | 109.67 [108.00, 111.00] | 109.67 [109.00, 111.00] |
| separate_ends--full_parent--chronological--budget-40 | 80.68 [80.44, 81.10] | 91.20 [90.68, 91.93] | 85.62 [85.26, 86.17] | 79.92 [79.81, 80.12] | 71.12 [70.19, 72.67] | 95.37 | 0.00 | 3.00 | 0.00 | 54.39 [54.18, 54.69] | 221.00 [220.00, 222.00] | 219.00 [217.00, 220.00] |
| separate_ends--full_parent--evidence--budget-05 | 67.99 [67.96, 68.04] | 76.50 [76.40, 76.71] | 71.99 [71.93, 72.12] | 72.26 [72.05, 72.67] | 44.10 | 95.37 | 0.00 | 3.00 | 0.00 | 6.30 [6.18, 6.36] | 44.67 [44.00, 45.00] | 43.00 |
| separate_ends--full_parent--evidence--budget-10 | 69.56 [68.51, 70.28] | 78.05 [77.02, 78.57] | 73.56 [72.51, 74.19] | 69.88 [69.25, 70.81] | 47.10 [46.27, 47.52] | 95.37 | 0.00 | 3.00 | 0.00 | 13.10 [12.92, 13.25] | 84.00 [83.00, 85.00] | 81.33 [81.00, 82.00] |
| separate_ends--full_parent--evidence--budget-20 | 73.79 [73.18, 74.09] | 82.19 [81.37, 82.61] | 77.77 [77.06, 78.12] | 71.43 [70.81, 72.05] | 57.66 [57.14, 58.39] | 95.37 | 0.00 | 3.00 | 0.00 | 26.62 [26.52, 26.72] | 150.00 [149.00, 151.00] | 144.67 [143.00, 146.00] |
| separate_ends--full_parent--evidence--budget-40 | 80.76 [80.61, 80.83] | 90.37 | 85.30 [85.21, 85.34] | 76.60 [76.09, 77.33] | 73.81 [72.98, 74.53] | 95.37 | 0.00 | 3.00 | 0.00 | 53.70 [53.51, 54.04] | 244.33 [243.00, 246.00] | 239.00 [237.00, 240.00] |
| typed_starts--full_parent--chronological--budget-05 | 64.15 [64.09, 64.19] | 72.26 [72.05, 72.36] | 67.96 [67.84, 68.03] | 72.88 [72.05, 73.60] | 41.61 [41.30, 41.93] | 95.37 | 0.00 | 3.00 | 0.00 | 6.53 [6.49, 6.56] | 29.00 [28.00, 30.00] | 27.67 [26.00, 29.00] |
| typed_starts--full_parent--chronological--budget-10 | 66.54 [66.30, 66.76] | 75.16 [74.84, 75.47] | 70.59 [70.45, 70.85] | 76.81 [76.71, 77.02] | 45.45 [44.72, 45.96] | 95.37 | 0.00 | 3.00 | 0.00 | 13.19 [13.10, 13.23] | 54.67 [54.00, 55.00] | 54.33 [53.00, 56.00] |
| typed_starts--full_parent--chronological--budget-20 | 71.47 [70.92, 71.90] | 81.16 [81.06, 81.37] | 76.01 [75.65, 76.20] | 79.81 [79.50, 80.12] | 54.04 [53.73, 54.66] | 95.37 | 0.00 | 3.00 | 0.00 | 27.25 [27.14, 27.42] | 106.33 [105.00, 108.00] | 108.00 [107.00, 109.00] |
| typed_starts--full_parent--chronological--budget-40 | 79.78 [79.51, 80.27] | 90.27 [89.44, 90.99] | 84.70 [84.21, 85.30] | 87.37 [86.65, 87.89] | 70.60 [68.94, 71.43] | 95.37 | 0.00 | 3.00 | 0.00 | 53.43 [52.99, 54.22] | 207.67 [205.00, 211.00] | 208.00 [203.00, 213.00] |
| typed_starts--full_parent--evidence--budget-05 | 67.68 [67.40, 67.96] | 76.09 [75.78, 76.40] | 71.64 [71.35, 71.93] | 73.40 [73.29, 73.60] | 43.58 [43.17, 44.10] | 95.37 | 0.00 | 3.00 | 0.00 | 6.27 [6.22, 6.34] | 42.67 [42.00, 43.00] | 40.67 [40.00, 41.00] |
| typed_starts--full_parent--evidence--budget-10 | 69.69 [68.89, 70.28] | 77.85 [77.02, 78.57] | 73.55 [72.73, 74.19] | 73.08 [72.67, 73.60] | 48.14 [47.52, 49.38] | 95.37 | 0.00 | 3.00 | 0.00 | 13.04 [13.02, 13.05] | 79.33 [79.00, 80.00] | 75.00 [74.00, 76.00] |
| typed_starts--full_parent--evidence--budget-20 | 73.49 [72.70, 74.52] | 82.09 [81.06, 83.54] | 77.55 [76.65, 78.77] | 77.85 [77.02, 78.88] | 57.35 [55.90, 58.70] | 95.37 | 0.00 | 3.00 | 0.00 | 26.51 [26.33, 26.79] | 138.00 [136.00, 141.00] | 133.67 [131.00, 136.00] |
| typed_starts--full_parent--evidence--budget-40 | 79.58 [79.28, 79.95] | 89.54 [89.13, 90.37] | 84.27 [83.92, 84.84] | 84.78 [83.54, 85.71] | 70.70 [68.94, 72.05] | 95.37 | 0.00 | 3.00 | 0.00 | 52.17 [51.98, 52.47] | 213.33 [210.00, 219.00] | 211.33 [206.00, 217.00] |

## Review workload and candidate diagnostics

Typed metrics in these reviewed rows still describe original selected candidates, not gold-corrected output. They expose what the review queue selected. Full-parent output may contain additional gold rallies beyond these candidates. Null acceptance counts mean that the full-parent mode does not classify individual candidate acceptance.

| Arm | Flags reviewed | Candidates reviewed | Accepted candidates | Rejected candidates | Corrected types | Original wrong types @1s | Original joint F1 @1s % | Unused budget min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_start--full_parent--chronological--budget-05 | 29.00 [28.00, 30.00] | 29.00 [28.00, 30.00] | n/a | n/a | n/a | 0.00 | 5.08 [3.97, 5.65] | 0.36 [0.31, 0.41] |
| first_start--full_parent--chronological--budget-10 | 54.33 [53.00, 55.00] | 54.33 [53.00, 55.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 9.49 [8.47, 10.53] | 0.54 [0.47, 0.69] |
| first_start--full_parent--chronological--budget-20 | 106.00 [105.00, 107.00] | 106.00 [105.00, 107.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 15.00 [13.46, 16.20] | 0.30 [0.08, 0.42] |
| first_start--full_parent--chronological--budget-40 | 208.00 [205.00, 212.00] | 208.00 [205.00, 212.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 21.26 [19.92, 21.97] | 2.12 [0.96, 2.84] |
| first_start--full_parent--evidence--budget-05 | 42.67 [42.00, 43.00] | 42.67 [42.00, 43.00] | n/a | n/a | n/a | 0.00 | 6.71 [5.98, 7.08] | 0.60 [0.55, 0.63] |
| first_start--full_parent--evidence--budget-10 | 79.33 [79.00, 80.00] | 79.33 [79.00, 80.00] | n/a | n/a | n/a | 0.00 | 11.70 [9.90, 13.83] | 0.74 [0.72, 0.75] |
| first_start--full_parent--evidence--budget-20 | 137.67 [136.00, 141.00] | 137.67 [136.00, 141.00] | n/a | n/a | n/a | 0.00 | 16.99 [15.62, 19.31] | 1.12 [0.78, 1.33] |
| first_start--full_parent--evidence--budget-40 | 211.67 [208.00, 218.00] | 211.67 [208.00, 218.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 21.86 [20.22, 23.26] | 3.31 [3.04, 3.77] |
| first_start--proposal_confirmation--chronological--budget-05 | 29.00 [28.00, 30.00] | 29.00 [28.00, 30.00] | 21.33 [18.00, 25.00] | 7.67 [5.00, 10.00] | 0.00 | 0.00 | 5.08 [3.97, 5.65] | 0.36 [0.31, 0.41] |
| first_start--proposal_confirmation--chronological--budget-10 | 54.33 [53.00, 55.00] | 54.33 [53.00, 55.00] | 44.33 [42.00, 47.00] | 10.00 [8.00, 13.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 9.49 [8.47, 10.53] | 0.54 [0.47, 0.69] |
| first_start--proposal_confirmation--chronological--budget-20 | 106.00 [105.00, 107.00] | 106.00 [105.00, 107.00] | 85.33 [82.00, 89.00] | 20.67 [16.00, 25.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 15.00 [13.46, 16.20] | 0.30 [0.08, 0.42] |
| first_start--proposal_confirmation--chronological--budget-40 | 208.00 [205.00, 212.00] | 208.00 [205.00, 212.00] | 166.00 [160.00, 174.00] | 42.00 [38.00, 45.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 21.26 [19.92, 21.97] | 2.12 [0.96, 2.84] |
| first_start--proposal_confirmation--evidence--budget-05 | 42.67 [42.00, 43.00] | 42.67 [42.00, 43.00] | 35.33 [33.00, 37.00] | 7.33 [5.00, 10.00] | 0.00 | 0.00 | 6.71 [5.98, 7.08] | 0.60 [0.55, 0.63] |
| first_start--proposal_confirmation--evidence--budget-10 | 79.33 [79.00, 80.00] | 79.33 [79.00, 80.00] | 63.00 [61.00, 66.00] | 16.33 [14.00, 18.00] | 0.00 | 0.00 | 11.70 [9.90, 13.83] | 0.74 [0.72, 0.75] |
| first_start--proposal_confirmation--evidence--budget-20 | 137.67 [136.00, 141.00] | 137.67 [136.00, 141.00] | 111.00 [106.00, 117.00] | 26.67 [24.00, 30.00] | 0.00 | 0.00 | 16.99 [15.62, 19.31] | 1.12 [0.78, 1.33] |
| first_start--proposal_confirmation--evidence--budget-40 | 211.67 [208.00, 218.00] | 211.67 [208.00, 218.00] | 168.00 [163.00, 176.00] | 43.67 [42.00, 45.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 21.86 [20.22, 23.26] | 3.31 [3.04, 3.77] |
| head_refined--full_parent--chronological--budget-05 | 59.33 [57.00, 61.00] | 33.67 [33.00, 34.00] | n/a | n/a | n/a | 0.00 | 8.76 [7.26, 10.64] | 0.21 [0.14, 0.29] |
| head_refined--full_parent--chronological--budget-10 | 107.00 [105.00, 109.00] | 63.33 [63.00, 64.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 13.97 [10.31, 17.10] | 0.42 [0.39, 0.45] |
| head_refined--full_parent--chronological--budget-20 | 207.33 [199.00, 216.00] | 121.33 [118.00, 123.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 21.74 [17.04, 26.76] | 0.48 [0.42, 0.58] |
| head_refined--full_parent--chronological--budget-40 | 412.33 [406.00, 419.00] | 241.33 [240.00, 243.00] | n/a | n/a | n/a | 1.33 [0.00, 2.00] | 37.15 [33.45, 42.93] | 0.65 [0.56, 0.76] |
| head_refined--full_parent--evidence--budget-05 | 74.67 [72.00, 77.00] | 45.67 [44.00, 47.00] | n/a | n/a | n/a | 0.00 | 13.02 [11.44, 14.63] | 0.52 [0.41, 0.71] |
| head_refined--full_parent--evidence--budget-10 | 136.67 [133.00, 140.00] | 85.33 [84.00, 87.00] | n/a | n/a | n/a | 0.00 | 22.26 [19.61, 26.11] | 0.70 [0.53, 0.83] |
| head_refined--full_parent--evidence--budget-20 | 252.00 [250.00, 254.00] | 152.67 [152.00, 153.00] | n/a | n/a | n/a | 0.00 | 31.41 [27.91, 36.17] | 0.97 [0.92, 1.00] |
| head_refined--full_parent--evidence--budget-40 | 434.00 [428.00, 441.00] | 257.33 [255.00, 259.00] | n/a | n/a | n/a | 0.00 | 41.02 [38.06, 45.22] | 1.58 [1.10, 1.96] |
| head_refined--proposal_confirmation--chronological--budget-05 | 59.33 [57.00, 61.00] | 33.67 [33.00, 34.00] | 27.67 [25.00, 30.00] | 6.00 [4.00, 8.00] | 0.00 | 0.00 | 8.76 [7.26, 10.64] | 0.21 [0.14, 0.29] |
| head_refined--proposal_confirmation--chronological--budget-10 | 107.00 [105.00, 109.00] | 63.33 [63.00, 64.00] | 53.67 [51.00, 56.00] | 9.67 [8.00, 12.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 13.97 [10.31, 17.10] | 0.42 [0.39, 0.45] |
| head_refined--proposal_confirmation--chronological--budget-20 | 207.33 [199.00, 216.00] | 121.33 [118.00, 123.00] | 99.00 [98.00, 101.00] | 22.33 [20.00, 25.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 21.74 [17.04, 26.76] | 0.48 [0.42, 0.58] |
| head_refined--proposal_confirmation--chronological--budget-40 | 412.33 [406.00, 419.00] | 241.33 [240.00, 243.00] | 195.67 [194.00, 197.00] | 45.67 [43.00, 47.00] | 1.33 [0.00, 2.00] | 1.33 [0.00, 2.00] | 37.15 [33.45, 42.93] | 0.65 [0.56, 0.76] |
| head_refined--proposal_confirmation--evidence--budget-05 | 74.67 [72.00, 77.00] | 45.67 [44.00, 47.00] | 38.33 [38.00, 39.00] | 7.33 [6.00, 9.00] | 0.00 | 0.00 | 13.02 [11.44, 14.63] | 0.52 [0.41, 0.71] |
| head_refined--proposal_confirmation--evidence--budget-10 | 136.67 [133.00, 140.00] | 85.33 [84.00, 87.00] | 72.67 [72.00, 73.00] | 12.67 [11.00, 15.00] | 0.00 | 0.00 | 22.26 [19.61, 26.11] | 0.70 [0.53, 0.83] |
| head_refined--proposal_confirmation--evidence--budget-20 | 252.00 [250.00, 254.00] | 152.67 [152.00, 153.00] | 128.00 [127.00, 130.00] | 24.67 [23.00, 26.00] | 0.00 | 0.00 | 31.41 [27.91, 36.17] | 0.97 [0.92, 1.00] |
| head_refined--proposal_confirmation--evidence--budget-40 | 434.00 [428.00, 441.00] | 257.33 [255.00, 259.00] | 210.00 [208.00, 211.00] | 47.33 [44.00, 51.00] | 0.00 | 0.00 | 41.02 [38.06, 45.22] | 1.58 [1.10, 1.96] |
| separate_ends--full_parent--chronological--budget-05 | 58.67 [58.00, 60.00] | 33.00 [32.00, 34.00] | n/a | n/a | n/a | 0.00 | 5.48 [5.08, 6.23] | 0.27 [0.23, 0.29] |
| separate_ends--full_parent--chronological--budget-10 | 106.00 [103.00, 108.00] | 63.33 [63.00, 64.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 9.51 [8.95, 10.08] | 0.37 [0.33, 0.43] |
| separate_ends--full_parent--chronological--budget-20 | 207.67 [198.00, 214.00] | 121.00 [118.00, 123.00] | n/a | n/a | n/a | 0.67 [0.00, 1.00] | 17.31 [16.36, 18.48] | 0.42 [0.33, 0.46] |
| separate_ends--full_parent--chronological--budget-40 | 408.33 [401.00, 417.00] | 240.33 [239.00, 242.00] | n/a | n/a | n/a | 1.33 [0.00, 2.00] | 30.98 [28.62, 32.33] | 0.74 [0.44, 0.96] |
| separate_ends--full_parent--evidence--budget-05 | 71.00 [66.00, 75.00] | 44.67 [44.00, 45.00] | n/a | n/a | n/a | 0.00 | 10.25 [9.39, 11.44] | 0.59 [0.53, 0.71] |
| separate_ends--full_parent--evidence--budget-10 | 135.67 [134.00, 138.00] | 85.00 [84.00, 87.00] | n/a | n/a | n/a | 0.00 | 17.63 [16.62, 18.50] | 0.68 [0.53, 0.86] |
| separate_ends--full_parent--evidence--budget-20 | 247.67 [246.00, 251.00] | 151.33 [150.00, 153.00] | n/a | n/a | n/a | 0.00 | 26.97 [25.82, 29.26] | 0.95 [0.84, 1.04] |
| separate_ends--full_parent--evidence--budget-40 | 426.67 [418.00, 438.00] | 256.00 [255.00, 258.00] | n/a | n/a | n/a | 0.00 | 34.84 [32.48, 36.49] | 1.43 [1.10, 1.63] |
| separate_ends--proposal_confirmation--chronological--budget-05 | 58.67 [58.00, 60.00] | 33.00 [32.00, 34.00] | 19.67 [17.00, 22.00] | 13.33 [11.00, 15.00] | 0.00 | 0.00 | 5.48 [5.08, 6.23] | 0.27 [0.23, 0.29] |
| separate_ends--proposal_confirmation--chronological--budget-10 | 106.00 [103.00, 108.00] | 63.33 [63.00, 64.00] | 41.00 [39.00, 45.00] | 22.33 [19.00, 24.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 9.51 [8.95, 10.08] | 0.37 [0.33, 0.43] |
| separate_ends--proposal_confirmation--chronological--budget-20 | 207.67 [198.00, 214.00] | 121.00 [118.00, 123.00] | 79.00 [75.00, 86.00] | 42.00 [36.00, 47.00] | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 17.31 [16.36, 18.48] | 0.42 [0.33, 0.46] |
| separate_ends--proposal_confirmation--chronological--budget-40 | 408.33 [401.00, 417.00] | 240.33 [239.00, 242.00] | 162.00 [156.00, 170.00] | 78.33 [72.00, 83.00] | 1.33 [0.00, 2.00] | 1.33 [0.00, 2.00] | 30.98 [28.62, 32.33] | 0.74 [0.44, 0.96] |
| separate_ends--proposal_confirmation--evidence--budget-05 | 71.00 [66.00, 75.00] | 44.67 [44.00, 45.00] | 32.00 [29.00, 35.00] | 12.67 [10.00, 16.00] | 0.00 | 0.00 | 10.25 [9.39, 11.44] | 0.59 [0.53, 0.71] |
| separate_ends--proposal_confirmation--evidence--budget-10 | 135.67 [134.00, 138.00] | 85.00 [84.00, 87.00] | 58.67 [55.00, 63.00] | 26.33 [21.00, 29.00] | 0.00 | 0.00 | 17.63 [16.62, 18.50] | 0.68 [0.53, 0.86] |
| separate_ends--proposal_confirmation--evidence--budget-20 | 247.67 [246.00, 251.00] | 151.33 [150.00, 153.00] | 105.33 [97.00, 112.00] | 46.00 [39.00, 53.00] | 0.00 | 0.00 | 26.97 [25.82, 29.26] | 0.95 [0.84, 1.04] |
| separate_ends--proposal_confirmation--evidence--budget-40 | 426.67 [418.00, 438.00] | 256.00 [255.00, 258.00] | 173.67 [166.00, 186.00] | 82.33 [72.00, 89.00] | 0.00 | 0.00 | 34.84 [32.48, 36.49] | 1.43 [1.10, 1.63] |
| typed_starts--full_parent--chronological--budget-05 | 29.67 [29.00, 30.00] | 30.00 [29.00, 31.00] | n/a | n/a | n/a | 0.00 | 4.88 [3.95, 5.63] | 0.36 [0.33, 0.41] |
| typed_starts--full_parent--chronological--budget-10 | 58.33 [56.00, 60.00] | 59.33 [57.00, 61.00] | n/a | n/a | n/a | 0.33 [0.00, 1.00] | 9.21 [7.85, 10.42] | 0.59 [0.55, 0.68] |
| typed_starts--full_parent--chronological--budget-20 | 117.33 [116.00, 119.00] | 118.33 [117.00, 120.00] | n/a | n/a | n/a | 0.67 [0.00, 1.00] | 14.15 [11.76, 16.22] | 0.31 [0.14, 0.43] |
| typed_starts--full_parent--chronological--budget-40 | 227.33 [225.00, 231.00] | 229.67 [228.00, 232.00] | n/a | n/a | n/a | 1.67 [1.00, 2.00] | 19.52 [17.72, 20.69] | 1.71 [0.91, 2.14] |
| typed_starts--full_parent--evidence--budget-05 | 43.00 | 43.00 | n/a | n/a | n/a | 0.00 | 6.52 [5.98, 7.07] | 0.62 [0.55, 0.67] |
| typed_starts--full_parent--evidence--budget-10 | 80.33 [80.00, 81.00] | 80.33 [80.00, 81.00] | n/a | n/a | n/a | 0.00 | 11.68 [9.85, 13.83] | 0.75 [0.73, 0.76] |
| typed_starts--full_parent--evidence--budget-20 | 141.33 [139.00, 144.00] | 141.67 [139.00, 144.00] | n/a | n/a | n/a | 0.00 | 16.42 [14.99, 18.34] | 1.05 [0.78, 1.24] |
| typed_starts--full_parent--evidence--budget-40 | 230.33 [228.00, 235.00] | 233.00 [231.00, 236.00] | n/a | n/a | n/a | 1.33 [1.00, 2.00] | 20.13 [18.02, 21.66] | 2.97 [2.67, 3.16] |
| typed_starts--proposal_confirmation--chronological--budget-05 | 29.67 [29.00, 30.00] | 30.00 [29.00, 31.00] | 21.67 [18.00, 25.00] | 8.33 [5.00, 11.00] | 0.00 | 0.00 | 4.88 [3.95, 5.63] | 0.36 [0.33, 0.41] |
| typed_starts--proposal_confirmation--chronological--budget-10 | 58.33 [56.00, 60.00] | 59.33 [57.00, 61.00] | 44.67 [42.00, 48.00] | 14.67 [12.00, 19.00] | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 9.21 [7.85, 10.42] | 0.59 [0.55, 0.68] |
| typed_starts--proposal_confirmation--chronological--budget-20 | 117.33 [116.00, 119.00] | 118.33 [117.00, 120.00] | 87.33 [85.00, 91.00] | 31.00 [27.00, 35.00] | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 14.15 [11.76, 16.22] | 0.31 [0.14, 0.43] |
| typed_starts--proposal_confirmation--chronological--budget-40 | 227.33 [225.00, 231.00] | 229.67 [228.00, 232.00] | 169.33 [163.00, 178.00] | 60.33 [54.00, 65.00] | 1.67 [1.00, 2.00] | 1.67 [1.00, 2.00] | 19.52 [17.72, 20.69] | 1.71 [0.91, 2.14] |
| typed_starts--proposal_confirmation--evidence--budget-05 | 43.00 | 43.00 | 35.00 [33.00, 36.00] | 8.00 [7.00, 10.00] | 0.00 | 0.00 | 6.52 [5.98, 7.07] | 0.62 [0.55, 0.67] |
| typed_starts--proposal_confirmation--evidence--budget-10 | 80.33 [80.00, 81.00] | 80.33 [80.00, 81.00] | 63.00 [61.00, 66.00] | 17.33 [14.00, 19.00] | 0.00 | 0.00 | 11.68 [9.85, 13.83] | 0.75 [0.73, 0.76] |
| typed_starts--proposal_confirmation--evidence--budget-20 | 141.33 [139.00, 144.00] | 141.67 [139.00, 144.00] | 111.67 [107.00, 117.00] | 30.00 [27.00, 32.00] | 0.00 | 0.00 | 16.42 [14.99, 18.34] | 1.05 [0.78, 1.24] |
| typed_starts--proposal_confirmation--evidence--budget-40 | 230.33 [228.00, 235.00] | 233.00 [231.00, 236.00] | 171.67 [167.00, 179.00] | 61.33 [57.00, 65.00] | 1.33 [1.00, 2.00] | 1.33 [1.00, 2.00] | 20.13 [18.02, 21.66] | 2.97 [2.67, 3.16] |

## Source-group sensitivity

Below: production, all automatic arms, and 10% evidence-order review arms. Every arm and every numeric metric is retained for all source groups and individual recordings in summary JSON, including all padding and tolerance cases. These slices are not independent repetitions of the dataset.

### source-group-005

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses | Review min | Parent jobs | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 62.11 | 69.41 | 65.56 | 71.76 | 35.29 | 94.71 | 0.00 | 2.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--first_start | 58.60 [55.79, 60.00] | 65.49 [62.35, 67.06] | 61.85 [58.89, 63.33] | 66.27 [63.53, 71.76] | 35.29 | 87.23 [85.67, 88.36] | 43.59 [37.01, 52.68] | 2.67 [2.00, 3.00] | 0.67 [0.00, 1.00] | 0.00 | 0.00 | 0.00 |
| automatic--head_refined | 55.42 [50.48, 63.37] | 67.06 [62.35, 75.29] | 60.68 [55.79, 68.82] | 83.53 [80.00, 87.06] | 41.96 [37.65, 47.06] | 78.86 [78.07, 79.40] | 92.37 [89.26, 96.99] | 2.33 [2.00, 3.00] | 0.33 [0.00, 1.00] | 0.00 | 0.00 | 0.00 |
| automatic--separate_ends | 51.83 [47.62, 55.45] | 62.75 [58.82, 65.88] | 56.76 [52.63, 60.22] | 67.06 [63.53, 72.94] | 33.73 [31.76, 35.29] | 75.73 [73.82, 77.64] | 110.60 [99.47, 121.74] | 2.67 [2.00, 3.00] | 0.67 [0.00, 1.00] | 0.00 | 0.00 | 0.00 |
| automatic--typed_starts | 54.06 [52.38, 55.34] | 65.49 [64.71, 67.06] | 59.22 [57.89, 60.64] | 67.06 [63.53, 72.94] | 35.29 | 87.23 [85.67, 88.36] | 43.59 [37.01, 52.68] | 2.67 [2.00, 3.00] | 0.67 [0.00, 1.00] | 0.00 | 0.00 | 0.00 |
| first_start--full_parent--evidence--budget-10 | 69.32 [68.82, 69.57] | 75.29 | 72.18 [71.91, 72.32] | 72.94 [71.76, 74.12] | 43.92 [43.53, 44.71] | 94.71 | 0.00 | 2.00 | 0.00 | 3.75 [3.65, 3.80] | 23.67 [23.00, 24.00] | 21.00 [20.00, 22.00] |
| first_start--proposal_confirmation--evidence--budget-10 | 62.11 | 69.41 | 65.56 | 76.47 [74.12, 78.82] | 35.29 | 94.71 | 0.00 | 2.00 | 0.00 | 3.75 [3.65, 3.80] | 23.67 [23.00, 24.00] | 21.00 [20.00, 22.00] |
| head_refined--full_parent--evidence--budget-10 | 69.42 [68.82, 69.89] | 75.69 [75.29, 76.47] | 72.42 [71.91, 73.03] | 69.02 [68.24, 69.41] | 43.92 [42.35, 44.71] | 94.71 | 0.00 | 2.00 | 0.00 | 3.75 [3.67, 3.80] | 25.00 | 22.67 [22.00, 23.00] |
| head_refined--proposal_confirmation--evidence--budget-10 | 67.37 | 75.29 | 71.11 | 71.37 [70.59, 71.76] | 43.92 [42.35, 44.71] | 94.71 | 0.00 | 2.00 | 0.00 | 3.75 [3.67, 3.80] | 25.00 | 22.67 [22.00, 23.00] |
| separate_ends--full_parent--evidence--budget-10 | 69.18 [68.82, 69.89] | 75.69 [75.29, 76.47] | 72.28 [71.91, 73.03] | 70.20 [68.24, 71.76] | 44.31 [43.53, 44.71] | 94.71 | 0.00 | 2.00 | 0.00 | 3.79 [3.72, 3.85] | 24.67 [24.00, 25.00] | 22.67 [22.00, 23.00] |
| separate_ends--proposal_confirmation--evidence--budget-10 | 66.32 [65.26, 67.37] | 74.12 [72.94, 75.29] | 70.00 [68.89, 71.11] | 75.69 [74.12, 76.47] | 41.18 [38.82, 43.53] | 94.71 | 0.00 | 2.00 | 0.00 | 3.79 [3.72, 3.85] | 24.67 [24.00, 25.00] | 22.67 [22.00, 23.00] |
| typed_starts--full_parent--evidence--budget-10 | 69.32 [68.82, 69.57] | 75.29 | 72.18 [71.91, 72.32] | 72.16 [70.59, 74.12] | 43.92 [43.53, 44.71] | 94.71 | 0.00 | 2.00 | 0.00 | 3.73 [3.65, 3.80] | 23.67 [23.00, 24.00] | 21.00 [20.00, 22.00] |
| typed_starts--proposal_confirmation--evidence--budget-10 | 62.11 | 69.41 | 65.56 | 76.47 [74.12, 78.82] | 35.29 | 94.71 | 0.00 | 2.00 | 0.00 | 3.73 [3.65, 3.80] | 23.67 [23.00, 24.00] | 21.00 [20.00, 22.00] |

### source-group-007

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses | Review min | Parent jobs | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 50.00 | 58.88 | 54.08 | 57.94 | 29.91 | 94.19 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--first_start | 51.59 [50.00, 53.17] | 60.75 [58.88, 62.62] | 55.79 [54.08, 57.51] | 66.98 [62.62, 72.90] | 29.91 | 88.72 [86.86, 89.94] | 34.08 [26.49, 45.66] | 1.67 [1.00, 3.00] | 0.67 [0.00, 2.00] | 0.00 | 0.00 | 0.00 |
| automatic--head_refined | 56.97 [54.01, 58.70] | 72.59 [69.16, 75.70] | 63.84 [60.66, 66.12] | 80.06 [78.50, 82.24] | 39.25 [34.58, 46.73] | 78.81 [76.93, 80.31] | 95.80 [86.44, 107.49] | 1.33 [1.00, 2.00] | 0.33 [0.00, 1.00] | 0.00 | 0.00 | 0.00 |
| automatic--separate_ends | 54.28 [53.62, 55.47] | 69.16 [67.29, 71.03] | 60.82 [59.75, 62.30] | 69.78 [65.42, 75.70] | 37.69 [35.51, 42.06] | 75.89 [72.61, 78.65] | 113.97 [96.74, 134.38] | 2.67 [2.00, 4.00] | 1.67 [1.00, 3.00] | 0.00 | 0.00 | 0.00 |
| automatic--typed_starts | 48.91 [45.99, 50.75] | 62.31 [58.88, 64.49] | 54.80 [51.64, 56.43] | 69.78 [65.42, 75.70] | 29.91 | 88.72 [86.86, 89.94] | 34.08 [26.49, 45.66] | 1.67 [1.00, 3.00] | 0.67 [0.00, 2.00] | 0.00 | 0.00 | 0.00 |
| first_start--full_parent--evidence--budget-10 | 55.46 [54.84, 56.00] | 64.80 [63.55, 65.42] | 59.77 [58.87, 60.34] | 60.44 [58.88, 62.62] | 33.96 [32.71, 36.45] | 94.19 | 0.00 | 1.00 | 0.00 | 4.20 [4.14, 4.23] | 24.00 [23.00, 25.00] | 23.00 |
| first_start--proposal_confirmation--evidence--budget-10 | 50.79 | 59.81 | 54.94 | 61.06 [59.81, 63.55] | 29.91 | 94.19 | 0.00 | 1.00 | 0.00 | 4.20 [4.14, 4.23] | 24.00 [23.00, 25.00] | 23.00 |
| head_refined--full_parent--evidence--budget-10 | 53.97 [53.17, 54.76] | 63.55 [62.62, 64.49] | 58.37 [57.51, 59.23] | 60.44 [57.94, 62.62] | 33.96 [32.71, 35.51] | 94.19 | 0.00 | 1.00 | 0.00 | 4.25 [4.09, 4.39] | 25.67 [25.00, 26.00] | 25.67 [25.00, 26.00] |
| head_refined--proposal_confirmation--evidence--budget-10 | 53.17 [52.38, 53.97] | 62.62 [61.68, 63.55] | 57.51 [56.65, 58.37] | 59.50 [57.94, 60.75] | 33.33 [32.71, 33.64] | 94.19 | 0.00 | 1.00 | 0.00 | 4.25 [4.09, 4.39] | 25.67 [25.00, 26.00] | 25.67 [25.00, 26.00] |
| separate_ends--full_parent--evidence--budget-10 | 53.97 [53.17, 54.76] | 63.55 [62.62, 64.49] | 58.37 [57.51, 59.23] | 60.44 [57.94, 62.62] | 33.33 [32.71, 33.64] | 94.19 | 0.00 | 1.00 | 0.00 | 4.24 [4.09, 4.36] | 25.67 [25.00, 26.00] | 25.67 [25.00, 26.00] |
| separate_ends--proposal_confirmation--evidence--budget-10 | 53.17 [52.38, 53.97] | 62.62 [61.68, 63.55] | 57.51 [56.65, 58.37] | 61.06 [59.81, 62.62] | 33.64 [32.71, 34.58] | 94.19 | 0.00 | 1.00 | 0.00 | 4.24 [4.09, 4.36] | 25.67 [25.00, 26.00] | 25.67 [25.00, 26.00] |
| typed_starts--full_parent--evidence--budget-10 | 55.46 [54.84, 56.00] | 64.80 [63.55, 65.42] | 59.77 [58.87, 60.34] | 60.44 [58.88, 62.62] | 33.96 [32.71, 36.45] | 94.19 | 0.00 | 1.00 | 0.00 | 4.20 [4.14, 4.23] | 24.00 [23.00, 25.00] | 23.00 |
| typed_starts--proposal_confirmation--evidence--budget-10 | 50.79 | 59.81 | 54.94 | 61.06 [59.81, 63.55] | 29.91 | 94.19 | 0.00 | 1.00 | 0.00 | 4.20 [4.14, 4.23] | 24.00 [23.00, 25.00] | 23.00 |

### source-group-009

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses | Review min | Parent jobs | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 86.08 | 91.89 | 88.89 | 87.84 | 54.05 | 97.74 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--first_start | 86.08 | 91.89 | 88.89 | 85.14 [83.78, 86.49] | 54.05 | 97.04 [96.84, 97.15] | 5.23 [4.37, 6.70] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_refined | 85.69 [81.93, 87.65] | 94.14 [91.89, 95.95] | 89.72 [86.62, 91.61] | 97.75 [97.30, 98.65] | 79.73 [74.32, 85.14] | 90.93 [89.64, 93.19] | 50.58 [33.76, 60.14] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--separate_ends | 86.52 [81.93, 88.89] | 95.05 [91.89, 97.30] | 90.58 [86.62, 92.90] | 85.14 [83.78, 86.49] | 78.38 [75.68, 79.73] | 91.77 [90.64, 93.46] | 44.34 [31.77, 52.75] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--typed_starts | 82.41 [79.52, 85.00] | 90.54 [89.19, 91.89] | 86.28 [84.08, 88.31] | 85.14 [83.78, 86.49] | 54.05 | 97.04 [96.84, 97.15] | 5.23 [4.37, 6.70] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| first_start--full_parent--evidence--budget-10 | 91.49 [89.74, 92.41] | 96.85 [94.59, 98.65] | 94.09 [92.11, 95.42] | 86.04 [85.14, 86.49] | 69.82 [68.92, 71.62] | 97.74 | 0.00 | 0.00 | 0.00 | 2.94 [2.90, 2.98] | 18.00 | 17.33 [17.00, 18.00] |
| first_start--proposal_confirmation--evidence--budget-10 | 86.92 [86.08, 87.34] | 92.79 [91.89, 93.24] | 89.76 [88.89, 90.20] | 86.04 [85.14, 86.49] | 54.05 | 97.74 | 0.00 | 0.00 | 0.00 | 2.94 [2.90, 2.98] | 18.00 | 17.33 [17.00, 18.00] |
| head_refined--full_parent--evidence--budget-10 | 91.95 [91.14, 92.41] | 97.75 [97.30, 98.65] | 94.76 [94.12, 95.42] | 75.68 | 66.22 | 97.74 | 0.00 | 0.00 | 0.00 | 2.92 [2.88, 2.95] | 20.00 | 19.67 [19.00, 20.00] |
| head_refined--proposal_confirmation--evidence--budget-10 | 91.56 [91.14, 92.41] | 97.75 [97.30, 98.65] | 94.55 [94.12, 95.42] | 75.68 | 67.12 [66.22, 67.57] | 97.74 | 0.00 | 0.00 | 0.00 | 2.92 [2.88, 2.95] | 20.00 | 19.67 [19.00, 20.00] |
| separate_ends--full_parent--evidence--budget-10 | 91.95 [91.14, 92.41] | 97.75 [97.30, 98.65] | 94.76 [94.12, 95.42] | 75.68 | 66.22 | 97.74 | 0.00 | 0.00 | 0.00 | 2.92 [2.88, 2.95] | 20.00 | 19.67 [19.00, 20.00] |
| separate_ends--proposal_confirmation--evidence--budget-10 | 90.30 [89.87, 91.14] | 96.40 [95.95, 97.30] | 93.25 [92.81, 94.12] | 82.43 [81.08, 83.78] | 64.86 [63.51, 66.22] | 97.74 | 0.00 | 0.00 | 0.00 | 2.92 [2.88, 2.95] | 20.00 | 19.67 [19.00, 20.00] |
| typed_starts--full_parent--evidence--budget-10 | 91.49 [89.74, 92.41] | 96.85 [94.59, 98.65] | 94.09 [92.11, 95.42] | 86.04 [85.14, 86.49] | 69.82 [68.92, 71.62] | 97.74 | 0.00 | 0.00 | 0.00 | 2.94 [2.90, 2.98] | 18.00 | 17.33 [17.00, 18.00] |
| typed_starts--proposal_confirmation--evidence--budget-10 | 86.92 [86.08, 87.34] | 92.79 [91.89, 93.24] | 89.76 [88.89, 90.20] | 86.04 [85.14, 86.49] | 54.05 | 97.74 | 0.00 | 0.00 | 0.00 | 2.94 [2.90, 2.98] | 18.00 | 17.33 [17.00, 18.00] |

### source-group-012

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed end R @1s % | Raw core R % | Raw core lost s | Complete misses | Additional misses | Review min | Parent jobs | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 59.38 | 67.86 | 63.33 | 76.79 | 35.71 | 93.89 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--first_start | 68.23 [65.62, 71.88] | 77.98 [75.00, 82.14] | 72.78 [70.00, 76.67] | 79.17 [76.79, 82.14] | 35.71 | 92.33 [91.97, 92.84] | 6.86 [4.64, 8.42] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_refined | 72.75 [70.15, 74.24] | 85.71 [83.93, 87.50] | 78.70 [76.42, 80.33] | 96.43 [94.64, 98.21] | 70.83 [67.86, 73.21] | 86.86 [85.97, 87.54] | 30.87 [27.87, 34.75] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--separate_ends | 71.74 [68.66, 73.85] | 84.52 [82.14, 85.71] | 77.61 [74.80, 79.34] | 79.76 [76.79, 82.14] | 63.69 [60.71, 66.07] | 89.17 [88.67, 89.93] | 20.72 [17.38, 22.92] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--typed_starts | 65.68 [61.19, 69.70] | 77.38 [73.21, 82.14] | 71.05 [66.67, 75.41] | 79.76 [76.79, 82.14] | 35.71 | 92.33 [91.97, 92.84] | 6.86 [4.64, 8.42] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| first_start--full_parent--evidence--budget-10 | 71.35 [67.19, 73.44] | 81.55 [76.79, 83.93] | 76.11 [71.67, 78.33] | 81.55 [80.36, 82.14] | 52.98 [50.00, 55.36] | 93.89 | 0.00 | 0.00 | 0.00 | 2.16 [2.12, 2.19] | 13.67 [13.00, 14.00] | 13.67 [13.00, 14.00] |
| first_start--proposal_confirmation--evidence--budget-10 | 67.19 | 76.79 | 71.67 | 81.55 [80.36, 82.14] | 35.71 | 93.89 | 0.00 | 0.00 | 0.00 | 2.16 [2.12, 2.19] | 13.67 [13.00, 14.00] | 13.67 [13.00, 14.00] |
| head_refined--full_parent--evidence--budget-10 | 72.92 [68.75, 75.00] | 83.33 [78.57, 85.71] | 77.78 [73.33, 80.00] | 79.76 [78.57, 80.36] | 50.60 [50.00, 51.79] | 93.89 | 0.00 | 0.00 | 0.00 | 2.16 [2.13, 2.21] | 14.00 | 14.00 |
| head_refined--proposal_confirmation--evidence--budget-10 | 72.92 [68.75, 75.00] | 83.33 [78.57, 85.71] | 77.78 [73.33, 80.00] | 79.76 [78.57, 80.36] | 48.81 [48.21, 50.00] | 93.89 | 0.00 | 0.00 | 0.00 | 2.16 [2.13, 2.21] | 14.00 | 14.00 |
| separate_ends--full_parent--evidence--budget-10 | 73.31 [68.75, 76.19] | 83.33 [78.57, 85.71] | 78.00 [73.33, 80.67] | 79.76 [78.57, 80.36] | 52.38 [50.00, 53.57] | 93.89 | 0.00 | 0.00 | 0.00 | 2.15 [2.12, 2.16] | 13.67 [13.00, 14.00] | 13.33 [13.00, 14.00] |
| separate_ends--proposal_confirmation--evidence--budget-10 | 71.88 [68.75, 73.44] | 82.14 [78.57, 83.93] | 76.67 [73.33, 78.33] | 80.95 [78.57, 82.14] | 49.40 [48.21, 50.00] | 93.89 | 0.00 | 0.00 | 0.00 | 2.15 [2.12, 2.16] | 13.67 [13.00, 14.00] | 13.33 [13.00, 14.00] |
| typed_starts--full_parent--evidence--budget-10 | 71.35 [67.19, 73.44] | 81.55 [76.79, 83.93] | 76.11 [71.67, 78.33] | 81.55 [80.36, 82.14] | 52.98 [50.00, 55.36] | 93.89 | 0.00 | 0.00 | 0.00 | 2.16 [2.12, 2.19] | 13.67 [13.00, 14.00] | 13.67 [13.00, 14.00] |
| typed_starts--proposal_confirmation--evidence--budget-10 | 67.19 | 76.79 | 71.67 | 81.55 [80.36, 82.14] | 35.71 | 93.89 | 0.00 | 0.00 | 0.00 | 2.16 [2.12, 2.19] | 13.67 [13.00, 14.00] | 13.67 [13.00, 14.00] |

## Limits and reproducibility

Observed start/end localization is a rally-boundary proxy, not serve-side, point-winner or reconstructed-score accuracy. Both human modes assume perfect permitted actions. Results are development estimates; previous model and decoder choices already used this footage. Fresh footage and measured human review are still needed before product promotion.

The registered result files bind all source/input hashes and independent plan, queue, human, workload, duration, identity and typed-metric audits. The summary verifies the full 12 automatic/192 reviewed outcome matrix, exact four-padding export invariance, workload totals, and zero full-parent core loss. It intentionally permits and exposes automatic/proposal-only event-timeline losses. `summary.json` contains 69 aggregate arms over pooled, source-group and recording scopes, with numeric mean/min/max and available-seed counts.

Artifacts: `private-reference-0192`. Contract SHA-256: `93697ad54ff3392e30cedb8282b77841736e23ab4afb7db933b7f99d8b6b4d5c`. Immutable summary SHA-256: `b2764e0bf0489f07439c74f59d8c2da472d190fa48e2c2e51c537a65c022d250`. Numerical freeze contains 44 source files and 88 qualification tests; a further 12 post-run summarizer tests passed. The byte snapshot includes frozen inputs, numerical sources, every result and supplementary reporting; byte verification does not imply a relocated replay.
