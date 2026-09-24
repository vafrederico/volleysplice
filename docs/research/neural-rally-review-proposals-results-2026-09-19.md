# Rally-preserving review proposal experiment

<!-- ROOT EXECUTIVE FINDINGS START -->
## Findings and product implications

**The experiment is complete: 432 fixed budgeted outcomes, with separate export, rally-event and observed-start metrics.** Original rally identities and touching boundaries are preserved. The new review inventories explicitly flag one-to-many/many-to-one predictions, propose event-shaped missed play from existing live scores, and optionally use the existing start/end heads. Eighty pre-execution tests, including the end-to-end adapter test, passed; every real candidate, queue, local edit, event metric and four-padding duration result passed independent checks. No detector or learned review head was trained and production was not modified.

The user's score-tracking requirement changes how the results must be read. Production's automatic **83.76% export F1** corresponds to only **66.47% rally-event F1** and **71.74% observed-start recall within one second**. DINO global's automatic **93.07% export F1** corresponds to **77.16% event F1** and **82.61% observed-start recall**. Padded export coverage is therefore insufficient evidence of correct rally/serve separation. Rally-event F1 uses one-to-one raw-event matching at IoU >=0.5; the observed-start metric excludes synthetic review-window-edge starts. Neither is serving-side or score accuracy.

**Local event proposals are a useful improvement for the individual neural detector.** With DINO global and the fixed evidence order, the old inventory at the 10% cap gives **90.36% P / 97.60% R / 93.84% export F1**, **80.95% event F1**, and **84.37% observed-start recall**, requiring **13.56 minutes**, **66.67 edit regions** and **61.33 true rallies** in those regions. The new local event inventory gives **91.80% P / 97.83% R / 94.71% export F1**, **85.37% event F1**, and **85.09% observed-start recall**, requiring **12.31 minutes**, **62 edit regions** and **48.33 true rallies**. That is +0.87 export-F1 points, +4.42 event-F1 points and 1.25 fewer playback minutes, while material split errors fall from 7.33 to 4.33. Completely missed rallies fall from 15.67 to 13.67; this remains materially worse than production's three automatic complete misses.

The benefit transfers to compact and DINO short boost at the same 10% cap and evidence order. Compact's event F1 rises **74.40% -> 75.89%** and export F1 **91.70% -> 92.09%**, with playback **13.64 -> 13.05 minutes**. DINO short boost's event F1 rises **76.95% -> 82.38%** and export F1 **93.06% -> 94.11%**, with playback **13.57 -> 13.04 minutes**. Across all three models and four caps, both new individual inventories improve mean event F1 over the legacy inventory in all **24 paired comparisons** with evidence ordering. These are descriptive development comparisons, not 24 independent datasets or a statistical generalization claim.

**The production combinations have a real tradeoff.** With evidence ordering, local event proposals improve event F1 over the old inventory in all **12 model/cap comparisons**, but observed-start recall is lower in **11** and tied in **one**. Adding head-guided proposals improves observed-start recall over the legacy inventory in all **12** comparisons, but often reduces export/event F1. At the 40% cap, production plus DINO global local events gives **88.17% P / 99.87% R / 93.66% export F1**, **89.33% event F1**, and **85.20% observed-start recall**, with **55.05 minutes**, **271.33 edit regions** and **205 true rallies** reviewed. Material merged predictions fall from production's 7 to 1, material split true rallies from 4 to 0, and complete misses from 3 to 1. However, the legacy DINO-global queue at that cap uses only 44.66 minutes and achieves higher observed-start recall (87.47%), so these are equal-cap, not equal-realized-time comparisons.

For a phone/browser-oriented production combination at that 40% cap, compact local events gives **93.07% export F1 / 87.41% event F1 / 85.40% observed-start recall** in **54.98 minutes**. Compact with head-guided proposals instead gives **89.39% / 81.15% / 92.44%**, respectively, in **55.05 minutes**. At the more practical 10% cap, production plus compact heads gives **74.33% P / 99.52% R / 85.10% export F1**, **71.41% event F1** and **78.05% observed-start recall** in **13.70 minutes**, involving **63 edit regions** and **58.33 true rallies**. The heads are useful timing signals, but the present ordering spends review effort on boundaries instead of enough whole-event cleanup. There is no single uniformly better review policy.

For standalone DINO global at the 40% cap, the head-guided inventory with evidence order gives **94.81% P / 98.77% R / 96.74% export F1**, **91.85% event F1** and **93.33% observed-start precision / 92.65% recall / 92.98% F1**, using **39.49 minutes**, **236 edit regions** and **183.33 true rallies**. It still leaves 8.33 completely missed rallies, 0.33 material merges and 5 material splits. Across both queue orders, the primary-metric leader at this cap is the same model/inventory with chronological order: **96.80% export F1**, **91.52% event F1**, **92.24% observed-start recall**, **39.46 minutes**. The evidence heuristic is therefore not established as a better overall sorter. For production head-guided proposals, it loses to chronological order on export F1 in all 12 model/cap pairs; it improves production local-event event F1 in all 12 corresponding pairs. Full rankings retain both orders.

**Recommendation:** keep a separate rally-event timeline with start/end markers and IDs, and derive padded/joined export ranges from it. A joined export must never imply one score-tracking rally. Retain the new explicit split/merge and event-shaped proposal generation for further development; it improves useful event-level quality. Do not promote the current generic evidence ranker or merge all review purposes into one queue. The next bounded follow-up should prioritize three explicit correction types—missing/false rally, split/merge, and serve-start timing—under a fixed budget, with full-event context when needed to avoid censored partial events. A learned review scorer could be compared after those targets and validation splits are fixed. A score-tracking pipeline still needs serving-side/result logic and its own end-to-end evaluation.

All reported review outcomes assume perfect localized human occupancy and boundary correction; they are not binary keep/remove scores or observed human performance. A “proposal” is a machine flag, overlapping flags can share an edit region, and one playback clip can contain several edit regions/rallies. Counts are three-seed means; actual playback is footage at 1x, not reviewer wall-clock labor. Local inventories can exhaust before the nominal cap, especially standalone DINO local events (22.24 minutes at the 40% cap). All 144 aggregate arms pass the predeclared non-worsening screen against their own automatic baseline, but this does not establish score-tracking readiness or erase the tradeoffs between arms.

The production comparison plot (ledger `private-reference-0168`) and individual-model plot (ledger `private-reference-0169`) show export F1, event F1 and observed-start F1 against actual playback. Both use the fixed evidence order; chronological comparisons remain in the tables.

The initial run stopped before producing result files because one independent workload-check adapter received a mapping instead of an interval pair. Its registration and source bytes remain preserved. The v2 run fixes that adapter and adds end-to-end synthetic coverage; all candidate rules, thresholds, budgets and metric definitions are unchanged. Shipped production exposure and prior development selection remain limitations; the protected test was not opened.
<!-- ROOT EXECUTIVE FINDINGS END -->

This is a fixed development comparison on eight exact-label grass/indoor videos, four source groups and 322 rallies. Every listed value averages three seed results after pooling recording-level numerators and denominators within each seed. Seed replicas do not increase the video count. The neural detectors were held fixed; no learned review head was trained.

Export quality uses `F1_padP_coreR` at the declared two-second symmetric padding and strict positive gap <3-second joining. The other three padding cases are sensitivity results. Raw event identities remain separate even when exports join. Event P/R/F1 uses general one-to-one IoU>=0.5 matching. Observed start localization at one second is a serve-contact proxy; synthetic edit-window edges are excluded. This does not evaluate serving side, point winner or reconstructed scores.

The human perfectly corrects occupancy and start/end markers only inside selected edit windows, with no oracle endpoints imported from outside those windows. Playback includes extra context and is charged as union footage at 1x, not measured reviewer labor. A proposal is a decision range and is not necessarily a rally; true-rally counts deduplicate labeled rally identities intersecting edit regions.

Inventory `legacy` uses prior whole-component/five-second flags; `local_events` uses localized disagreement, split/merge and sustained-live proposals; `local_heads` additionally uses trained start/end heads. `chronological` orders by time and `evidence` uses fixed label-blind evidence per standalone playback cost. Review caps are per recording and unused budget is reported. The descriptive screen requires non-worsening mean event F1, observed-start F1 and recall, and no increase in complete rally misses against the same automatic baseline. Passing is not score-tracking validation.

## Automatic baselines

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_boost | 88.30 | 92.41 | 90.31 | 69.23 | 69.67 | 69.45 | 74.38 | 74.84 | 74.61 | 0.00 | 0.00 | 0.00 | baseline |
| dino_boost | 88.47 | 96.95 | 92.51 | 71.71 | 76.81 | 74.17 | 78.69 | 84.27 | 81.37 | 0.00 | 0.00 | 0.00 | baseline |
| dino_global | 89.50 | 96.96 | 93.07 | 75.54 | 78.88 | 77.16 | 79.07 | 82.61 | 80.78 | 0.00 | 0.00 | 0.00 | baseline |
| production | 72.45 | 99.27 | 83.76 | 62.64 | 70.81 | 66.47 | 63.64 | 71.74 | 67.45 | 0.00 | 0.00 | 0.00 | baseline |

## Every fixed arm at the target padding

### production--budget-05

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--dino_global--legacy--chronological--budget-05 | 74.40 | 99.47 | 85.12 | 66.07 | 72.98 | 69.36 | 66.70 | 73.60 | 69.98 | 6.22 | 26.00 | 16.00 | yes |
| production--dino_boost--legacy--evidence--budget-05 | 74.33 | 99.40 | 85.06 | 65.46 | 72.77 | 68.92 | 66.73 | 74.12 | 70.23 | 6.33 | 18.33 | 13.33 | yes |
| production--compact_boost--legacy--chronological--budget-05 | 74.33 | 99.36 | 85.04 | 65.73 | 72.67 | 69.03 | 66.20 | 73.19 | 69.52 | 6.31 | 25.00 | 17.00 | yes |
| production--dino_boost--legacy--chronological--budget-05 | 74.23 | 99.43 | 85.00 | 65.79 | 72.88 | 69.15 | 66.23 | 73.29 | 69.58 | 6.20 | 25.00 | 15.33 | yes |
| production--dino_global--legacy--evidence--budget-05 | 74.18 | 99.36 | 84.94 | 65.06 | 72.67 | 68.66 | 66.08 | 73.81 | 69.73 | 6.51 | 18.67 | 14.67 | yes |
| production--dino_global--local_events--chronological--budget-05 | 74.09 | 99.27 | 84.85 | 64.93 | 72.05 | 68.30 | 66.39 | 73.60 | 69.81 | 6.44 | 47.00 | 19.00 | yes |
| production--dino_global--local_heads--chronological--budget-05 | 74.08 | 99.27 | 84.85 | 64.90 | 72.15 | 68.33 | 66.42 | 73.71 | 69.87 | 6.48 | 83.33 | 18.67 | yes |
| production--compact_boost--local_events--chronological--budget-05 | 74.07 | 99.27 | 84.84 | 64.93 | 72.05 | 68.30 | 66.29 | 73.50 | 69.71 | 6.46 | 45.00 | 18.67 | yes |
| production--compact_boost--local_heads--chronological--budget-05 | 74.06 | 99.27 | 84.83 | 64.99 | 72.26 | 68.43 | 66.32 | 73.60 | 69.77 | 6.51 | 76.33 | 19.00 | yes |
| production--dino_boost--local_heads--chronological--budget-05 | 74.05 | 99.27 | 84.83 | 64.90 | 72.15 | 68.33 | 66.26 | 73.60 | 69.74 | 6.51 | 96.33 | 18.67 | yes |
| production--dino_boost--local_events--chronological--budget-05 | 74.02 | 99.27 | 84.80 | 64.80 | 72.05 | 68.24 | 66.20 | 73.60 | 69.71 | 6.42 | 49.00 | 19.00 | yes |
| production--dino_global--local_events--evidence--budget-05 | 73.62 | 99.60 | 84.66 | 65.48 | 74.02 | 69.49 | 65.53 | 73.81 | 69.43 | 6.65 | 51.33 | 25.33 | yes |
| production--compact_boost--legacy--evidence--budget-05 | 73.76 | 99.31 | 84.65 | 64.23 | 71.74 | 67.78 | 66.27 | 74.02 | 69.93 | 6.57 | 18.00 | 15.67 | yes |
| production--dino_boost--local_events--evidence--budget-05 | 73.56 | 99.53 | 84.59 | 65.35 | 73.40 | 69.14 | 65.80 | 73.71 | 69.53 | 6.57 | 54.67 | 22.67 | yes |
| production--dino_boost--local_heads--evidence--budget-05 | 73.54 | 99.53 | 84.58 | 64.81 | 73.40 | 68.83 | 66.06 | 74.53 | 70.04 | 6.77 | 83.00 | 25.67 | yes |
| production--dino_global--local_heads--evidence--budget-05 | 73.50 | 99.54 | 84.56 | 64.20 | 73.50 | 68.53 | 66.03 | 74.84 | 70.16 | 6.76 | 80.00 | 27.67 | yes |
| production--compact_boost--local_heads--evidence--budget-05 | 73.48 | 99.50 | 84.54 | 65.26 | 73.71 | 69.23 | 66.79 | 74.95 | 70.63 | 6.73 | 84.67 | 25.33 | yes |
| production--compact_boost--local_events--evidence--budget-05 | 73.31 | 99.51 | 84.42 | 65.10 | 72.98 | 68.81 | 65.93 | 73.71 | 69.60 | 6.62 | 55.67 | 24.67 | yes |

### individual--budget-05

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| individual--dino_global--local_heads--evidence--budget-05 | 90.77 | 97.37 | 93.94 | 80.49 | 81.68 | 81.06 | 82.95 | 83.85 | 83.38 | 6.53 | 77.33 | 26.33 | yes |
| individual--dino_global--local_events--evidence--budget-05 | 90.65 | 97.41 | 93.90 | 79.83 | 81.68 | 80.73 | 82.10 | 83.85 | 82.95 | 6.51 | 43.33 | 26.67 | yes |
| individual--dino_global--local_events--chronological--budget-05 | 90.45 | 97.60 | 93.87 | 79.98 | 82.30 | 81.11 | 82.20 | 84.37 | 83.26 | 6.35 | 48.33 | 27.33 | yes |
| individual--dino_global--local_heads--chronological--budget-05 | 90.32 | 97.38 | 93.71 | 78.19 | 81.26 | 79.68 | 80.91 | 83.95 | 82.39 | 6.25 | 56.00 | 29.00 | yes |
| individual--dino_global--legacy--evidence--budget-05 | 89.90 | 97.31 | 93.44 | 77.00 | 80.95 | 78.91 | 80.07 | 83.23 | 81.60 | 6.66 | 41.00 | 29.67 | yes |
| individual--dino_global--legacy--chronological--budget-05 | 89.88 | 97.27 | 93.42 | 77.76 | 80.64 | 79.16 | 80.69 | 83.64 | 82.13 | 6.37 | 45.33 | 25.00 | yes |
| individual--dino_boost--local_events--evidence--budget-05 | 89.77 | 97.25 | 93.35 | 76.64 | 79.71 | 78.14 | 82.10 | 85.30 | 83.66 | 6.59 | 41.00 | 25.33 | yes |
| individual--dino_boost--local_heads--evidence--budget-05 | 89.68 | 97.23 | 93.30 | 76.40 | 79.09 | 77.72 | 82.37 | 84.99 | 83.66 | 6.54 | 72.67 | 26.00 | yes |
| individual--dino_boost--local_events--chronological--budget-05 | 89.48 | 97.33 | 93.24 | 76.34 | 80.12 | 78.18 | 81.86 | 85.40 | 83.59 | 6.39 | 47.00 | 25.00 | yes |
| individual--dino_boost--local_heads--chronological--budget-05 | 89.16 | 97.06 | 92.94 | 73.87 | 78.67 | 76.19 | 80.31 | 85.09 | 82.63 | 6.27 | 54.67 | 29.00 | yes |
| individual--dino_boost--legacy--chronological--budget-05 | 88.93 | 97.01 | 92.79 | 73.37 | 78.05 | 75.63 | 80.21 | 84.99 | 82.52 | 6.35 | 44.67 | 20.67 | yes |
| individual--dino_boost--legacy--evidence--budget-05 | 88.91 | 96.99 | 92.77 | 72.63 | 77.95 | 75.19 | 80.63 | 85.09 | 82.79 | 6.69 | 38.33 | 27.33 | yes |
| individual--compact_boost--local_events--chronological--budget-05 | 89.02 | 93.50 | 91.20 | 73.45 | 73.60 | 73.53 | 77.18 | 77.02 | 77.10 | 6.38 | 46.00 | 26.00 | yes |
| individual--compact_boost--local_events--evidence--budget-05 | 89.43 | 93.02 | 91.19 | 72.35 | 71.53 | 71.94 | 77.55 | 76.50 | 77.02 | 6.55 | 29.33 | 23.67 | yes |
| individual--compact_boost--local_heads--evidence--budget-05 | 89.50 | 92.91 | 91.17 | 73.11 | 72.05 | 72.57 | 78.19 | 76.81 | 77.49 | 6.56 | 59.33 | 26.33 | yes |
| individual--compact_boost--local_heads--chronological--budget-05 | 88.96 | 93.43 | 91.13 | 72.76 | 72.98 | 72.87 | 76.94 | 77.02 | 76.98 | 6.53 | 59.00 | 30.67 | yes |
| individual--compact_boost--legacy--evidence--budget-05 | 88.87 | 93.42 | 91.08 | 70.48 | 71.43 | 70.95 | 75.70 | 75.78 | 75.74 | 6.71 | 41.00 | 26.33 | yes |
| individual--compact_boost--legacy--chronological--budget-05 | 88.95 | 93.27 | 91.05 | 71.44 | 71.74 | 71.59 | 76.26 | 76.50 | 76.38 | 6.50 | 46.33 | 21.33 | yes |

### production--budget-10

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--dino_boost--legacy--evidence--budget-10 | 76.16 | 99.45 | 86.26 | 67.73 | 74.53 | 70.97 | 69.09 | 75.88 | 72.32 | 13.26 | 37.33 | 29.33 | yes |
| production--dino_global--legacy--chronological--budget-10 | 75.86 | 99.50 | 86.09 | 68.77 | 75.67 | 72.05 | 68.93 | 75.78 | 72.19 | 13.16 | 48.33 | 36.00 | yes |
| production--dino_boost--legacy--chronological--budget-10 | 75.76 | 99.49 | 86.01 | 68.54 | 75.57 | 71.89 | 68.86 | 75.78 | 72.15 | 13.14 | 48.00 | 35.00 | yes |
| production--dino_global--legacy--evidence--budget-10 | 75.79 | 99.42 | 86.01 | 67.32 | 74.64 | 70.79 | 68.60 | 75.98 | 72.10 | 13.37 | 37.67 | 32.00 | yes |
| production--compact_boost--legacy--chronological--budget-10 | 75.71 | 99.40 | 85.95 | 68.04 | 75.16 | 71.42 | 68.74 | 75.78 | 72.08 | 13.31 | 47.33 | 39.33 | yes |
| production--compact_boost--local_events--chronological--budget-10 | 75.32 | 99.27 | 85.66 | 67.01 | 74.64 | 70.62 | 68.78 | 76.40 | 72.39 | 13.20 | 92.00 | 42.33 | yes |
| production--compact_boost--legacy--evidence--budget-10 | 75.23 | 99.40 | 85.64 | 65.99 | 73.50 | 69.54 | 67.94 | 75.67 | 71.60 | 13.19 | 37.00 | 33.00 | yes |
| production--dino_global--local_events--chronological--budget-10 | 75.28 | 99.27 | 85.63 | 66.79 | 74.33 | 70.36 | 68.78 | 76.40 | 72.39 | 13.11 | 95.00 | 42.33 | yes |
| production--dino_boost--local_events--evidence--budget-10 | 74.98 | 99.73 | 85.60 | 68.06 | 76.09 | 71.85 | 67.63 | 75.26 | 71.24 | 13.64 | 111.67 | 49.00 | yes |
| production--dino_global--local_heads--chronological--budget-10 | 75.23 | 99.27 | 85.60 | 66.82 | 74.43 | 70.42 | 68.69 | 76.29 | 72.29 | 13.14 | 172.00 | 42.00 | yes |
| production--compact_boost--local_heads--chronological--budget-10 | 75.19 | 99.27 | 85.57 | 66.88 | 74.64 | 70.55 | 68.78 | 76.40 | 72.39 | 13.24 | 157.00 | 42.33 | yes |
| production--dino_boost--local_events--chronological--budget-10 | 75.13 | 99.32 | 85.55 | 66.48 | 74.12 | 70.09 | 68.99 | 76.71 | 72.65 | 13.27 | 98.67 | 43.33 | yes |
| production--dino_boost--local_heads--chronological--budget-10 | 75.12 | 99.32 | 85.54 | 66.48 | 74.33 | 70.19 | 68.62 | 76.29 | 72.25 | 13.23 | 200.00 | 42.00 | yes |
| production--dino_boost--local_heads--evidence--budget-10 | 74.91 | 99.65 | 85.53 | 67.98 | 76.92 | 72.17 | 68.44 | 77.02 | 72.48 | 13.69 | 185.33 | 53.67 | yes |
| production--dino_global--local_events--evidence--budget-10 | 74.81 | 99.73 | 85.49 | 68.66 | 77.12 | 72.64 | 67.72 | 75.57 | 71.43 | 13.62 | 110.33 | 53.33 | yes |
| production--dino_global--local_heads--evidence--budget-10 | 74.72 | 99.70 | 85.42 | 67.60 | 77.33 | 72.14 | 68.54 | 77.33 | 72.67 | 13.64 | 178.00 | 56.67 | yes |
| production--compact_boost--local_events--evidence--budget-10 | 74.74 | 99.56 | 85.38 | 68.56 | 76.09 | 72.13 | 67.92 | 74.95 | 71.26 | 13.59 | 115.67 | 53.00 | yes |
| production--compact_boost--local_heads--evidence--budget-10 | 74.33 | 99.52 | 85.10 | 67.12 | 76.29 | 71.41 | 69.62 | 78.05 | 73.60 | 13.70 | 175.00 | 58.33 | yes |

### individual--budget-10

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| individual--dino_global--local_heads--evidence--budget-10 | 91.86 | 97.76 | 94.71 | 84.51 | 85.09 | 84.79 | 85.64 | 85.61 | 85.61 | 13.46 | 149.33 | 61.00 | yes |
| individual--dino_global--local_events--evidence--budget-10 | 91.80 | 97.83 | 94.71 | 85.35 | 85.40 | 85.37 | 85.74 | 85.09 | 85.41 | 12.31 | 89.33 | 48.33 | yes |
| individual--dino_global--local_events--chronological--budget-10 | 91.44 | 97.96 | 94.58 | 84.32 | 85.40 | 84.85 | 85.38 | 85.71 | 85.54 | 12.25 | 96.67 | 48.67 | yes |
| individual--dino_global--local_heads--chronological--budget-10 | 91.40 | 97.65 | 94.41 | 80.75 | 83.23 | 81.95 | 83.78 | 85.61 | 84.66 | 13.39 | 124.33 | 61.67 | yes |
| individual--dino_boost--local_events--evidence--budget-10 | 90.74 | 97.75 | 94.11 | 81.55 | 83.23 | 82.38 | 85.94 | 87.16 | 86.54 | 13.04 | 89.33 | 53.33 | yes |
| individual--dino_boost--local_heads--evidence--budget-10 | 90.74 | 97.43 | 93.96 | 80.26 | 81.68 | 80.96 | 85.49 | 86.54 | 86.01 | 13.47 | 147.00 | 58.67 | yes |
| individual--dino_boost--local_events--chronological--budget-10 | 90.35 | 97.76 | 93.90 | 80.14 | 83.13 | 81.61 | 84.46 | 86.54 | 85.49 | 12.83 | 94.67 | 51.67 | yes |
| individual--dino_global--legacy--evidence--budget-10 | 90.36 | 97.60 | 93.84 | 78.91 | 83.13 | 80.95 | 81.65 | 84.37 | 82.97 | 13.56 | 83.33 | 61.33 | yes |
| individual--dino_global--legacy--chronological--budget-10 | 90.35 | 97.44 | 93.75 | 78.44 | 81.47 | 79.92 | 81.62 | 84.37 | 82.96 | 13.35 | 95.33 | 52.33 | yes |
| individual--dino_boost--local_heads--chronological--budget-10 | 90.03 | 97.30 | 93.52 | 76.48 | 80.43 | 78.40 | 82.88 | 86.02 | 84.42 | 13.20 | 118.33 | 59.33 | yes |
| individual--dino_boost--legacy--chronological--budget-10 | 89.39 | 97.08 | 93.07 | 74.68 | 79.30 | 76.91 | 81.09 | 85.51 | 83.23 | 13.28 | 94.33 | 46.00 | yes |
| individual--dino_boost--legacy--evidence--budget-10 | 89.29 | 97.17 | 93.06 | 74.30 | 79.81 | 76.95 | 82.14 | 86.34 | 84.17 | 13.57 | 80.00 | 58.33 | yes |
| individual--compact_boost--local_heads--evidence--budget-10 | 90.59 | 93.71 | 92.12 | 77.41 | 75.16 | 76.27 | 81.52 | 78.47 | 79.96 | 13.29 | 123.33 | 54.67 | yes |
| individual--compact_boost--local_events--chronological--budget-10 | 90.02 | 94.28 | 92.10 | 76.21 | 76.29 | 76.25 | 79.71 | 79.30 | 79.50 | 12.81 | 88.67 | 51.67 | yes |
| individual--compact_boost--local_events--evidence--budget-10 | 90.76 | 93.47 | 92.09 | 77.40 | 74.43 | 75.89 | 81.35 | 77.64 | 79.45 | 13.05 | 70.00 | 49.00 | yes |
| individual--compact_boost--local_heads--chronological--budget-10 | 89.81 | 94.08 | 91.89 | 74.96 | 75.57 | 75.26 | 79.43 | 79.50 | 79.46 | 13.00 | 117.00 | 59.67 | yes |
| individual--compact_boost--legacy--evidence--budget-10 | 89.37 | 94.16 | 91.70 | 73.75 | 75.05 | 74.40 | 78.92 | 78.67 | 78.80 | 13.64 | 87.67 | 54.33 | yes |
| individual--compact_boost--legacy--chronological--budget-10 | 89.38 | 93.74 | 91.51 | 73.36 | 74.12 | 73.74 | 77.65 | 78.05 | 77.85 | 13.24 | 96.00 | 46.33 | yes |

### production--budget-20

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--dino_boost--legacy--evidence--budget-20 | 80.48 | 99.58 | 89.02 | 73.37 | 79.30 | 76.22 | 74.71 | 80.43 | 77.47 | 26.71 | 81.33 | 65.00 | yes |
| production--dino_global--legacy--chronological--budget-20 | 80.04 | 99.60 | 88.76 | 75.72 | 81.99 | 78.73 | 75.12 | 81.26 | 78.07 | 26.08 | 91.00 | 72.33 | yes |
| production--dino_boost--legacy--chronological--budget-20 | 79.93 | 99.64 | 88.70 | 75.09 | 81.16 | 78.01 | 75.91 | 81.88 | 78.78 | 26.73 | 94.67 | 71.67 | yes |
| production--dino_global--legacy--evidence--budget-20 | 79.65 | 99.50 | 88.48 | 72.32 | 79.50 | 75.74 | 74.03 | 81.16 | 77.43 | 26.25 | 77.00 | 69.00 | yes |
| production--compact_boost--legacy--chronological--budget-20 | 79.36 | 99.51 | 88.30 | 73.81 | 80.23 | 76.88 | 73.66 | 79.92 | 76.66 | 25.90 | 90.00 | 75.33 | yes |
| production--compact_boost--legacy--evidence--budget-20 | 79.04 | 99.48 | 88.09 | 70.67 | 78.05 | 74.18 | 72.45 | 80.02 | 76.05 | 26.03 | 76.33 | 69.00 | yes |
| production--compact_boost--local_events--evidence--budget-20 | 78.65 | 99.64 | 87.91 | 73.22 | 81.78 | 77.26 | 73.25 | 79.92 | 76.44 | 27.45 | 205.00 | 106.67 | yes |
| production--dino_global--local_events--evidence--budget-20 | 78.42 | 99.81 | 87.83 | 73.58 | 81.57 | 77.37 | 70.64 | 77.43 | 73.88 | 27.49 | 207.00 | 109.33 | yes |
| production--dino_boost--local_events--evidence--budget-20 | 78.12 | 99.81 | 87.64 | 72.58 | 80.85 | 76.49 | 69.82 | 77.12 | 73.29 | 27.48 | 215.67 | 105.67 | yes |
| production--dino_global--local_heads--chronological--budget-20 | 77.94 | 99.47 | 87.40 | 70.74 | 78.57 | 74.45 | 72.66 | 80.33 | 76.30 | 26.98 | 356.33 | 86.67 | yes |
| production--dino_global--local_events--chronological--budget-20 | 77.92 | 99.47 | 87.39 | 70.80 | 78.57 | 74.48 | 72.70 | 80.23 | 76.28 | 27.05 | 201.00 | 88.67 | yes |
| production--dino_boost--local_events--chronological--budget-20 | 77.78 | 99.47 | 87.30 | 70.41 | 77.85 | 73.94 | 72.56 | 79.92 | 76.06 | 26.97 | 207.33 | 86.33 | yes |
| production--dino_boost--local_heads--chronological--budget-20 | 77.77 | 99.47 | 87.30 | 70.15 | 77.85 | 73.80 | 72.58 | 80.02 | 76.12 | 27.00 | 409.67 | 84.67 | yes |
| production--compact_boost--local_heads--chronological--budget-20 | 77.71 | 99.45 | 87.25 | 70.15 | 77.85 | 73.80 | 72.49 | 79.92 | 76.02 | 27.10 | 327.00 | 86.67 | yes |
| production--compact_boost--local_events--chronological--budget-20 | 77.71 | 99.45 | 87.25 | 70.27 | 78.05 | 73.96 | 72.50 | 79.71 | 75.94 | 27.05 | 193.67 | 87.33 | yes |
| production--dino_boost--local_heads--evidence--budget-20 | 77.12 | 99.76 | 86.99 | 70.99 | 81.06 | 75.69 | 71.85 | 80.85 | 76.08 | 27.52 | 367.67 | 115.67 | yes |
| production--dino_global--local_heads--evidence--budget-20 | 76.69 | 99.76 | 86.71 | 70.21 | 81.47 | 75.42 | 73.13 | 82.82 | 77.67 | 27.49 | 347.33 | 123.33 | yes |
| production--compact_boost--local_heads--evidence--budget-20 | 76.55 | 99.55 | 86.55 | 70.46 | 81.47 | 75.56 | 75.62 | 85.09 | 80.08 | 27.48 | 311.33 | 127.00 | yes |

### individual--budget-20

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| individual--dino_global--local_heads--chronological--budget-20 | 93.19 | 98.11 | 95.58 | 85.85 | 87.99 | 86.89 | 88.53 | 89.03 | 88.76 | 24.82 | 227.00 | 110.67 | yes |
| individual--dino_global--local_events--chronological--budget-20 | 92.69 | 98.51 | 95.51 | 88.75 | 88.92 | 88.83 | 88.69 | 87.68 | 88.18 | 18.86 | 142.00 | 73.00 | yes |
| individual--dino_global--local_heads--evidence--budget-20 | 92.96 | 98.20 | 95.50 | 87.96 | 89.54 | 88.73 | 89.16 | 89.13 | 89.13 | 25.04 | 246.67 | 123.33 | yes |
| individual--dino_global--local_events--evidence--budget-20 | 92.84 | 98.32 | 95.49 | 89.81 | 89.23 | 89.52 | 88.82 | 87.16 | 87.98 | 19.05 | 136.00 | 76.00 | yes |
| individual--dino_boost--local_events--evidence--budget-20 | 91.93 | 98.21 | 94.96 | 86.55 | 87.27 | 86.90 | 89.43 | 89.13 | 89.27 | 19.91 | 140.33 | 78.67 | yes |
| individual--dino_boost--local_heads--evidence--budget-20 | 92.04 | 97.98 | 94.91 | 84.73 | 86.75 | 85.73 | 88.91 | 88.72 | 88.81 | 25.79 | 256.33 | 123.00 | yes |
| individual--dino_boost--local_events--chronological--budget-20 | 91.63 | 98.35 | 94.87 | 84.70 | 86.54 | 85.60 | 88.58 | 88.82 | 88.69 | 19.80 | 149.00 | 76.33 | yes |
| individual--dino_boost--local_heads--chronological--budget-20 | 91.67 | 98.03 | 94.73 | 81.99 | 85.30 | 83.61 | 87.27 | 87.99 | 87.63 | 25.54 | 230.00 | 110.00 | yes |
| individual--dino_global--legacy--chronological--budget-20 | 91.61 | 97.89 | 94.64 | 82.14 | 84.37 | 83.23 | 85.22 | 86.34 | 85.76 | 27.02 | 196.67 | 98.33 | yes |
| individual--dino_global--legacy--evidence--budget-20 | 91.37 | 98.11 | 94.61 | 82.37 | 86.54 | 84.39 | 85.81 | 87.27 | 86.51 | 27.41 | 180.33 | 123.33 | yes |
| individual--dino_boost--legacy--chronological--budget-20 | 90.59 | 97.50 | 93.91 | 78.28 | 82.09 | 80.14 | 84.60 | 87.47 | 86.01 | 27.06 | 196.33 | 92.00 | yes |
| individual--dino_boost--legacy--evidence--budget-20 | 90.40 | 97.59 | 93.86 | 77.21 | 82.40 | 79.72 | 86.30 | 88.61 | 87.44 | 27.46 | 172.33 | 122.00 | yes |
| individual--compact_boost--local_heads--evidence--budget-20 | 92.22 | 95.40 | 93.78 | 83.94 | 81.06 | 82.47 | 87.56 | 83.02 | 85.23 | 24.28 | 229.00 | 100.00 | yes |
| individual--compact_boost--local_events--evidence--budget-20 | 92.39 | 95.12 | 93.73 | 83.41 | 80.12 | 81.74 | 86.47 | 81.99 | 84.17 | 23.58 | 133.33 | 90.33 | yes |
| individual--compact_boost--local_events--chronological--budget-20 | 91.54 | 95.91 | 93.67 | 81.02 | 81.37 | 81.19 | 84.63 | 83.75 | 84.18 | 23.34 | 161.67 | 91.00 | yes |
| individual--compact_boost--local_heads--chronological--budget-20 | 91.56 | 95.31 | 93.39 | 79.94 | 80.43 | 80.19 | 84.28 | 83.75 | 84.01 | 24.11 | 211.67 | 100.67 | yes |
| individual--compact_boost--legacy--evidence--budget-20 | 90.64 | 95.44 | 92.97 | 78.29 | 79.50 | 78.89 | 83.98 | 81.88 | 82.91 | 27.43 | 183.33 | 111.00 | yes |
| individual--compact_boost--legacy--chronological--budget-20 | 90.54 | 94.50 | 92.48 | 76.76 | 77.64 | 77.20 | 81.08 | 80.75 | 80.91 | 27.03 | 194.33 | 93.33 | yes |

### production--budget-40

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--dino_global--local_events--evidence--budget-40 | 88.17 | 99.87 | 93.66 | 86.60 | 92.24 | 89.33 | 81.73 | 85.20 | 83.42 | 55.05 | 392.00 | 205.00 | yes |
| production--dino_boost--legacy--chronological--budget-40 | 87.64 | 99.79 | 93.32 | 84.43 | 87.58 | 85.98 | 85.19 | 88.10 | 86.62 | 45.43 | 155.67 | 117.67 | yes |
| production--dino_boost--local_events--evidence--budget-40 | 87.48 | 99.92 | 93.29 | 85.51 | 92.24 | 88.74 | 81.28 | 85.82 | 83.49 | 55.03 | 410.33 | 202.33 | yes |
| production--dino_boost--legacy--evidence--budget-40 | 87.48 | 99.73 | 93.20 | 82.89 | 87.27 | 85.02 | 83.73 | 87.89 | 85.76 | 45.62 | 149.00 | 116.33 | yes |
| production--dino_global--legacy--chronological--budget-40 | 87.44 | 99.77 | 93.20 | 83.91 | 87.47 | 85.66 | 84.48 | 87.89 | 86.15 | 44.44 | 149.33 | 117.00 | yes |
| production--dino_global--legacy--evidence--budget-40 | 87.28 | 99.69 | 93.07 | 83.02 | 87.58 | 85.24 | 83.26 | 87.47 | 85.31 | 44.66 | 143.67 | 115.33 | yes |
| production--compact_boost--local_events--evidence--budget-40 | 87.21 | 99.78 | 93.07 | 84.18 | 90.89 | 87.41 | 81.36 | 85.40 | 83.33 | 54.98 | 385.33 | 198.00 | yes |
| production--compact_boost--legacy--chronological--budget-40 | 86.84 | 99.53 | 92.75 | 82.27 | 86.44 | 84.30 | 82.82 | 86.85 | 84.79 | 44.91 | 151.67 | 125.00 | yes |
| production--compact_boost--legacy--evidence--budget-40 | 86.24 | 99.53 | 92.41 | 80.76 | 85.61 | 83.12 | 81.71 | 86.44 | 84.01 | 44.95 | 145.00 | 121.00 | yes |
| production--dino_global--local_events--chronological--budget-40 | 85.07 | 99.60 | 91.76 | 80.72 | 86.23 | 83.38 | 82.22 | 86.65 | 84.38 | 54.68 | 405.00 | 172.00 | yes |
| production--dino_global--local_heads--chronological--budget-40 | 84.96 | 99.60 | 91.70 | 80.50 | 86.34 | 83.32 | 82.17 | 86.85 | 84.45 | 54.66 | 720.00 | 170.67 | yes |
| production--dino_boost--local_events--chronological--budget-40 | 84.82 | 99.60 | 91.62 | 80.33 | 86.23 | 83.17 | 82.05 | 87.06 | 84.48 | 54.65 | 411.00 | 170.00 | yes |
| production--dino_boost--local_heads--chronological--budget-40 | 84.77 | 99.60 | 91.59 | 80.10 | 86.23 | 83.05 | 81.77 | 86.85 | 84.24 | 54.62 | 816.00 | 169.33 | yes |
| production--compact_boost--local_heads--chronological--budget-40 | 84.42 | 99.60 | 91.38 | 79.92 | 86.13 | 82.91 | 81.82 | 86.65 | 84.16 | 54.67 | 667.67 | 169.33 | yes |
| production--compact_boost--local_events--chronological--budget-40 | 84.34 | 99.54 | 91.31 | 80.14 | 86.02 | 82.98 | 81.81 | 86.13 | 83.91 | 54.56 | 398.33 | 169.67 | yes |
| production--dino_boost--local_heads--evidence--budget-40 | 81.01 | 99.83 | 89.44 | 74.85 | 88.41 | 81.06 | 80.18 | 90.48 | 85.02 | 55.10 | 764.67 | 235.33 | yes |
| production--compact_boost--local_heads--evidence--budget-40 | 81.02 | 99.69 | 89.39 | 75.14 | 88.20 | 81.15 | 82.46 | 92.44 | 87.17 | 55.05 | 606.67 | 243.33 | yes |
| production--dino_global--local_heads--evidence--budget-40 | 80.77 | 99.81 | 89.29 | 74.37 | 88.61 | 80.87 | 79.47 | 90.17 | 84.48 | 55.10 | 682.00 | 238.67 | yes |

### individual--budget-40

Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.

| Arm | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | Proposals selected | True rallies in edit regions | Non-worsening screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| individual--dino_global--local_heads--chronological--budget-40 | 94.84 | 98.85 | 96.80 | 90.92 | 92.13 | 91.52 | 93.49 | 92.24 | 92.85 | 39.46 | 356.00 | 174.00 | yes |
| individual--dino_global--local_heads--evidence--budget-40 | 94.81 | 98.77 | 96.74 | 90.98 | 92.75 | 91.85 | 93.33 | 92.65 | 92.98 | 39.49 | 360.00 | 183.33 | yes |
| individual--dino_boost--local_heads--chronological--budget-40 | 94.36 | 98.74 | 96.49 | 90.00 | 91.30 | 90.65 | 94.26 | 91.82 | 93.02 | 47.12 | 426.00 | 204.67 | yes |
| individual--dino_global--legacy--chronological--budget-40 | 94.22 | 98.71 | 96.41 | 88.17 | 89.96 | 89.05 | 91.23 | 90.17 | 90.68 | 54.65 | 394.67 | 196.33 | yes |
| individual--dino_boost--local_heads--evidence--budget-40 | 94.10 | 98.83 | 96.40 | 90.42 | 92.75 | 91.57 | 94.29 | 92.34 | 93.30 | 47.26 | 426.67 | 214.00 | yes |
| individual--dino_global--legacy--evidence--budget-40 | 93.83 | 98.54 | 96.12 | 89.70 | 91.61 | 90.64 | 93.91 | 91.10 | 92.47 | 54.99 | 376.33 | 234.67 | yes |
| individual--dino_global--local_events--chronological--budget-40 | 92.99 | 98.94 | 95.87 | 90.46 | 91.41 | 90.93 | 89.58 | 89.13 | 89.35 | 22.24 | 164.00 | 87.00 | yes |
| individual--dino_global--local_events--evidence--budget-40 | 92.99 | 98.94 | 95.87 | 90.46 | 91.41 | 90.93 | 89.58 | 89.13 | 89.35 | 22.24 | 164.00 | 87.00 | yes |
| individual--compact_boost--local_heads--evidence--budget-40 | 94.01 | 97.72 | 95.82 | 88.86 | 88.92 | 88.89 | 93.14 | 89.96 | 91.52 | 41.31 | 373.67 | 171.33 | yes |
| individual--compact_boost--local_events--evidence--budget-40 | 93.81 | 97.77 | 95.75 | 89.50 | 88.10 | 88.79 | 91.22 | 88.20 | 89.68 | 35.76 | 238.33 | 131.67 | yes |
| individual--compact_boost--local_heads--chronological--budget-40 | 93.82 | 97.70 | 95.72 | 88.10 | 88.72 | 88.40 | 92.06 | 90.06 | 91.05 | 41.13 | 370.33 | 164.67 | yes |
| individual--compact_boost--local_events--chronological--budget-40 | 93.62 | 97.79 | 95.66 | 89.24 | 88.30 | 88.77 | 91.04 | 88.30 | 89.65 | 35.74 | 241.00 | 131.33 | yes |
| individual--dino_boost--legacy--chronological--budget-40 | 92.93 | 98.45 | 95.60 | 84.91 | 88.51 | 86.67 | 90.02 | 90.58 | 90.30 | 54.67 | 400.67 | 185.00 | yes |
| individual--dino_boost--legacy--evidence--budget-40 | 92.66 | 98.37 | 95.43 | 86.10 | 89.34 | 87.68 | 95.20 | 93.27 | 94.21 | 54.94 | 380.00 | 242.33 | yes |
| individual--dino_boost--local_events--chronological--budget-40 | 92.30 | 98.61 | 95.35 | 87.57 | 89.23 | 88.39 | 90.34 | 89.96 | 90.14 | 24.40 | 178.33 | 92.33 | yes |
| individual--dino_boost--local_events--evidence--budget-40 | 92.30 | 98.61 | 95.35 | 87.57 | 89.23 | 88.39 | 90.34 | 89.96 | 90.14 | 24.40 | 178.33 | 92.33 | yes |
| individual--compact_boost--legacy--evidence--budget-40 | 93.14 | 97.28 | 95.16 | 87.43 | 87.16 | 87.30 | 94.20 | 89.03 | 91.54 | 54.96 | 387.33 | 213.67 | yes |
| individual--compact_boost--legacy--chronological--budget-40 | 92.93 | 96.65 | 94.75 | 84.19 | 84.89 | 84.54 | 88.65 | 86.54 | 87.58 | 54.71 | 400.00 | 181.00 | yes |

## Rally separation and boundary censoring

Merge/split counts first use any positive overlap. Material variants require at least min(0.5 seconds, 10% of gold rally duration). Complete misses have no raw overlap at all; they differ from IoU-based unmatched events. Synthetic/censored boundaries can leave partial event objects, so event F1 need not improve monotonically as more video is reviewed.

| Arm | Raw predictions | Complete misses | Merged predictions | Split gold rallies | Material merges | Material splits | Censored starts | Censored ends | Unobserved starts | Unobserved ends | Touched rallies with uneditable boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--compact_boost--legacy--chronological--budget-05 | 356.00 | 3.00 | 7.00 | 4.00 | 7.00 | 4.00 | 0.00 | 1.00 | 0.00 | 2.00 | 1.00 |
| production--compact_boost--legacy--chronological--budget-10 | 355.67 | 3.00 | 5.00 | 3.33 | 5.00 | 3.33 | 0.00 | 2.33 | 0.67 | 4.00 | 2.67 |
| production--compact_boost--legacy--chronological--budget-20 | 350.00 | 3.00 | 3.67 | 3.33 | 3.67 | 3.33 | 0.00 | 2.67 | 0.67 | 5.33 | 3.00 |
| production--compact_boost--legacy--chronological--budget-40 | 338.33 | 3.00 | 0.67 | 3.00 | 0.67 | 3.00 | 0.00 | 2.67 | 0.67 | 5.33 | 3.33 |
| production--compact_boost--legacy--evidence--budget-05 | 359.67 | 3.00 | 6.00 | 2.67 | 6.00 | 2.67 | 0.00 | 0.00 | 0.67 | 1.67 | 0.33 |
| production--compact_boost--legacy--evidence--budget-10 | 358.67 | 3.00 | 5.33 | 2.67 | 5.33 | 2.67 | 0.00 | 0.33 | 0.67 | 3.00 | 0.67 |
| production--compact_boost--legacy--evidence--budget-20 | 355.67 | 3.00 | 3.67 | 2.67 | 3.67 | 2.67 | 0.00 | 0.33 | 0.67 | 3.00 | 0.67 |
| production--compact_boost--legacy--evidence--budget-40 | 341.33 | 3.00 | 1.67 | 3.00 | 1.67 | 3.00 | 0.00 | 2.67 | 1.67 | 5.33 | 3.33 |
| production--compact_boost--local_events--chronological--budget-05 | 357.33 | 3.00 | 7.00 | 4.00 | 7.00 | 4.00 | 0.00 | 1.00 | 0.33 | 2.00 | 3.67 |
| production--compact_boost--local_events--chronological--budget-10 | 358.67 | 3.00 | 5.00 | 3.00 | 5.00 | 3.00 | 0.00 | 1.00 | 1.00 | 4.67 | 7.33 |
| production--compact_boost--local_events--chronological--budget-20 | 357.67 | 2.33 | 3.67 | 2.00 | 3.67 | 2.00 | 0.00 | 1.00 | 3.67 | 9.00 | 12.33 |
| production--compact_boost--local_events--chronological--budget-40 | 345.67 | 2.33 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 2.67 | 6.67 | 17.33 | 19.33 |
| production--compact_boost--local_events--evidence--budget-05 | 361.00 | 3.00 | 6.00 | 2.33 | 6.00 | 2.33 | 0.00 | 0.67 | 1.00 | 5.00 | 13.33 |
| production--compact_boost--local_events--evidence--budget-10 | 357.33 | 2.67 | 5.67 | 1.33 | 5.67 | 1.33 | 0.00 | 1.67 | 2.67 | 8.00 | 31.67 |
| production--compact_boost--local_events--evidence--budget-20 | 359.67 | 2.33 | 3.67 | 1.67 | 3.67 | 1.33 | 0.67 | 1.67 | 9.00 | 11.67 | 65.33 |
| production--compact_boost--local_events--evidence--budget-40 | 347.67 | 0.67 | 1.67 | 1.67 | 1.67 | 1.33 | 1.33 | 2.67 | 10.67 | 27.00 | 105.00 |
| production--compact_boost--local_heads--chronological--budget-05 | 358.00 | 3.00 | 6.33 | 4.00 | 6.33 | 4.00 | 0.00 | 1.00 | 0.67 | 2.33 | 4.33 |
| production--compact_boost--local_heads--chronological--budget-10 | 359.33 | 3.00 | 5.00 | 3.00 | 5.00 | 3.00 | 0.00 | 1.00 | 1.67 | 4.00 | 6.67 |
| production--compact_boost--local_heads--chronological--budget-20 | 357.33 | 2.33 | 3.67 | 2.00 | 3.67 | 2.00 | 0.00 | 1.00 | 2.33 | 7.33 | 10.67 |
| production--compact_boost--local_heads--chronological--budget-40 | 347.00 | 2.33 | 0.67 | 1.00 | 0.67 | 1.00 | 0.00 | 3.00 | 6.00 | 15.67 | 16.33 |
| production--compact_boost--local_heads--evidence--budget-05 | 363.67 | 3.00 | 6.00 | 2.67 | 6.00 | 2.67 | 0.00 | 0.33 | 2.67 | 6.00 | 11.67 |
| production--compact_boost--local_heads--evidence--budget-10 | 366.00 | 3.00 | 5.33 | 2.00 | 5.33 | 2.00 | 0.33 | 0.33 | 5.33 | 8.67 | 31.00 |
| production--compact_boost--local_heads--evidence--budget-20 | 372.33 | 2.67 | 4.33 | 1.67 | 4.33 | 1.67 | 0.00 | 0.67 | 10.33 | 13.00 | 82.00 |
| production--compact_boost--local_heads--evidence--budget-40 | 378.00 | 2.00 | 3.00 | 1.67 | 3.00 | 1.67 | 0.33 | 2.00 | 17.67 | 21.33 | 139.67 |
| production--dino_global--legacy--chronological--budget-05 | 355.67 | 2.33 | 7.00 | 4.00 | 7.00 | 4.00 | 0.00 | 1.00 | 0.33 | 2.00 | 1.33 |
| production--dino_global--legacy--chronological--budget-10 | 354.33 | 2.33 | 5.00 | 3.00 | 5.00 | 3.00 | 0.00 | 1.00 | 0.33 | 2.67 | 2.00 |
| production--dino_global--legacy--chronological--budget-20 | 348.67 | 2.33 | 3.33 | 3.00 | 3.33 | 3.00 | 0.00 | 1.33 | 0.33 | 3.33 | 2.00 |
| production--dino_global--legacy--chronological--budget-40 | 335.67 | 2.00 | 0.33 | 2.00 | 0.33 | 2.00 | 0.00 | 2.67 | 0.67 | 5.33 | 3.33 |
| production--dino_global--legacy--evidence--budget-05 | 359.67 | 3.00 | 5.33 | 3.00 | 5.33 | 3.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| production--dino_global--legacy--evidence--budget-10 | 357.00 | 3.00 | 4.33 | 2.33 | 4.33 | 2.33 | 0.00 | 0.00 | 1.33 | 1.33 | 0.00 |
| production--dino_global--legacy--evidence--budget-20 | 354.00 | 3.00 | 2.00 | 2.00 | 2.00 | 2.00 | 0.00 | 0.67 | 2.00 | 2.33 | 1.00 |
| production--dino_global--legacy--evidence--budget-40 | 339.67 | 2.00 | 0.33 | 2.00 | 0.33 | 2.00 | 0.00 | 2.00 | 2.33 | 5.00 | 2.67 |
| production--dino_global--local_events--chronological--budget-05 | 357.33 | 3.00 | 7.00 | 4.00 | 7.00 | 4.00 | 0.00 | 0.67 | 0.33 | 1.67 | 3.67 |
| production--dino_global--local_events--chronological--budget-10 | 358.33 | 3.00 | 5.00 | 3.00 | 5.00 | 3.00 | 0.00 | 0.67 | 0.67 | 4.00 | 7.33 |
| production--dino_global--local_events--chronological--budget-20 | 357.33 | 2.00 | 3.00 | 2.00 | 3.00 | 2.00 | 0.00 | 1.00 | 2.00 | 7.00 | 12.33 |
| production--dino_global--local_events--chronological--budget-40 | 344.00 | 2.00 | 0.67 | 1.00 | 0.67 | 1.00 | 0.00 | 2.33 | 4.67 | 15.67 | 21.33 |
| production--dino_global--local_events--evidence--budget-05 | 364.00 | 2.67 | 4.00 | 2.33 | 4.00 | 2.33 | 0.00 | 0.00 | 1.33 | 3.67 | 11.67 |
| production--dino_global--local_events--evidence--budget-10 | 361.67 | 2.00 | 3.33 | 1.00 | 3.33 | 1.00 | 0.00 | 0.33 | 2.33 | 6.00 | 27.00 |
| production--dino_global--local_events--evidence--budget-20 | 357.00 | 1.33 | 2.67 | 0.00 | 2.67 | 0.00 | 0.00 | 1.00 | 4.00 | 10.00 | 66.33 |
| production--dino_global--local_events--evidence--budget-40 | 343.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.33 | 2.00 | 8.00 | 22.00 | 118.67 |
| production--dino_global--local_heads--chronological--budget-05 | 358.00 | 3.00 | 6.67 | 4.00 | 6.67 | 4.00 | 0.00 | 0.67 | 0.67 | 2.00 | 3.00 |
| production--dino_global--local_heads--chronological--budget-10 | 358.67 | 3.00 | 5.00 | 3.00 | 5.00 | 3.00 | 0.00 | 0.67 | 1.00 | 3.67 | 6.67 |
| production--dino_global--local_heads--chronological--budget-20 | 357.67 | 2.00 | 3.00 | 2.00 | 3.00 | 2.00 | 0.00 | 0.67 | 1.67 | 5.67 | 9.00 |
| production--dino_global--local_heads--chronological--budget-40 | 345.33 | 2.00 | 0.33 | 1.00 | 0.33 | 1.00 | 0.00 | 2.33 | 5.33 | 13.67 | 17.67 |
| production--dino_global--local_heads--evidence--budget-05 | 368.67 | 3.00 | 4.33 | 3.00 | 4.33 | 3.00 | 0.00 | 0.00 | 3.67 | 5.00 | 12.33 |
| production--dino_global--local_heads--evidence--budget-10 | 368.33 | 2.00 | 3.00 | 1.67 | 3.00 | 1.67 | 0.00 | 0.00 | 5.00 | 7.67 | 29.67 |
| production--dino_global--local_heads--evidence--budget-20 | 373.67 | 1.67 | 2.67 | 0.33 | 2.67 | 0.33 | 0.33 | 0.33 | 9.00 | 13.33 | 77.00 |
| production--dino_global--local_heads--evidence--budget-40 | 383.67 | 1.33 | 1.67 | 0.00 | 1.67 | 0.00 | 0.00 | 1.67 | 18.33 | 21.67 | 123.33 |
| production--dino_boost--legacy--chronological--budget-05 | 356.67 | 2.67 | 7.00 | 4.33 | 7.00 | 4.00 | 0.33 | 1.00 | 0.33 | 2.00 | 1.33 |
| production--dino_boost--legacy--chronological--budget-10 | 355.00 | 2.33 | 5.33 | 3.33 | 5.33 | 3.00 | 0.33 | 1.00 | 0.67 | 2.33 | 1.33 |
| production--dino_boost--legacy--chronological--budget-20 | 348.00 | 2.00 | 4.00 | 3.33 | 4.00 | 3.00 | 0.33 | 1.33 | 0.67 | 3.67 | 1.67 |
| production--dino_boost--legacy--chronological--budget-40 | 334.00 | 1.33 | 1.00 | 2.33 | 1.00 | 2.00 | 0.33 | 2.00 | 1.00 | 5.00 | 2.67 |
| production--dino_boost--legacy--evidence--budget-05 | 358.00 | 3.00 | 5.67 | 3.00 | 5.67 | 3.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| production--dino_boost--legacy--evidence--budget-10 | 354.33 | 3.00 | 4.67 | 2.33 | 4.67 | 2.33 | 0.00 | 0.00 | 1.33 | 1.33 | 0.00 |
| production--dino_boost--legacy--evidence--budget-20 | 348.00 | 2.33 | 3.67 | 2.33 | 3.67 | 2.33 | 0.00 | 0.00 | 2.00 | 1.67 | 0.00 |
| production--dino_boost--legacy--evidence--budget-40 | 339.00 | 1.33 | 1.00 | 2.33 | 1.00 | 2.00 | 0.33 | 1.33 | 2.00 | 5.00 | 1.67 |
| production--dino_boost--local_events--chronological--budget-05 | 358.00 | 3.00 | 7.00 | 4.00 | 7.00 | 4.00 | 0.00 | 1.00 | 0.00 | 2.00 | 4.67 |
| production--dino_boost--local_events--chronological--budget-10 | 359.00 | 2.33 | 5.33 | 3.00 | 5.33 | 3.00 | 0.00 | 1.00 | 1.00 | 3.33 | 9.00 |
| production--dino_boost--local_events--chronological--budget-20 | 356.00 | 2.00 | 3.67 | 2.00 | 3.67 | 2.00 | 0.00 | 1.00 | 1.33 | 5.67 | 11.67 |
| production--dino_boost--local_events--chronological--budget-40 | 345.67 | 2.00 | 0.67 | 1.00 | 0.67 | 1.00 | 0.00 | 2.33 | 4.00 | 14.00 | 20.33 |
| production--dino_boost--local_events--evidence--budget-05 | 361.67 | 3.00 | 5.33 | 2.00 | 5.33 | 2.00 | 0.00 | 0.00 | 1.00 | 3.67 | 10.00 |
| production--dino_boost--local_events--evidence--budget-10 | 360.00 | 2.00 | 3.67 | 1.00 | 3.67 | 1.00 | 0.00 | 0.33 | 1.67 | 5.33 | 25.33 |
| production--dino_boost--local_events--evidence--budget-20 | 358.67 | 2.00 | 2.67 | 0.00 | 2.67 | 0.00 | 0.00 | 0.67 | 3.00 | 9.67 | 59.67 |
| production--dino_boost--local_events--evidence--budget-40 | 347.33 | 0.67 | 1.00 | 0.00 | 1.00 | 0.00 | 0.67 | 1.00 | 8.33 | 18.00 | 115.67 |
| production--dino_boost--local_heads--chronological--budget-05 | 358.00 | 3.00 | 6.67 | 4.00 | 6.67 | 4.00 | 0.00 | 1.00 | 0.33 | 2.33 | 4.00 |
| production--dino_boost--local_heads--chronological--budget-10 | 360.00 | 2.33 | 5.33 | 3.00 | 5.33 | 3.00 | 0.00 | 1.00 | 2.00 | 3.67 | 6.67 |
| production--dino_boost--local_heads--chronological--budget-20 | 357.33 | 2.00 | 3.67 | 2.00 | 3.67 | 2.00 | 0.00 | 1.00 | 2.33 | 7.00 | 10.00 |
| production--dino_boost--local_heads--chronological--budget-40 | 346.67 | 2.00 | 0.67 | 1.00 | 0.67 | 1.00 | 0.00 | 2.33 | 4.67 | 13.67 | 19.33 |
| production--dino_boost--local_heads--evidence--budget-05 | 364.67 | 2.67 | 5.33 | 2.00 | 5.33 | 2.00 | 0.00 | 0.67 | 1.33 | 5.33 | 10.33 |
| production--dino_boost--local_heads--evidence--budget-10 | 364.33 | 2.33 | 3.67 | 1.33 | 3.67 | 1.33 | 0.00 | 0.67 | 2.00 | 7.33 | 23.67 |
| production--dino_boost--local_heads--evidence--budget-20 | 367.67 | 2.00 | 2.67 | 0.00 | 2.67 | 0.00 | 0.00 | 2.00 | 5.33 | 14.00 | 64.00 |
| production--dino_boost--local_heads--evidence--budget-40 | 380.33 | 1.00 | 2.00 | 0.00 | 2.00 | 0.00 | 0.00 | 2.67 | 17.00 | 23.00 | 125.33 |
| individual--compact_boost--legacy--chronological--budget-05 | 323.33 | 32.00 | 2.00 | 11.33 | 2.00 | 10.33 | 0.00 | 0.67 | 0.33 | 1.33 | 10.33 |
| individual--compact_boost--legacy--chronological--budget-10 | 325.33 | 29.00 | 2.00 | 10.33 | 2.00 | 9.33 | 0.33 | 1.00 | 1.67 | 3.00 | 20.67 |
| individual--compact_boost--legacy--chronological--budget-20 | 325.67 | 24.00 | 2.00 | 8.33 | 2.00 | 7.67 | 0.67 | 0.67 | 5.00 | 10.00 | 43.33 |
| individual--compact_boost--legacy--chronological--budget-40 | 324.67 | 16.00 | 2.00 | 4.67 | 2.00 | 4.67 | 1.33 | 2.00 | 10.33 | 22.67 | 78.33 |
| individual--compact_boost--legacy--evidence--budget-05 | 326.33 | 31.33 | 2.00 | 13.00 | 2.00 | 12.00 | 3.33 | 0.33 | 4.00 | 1.67 | 17.67 |
| individual--compact_boost--legacy--evidence--budget-10 | 327.67 | 28.67 | 2.00 | 12.00 | 2.00 | 11.33 | 3.00 | 2.33 | 6.67 | 5.67 | 33.67 |
| individual--compact_boost--legacy--evidence--budget-20 | 327.00 | 24.67 | 2.00 | 10.00 | 2.00 | 9.33 | 4.33 | 4.67 | 13.00 | 12.00 | 71.33 |
| individual--compact_boost--legacy--evidence--budget-40 | 321.00 | 16.33 | 2.00 | 2.00 | 2.00 | 2.00 | 2.67 | 7.00 | 16.67 | 26.33 | 132.33 |
| individual--compact_boost--local_events--chronological--budget-05 | 322.67 | 30.00 | 2.00 | 10.00 | 2.00 | 9.00 | 1.00 | 3.33 | 1.33 | 4.00 | 9.67 |
| individual--compact_boost--local_events--chronological--budget-10 | 322.33 | 26.33 | 2.00 | 10.00 | 2.00 | 9.33 | 1.67 | 5.33 | 2.00 | 7.00 | 15.33 |
| individual--compact_boost--local_events--chronological--budget-20 | 323.33 | 19.00 | 2.00 | 9.33 | 2.00 | 9.33 | 4.00 | 9.67 | 4.67 | 12.00 | 24.33 |
| individual--compact_boost--local_events--chronological--budget-40 | 318.67 | 12.00 | 2.00 | 6.67 | 2.00 | 6.67 | 5.67 | 12.67 | 6.33 | 15.67 | 33.33 |
| individual--compact_boost--local_events--evidence--budget-05 | 318.33 | 33.33 | 2.00 | 10.33 | 2.00 | 9.67 | 0.67 | 3.33 | 0.67 | 3.67 | 5.00 |
| individual--compact_boost--local_events--evidence--budget-10 | 309.67 | 33.00 | 2.00 | 8.33 | 2.00 | 8.00 | 2.00 | 4.67 | 2.33 | 5.00 | 11.33 |
| individual--compact_boost--local_events--evidence--budget-20 | 309.33 | 27.00 | 2.00 | 7.00 | 2.00 | 7.00 | 3.67 | 8.67 | 4.00 | 9.67 | 18.67 |
| individual--compact_boost--local_events--evidence--budget-40 | 317.00 | 12.33 | 2.00 | 6.67 | 2.00 | 6.67 | 5.00 | 12.67 | 5.67 | 15.33 | 32.33 |
| individual--compact_boost--local_heads--chronological--budget-05 | 323.00 | 31.33 | 2.00 | 10.33 | 2.00 | 9.33 | 0.33 | 4.33 | 0.67 | 5.33 | 18.00 |
| individual--compact_boost--local_heads--chronological--budget-10 | 324.67 | 27.67 | 2.00 | 10.00 | 2.00 | 9.00 | 0.33 | 8.00 | 2.33 | 10.00 | 32.33 |
| individual--compact_boost--local_heads--chronological--budget-20 | 324.00 | 22.33 | 1.67 | 8.33 | 1.67 | 7.67 | 1.00 | 13.67 | 4.00 | 23.67 | 52.00 |
| individual--compact_boost--local_heads--chronological--budget-40 | 324.33 | 13.33 | 1.67 | 5.67 | 1.67 | 5.00 | 4.33 | 20.00 | 9.33 | 33.67 | 82.67 |
| individual--compact_boost--local_heads--evidence--budget-05 | 317.33 | 33.67 | 2.00 | 10.67 | 2.00 | 10.33 | 0.67 | 3.33 | 1.00 | 4.00 | 12.67 |
| individual--compact_boost--local_heads--evidence--budget-10 | 312.67 | 31.00 | 2.00 | 8.33 | 2.00 | 8.00 | 2.00 | 5.67 | 2.67 | 6.33 | 27.00 |
| individual--compact_boost--local_heads--evidence--budget-20 | 311.00 | 25.33 | 2.00 | 7.00 | 2.00 | 6.67 | 3.67 | 9.67 | 5.67 | 11.33 | 50.33 |
| individual--compact_boost--local_heads--evidence--budget-40 | 322.33 | 13.33 | 1.67 | 5.67 | 1.67 | 5.33 | 5.00 | 17.33 | 11.33 | 27.00 | 86.33 |
| individual--dino_global--legacy--chronological--budget-05 | 334.33 | 17.33 | 0.67 | 8.00 | 0.67 | 7.00 | 0.33 | 0.00 | 0.33 | 1.33 | 13.33 |
| individual--dino_global--legacy--chronological--budget-10 | 334.67 | 16.00 | 0.67 | 7.33 | 0.67 | 6.33 | 0.33 | 0.00 | 1.67 | 3.33 | 28.33 |
| individual--dino_global--legacy--chronological--budget-20 | 331.00 | 13.33 | 0.67 | 5.00 | 0.67 | 4.67 | 0.33 | 1.33 | 4.67 | 9.33 | 49.00 |
| individual--dino_global--legacy--chronological--budget-40 | 328.67 | 9.00 | 0.67 | 3.33 | 0.67 | 3.33 | 2.00 | 3.33 | 10.33 | 24.00 | 96.00 |
| individual--dino_global--legacy--evidence--budget-05 | 339.00 | 16.67 | 0.67 | 8.33 | 0.67 | 7.33 | 1.67 | 0.33 | 4.00 | 1.67 | 22.00 |
| individual--dino_global--legacy--evidence--budget-10 | 339.67 | 15.67 | 0.67 | 8.00 | 0.67 | 7.33 | 2.00 | 1.33 | 6.67 | 4.33 | 45.33 |
| individual--dino_global--legacy--evidence--budget-20 | 338.67 | 12.33 | 0.67 | 6.00 | 0.67 | 5.33 | 2.00 | 2.00 | 11.00 | 7.00 | 84.00 |
| individual--dino_global--legacy--evidence--budget-40 | 329.00 | 10.33 | 0.67 | 2.67 | 0.67 | 2.33 | 2.67 | 6.00 | 16.67 | 20.00 | 159.33 |
| individual--dino_global--local_events--chronological--budget-05 | 331.67 | 14.67 | 0.67 | 6.33 | 0.67 | 5.67 | 0.00 | 2.00 | 1.00 | 4.67 | 9.67 |
| individual--dino_global--local_events--chronological--budget-10 | 326.33 | 12.67 | 0.67 | 5.33 | 0.67 | 5.00 | 0.67 | 3.00 | 3.00 | 9.67 | 15.67 |
| individual--dino_global--local_events--chronological--budget-20 | 322.67 | 9.00 | 0.67 | 5.00 | 0.67 | 4.67 | 1.33 | 4.00 | 4.33 | 13.00 | 21.00 |
| individual--dino_global--local_events--chronological--budget-40 | 325.33 | 6.33 | 0.67 | 5.33 | 0.67 | 5.00 | 2.00 | 4.33 | 5.00 | 14.33 | 23.00 |
| individual--dino_global--local_events--evidence--budget-05 | 329.67 | 16.67 | 0.67 | 7.33 | 0.67 | 6.67 | 0.33 | 2.00 | 0.67 | 3.33 | 5.33 |
| individual--dino_global--local_events--evidence--budget-10 | 322.33 | 13.67 | 0.67 | 4.67 | 0.67 | 4.33 | 1.00 | 3.00 | 2.67 | 7.00 | 11.67 |
| individual--dino_global--local_events--evidence--budget-20 | 320.00 | 12.00 | 0.67 | 5.00 | 0.67 | 4.67 | 1.33 | 4.00 | 4.00 | 12.67 | 19.67 |
| individual--dino_global--local_events--evidence--budget-40 | 325.33 | 6.33 | 0.67 | 5.33 | 0.67 | 5.00 | 2.00 | 4.33 | 5.00 | 14.33 | 23.00 |
| individual--dino_global--local_heads--chronological--budget-05 | 335.00 | 16.67 | 0.67 | 8.33 | 0.67 | 7.33 | 0.00 | 1.67 | 0.67 | 5.00 | 16.00 |
| individual--dino_global--local_heads--chronological--budget-10 | 332.33 | 14.33 | 0.67 | 7.33 | 0.67 | 6.67 | 0.33 | 2.33 | 3.00 | 7.00 | 34.67 |
| individual--dino_global--local_heads--chronological--budget-20 | 330.33 | 11.67 | 0.33 | 6.33 | 0.33 | 5.67 | 1.33 | 4.33 | 6.33 | 15.00 | 59.33 |
| individual--dino_global--local_heads--chronological--budget-40 | 326.33 | 8.67 | 0.33 | 4.33 | 0.33 | 4.00 | 2.00 | 7.00 | 8.67 | 22.67 | 90.67 |
| individual--dino_global--local_heads--evidence--budget-05 | 327.00 | 17.00 | 0.67 | 7.00 | 0.67 | 6.33 | 0.67 | 1.33 | 1.33 | 2.33 | 10.00 |
| individual--dino_global--local_heads--evidence--budget-10 | 324.33 | 15.33 | 0.67 | 6.33 | 0.67 | 5.67 | 0.67 | 3.67 | 2.33 | 7.67 | 31.33 |
| individual--dino_global--local_heads--evidence--budget-20 | 328.00 | 11.67 | 0.33 | 5.67 | 0.33 | 5.33 | 1.33 | 5.33 | 6.00 | 12.00 | 70.33 |
| individual--dino_global--local_heads--evidence--budget-40 | 328.33 | 8.33 | 0.33 | 5.67 | 0.33 | 5.00 | 2.00 | 6.67 | 8.67 | 20.67 | 101.00 |
| individual--dino_boost--legacy--chronological--budget-05 | 342.67 | 14.67 | 1.33 | 8.67 | 1.33 | 7.33 | 0.00 | 0.67 | 1.33 | 1.67 | 8.33 |
| individual--dino_boost--legacy--chronological--budget-10 | 342.00 | 14.00 | 1.33 | 7.67 | 1.33 | 6.33 | 0.00 | 0.67 | 2.33 | 5.00 | 19.00 |
| individual--dino_boost--legacy--chronological--budget-20 | 337.67 | 12.33 | 1.33 | 6.33 | 1.33 | 5.00 | 0.00 | 3.00 | 4.67 | 12.00 | 38.00 |
| individual--dino_boost--legacy--chronological--budget-40 | 335.67 | 8.33 | 1.33 | 3.67 | 1.33 | 2.67 | 0.33 | 4.67 | 11.67 | 26.67 | 70.67 |
| individual--dino_boost--legacy--evidence--budget-05 | 345.67 | 14.33 | 1.33 | 9.00 | 1.33 | 8.00 | 0.67 | 2.33 | 5.67 | 3.67 | 21.67 |
| individual--dino_boost--legacy--evidence--budget-10 | 346.00 | 13.33 | 1.33 | 8.33 | 1.33 | 7.33 | 1.00 | 3.67 | 7.33 | 6.67 | 46.33 |
| individual--dino_boost--legacy--evidence--budget-20 | 343.67 | 11.33 | 1.00 | 7.00 | 1.00 | 6.33 | 1.67 | 4.00 | 13.00 | 9.67 | 94.67 |
| individual--dino_boost--legacy--evidence--budget-40 | 334.33 | 8.67 | 1.00 | 3.67 | 1.00 | 3.33 | 1.00 | 9.67 | 18.67 | 22.00 | 172.67 |
| individual--dino_boost--local_events--chronological--budget-05 | 338.00 | 13.67 | 1.33 | 8.33 | 1.33 | 7.00 | 0.33 | 3.00 | 2.00 | 7.00 | 11.33 |
| individual--dino_boost--local_events--chronological--budget-10 | 334.00 | 12.00 | 1.33 | 8.33 | 1.33 | 7.00 | 1.00 | 5.00 | 4.00 | 10.33 | 19.67 |
| individual--dino_boost--local_events--chronological--budget-20 | 329.00 | 8.00 | 1.33 | 7.67 | 1.33 | 7.00 | 1.33 | 7.00 | 6.00 | 17.00 | 27.33 |
| individual--dino_boost--local_events--chronological--budget-40 | 328.00 | 6.00 | 1.33 | 7.67 | 1.33 | 7.00 | 2.33 | 8.00 | 7.33 | 19.33 | 29.33 |
| individual--dino_boost--local_events--evidence--budget-05 | 335.00 | 15.00 | 1.33 | 8.00 | 1.33 | 6.67 | 0.33 | 2.33 | 0.33 | 3.33 | 5.00 |
| individual--dino_boost--local_events--evidence--budget-10 | 328.67 | 13.33 | 1.33 | 7.00 | 1.33 | 6.33 | 0.67 | 4.67 | 2.00 | 7.00 | 15.33 |
| individual--dino_boost--local_events--evidence--budget-20 | 324.67 | 9.33 | 1.33 | 6.67 | 1.33 | 6.00 | 1.00 | 7.33 | 3.67 | 13.67 | 22.33 |
| individual--dino_boost--local_events--evidence--budget-40 | 328.00 | 6.00 | 1.33 | 7.67 | 1.33 | 7.00 | 2.33 | 8.00 | 7.33 | 19.33 | 29.33 |
| individual--dino_boost--local_heads--chronological--budget-05 | 343.00 | 14.33 | 1.33 | 9.00 | 1.33 | 7.33 | 0.00 | 1.67 | 1.67 | 5.00 | 17.33 |
| individual--dino_boost--local_heads--chronological--budget-10 | 338.67 | 13.67 | 1.33 | 7.67 | 1.33 | 6.00 | 1.00 | 4.67 | 4.33 | 10.00 | 35.00 |
| individual--dino_boost--local_heads--chronological--budget-20 | 335.00 | 10.67 | 1.00 | 6.33 | 1.00 | 5.00 | 2.33 | 7.67 | 10.33 | 20.67 | 61.67 |
| individual--dino_boost--local_heads--chronological--budget-40 | 326.67 | 7.67 | 1.00 | 4.67 | 1.00 | 3.67 | 3.33 | 12.00 | 13.00 | 32.00 | 106.67 |
| individual--dino_boost--local_heads--evidence--budget-05 | 333.33 | 14.67 | 1.33 | 8.00 | 1.33 | 7.00 | 0.33 | 1.67 | 1.00 | 2.00 | 11.00 |
| individual--dino_boost--local_heads--evidence--budget-10 | 327.67 | 14.33 | 1.33 | 7.33 | 1.33 | 7.00 | 1.00 | 2.67 | 1.67 | 4.67 | 29.00 |
| individual--dino_boost--local_heads--evidence--budget-20 | 329.67 | 11.67 | 1.00 | 7.33 | 1.00 | 7.00 | 2.33 | 6.33 | 8.33 | 11.67 | 69.33 |
| individual--dino_boost--local_heads--evidence--budget-40 | 330.33 | 6.67 | 1.00 | 5.00 | 1.00 | 4.67 | 3.67 | 12.00 | 15.00 | 28.33 | 114.33 |

## Changes against the same automatic baseline

Quality deltas are percentage points. Miss/merge/split deltas are counts, where a reduction is better.

| Arm | Export F1 delta pp | Event F1 delta pp | Event recall delta pp | Observed start F1 delta pp | Observed start recall delta pp | Complete misses delta | Material merges delta | Material splits delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--compact_boost--legacy--chronological--budget-05 | 1.28 | 2.55 | 1.86 | 2.07 | 1.45 | 0.00 | 0.00 | 0.00 |
| production--compact_boost--legacy--chronological--budget-10 | 2.19 | 4.95 | 4.35 | 4.64 | 4.04 | 0.00 | -2.00 | -0.67 |
| production--compact_boost--legacy--chronological--budget-20 | 4.54 | 10.41 | 9.42 | 9.22 | 8.18 | 0.00 | -3.33 | -0.67 |
| production--compact_boost--legacy--chronological--budget-40 | 8.99 | 17.83 | 15.63 | 17.35 | 15.11 | 0.00 | -6.33 | -1.00 |
| production--compact_boost--legacy--evidence--budget-05 | 0.88 | 1.30 | 0.93 | 2.48 | 2.28 | 0.00 | -1.00 | -1.33 |
| production--compact_boost--legacy--evidence--budget-10 | 1.88 | 3.07 | 2.69 | 4.15 | 3.93 | 0.00 | -1.67 | -1.33 |
| production--compact_boost--legacy--evidence--budget-20 | 4.33 | 7.70 | 7.25 | 8.60 | 8.28 | 0.00 | -3.33 | -1.33 |
| production--compact_boost--legacy--evidence--budget-40 | 8.65 | 16.64 | 14.80 | 16.56 | 14.70 | 0.00 | -5.33 | -1.00 |
| production--compact_boost--local_events--chronological--budget-05 | 1.08 | 1.83 | 1.24 | 2.27 | 1.76 | 0.00 | 0.00 | 0.00 |
| production--compact_boost--local_events--chronological--budget-10 | 1.89 | 4.14 | 3.83 | 4.94 | 4.66 | 0.00 | -2.00 | -1.00 |
| production--compact_boost--local_events--chronological--budget-20 | 3.49 | 7.48 | 7.25 | 8.49 | 7.97 | -0.67 | -3.33 | -2.00 |
| production--compact_boost--local_events--chronological--budget-40 | 7.55 | 16.50 | 15.22 | 16.47 | 14.39 | -0.67 | -6.00 | -3.00 |
| production--compact_boost--local_events--evidence--budget-05 | 0.66 | 2.34 | 2.17 | 2.15 | 1.97 | 0.00 | -1.00 | -1.67 |
| production--compact_boost--local_events--evidence--budget-10 | 1.62 | 5.66 | 5.28 | 3.81 | 3.21 | -0.33 | -1.33 | -2.67 |
| production--compact_boost--local_events--evidence--budget-20 | 4.15 | 10.79 | 10.97 | 8.99 | 8.18 | -0.67 | -3.33 | -2.67 |
| production--compact_boost--local_events--evidence--budget-40 | 9.31 | 20.94 | 20.08 | 15.89 | 13.66 | -2.33 | -5.33 | -2.67 |
| production--compact_boost--local_heads--chronological--budget-05 | 1.07 | 1.96 | 1.45 | 2.33 | 1.86 | 0.00 | -0.67 | 0.00 |
| production--compact_boost--local_heads--chronological--budget-10 | 1.81 | 4.08 | 3.83 | 4.94 | 4.66 | 0.00 | -2.00 | -1.00 |
| production--compact_boost--local_heads--chronological--budget-20 | 3.49 | 7.32 | 7.04 | 8.58 | 8.18 | -0.67 | -3.33 | -2.00 |
| production--compact_boost--local_heads--chronological--budget-40 | 7.62 | 16.44 | 15.32 | 16.72 | 14.91 | -0.67 | -6.33 | -3.00 |
| production--compact_boost--local_heads--evidence--budget-05 | 0.77 | 2.75 | 2.90 | 3.19 | 3.21 | 0.00 | -1.00 | -1.33 |
| production--compact_boost--local_heads--evidence--budget-10 | 1.34 | 4.94 | 5.49 | 6.15 | 6.31 | 0.00 | -1.67 | -2.00 |
| production--compact_boost--local_heads--evidence--budget-20 | 2.78 | 9.09 | 10.66 | 12.63 | 13.35 | -0.33 | -2.67 | -2.33 |
| production--compact_boost--local_heads--evidence--budget-40 | 5.63 | 14.67 | 17.39 | 19.72 | 20.70 | -1.00 | -4.00 | -2.33 |
| production--dino_global--legacy--chronological--budget-05 | 1.36 | 2.88 | 2.17 | 2.54 | 1.86 | -0.67 | 0.00 | 0.00 |
| production--dino_global--legacy--chronological--budget-10 | 2.33 | 5.58 | 4.87 | 4.74 | 4.04 | -0.67 | -2.00 | -1.00 |
| production--dino_global--legacy--chronological--budget-20 | 5.00 | 12.25 | 11.18 | 10.63 | 9.52 | -0.67 | -3.67 | -1.00 |
| production--dino_global--legacy--chronological--budget-40 | 9.44 | 19.18 | 16.67 | 18.70 | 16.15 | -1.00 | -6.67 | -2.00 |
| production--dino_global--legacy--evidence--budget-05 | 1.18 | 2.18 | 1.86 | 2.29 | 2.07 | 0.00 | -1.67 | -1.00 |
| production--dino_global--legacy--evidence--budget-10 | 2.25 | 4.32 | 3.83 | 4.66 | 4.24 | 0.00 | -2.67 | -1.67 |
| production--dino_global--legacy--evidence--budget-20 | 4.71 | 9.27 | 8.70 | 9.99 | 9.42 | 0.00 | -5.00 | -2.00 |
| production--dino_global--legacy--evidence--budget-40 | 9.31 | 18.76 | 16.77 | 17.87 | 15.73 | -1.00 | -6.67 | -2.00 |
| production--dino_global--local_events--chronological--budget-05 | 1.09 | 1.83 | 1.24 | 2.36 | 1.86 | 0.00 | 0.00 | 0.00 |
| production--dino_global--local_events--chronological--budget-10 | 1.87 | 3.89 | 3.52 | 4.94 | 4.66 | 0.00 | -2.00 | -1.00 |
| production--dino_global--local_events--chronological--budget-20 | 3.62 | 8.01 | 7.76 | 8.83 | 8.49 | -1.00 | -4.00 | -2.00 |
| production--dino_global--local_events--chronological--budget-40 | 8.00 | 16.91 | 15.42 | 16.93 | 14.91 | -1.00 | -6.33 | -3.00 |
| production--dino_global--local_events--evidence--budget-05 | 0.90 | 3.01 | 3.21 | 1.98 | 2.07 | -0.33 | -3.00 | -1.67 |
| production--dino_global--local_events--evidence--budget-10 | 1.73 | 6.17 | 6.31 | 3.98 | 3.83 | -1.00 | -3.67 | -3.00 |
| production--dino_global--local_events--evidence--budget-20 | 4.07 | 10.90 | 10.77 | 6.43 | 5.69 | -1.67 | -4.33 | -4.00 |
| production--dino_global--local_events--evidence--budget-40 | 9.89 | 22.85 | 21.43 | 15.98 | 13.46 | -2.00 | -6.00 | -4.00 |
| production--dino_global--local_heads--chronological--budget-05 | 1.08 | 1.86 | 1.35 | 2.43 | 1.97 | 0.00 | -0.33 | 0.00 |
| production--dino_global--local_heads--chronological--budget-10 | 1.83 | 3.95 | 3.62 | 4.85 | 4.55 | 0.00 | -2.00 | -1.00 |
| production--dino_global--local_heads--chronological--budget-20 | 3.64 | 7.98 | 7.76 | 8.86 | 8.59 | -1.00 | -4.00 | -2.00 |
| production--dino_global--local_heads--chronological--budget-40 | 7.94 | 16.84 | 15.53 | 17.00 | 15.11 | -1.00 | -6.67 | -3.00 |
| production--dino_global--local_heads--evidence--budget-05 | 0.80 | 2.06 | 2.69 | 2.72 | 3.11 | 0.00 | -2.67 | -1.00 |
| production--dino_global--local_heads--evidence--budget-10 | 1.66 | 5.67 | 6.52 | 5.22 | 5.59 | -1.00 | -4.00 | -2.33 |
| production--dino_global--local_heads--evidence--budget-20 | 2.95 | 8.95 | 10.66 | 10.23 | 11.08 | -1.33 | -4.33 | -3.67 |
| production--dino_global--local_heads--evidence--budget-40 | 5.52 | 14.40 | 17.81 | 17.04 | 18.43 | -1.67 | -5.33 | -4.00 |
| production--dino_boost--legacy--chronological--budget-05 | 1.24 | 2.68 | 2.07 | 2.14 | 1.55 | -0.33 | 0.00 | 0.00 |
| production--dino_boost--legacy--chronological--budget-10 | 2.25 | 5.41 | 4.76 | 4.71 | 4.04 | -0.67 | -1.67 | -1.00 |
| production--dino_boost--legacy--chronological--budget-20 | 4.94 | 11.54 | 10.35 | 11.34 | 10.14 | -1.00 | -3.00 | -1.00 |
| production--dino_boost--legacy--chronological--budget-40 | 9.56 | 19.50 | 16.77 | 19.17 | 16.36 | -1.67 | -6.00 | -2.00 |
| production--dino_boost--legacy--evidence--budget-05 | 1.29 | 2.45 | 1.97 | 2.79 | 2.38 | 0.00 | -1.33 | -1.00 |
| production--dino_boost--legacy--evidence--budget-10 | 2.50 | 4.50 | 3.73 | 4.88 | 4.14 | 0.00 | -2.33 | -1.67 |
| production--dino_boost--legacy--evidence--budget-20 | 5.25 | 9.75 | 8.49 | 10.02 | 8.70 | -0.67 | -3.33 | -1.67 |
| production--dino_boost--legacy--evidence--budget-40 | 9.44 | 18.55 | 16.46 | 18.31 | 16.15 | -1.67 | -6.00 | -2.00 |
| production--dino_boost--local_events--chronological--budget-05 | 1.04 | 1.76 | 1.24 | 2.26 | 1.86 | 0.00 | 0.00 | 0.00 |
| production--dino_boost--local_events--chronological--budget-10 | 1.79 | 3.62 | 3.31 | 5.20 | 4.97 | -0.67 | -1.67 | -1.00 |
| production--dino_boost--local_events--chronological--budget-20 | 3.53 | 7.47 | 7.04 | 8.61 | 8.18 | -1.00 | -3.33 | -2.00 |
| production--dino_boost--local_events--chronological--budget-40 | 7.85 | 16.70 | 15.42 | 17.04 | 15.32 | -1.00 | -6.33 | -3.00 |
| production--dino_boost--local_events--evidence--budget-05 | 0.83 | 2.66 | 2.59 | 2.09 | 1.97 | 0.00 | -1.67 | -2.00 |
| production--dino_boost--local_events--evidence--budget-10 | 1.84 | 5.38 | 5.28 | 3.79 | 3.52 | -1.00 | -3.33 | -3.00 |
| production--dino_boost--local_events--evidence--budget-20 | 3.88 | 10.02 | 10.04 | 5.85 | 5.38 | -1.00 | -4.33 | -4.00 |
| production--dino_boost--local_events--evidence--budget-40 | 9.52 | 22.27 | 21.43 | 16.04 | 14.08 | -2.33 | -6.00 | -4.00 |
| production--dino_boost--local_heads--chronological--budget-05 | 1.06 | 1.86 | 1.35 | 2.29 | 1.86 | 0.00 | -0.33 | 0.00 |
| production--dino_boost--local_heads--chronological--budget-10 | 1.78 | 3.71 | 3.52 | 4.81 | 4.55 | -0.67 | -1.67 | -1.00 |
| production--dino_boost--local_heads--chronological--budget-20 | 3.53 | 7.32 | 7.04 | 8.67 | 8.28 | -1.00 | -3.33 | -2.00 |
| production--dino_boost--local_heads--chronological--budget-40 | 7.83 | 16.58 | 15.42 | 16.79 | 15.11 | -1.00 | -6.33 | -3.00 |
| production--dino_boost--local_heads--evidence--budget-05 | 0.82 | 2.36 | 2.59 | 2.59 | 2.80 | -0.33 | -1.67 | -2.00 |
| production--dino_boost--local_heads--evidence--budget-10 | 1.76 | 5.70 | 6.11 | 5.03 | 5.28 | -0.67 | -3.33 | -2.67 |
| production--dino_boost--local_heads--evidence--budget-20 | 3.23 | 9.21 | 10.25 | 8.64 | 9.11 | -1.00 | -4.33 | -4.00 |
| production--dino_boost--local_heads--evidence--budget-40 | 5.68 | 14.59 | 17.60 | 17.57 | 18.74 | -2.00 | -5.00 | -4.00 |
| individual--compact_boost--legacy--chronological--budget-05 | 0.75 | 2.14 | 2.07 | 1.77 | 1.66 | -2.00 | 0.00 | -0.67 |
| individual--compact_boost--legacy--chronological--budget-10 | 1.20 | 4.29 | 4.45 | 3.24 | 3.21 | -5.00 | 0.00 | -1.67 |
| individual--compact_boost--legacy--chronological--budget-20 | 2.17 | 7.75 | 7.97 | 6.30 | 5.90 | -10.00 | 0.00 | -3.33 |
| individual--compact_boost--legacy--chronological--budget-40 | 4.44 | 15.09 | 15.22 | 12.97 | 11.70 | -18.00 | 0.00 | -6.33 |
| individual--compact_boost--legacy--evidence--budget-05 | 0.78 | 1.50 | 1.76 | 1.13 | 0.93 | -2.67 | 0.00 | 1.00 |
| individual--compact_boost--legacy--evidence--budget-10 | 1.39 | 4.95 | 5.38 | 4.19 | 3.83 | -5.33 | 0.00 | 0.33 |
| individual--compact_boost--legacy--evidence--budget-20 | 2.66 | 9.44 | 9.83 | 8.30 | 7.04 | -9.33 | 0.00 | -1.67 |
| individual--compact_boost--legacy--evidence--budget-40 | 4.85 | 17.85 | 17.49 | 16.93 | 14.18 | -17.67 | 0.00 | -9.00 |
| individual--compact_boost--local_events--chronological--budget-05 | 0.90 | 4.08 | 3.93 | 2.49 | 2.17 | -4.00 | 0.00 | -2.00 |
| individual--compact_boost--local_events--chronological--budget-10 | 1.80 | 6.80 | 6.63 | 4.89 | 4.45 | -7.67 | 0.00 | -1.67 |
| individual--compact_boost--local_events--chronological--budget-20 | 3.37 | 11.74 | 11.70 | 9.57 | 8.90 | -15.00 | 0.00 | -1.67 |
| individual--compact_boost--local_events--chronological--budget-40 | 5.35 | 19.32 | 18.63 | 15.04 | 13.46 | -22.00 | 0.00 | -4.33 |
| individual--compact_boost--local_events--evidence--budget-05 | 0.88 | 2.49 | 1.86 | 2.41 | 1.66 | -0.67 | 0.00 | -1.33 |
| individual--compact_boost--local_events--evidence--budget-10 | 1.78 | 6.44 | 4.76 | 4.84 | 2.80 | -1.00 | 0.00 | -3.00 |
| individual--compact_boost--local_events--evidence--budget-20 | 3.43 | 12.29 | 10.46 | 9.56 | 7.14 | -7.00 | 0.00 | -4.00 |
| individual--compact_boost--local_events--evidence--budget-40 | 5.44 | 19.34 | 18.43 | 15.07 | 13.35 | -21.67 | 0.00 | -4.33 |
| individual--compact_boost--local_heads--chronological--budget-05 | 0.83 | 3.42 | 3.31 | 2.37 | 2.17 | -2.67 | 0.00 | -1.67 |
| individual--compact_boost--local_heads--chronological--budget-10 | 1.58 | 5.81 | 5.90 | 4.85 | 4.66 | -6.33 | 0.00 | -2.00 |
| individual--compact_boost--local_heads--chronological--budget-20 | 3.09 | 10.74 | 10.77 | 9.40 | 8.90 | -11.67 | -0.33 | -3.33 |
| individual--compact_boost--local_heads--chronological--budget-40 | 5.41 | 18.95 | 19.05 | 16.44 | 15.22 | -20.67 | -0.33 | -6.00 |
| individual--compact_boost--local_heads--evidence--budget-05 | 0.86 | 3.12 | 2.38 | 2.88 | 1.97 | -0.33 | 0.00 | -0.67 |
| individual--compact_boost--local_heads--evidence--budget-10 | 1.81 | 6.82 | 5.49 | 5.35 | 3.62 | -3.00 | 0.00 | -3.00 |
| individual--compact_boost--local_heads--evidence--budget-20 | 3.47 | 13.02 | 11.39 | 10.61 | 8.18 | -8.67 | 0.00 | -4.33 |
| individual--compact_boost--local_heads--evidence--budget-40 | 5.51 | 19.44 | 19.25 | 16.91 | 15.11 | -20.67 | -0.33 | -5.67 |
| individual--dino_global--legacy--chronological--budget-05 | 0.35 | 2.00 | 1.76 | 1.34 | 1.04 | -0.67 | 0.00 | -0.67 |
| individual--dino_global--legacy--chronological--budget-10 | 0.68 | 2.76 | 2.59 | 2.18 | 1.76 | -2.00 | 0.00 | -1.33 |
| individual--dino_global--legacy--chronological--budget-20 | 1.57 | 6.07 | 5.49 | 4.98 | 3.73 | -4.67 | 0.00 | -3.00 |
| individual--dino_global--legacy--chronological--budget-40 | 3.34 | 11.89 | 11.08 | 9.90 | 7.56 | -9.00 | 0.00 | -4.33 |
| individual--dino_global--legacy--evidence--budget-05 | 0.37 | 1.75 | 2.07 | 0.81 | 0.62 | -1.33 | 0.00 | -0.33 |
| individual--dino_global--legacy--evidence--budget-10 | 0.76 | 3.79 | 4.24 | 2.19 | 1.76 | -2.33 | 0.00 | -0.33 |
| individual--dino_global--legacy--evidence--budget-20 | 1.54 | 7.23 | 7.66 | 5.73 | 4.66 | -5.67 | 0.00 | -2.33 |
| individual--dino_global--legacy--evidence--budget-40 | 3.05 | 13.48 | 12.73 | 11.69 | 8.49 | -7.67 | 0.00 | -5.33 |
| individual--dino_global--local_events--chronological--budget-05 | 0.80 | 3.95 | 3.42 | 2.48 | 1.76 | -3.33 | 0.00 | -2.00 |
| individual--dino_global--local_events--chronological--budget-10 | 1.51 | 7.69 | 6.52 | 4.76 | 3.11 | -5.33 | 0.00 | -2.67 |
| individual--dino_global--local_events--chronological--budget-20 | 2.44 | 11.67 | 10.04 | 7.40 | 5.07 | -9.00 | 0.00 | -3.00 |
| individual--dino_global--local_events--chronological--budget-40 | 2.80 | 13.77 | 12.53 | 8.57 | 6.52 | -11.67 | 0.00 | -2.67 |
| individual--dino_global--local_events--evidence--budget-05 | 0.83 | 3.57 | 2.80 | 2.17 | 1.24 | -1.33 | 0.00 | -1.00 |
| individual--dino_global--local_events--evidence--budget-10 | 1.64 | 8.21 | 6.52 | 4.63 | 2.48 | -4.33 | 0.00 | -3.33 |
| individual--dino_global--local_events--evidence--budget-20 | 2.42 | 12.36 | 10.35 | 7.20 | 4.55 | -6.00 | 0.00 | -3.00 |
| individual--dino_global--local_events--evidence--budget-40 | 2.80 | 13.77 | 12.53 | 8.57 | 6.52 | -11.67 | 0.00 | -2.67 |
| individual--dino_global--local_heads--chronological--budget-05 | 0.63 | 2.52 | 2.38 | 1.61 | 1.35 | -1.33 | 0.00 | -0.33 |
| individual--dino_global--local_heads--chronological--budget-10 | 1.34 | 4.79 | 4.35 | 3.88 | 3.00 | -3.67 | 0.00 | -1.00 |
| individual--dino_global--local_heads--chronological--budget-20 | 2.51 | 9.74 | 9.11 | 7.97 | 6.42 | -6.33 | -0.33 | -2.00 |
| individual--dino_global--local_heads--chronological--budget-40 | 3.73 | 14.36 | 13.25 | 12.07 | 9.63 | -9.33 | -0.33 | -3.67 |
| individual--dino_global--local_heads--evidence--budget-05 | 0.87 | 3.91 | 2.80 | 2.60 | 1.24 | -1.00 | 0.00 | -1.33 |
| individual--dino_global--local_heads--evidence--budget-10 | 1.64 | 7.63 | 6.21 | 4.83 | 3.00 | -2.67 | 0.00 | -2.00 |
| individual--dino_global--local_heads--evidence--budget-20 | 2.43 | 11.57 | 10.66 | 8.35 | 6.52 | -6.33 | -0.33 | -2.33 |
| individual--dino_global--local_heads--evidence--budget-40 | 3.67 | 14.70 | 13.87 | 12.20 | 10.04 | -9.67 | -0.33 | -2.67 |
| individual--dino_boost--legacy--chronological--budget-05 | 0.28 | 1.47 | 1.24 | 1.15 | 0.72 | -0.33 | 0.00 | -1.00 |
| individual--dino_boost--legacy--chronological--budget-10 | 0.56 | 2.75 | 2.48 | 1.86 | 1.24 | -1.00 | 0.00 | -2.00 |
| individual--dino_boost--legacy--chronological--budget-20 | 1.40 | 5.97 | 5.28 | 4.64 | 3.21 | -2.67 | 0.00 | -3.33 |
| individual--dino_boost--legacy--chronological--budget-40 | 3.09 | 12.50 | 11.70 | 8.93 | 6.31 | -6.67 | 0.00 | -5.67 |
| individual--dino_boost--legacy--evidence--budget-05 | 0.25 | 1.02 | 1.14 | 1.42 | 0.83 | -0.67 | 0.00 | -0.33 |
| individual--dino_boost--legacy--evidence--budget-10 | 0.54 | 2.79 | 3.00 | 2.80 | 2.07 | -1.67 | 0.00 | -1.00 |
| individual--dino_boost--legacy--evidence--budget-20 | 1.34 | 5.55 | 5.59 | 6.07 | 4.35 | -3.67 | -0.33 | -2.00 |
| individual--dino_boost--legacy--evidence--budget-40 | 2.91 | 13.51 | 12.53 | 12.83 | 9.01 | -6.33 | -0.33 | -5.00 |
| individual--dino_boost--local_events--chronological--budget-05 | 0.72 | 4.02 | 3.31 | 2.22 | 1.14 | -1.33 | 0.00 | -1.33 |
| individual--dino_boost--local_events--chronological--budget-10 | 1.39 | 7.44 | 6.31 | 4.11 | 2.28 | -3.00 | 0.00 | -1.33 |
| individual--dino_boost--local_events--chronological--budget-20 | 2.35 | 11.44 | 9.73 | 7.31 | 4.55 | -7.00 | 0.00 | -1.33 |
| individual--dino_boost--local_events--chronological--budget-40 | 2.84 | 14.22 | 12.42 | 8.77 | 5.69 | -9.00 | 0.00 | -1.33 |
| individual--dino_boost--local_events--evidence--budget-05 | 0.84 | 3.97 | 2.90 | 2.29 | 1.04 | 0.00 | 0.00 | -1.67 |
| individual--dino_boost--local_events--evidence--budget-10 | 1.60 | 8.21 | 6.42 | 5.17 | 2.90 | -1.67 | 0.00 | -2.00 |
| individual--dino_boost--local_events--evidence--budget-20 | 2.45 | 12.74 | 10.46 | 7.90 | 4.87 | -5.67 | 0.00 | -2.33 |
| individual--dino_boost--local_events--evidence--budget-40 | 2.84 | 14.22 | 12.42 | 8.77 | 5.69 | -9.00 | 0.00 | -1.33 |
| individual--dino_boost--local_heads--chronological--budget-05 | 0.42 | 2.02 | 1.86 | 1.25 | 0.83 | -0.67 | 0.00 | -1.00 |
| individual--dino_boost--local_heads--chronological--budget-10 | 1.01 | 4.24 | 3.62 | 3.05 | 1.76 | -1.33 | 0.00 | -2.33 |
| individual--dino_boost--local_heads--chronological--budget-20 | 2.22 | 9.45 | 8.49 | 6.26 | 3.73 | -4.33 | -0.33 | -3.33 |
| individual--dino_boost--local_heads--chronological--budget-40 | 3.98 | 16.48 | 14.49 | 11.65 | 7.56 | -7.33 | -0.33 | -4.67 |
| individual--dino_boost--local_heads--evidence--budget-05 | 0.78 | 3.55 | 2.28 | 2.28 | 0.72 | -0.33 | 0.00 | -1.33 |
| individual--dino_boost--local_heads--evidence--budget-10 | 1.45 | 6.80 | 4.87 | 4.64 | 2.28 | -0.67 | 0.00 | -1.33 |
| individual--dino_boost--local_heads--evidence--budget-20 | 2.40 | 11.56 | 9.94 | 7.44 | 4.45 | -3.33 | -0.33 | -1.33 |
| individual--dino_boost--local_heads--evidence--budget-40 | 3.89 | 17.40 | 15.94 | 11.92 | 8.07 | -8.33 | -0.33 | -3.67 |

## Workload and unused budget

| Arm | Cap % | Actual footage % | Playback min | Unused cap min | Edit min | Selected proposals | Available proposals | Edit regions | Playback clips | True rallies in edit regions | True rallies visible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--compact_boost--legacy--chronological--budget-05 | 5.00 | 4.58 | 6.31 | 0.58 | 4.79 | 25.00 | 198.67 | 25.00 | 24.00 | 17.00 | 18.00 |
| production--compact_boost--legacy--chronological--budget-10 | 10.00 | 9.66 | 13.31 | 0.47 | 10.28 | 47.33 | 198.67 | 46.67 | 42.67 | 39.33 | 40.33 |
| production--compact_boost--legacy--chronological--budget-20 | 20.00 | 18.79 | 25.90 | 1.67 | 20.35 | 90.00 | 198.67 | 85.67 | 75.67 | 75.33 | 77.00 |
| production--compact_boost--legacy--chronological--budget-40 | 40.00 | 32.58 | 44.91 | 10.22 | 35.95 | 151.67 | 198.67 | 141.67 | 105.33 | 125.00 | 126.33 |
| production--compact_boost--legacy--evidence--budget-05 | 5.00 | 4.76 | 6.57 | 0.32 | 5.50 | 18.00 | 198.67 | 18.00 | 18.00 | 15.67 | 15.67 |
| production--compact_boost--legacy--evidence--budget-10 | 10.00 | 9.57 | 13.19 | 0.59 | 10.98 | 37.00 | 198.67 | 35.67 | 34.67 | 33.00 | 33.00 |
| production--compact_boost--legacy--evidence--budget-20 | 20.00 | 18.89 | 26.03 | 1.53 | 21.34 | 76.33 | 198.67 | 73.33 | 68.00 | 69.00 | 69.00 |
| production--compact_boost--legacy--evidence--budget-40 | 40.00 | 32.61 | 44.95 | 10.18 | 36.02 | 145.00 | 198.67 | 137.33 | 112.33 | 121.00 | 123.00 |
| production--compact_boost--local_events--chronological--budget-05 | 5.00 | 4.69 | 6.46 | 0.43 | 4.50 | 45.00 | 756.00 | 34.00 | 22.00 | 18.67 | 19.33 |
| production--compact_boost--local_events--chronological--budget-10 | 10.00 | 9.58 | 13.20 | 0.58 | 9.18 | 92.00 | 756.00 | 70.33 | 38.33 | 42.33 | 42.33 |
| production--compact_boost--local_events--chronological--budget-20 | 20.00 | 19.62 | 27.05 | 0.52 | 18.58 | 193.67 | 756.00 | 140.33 | 81.33 | 87.33 | 87.33 |
| production--compact_boost--local_events--chronological--budget-40 | 40.00 | 39.58 | 54.56 | 0.57 | 37.30 | 398.33 | 756.00 | 289.67 | 138.33 | 169.67 | 170.00 |
| production--compact_boost--local_events--evidence--budget-05 | 5.00 | 4.80 | 6.62 | 0.28 | 4.86 | 55.67 | 756.00 | 27.67 | 26.67 | 24.67 | 25.33 |
| production--compact_boost--local_events--evidence--budget-10 | 10.00 | 9.86 | 13.59 | 0.19 | 9.73 | 115.67 | 756.00 | 60.67 | 58.00 | 53.00 | 56.00 |
| production--compact_boost--local_events--evidence--budget-20 | 20.00 | 19.91 | 27.45 | 0.12 | 19.45 | 205.00 | 756.00 | 125.00 | 117.00 | 106.67 | 112.67 |
| production--compact_boost--local_events--evidence--budget-40 | 40.00 | 39.89 | 54.98 | 0.15 | 38.36 | 385.33 | 756.00 | 266.33 | 204.67 | 198.00 | 208.00 |
| production--compact_boost--local_heads--chronological--budget-05 | 5.00 | 4.72 | 6.51 | 0.38 | 4.57 | 76.33 | 1280.33 | 32.67 | 21.67 | 19.00 | 19.33 |
| production--compact_boost--local_heads--chronological--budget-10 | 10.00 | 9.60 | 13.24 | 0.54 | 9.31 | 157.00 | 1280.33 | 68.00 | 37.33 | 42.33 | 42.33 |
| production--compact_boost--local_heads--chronological--budget-20 | 20.00 | 19.66 | 27.10 | 0.46 | 18.94 | 327.00 | 1280.33 | 136.67 | 79.00 | 86.67 | 86.67 |
| production--compact_boost--local_heads--chronological--budget-40 | 40.00 | 39.66 | 54.67 | 0.47 | 38.23 | 667.67 | 1280.33 | 279.67 | 136.33 | 169.33 | 169.33 |
| production--compact_boost--local_heads--evidence--budget-05 | 5.00 | 4.89 | 6.73 | 0.16 | 5.04 | 84.67 | 1280.33 | 27.00 | 26.33 | 25.33 | 25.67 |
| production--compact_boost--local_heads--evidence--budget-10 | 10.00 | 9.94 | 13.70 | 0.09 | 9.72 | 175.00 | 1280.33 | 63.00 | 59.33 | 58.33 | 60.33 |
| production--compact_boost--local_heads--evidence--budget-20 | 20.00 | 19.94 | 27.48 | 0.08 | 18.48 | 311.33 | 1280.33 | 139.33 | 127.00 | 127.00 | 130.67 |
| production--compact_boost--local_heads--evidence--budget-40 | 40.00 | 39.94 | 55.05 | 0.08 | 36.19 | 606.67 | 1280.33 | 295.67 | 235.00 | 243.33 | 245.00 |
| production--dino_global--legacy--chronological--budget-05 | 5.00 | 4.51 | 6.22 | 0.67 | 4.73 | 26.00 | 188.67 | 24.33 | 23.33 | 16.00 | 17.00 |
| production--dino_global--legacy--chronological--budget-10 | 10.00 | 9.55 | 13.16 | 0.62 | 10.25 | 48.33 | 188.67 | 44.67 | 41.00 | 36.00 | 37.33 |
| production--dino_global--legacy--chronological--budget-20 | 20.00 | 18.92 | 26.08 | 1.49 | 20.84 | 91.00 | 188.67 | 82.67 | 70.33 | 72.33 | 74.67 |
| production--dino_global--legacy--chronological--budget-40 | 40.00 | 32.24 | 44.44 | 10.70 | 35.80 | 149.33 | 188.67 | 139.00 | 104.00 | 117.00 | 119.67 |
| production--dino_global--legacy--evidence--budget-05 | 5.00 | 4.72 | 6.51 | 0.39 | 5.43 | 18.67 | 188.67 | 18.67 | 18.33 | 14.67 | 15.67 |
| production--dino_global--legacy--evidence--budget-10 | 10.00 | 9.70 | 13.37 | 0.41 | 11.14 | 37.67 | 188.67 | 36.33 | 35.33 | 32.00 | 34.00 |
| production--dino_global--legacy--evidence--budget-20 | 20.00 | 19.04 | 26.25 | 1.32 | 21.58 | 77.00 | 188.67 | 73.67 | 66.33 | 69.00 | 71.33 |
| production--dino_global--legacy--evidence--budget-40 | 40.00 | 32.40 | 44.66 | 10.47 | 36.09 | 143.67 | 188.67 | 134.67 | 110.33 | 115.33 | 119.33 |
| production--dino_global--local_events--chronological--budget-05 | 5.00 | 4.67 | 6.44 | 0.45 | 4.52 | 47.00 | 757.33 | 34.67 | 22.00 | 19.00 | 19.67 |
| production--dino_global--local_events--chronological--budget-10 | 10.00 | 9.51 | 13.11 | 0.68 | 9.04 | 95.00 | 757.33 | 71.67 | 38.00 | 42.33 | 42.33 |
| production--dino_global--local_events--chronological--budget-20 | 20.00 | 19.63 | 27.05 | 0.51 | 18.52 | 201.00 | 757.33 | 143.33 | 80.67 | 88.67 | 88.67 |
| production--dino_global--local_events--chronological--budget-40 | 40.00 | 39.67 | 54.68 | 0.46 | 37.19 | 405.00 | 757.33 | 294.67 | 142.67 | 172.00 | 173.00 |
| production--dino_global--local_events--evidence--budget-05 | 5.00 | 4.82 | 6.65 | 0.24 | 4.98 | 51.33 | 757.33 | 26.67 | 26.33 | 25.33 | 26.33 |
| production--dino_global--local_events--evidence--budget-10 | 10.00 | 9.88 | 13.62 | 0.16 | 9.86 | 110.33 | 757.33 | 60.33 | 56.00 | 53.33 | 55.67 |
| production--dino_global--local_events--evidence--budget-20 | 20.00 | 19.94 | 27.49 | 0.08 | 19.41 | 207.00 | 757.33 | 126.33 | 115.67 | 109.33 | 112.67 |
| production--dino_global--local_events--evidence--budget-40 | 40.00 | 39.94 | 55.05 | 0.08 | 38.02 | 392.00 | 757.33 | 271.33 | 210.67 | 205.00 | 217.33 |
| production--dino_global--local_heads--chronological--budget-05 | 5.00 | 4.70 | 6.48 | 0.42 | 4.56 | 83.33 | 1370.33 | 34.67 | 21.67 | 18.67 | 19.67 |
| production--dino_global--local_heads--chronological--budget-10 | 10.00 | 9.53 | 13.14 | 0.64 | 9.22 | 172.00 | 1370.33 | 69.67 | 37.00 | 42.00 | 42.33 |
| production--dino_global--local_heads--chronological--budget-20 | 20.00 | 19.58 | 26.98 | 0.58 | 18.87 | 356.33 | 1370.33 | 137.67 | 78.67 | 86.67 | 86.67 |
| production--dino_global--local_heads--chronological--budget-40 | 40.00 | 39.65 | 54.66 | 0.48 | 37.89 | 720.00 | 1370.33 | 282.67 | 139.00 | 170.67 | 171.00 |
| production--dino_global--local_heads--evidence--budget-05 | 5.00 | 4.90 | 6.76 | 0.13 | 5.05 | 80.00 | 1370.33 | 27.33 | 26.33 | 27.67 | 28.33 |
| production--dino_global--local_heads--evidence--budget-10 | 10.00 | 9.90 | 13.64 | 0.14 | 9.86 | 178.00 | 1370.33 | 59.00 | 54.67 | 56.67 | 58.33 |
| production--dino_global--local_heads--evidence--budget-20 | 20.00 | 19.94 | 27.49 | 0.08 | 18.73 | 347.33 | 1370.33 | 135.00 | 120.00 | 123.33 | 125.67 |
| production--dino_global--local_heads--evidence--budget-40 | 40.00 | 39.98 | 55.10 | 0.03 | 36.11 | 682.00 | 1370.33 | 297.00 | 230.33 | 238.67 | 241.33 |
| production--dino_boost--legacy--chronological--budget-05 | 5.00 | 4.50 | 6.20 | 0.69 | 4.75 | 25.00 | 191.00 | 24.00 | 23.67 | 15.33 | 16.67 |
| production--dino_boost--legacy--chronological--budget-10 | 10.00 | 9.54 | 13.14 | 0.64 | 10.29 | 48.00 | 191.00 | 44.33 | 42.33 | 35.00 | 38.00 |
| production--dino_boost--legacy--chronological--budget-20 | 20.00 | 19.40 | 26.73 | 0.83 | 21.17 | 94.67 | 191.00 | 86.67 | 75.00 | 71.67 | 75.33 |
| production--dino_boost--legacy--chronological--budget-40 | 40.00 | 32.96 | 45.43 | 9.70 | 36.37 | 155.67 | 191.00 | 144.00 | 106.67 | 117.67 | 122.00 |
| production--dino_boost--legacy--evidence--budget-05 | 5.00 | 4.59 | 6.33 | 0.56 | 5.29 | 18.33 | 191.00 | 18.33 | 17.67 | 13.33 | 14.67 |
| production--dino_boost--legacy--evidence--budget-10 | 10.00 | 9.62 | 13.26 | 0.52 | 10.97 | 37.33 | 191.00 | 37.00 | 36.00 | 29.33 | 31.67 |
| production--dino_boost--legacy--evidence--budget-20 | 20.00 | 19.38 | 26.71 | 0.85 | 21.63 | 81.33 | 191.00 | 79.67 | 71.00 | 65.00 | 68.33 |
| production--dino_boost--legacy--evidence--budget-40 | 40.00 | 33.10 | 45.62 | 9.51 | 36.71 | 149.00 | 191.00 | 140.33 | 111.00 | 116.33 | 121.00 |
| production--dino_boost--local_events--chronological--budget-05 | 5.00 | 4.66 | 6.42 | 0.47 | 4.56 | 49.00 | 778.00 | 33.33 | 20.67 | 19.00 | 19.00 |
| production--dino_boost--local_events--chronological--budget-10 | 10.00 | 9.63 | 13.27 | 0.51 | 9.12 | 98.67 | 778.00 | 71.00 | 38.33 | 43.33 | 43.33 |
| production--dino_boost--local_events--chronological--budget-20 | 20.00 | 19.56 | 26.97 | 0.60 | 18.72 | 207.33 | 778.00 | 139.00 | 76.67 | 86.33 | 86.33 |
| production--dino_boost--local_events--chronological--budget-40 | 40.00 | 39.65 | 54.65 | 0.48 | 37.56 | 411.00 | 778.00 | 286.00 | 136.00 | 170.00 | 170.33 |
| production--dino_boost--local_events--evidence--budget-05 | 5.00 | 4.77 | 6.57 | 0.32 | 5.02 | 54.67 | 778.00 | 24.67 | 23.67 | 22.67 | 22.67 |
| production--dino_boost--local_events--evidence--budget-10 | 10.00 | 9.90 | 13.64 | 0.14 | 10.22 | 111.67 | 778.00 | 54.33 | 50.33 | 49.00 | 49.33 |
| production--dino_boost--local_events--evidence--budget-20 | 20.00 | 19.93 | 27.48 | 0.09 | 19.91 | 215.67 | 778.00 | 118.67 | 105.33 | 105.67 | 107.33 |
| production--dino_boost--local_events--evidence--budget-40 | 40.00 | 39.93 | 55.03 | 0.10 | 38.88 | 410.33 | 778.00 | 259.67 | 200.33 | 202.33 | 211.67 |
| production--dino_boost--local_heads--chronological--budget-05 | 5.00 | 4.72 | 6.51 | 0.39 | 4.64 | 96.33 | 1560.33 | 34.33 | 20.67 | 18.67 | 19.00 |
| production--dino_boost--local_heads--chronological--budget-10 | 10.00 | 9.60 | 13.23 | 0.56 | 9.29 | 200.00 | 1560.33 | 68.33 | 36.00 | 42.00 | 42.33 |
| production--dino_boost--local_heads--chronological--budget-20 | 20.00 | 19.59 | 27.00 | 0.56 | 19.07 | 409.67 | 1560.33 | 135.67 | 75.67 | 84.67 | 85.00 |
| production--dino_boost--local_heads--chronological--budget-40 | 40.00 | 39.63 | 54.62 | 0.52 | 38.34 | 816.00 | 1560.33 | 277.00 | 134.00 | 169.33 | 169.67 |
| production--dino_boost--local_heads--evidence--budget-05 | 5.00 | 4.91 | 6.77 | 0.12 | 5.14 | 83.00 | 1560.33 | 25.67 | 25.00 | 25.67 | 25.67 |
| production--dino_boost--local_heads--evidence--budget-10 | 10.00 | 9.94 | 13.69 | 0.09 | 10.20 | 185.33 | 1560.33 | 55.00 | 51.00 | 53.67 | 54.00 |
| production--dino_boost--local_heads--evidence--budget-20 | 20.00 | 19.97 | 27.52 | 0.05 | 19.40 | 367.67 | 1560.33 | 126.67 | 110.67 | 115.67 | 116.67 |
| production--dino_boost--local_heads--evidence--budget-40 | 40.00 | 39.98 | 55.10 | 0.03 | 36.85 | 764.67 | 1560.33 | 287.33 | 223.67 | 235.33 | 238.67 |
| individual--compact_boost--legacy--chronological--budget-05 | 5.00 | 4.71 | 6.50 | 0.39 | 4.51 | 46.33 | 689.33 | 31.67 | 18.67 | 21.33 | 22.00 |
| individual--compact_boost--legacy--chronological--budget-10 | 10.00 | 9.60 | 13.24 | 0.55 | 9.31 | 96.00 | 689.33 | 63.67 | 34.67 | 46.33 | 47.33 |
| individual--compact_boost--legacy--chronological--budget-20 | 20.00 | 19.61 | 27.03 | 0.54 | 18.88 | 194.33 | 689.33 | 129.67 | 68.00 | 93.33 | 94.67 |
| individual--compact_boost--legacy--chronological--budget-40 | 40.00 | 39.69 | 54.71 | 0.43 | 38.93 | 400.00 | 689.33 | 257.00 | 121.67 | 181.00 | 183.33 |
| individual--compact_boost--legacy--evidence--budget-05 | 5.00 | 4.87 | 6.71 | 0.18 | 4.80 | 41.00 | 689.33 | 30.33 | 29.33 | 26.33 | 28.67 |
| individual--compact_boost--legacy--evidence--budget-10 | 10.00 | 9.90 | 13.64 | 0.14 | 9.61 | 87.67 | 689.33 | 62.33 | 58.00 | 54.33 | 57.33 |
| individual--compact_boost--legacy--evidence--budget-20 | 20.00 | 19.90 | 27.43 | 0.14 | 19.04 | 183.33 | 689.33 | 129.67 | 116.33 | 111.00 | 116.33 |
| individual--compact_boost--legacy--evidence--budget-40 | 40.00 | 39.88 | 54.96 | 0.17 | 38.15 | 387.33 | 689.33 | 265.00 | 198.33 | 213.67 | 221.67 |
| individual--compact_boost--local_events--chronological--budget-05 | 5.00 | 4.63 | 6.38 | 0.51 | 4.33 | 46.00 | 247.33 | 31.00 | 30.33 | 26.00 | 27.33 |
| individual--compact_boost--local_events--chronological--budget-10 | 10.00 | 9.30 | 12.81 | 0.97 | 8.78 | 88.67 | 247.33 | 60.67 | 58.33 | 51.67 | 54.33 |
| individual--compact_boost--local_events--chronological--budget-20 | 20.00 | 16.93 | 23.34 | 4.23 | 16.12 | 161.67 | 247.33 | 107.33 | 97.00 | 91.00 | 97.00 |
| individual--compact_boost--local_events--chronological--budget-40 | 40.00 | 25.93 | 35.74 | 19.39 | 24.99 | 241.00 | 247.33 | 160.00 | 141.00 | 131.33 | 139.00 |
| individual--compact_boost--local_events--evidence--budget-05 | 5.00 | 4.75 | 6.55 | 0.34 | 4.84 | 29.33 | 247.33 | 25.67 | 25.67 | 23.67 | 24.67 |
| individual--compact_boost--local_events--evidence--budget-10 | 10.00 | 9.47 | 13.05 | 0.73 | 9.49 | 70.00 | 247.33 | 53.67 | 52.67 | 49.00 | 50.00 |
| individual--compact_boost--local_events--evidence--budget-20 | 20.00 | 17.11 | 23.58 | 3.98 | 16.92 | 133.33 | 247.33 | 100.33 | 96.33 | 90.33 | 94.00 |
| individual--compact_boost--local_events--evidence--budget-40 | 40.00 | 25.95 | 35.76 | 19.37 | 25.13 | 238.33 | 247.33 | 159.00 | 142.00 | 131.67 | 139.33 |
| individual--compact_boost--local_heads--chronological--budget-05 | 5.00 | 4.73 | 6.53 | 0.37 | 4.29 | 59.00 | 423.67 | 35.00 | 31.00 | 30.67 | 31.33 |
| individual--compact_boost--local_heads--chronological--budget-10 | 10.00 | 9.43 | 13.00 | 0.78 | 8.46 | 117.00 | 423.67 | 69.67 | 59.00 | 59.67 | 61.33 |
| individual--compact_boost--local_heads--chronological--budget-20 | 20.00 | 17.49 | 24.11 | 3.46 | 15.73 | 211.67 | 423.67 | 127.33 | 102.00 | 100.67 | 105.33 |
| individual--compact_boost--local_heads--chronological--budget-40 | 40.00 | 29.84 | 41.13 | 14.00 | 27.20 | 370.33 | 423.67 | 215.33 | 169.33 | 164.67 | 172.33 |
| individual--compact_boost--local_heads--evidence--budget-05 | 5.00 | 4.76 | 6.56 | 0.33 | 4.67 | 59.33 | 423.67 | 28.67 | 28.67 | 26.33 | 27.00 |
| individual--compact_boost--local_heads--evidence--budget-10 | 10.00 | 9.64 | 13.29 | 0.50 | 9.25 | 123.33 | 423.67 | 61.00 | 59.00 | 54.67 | 57.33 |
| individual--compact_boost--local_heads--evidence--budget-20 | 20.00 | 17.62 | 24.28 | 3.29 | 16.73 | 229.00 | 423.67 | 115.00 | 108.00 | 100.00 | 105.67 |
| individual--compact_boost--local_heads--evidence--budget-40 | 40.00 | 29.97 | 41.31 | 13.82 | 27.43 | 373.67 | 423.67 | 213.33 | 179.33 | 171.33 | 178.67 |
| individual--dino_global--legacy--chronological--budget-05 | 5.00 | 4.62 | 6.37 | 0.52 | 4.33 | 45.33 | 624.33 | 33.33 | 19.67 | 25.00 | 25.00 |
| individual--dino_global--legacy--chronological--budget-10 | 10.00 | 9.68 | 13.35 | 0.43 | 9.14 | 95.33 | 624.33 | 69.33 | 39.67 | 52.33 | 52.33 |
| individual--dino_global--legacy--chronological--budget-20 | 20.00 | 19.60 | 27.02 | 0.55 | 18.76 | 196.67 | 624.33 | 136.33 | 71.67 | 98.33 | 99.00 |
| individual--dino_global--legacy--chronological--budget-40 | 40.00 | 39.65 | 54.65 | 0.49 | 37.85 | 394.67 | 624.33 | 275.00 | 138.67 | 196.33 | 198.33 |
| individual--dino_global--legacy--evidence--budget-05 | 5.00 | 4.83 | 6.66 | 0.23 | 4.63 | 41.00 | 624.33 | 32.00 | 31.00 | 29.67 | 31.67 |
| individual--dino_global--legacy--evidence--budget-10 | 10.00 | 9.84 | 13.56 | 0.22 | 9.31 | 83.33 | 624.33 | 66.67 | 63.33 | 61.33 | 65.33 |
| individual--dino_global--legacy--evidence--budget-20 | 20.00 | 19.88 | 27.41 | 0.16 | 18.68 | 180.33 | 624.33 | 138.67 | 121.33 | 123.33 | 128.67 |
| individual--dino_global--legacy--evidence--budget-40 | 40.00 | 39.90 | 54.99 | 0.14 | 37.11 | 376.33 | 624.33 | 283.67 | 213.67 | 234.67 | 241.00 |
| individual--dino_global--local_events--chronological--budget-05 | 5.00 | 4.60 | 6.35 | 0.55 | 4.07 | 48.33 | 164.00 | 34.33 | 33.67 | 27.33 | 31.33 |
| individual--dino_global--local_events--chronological--budget-10 | 10.00 | 8.89 | 12.25 | 1.54 | 7.86 | 96.67 | 164.00 | 66.67 | 62.67 | 48.67 | 57.00 |
| individual--dino_global--local_events--chronological--budget-20 | 20.00 | 13.68 | 18.86 | 8.71 | 12.18 | 142.00 | 164.00 | 100.33 | 92.67 | 73.00 | 84.67 |
| individual--dino_global--local_events--chronological--budget-40 | 40.00 | 16.13 | 22.24 | 32.90 | 14.42 | 164.00 | 164.00 | 117.00 | 108.00 | 87.00 | 98.33 |
| individual--dino_global--local_events--evidence--budget-05 | 5.00 | 4.72 | 6.51 | 0.38 | 4.41 | 43.33 | 164.00 | 32.00 | 31.67 | 26.67 | 28.33 |
| individual--dino_global--local_events--evidence--budget-10 | 10.00 | 8.93 | 12.31 | 1.47 | 8.23 | 89.33 | 164.00 | 62.00 | 60.33 | 48.33 | 56.00 |
| individual--dino_global--local_events--evidence--budget-20 | 20.00 | 13.82 | 19.05 | 8.52 | 12.46 | 136.00 | 164.00 | 99.33 | 93.33 | 76.00 | 86.67 |
| individual--dino_global--local_events--evidence--budget-40 | 40.00 | 16.13 | 22.24 | 32.90 | 14.42 | 164.00 | 164.00 | 117.00 | 108.00 | 87.00 | 98.33 |
| individual--dino_global--local_heads--chronological--budget-05 | 5.00 | 4.54 | 6.25 | 0.64 | 3.86 | 56.00 | 388.00 | 37.67 | 32.33 | 29.00 | 32.33 |
| individual--dino_global--local_heads--chronological--budget-10 | 10.00 | 9.72 | 13.39 | 0.39 | 8.26 | 124.33 | 388.00 | 78.67 | 66.00 | 61.67 | 65.67 |
| individual--dino_global--local_heads--chronological--budget-20 | 20.00 | 18.01 | 24.82 | 2.75 | 15.19 | 227.00 | 388.00 | 148.00 | 118.67 | 110.67 | 116.67 |
| individual--dino_global--local_heads--chronological--budget-40 | 40.00 | 28.63 | 39.46 | 15.67 | 24.11 | 356.00 | 388.00 | 236.33 | 187.67 | 174.00 | 182.33 |
| individual--dino_global--local_heads--evidence--budget-05 | 5.00 | 4.74 | 6.53 | 0.36 | 4.39 | 77.33 | 388.00 | 32.67 | 31.00 | 26.33 | 27.67 |
| individual--dino_global--local_heads--evidence--budget-10 | 10.00 | 9.76 | 13.46 | 0.33 | 8.67 | 149.33 | 388.00 | 73.00 | 69.00 | 61.00 | 64.33 |
| individual--dino_global--local_heads--evidence--budget-20 | 20.00 | 18.17 | 25.04 | 2.53 | 15.53 | 246.67 | 388.00 | 146.00 | 130.33 | 123.33 | 127.67 |
| individual--dino_global--local_heads--evidence--budget-40 | 40.00 | 28.65 | 39.49 | 15.64 | 24.18 | 360.00 | 388.00 | 236.00 | 196.00 | 183.33 | 191.00 |
| individual--dino_boost--legacy--chronological--budget-05 | 5.00 | 4.61 | 6.35 | 0.54 | 4.46 | 44.67 | 712.67 | 34.33 | 16.33 | 20.67 | 22.00 |
| individual--dino_boost--legacy--chronological--budget-10 | 10.00 | 9.64 | 13.28 | 0.50 | 9.29 | 94.33 | 712.67 | 71.67 | 31.00 | 46.00 | 47.33 |
| individual--dino_boost--legacy--chronological--budget-20 | 20.00 | 19.63 | 27.06 | 0.50 | 18.92 | 196.33 | 712.67 | 139.00 | 61.00 | 92.00 | 93.67 |
| individual--dino_boost--legacy--chronological--budget-40 | 40.00 | 39.67 | 54.67 | 0.46 | 38.57 | 400.67 | 712.67 | 275.33 | 118.00 | 185.00 | 187.33 |
| individual--dino_boost--legacy--evidence--budget-05 | 5.00 | 4.86 | 6.69 | 0.20 | 4.64 | 38.33 | 712.67 | 33.33 | 31.67 | 27.33 | 29.00 |
| individual--dino_boost--legacy--evidence--budget-10 | 10.00 | 9.84 | 13.57 | 0.22 | 9.31 | 80.00 | 712.67 | 67.67 | 63.33 | 58.33 | 60.00 |
| individual--dino_boost--legacy--evidence--budget-20 | 20.00 | 19.92 | 27.46 | 0.11 | 18.58 | 172.33 | 712.67 | 140.33 | 126.33 | 122.00 | 126.67 |
| individual--dino_boost--legacy--evidence--budget-40 | 40.00 | 39.86 | 54.94 | 0.19 | 37.49 | 380.00 | 712.67 | 287.00 | 223.00 | 242.33 | 246.67 |
| individual--dino_boost--local_events--chronological--budget-05 | 5.00 | 4.63 | 6.39 | 0.50 | 4.09 | 47.00 | 178.33 | 35.00 | 34.33 | 25.00 | 30.00 |
| individual--dino_boost--local_events--chronological--budget-10 | 10.00 | 9.31 | 12.83 | 0.95 | 8.20 | 94.67 | 178.33 | 70.00 | 66.67 | 51.67 | 59.00 |
| individual--dino_boost--local_events--chronological--budget-20 | 20.00 | 14.36 | 19.80 | 7.77 | 12.81 | 149.00 | 178.33 | 105.67 | 96.67 | 76.33 | 86.33 |
| individual--dino_boost--local_events--chronological--budget-40 | 40.00 | 17.70 | 24.40 | 30.74 | 15.97 | 178.33 | 178.33 | 126.33 | 114.67 | 92.33 | 103.67 |
| individual--dino_boost--local_events--evidence--budget-05 | 5.00 | 4.78 | 6.59 | 0.30 | 4.60 | 41.00 | 178.33 | 30.00 | 30.00 | 25.33 | 26.67 |
| individual--dino_boost--local_events--evidence--budget-10 | 10.00 | 9.46 | 13.04 | 0.74 | 8.78 | 89.33 | 178.33 | 64.33 | 62.67 | 53.33 | 56.33 |
| individual--dino_boost--local_events--evidence--budget-20 | 20.00 | 14.44 | 19.91 | 7.66 | 13.18 | 140.33 | 178.33 | 100.67 | 94.67 | 78.67 | 85.00 |
| individual--dino_boost--local_events--evidence--budget-40 | 40.00 | 17.70 | 24.40 | 30.74 | 15.97 | 178.33 | 178.33 | 126.33 | 114.67 | 92.33 | 103.67 |
| individual--dino_boost--local_heads--chronological--budget-05 | 5.00 | 4.55 | 6.27 | 0.62 | 3.86 | 54.67 | 480.00 | 38.33 | 31.33 | 29.00 | 30.33 |
| individual--dino_boost--local_heads--chronological--budget-10 | 10.00 | 9.58 | 13.20 | 0.58 | 8.06 | 118.33 | 480.00 | 78.67 | 61.33 | 59.33 | 61.00 |
| individual--dino_boost--local_heads--chronological--budget-20 | 20.00 | 18.53 | 25.54 | 2.02 | 15.61 | 230.00 | 480.00 | 151.33 | 115.00 | 110.00 | 114.67 |
| individual--dino_boost--local_heads--chronological--budget-40 | 40.00 | 34.19 | 47.12 | 8.01 | 29.00 | 426.00 | 480.00 | 277.33 | 213.33 | 204.67 | 211.67 |
| individual--dino_boost--local_heads--evidence--budget-05 | 5.00 | 4.75 | 6.54 | 0.35 | 4.55 | 72.67 | 480.00 | 30.00 | 29.33 | 26.00 | 27.33 |
| individual--dino_boost--local_heads--evidence--budget-10 | 10.00 | 9.77 | 13.47 | 0.31 | 8.98 | 147.00 | 480.00 | 67.67 | 66.33 | 58.67 | 60.67 |
| individual--dino_boost--local_heads--evidence--budget-20 | 20.00 | 18.71 | 25.79 | 1.78 | 16.27 | 256.33 | 480.00 | 142.67 | 134.67 | 123.00 | 127.33 |
| individual--dino_boost--local_heads--evidence--budget-40 | 40.00 | 34.29 | 47.26 | 7.87 | 29.02 | 426.67 | 480.00 | 277.33 | 227.67 | 214.00 | 222.33 |

## All four padding cases and export accounting

Correctly removed time is valid unwanted footage excluded from export. Incorrectly removed time is wanted padded-human export omitted. Incorrect export is unwanted footage retained. Model-minus-human duration is signed.

| Arm | Pad s | P_pad % | R_core % | F1_padP_coreR % | Model export min | Human export min | Difference min | Correctly removed min | Incorrectly removed min | Incorrect export min | Missed core s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_boost | 0 | 84.24 | 82.00 | 83.10 | 38.73 | 39.79 | -1.05 | 91.94 | 7.16 | 6.11 | 429.79 |
| compact_boost | 1 | 86.47 | 89.60 | 88.00 | 49.06 | 50.52 | -1.46 | 80.67 | 8.11 | 6.65 | 248.24 |
| compact_boost | 2 | 88.30 | 92.41 | 90.31 | 59.06 | 61.25 | -2.19 | 69.67 | 9.11 | 6.91 | 181.20 |
| compact_boost | 3 | 89.71 | 93.88 | 91.75 | 69.09 | 72.05 | -2.97 | 58.67 | 10.08 | 7.11 | 146.03 |
| dino_boost | 0 | 84.34 | 86.22 | 85.25 | 40.70 | 39.79 | 0.91 | 91.65 | 5.48 | 6.39 | 328.89 |
| dino_boost | 1 | 86.69 | 94.63 | 90.48 | 51.69 | 50.52 | 1.17 | 80.41 | 5.73 | 6.90 | 128.08 |
| dino_boost | 2 | 88.47 | 96.95 | 92.51 | 62.33 | 61.25 | 1.08 | 69.38 | 6.13 | 7.20 | 72.70 |
| dino_boost | 3 | 90.08 | 97.84 | 93.79 | 72.74 | 72.05 | 0.69 | 58.55 | 6.55 | 7.23 | 51.68 |
| dino_global | 0 | 85.81 | 87.32 | 86.53 | 40.52 | 39.79 | 0.74 | 92.27 | 5.04 | 5.78 | 302.60 |
| dino_global | 1 | 87.96 | 94.96 | 91.31 | 51.28 | 50.52 | 0.76 | 81.11 | 5.45 | 6.21 | 120.25 |
| dino_global | 2 | 89.50 | 96.96 | 93.07 | 61.78 | 61.25 | 0.53 | 70.07 | 5.98 | 6.51 | 72.57 |
| dino_global | 3 | 90.73 | 97.64 | 94.05 | 72.30 | 72.05 | 0.25 | 59.05 | 6.48 | 6.73 | 56.46 |
| production | 0 | 65.50 | 95.75 | 77.79 | 58.16 | 39.79 | 18.37 | 77.98 | 1.69 | 20.07 | 101.49 |
| production | 1 | 69.40 | 98.30 | 81.36 | 70.39 | 50.52 | 19.87 | 65.77 | 1.67 | 21.54 | 40.52 |
| production | 2 | 72.45 | 99.27 | 83.76 | 82.38 | 61.25 | 21.13 | 53.88 | 1.57 | 22.70 | 17.47 |
| production | 3 | 75.02 | 99.49 | 85.54 | 93.95 | 72.05 | 21.90 | 42.31 | 1.57 | 23.47 | 12.18 |
| production--compact_boost--legacy--chronological--budget-05 | 0 | 67.48 | 96.11 | 79.29 | 56.66 | 39.79 | 16.88 | 79.62 | 1.55 | 18.43 | 92.94 |
| production--compact_boost--legacy--chronological--budget-05 | 1 | 71.29 | 98.48 | 82.71 | 68.72 | 50.52 | 18.20 | 67.59 | 1.53 | 19.73 | 36.19 |
| production--compact_boost--legacy--chronological--budget-05 | 2 | 74.33 | 99.36 | 85.04 | 80.49 | 61.25 | 19.23 | 55.92 | 1.43 | 20.66 | 15.25 |
| production--compact_boost--legacy--chronological--budget-05 | 3 | 76.86 | 99.55 | 86.74 | 91.89 | 72.05 | 19.84 | 44.51 | 1.43 | 21.27 | 10.77 |
| production--compact_boost--legacy--chronological--budget-10 | 0 | 69.50 | 96.74 | 80.89 | 55.38 | 39.79 | 15.60 | 81.16 | 1.30 | 16.89 | 77.73 |
| production--compact_boost--legacy--chronological--budget-10 | 1 | 73.04 | 98.88 | 84.02 | 67.38 | 50.52 | 16.86 | 69.15 | 1.31 | 18.17 | 26.77 |
| production--compact_boost--legacy--chronological--budget-10 | 2 | 75.71 | 99.40 | 85.95 | 79.12 | 61.25 | 17.87 | 57.36 | 1.35 | 19.22 | 14.21 |
| production--compact_boost--legacy--chronological--budget-10 | 3 | 78.07 | 99.58 | 87.52 | 90.56 | 72.05 | 18.51 | 45.92 | 1.35 | 19.86 | 10.06 |
| production--compact_boost--legacy--chronological--budget-20 | 0 | 73.89 | 97.25 | 83.98 | 52.36 | 39.79 | 12.58 | 84.38 | 1.09 | 13.67 | 65.66 |
| production--compact_boost--legacy--chronological--budget-20 | 1 | 77.02 | 99.11 | 86.68 | 64.16 | 50.52 | 13.64 | 72.57 | 1.11 | 14.74 | 21.26 |
| production--compact_boost--legacy--chronological--budget-20 | 2 | 79.36 | 99.51 | 88.30 | 75.72 | 61.25 | 14.47 | 60.95 | 1.16 | 15.63 | 11.63 |
| production--compact_boost--legacy--chronological--budget-20 | 3 | 81.42 | 99.64 | 89.62 | 87.00 | 72.05 | 14.94 | 49.62 | 1.22 | 16.16 | 8.48 |
| production--compact_boost--legacy--chronological--budget-40 | 0 | 82.74 | 97.73 | 89.61 | 46.99 | 39.79 | 7.21 | 89.94 | 0.90 | 8.11 | 54.22 |
| production--compact_boost--legacy--chronological--budget-40 | 1 | 85.12 | 99.20 | 91.62 | 58.25 | 50.52 | 7.73 | 78.65 | 0.94 | 8.67 | 18.98 |
| production--compact_boost--legacy--chronological--budget-40 | 2 | 86.84 | 99.53 | 92.75 | 69.35 | 61.25 | 8.09 | 67.45 | 1.03 | 9.13 | 11.34 |
| production--compact_boost--legacy--chronological--budget-40 | 3 | 87.87 | 99.64 | 93.39 | 80.73 | 72.05 | 8.68 | 55.98 | 1.12 | 9.80 | 8.48 |
| production--compact_boost--legacy--evidence--budget-05 | 0 | 67.10 | 96.09 | 79.02 | 56.98 | 39.79 | 17.19 | 79.30 | 1.56 | 18.75 | 93.33 |
| production--compact_boost--legacy--evidence--budget-05 | 1 | 70.88 | 98.47 | 82.43 | 69.05 | 50.52 | 18.53 | 67.21 | 1.58 | 20.11 | 36.64 |
| production--compact_boost--legacy--evidence--budget-05 | 2 | 73.76 | 99.31 | 84.65 | 80.98 | 61.25 | 19.73 | 55.33 | 1.52 | 21.25 | 16.43 |
| production--compact_boost--legacy--evidence--budget-05 | 3 | 76.13 | 99.52 | 86.27 | 92.64 | 72.05 | 20.58 | 43.67 | 1.53 | 22.11 | 11.47 |
| production--compact_boost--legacy--evidence--budget-10 | 0 | 69.10 | 96.36 | 80.49 | 55.48 | 39.79 | 15.70 | 80.90 | 1.45 | 17.14 | 86.87 |
| production--compact_boost--legacy--evidence--budget-10 | 1 | 72.62 | 98.64 | 83.65 | 67.53 | 50.52 | 17.01 | 68.82 | 1.48 | 18.49 | 32.51 |
| production--compact_boost--legacy--evidence--budget-10 | 2 | 75.23 | 99.40 | 85.64 | 79.50 | 61.25 | 18.24 | 56.89 | 1.45 | 19.69 | 14.35 |
| production--compact_boost--legacy--evidence--budget-10 | 3 | 77.40 | 99.58 | 87.10 | 91.18 | 72.05 | 19.13 | 45.17 | 1.48 | 20.60 | 10.06 |
| production--compact_boost--legacy--evidence--budget-20 | 0 | 73.71 | 96.70 | 83.65 | 52.20 | 39.79 | 12.41 | 84.32 | 1.31 | 13.73 | 78.79 |
| production--compact_boost--legacy--evidence--budget-20 | 1 | 76.70 | 98.78 | 86.35 | 64.11 | 50.52 | 13.59 | 72.38 | 1.35 | 14.93 | 29.11 |
| production--compact_boost--legacy--evidence--budget-20 | 2 | 79.04 | 99.48 | 88.09 | 75.83 | 61.25 | 14.57 | 60.69 | 1.32 | 15.89 | 12.39 |
| production--compact_boost--legacy--evidence--budget-20 | 3 | 80.68 | 99.62 | 89.15 | 87.60 | 72.05 | 15.55 | 48.85 | 1.38 | 16.93 | 9.01 |
| production--compact_boost--legacy--evidence--budget-40 | 0 | 82.15 | 97.49 | 89.16 | 47.21 | 39.79 | 7.43 | 89.62 | 1.00 | 8.43 | 59.94 |
| production--compact_boost--legacy--evidence--budget-40 | 1 | 84.45 | 99.09 | 91.19 | 58.59 | 50.52 | 8.07 | 78.20 | 1.04 | 9.11 | 21.74 |
| production--compact_boost--legacy--evidence--budget-40 | 2 | 86.24 | 99.53 | 92.41 | 69.75 | 61.25 | 8.50 | 66.99 | 1.10 | 9.60 | 11.27 |
| production--compact_boost--legacy--evidence--budget-40 | 3 | 87.19 | 99.64 | 93.00 | 81.29 | 72.05 | 9.23 | 55.36 | 1.18 | 10.42 | 8.48 |
| production--compact_boost--local_events--chronological--budget-05 | 0 | 67.22 | 96.03 | 79.08 | 56.84 | 39.79 | 17.06 | 79.41 | 1.58 | 18.64 | 94.71 |
| production--compact_boost--local_events--chronological--budget-05 | 1 | 71.03 | 98.39 | 82.50 | 68.94 | 50.52 | 18.42 | 67.34 | 1.55 | 19.97 | 38.38 |
| production--compact_boost--local_events--chronological--budget-05 | 2 | 74.07 | 99.27 | 84.84 | 80.73 | 61.25 | 19.48 | 55.65 | 1.45 | 20.94 | 17.33 |
| production--compact_boost--local_events--chronological--budget-05 | 3 | 76.56 | 99.49 | 86.53 | 92.22 | 72.05 | 20.16 | 44.16 | 1.46 | 21.62 | 12.18 |
| production--compact_boost--local_events--chronological--budget-10 | 0 | 69.06 | 96.38 | 80.46 | 55.53 | 39.79 | 15.74 | 80.87 | 1.44 | 17.18 | 86.50 |
| production--compact_boost--local_events--chronological--budget-10 | 1 | 72.58 | 98.55 | 83.59 | 67.61 | 50.52 | 17.09 | 68.77 | 1.45 | 18.54 | 34.61 |
| production--compact_boost--local_events--chronological--budget-10 | 2 | 75.32 | 99.27 | 85.66 | 79.44 | 61.25 | 18.19 | 56.98 | 1.41 | 19.60 | 17.33 |
| production--compact_boost--local_events--chronological--budget-10 | 3 | 77.68 | 99.49 | 87.24 | 90.93 | 72.05 | 18.88 | 45.48 | 1.42 | 20.30 | 12.18 |
| production--compact_boost--local_events--chronological--budget-20 | 0 | 72.48 | 97.23 | 83.05 | 53.37 | 39.79 | 13.59 | 83.36 | 1.10 | 14.69 | 66.17 |
| production--compact_boost--local_events--chronological--budget-20 | 1 | 75.47 | 99.00 | 85.65 | 65.49 | 50.52 | 14.97 | 71.25 | 1.09 | 16.06 | 23.87 |
| production--compact_boost--local_events--chronological--budget-20 | 2 | 77.71 | 99.45 | 87.25 | 77.39 | 61.25 | 16.14 | 59.33 | 1.11 | 17.25 | 13.12 |
| production--compact_boost--local_events--chronological--budget-20 | 3 | 79.77 | 99.62 | 88.60 | 88.96 | 72.05 | 16.91 | 47.78 | 1.09 | 18.00 | 8.97 |
| production--compact_boost--local_events--chronological--budget-40 | 0 | 80.44 | 97.91 | 88.32 | 48.43 | 39.79 | 8.64 | 88.57 | 0.83 | 9.48 | 49.91 |
| production--compact_boost--local_events--chronological--budget-40 | 1 | 82.71 | 99.21 | 90.21 | 60.09 | 50.52 | 9.57 | 76.93 | 0.82 | 10.39 | 18.86 |
| production--compact_boost--local_events--chronological--budget-40 | 2 | 84.34 | 99.54 | 91.31 | 71.62 | 61.25 | 10.36 | 65.37 | 0.85 | 11.21 | 11.05 |
| production--compact_boost--local_events--chronological--budget-40 | 3 | 85.90 | 99.68 | 92.28 | 82.84 | 72.05 | 10.78 | 54.10 | 0.90 | 11.68 | 7.61 |
| production--compact_boost--local_events--evidence--budget-05 | 0 | 66.87 | 96.80 | 79.10 | 57.59 | 39.79 | 17.81 | 78.97 | 1.27 | 19.08 | 76.32 |
| production--compact_boost--local_events--evidence--budget-05 | 1 | 70.48 | 98.95 | 82.32 | 69.82 | 50.52 | 19.30 | 66.71 | 1.31 | 20.61 | 25.03 |
| production--compact_boost--local_events--evidence--budget-05 | 2 | 73.31 | 99.51 | 84.42 | 81.71 | 61.25 | 20.46 | 54.77 | 1.35 | 21.81 | 11.63 |
| production--compact_boost--local_events--evidence--budget-05 | 3 | 75.74 | 99.65 | 86.07 | 93.36 | 72.05 | 21.31 | 43.13 | 1.34 | 22.65 | 8.32 |
| production--compact_boost--local_events--evidence--budget-10 | 0 | 68.53 | 97.31 | 80.42 | 56.50 | 39.79 | 16.71 | 80.27 | 1.07 | 17.78 | 64.11 |
| production--compact_boost--local_events--evidence--budget-10 | 1 | 71.99 | 99.10 | 83.40 | 68.59 | 50.52 | 18.07 | 68.10 | 1.14 | 19.21 | 21.54 |
| production--compact_boost--local_events--evidence--budget-10 | 2 | 74.74 | 99.56 | 85.38 | 80.38 | 61.25 | 19.12 | 56.28 | 1.18 | 20.30 | 10.45 |
| production--compact_boost--local_events--evidence--budget-10 | 3 | 76.96 | 99.67 | 86.86 | 92.11 | 72.05 | 20.05 | 44.56 | 1.17 | 21.22 | 7.77 |
| production--compact_boost--local_events--evidence--budget-20 | 0 | 73.48 | 97.68 | 83.87 | 52.90 | 39.79 | 13.11 | 84.01 | 0.92 | 14.04 | 55.39 |
| production--compact_boost--local_events--evidence--budget-20 | 1 | 76.39 | 99.22 | 86.31 | 64.85 | 50.52 | 14.33 | 71.99 | 0.99 | 15.32 | 18.67 |
| production--compact_boost--local_events--evidence--budget-20 | 2 | 78.65 | 99.64 | 87.91 | 76.56 | 61.25 | 15.31 | 60.23 | 1.04 | 16.35 | 8.57 |
| production--compact_boost--local_events--evidence--budget-20 | 3 | 80.37 | 99.73 | 89.01 | 88.39 | 72.05 | 16.34 | 48.42 | 1.02 | 17.36 | 6.34 |
| production--compact_boost--local_events--evidence--budget-40 | 0 | 83.76 | 98.43 | 90.50 | 46.76 | 39.79 | 6.97 | 90.45 | 0.62 | 7.60 | 37.41 |
| production--compact_boost--local_events--evidence--budget-40 | 1 | 85.77 | 99.45 | 92.10 | 58.16 | 50.52 | 7.64 | 79.03 | 0.64 | 8.28 | 13.05 |
| production--compact_boost--local_events--evidence--budget-40 | 2 | 87.21 | 99.78 | 93.07 | 69.50 | 61.25 | 8.24 | 67.69 | 0.65 | 8.89 | 5.27 |
| production--compact_boost--local_events--evidence--budget-40 | 3 | 87.92 | 99.87 | 93.51 | 81.22 | 72.05 | 9.16 | 55.96 | 0.65 | 9.82 | 3.04 |
| production--compact_boost--local_heads--chronological--budget-05 | 0 | 67.23 | 96.03 | 79.09 | 56.83 | 39.79 | 17.05 | 79.42 | 1.58 | 18.63 | 94.71 |
| production--compact_boost--local_heads--chronological--budget-05 | 1 | 71.02 | 98.39 | 82.49 | 68.95 | 50.52 | 18.43 | 67.33 | 1.55 | 19.98 | 38.38 |
| production--compact_boost--local_heads--chronological--budget-05 | 2 | 74.06 | 99.27 | 84.83 | 80.74 | 61.25 | 19.49 | 55.64 | 1.45 | 20.94 | 17.33 |
| production--compact_boost--local_heads--chronological--budget-05 | 3 | 76.55 | 99.49 | 86.52 | 92.23 | 72.05 | 20.17 | 44.15 | 1.46 | 21.63 | 12.18 |
| production--compact_boost--local_heads--chronological--budget-10 | 0 | 68.97 | 96.38 | 80.40 | 55.60 | 39.79 | 15.81 | 80.80 | 1.44 | 17.25 | 86.50 |
| production--compact_boost--local_heads--chronological--budget-10 | 1 | 72.48 | 98.55 | 83.53 | 67.70 | 50.52 | 17.18 | 68.68 | 1.45 | 18.63 | 34.61 |
| production--compact_boost--local_heads--chronological--budget-10 | 2 | 75.19 | 99.27 | 85.57 | 79.58 | 61.25 | 18.33 | 56.84 | 1.41 | 19.74 | 17.33 |
| production--compact_boost--local_heads--chronological--budget-10 | 3 | 77.56 | 99.49 | 87.17 | 91.07 | 72.05 | 19.02 | 45.35 | 1.42 | 20.43 | 12.18 |
| production--compact_boost--local_heads--chronological--budget-20 | 0 | 72.44 | 97.20 | 83.01 | 53.39 | 39.79 | 13.60 | 83.33 | 1.11 | 14.72 | 66.85 |
| production--compact_boost--local_heads--chronological--budget-20 | 1 | 75.44 | 98.99 | 85.62 | 65.50 | 50.52 | 14.98 | 71.23 | 1.11 | 16.09 | 24.09 |
| production--compact_boost--local_heads--chronological--budget-20 | 2 | 77.71 | 99.45 | 87.25 | 77.37 | 61.25 | 16.12 | 59.34 | 1.12 | 17.24 | 13.12 |
| production--compact_boost--local_heads--chronological--budget-20 | 3 | 79.71 | 99.62 | 88.56 | 89.01 | 72.05 | 16.96 | 47.72 | 1.10 | 18.06 | 8.97 |
| production--compact_boost--local_heads--chronological--budget-40 | 0 | 80.55 | 98.01 | 88.43 | 48.41 | 39.79 | 8.62 | 88.63 | 0.79 | 9.41 | 47.47 |
| production--compact_boost--local_heads--chronological--budget-40 | 1 | 82.82 | 99.31 | 90.32 | 60.05 | 50.52 | 9.53 | 77.00 | 0.78 | 10.32 | 16.58 |
| production--compact_boost--local_heads--chronological--budget-40 | 2 | 84.42 | 99.60 | 91.38 | 71.60 | 61.25 | 10.34 | 65.43 | 0.81 | 11.15 | 9.64 |
| production--compact_boost--local_heads--chronological--budget-40 | 3 | 85.91 | 99.71 | 92.30 | 82.87 | 72.05 | 10.82 | 54.10 | 0.86 | 11.68 | 6.90 |
| production--compact_boost--local_heads--evidence--budget-05 | 0 | 67.17 | 96.67 | 79.27 | 57.25 | 39.79 | 17.47 | 79.25 | 1.33 | 18.79 | 79.57 |
| production--compact_boost--local_heads--evidence--budget-05 | 1 | 70.73 | 98.96 | 82.50 | 69.53 | 50.52 | 19.01 | 66.96 | 1.34 | 20.35 | 24.89 |
| production--compact_boost--local_heads--evidence--budget-05 | 2 | 73.48 | 99.50 | 84.54 | 81.46 | 61.25 | 20.20 | 54.98 | 1.40 | 21.60 | 11.84 |
| production--compact_boost--local_heads--evidence--budget-05 | 3 | 75.86 | 99.64 | 86.14 | 93.14 | 72.05 | 21.08 | 43.30 | 1.40 | 22.48 | 8.48 |
| production--compact_boost--local_heads--evidence--budget-10 | 0 | 68.43 | 97.10 | 80.28 | 56.46 | 39.79 | 16.67 | 80.22 | 1.15 | 17.83 | 69.29 |
| production--compact_boost--local_heads--evidence--budget-10 | 1 | 71.70 | 99.04 | 83.18 | 68.76 | 50.52 | 18.24 | 67.85 | 1.22 | 19.46 | 23.02 |
| production--compact_boost--local_heads--evidence--budget-10 | 2 | 74.33 | 99.52 | 85.10 | 80.70 | 61.25 | 19.44 | 55.86 | 1.27 | 20.72 | 11.35 |
| production--compact_boost--local_heads--evidence--budget-10 | 3 | 76.61 | 99.64 | 86.62 | 92.39 | 72.05 | 20.34 | 44.17 | 1.28 | 21.61 | 8.48 |
| production--compact_boost--local_heads--evidence--budget-20 | 0 | 71.47 | 97.39 | 82.44 | 54.22 | 39.79 | 14.43 | 82.58 | 1.04 | 15.47 | 62.20 |
| production--compact_boost--local_heads--evidence--budget-20 | 1 | 74.28 | 99.10 | 84.91 | 66.54 | 50.52 | 16.02 | 70.20 | 1.09 | 17.12 | 21.47 |
| production--compact_boost--local_heads--evidence--budget-20 | 2 | 76.55 | 99.55 | 86.55 | 78.52 | 61.25 | 17.27 | 58.17 | 1.15 | 18.41 | 10.78 |
| production--compact_boost--local_heads--evidence--budget-20 | 3 | 78.50 | 99.67 | 87.82 | 90.34 | 72.05 | 18.29 | 46.35 | 1.14 | 19.43 | 7.94 |
| production--compact_boost--local_heads--evidence--budget-40 | 0 | 77.58 | 97.98 | 86.59 | 50.25 | 39.79 | 10.47 | 86.78 | 0.80 | 11.27 | 48.15 |
| production--compact_boost--local_heads--evidence--budget-40 | 1 | 79.40 | 99.31 | 88.25 | 62.57 | 50.52 | 12.06 | 74.42 | 0.83 | 12.89 | 16.43 |
| production--compact_boost--local_heads--evidence--budget-40 | 2 | 81.02 | 99.69 | 89.39 | 74.52 | 61.25 | 13.27 | 62.43 | 0.88 | 14.15 | 7.46 |
| production--compact_boost--local_heads--evidence--budget-40 | 3 | 82.42 | 99.76 | 90.26 | 86.40 | 72.05 | 14.34 | 50.59 | 0.85 | 15.19 | 5.66 |
| production--dino_global--legacy--chronological--budget-05 | 0 | 67.61 | 96.31 | 79.45 | 56.67 | 39.79 | 16.89 | 79.69 | 1.47 | 18.35 | 88.14 |
| production--dino_global--legacy--chronological--budget-05 | 1 | 71.42 | 98.66 | 82.86 | 68.74 | 50.52 | 18.22 | 67.67 | 1.42 | 19.65 | 32.08 |
| production--dino_global--legacy--chronological--budget-05 | 2 | 74.40 | 99.47 | 85.12 | 80.55 | 61.25 | 19.30 | 55.96 | 1.33 | 20.62 | 12.76 |
| production--dino_global--legacy--chronological--budget-05 | 3 | 76.96 | 99.64 | 86.84 | 91.93 | 72.05 | 19.87 | 44.60 | 1.31 | 21.18 | 8.61 |
| production--dino_global--legacy--chronological--budget-10 | 0 | 69.67 | 96.82 | 81.03 | 55.29 | 39.79 | 15.51 | 81.28 | 1.27 | 16.77 | 75.99 |
| production--dino_global--legacy--chronological--budget-10 | 1 | 73.23 | 98.95 | 84.17 | 67.27 | 50.52 | 16.75 | 69.31 | 1.26 | 18.01 | 25.17 |
| production--dino_global--legacy--chronological--budget-10 | 2 | 75.86 | 99.50 | 86.09 | 79.10 | 61.25 | 17.84 | 57.49 | 1.25 | 19.09 | 11.90 |
| production--dino_global--legacy--chronological--budget-10 | 3 | 78.22 | 99.66 | 87.65 | 90.53 | 72.05 | 18.48 | 46.07 | 1.24 | 19.71 | 8.08 |
| production--dino_global--legacy--chronological--budget-20 | 0 | 74.72 | 97.18 | 84.48 | 51.75 | 39.79 | 11.96 | 84.96 | 1.12 | 13.08 | 67.22 |
| production--dino_global--legacy--chronological--budget-20 | 1 | 77.83 | 99.14 | 87.20 | 63.48 | 50.52 | 12.96 | 73.24 | 1.11 | 14.07 | 20.54 |
| production--dino_global--legacy--chronological--budget-20 | 2 | 80.04 | 99.60 | 88.76 | 75.10 | 61.25 | 13.84 | 61.60 | 1.14 | 14.99 | 9.51 |
| production--dino_global--legacy--chronological--budget-20 | 3 | 81.82 | 99.73 | 89.90 | 86.60 | 72.05 | 14.55 | 50.04 | 1.19 | 15.74 | 6.37 |
| production--dino_global--legacy--chronological--budget-40 | 0 | 83.39 | 97.94 | 90.08 | 46.73 | 39.79 | 6.94 | 90.28 | 0.82 | 7.76 | 49.20 |
| production--dino_global--legacy--chronological--budget-40 | 1 | 85.76 | 99.45 | 92.10 | 57.90 | 50.52 | 7.38 | 79.07 | 0.87 | 8.25 | 13.14 |
| production--dino_global--legacy--chronological--budget-40 | 2 | 87.44 | 99.77 | 93.20 | 68.99 | 61.25 | 7.74 | 67.92 | 0.93 | 8.66 | 5.50 |
| production--dino_global--legacy--chronological--budget-40 | 3 | 88.44 | 99.85 | 93.80 | 80.38 | 72.05 | 8.32 | 56.49 | 0.97 | 9.29 | 3.70 |
| production--dino_global--legacy--evidence--budget-05 | 0 | 67.74 | 96.17 | 79.49 | 56.49 | 39.79 | 16.70 | 79.82 | 1.52 | 18.23 | 91.47 |
| production--dino_global--legacy--evidence--budget-05 | 1 | 71.40 | 98.58 | 82.82 | 68.59 | 50.52 | 18.07 | 67.70 | 1.54 | 19.62 | 34.01 |
| production--dino_global--legacy--evidence--budget-05 | 2 | 74.18 | 99.36 | 84.94 | 80.54 | 61.25 | 19.28 | 55.79 | 1.51 | 20.79 | 15.39 |
| production--dino_global--legacy--evidence--budget-05 | 3 | 76.53 | 99.55 | 86.54 | 92.17 | 72.05 | 20.12 | 44.15 | 1.51 | 21.63 | 10.77 |
| production--dino_global--legacy--evidence--budget-10 | 0 | 69.84 | 96.59 | 81.07 | 55.02 | 39.79 | 15.24 | 81.45 | 1.36 | 16.59 | 81.41 |
| production--dino_global--legacy--evidence--budget-10 | 1 | 73.29 | 98.75 | 84.13 | 66.99 | 50.52 | 16.47 | 69.42 | 1.43 | 17.90 | 29.79 |
| production--dino_global--legacy--evidence--budget-10 | 2 | 75.79 | 99.42 | 86.01 | 78.95 | 61.25 | 17.69 | 57.47 | 1.42 | 19.11 | 13.86 |
| production--dino_global--legacy--evidence--budget-10 | 3 | 77.88 | 99.58 | 87.40 | 90.68 | 72.05 | 18.63 | 45.72 | 1.43 | 20.06 | 10.06 |
| production--dino_global--legacy--evidence--budget-20 | 0 | 74.82 | 97.05 | 84.50 | 51.61 | 39.79 | 11.83 | 85.05 | 1.17 | 13.00 | 70.36 |
| production--dino_global--legacy--evidence--budget-20 | 1 | 77.60 | 98.98 | 87.00 | 63.47 | 50.52 | 12.95 | 73.10 | 1.26 | 14.22 | 24.27 |
| production--dino_global--legacy--evidence--budget-20 | 2 | 79.65 | 99.50 | 88.48 | 75.26 | 61.25 | 14.00 | 61.27 | 1.31 | 15.31 | 11.89 |
| production--dino_global--legacy--evidence--budget-20 | 3 | 81.26 | 99.62 | 89.51 | 86.99 | 72.05 | 14.94 | 49.48 | 1.36 | 16.30 | 9.01 |
| production--dino_global--legacy--evidence--budget-40 | 0 | 83.76 | 97.69 | 90.19 | 46.41 | 39.79 | 6.62 | 90.51 | 0.92 | 7.54 | 55.21 |
| production--dino_global--legacy--evidence--budget-40 | 1 | 85.75 | 99.29 | 92.03 | 57.77 | 50.52 | 7.25 | 79.08 | 0.98 | 8.23 | 16.93 |
| production--dino_global--legacy--evidence--budget-40 | 2 | 87.28 | 99.69 | 93.07 | 69.02 | 61.25 | 7.76 | 67.80 | 1.02 | 8.78 | 7.39 |
| production--dino_global--legacy--evidence--budget-40 | 3 | 88.02 | 99.78 | 93.53 | 80.66 | 72.05 | 8.60 | 56.12 | 1.06 | 9.66 | 5.24 |
| production--dino_global--local_events--chronological--budget-05 | 0 | 67.25 | 96.03 | 79.10 | 56.82 | 39.79 | 17.03 | 79.44 | 1.58 | 18.61 | 94.66 |
| production--dino_global--local_events--chronological--budget-05 | 1 | 71.05 | 98.39 | 82.52 | 68.91 | 50.52 | 18.39 | 67.37 | 1.55 | 19.95 | 38.38 |
| production--dino_global--local_events--chronological--budget-05 | 2 | 74.09 | 99.27 | 84.85 | 80.71 | 61.25 | 19.46 | 55.67 | 1.45 | 20.91 | 17.33 |
| production--dino_global--local_events--chronological--budget-05 | 3 | 76.58 | 99.49 | 86.54 | 92.20 | 72.05 | 20.14 | 44.18 | 1.46 | 21.60 | 12.18 |
| production--dino_global--local_events--chronological--budget-10 | 0 | 69.00 | 96.37 | 80.42 | 55.57 | 39.79 | 15.78 | 80.82 | 1.44 | 17.22 | 86.54 |
| production--dino_global--local_events--chronological--budget-10 | 1 | 72.53 | 98.55 | 83.56 | 67.65 | 50.52 | 17.13 | 68.73 | 1.45 | 18.58 | 34.61 |
| production--dino_global--local_events--chronological--budget-10 | 2 | 75.28 | 99.27 | 85.63 | 79.48 | 61.25 | 18.23 | 56.94 | 1.41 | 19.64 | 17.33 |
| production--dino_global--local_events--chronological--budget-10 | 3 | 77.64 | 99.49 | 87.22 | 90.98 | 72.05 | 18.92 | 45.44 | 1.42 | 20.34 | 12.18 |
| production--dino_global--local_events--chronological--budget-20 | 0 | 72.62 | 97.20 | 83.13 | 53.25 | 39.79 | 13.46 | 83.47 | 1.11 | 14.58 | 66.87 |
| production--dino_global--local_events--chronological--budget-20 | 1 | 75.65 | 99.00 | 85.77 | 65.33 | 50.52 | 14.81 | 71.41 | 1.09 | 15.91 | 23.78 |
| production--dino_global--local_events--chronological--budget-20 | 2 | 77.92 | 99.47 | 87.39 | 77.20 | 61.25 | 15.95 | 59.53 | 1.10 | 17.05 | 12.58 |
| production--dino_global--local_events--chronological--budget-20 | 3 | 79.92 | 99.65 | 88.70 | 88.82 | 72.05 | 16.77 | 47.94 | 1.07 | 17.84 | 8.43 |
| production--dino_global--local_events--chronological--budget-40 | 0 | 81.20 | 97.99 | 88.81 | 48.02 | 39.79 | 8.23 | 89.02 | 0.80 | 9.03 | 47.89 |
| production--dino_global--local_events--chronological--budget-40 | 1 | 83.48 | 99.30 | 90.71 | 59.58 | 50.52 | 9.06 | 77.47 | 0.78 | 9.84 | 16.63 |
| production--dino_global--local_events--chronological--budget-40 | 2 | 85.07 | 99.60 | 91.76 | 71.07 | 61.25 | 9.82 | 65.97 | 0.79 | 10.61 | 9.45 |
| production--dino_global--local_events--chronological--budget-40 | 3 | 86.51 | 99.73 | 92.65 | 82.33 | 72.05 | 10.27 | 54.67 | 0.83 | 11.11 | 6.38 |
| production--dino_global--local_events--evidence--budget-05 | 0 | 67.35 | 97.03 | 79.51 | 57.32 | 39.79 | 17.53 | 79.34 | 1.18 | 18.71 | 70.93 |
| production--dino_global--local_events--evidence--budget-05 | 1 | 70.86 | 99.13 | 82.64 | 69.63 | 50.52 | 19.11 | 67.02 | 1.18 | 20.29 | 20.71 |
| production--dino_global--local_events--evidence--budget-05 | 2 | 73.62 | 99.60 | 84.66 | 81.55 | 61.25 | 20.30 | 55.07 | 1.21 | 21.51 | 9.65 |
| production--dino_global--local_events--evidence--budget-05 | 3 | 75.90 | 99.69 | 86.18 | 93.34 | 72.05 | 21.28 | 43.29 | 1.21 | 22.49 | 7.51 |
| production--dino_global--local_events--evidence--budget-10 | 0 | 68.90 | 97.58 | 80.77 | 56.35 | 39.79 | 16.56 | 80.52 | 0.96 | 17.52 | 57.80 |
| production--dino_global--local_events--evidence--budget-10 | 1 | 72.14 | 99.34 | 83.58 | 68.66 | 50.52 | 18.14 | 68.18 | 0.99 | 19.13 | 15.85 |
| production--dino_global--local_events--evidence--budget-10 | 2 | 74.81 | 99.73 | 85.49 | 80.54 | 61.25 | 19.28 | 56.29 | 1.00 | 20.29 | 6.56 |
| production--dino_global--local_events--evidence--budget-10 | 3 | 76.91 | 99.79 | 86.87 | 92.42 | 72.05 | 20.36 | 44.44 | 0.97 | 21.34 | 5.01 |
| production--dino_global--local_events--evidence--budget-20 | 0 | 73.36 | 98.06 | 83.93 | 53.18 | 39.79 | 13.40 | 83.88 | 0.77 | 14.17 | 46.29 |
| production--dino_global--local_events--evidence--budget-20 | 1 | 76.10 | 99.48 | 86.23 | 65.34 | 50.52 | 14.82 | 71.70 | 0.80 | 15.62 | 12.33 |
| production--dino_global--local_events--evidence--budget-20 | 2 | 78.42 | 99.81 | 87.83 | 77.08 | 61.25 | 15.83 | 59.94 | 0.81 | 16.64 | 4.58 |
| production--dino_global--local_events--evidence--budget-20 | 3 | 80.13 | 99.86 | 88.91 | 88.97 | 72.05 | 16.91 | 48.10 | 0.77 | 17.68 | 3.34 |
| production--dino_global--local_events--evidence--budget-40 | 0 | 84.97 | 98.48 | 91.23 | 46.12 | 39.79 | 6.33 | 91.11 | 0.61 | 6.93 | 36.30 |
| production--dino_global--local_events--evidence--budget-40 | 1 | 86.80 | 99.63 | 92.77 | 57.48 | 50.52 | 6.96 | 79.72 | 0.63 | 7.59 | 8.92 |
| production--dino_global--local_events--evidence--budget-40 | 2 | 88.17 | 99.87 | 93.66 | 68.73 | 61.25 | 7.48 | 68.44 | 0.66 | 8.14 | 3.03 |
| production--dino_global--local_events--evidence--budget-40 | 3 | 88.87 | 99.91 | 94.07 | 80.32 | 72.05 | 8.27 | 56.83 | 0.68 | 8.95 | 2.11 |
| production--dino_global--local_heads--chronological--budget-05 | 0 | 67.26 | 96.04 | 79.11 | 56.81 | 39.79 | 17.02 | 79.45 | 1.58 | 18.60 | 94.63 |
| production--dino_global--local_heads--chronological--budget-05 | 1 | 71.04 | 98.39 | 82.51 | 68.93 | 50.52 | 18.41 | 67.35 | 1.55 | 19.96 | 38.38 |
| production--dino_global--local_heads--chronological--budget-05 | 2 | 74.08 | 99.27 | 84.85 | 80.72 | 61.25 | 19.47 | 55.66 | 1.45 | 20.92 | 17.33 |
| production--dino_global--local_heads--chronological--budget-05 | 3 | 76.57 | 99.49 | 86.54 | 92.21 | 72.05 | 20.15 | 44.17 | 1.45 | 21.61 | 12.18 |
| production--dino_global--local_heads--chronological--budget-10 | 0 | 69.02 | 96.38 | 80.43 | 55.56 | 39.79 | 15.77 | 80.83 | 1.44 | 17.21 | 86.51 |
| production--dino_global--local_heads--chronological--budget-10 | 1 | 72.52 | 98.55 | 83.55 | 67.66 | 50.52 | 17.14 | 68.72 | 1.45 | 18.59 | 34.61 |
| production--dino_global--local_heads--chronological--budget-10 | 2 | 75.23 | 99.27 | 85.60 | 79.54 | 61.25 | 18.28 | 56.88 | 1.41 | 19.70 | 17.33 |
| production--dino_global--local_heads--chronological--budget-10 | 3 | 77.60 | 99.49 | 87.19 | 91.03 | 72.05 | 18.98 | 45.39 | 1.42 | 20.39 | 12.18 |
| production--dino_global--local_heads--chronological--budget-20 | 0 | 72.70 | 97.16 | 83.17 | 53.17 | 39.79 | 13.38 | 83.53 | 1.13 | 14.51 | 67.73 |
| production--dino_global--local_heads--chronological--budget-20 | 1 | 75.68 | 98.99 | 85.78 | 65.29 | 50.52 | 14.77 | 71.43 | 1.11 | 15.88 | 24.00 |
| production--dino_global--local_heads--chronological--budget-20 | 2 | 77.94 | 99.47 | 87.40 | 77.16 | 61.25 | 15.91 | 59.56 | 1.12 | 17.02 | 12.58 |
| production--dino_global--local_heads--chronological--budget-20 | 3 | 79.90 | 99.65 | 88.68 | 88.83 | 72.05 | 16.78 | 47.92 | 1.08 | 17.86 | 8.43 |
| production--dino_global--local_heads--chronological--budget-40 | 0 | 81.12 | 98.00 | 88.77 | 48.07 | 39.79 | 8.28 | 88.97 | 0.79 | 9.08 | 47.64 |
| production--dino_global--local_heads--chronological--budget-40 | 1 | 83.39 | 99.30 | 90.65 | 59.65 | 50.52 | 9.13 | 77.41 | 0.77 | 9.91 | 16.67 |
| production--dino_global--local_heads--chronological--budget-40 | 2 | 84.96 | 99.60 | 91.70 | 71.17 | 61.25 | 9.92 | 65.88 | 0.79 | 10.71 | 9.45 |
| production--dino_global--local_heads--chronological--budget-40 | 3 | 86.36 | 99.73 | 92.57 | 82.47 | 72.05 | 10.42 | 54.53 | 0.83 | 11.25 | 6.38 |
| production--dino_global--local_heads--evidence--budget-05 | 0 | 67.31 | 96.82 | 79.41 | 57.23 | 39.79 | 17.44 | 79.34 | 1.27 | 18.71 | 75.90 |
| production--dino_global--local_heads--evidence--budget-05 | 1 | 70.75 | 99.04 | 82.54 | 69.59 | 50.52 | 19.08 | 66.96 | 1.28 | 20.35 | 22.97 |
| production--dino_global--local_heads--evidence--budget-05 | 2 | 73.50 | 99.54 | 84.56 | 81.54 | 61.25 | 20.29 | 54.97 | 1.32 | 21.61 | 11.05 |
| production--dino_global--local_heads--evidence--budget-05 | 3 | 75.81 | 99.64 | 86.11 | 93.31 | 72.05 | 21.25 | 43.21 | 1.32 | 22.57 | 8.58 |
| production--dino_global--local_heads--evidence--budget-10 | 0 | 69.11 | 97.48 | 80.88 | 56.12 | 39.79 | 16.33 | 80.71 | 1.00 | 17.34 | 60.17 |
| production--dino_global--local_heads--evidence--budget-10 | 1 | 72.17 | 99.28 | 83.58 | 68.54 | 50.52 | 18.02 | 68.24 | 1.05 | 19.07 | 17.22 |
| production--dino_global--local_heads--evidence--budget-10 | 2 | 74.72 | 99.70 | 85.42 | 80.55 | 61.25 | 19.30 | 56.22 | 1.06 | 20.36 | 7.22 |
| production--dino_global--local_heads--evidence--budget-10 | 3 | 76.89 | 99.78 | 86.85 | 92.36 | 72.05 | 20.31 | 44.44 | 1.03 | 21.34 | 5.33 |
| production--dino_global--local_heads--evidence--budget-20 | 0 | 71.97 | 97.89 | 82.95 | 54.12 | 39.79 | 14.33 | 82.88 | 0.84 | 15.17 | 50.30 |
| production--dino_global--local_heads--evidence--budget-20 | 1 | 74.46 | 99.40 | 85.14 | 66.67 | 50.52 | 16.15 | 70.28 | 0.88 | 17.03 | 14.30 |
| production--dino_global--local_heads--evidence--budget-20 | 2 | 76.69 | 99.76 | 86.71 | 78.71 | 61.25 | 17.46 | 58.23 | 0.89 | 18.35 | 5.72 |
| production--dino_global--local_heads--evidence--budget-20 | 3 | 78.60 | 99.83 | 87.95 | 90.57 | 72.05 | 18.52 | 46.39 | 0.87 | 19.39 | 4.10 |
| production--dino_global--local_heads--evidence--budget-40 | 0 | 77.72 | 98.62 | 86.93 | 50.49 | 39.79 | 10.70 | 86.80 | 0.55 | 11.25 | 33.06 |
| production--dino_global--local_heads--evidence--budget-40 | 1 | 79.14 | 99.54 | 88.17 | 63.10 | 50.52 | 12.58 | 74.15 | 0.58 | 13.17 | 10.97 |
| production--dino_global--local_heads--evidence--budget-40 | 2 | 80.77 | 99.81 | 89.29 | 75.11 | 61.25 | 13.85 | 62.14 | 0.59 | 14.44 | 4.53 |
| production--dino_global--local_heads--evidence--budget-40 | 3 | 82.20 | 99.86 | 90.17 | 86.98 | 72.05 | 14.92 | 50.29 | 0.56 | 15.49 | 3.34 |
| production--dino_boost--legacy--chronological--budget-05 | 0 | 67.42 | 96.25 | 79.29 | 56.80 | 39.79 | 17.02 | 79.54 | 1.49 | 18.51 | 89.45 |
| production--dino_boost--legacy--chronological--budget-05 | 1 | 71.23 | 98.58 | 82.70 | 68.88 | 50.52 | 18.36 | 67.50 | 1.46 | 19.81 | 33.93 |
| production--dino_boost--legacy--chronological--budget-05 | 2 | 74.23 | 99.43 | 85.00 | 80.71 | 61.25 | 19.45 | 55.78 | 1.35 | 20.80 | 13.66 |
| production--dino_boost--legacy--chronological--budget-05 | 3 | 76.80 | 99.60 | 86.73 | 92.07 | 72.05 | 20.02 | 44.42 | 1.35 | 21.36 | 9.52 |
| production--dino_boost--legacy--chronological--budget-10 | 0 | 69.54 | 96.85 | 80.95 | 55.41 | 39.79 | 15.62 | 81.17 | 1.25 | 16.88 | 75.21 |
| production--dino_boost--legacy--chronological--budget-10 | 1 | 73.13 | 98.95 | 84.10 | 67.38 | 50.52 | 16.86 | 69.21 | 1.24 | 18.11 | 25.16 |
| production--dino_boost--legacy--chronological--budget-10 | 2 | 75.76 | 99.49 | 86.01 | 79.22 | 61.25 | 17.97 | 57.37 | 1.24 | 19.21 | 12.26 |
| production--dino_boost--legacy--chronological--budget-10 | 3 | 78.07 | 99.65 | 87.55 | 90.71 | 72.05 | 18.65 | 45.89 | 1.24 | 19.89 | 8.45 |
| production--dino_boost--legacy--chronological--budget-20 | 0 | 74.39 | 97.26 | 84.30 | 52.02 | 39.79 | 12.23 | 84.72 | 1.09 | 13.32 | 65.37 |
| production--dino_boost--legacy--chronological--budget-20 | 1 | 77.66 | 99.22 | 87.12 | 63.68 | 50.52 | 13.16 | 73.09 | 1.07 | 14.23 | 18.64 |
| production--dino_boost--legacy--chronological--budget-20 | 2 | 79.93 | 99.64 | 88.70 | 75.27 | 61.25 | 14.02 | 61.47 | 1.09 | 15.11 | 8.61 |
| production--dino_boost--legacy--chronological--budget-20 | 3 | 81.74 | 99.76 | 89.86 | 86.77 | 72.05 | 14.71 | 49.94 | 1.13 | 15.84 | 5.80 |
| production--dino_boost--legacy--chronological--budget-40 | 0 | 83.36 | 97.92 | 90.05 | 46.74 | 39.79 | 6.95 | 90.27 | 0.83 | 7.78 | 49.65 |
| production--dino_boost--legacy--chronological--budget-40 | 1 | 85.94 | 99.46 | 92.21 | 57.79 | 50.52 | 7.27 | 79.19 | 0.86 | 8.13 | 12.89 |
| production--dino_boost--legacy--chronological--budget-40 | 2 | 87.64 | 99.79 | 93.32 | 68.87 | 61.25 | 7.62 | 68.06 | 0.90 | 8.52 | 4.92 |
| production--dino_boost--legacy--chronological--budget-40 | 3 | 88.72 | 99.89 | 93.98 | 80.18 | 72.05 | 8.13 | 56.74 | 0.92 | 9.04 | 2.65 |
| production--dino_boost--legacy--evidence--budget-05 | 0 | 67.76 | 96.23 | 79.53 | 56.50 | 39.79 | 16.72 | 79.83 | 1.50 | 18.21 | 89.94 |
| production--dino_boost--legacy--evidence--budget-05 | 1 | 71.49 | 98.63 | 82.89 | 68.54 | 50.52 | 18.03 | 67.77 | 1.52 | 19.54 | 32.64 |
| production--dino_boost--legacy--evidence--budget-05 | 2 | 74.33 | 99.40 | 85.06 | 80.41 | 61.25 | 19.16 | 55.94 | 1.49 | 20.64 | 14.35 |
| production--dino_boost--legacy--evidence--budget-05 | 3 | 76.69 | 99.58 | 86.65 | 92.02 | 72.05 | 19.96 | 44.33 | 1.49 | 21.45 | 10.06 |
| production--dino_boost--legacy--evidence--budget-10 | 0 | 69.94 | 96.52 | 81.11 | 54.90 | 39.79 | 15.12 | 81.54 | 1.39 | 16.50 | 83.14 |
| production--dino_boost--legacy--evidence--budget-10 | 1 | 73.50 | 98.72 | 84.27 | 66.75 | 50.52 | 16.23 | 69.63 | 1.45 | 17.69 | 30.45 |
| production--dino_boost--legacy--evidence--budget-10 | 2 | 76.16 | 99.45 | 86.26 | 78.55 | 61.25 | 17.30 | 57.85 | 1.43 | 18.73 | 13.25 |
| production--dino_boost--legacy--evidence--budget-10 | 3 | 78.24 | 99.60 | 87.64 | 90.26 | 72.05 | 18.20 | 46.14 | 1.44 | 19.64 | 9.53 |
| production--dino_boost--legacy--evidence--budget-20 | 0 | 74.94 | 96.85 | 84.50 | 51.42 | 39.79 | 11.63 | 85.16 | 1.25 | 12.89 | 75.21 |
| production--dino_boost--legacy--evidence--budget-20 | 1 | 78.09 | 98.91 | 87.28 | 63.03 | 50.52 | 12.51 | 73.51 | 1.30 | 13.81 | 26.04 |
| production--dino_boost--legacy--evidence--budget-20 | 2 | 80.48 | 99.58 | 89.02 | 74.55 | 61.25 | 13.30 | 62.03 | 1.25 | 14.55 | 10.08 |
| production--dino_boost--legacy--evidence--budget-20 | 3 | 82.03 | 99.71 | 90.01 | 86.28 | 72.05 | 14.22 | 50.28 | 1.28 | 15.50 | 7.03 |
| production--dino_boost--legacy--evidence--budget-40 | 0 | 83.65 | 97.75 | 90.15 | 46.50 | 39.79 | 6.71 | 90.44 | 0.89 | 7.60 | 53.62 |
| production--dino_boost--legacy--evidence--budget-40 | 1 | 85.92 | 99.35 | 92.15 | 57.71 | 50.52 | 7.19 | 79.19 | 0.94 | 8.13 | 15.55 |
| production--dino_boost--legacy--evidence--budget-40 | 2 | 87.48 | 99.73 | 93.20 | 68.92 | 61.25 | 7.66 | 67.95 | 0.97 | 8.63 | 6.55 |
| production--dino_boost--legacy--evidence--budget-40 | 3 | 88.22 | 99.83 | 93.67 | 80.59 | 72.05 | 8.54 | 56.28 | 0.96 | 9.50 | 4.01 |
| production--dino_boost--local_events--chronological--budget-05 | 0 | 67.17 | 96.03 | 79.05 | 56.88 | 39.79 | 17.10 | 79.37 | 1.58 | 18.67 | 94.66 |
| production--dino_boost--local_events--chronological--budget-05 | 1 | 70.97 | 98.39 | 82.46 | 68.99 | 50.52 | 18.48 | 67.29 | 1.55 | 20.03 | 38.38 |
| production--dino_boost--local_events--chronological--budget-05 | 2 | 74.02 | 99.27 | 84.80 | 80.79 | 61.25 | 19.54 | 55.59 | 1.45 | 20.99 | 17.33 |
| production--dino_boost--local_events--chronological--budget-05 | 3 | 76.51 | 99.49 | 86.50 | 92.27 | 72.05 | 20.22 | 44.10 | 1.46 | 21.67 | 12.18 |
| production--dino_boost--local_events--chronological--budget-10 | 0 | 68.77 | 96.42 | 80.28 | 55.78 | 39.79 | 15.99 | 80.63 | 1.43 | 17.42 | 85.54 |
| production--dino_boost--local_events--chronological--budget-10 | 1 | 72.35 | 98.60 | 83.46 | 67.87 | 50.52 | 17.35 | 68.55 | 1.41 | 18.77 | 33.53 |
| production--dino_boost--local_events--chronological--budget-10 | 2 | 75.13 | 99.32 | 85.55 | 79.72 | 61.25 | 18.47 | 56.76 | 1.35 | 19.82 | 16.24 |
| production--dino_boost--local_events--chronological--budget-10 | 3 | 77.53 | 99.54 | 87.17 | 91.22 | 72.05 | 19.16 | 45.29 | 1.33 | 20.49 | 11.09 |
| production--dino_boost--local_events--chronological--budget-20 | 0 | 72.36 | 97.21 | 82.96 | 53.45 | 39.79 | 13.66 | 83.27 | 1.11 | 14.77 | 66.69 |
| production--dino_boost--local_events--chronological--budget-20 | 1 | 75.47 | 99.01 | 85.65 | 65.50 | 50.52 | 14.98 | 71.25 | 1.09 | 16.07 | 23.55 |
| production--dino_boost--local_events--chronological--budget-20 | 2 | 77.78 | 99.47 | 87.30 | 77.34 | 61.25 | 16.09 | 59.39 | 1.10 | 17.19 | 12.58 |
| production--dino_boost--local_events--chronological--budget-20 | 3 | 79.76 | 99.65 | 88.60 | 89.01 | 72.05 | 16.95 | 47.76 | 1.07 | 18.02 | 8.43 |
| production--dino_boost--local_events--chronological--budget-40 | 0 | 80.99 | 98.00 | 88.69 | 48.15 | 39.79 | 8.36 | 88.89 | 0.79 | 9.16 | 47.64 |
| production--dino_boost--local_events--chronological--budget-40 | 1 | 83.25 | 99.30 | 90.57 | 59.76 | 50.52 | 9.24 | 77.30 | 0.77 | 10.01 | 16.60 |
| production--dino_boost--local_events--chronological--budget-40 | 2 | 84.82 | 99.60 | 91.62 | 71.29 | 61.25 | 10.04 | 65.76 | 0.79 | 10.83 | 9.45 |
| production--dino_boost--local_events--chronological--budget-40 | 3 | 86.23 | 99.73 | 92.49 | 82.60 | 72.05 | 10.54 | 54.41 | 0.83 | 11.37 | 6.38 |
| production--dino_boost--local_events--evidence--budget-05 | 0 | 67.17 | 96.78 | 79.30 | 57.32 | 39.79 | 17.54 | 79.23 | 1.28 | 18.82 | 76.94 |
| production--dino_boost--local_events--evidence--budget-05 | 1 | 70.71 | 98.99 | 82.49 | 69.61 | 50.52 | 19.09 | 66.93 | 1.30 | 20.39 | 24.18 |
| production--dino_boost--local_events--evidence--budget-05 | 2 | 73.56 | 99.53 | 84.59 | 81.46 | 61.25 | 20.20 | 55.04 | 1.34 | 21.54 | 11.23 |
| production--dino_boost--local_events--evidence--budget-05 | 3 | 75.88 | 99.65 | 86.16 | 93.18 | 72.05 | 21.13 | 43.31 | 1.35 | 22.47 | 8.27 |
| production--dino_boost--local_events--evidence--budget-10 | 0 | 69.05 | 97.33 | 80.79 | 56.08 | 39.79 | 16.29 | 80.69 | 1.06 | 17.36 | 63.79 |
| production--dino_boost--local_events--evidence--budget-10 | 1 | 72.31 | 99.26 | 83.67 | 68.36 | 50.52 | 17.84 | 68.39 | 1.09 | 18.93 | 17.63 |
| production--dino_boost--local_events--evidence--budget-10 | 2 | 74.98 | 99.73 | 85.60 | 80.22 | 61.25 | 18.97 | 56.51 | 1.10 | 20.07 | 6.44 |
| production--dino_boost--local_events--evidence--budget-10 | 3 | 77.14 | 99.81 | 87.02 | 92.00 | 72.05 | 19.95 | 44.75 | 1.08 | 21.03 | 4.64 |
| production--dino_boost--local_events--evidence--budget-20 | 0 | 73.05 | 98.07 | 83.73 | 53.41 | 39.79 | 13.62 | 83.65 | 0.77 | 14.39 | 46.15 |
| production--dino_boost--local_events--evidence--budget-20 | 1 | 75.80 | 99.48 | 86.04 | 65.57 | 50.52 | 15.05 | 71.44 | 0.82 | 15.87 | 12.41 |
| production--dino_boost--local_events--evidence--budget-20 | 2 | 78.12 | 99.81 | 87.64 | 77.34 | 61.25 | 16.09 | 59.66 | 0.83 | 16.92 | 4.62 |
| production--dino_boost--local_events--evidence--budget-20 | 3 | 79.89 | 99.84 | 88.76 | 89.16 | 72.05 | 17.10 | 47.85 | 0.82 | 17.93 | 3.73 |
| production--dino_boost--local_events--evidence--budget-40 | 0 | 84.25 | 98.58 | 90.86 | 46.55 | 39.79 | 6.77 | 90.72 | 0.56 | 7.33 | 33.83 |
| production--dino_boost--local_events--evidence--budget-40 | 1 | 86.13 | 99.65 | 92.40 | 57.98 | 50.52 | 7.46 | 79.27 | 0.58 | 8.04 | 8.40 |
| production--dino_boost--local_events--evidence--budget-40 | 2 | 87.48 | 99.92 | 93.29 | 69.34 | 61.25 | 8.09 | 67.90 | 0.59 | 8.68 | 1.99 |
| production--dino_boost--local_events--evidence--budget-40 | 3 | 88.19 | 99.96 | 93.71 | 81.03 | 72.05 | 8.98 | 56.21 | 0.59 | 9.57 | 0.97 |
| production--dino_boost--local_heads--chronological--budget-05 | 0 | 67.22 | 96.03 | 79.08 | 56.84 | 39.79 | 17.06 | 79.41 | 1.58 | 18.64 | 94.66 |
| production--dino_boost--local_heads--chronological--budget-05 | 1 | 71.01 | 98.39 | 82.49 | 68.96 | 50.52 | 18.44 | 67.32 | 1.55 | 19.99 | 38.38 |
| production--dino_boost--local_heads--chronological--budget-05 | 2 | 74.05 | 99.27 | 84.83 | 80.75 | 61.25 | 19.50 | 55.62 | 1.45 | 20.96 | 17.33 |
| production--dino_boost--local_heads--chronological--budget-05 | 3 | 76.54 | 99.49 | 86.52 | 92.24 | 72.05 | 20.18 | 44.14 | 1.46 | 21.64 | 12.18 |
| production--dino_boost--local_heads--chronological--budget-10 | 0 | 68.79 | 96.42 | 80.29 | 55.76 | 39.79 | 15.98 | 80.64 | 1.43 | 17.40 | 85.54 |
| production--dino_boost--local_heads--chronological--budget-10 | 1 | 72.36 | 98.60 | 83.46 | 67.87 | 50.52 | 17.35 | 68.55 | 1.41 | 18.76 | 33.53 |
| production--dino_boost--local_heads--chronological--budget-10 | 2 | 75.12 | 99.32 | 85.54 | 79.73 | 61.25 | 18.48 | 56.75 | 1.35 | 19.83 | 16.24 |
| production--dino_boost--local_heads--chronological--budget-10 | 3 | 77.50 | 99.54 | 87.15 | 91.25 | 72.05 | 19.19 | 45.25 | 1.33 | 20.53 | 11.09 |
| production--dino_boost--local_heads--chronological--budget-20 | 0 | 72.45 | 97.18 | 83.01 | 53.37 | 39.79 | 13.58 | 83.34 | 1.12 | 14.70 | 67.30 |
| production--dino_boost--local_heads--chronological--budget-20 | 1 | 75.50 | 99.00 | 85.67 | 65.45 | 50.52 | 14.94 | 71.28 | 1.10 | 16.04 | 23.78 |
| production--dino_boost--local_heads--chronological--budget-20 | 2 | 77.77 | 99.47 | 87.30 | 77.33 | 61.25 | 16.08 | 59.39 | 1.11 | 17.19 | 12.58 |
| production--dino_boost--local_heads--chronological--budget-20 | 3 | 79.73 | 99.65 | 88.58 | 89.02 | 72.05 | 16.97 | 47.74 | 1.08 | 18.04 | 8.43 |
| production--dino_boost--local_heads--chronological--budget-40 | 0 | 80.96 | 98.02 | 88.68 | 48.17 | 39.79 | 8.38 | 88.88 | 0.79 | 9.17 | 47.37 |
| production--dino_boost--local_heads--chronological--budget-40 | 1 | 83.21 | 99.30 | 90.55 | 59.79 | 50.52 | 9.27 | 77.28 | 0.77 | 10.04 | 16.60 |
| production--dino_boost--local_heads--chronological--budget-40 | 2 | 84.77 | 99.60 | 91.59 | 71.33 | 61.25 | 10.08 | 65.72 | 0.78 | 10.86 | 9.45 |
| production--dino_boost--local_heads--chronological--budget-40 | 3 | 86.16 | 99.73 | 92.45 | 82.67 | 72.05 | 10.61 | 54.34 | 0.82 | 11.44 | 6.38 |
| production--dino_boost--local_heads--evidence--budget-05 | 0 | 67.22 | 96.84 | 79.36 | 57.31 | 39.79 | 17.53 | 79.26 | 1.26 | 18.79 | 75.55 |
| production--dino_boost--local_heads--evidence--budget-05 | 1 | 70.75 | 98.99 | 82.52 | 69.59 | 50.52 | 19.07 | 66.96 | 1.29 | 20.36 | 24.02 |
| production--dino_boost--local_heads--evidence--budget-05 | 2 | 73.54 | 99.53 | 84.58 | 81.50 | 61.25 | 20.25 | 55.02 | 1.32 | 21.57 | 11.18 |
| production--dino_boost--local_heads--evidence--budget-05 | 3 | 75.87 | 99.67 | 86.16 | 93.23 | 72.05 | 21.18 | 43.28 | 1.32 | 22.50 | 7.89 |
| production--dino_boost--local_heads--evidence--budget-10 | 0 | 69.15 | 97.34 | 80.85 | 56.01 | 39.79 | 16.22 | 80.77 | 1.06 | 17.28 | 63.61 |
| production--dino_boost--local_heads--evidence--budget-10 | 1 | 72.30 | 99.18 | 83.63 | 68.33 | 50.52 | 17.81 | 68.39 | 1.12 | 18.93 | 19.66 |
| production--dino_boost--local_heads--evidence--budget-10 | 2 | 74.91 | 99.65 | 85.53 | 80.25 | 61.25 | 18.99 | 56.44 | 1.14 | 20.14 | 8.39 |
| production--dino_boost--local_heads--evidence--budget-10 | 3 | 77.02 | 99.76 | 86.92 | 92.08 | 72.05 | 20.03 | 44.61 | 1.14 | 21.17 | 5.76 |
| production--dino_boost--local_heads--evidence--budget-20 | 0 | 72.21 | 97.92 | 83.12 | 53.96 | 39.79 | 14.17 | 83.05 | 0.83 | 15.00 | 49.72 |
| production--dino_boost--local_heads--evidence--budget-20 | 1 | 74.89 | 99.40 | 85.42 | 66.28 | 50.52 | 15.77 | 70.67 | 0.88 | 16.65 | 14.28 |
| production--dino_boost--local_heads--evidence--budget-20 | 2 | 77.12 | 99.76 | 86.99 | 78.26 | 61.25 | 17.01 | 58.67 | 0.90 | 17.91 | 5.71 |
| production--dino_boost--local_heads--evidence--budget-20 | 3 | 78.98 | 99.81 | 88.18 | 90.09 | 72.05 | 18.04 | 46.84 | 0.90 | 18.94 | 4.43 |
| production--dino_boost--local_heads--evidence--budget-40 | 0 | 77.78 | 98.70 | 87.00 | 50.49 | 39.79 | 10.71 | 86.82 | 0.52 | 11.22 | 30.99 |
| production--dino_boost--local_heads--evidence--budget-40 | 1 | 79.47 | 99.61 | 88.40 | 62.91 | 50.52 | 12.39 | 74.39 | 0.54 | 12.92 | 9.36 |
| production--dino_boost--local_heads--evidence--budget-40 | 2 | 81.01 | 99.83 | 89.44 | 74.96 | 61.25 | 13.71 | 62.34 | 0.53 | 14.24 | 4.01 |
| production--dino_boost--local_heads--evidence--budget-40 | 3 | 82.30 | 99.88 | 90.24 | 86.94 | 72.05 | 14.88 | 50.39 | 0.51 | 15.39 | 2.87 |
| individual--compact_boost--legacy--chronological--budget-05 | 0 | 84.93 | 83.52 | 84.22 | 39.13 | 39.79 | -0.65 | 92.15 | 6.56 | 5.90 | 393.30 |
| individual--compact_boost--legacy--chronological--budget-05 | 1 | 87.11 | 90.65 | 88.84 | 49.47 | 50.52 | -1.05 | 80.93 | 7.43 | 6.38 | 223.09 |
| individual--compact_boost--legacy--chronological--budget-05 | 2 | 88.95 | 93.27 | 91.05 | 59.46 | 61.25 | -1.79 | 70.01 | 8.37 | 6.58 | 160.73 |
| individual--compact_boost--legacy--chronological--budget-05 | 3 | 90.31 | 94.61 | 92.41 | 69.52 | 72.05 | -2.54 | 59.04 | 9.28 | 6.74 | 128.79 |
| individual--compact_boost--legacy--chronological--budget-10 | 0 | 85.62 | 84.79 | 85.19 | 39.41 | 39.79 | -0.38 | 92.37 | 6.05 | 5.68 | 363.13 |
| individual--compact_boost--legacy--chronological--budget-10 | 1 | 87.63 | 91.30 | 89.42 | 49.80 | 50.52 | -0.72 | 81.15 | 6.89 | 6.17 | 207.74 |
| individual--compact_boost--legacy--chronological--budget-10 | 2 | 89.38 | 93.74 | 91.51 | 59.86 | 61.25 | -1.39 | 70.22 | 7.75 | 6.36 | 149.52 |
| individual--compact_boost--legacy--chronological--budget-10 | 3 | 90.68 | 95.01 | 92.79 | 70.00 | 72.05 | -2.05 | 59.25 | 8.58 | 6.53 | 119.24 |
| individual--compact_boost--legacy--chronological--budget-20 | 0 | 87.39 | 86.78 | 87.08 | 39.52 | 39.79 | -0.27 | 93.06 | 5.26 | 4.99 | 315.47 |
| individual--compact_boost--legacy--chronological--budget-20 | 1 | 89.12 | 92.40 | 90.72 | 50.02 | 50.52 | -0.50 | 81.86 | 5.95 | 5.45 | 181.46 |
| individual--compact_boost--legacy--chronological--budget-20 | 2 | 90.54 | 94.50 | 92.48 | 60.31 | 61.25 | -0.94 | 70.87 | 6.65 | 5.71 | 131.34 |
| individual--compact_boost--legacy--chronological--budget-20 | 3 | 91.73 | 95.59 | 93.62 | 70.57 | 72.05 | -1.49 | 59.94 | 7.33 | 5.84 | 105.22 |
| individual--compact_boost--legacy--chronological--budget-40 | 0 | 90.56 | 91.58 | 91.06 | 40.24 | 39.79 | 0.45 | 94.24 | 3.35 | 3.80 | 201.04 |
| individual--compact_boost--legacy--chronological--budget-40 | 1 | 91.94 | 95.34 | 93.60 | 50.82 | 50.52 | 0.30 | 83.21 | 3.80 | 4.10 | 111.35 |
| individual--compact_boost--legacy--chronological--budget-40 | 2 | 92.93 | 96.65 | 94.75 | 61.29 | 61.25 | 0.04 | 72.24 | 4.30 | 4.34 | 80.03 |
| individual--compact_boost--legacy--chronological--budget-40 | 3 | 93.76 | 97.31 | 95.50 | 71.78 | 72.05 | -0.27 | 61.29 | 4.76 | 4.48 | 64.17 |
| individual--compact_boost--legacy--evidence--budget-05 | 0 | 84.96 | 83.70 | 84.32 | 39.21 | 39.79 | -0.58 | 92.14 | 6.48 | 5.91 | 388.99 |
| individual--compact_boost--legacy--evidence--budget-05 | 1 | 87.11 | 90.81 | 88.91 | 49.55 | 50.52 | -0.97 | 80.92 | 7.36 | 6.40 | 219.50 |
| individual--compact_boost--legacy--evidence--budget-05 | 2 | 88.87 | 93.42 | 91.08 | 59.62 | 61.25 | -1.63 | 69.94 | 8.28 | 6.64 | 157.10 |
| individual--compact_boost--legacy--evidence--budget-05 | 3 | 90.19 | 94.78 | 92.42 | 69.72 | 72.05 | -2.33 | 58.93 | 9.18 | 6.85 | 124.72 |
| individual--compact_boost--legacy--evidence--budget-10 | 0 | 85.75 | 85.26 | 85.50 | 39.57 | 39.79 | -0.22 | 92.40 | 5.87 | 5.65 | 351.93 |
| individual--compact_boost--legacy--evidence--budget-10 | 1 | 87.68 | 91.82 | 89.70 | 49.99 | 50.52 | -0.53 | 81.15 | 6.70 | 6.17 | 195.19 |
| individual--compact_boost--legacy--evidence--budget-10 | 2 | 89.37 | 94.16 | 91.70 | 60.08 | 61.25 | -1.17 | 70.19 | 7.56 | 6.39 | 139.32 |
| individual--compact_boost--legacy--evidence--budget-10 | 3 | 90.64 | 95.42 | 92.97 | 70.22 | 72.05 | -1.83 | 59.20 | 8.41 | 6.58 | 109.41 |
| individual--compact_boost--legacy--evidence--budget-20 | 0 | 87.43 | 88.07 | 87.74 | 40.09 | 39.79 | 0.30 | 93.00 | 4.75 | 5.05 | 284.72 |
| individual--compact_boost--legacy--evidence--budget-20 | 1 | 89.20 | 93.47 | 91.28 | 50.44 | 50.52 | -0.08 | 81.86 | 5.53 | 5.45 | 155.87 |
| individual--compact_boost--legacy--evidence--budget-20 | 2 | 90.64 | 95.44 | 92.97 | 60.64 | 61.25 | -0.61 | 70.90 | 6.29 | 5.69 | 108.87 |
| individual--compact_boost--legacy--evidence--budget-20 | 3 | 91.69 | 96.44 | 94.00 | 70.87 | 72.05 | -1.18 | 59.89 | 7.08 | 5.89 | 84.92 |
| individual--compact_boost--legacy--evidence--budget-40 | 0 | 90.82 | 92.77 | 91.78 | 40.65 | 39.79 | 0.86 | 94.31 | 2.87 | 3.74 | 172.48 |
| individual--compact_boost--legacy--evidence--budget-40 | 1 | 91.99 | 96.12 | 94.01 | 51.21 | 50.52 | 0.69 | 83.21 | 3.42 | 4.11 | 92.62 |
| individual--compact_boost--legacy--evidence--budget-40 | 2 | 93.14 | 97.28 | 95.16 | 61.52 | 61.25 | 0.27 | 72.35 | 3.96 | 4.23 | 64.84 |
| individual--compact_boost--legacy--evidence--budget-40 | 3 | 93.86 | 97.90 | 95.84 | 71.97 | 72.05 | -0.08 | 61.36 | 4.51 | 4.42 | 50.02 |
| individual--compact_boost--local_events--chronological--budget-05 | 0 | 85.05 | 83.98 | 84.51 | 39.30 | 39.79 | -0.49 | 92.16 | 6.37 | 5.88 | 382.31 |
| individual--compact_boost--local_events--chronological--budget-05 | 1 | 87.18 | 90.96 | 89.02 | 49.66 | 50.52 | -0.86 | 80.94 | 7.24 | 6.38 | 215.84 |
| individual--compact_boost--local_events--chronological--budget-05 | 2 | 89.02 | 93.50 | 91.20 | 59.69 | 61.25 | -1.57 | 70.02 | 8.13 | 6.56 | 155.10 |
| individual--compact_boost--local_events--chronological--budget-05 | 3 | 90.35 | 94.80 | 92.52 | 69.81 | 72.05 | -2.25 | 59.04 | 8.99 | 6.74 | 124.17 |
| individual--compact_boost--local_events--chronological--budget-10 | 0 | 86.15 | 85.74 | 85.93 | 39.60 | 39.79 | -0.18 | 92.56 | 5.67 | 5.49 | 340.43 |
| individual--compact_boost--local_events--chronological--budget-10 | 1 | 88.32 | 92.06 | 90.15 | 49.92 | 50.52 | -0.60 | 81.48 | 6.44 | 5.83 | 189.42 |
| individual--compact_boost--local_events--chronological--budget-10 | 2 | 90.02 | 94.28 | 92.10 | 60.01 | 61.25 | -1.24 | 70.59 | 7.23 | 5.99 | 136.43 |
| individual--compact_boost--local_events--chronological--budget-10 | 3 | 91.32 | 95.42 | 93.33 | 70.16 | 72.05 | -1.90 | 59.69 | 7.99 | 6.09 | 109.28 |
| individual--compact_boost--local_events--chronological--budget-20 | 0 | 88.00 | 88.37 | 88.17 | 39.96 | 39.79 | 0.18 | 93.24 | 4.63 | 4.81 | 277.61 |
| individual--compact_boost--local_events--chronological--budget-20 | 1 | 90.06 | 93.92 | 91.95 | 50.33 | 50.52 | -0.19 | 82.31 | 5.20 | 5.01 | 145.10 |
| individual--compact_boost--local_events--chronological--budget-20 | 2 | 91.54 | 95.91 | 93.67 | 60.61 | 61.25 | -0.65 | 71.45 | 5.78 | 5.13 | 97.59 |
| individual--compact_boost--local_events--chronological--budget-20 | 3 | 92.62 | 96.92 | 94.72 | 70.98 | 72.05 | -1.07 | 60.54 | 6.31 | 5.24 | 73.59 |
| individual--compact_boost--local_events--chronological--budget-40 | 0 | 90.18 | 91.96 | 91.05 | 40.58 | 39.79 | 0.80 | 94.05 | 3.20 | 4.00 | 191.97 |
| individual--compact_boost--local_events--chronological--budget-40 | 1 | 92.25 | 96.38 | 94.26 | 50.90 | 50.52 | 0.38 | 83.36 | 3.58 | 3.96 | 86.42 |
| individual--compact_boost--local_events--chronological--budget-40 | 2 | 93.62 | 97.79 | 95.66 | 61.20 | 61.25 | -0.05 | 72.67 | 3.97 | 3.91 | 52.70 |
| individual--compact_boost--local_events--chronological--budget-40 | 3 | 94.53 | 98.38 | 96.41 | 71.63 | 72.05 | -0.43 | 61.85 | 4.35 | 3.93 | 38.71 |
| individual--compact_boost--local_events--evidence--budget-05 | 0 | 85.60 | 83.42 | 84.49 | 38.78 | 39.79 | -1.00 | 92.45 | 6.60 | 5.59 | 395.77 |
| individual--compact_boost--local_events--evidence--budget-05 | 1 | 87.76 | 90.39 | 89.05 | 48.94 | 50.52 | -1.58 | 81.31 | 7.58 | 6.00 | 229.50 |
| individual--compact_boost--local_events--evidence--budget-05 | 2 | 89.43 | 93.02 | 91.19 | 58.91 | 61.25 | -2.35 | 70.35 | 8.58 | 6.23 | 166.64 |
| individual--compact_boost--local_events--evidence--budget-05 | 3 | 90.71 | 94.35 | 92.49 | 68.88 | 72.05 | -3.17 | 59.37 | 9.57 | 6.41 | 134.96 |
| individual--compact_boost--local_events--evidence--budget-10 | 0 | 87.03 | 84.92 | 85.95 | 38.83 | 39.79 | -0.95 | 93.00 | 6.00 | 5.05 | 359.94 |
| individual--compact_boost--local_events--evidence--budget-10 | 1 | 89.13 | 91.14 | 90.12 | 48.76 | 50.52 | -1.76 | 82.01 | 7.06 | 5.31 | 211.55 |
| individual--compact_boost--local_events--evidence--budget-10 | 2 | 90.76 | 93.47 | 92.09 | 58.56 | 61.25 | -2.69 | 71.16 | 8.11 | 5.42 | 155.94 |
| individual--compact_boost--local_events--evidence--budget-10 | 3 | 91.95 | 94.65 | 93.28 | 68.45 | 72.05 | -3.60 | 60.27 | 9.12 | 5.51 | 127.78 |
| individual--compact_boost--local_events--evidence--budget-20 | 0 | 88.95 | 88.04 | 88.48 | 39.39 | 39.79 | -0.40 | 93.69 | 4.76 | 4.36 | 285.50 |
| individual--compact_boost--local_events--evidence--budget-20 | 1 | 90.90 | 93.26 | 92.06 | 49.39 | 50.52 | -1.13 | 82.81 | 5.63 | 4.50 | 160.98 |
| individual--compact_boost--local_events--evidence--budget-20 | 2 | 92.39 | 95.12 | 93.73 | 59.26 | 61.25 | -1.99 | 72.06 | 6.51 | 4.52 | 116.45 |
| individual--compact_boost--local_events--evidence--budget-20 | 3 | 93.39 | 96.00 | 94.68 | 69.26 | 72.05 | -2.79 | 61.20 | 7.37 | 4.58 | 95.52 |
| individual--compact_boost--local_events--evidence--budget-40 | 0 | 90.32 | 91.96 | 91.12 | 40.52 | 39.79 | 0.74 | 94.11 | 3.20 | 3.94 | 191.94 |
| individual--compact_boost--local_events--evidence--budget-40 | 1 | 92.43 | 96.37 | 94.35 | 50.79 | 50.52 | 0.27 | 83.46 | 3.59 | 3.85 | 86.67 |
| individual--compact_boost--local_events--evidence--budget-40 | 2 | 93.81 | 97.77 | 95.75 | 61.05 | 61.25 | -0.20 | 72.79 | 3.99 | 3.79 | 53.14 |
| individual--compact_boost--local_events--evidence--budget-40 | 3 | 94.72 | 98.35 | 96.50 | 71.45 | 72.05 | -0.61 | 62.00 | 4.38 | 3.78 | 39.28 |
| individual--compact_boost--local_heads--chronological--budget-05 | 0 | 85.20 | 84.04 | 84.60 | 39.26 | 39.79 | -0.53 | 92.23 | 6.35 | 5.82 | 381.01 |
| individual--compact_boost--local_heads--chronological--budget-05 | 1 | 87.20 | 90.94 | 89.03 | 49.62 | 50.52 | -0.90 | 80.96 | 7.26 | 6.36 | 216.28 |
| individual--compact_boost--local_heads--chronological--budget-05 | 2 | 88.96 | 93.43 | 91.13 | 59.65 | 61.25 | -1.60 | 69.99 | 8.19 | 6.59 | 156.90 |
| individual--compact_boost--local_heads--chronological--budget-05 | 3 | 90.27 | 94.71 | 92.44 | 69.74 | 72.05 | -2.31 | 58.99 | 9.10 | 6.79 | 126.27 |
| individual--compact_boost--local_heads--chronological--budget-10 | 0 | 86.32 | 85.65 | 85.97 | 39.49 | 39.79 | -0.30 | 92.63 | 5.71 | 5.41 | 342.56 |
| individual--compact_boost--local_heads--chronological--budget-10 | 1 | 88.22 | 91.94 | 90.03 | 49.89 | 50.52 | -0.62 | 81.43 | 6.51 | 5.89 | 192.43 |
| individual--compact_boost--local_heads--chronological--budget-10 | 2 | 89.81 | 94.08 | 91.89 | 60.02 | 61.25 | -1.23 | 70.46 | 7.36 | 6.12 | 141.27 |
| individual--compact_boost--local_heads--chronological--budget-10 | 3 | 91.05 | 95.22 | 93.08 | 70.18 | 72.05 | -1.88 | 59.49 | 8.16 | 6.29 | 114.21 |
| individual--compact_boost--local_heads--chronological--budget-20 | 0 | 88.53 | 87.98 | 88.24 | 39.55 | 39.79 | -0.24 | 93.51 | 4.78 | 4.54 | 286.99 |
| individual--compact_boost--local_heads--chronological--budget-20 | 1 | 90.29 | 93.51 | 91.87 | 49.93 | 50.52 | -0.58 | 82.46 | 5.44 | 4.86 | 154.89 |
| individual--compact_boost--local_heads--chronological--budget-20 | 2 | 91.56 | 95.31 | 93.39 | 60.22 | 61.25 | -1.03 | 71.49 | 6.12 | 5.09 | 111.99 |
| individual--compact_boost--local_heads--chronological--budget-20 | 3 | 92.64 | 96.26 | 94.41 | 70.49 | 72.05 | -1.57 | 60.59 | 6.76 | 5.19 | 89.22 |
| individual--compact_boost--local_heads--chronological--budget-40 | 0 | 91.19 | 92.71 | 91.94 | 40.46 | 39.79 | 0.67 | 94.48 | 2.90 | 3.57 | 173.92 |
| individual--compact_boost--local_heads--chronological--budget-40 | 1 | 92.79 | 96.60 | 94.65 | 50.86 | 50.52 | 0.34 | 83.64 | 3.34 | 3.67 | 81.14 |
| individual--compact_boost--local_heads--chronological--budget-40 | 2 | 93.82 | 97.70 | 95.72 | 61.29 | 61.25 | 0.04 | 72.79 | 3.76 | 3.79 | 54.86 |
| individual--compact_boost--local_heads--chronological--budget-40 | 3 | 94.64 | 98.19 | 96.38 | 71.72 | 72.05 | -0.33 | 61.93 | 4.18 | 3.85 | 43.20 |
| individual--compact_boost--local_heads--evidence--budget-05 | 0 | 85.73 | 83.67 | 84.68 | 38.84 | 39.79 | -0.94 | 92.50 | 6.50 | 5.55 | 389.84 |
| individual--compact_boost--local_heads--evidence--budget-05 | 1 | 87.79 | 90.45 | 89.09 | 48.97 | 50.52 | -1.55 | 81.32 | 7.54 | 5.99 | 228.06 |
| individual--compact_boost--local_heads--evidence--budget-05 | 2 | 89.50 | 92.91 | 91.17 | 58.89 | 61.25 | -2.36 | 70.39 | 8.56 | 6.20 | 169.36 |
| individual--compact_boost--local_heads--evidence--budget-05 | 3 | 90.80 | 94.21 | 92.47 | 68.84 | 72.05 | -3.21 | 59.44 | 9.55 | 6.34 | 138.31 |
| individual--compact_boost--local_heads--evidence--budget-10 | 0 | 86.97 | 85.52 | 86.23 | 39.14 | 39.79 | -0.65 | 92.94 | 5.76 | 5.11 | 345.69 |
| individual--compact_boost--local_heads--evidence--budget-10 | 1 | 89.00 | 91.58 | 90.27 | 49.16 | 50.52 | -1.35 | 81.90 | 6.77 | 5.42 | 201.06 |
| individual--compact_boost--local_heads--evidence--budget-10 | 2 | 90.59 | 93.71 | 92.12 | 59.05 | 61.25 | -2.20 | 71.01 | 7.77 | 5.57 | 150.09 |
| individual--compact_boost--local_heads--evidence--budget-10 | 3 | 91.71 | 94.83 | 93.24 | 69.05 | 72.05 | -3.01 | 60.05 | 8.74 | 5.73 | 123.39 |
| individual--compact_boost--local_heads--evidence--budget-20 | 0 | 88.76 | 88.68 | 88.71 | 39.77 | 39.79 | -0.02 | 93.56 | 4.50 | 4.48 | 270.19 |
| individual--compact_boost--local_heads--evidence--budget-20 | 1 | 90.77 | 93.78 | 92.24 | 49.80 | 50.52 | -0.72 | 82.70 | 5.33 | 4.61 | 148.41 |
| individual--compact_boost--local_heads--evidence--budget-20 | 2 | 92.22 | 95.40 | 93.78 | 59.74 | 61.25 | -1.51 | 71.93 | 6.17 | 4.66 | 109.86 |
| individual--compact_boost--local_heads--evidence--budget-20 | 3 | 93.15 | 96.25 | 94.67 | 69.82 | 72.05 | -2.23 | 60.98 | 7.03 | 4.79 | 89.49 |
| individual--compact_boost--local_heads--evidence--budget-40 | 0 | 91.24 | 92.90 | 92.05 | 40.52 | 39.79 | 0.73 | 94.49 | 2.83 | 3.56 | 169.59 |
| individual--compact_boost--local_heads--evidence--budget-40 | 1 | 92.89 | 96.63 | 94.72 | 50.89 | 50.52 | 0.37 | 83.69 | 3.25 | 3.63 | 80.36 |
| individual--compact_boost--local_heads--evidence--budget-40 | 2 | 94.01 | 97.72 | 95.82 | 61.24 | 61.25 | -0.01 | 72.90 | 3.69 | 3.68 | 54.34 |
| individual--compact_boost--local_heads--evidence--budget-40 | 3 | 94.80 | 98.25 | 96.49 | 71.64 | 72.05 | -0.41 | 62.05 | 4.15 | 3.73 | 41.75 |
| individual--dino_global--legacy--chronological--budget-05 | 0 | 86.36 | 88.38 | 87.33 | 40.75 | 39.79 | 0.97 | 92.46 | 4.62 | 5.59 | 277.32 |
| individual--dino_global--legacy--chronological--budget-05 | 1 | 88.40 | 95.44 | 91.77 | 51.48 | 50.52 | 0.96 | 81.31 | 5.04 | 6.00 | 108.74 |
| individual--dino_global--legacy--chronological--budget-05 | 2 | 89.88 | 97.27 | 93.42 | 62.00 | 61.25 | 0.75 | 70.28 | 5.55 | 6.30 | 65.16 |
| individual--dino_global--legacy--chronological--budget-05 | 3 | 91.07 | 97.86 | 94.33 | 72.52 | 72.05 | 0.46 | 59.27 | 6.04 | 6.51 | 51.14 |
| individual--dino_global--legacy--chronological--budget-10 | 0 | 86.96 | 89.14 | 88.01 | 40.82 | 39.79 | 1.03 | 92.70 | 4.32 | 5.35 | 259.17 |
| individual--dino_global--legacy--chronological--budget-10 | 1 | 88.93 | 95.72 | 92.18 | 51.56 | 50.52 | 1.04 | 81.58 | 4.70 | 5.74 | 102.23 |
| individual--dino_global--legacy--chronological--budget-10 | 2 | 90.35 | 97.44 | 93.75 | 62.09 | 61.25 | 0.84 | 70.56 | 5.18 | 6.02 | 61.05 |
| individual--dino_global--legacy--chronological--budget-10 | 3 | 91.49 | 98.00 | 94.62 | 72.63 | 72.05 | 0.57 | 59.57 | 5.64 | 6.21 | 47.79 |
| individual--dino_global--legacy--chronological--budget-20 | 0 | 88.54 | 90.94 | 89.71 | 40.88 | 39.79 | 1.10 | 93.35 | 3.61 | 4.70 | 216.36 |
| individual--dino_global--legacy--chronological--budget-20 | 1 | 90.32 | 96.46 | 93.28 | 51.59 | 50.52 | 1.07 | 82.30 | 3.94 | 5.01 | 84.54 |
| individual--dino_global--legacy--chronological--budget-20 | 2 | 91.61 | 97.89 | 94.64 | 62.15 | 61.25 | 0.90 | 71.35 | 4.34 | 5.23 | 50.34 |
| individual--dino_global--legacy--chronological--budget-20 | 3 | 92.62 | 98.32 | 95.38 | 72.73 | 72.05 | 0.67 | 60.39 | 4.71 | 5.39 | 40.09 |
| individual--dino_global--legacy--chronological--budget-40 | 0 | 91.93 | 94.11 | 93.00 | 40.74 | 39.79 | 0.96 | 94.75 | 2.34 | 3.30 | 140.48 |
| individual--dino_global--legacy--chronological--budget-40 | 1 | 93.36 | 97.73 | 95.49 | 51.36 | 50.52 | 0.84 | 83.90 | 2.57 | 3.42 | 54.25 |
| individual--dino_global--legacy--chronological--budget-40 | 2 | 94.22 | 98.71 | 96.41 | 62.00 | 61.25 | 0.75 | 72.99 | 2.84 | 3.59 | 30.90 |
| individual--dino_global--legacy--chronological--budget-40 | 3 | 94.90 | 98.95 | 96.88 | 72.68 | 72.05 | 0.62 | 62.06 | 3.09 | 3.72 | 24.95 |
| individual--dino_global--legacy--evidence--budget-05 | 0 | 86.34 | 88.37 | 87.32 | 40.76 | 39.79 | 0.97 | 92.45 | 4.63 | 5.60 | 277.51 |
| individual--dino_global--legacy--evidence--budget-05 | 1 | 88.40 | 95.45 | 91.77 | 51.52 | 50.52 | 1.00 | 81.31 | 5.00 | 6.01 | 108.67 |
| individual--dino_global--legacy--evidence--budget-05 | 2 | 89.90 | 97.31 | 93.44 | 62.05 | 61.25 | 0.80 | 70.28 | 5.50 | 6.30 | 64.30 |
| individual--dino_global--legacy--evidence--budget-05 | 3 | 91.01 | 97.92 | 94.33 | 72.65 | 72.05 | 0.60 | 59.22 | 5.96 | 6.56 | 49.73 |
| individual--dino_global--legacy--evidence--budget-10 | 0 | 86.96 | 89.36 | 88.13 | 40.92 | 39.79 | 1.13 | 92.69 | 4.23 | 5.36 | 253.95 |
| individual--dino_global--legacy--evidence--budget-10 | 1 | 88.95 | 95.94 | 92.30 | 51.68 | 50.52 | 1.16 | 81.58 | 4.57 | 5.73 | 96.98 |
| individual--dino_global--legacy--evidence--budget-10 | 2 | 90.36 | 97.60 | 93.84 | 62.22 | 61.25 | 0.97 | 70.56 | 5.05 | 6.02 | 57.24 |
| individual--dino_global--legacy--evidence--budget-10 | 3 | 91.42 | 98.16 | 94.66 | 72.84 | 72.05 | 0.78 | 59.51 | 5.49 | 6.27 | 43.90 |
| individual--dino_global--legacy--evidence--budget-20 | 0 | 88.40 | 91.18 | 89.74 | 41.07 | 39.79 | 1.29 | 93.25 | 3.51 | 4.79 | 210.45 |
| individual--dino_global--legacy--evidence--budget-20 | 1 | 90.15 | 96.66 | 93.28 | 51.87 | 50.52 | 1.35 | 82.18 | 3.78 | 5.13 | 79.67 |
| individual--dino_global--legacy--evidence--budget-20 | 2 | 91.37 | 98.11 | 94.61 | 62.51 | 61.25 | 1.26 | 71.16 | 4.16 | 5.42 | 45.07 |
| individual--dino_global--legacy--evidence--budget-20 | 3 | 92.38 | 98.61 | 95.38 | 73.13 | 72.05 | 1.08 | 60.18 | 4.52 | 5.60 | 33.28 |
| individual--dino_global--legacy--evidence--budget-40 | 0 | 91.53 | 94.40 | 92.93 | 41.04 | 39.79 | 1.26 | 94.56 | 2.23 | 3.49 | 133.77 |
| individual--dino_global--legacy--evidence--budget-40 | 1 | 92.91 | 97.59 | 95.19 | 51.63 | 50.52 | 1.11 | 83.64 | 2.56 | 3.67 | 57.43 |
| individual--dino_global--legacy--evidence--budget-40 | 2 | 93.83 | 98.54 | 96.12 | 62.21 | 61.25 | 0.96 | 72.73 | 2.89 | 3.85 | 34.90 |
| individual--dino_global--legacy--evidence--budget-40 | 3 | 94.57 | 98.87 | 96.67 | 72.81 | 72.05 | 0.76 | 61.82 | 3.21 | 3.96 | 27.03 |
| individual--dino_global--local_events--chronological--budget-05 | 0 | 86.99 | 88.91 | 87.91 | 40.70 | 39.79 | 0.91 | 92.72 | 4.41 | 5.32 | 264.80 |
| individual--dino_global--local_events--chronological--budget-05 | 1 | 89.02 | 95.83 | 92.29 | 51.42 | 50.52 | 0.90 | 81.64 | 4.77 | 5.67 | 99.43 |
| individual--dino_global--local_events--chronological--budget-05 | 2 | 90.45 | 97.60 | 93.87 | 61.99 | 61.25 | 0.74 | 70.63 | 5.21 | 5.95 | 57.35 |
| individual--dino_global--local_events--chronological--budget-05 | 3 | 91.55 | 98.15 | 94.72 | 72.60 | 72.05 | 0.55 | 59.62 | 5.62 | 6.16 | 44.26 |
| individual--dino_global--local_events--chronological--budget-10 | 0 | 88.19 | 90.18 | 89.15 | 40.71 | 39.79 | 0.93 | 93.21 | 3.91 | 4.84 | 234.47 |
| individual--dino_global--local_events--chronological--budget-10 | 1 | 90.12 | 96.41 | 93.14 | 51.37 | 50.52 | 0.85 | 82.21 | 4.25 | 5.10 | 85.81 |
| individual--dino_global--local_events--chronological--budget-10 | 2 | 91.44 | 97.96 | 94.58 | 61.94 | 61.25 | 0.69 | 71.26 | 4.63 | 5.32 | 48.66 |
| individual--dino_global--local_events--chronological--budget-10 | 3 | 92.48 | 98.43 | 95.36 | 72.52 | 72.05 | 0.47 | 60.31 | 5.01 | 5.47 | 37.39 |
| individual--dino_global--local_events--chronological--budget-20 | 0 | 89.41 | 91.74 | 90.55 | 40.84 | 39.79 | 1.06 | 93.70 | 3.29 | 4.34 | 197.14 |
| individual--dino_global--local_events--chronological--budget-20 | 1 | 91.44 | 97.25 | 94.25 | 51.37 | 50.52 | 0.85 | 82.90 | 3.56 | 4.41 | 65.70 |
| individual--dino_global--local_events--chronological--budget-20 | 2 | 92.69 | 98.51 | 95.51 | 61.93 | 61.25 | 0.67 | 72.05 | 3.86 | 4.54 | 35.50 |
| individual--dino_global--local_events--chronological--budget-20 | 3 | 93.70 | 98.88 | 96.22 | 72.49 | 72.05 | 0.44 | 61.20 | 4.14 | 4.58 | 26.66 |
| individual--dino_global--local_events--chronological--budget-40 | 0 | 89.84 | 92.66 | 91.22 | 41.05 | 39.79 | 1.26 | 93.86 | 2.92 | 4.19 | 175.30 |
| individual--dino_global--local_events--chronological--budget-40 | 1 | 91.79 | 97.79 | 94.69 | 51.66 | 50.52 | 1.14 | 83.06 | 3.11 | 4.25 | 52.64 |
| individual--dino_global--local_events--chronological--budget-40 | 2 | 92.99 | 98.94 | 95.87 | 62.31 | 61.25 | 1.06 | 72.20 | 3.32 | 4.38 | 25.27 |
| individual--dino_global--local_events--chronological--budget-40 | 3 | 93.94 | 99.28 | 96.53 | 72.95 | 72.05 | 0.90 | 61.35 | 3.53 | 4.43 | 17.22 |
| individual--dino_global--local_events--evidence--budget-05 | 0 | 87.24 | 88.90 | 88.03 | 40.57 | 39.79 | 0.79 | 92.84 | 4.42 | 5.21 | 265.03 |
| individual--dino_global--local_events--evidence--budget-05 | 1 | 89.26 | 95.68 | 92.34 | 51.19 | 50.52 | 0.67 | 81.79 | 4.85 | 5.53 | 103.19 |
| individual--dino_global--local_events--evidence--budget-05 | 2 | 90.65 | 97.41 | 93.90 | 61.68 | 61.25 | 0.43 | 70.79 | 5.36 | 5.79 | 61.87 |
| individual--dino_global--local_events--evidence--budget-05 | 3 | 91.75 | 97.96 | 94.74 | 72.17 | 72.05 | 0.12 | 59.79 | 5.87 | 5.98 | 48.69 |
| individual--dino_global--local_events--evidence--budget-10 | 0 | 88.56 | 90.31 | 89.40 | 40.60 | 39.79 | 0.82 | 93.38 | 3.86 | 4.67 | 231.43 |
| individual--dino_global--local_events--evidence--budget-10 | 1 | 90.52 | 96.34 | 93.32 | 51.11 | 50.52 | 0.59 | 82.44 | 4.28 | 4.87 | 87.45 |
| individual--dino_global--local_events--evidence--budget-10 | 2 | 91.80 | 97.83 | 94.71 | 61.62 | 61.25 | 0.37 | 71.51 | 4.71 | 5.08 | 51.78 |
| individual--dino_global--local_events--evidence--budget-10 | 3 | 92.83 | 98.30 | 95.47 | 72.11 | 72.05 | 0.06 | 60.58 | 5.14 | 5.19 | 40.66 |
| individual--dino_global--local_events--evidence--budget-20 | 0 | 89.71 | 91.83 | 90.74 | 40.75 | 39.79 | 0.97 | 93.83 | 3.25 | 4.22 | 195.00 |
| individual--dino_global--local_events--evidence--budget-20 | 1 | 91.62 | 97.12 | 94.28 | 51.21 | 50.52 | 0.69 | 83.00 | 3.62 | 4.31 | 68.68 |
| individual--dino_global--local_events--evidence--budget-20 | 2 | 92.84 | 98.32 | 95.49 | 61.68 | 61.25 | 0.43 | 72.15 | 4.01 | 4.43 | 40.07 |
| individual--dino_global--local_events--evidence--budget-20 | 3 | 93.80 | 98.69 | 96.18 | 72.14 | 72.05 | 0.09 | 61.29 | 4.39 | 4.48 | 31.24 |
| individual--dino_global--local_events--evidence--budget-40 | 0 | 89.84 | 92.66 | 91.22 | 41.05 | 39.79 | 1.26 | 93.86 | 2.92 | 4.19 | 175.30 |
| individual--dino_global--local_events--evidence--budget-40 | 1 | 91.79 | 97.79 | 94.69 | 51.66 | 50.52 | 1.14 | 83.06 | 3.11 | 4.25 | 52.64 |
| individual--dino_global--local_events--evidence--budget-40 | 2 | 92.99 | 98.94 | 95.87 | 62.31 | 61.25 | 1.06 | 72.20 | 3.32 | 4.38 | 25.27 |
| individual--dino_global--local_events--evidence--budget-40 | 3 | 93.94 | 99.28 | 96.53 | 72.95 | 72.05 | 0.90 | 61.35 | 3.53 | 4.43 | 17.22 |
| individual--dino_global--local_heads--chronological--budget-05 | 0 | 86.89 | 88.70 | 87.76 | 40.65 | 39.79 | 0.86 | 92.69 | 4.50 | 5.36 | 269.73 |
| individual--dino_global--local_heads--chronological--budget-05 | 1 | 88.88 | 95.62 | 92.11 | 51.35 | 50.52 | 0.83 | 81.58 | 4.91 | 5.74 | 104.64 |
| individual--dino_global--local_heads--chronological--budget-05 | 2 | 90.32 | 97.38 | 93.71 | 61.87 | 61.25 | 0.61 | 70.57 | 5.40 | 6.02 | 62.49 |
| individual--dino_global--local_heads--chronological--budget-05 | 3 | 91.38 | 97.97 | 94.55 | 72.46 | 72.05 | 0.41 | 59.50 | 5.87 | 6.27 | 48.54 |
| individual--dino_global--local_heads--chronological--budget-10 | 0 | 88.21 | 89.81 | 88.99 | 40.53 | 39.79 | 0.74 | 93.25 | 4.06 | 4.80 | 243.33 |
| individual--dino_global--local_heads--chronological--budget-10 | 1 | 90.10 | 96.06 | 92.97 | 51.16 | 50.52 | 0.64 | 82.23 | 4.45 | 5.09 | 94.11 |
| individual--dino_global--local_heads--chronological--budget-10 | 2 | 91.40 | 97.65 | 94.41 | 61.71 | 61.25 | 0.46 | 71.25 | 4.88 | 5.33 | 56.10 |
| individual--dino_global--local_heads--chronological--budget-10 | 3 | 92.42 | 98.17 | 95.20 | 72.27 | 72.05 | 0.22 | 60.28 | 5.29 | 5.50 | 43.70 |
| individual--dino_global--local_heads--chronological--budget-20 | 0 | 90.43 | 91.61 | 91.00 | 40.33 | 39.79 | 0.54 | 94.17 | 3.34 | 3.88 | 200.39 |
| individual--dino_global--local_heads--chronological--budget-20 | 1 | 92.04 | 96.79 | 94.35 | 50.91 | 50.52 | 0.39 | 83.24 | 3.68 | 4.07 | 76.52 |
| individual--dino_global--local_heads--chronological--budget-20 | 2 | 93.19 | 98.11 | 95.58 | 61.41 | 61.25 | 0.16 | 72.38 | 4.04 | 4.20 | 45.21 |
| individual--dino_global--local_heads--chronological--budget-20 | 3 | 93.99 | 98.55 | 96.21 | 72.02 | 72.05 | -0.04 | 61.44 | 4.38 | 4.34 | 34.68 |
| individual--dino_global--local_heads--chronological--budget-40 | 0 | 92.57 | 94.35 | 93.44 | 40.56 | 39.79 | 0.77 | 95.03 | 2.25 | 3.02 | 134.95 |
| individual--dino_global--local_heads--chronological--budget-40 | 1 | 93.94 | 97.99 | 95.92 | 51.09 | 50.52 | 0.57 | 84.21 | 2.53 | 3.10 | 47.88 |
| individual--dino_global--local_heads--chronological--budget-40 | 2 | 94.84 | 98.85 | 96.80 | 61.62 | 61.25 | 0.36 | 73.40 | 2.82 | 3.18 | 27.41 |
| individual--dino_global--local_heads--chronological--budget-40 | 3 | 95.47 | 99.11 | 97.26 | 72.23 | 72.05 | 0.18 | 62.50 | 3.10 | 3.28 | 21.23 |
| individual--dino_global--local_heads--evidence--budget-05 | 0 | 87.22 | 89.00 | 88.08 | 40.63 | 39.79 | 0.84 | 92.83 | 4.38 | 5.22 | 262.54 |
| individual--dino_global--local_heads--evidence--budget-05 | 1 | 89.32 | 95.71 | 92.39 | 51.14 | 50.52 | 0.62 | 81.82 | 4.87 | 5.49 | 102.36 |
| individual--dino_global--local_heads--evidence--budget-05 | 2 | 90.77 | 97.37 | 93.94 | 61.55 | 61.25 | 0.30 | 70.88 | 5.40 | 5.70 | 62.85 |
| individual--dino_global--local_heads--evidence--budget-05 | 3 | 91.93 | 97.89 | 94.81 | 71.96 | 72.05 | -0.09 | 59.95 | 5.92 | 5.83 | 50.39 |
| individual--dino_global--local_heads--evidence--budget-10 | 0 | 88.50 | 90.36 | 89.40 | 40.65 | 39.79 | 0.86 | 93.35 | 3.84 | 4.70 | 230.10 |
| individual--dino_global--local_heads--evidence--budget-10 | 1 | 90.48 | 96.28 | 93.28 | 51.09 | 50.52 | 0.58 | 82.43 | 4.31 | 4.88 | 88.81 |
| individual--dino_global--local_heads--evidence--budget-10 | 2 | 91.86 | 97.76 | 94.71 | 61.50 | 61.25 | 0.24 | 71.56 | 4.78 | 5.02 | 53.57 |
| individual--dino_global--local_heads--evidence--budget-10 | 3 | 92.87 | 98.29 | 95.50 | 71.98 | 72.05 | -0.08 | 60.63 | 5.22 | 5.15 | 40.74 |
| individual--dino_global--local_heads--evidence--budget-20 | 0 | 89.97 | 92.31 | 91.11 | 40.84 | 39.79 | 1.06 | 93.93 | 3.06 | 4.12 | 183.68 |
| individual--dino_global--local_heads--evidence--budget-20 | 1 | 91.71 | 97.05 | 94.29 | 51.36 | 50.52 | 0.84 | 83.04 | 3.43 | 4.28 | 70.45 |
| individual--dino_global--local_heads--evidence--budget-20 | 2 | 92.96 | 98.20 | 95.50 | 61.81 | 61.25 | 0.56 | 72.21 | 3.81 | 4.37 | 43.01 |
| individual--dino_global--local_heads--evidence--budget-20 | 3 | 93.84 | 98.65 | 96.18 | 72.36 | 72.05 | 0.31 | 61.31 | 4.17 | 4.47 | 32.16 |
| individual--dino_global--local_heads--evidence--budget-40 | 0 | 92.47 | 94.39 | 93.42 | 40.62 | 39.79 | 0.83 | 94.99 | 2.23 | 3.06 | 133.92 |
| individual--dino_global--local_heads--evidence--budget-40 | 1 | 93.86 | 97.91 | 95.83 | 51.18 | 50.52 | 0.66 | 84.17 | 2.49 | 3.15 | 49.98 |
| individual--dino_global--local_heads--evidence--budget-40 | 2 | 94.81 | 98.77 | 96.74 | 61.70 | 61.25 | 0.45 | 73.37 | 2.76 | 3.21 | 29.44 |
| individual--dino_global--local_heads--evidence--budget-40 | 3 | 95.42 | 99.10 | 97.23 | 72.37 | 72.05 | 0.31 | 62.46 | 3.01 | 3.32 | 21.38 |
| individual--dino_boost--legacy--chronological--budget-05 | 0 | 84.85 | 86.94 | 85.87 | 40.79 | 39.79 | 1.00 | 91.85 | 5.20 | 6.20 | 311.74 |
| individual--dino_boost--legacy--chronological--budget-05 | 1 | 87.19 | 94.80 | 90.83 | 51.71 | 50.52 | 1.19 | 80.67 | 5.45 | 6.65 | 124.06 |
| individual--dino_boost--legacy--chronological--budget-05 | 2 | 88.93 | 97.01 | 92.79 | 62.33 | 61.25 | 1.07 | 69.67 | 5.84 | 6.92 | 71.31 |
| individual--dino_boost--legacy--chronological--budget-05 | 3 | 90.48 | 97.86 | 94.02 | 72.73 | 72.05 | 0.67 | 58.84 | 6.26 | 6.94 | 51.13 |
| individual--dino_boost--legacy--chronological--budget-10 | 0 | 85.62 | 87.66 | 86.61 | 40.76 | 39.79 | 0.97 | 92.17 | 4.91 | 5.88 | 294.51 |
| individual--dino_boost--legacy--chronological--budget-10 | 1 | 87.77 | 94.94 | 91.20 | 51.67 | 50.52 | 1.15 | 80.97 | 5.19 | 6.34 | 120.83 |
| individual--dino_boost--legacy--chronological--budget-10 | 2 | 89.39 | 97.08 | 93.07 | 62.33 | 61.25 | 1.08 | 69.95 | 5.55 | 6.63 | 69.82 |
| individual--dino_boost--legacy--chronological--budget-10 | 3 | 90.90 | 97.91 | 94.27 | 72.74 | 72.05 | 0.68 | 59.15 | 5.95 | 6.63 | 49.97 |
| individual--dino_boost--legacy--chronological--budget-20 | 0 | 87.20 | 89.23 | 88.19 | 40.74 | 39.79 | 0.95 | 92.81 | 4.28 | 5.23 | 256.98 |
| individual--dino_boost--legacy--chronological--budget-20 | 1 | 89.15 | 95.64 | 92.27 | 51.62 | 50.52 | 1.10 | 81.70 | 4.52 | 5.62 | 104.10 |
| individual--dino_boost--legacy--chronological--budget-20 | 2 | 90.59 | 97.50 | 93.91 | 62.29 | 61.25 | 1.04 | 70.70 | 4.84 | 5.88 | 59.70 |
| individual--dino_boost--legacy--chronological--budget-20 | 3 | 91.97 | 98.17 | 94.96 | 72.72 | 72.05 | 0.66 | 59.92 | 5.19 | 5.86 | 43.75 |
| individual--dino_boost--legacy--chronological--budget-40 | 0 | 90.29 | 92.83 | 91.53 | 40.92 | 39.79 | 1.14 | 94.06 | 2.85 | 3.99 | 171.21 |
| individual--dino_boost--legacy--chronological--budget-40 | 1 | 91.79 | 97.12 | 94.37 | 51.75 | 50.52 | 1.23 | 83.05 | 3.04 | 4.26 | 68.71 |
| individual--dino_boost--legacy--chronological--budget-40 | 2 | 92.93 | 98.45 | 95.60 | 62.41 | 61.25 | 1.16 | 72.15 | 3.27 | 4.43 | 36.90 |
| individual--dino_boost--legacy--chronological--budget-40 | 3 | 93.89 | 98.90 | 96.33 | 73.01 | 72.05 | 0.95 | 61.31 | 3.52 | 4.47 | 26.21 |
| individual--dino_boost--legacy--evidence--budget-05 | 0 | 84.86 | 86.92 | 85.86 | 40.77 | 39.79 | 0.99 | 91.86 | 5.20 | 6.19 | 312.30 |
| individual--dino_boost--legacy--evidence--budget-05 | 1 | 87.18 | 94.78 | 90.81 | 51.67 | 50.52 | 1.15 | 80.67 | 5.50 | 6.64 | 124.52 |
| individual--dino_boost--legacy--evidence--budget-05 | 2 | 88.91 | 96.99 | 92.77 | 62.28 | 61.25 | 1.03 | 69.65 | 5.90 | 6.93 | 71.75 |
| individual--dino_boost--legacy--evidence--budget-05 | 3 | 90.44 | 97.86 | 94.00 | 72.70 | 72.05 | 0.64 | 58.82 | 6.32 | 6.96 | 51.16 |
| individual--dino_boost--legacy--evidence--budget-10 | 0 | 85.33 | 87.67 | 86.47 | 40.89 | 39.79 | 1.11 | 92.03 | 4.91 | 6.01 | 294.36 |
| individual--dino_boost--legacy--evidence--budget-10 | 1 | 87.62 | 95.06 | 91.18 | 51.78 | 50.52 | 1.26 | 80.89 | 5.16 | 6.43 | 117.87 |
| individual--dino_boost--legacy--evidence--budget-10 | 2 | 89.29 | 97.17 | 93.06 | 62.41 | 61.25 | 1.16 | 69.88 | 5.54 | 6.70 | 67.56 |
| individual--dino_boost--legacy--evidence--budget-10 | 3 | 90.77 | 98.01 | 94.25 | 72.86 | 72.05 | 0.81 | 59.04 | 5.93 | 6.74 | 47.53 |
| individual--dino_boost--legacy--evidence--budget-20 | 0 | 86.57 | 89.47 | 87.99 | 41.13 | 39.79 | 1.35 | 92.51 | 4.19 | 5.54 | 251.42 |
| individual--dino_boost--legacy--evidence--budget-20 | 1 | 88.88 | 95.80 | 92.20 | 51.85 | 50.52 | 1.33 | 81.54 | 4.45 | 5.78 | 100.36 |
| individual--dino_boost--legacy--evidence--budget-20 | 2 | 90.40 | 97.59 | 93.86 | 62.46 | 61.25 | 1.21 | 70.58 | 4.80 | 6.00 | 57.44 |
| individual--dino_boost--legacy--evidence--budget-20 | 3 | 91.74 | 98.34 | 94.92 | 72.94 | 72.05 | 0.89 | 59.75 | 5.14 | 6.03 | 39.64 |
| individual--dino_boost--legacy--evidence--budget-40 | 0 | 89.48 | 93.09 | 91.24 | 41.40 | 39.79 | 1.61 | 93.68 | 2.75 | 4.36 | 165.03 |
| individual--dino_boost--legacy--evidence--budget-40 | 1 | 91.41 | 97.09 | 94.16 | 51.98 | 50.52 | 1.46 | 82.84 | 3.01 | 4.47 | 69.47 |
| individual--dino_boost--legacy--evidence--budget-40 | 2 | 92.66 | 98.37 | 95.43 | 62.55 | 61.25 | 1.30 | 71.98 | 3.30 | 4.60 | 38.95 |
| individual--dino_boost--legacy--evidence--budget-40 | 3 | 93.70 | 98.87 | 96.21 | 73.07 | 72.05 | 1.02 | 61.17 | 3.59 | 4.61 | 27.01 |
| individual--dino_boost--local_events--chronological--budget-05 | 0 | 85.45 | 87.52 | 86.46 | 40.78 | 39.79 | 0.99 | 92.09 | 4.96 | 5.95 | 297.84 |
| individual--dino_boost--local_events--chronological--budget-05 | 1 | 87.79 | 95.21 | 91.34 | 51.57 | 50.52 | 1.05 | 81.00 | 5.26 | 6.31 | 114.39 |
| individual--dino_boost--local_events--chronological--budget-05 | 2 | 89.48 | 97.33 | 93.24 | 62.18 | 61.25 | 0.92 | 70.03 | 5.63 | 6.55 | 63.79 |
| individual--dino_boost--local_events--chronological--budget-05 | 3 | 91.03 | 98.10 | 94.43 | 72.56 | 72.05 | 0.50 | 59.26 | 6.02 | 6.52 | 45.26 |
| individual--dino_boost--local_events--chronological--budget-10 | 0 | 86.33 | 89.05 | 87.66 | 41.06 | 39.79 | 1.28 | 92.41 | 4.36 | 5.63 | 261.30 |
| individual--dino_boost--local_events--chronological--budget-10 | 1 | 88.67 | 95.93 | 92.15 | 51.78 | 50.52 | 1.26 | 81.43 | 4.63 | 5.89 | 97.04 |
| individual--dino_boost--local_events--chronological--budget-10 | 2 | 90.35 | 97.76 | 93.90 | 62.31 | 61.25 | 1.06 | 70.55 | 4.97 | 6.03 | 53.50 |
| individual--dino_boost--local_events--chronological--budget-10 | 3 | 91.75 | 98.39 | 94.95 | 72.76 | 72.05 | 0.70 | 59.77 | 5.31 | 6.01 | 38.45 |
| individual--dino_boost--local_events--chronological--budget-20 | 0 | 87.52 | 90.55 | 89.00 | 41.18 | 39.79 | 1.39 | 92.90 | 3.76 | 5.15 | 225.63 |
| individual--dino_boost--local_events--chronological--budget-20 | 1 | 89.98 | 96.75 | 93.24 | 51.77 | 50.52 | 1.25 | 82.12 | 3.94 | 5.20 | 77.60 |
| individual--dino_boost--local_events--chronological--budget-20 | 2 | 91.63 | 98.35 | 94.87 | 62.32 | 61.25 | 1.06 | 71.36 | 4.16 | 5.23 | 39.38 |
| individual--dino_boost--local_events--chronological--budget-20 | 3 | 92.85 | 98.87 | 95.76 | 72.88 | 72.05 | 0.83 | 60.56 | 4.39 | 5.21 | 26.94 |
| individual--dino_boost--local_events--chronological--budget-40 | 0 | 88.36 | 91.50 | 89.90 | 41.21 | 39.79 | 1.43 | 93.24 | 3.38 | 4.81 | 202.90 |
| individual--dino_boost--local_events--chronological--budget-40 | 1 | 90.73 | 97.19 | 93.84 | 51.78 | 50.52 | 1.27 | 82.51 | 3.54 | 4.81 | 67.18 |
| individual--dino_boost--local_events--chronological--budget-40 | 2 | 92.30 | 98.61 | 95.35 | 62.35 | 61.25 | 1.10 | 71.78 | 3.71 | 4.80 | 33.09 |
| individual--dino_boost--local_events--chronological--budget-40 | 3 | 93.47 | 99.06 | 96.18 | 72.93 | 72.05 | 0.87 | 61.01 | 3.90 | 4.77 | 22.53 |
| individual--dino_boost--local_events--evidence--budget-05 | 0 | 85.65 | 87.60 | 86.60 | 40.71 | 39.79 | 0.93 | 92.19 | 4.93 | 5.86 | 295.99 |
| individual--dino_boost--local_events--evidence--budget-05 | 1 | 88.09 | 95.19 | 91.50 | 51.42 | 50.52 | 0.90 | 81.17 | 5.24 | 6.14 | 114.83 |
| individual--dino_boost--local_events--evidence--budget-05 | 2 | 89.77 | 97.25 | 93.35 | 61.96 | 61.25 | 0.71 | 70.23 | 5.65 | 6.36 | 65.75 |
| individual--dino_boost--local_events--evidence--budget-05 | 3 | 91.23 | 98.00 | 94.49 | 72.34 | 72.05 | 0.29 | 59.42 | 6.07 | 6.36 | 47.68 |
| individual--dino_boost--local_events--evidence--budget-10 | 0 | 86.82 | 89.20 | 87.99 | 40.89 | 39.79 | 1.11 | 92.65 | 4.30 | 5.40 | 257.72 |
| individual--dino_boost--local_events--evidence--budget-10 | 1 | 89.13 | 95.96 | 92.41 | 51.51 | 50.52 | 0.99 | 81.70 | 4.62 | 5.61 | 96.55 |
| individual--dino_boost--local_events--evidence--budget-10 | 2 | 90.74 | 97.75 | 94.11 | 62.03 | 61.25 | 0.78 | 70.82 | 4.98 | 5.76 | 53.68 |
| individual--dino_boost--local_events--evidence--budget-10 | 3 | 92.06 | 98.37 | 95.11 | 72.45 | 72.05 | 0.39 | 60.02 | 5.37 | 5.76 | 38.94 |
| individual--dino_boost--local_events--evidence--budget-20 | 0 | 87.86 | 90.65 | 89.23 | 41.06 | 39.79 | 1.28 | 93.05 | 3.72 | 5.00 | 223.10 |
| individual--dino_boost--local_events--evidence--budget-20 | 1 | 90.30 | 96.65 | 93.36 | 51.54 | 50.52 | 1.02 | 82.31 | 3.99 | 5.01 | 79.90 |
| individual--dino_boost--local_events--evidence--budget-20 | 2 | 91.93 | 98.21 | 94.96 | 62.00 | 61.25 | 0.75 | 71.57 | 4.26 | 5.01 | 42.80 |
| individual--dino_boost--local_events--evidence--budget-20 | 3 | 93.09 | 98.72 | 95.82 | 72.51 | 72.05 | 0.45 | 60.76 | 4.57 | 5.02 | 30.49 |
| individual--dino_boost--local_events--evidence--budget-40 | 0 | 88.36 | 91.50 | 89.90 | 41.21 | 39.79 | 1.43 | 93.24 | 3.38 | 4.81 | 202.90 |
| individual--dino_boost--local_events--evidence--budget-40 | 1 | 90.73 | 97.19 | 93.84 | 51.78 | 50.52 | 1.27 | 82.51 | 3.54 | 4.81 | 67.18 |
| individual--dino_boost--local_events--evidence--budget-40 | 2 | 92.30 | 98.61 | 95.35 | 62.35 | 61.25 | 1.10 | 71.78 | 3.71 | 4.80 | 33.09 |
| individual--dino_boost--local_events--evidence--budget-40 | 3 | 93.47 | 99.06 | 96.18 | 72.93 | 72.05 | 0.87 | 61.01 | 3.90 | 4.77 | 22.53 |
| individual--dino_boost--local_heads--chronological--budget-05 | 0 | 85.28 | 87.25 | 86.24 | 40.72 | 39.79 | 0.94 | 92.04 | 5.07 | 6.01 | 304.36 |
| individual--dino_boost--local_heads--chronological--budget-05 | 1 | 87.50 | 94.93 | 91.05 | 51.65 | 50.52 | 1.13 | 80.84 | 5.34 | 6.47 | 121.09 |
| individual--dino_boost--local_heads--chronological--budget-05 | 2 | 89.16 | 97.06 | 92.94 | 62.30 | 61.25 | 1.05 | 69.81 | 5.72 | 6.77 | 70.08 |
| individual--dino_boost--local_heads--chronological--budget-05 | 3 | 90.72 | 97.89 | 94.16 | 72.69 | 72.05 | 0.63 | 59.02 | 6.13 | 6.76 | 50.47 |
| individual--dino_boost--local_heads--chronological--budget-10 | 0 | 86.45 | 88.24 | 87.33 | 40.63 | 39.79 | 0.84 | 92.53 | 4.68 | 5.52 | 280.75 |
| individual--dino_boost--local_heads--chronological--budget-10 | 1 | 88.49 | 95.31 | 91.76 | 51.48 | 50.52 | 0.96 | 81.37 | 4.98 | 5.94 | 112.04 |
| individual--dino_boost--local_heads--chronological--budget-10 | 2 | 90.03 | 97.30 | 93.52 | 62.12 | 61.25 | 0.86 | 70.38 | 5.34 | 6.21 | 64.42 |
| individual--dino_boost--local_heads--chronological--budget-10 | 3 | 91.53 | 98.04 | 94.67 | 72.47 | 72.05 | 0.41 | 59.63 | 5.74 | 6.15 | 46.86 |
| individual--dino_boost--local_heads--chronological--budget-20 | 0 | 88.59 | 90.48 | 89.51 | 40.66 | 39.79 | 0.87 | 93.39 | 3.79 | 4.66 | 227.32 |
| individual--dino_boost--local_heads--chronological--budget-20 | 1 | 90.30 | 96.39 | 93.24 | 51.50 | 50.52 | 0.98 | 82.30 | 4.03 | 5.02 | 86.07 |
| individual--dino_boost--local_heads--chronological--budget-20 | 2 | 91.67 | 98.03 | 94.73 | 62.10 | 61.25 | 0.84 | 71.39 | 4.35 | 5.19 | 47.13 |
| individual--dino_boost--local_heads--chronological--budget-20 | 3 | 92.85 | 98.56 | 95.62 | 72.59 | 72.05 | 0.54 | 60.57 | 4.67 | 5.20 | 34.32 |
| individual--dino_boost--local_heads--chronological--budget-40 | 0 | 91.63 | 94.17 | 92.88 | 40.90 | 39.79 | 1.11 | 94.62 | 2.32 | 3.43 | 139.13 |
| individual--dino_boost--local_heads--chronological--budget-40 | 1 | 93.25 | 97.85 | 95.49 | 51.49 | 50.52 | 0.97 | 83.83 | 2.51 | 3.48 | 51.43 |
| individual--dino_boost--local_heads--chronological--budget-40 | 2 | 94.36 | 98.74 | 96.49 | 62.00 | 61.25 | 0.75 | 73.08 | 2.75 | 3.50 | 30.16 |
| individual--dino_boost--local_heads--chronological--budget-40 | 3 | 95.22 | 99.01 | 97.08 | 72.52 | 72.05 | 0.47 | 62.31 | 3.00 | 3.47 | 23.64 |
| individual--dino_boost--local_heads--evidence--budget-05 | 0 | 85.73 | 87.59 | 86.64 | 40.66 | 39.79 | 0.88 | 92.23 | 4.94 | 5.81 | 296.25 |
| individual--dino_boost--local_heads--evidence--budget-05 | 1 | 88.00 | 95.21 | 91.45 | 51.43 | 50.52 | 0.91 | 81.13 | 5.27 | 6.19 | 114.40 |
| individual--dino_boost--local_heads--evidence--budget-05 | 2 | 89.68 | 97.23 | 93.30 | 61.96 | 61.25 | 0.71 | 70.17 | 5.70 | 6.41 | 66.13 |
| individual--dino_boost--local_heads--evidence--budget-05 | 3 | 91.15 | 97.96 | 94.43 | 72.34 | 72.05 | 0.28 | 59.37 | 6.13 | 6.41 | 48.60 |
| individual--dino_boost--local_heads--evidence--budget-10 | 0 | 86.79 | 88.83 | 87.79 | 40.74 | 39.79 | 0.95 | 92.65 | 4.44 | 5.40 | 266.64 |
| individual--dino_boost--local_heads--evidence--budget-10 | 1 | 89.15 | 95.64 | 92.27 | 51.30 | 50.52 | 0.78 | 81.73 | 4.80 | 5.58 | 104.01 |
| individual--dino_boost--local_heads--evidence--budget-10 | 2 | 90.74 | 97.43 | 93.96 | 61.74 | 61.25 | 0.49 | 70.85 | 5.24 | 5.73 | 61.41 |
| individual--dino_boost--local_heads--evidence--budget-10 | 3 | 92.20 | 98.08 | 95.04 | 72.02 | 72.05 | -0.04 | 60.15 | 5.66 | 5.63 | 45.83 |
| individual--dino_boost--local_heads--evidence--budget-20 | 0 | 88.18 | 91.07 | 89.59 | 41.10 | 39.79 | 1.32 | 93.18 | 3.55 | 4.87 | 213.27 |
| individual--dino_boost--local_heads--evidence--budget-20 | 1 | 90.49 | 96.56 | 93.42 | 51.55 | 50.52 | 1.03 | 82.40 | 3.88 | 4.91 | 82.04 |
| individual--dino_boost--local_heads--evidence--budget-20 | 2 | 92.04 | 97.98 | 94.91 | 61.93 | 61.25 | 0.67 | 71.64 | 4.27 | 4.94 | 48.13 |
| individual--dino_boost--local_heads--evidence--budget-20 | 3 | 93.26 | 98.47 | 95.79 | 72.30 | 72.05 | 0.24 | 60.90 | 4.63 | 4.88 | 36.62 |
| individual--dino_boost--local_heads--evidence--budget-40 | 0 | 91.25 | 94.68 | 92.92 | 41.29 | 39.79 | 1.51 | 94.43 | 2.12 | 3.62 | 127.00 |
| individual--dino_boost--local_heads--evidence--budget-40 | 1 | 92.94 | 98.01 | 95.41 | 51.87 | 50.52 | 1.35 | 83.65 | 2.32 | 3.67 | 47.43 |
| individual--dino_boost--local_heads--evidence--budget-40 | 2 | 94.10 | 98.83 | 96.40 | 62.40 | 61.25 | 1.14 | 72.89 | 2.54 | 3.69 | 28.04 |
| individual--dino_boost--local_heads--evidence--budget-40 | 3 | 95.00 | 99.08 | 97.00 | 72.95 | 72.05 | 0.90 | 62.13 | 2.75 | 3.65 | 22.00 |

## Source-group results at target padding

| Arm | Source group | P_pad % | R_core % | F1_padP_coreR % | Event F1 % | Observed start P @1s % | Observed start R @1s % | Observed start F1 @1s % | Playback min | True rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production--compact_boost--legacy--chronological--budget-05 | source-group-005 | 72.76 | 99.13 | 83.92 | 66.79 | 66.55 | 73.33 | 69.78 | 1.81 | 5.67 |
| production--compact_boost--legacy--chronological--budget-05 | source-group-007 | 60.91 | 98.95 | 75.40 | 56.52 | 52.85 | 60.75 | 56.52 | 2.08 | 5.00 |
| production--compact_boost--legacy--chronological--budget-05 | source-group-009 | 91.51 | 100.00 | 95.57 | 92.89 | 85.96 | 88.29 | 87.11 | 1.32 | 3.33 |
| production--compact_boost--legacy--chronological--budget-05 | source-group-012 | 80.46 | 99.18 | 88.84 | 66.48 | 67.90 | 76.79 | 72.07 | 1.09 | 3.00 |
| production--compact_boost--legacy--chronological--budget-10 | source-group-005 | 74.94 | 99.13 | 85.36 | 70.52 | 71.34 | 78.04 | 74.54 | 3.86 | 12.00 |
| production--compact_boost--legacy--chronological--budget-10 | source-group-007 | 62.16 | 98.95 | 76.36 | 59.16 | 55.38 | 64.17 | 59.45 | 4.35 | 12.33 |
| production--compact_boost--legacy--chronological--budget-10 | source-group-009 | 92.19 | 100.00 | 95.94 | 94.64 | 87.18 | 88.74 | 87.95 | 2.89 | 8.00 |
| production--compact_boost--legacy--chronological--budget-10 | source-group-012 | 81.48 | 99.41 | 89.56 | 67.42 | 69.16 | 77.38 | 73.04 | 2.21 | 7.00 |
| production--compact_boost--legacy--chronological--budget-20 | source-group-005 | 78.54 | 99.13 | 87.64 | 76.93 | 75.37 | 81.57 | 78.34 | 7.75 | 23.00 |
| production--compact_boost--legacy--chronological--budget-20 | source-group-007 | 66.64 | 98.95 | 79.64 | 66.09 | 61.20 | 69.78 | 65.21 | 8.88 | 25.67 |
| production--compact_boost--legacy--chronological--budget-20 | source-group-009 | 93.16 | 100.00 | 96.46 | 95.30 | 88.89 | 90.09 | 89.49 | 4.90 | 14.33 |
| production--compact_boost--legacy--chronological--budget-20 | source-group-012 | 86.13 | 100.00 | 92.55 | 74.50 | 77.35 | 83.33 | 80.23 | 4.37 | 12.33 |
| production--compact_boost--legacy--chronological--budget-40 | source-group-005 | 87.52 | 99.18 | 92.99 | 84.05 | 87.55 | 88.24 | 87.89 | 15.67 | 43.33 |
| production--compact_boost--legacy--chronological--budget-40 | source-group-007 | 79.48 | 98.95 | 88.15 | 80.71 | 75.36 | 82.87 | 78.93 | 17.97 | 50.33 |
| production--compact_boost--legacy--chronological--budget-40 | source-group-009 | 93.23 | 100.00 | 96.49 | 95.30 | 88.89 | 90.09 | 89.49 | 5.12 | 15.00 |
| production--compact_boost--legacy--chronological--budget-40 | source-group-012 | 89.16 | 100.00 | 94.27 | 77.48 | 83.16 | 88.10 | 85.55 | 6.15 | 16.33 |
| production--compact_boost--legacy--evidence--budget-05 | source-group-005 | 72.81 | 99.13 | 83.95 | 66.17 | 66.44 | 73.73 | 69.89 | 1.91 | 5.00 |
| production--compact_boost--legacy--evidence--budget-05 | source-group-007 | 60.50 | 98.93 | 75.08 | 56.32 | 52.79 | 61.68 | 56.89 | 2.17 | 5.67 |
| production--compact_boost--legacy--evidence--budget-05 | source-group-009 | 89.87 | 100.00 | 94.67 | 90.07 | 85.28 | 88.74 | 86.98 | 1.40 | 3.00 |
| production--compact_boost--legacy--evidence--budget-05 | source-group-012 | 79.75 | 98.94 | 88.32 | 64.26 | 69.49 | 78.57 | 73.75 | 1.09 | 2.00 |
| production--compact_boost--legacy--evidence--budget-10 | source-group-005 | 74.59 | 99.13 | 85.12 | 68.78 | 68.91 | 76.47 | 72.49 | 3.80 | 10.00 |
| production--compact_boost--legacy--evidence--budget-10 | source-group-007 | 62.16 | 98.93 | 76.35 | 58.50 | 54.43 | 63.24 | 58.50 | 4.35 | 10.67 |
| production--compact_boost--legacy--evidence--budget-10 | source-group-009 | 90.85 | 100.00 | 95.21 | 91.15 | 86.96 | 90.09 | 88.49 | 2.89 | 7.33 |
| production--compact_boost--legacy--evidence--budget-10 | source-group-012 | 80.60 | 99.41 | 89.03 | 64.81 | 70.03 | 79.17 | 74.31 | 2.15 | 5.00 |
| production--compact_boost--legacy--evidence--budget-20 | source-group-005 | 78.51 | 99.17 | 87.64 | 73.00 | 74.12 | 81.96 | 77.84 | 7.75 | 21.00 |
| production--compact_boost--legacy--evidence--budget-20 | source-group-007 | 66.30 | 98.93 | 79.39 | 62.25 | 59.52 | 69.16 | 63.98 | 8.98 | 22.33 |
| production--compact_boost--legacy--evidence--budget-20 | source-group-009 | 93.12 | 100.00 | 96.44 | 95.30 | 88.89 | 90.09 | 89.49 | 4.90 | 14.00 |
| production--compact_boost--legacy--evidence--budget-20 | source-group-012 | 84.89 | 99.80 | 91.74 | 72.70 | 75.96 | 84.52 | 80.01 | 4.40 | 11.67 |
| production--compact_boost--legacy--evidence--budget-40 | source-group-005 | 88.09 | 99.19 | 93.31 | 85.21 | 87.22 | 88.24 | 87.72 | 15.66 | 42.67 |
| production--compact_boost--legacy--evidence--budget-40 | source-group-007 | 77.34 | 98.95 | 86.82 | 76.43 | 72.58 | 81.62 | 76.83 | 18.02 | 47.00 |
| production--compact_boost--legacy--evidence--budget-40 | source-group-009 | 93.23 | 100.00 | 96.49 | 95.30 | 88.89 | 90.09 | 89.49 | 5.12 | 15.00 |
| production--compact_boost--legacy--evidence--budget-40 | source-group-012 | 89.16 | 100.00 | 94.27 | 77.48 | 83.16 | 88.10 | 85.55 | 6.15 | 16.33 |
| production--compact_boost--local_events--chronological--budget-05 | source-group-005 | 72.64 | 99.13 | 83.84 | 66.91 | 65.96 | 72.94 | 69.27 | 1.92 | 6.00 |
| production--compact_boost--local_events--chronological--budget-05 | source-group-007 | 60.88 | 98.95 | 75.38 | 56.52 | 52.57 | 60.44 | 56.23 | 2.06 | 4.67 |
| production--compact_boost--local_events--chronological--budget-05 | source-group-009 | 91.04 | 100.00 | 95.31 | 90.67 | 88.16 | 90.54 | 89.33 | 1.41 | 4.00 |
| production--compact_boost--local_events--chronological--budget-05 | source-group-012 | 79.68 | 98.70 | 88.17 | 65.00 | 67.19 | 76.79 | 71.67 | 1.07 | 4.00 |
| production--compact_boost--local_events--chronological--budget-10 | source-group-005 | 74.11 | 99.13 | 84.81 | 69.74 | 68.31 | 76.08 | 71.98 | 3.74 | 12.33 |
| production--compact_boost--local_events--chronological--budget-10 | source-group-007 | 62.31 | 98.95 | 76.46 | 58.59 | 56.72 | 65.73 | 60.89 | 4.40 | 13.00 |
| production--compact_boost--local_events--chronological--budget-10 | source-group-009 | 91.65 | 100.00 | 95.64 | 92.62 | 90.67 | 91.89 | 91.28 | 2.94 | 9.00 |
| production--compact_boost--local_events--chronological--budget-10 | source-group-012 | 80.71 | 98.70 | 88.81 | 67.78 | 67.19 | 76.79 | 71.67 | 2.13 | 8.00 |
| production--compact_boost--local_events--chronological--budget-20 | source-group-005 | 76.04 | 99.31 | 86.13 | 73.33 | 72.14 | 79.22 | 75.51 | 7.81 | 26.00 |
| production--compact_boost--local_events--chronological--budget-20 | source-group-007 | 65.80 | 98.95 | 79.04 | 64.46 | 61.41 | 70.40 | 65.60 | 8.85 | 27.33 |
| production--compact_boost--local_events--chronological--budget-20 | source-group-009 | 92.80 | 100.00 | 96.27 | 93.24 | 94.59 | 94.59 | 94.59 | 6.00 | 18.00 |
| production--compact_boost--local_events--chronological--budget-20 | source-group-012 | 82.31 | 99.41 | 90.06 | 69.44 | 68.75 | 78.57 | 73.33 | 4.39 | 16.00 |
| production--compact_boost--local_events--chronological--budget-40 | source-group-005 | 82.49 | 99.31 | 90.13 | 84.47 | 82.40 | 86.27 | 84.29 | 15.64 | 49.00 |
| production--compact_boost--local_events--chronological--budget-40 | source-group-007 | 76.15 | 99.28 | 86.19 | 76.54 | 75.00 | 81.31 | 78.02 | 17.97 | 52.00 |
| production--compact_boost--local_events--chronological--budget-40 | source-group-009 | 95.05 | 100.00 | 97.46 | 94.59 | 95.95 | 95.95 | 95.95 | 12.11 | 39.00 |
| production--compact_boost--local_events--chronological--budget-40 | source-group-012 | 86.26 | 99.41 | 92.37 | 78.51 | 76.67 | 82.14 | 79.31 | 8.85 | 29.67 |
| production--compact_boost--local_events--evidence--budget-05 | source-group-005 | 72.54 | 99.13 | 83.77 | 67.41 | 67.38 | 74.51 | 70.76 | 1.94 | 6.00 |
| production--compact_boost--local_events--evidence--budget-05 | source-group-007 | 59.95 | 99.09 | 74.70 | 56.73 | 52.27 | 61.06 | 56.33 | 2.16 | 7.67 |
| production--compact_boost--local_events--evidence--budget-05 | source-group-009 | 89.29 | 100.00 | 94.34 | 91.63 | 84.05 | 87.84 | 85.90 | 1.47 | 7.67 |
| production--compact_boost--local_events--evidence--budget-05 | source-group-012 | 79.47 | 99.80 | 88.48 | 65.56 | 68.59 | 77.98 | 72.98 | 1.05 | 3.33 |
| production--compact_boost--local_events--evidence--budget-10 | source-group-005 | 73.54 | 99.30 | 84.50 | 70.26 | 68.09 | 75.29 | 71.51 | 3.92 | 13.33 |
| production--compact_boost--local_events--evidence--budget-10 | source-group-007 | 62.05 | 99.11 | 76.32 | 60.54 | 54.70 | 63.55 | 58.79 | 4.48 | 16.33 |
| production--compact_boost--local_events--evidence--budget-10 | source-group-009 | 90.24 | 100.00 | 94.87 | 93.57 | 85.96 | 88.29 | 87.11 | 3.01 | 15.00 |
| production--compact_boost--local_events--evidence--budget-10 | source-group-012 | 80.30 | 99.80 | 89.00 | 70.45 | 72.13 | 78.57 | 75.21 | 2.18 | 8.33 |
| production--compact_boost--local_events--evidence--budget-20 | source-group-005 | 77.02 | 99.48 | 86.82 | 76.95 | 73.38 | 80.00 | 76.55 | 7.90 | 26.33 |
| production--compact_boost--local_events--evidence--budget-20 | source-group-007 | 66.89 | 99.11 | 79.87 | 65.81 | 60.06 | 69.78 | 64.56 | 9.02 | 31.67 |
| production--compact_boost--local_events--evidence--budget-20 | source-group-009 | 93.29 | 100.00 | 96.53 | 96.68 | 89.36 | 90.54 | 89.94 | 6.10 | 29.67 |
| production--compact_boost--local_events--evidence--budget-20 | source-group-012 | 83.25 | 100.00 | 90.86 | 75.93 | 80.33 | 85.12 | 82.65 | 4.43 | 19.00 |
| production--compact_boost--local_events--evidence--budget-40 | source-group-005 | 85.46 | 99.88 | 92.11 | 87.83 | 79.65 | 82.75 | 81.16 | 15.80 | 50.67 |
| production--compact_boost--local_events--evidence--budget-40 | source-group-007 | 78.48 | 99.26 | 87.65 | 78.96 | 69.78 | 76.95 | 73.18 | 18.10 | 55.67 |
| production--compact_boost--local_events--evidence--budget-40 | source-group-009 | 96.14 | 100.00 | 98.03 | 97.12 | 92.91 | 94.14 | 93.51 | 12.22 | 57.00 |
| production--compact_boost--local_events--evidence--budget-40 | source-group-012 | 92.37 | 100.00 | 96.03 | 91.01 | 92.93 | 94.05 | 93.47 | 8.87 | 34.67 |
| production--compact_boost--local_heads--chronological--budget-05 | source-group-005 | 72.60 | 99.13 | 83.82 | 67.41 | 66.08 | 73.33 | 69.52 | 1.94 | 6.33 |
| production--compact_boost--local_heads--chronological--budget-05 | source-group-007 | 60.88 | 98.95 | 75.38 | 56.52 | 52.57 | 60.44 | 56.23 | 2.06 | 4.67 |
| production--compact_boost--local_heads--chronological--budget-05 | source-group-009 | 91.04 | 100.00 | 95.31 | 90.67 | 88.16 | 90.54 | 89.33 | 1.42 | 4.00 |
| production--compact_boost--local_heads--chronological--budget-05 | source-group-012 | 79.68 | 98.70 | 88.17 | 65.00 | 67.19 | 76.79 | 71.67 | 1.09 | 4.00 |
| production--compact_boost--local_heads--chronological--budget-10 | source-group-005 | 74.04 | 99.13 | 84.77 | 69.74 | 68.31 | 76.08 | 71.98 | 3.77 | 12.33 |
| production--compact_boost--local_heads--chronological--budget-10 | source-group-007 | 62.03 | 98.95 | 76.26 | 58.42 | 56.72 | 65.73 | 60.89 | 4.39 | 13.00 |
| production--compact_boost--local_heads--chronological--budget-10 | source-group-009 | 91.65 | 100.00 | 95.64 | 92.62 | 90.67 | 91.89 | 91.28 | 2.93 | 9.00 |
| production--compact_boost--local_heads--chronological--budget-10 | source-group-012 | 80.71 | 98.70 | 88.81 | 67.78 | 67.19 | 76.79 | 71.67 | 2.15 | 8.00 |
| production--compact_boost--local_heads--chronological--budget-20 | source-group-005 | 75.98 | 99.31 | 86.10 | 73.09 | 72.14 | 79.22 | 75.51 | 7.75 | 25.33 |
| production--compact_boost--local_heads--chronological--budget-20 | source-group-007 | 65.84 | 98.95 | 79.07 | 64.17 | 61.45 | 71.03 | 65.89 | 8.89 | 27.33 |
| production--compact_boost--local_heads--chronological--budget-20 | source-group-009 | 92.80 | 100.00 | 96.26 | 93.24 | 94.59 | 94.59 | 94.59 | 6.03 | 18.00 |
| production--compact_boost--local_heads--chronological--budget-20 | source-group-012 | 82.31 | 99.41 | 90.06 | 69.44 | 68.75 | 78.57 | 73.33 | 4.42 | 16.00 |
| production--compact_boost--local_heads--chronological--budget-40 | source-group-005 | 82.46 | 99.31 | 90.11 | 84.15 | 82.16 | 86.67 | 84.35 | 15.69 | 49.00 |
| production--compact_boost--local_heads--chronological--budget-40 | source-group-007 | 76.47 | 99.51 | 86.48 | 76.61 | 75.29 | 82.55 | 78.75 | 18.00 | 51.67 |
| production--compact_boost--local_heads--chronological--budget-40 | source-group-009 | 95.05 | 100.00 | 97.46 | 94.59 | 95.95 | 95.95 | 95.95 | 12.16 | 39.33 |
| production--compact_boost--local_heads--chronological--budget-40 | source-group-012 | 86.17 | 99.41 | 92.32 | 78.51 | 76.67 | 82.14 | 79.31 | 8.81 | 29.33 |
| production--compact_boost--local_heads--evidence--budget-05 | source-group-005 | 72.75 | 99.12 | 83.91 | 67.53 | 67.86 | 74.51 | 71.03 | 1.96 | 5.67 |
| production--compact_boost--local_heads--evidence--budget-05 | source-group-007 | 59.89 | 98.93 | 74.61 | 57.39 | 53.04 | 62.62 | 57.43 | 2.19 | 8.33 |
| production--compact_boost--local_heads--evidence--budget-05 | source-group-009 | 89.82 | 100.00 | 94.63 | 90.79 | 87.14 | 91.44 | 89.24 | 1.48 | 7.33 |
| production--compact_boost--local_heads--evidence--budget-05 | source-group-012 | 79.76 | 100.00 | 88.74 | 67.59 | 67.71 | 77.38 | 72.22 | 1.10 | 4.00 |
| production--compact_boost--local_heads--evidence--budget-10 | source-group-005 | 73.95 | 99.20 | 84.73 | 69.90 | 69.43 | 75.69 | 72.42 | 3.94 | 13.33 |
| production--compact_boost--local_heads--evidence--budget-10 | source-group-007 | 60.47 | 98.93 | 75.06 | 59.16 | 56.18 | 66.67 | 60.97 | 4.51 | 19.33 |
| production--compact_boost--local_heads--evidence--budget-10 | source-group-009 | 90.67 | 100.00 | 95.11 | 91.91 | 88.81 | 92.79 | 90.75 | 3.04 | 16.00 |
| production--compact_boost--local_heads--evidence--budget-10 | source-group-012 | 80.58 | 100.00 | 89.24 | 71.80 | 73.42 | 83.93 | 78.31 | 2.21 | 9.67 |
| production--compact_boost--local_heads--evidence--budget-20 | source-group-005 | 76.30 | 99.28 | 86.29 | 75.51 | 77.98 | 84.71 | 81.21 | 7.91 | 30.67 |
| production--compact_boost--local_heads--evidence--budget-20 | source-group-007 | 63.38 | 98.94 | 77.26 | 64.34 | 64.06 | 76.64 | 69.79 | 9.06 | 40.33 |
| production--compact_boost--local_heads--evidence--budget-20 | source-group-009 | 91.66 | 100.00 | 95.65 | 92.78 | 91.46 | 96.40 | 93.86 | 6.08 | 33.67 |
| production--compact_boost--local_heads--evidence--budget-20 | source-group-012 | 82.07 | 100.00 | 90.15 | 76.13 | 76.01 | 86.90 | 81.09 | 4.44 | 22.33 |
| production--compact_boost--local_heads--evidence--budget-40 | source-group-005 | 81.29 | 99.49 | 89.47 | 81.71 | 85.47 | 92.16 | 88.68 | 15.82 | 61.67 |
| production--compact_boost--local_heads--evidence--budget-40 | source-group-007 | 68.72 | 99.28 | 81.22 | 70.47 | 74.11 | 89.10 | 80.91 | 18.14 | 78.00 |
| production--compact_boost--local_heads--evidence--budget-40 | source-group-009 | 94.32 | 100.00 | 97.07 | 94.54 | 93.95 | 97.75 | 95.81 | 12.20 | 60.33 |
| production--compact_boost--local_heads--evidence--budget-40 | source-group-012 | 85.50 | 100.00 | 92.18 | 85.01 | 81.16 | 92.26 | 86.35 | 8.89 | 43.33 |
| production--dino_global--legacy--chronological--budget-05 | source-group-005 | 73.02 | 99.38 | 84.18 | 68.66 | 68.23 | 74.90 | 71.41 | 1.85 | 5.00 |
| production--dino_global--legacy--chronological--budget-05 | source-group-007 | 60.89 | 98.95 | 75.39 | 56.52 | 52.85 | 60.75 | 56.52 | 2.07 | 5.00 |
| production--dino_global--legacy--chronological--budget-05 | source-group-009 | 91.24 | 100.00 | 95.42 | 91.56 | 85.96 | 88.29 | 87.11 | 1.23 | 3.00 |
| production--dino_global--legacy--chronological--budget-05 | source-group-012 | 80.82 | 99.41 | 89.16 | 67.23 | 68.25 | 76.79 | 72.27 | 1.07 | 3.00 |
| production--dino_global--legacy--chronological--budget-10 | source-group-005 | 74.84 | 99.38 | 85.38 | 72.25 | 71.17 | 78.43 | 74.63 | 3.86 | 11.00 |
| production--dino_global--legacy--chronological--budget-10 | source-group-007 | 62.32 | 98.95 | 76.48 | 59.25 | 55.26 | 63.86 | 59.25 | 4.26 | 11.67 |
| production--dino_global--legacy--chronological--budget-10 | source-group-009 | 92.11 | 100.00 | 95.90 | 93.51 | 87.56 | 88.74 | 88.14 | 2.90 | 8.00 |
| production--dino_global--legacy--chronological--budget-10 | source-group-012 | 82.30 | 99.61 | 90.13 | 69.69 | 70.27 | 77.38 | 73.65 | 2.15 | 5.33 |
| production--dino_global--legacy--chronological--budget-20 | source-group-005 | 79.21 | 99.38 | 88.15 | 80.75 | 76.74 | 82.75 | 79.63 | 7.73 | 21.67 |
| production--dino_global--legacy--chronological--budget-20 | source-group-007 | 67.49 | 99.06 | 80.28 | 67.54 | 62.81 | 71.03 | 66.67 | 8.84 | 24.00 |
| production--dino_global--legacy--chronological--budget-20 | source-group-009 | 93.70 | 100.00 | 96.75 | 95.75 | 90.22 | 91.44 | 90.83 | 5.14 | 15.33 |
| production--dino_global--legacy--chronological--budget-20 | source-group-012 | 86.41 | 100.00 | 92.71 | 75.78 | 78.58 | 85.12 | 81.72 | 4.37 | 11.33 |
| production--dino_global--legacy--chronological--budget-40 | source-group-005 | 88.21 | 99.65 | 93.57 | 87.07 | 87.47 | 87.45 | 87.46 | 14.52 | 37.33 |
| production--dino_global--legacy--chronological--budget-40 | source-group-007 | 80.11 | 99.45 | 88.74 | 80.54 | 78.06 | 85.36 | 81.55 | 17.79 | 47.00 |
| production--dino_global--legacy--chronological--budget-40 | source-group-009 | 93.91 | 100.00 | 96.86 | 95.75 | 90.22 | 91.44 | 90.83 | 5.70 | 17.33 |
| production--dino_global--legacy--chronological--budget-40 | source-group-012 | 89.52 | 100.00 | 94.47 | 80.47 | 85.63 | 88.69 | 87.13 | 6.43 | 15.33 |
| production--dino_global--legacy--evidence--budget-05 | source-group-005 | 73.89 | 99.13 | 84.67 | 70.00 | 66.67 | 74.51 | 70.37 | 1.88 | 5.00 |
| production--dino_global--legacy--evidence--budget-05 | source-group-007 | 60.73 | 98.93 | 75.26 | 56.36 | 52.02 | 60.12 | 55.78 | 2.13 | 4.67 |
| production--dino_global--legacy--evidence--budget-05 | source-group-009 | 90.10 | 100.00 | 94.79 | 90.07 | 85.71 | 89.19 | 87.42 | 1.41 | 3.00 |
| production--dino_global--legacy--evidence--budget-05 | source-group-012 | 79.79 | 99.18 | 88.43 | 63.33 | 68.75 | 78.57 | 73.33 | 1.08 | 2.00 |
| production--dino_global--legacy--evidence--budget-10 | source-group-005 | 75.97 | 99.21 | 86.05 | 72.25 | 70.48 | 77.65 | 73.88 | 3.87 | 10.00 |
| production--dino_global--legacy--evidence--budget-10 | source-group-007 | 62.65 | 98.93 | 76.71 | 59.25 | 54.45 | 62.93 | 58.38 | 4.41 | 10.00 |
| production--dino_global--legacy--evidence--budget-10 | source-group-009 | 91.13 | 100.00 | 95.36 | 91.15 | 87.83 | 90.99 | 89.38 | 2.92 | 7.67 |
| production--dino_global--legacy--evidence--budget-10 | source-group-012 | 80.37 | 99.41 | 88.88 | 65.18 | 70.23 | 78.57 | 74.16 | 2.17 | 4.33 |
| production--dino_global--legacy--evidence--budget-20 | source-group-005 | 80.42 | 99.25 | 88.85 | 76.76 | 76.63 | 83.53 | 79.92 | 7.75 | 21.00 |
| production--dino_global--legacy--evidence--budget-20 | source-group-007 | 67.00 | 98.93 | 79.89 | 64.84 | 61.56 | 71.34 | 66.09 | 8.99 | 22.00 |
| production--dino_global--legacy--evidence--budget-20 | source-group-009 | 93.20 | 100.00 | 96.48 | 94.45 | 89.43 | 91.44 | 90.42 | 5.14 | 15.33 |
| production--dino_global--legacy--evidence--budget-20 | source-group-012 | 83.91 | 99.80 | 91.17 | 71.81 | 76.40 | 82.74 | 79.44 | 4.37 | 10.67 |
| production--dino_global--legacy--evidence--budget-40 | source-group-005 | 89.25 | 99.65 | 94.15 | 87.46 | 88.63 | 88.24 | 88.43 | 14.54 | 36.67 |
| production--dino_global--legacy--evidence--budget-40 | source-group-007 | 78.87 | 99.14 | 87.85 | 79.11 | 74.04 | 83.49 | 78.47 | 18.00 | 46.00 |
| production--dino_global--legacy--evidence--budget-40 | source-group-009 | 93.91 | 100.00 | 96.86 | 95.75 | 90.22 | 91.44 | 90.83 | 5.70 | 17.33 |
| production--dino_global--legacy--evidence--budget-40 | source-group-012 | 89.52 | 100.00 | 94.47 | 80.47 | 85.63 | 88.69 | 87.13 | 6.43 | 15.33 |
| production--dino_global--local_events--chronological--budget-05 | source-group-005 | 72.67 | 99.13 | 83.86 | 66.91 | 65.96 | 72.94 | 69.28 | 1.92 | 6.00 |
| production--dino_global--local_events--chronological--budget-05 | source-group-007 | 60.91 | 98.95 | 75.41 | 56.52 | 52.85 | 60.75 | 56.52 | 2.08 | 5.00 |
| production--dino_global--local_events--chronological--budget-05 | source-group-009 | 91.04 | 100.00 | 95.31 | 90.67 | 88.16 | 90.54 | 89.33 | 1.40 | 4.00 |
| production--dino_global--local_events--chronological--budget-05 | source-group-012 | 79.68 | 98.70 | 88.17 | 65.00 | 67.19 | 76.79 | 71.67 | 1.04 | 4.00 |
| production--dino_global--local_events--chronological--budget-10 | source-group-005 | 74.10 | 99.13 | 84.81 | 69.50 | 67.96 | 75.69 | 71.61 | 3.72 | 11.67 |
| production--dino_global--local_events--chronological--budget-10 | source-group-007 | 62.19 | 98.95 | 76.38 | 58.30 | 56.99 | 66.04 | 61.18 | 4.37 | 13.67 |
| production--dino_global--local_events--chronological--budget-10 | source-group-009 | 91.66 | 100.00 | 95.65 | 92.62 | 90.67 | 91.89 | 91.28 | 2.93 | 9.00 |
| production--dino_global--local_events--chronological--budget-10 | source-group-012 | 80.76 | 98.70 | 88.84 | 67.22 | 67.19 | 76.79 | 71.67 | 2.09 | 8.00 |
| production--dino_global--local_events--chronological--budget-20 | source-group-005 | 76.48 | 99.41 | 86.45 | 74.91 | 72.60 | 80.00 | 76.12 | 7.82 | 26.00 |
| production--dino_global--local_events--chronological--budget-20 | source-group-007 | 65.94 | 98.95 | 79.14 | 64.94 | 61.56 | 71.34 | 66.09 | 8.88 | 28.67 |
| production--dino_global--local_events--chronological--budget-20 | source-group-009 | 92.80 | 100.00 | 96.27 | 93.24 | 94.59 | 94.59 | 94.59 | 5.98 | 18.33 |
| production--dino_global--local_events--chronological--budget-20 | source-group-012 | 82.54 | 99.41 | 90.19 | 69.08 | 69.11 | 78.57 | 73.54 | 4.38 | 15.67 |
| production--dino_global--local_events--chronological--budget-40 | source-group-005 | 83.74 | 99.41 | 90.90 | 86.53 | 83.46 | 87.06 | 85.22 | 15.75 | 50.33 |
| production--dino_global--local_events--chronological--budget-40 | source-group-007 | 77.41 | 99.45 | 87.06 | 76.51 | 75.43 | 82.24 | 78.69 | 17.94 | 52.67 |
| production--dino_global--local_events--chronological--budget-40 | source-group-009 | 95.07 | 100.00 | 97.47 | 94.59 | 95.95 | 95.95 | 95.95 | 12.14 | 39.67 |
| production--dino_global--local_events--chronological--budget-40 | source-group-012 | 86.07 | 99.41 | 92.26 | 77.71 | 76.67 | 82.14 | 79.31 | 8.84 | 29.33 |
| production--dino_global--local_events--evidence--budget-05 | source-group-005 | 72.50 | 99.30 | 83.81 | 69.61 | 66.78 | 74.90 | 70.61 | 1.92 | 7.33 |
| production--dino_global--local_events--evidence--budget-05 | source-group-007 | 60.59 | 99.10 | 75.20 | 57.19 | 51.32 | 60.75 | 55.64 | 2.22 | 7.67 |
| production--dino_global--local_events--evidence--budget-05 | source-group-009 | 89.59 | 100.00 | 94.51 | 91.59 | 84.79 | 87.84 | 86.28 | 1.44 | 7.00 |
| production--dino_global--local_events--evidence--budget-05 | source-group-012 | 79.55 | 100.00 | 88.61 | 65.56 | 68.75 | 78.57 | 73.33 | 1.08 | 3.33 |
| production--dino_global--local_events--evidence--budget-10 | source-group-005 | 73.31 | 99.66 | 84.48 | 71.78 | 67.36 | 76.08 | 71.45 | 3.93 | 15.33 |
| production--dino_global--local_events--evidence--budget-10 | source-group-007 | 62.06 | 99.27 | 76.38 | 60.31 | 53.14 | 63.24 | 57.75 | 4.46 | 17.00 |
| production--dino_global--local_events--evidence--budget-10 | source-group-009 | 90.78 | 100.00 | 95.17 | 93.09 | 87.23 | 89.19 | 88.20 | 3.01 | 14.00 |
| production--dino_global--local_events--evidence--budget-10 | source-group-012 | 80.49 | 100.00 | 89.19 | 72.53 | 74.59 | 80.36 | 77.36 | 2.22 | 7.00 |
| production--dino_global--local_events--evidence--budget-20 | source-group-005 | 77.32 | 99.90 | 87.15 | 77.27 | 70.92 | 78.43 | 74.48 | 7.89 | 28.33 |
| production--dino_global--local_events--evidence--budget-20 | source-group-007 | 66.91 | 99.36 | 79.97 | 66.29 | 56.41 | 65.73 | 60.71 | 9.05 | 32.00 |
| production--dino_global--local_events--evidence--budget-20 | source-group-009 | 93.35 | 100.00 | 96.56 | 95.76 | 89.00 | 90.99 | 89.98 | 6.10 | 29.67 |
| production--dino_global--local_events--evidence--budget-20 | source-group-012 | 81.44 | 100.00 | 89.77 | 76.00 | 76.72 | 80.36 | 78.49 | 4.44 | 19.33 |
| production--dino_global--local_events--evidence--budget-40 | source-group-005 | 88.16 | 99.98 | 93.66 | 89.13 | 83.16 | 83.92 | 83.49 | 15.80 | 53.00 |
| production--dino_global--local_events--evidence--budget-40 | source-group-007 | 78.78 | 99.53 | 87.95 | 82.53 | 67.98 | 74.77 | 71.21 | 18.12 | 60.67 |
| production--dino_global--local_events--evidence--budget-40 | source-group-009 | 96.95 | 100.00 | 98.45 | 98.24 | 93.03 | 95.95 | 94.46 | 12.24 | 57.67 |
| production--dino_global--local_events--evidence--budget-40 | source-group-012 | 92.36 | 100.00 | 96.02 | 91.45 | 93.40 | 92.86 | 93.13 | 8.89 | 33.67 |
| production--dino_global--local_heads--chronological--budget-05 | source-group-005 | 72.60 | 99.13 | 83.82 | 67.04 | 66.08 | 73.33 | 69.52 | 1.90 | 5.67 |
| production--dino_global--local_heads--chronological--budget-05 | source-group-007 | 60.91 | 98.95 | 75.41 | 56.52 | 52.85 | 60.75 | 56.52 | 2.10 | 5.00 |
| production--dino_global--local_heads--chronological--budget-05 | source-group-009 | 91.07 | 100.00 | 95.33 | 90.67 | 88.16 | 90.54 | 89.33 | 1.43 | 4.00 |
| production--dino_global--local_heads--chronological--budget-05 | source-group-012 | 79.68 | 98.70 | 88.17 | 65.00 | 67.19 | 76.79 | 71.67 | 1.04 | 4.00 |
| production--dino_global--local_heads--chronological--budget-10 | source-group-005 | 74.15 | 99.13 | 84.84 | 69.63 | 67.96 | 75.69 | 71.61 | 3.75 | 11.67 |
| production--dino_global--local_heads--chronological--budget-10 | source-group-007 | 62.04 | 98.95 | 76.26 | 58.42 | 56.72 | 65.73 | 60.89 | 4.36 | 13.33 |
| production--dino_global--local_heads--chronological--budget-10 | source-group-009 | 91.66 | 100.00 | 95.65 | 92.62 | 90.67 | 91.89 | 91.28 | 2.94 | 9.00 |
| production--dino_global--local_heads--chronological--budget-10 | source-group-012 | 80.76 | 98.70 | 88.84 | 67.22 | 67.19 | 76.79 | 71.67 | 2.10 | 8.00 |
| production--dino_global--local_heads--chronological--budget-20 | source-group-005 | 76.50 | 99.41 | 86.46 | 75.05 | 72.24 | 79.61 | 75.75 | 7.76 | 25.00 |
| production--dino_global--local_heads--chronological--budget-20 | source-group-007 | 65.98 | 98.95 | 79.17 | 64.75 | 61.77 | 71.96 | 66.48 | 8.86 | 28.00 |
| production--dino_global--local_heads--chronological--budget-20 | source-group-009 | 92.80 | 100.00 | 96.26 | 93.24 | 94.59 | 94.59 | 94.59 | 5.96 | 18.00 |
| production--dino_global--local_heads--chronological--budget-20 | source-group-012 | 82.54 | 99.41 | 90.19 | 69.08 | 69.11 | 78.57 | 73.54 | 4.40 | 15.67 |
| production--dino_global--local_heads--chronological--budget-40 | source-group-005 | 83.60 | 99.41 | 90.82 | 86.20 | 83.52 | 87.45 | 85.44 | 15.75 | 50.00 |
| production--dino_global--local_heads--chronological--budget-40 | source-group-007 | 77.26 | 99.45 | 86.96 | 76.47 | 75.28 | 82.55 | 78.75 | 17.97 | 52.67 |
| production--dino_global--local_heads--chronological--budget-40 | source-group-009 | 95.01 | 100.00 | 97.44 | 94.59 | 95.95 | 95.95 | 95.95 | 12.08 | 38.67 |
| production--dino_global--local_heads--chronological--budget-40 | source-group-012 | 86.01 | 99.41 | 92.23 | 77.93 | 76.67 | 82.14 | 79.31 | 8.85 | 29.33 |
| production--dino_global--local_heads--evidence--budget-05 | source-group-005 | 72.15 | 99.21 | 83.55 | 67.53 | 67.02 | 74.90 | 70.74 | 1.94 | 7.33 |
| production--dino_global--local_heads--evidence--budget-05 | source-group-007 | 60.30 | 99.10 | 74.98 | 56.70 | 51.83 | 61.68 | 56.33 | 2.25 | 9.00 |
| production--dino_global--local_heads--evidence--budget-05 | source-group-009 | 89.97 | 100.00 | 94.72 | 90.39 | 86.03 | 91.44 | 88.65 | 1.48 | 7.67 |
| production--dino_global--local_heads--evidence--budget-05 | source-group-012 | 79.65 | 99.80 | 88.59 | 65.55 | 68.23 | 77.98 | 72.78 | 1.09 | 3.67 |
| production--dino_global--local_heads--evidence--budget-10 | source-group-005 | 73.02 | 99.72 | 84.31 | 71.15 | 68.06 | 76.86 | 72.19 | 3.91 | 15.33 |
| production--dino_global--local_heads--evidence--budget-10 | source-group-007 | 62.00 | 99.10 | 76.28 | 60.03 | 54.52 | 65.73 | 59.60 | 4.48 | 18.00 |
| production--dino_global--local_heads--evidence--budget-10 | source-group-009 | 90.76 | 100.00 | 95.16 | 92.11 | 88.89 | 93.69 | 91.23 | 3.03 | 16.00 |
| production--dino_global--local_heads--evidence--budget-10 | source-group-012 | 80.53 | 100.00 | 89.22 | 72.32 | 72.93 | 78.57 | 75.65 | 2.22 | 7.33 |
| production--dino_global--local_heads--evidence--budget-20 | source-group-005 | 74.89 | 99.80 | 85.57 | 75.33 | 72.65 | 82.35 | 77.20 | 7.90 | 34.00 |
| production--dino_global--local_heads--evidence--budget-20 | source-group-007 | 64.35 | 99.27 | 78.08 | 63.62 | 60.71 | 74.14 | 66.76 | 9.05 | 38.00 |
| production--dino_global--local_heads--evidence--budget-20 | source-group-009 | 92.16 | 100.00 | 95.92 | 93.01 | 90.60 | 95.50 | 92.98 | 6.09 | 32.67 |
| production--dino_global--local_heads--evidence--budget-20 | source-group-012 | 82.20 | 100.00 | 90.23 | 76.84 | 78.21 | 83.33 | 80.69 | 4.44 | 18.67 |
| production--dino_global--local_heads--evidence--budget-40 | source-group-005 | 78.84 | 99.90 | 88.13 | 82.03 | 80.89 | 91.37 | 85.81 | 15.82 | 68.00 |
| production--dino_global--local_heads--evidence--budget-40 | source-group-007 | 70.42 | 99.37 | 82.42 | 72.03 | 71.03 | 86.29 | 77.92 | 18.15 | 73.00 |
| production--dino_global--local_heads--evidence--budget-40 | source-group-009 | 93.81 | 100.00 | 96.80 | 93.68 | 91.55 | 97.30 | 94.33 | 12.24 | 59.00 |
| production--dino_global--local_heads--evidence--budget-40 | source-group-012 | 84.61 | 100.00 | 91.66 | 80.66 | 79.67 | 86.31 | 82.86 | 8.89 | 38.67 |
| production--dino_boost--legacy--chronological--budget-05 | source-group-005 | 72.41 | 99.22 | 83.72 | 67.53 | 66.43 | 73.73 | 69.89 | 1.79 | 4.67 |
| production--dino_boost--legacy--chronological--budget-05 | source-group-007 | 60.87 | 98.95 | 75.37 | 56.81 | 52.85 | 60.75 | 56.52 | 2.08 | 5.00 |
| production--dino_boost--legacy--chronological--budget-05 | source-group-009 | 91.28 | 100.00 | 95.44 | 91.56 | 85.96 | 88.29 | 87.11 | 1.27 | 2.67 |
| production--dino_boost--legacy--chronological--budget-05 | source-group-012 | 80.82 | 99.41 | 89.16 | 67.23 | 68.25 | 76.79 | 72.27 | 1.07 | 3.00 |
| production--dino_boost--legacy--chronological--budget-10 | source-group-005 | 74.23 | 99.31 | 84.96 | 70.50 | 70.92 | 78.43 | 74.49 | 3.81 | 10.67 |
| production--dino_boost--legacy--chronological--budget-10 | source-group-007 | 62.46 | 98.95 | 76.58 | 60.20 | 55.40 | 63.86 | 59.33 | 4.25 | 11.33 |
| production--dino_boost--legacy--chronological--budget-10 | source-group-009 | 92.05 | 100.00 | 95.86 | 93.51 | 87.11 | 88.29 | 87.70 | 2.91 | 7.67 |
| production--dino_boost--legacy--chronological--budget-10 | source-group-012 | 82.35 | 99.61 | 90.16 | 69.49 | 70.43 | 77.98 | 74.01 | 2.17 | 5.33 |
| production--dino_boost--legacy--chronological--budget-20 | source-group-005 | 78.16 | 99.47 | 87.53 | 77.20 | 77.74 | 83.53 | 80.53 | 7.85 | 20.33 |
| production--dino_boost--legacy--chronological--budget-20 | source-group-007 | 67.77 | 99.11 | 80.50 | 67.84 | 63.33 | 71.03 | 66.96 | 8.80 | 23.00 |
| production--dino_boost--legacy--chronological--budget-20 | source-group-009 | 93.81 | 100.00 | 96.81 | 94.41 | 91.11 | 92.34 | 91.72 | 5.67 | 16.67 |
| production--dino_boost--legacy--chronological--budget-20 | source-group-012 | 86.64 | 100.00 | 92.84 | 78.07 | 79.24 | 86.31 | 82.62 | 4.42 | 11.67 |
| production--dino_boost--legacy--chronological--budget-40 | source-group-005 | 87.44 | 99.87 | 93.22 | 85.87 | 88.51 | 87.45 | 87.97 | 14.08 | 34.67 |
| production--dino_boost--legacy--chronological--budget-40 | source-group-007 | 80.98 | 99.33 | 89.21 | 80.97 | 78.62 | 84.74 | 81.56 | 17.96 | 46.33 |
| production--dino_boost--legacy--chronological--budget-40 | source-group-009 | 94.47 | 100.00 | 97.16 | 96.20 | 91.11 | 92.34 | 91.72 | 6.90 | 21.00 |
| production--dino_boost--legacy--chronological--budget-40 | source-group-012 | 89.33 | 100.00 | 94.36 | 82.56 | 85.80 | 89.88 | 87.79 | 6.48 | 15.67 |
| production--dino_boost--legacy--evidence--budget-05 | source-group-005 | 73.83 | 99.13 | 84.63 | 69.64 | 68.35 | 75.29 | 71.65 | 1.84 | 4.33 |
| production--dino_boost--legacy--evidence--budget-05 | source-group-007 | 61.08 | 98.93 | 75.53 | 56.73 | 52.16 | 60.12 | 55.86 | 2.04 | 4.00 |
| production--dino_boost--legacy--evidence--budget-05 | source-group-009 | 89.91 | 100.00 | 94.69 | 90.07 | 85.71 | 89.19 | 87.42 | 1.39 | 2.67 |
| production--dino_boost--legacy--evidence--budget-05 | source-group-012 | 80.14 | 99.41 | 88.74 | 64.63 | 69.64 | 79.17 | 74.10 | 1.06 | 2.33 |
| production--dino_boost--legacy--evidence--budget-10 | source-group-005 | 75.99 | 99.17 | 86.04 | 72.55 | 72.01 | 77.65 | 74.72 | 3.81 | 9.33 |
| production--dino_boost--legacy--evidence--budget-10 | source-group-007 | 63.50 | 98.93 | 77.35 | 59.13 | 54.20 | 62.31 | 57.97 | 4.36 | 8.67 |
| production--dino_boost--legacy--evidence--budget-10 | source-group-009 | 90.82 | 100.00 | 95.19 | 90.27 | 87.83 | 90.99 | 89.38 | 2.93 | 6.67 |
| production--dino_boost--legacy--evidence--budget-10 | source-group-012 | 80.82 | 99.61 | 89.24 | 67.04 | 71.12 | 79.17 | 74.93 | 2.16 | 4.67 |
| production--dino_boost--legacy--evidence--budget-20 | source-group-005 | 80.96 | 99.42 | 89.24 | 77.68 | 79.82 | 81.96 | 80.86 | 7.79 | 18.00 |
| production--dino_boost--legacy--evidence--budget-20 | source-group-007 | 68.47 | 98.93 | 80.93 | 64.82 | 60.64 | 69.16 | 64.62 | 8.91 | 19.67 |
| production--dino_boost--legacy--evidence--budget-20 | source-group-009 | 92.89 | 100.00 | 96.31 | 92.90 | 89.48 | 91.89 | 90.67 | 5.59 | 16.00 |
| production--dino_boost--legacy--evidence--budget-20 | source-group-012 | 85.05 | 100.00 | 91.92 | 74.99 | 77.16 | 84.52 | 80.68 | 4.42 | 11.33 |
| production--dino_boost--legacy--evidence--budget-40 | source-group-005 | 87.32 | 99.88 | 93.16 | 85.53 | 88.59 | 88.24 | 88.41 | 14.16 | 34.67 |
| production--dino_boost--legacy--evidence--budget-40 | source-group-007 | 80.54 | 99.06 | 88.84 | 78.53 | 74.65 | 83.49 | 78.82 | 18.07 | 45.00 |
| production--dino_boost--legacy--evidence--budget-40 | source-group-009 | 94.47 | 100.00 | 97.16 | 96.20 | 91.11 | 92.34 | 91.72 | 6.90 | 21.00 |
| production--dino_boost--legacy--evidence--budget-40 | source-group-012 | 89.33 | 100.00 | 94.36 | 82.56 | 85.80 | 89.88 | 87.79 | 6.48 | 15.67 |
| production--dino_boost--local_events--chronological--budget-05 | source-group-005 | 72.45 | 99.13 | 83.72 | 66.67 | 65.61 | 73.33 | 69.26 | 1.93 | 6.33 |
| production--dino_boost--local_events--chronological--budget-05 | source-group-007 | 60.88 | 98.95 | 75.38 | 56.52 | 52.57 | 60.44 | 56.23 | 2.07 | 4.67 |
| production--dino_boost--local_events--chronological--budget-05 | source-group-009 | 91.04 | 100.00 | 95.31 | 90.67 | 88.16 | 90.54 | 89.33 | 1.40 | 4.00 |
| production--dino_boost--local_events--chronological--budget-05 | source-group-012 | 79.65 | 98.70 | 88.16 | 65.00 | 67.19 | 76.79 | 71.67 | 1.02 | 4.00 |
| production--dino_boost--local_events--chronological--budget-10 | source-group-005 | 73.70 | 99.31 | 84.61 | 69.24 | 68.42 | 76.47 | 72.22 | 3.86 | 12.67 |
| production--dino_boost--local_events--chronological--budget-10 | source-group-007 | 62.07 | 98.95 | 76.28 | 58.01 | 57.26 | 66.36 | 61.47 | 4.38 | 13.67 |
| production--dino_boost--local_events--chronological--budget-10 | source-group-009 | 91.65 | 100.00 | 95.64 | 92.62 | 90.67 | 91.89 | 91.28 | 2.93 | 9.00 |
| production--dino_boost--local_events--chronological--budget-10 | source-group-012 | 80.82 | 98.70 | 88.87 | 66.67 | 67.19 | 76.79 | 71.67 | 2.11 | 8.00 |
| production--dino_boost--local_events--chronological--budget-20 | source-group-005 | 75.79 | 99.41 | 86.01 | 73.09 | 72.14 | 79.22 | 75.51 | 7.77 | 25.00 |
| production--dino_boost--local_events--chronological--budget-20 | source-group-007 | 65.94 | 98.95 | 79.14 | 64.94 | 61.29 | 71.03 | 65.80 | 8.86 | 28.00 |
| production--dino_boost--local_events--chronological--budget-20 | source-group-009 | 92.79 | 100.00 | 96.26 | 93.24 | 94.59 | 94.59 | 94.59 | 5.97 | 17.67 |
| production--dino_boost--local_events--chronological--budget-20 | source-group-012 | 82.80 | 99.41 | 90.35 | 68.72 | 69.48 | 78.57 | 73.74 | 4.37 | 15.67 |
| production--dino_boost--local_events--chronological--budget-40 | source-group-005 | 82.03 | 99.41 | 89.88 | 84.52 | 82.99 | 87.84 | 85.34 | 15.73 | 50.00 |
| production--dino_boost--local_events--chronological--budget-40 | source-group-007 | 77.93 | 99.45 | 87.38 | 77.76 | 75.35 | 82.87 | 78.93 | 17.94 | 51.33 |
| production--dino_boost--local_events--chronological--budget-40 | source-group-009 | 95.03 | 100.00 | 97.45 | 94.59 | 95.95 | 95.95 | 95.95 | 12.11 | 38.67 |
| production--dino_boost--local_events--chronological--budget-40 | source-group-012 | 86.24 | 99.41 | 92.36 | 77.14 | 76.67 | 82.14 | 79.31 | 8.87 | 30.00 |
| production--dino_boost--local_events--evidence--budget-05 | source-group-005 | 72.27 | 99.13 | 83.59 | 66.67 | 67.02 | 74.12 | 70.39 | 1.92 | 5.33 |
| production--dino_boost--local_events--evidence--budget-05 | source-group-007 | 60.44 | 99.01 | 75.06 | 57.47 | 51.71 | 61.37 | 56.12 | 2.15 | 7.67 |
| production--dino_boost--local_events--evidence--budget-05 | source-group-009 | 89.52 | 100.00 | 94.47 | 91.80 | 85.16 | 87.84 | 86.48 | 1.39 | 6.33 |
| production--dino_boost--local_events--evidence--budget-05 | source-group-012 | 79.93 | 100.00 | 88.85 | 67.22 | 68.95 | 77.98 | 73.18 | 1.11 | 3.33 |
| production--dino_boost--local_events--evidence--budget-10 | source-group-005 | 73.32 | 99.68 | 84.49 | 69.39 | 67.96 | 75.69 | 71.61 | 3.93 | 12.33 |
| production--dino_boost--local_events--evidence--budget-10 | source-group-007 | 62.29 | 99.27 | 76.54 | 60.06 | 53.40 | 63.55 | 58.04 | 4.46 | 16.67 |
| production--dino_boost--local_events--evidence--budget-10 | source-group-009 | 91.12 | 100.00 | 95.36 | 93.10 | 86.79 | 88.74 | 87.75 | 3.03 | 13.00 |
| production--dino_boost--local_events--evidence--budget-10 | source-group-012 | 80.52 | 100.00 | 89.21 | 72.16 | 73.05 | 79.17 | 75.97 | 2.22 | 7.00 |
| production--dino_boost--local_events--evidence--budget-20 | source-group-005 | 75.19 | 99.72 | 85.74 | 74.29 | 69.12 | 77.25 | 72.96 | 7.90 | 28.00 |
| production--dino_boost--local_events--evidence--budget-20 | source-group-007 | 67.48 | 99.52 | 80.42 | 66.76 | 56.25 | 66.04 | 60.75 | 9.05 | 31.00 |
| production--dino_boost--local_events--evidence--budget-20 | source-group-009 | 93.59 | 100.00 | 96.69 | 95.32 | 88.54 | 90.54 | 89.52 | 6.11 | 28.00 |
| production--dino_boost--local_events--evidence--budget-20 | source-group-012 | 81.32 | 100.00 | 89.70 | 75.27 | 75.83 | 80.36 | 78.03 | 4.42 | 18.67 |
| production--dino_boost--local_events--evidence--budget-40 | source-group-005 | 83.33 | 99.98 | 90.89 | 86.03 | 79.36 | 84.31 | 81.75 | 15.80 | 52.33 |
| production--dino_boost--local_events--evidence--budget-40 | source-group-007 | 80.53 | 99.70 | 89.09 | 83.14 | 68.46 | 75.70 | 71.90 | 18.12 | 57.33 |
| production--dino_boost--local_events--evidence--budget-40 | source-group-009 | 97.22 | 100.00 | 98.59 | 98.67 | 93.37 | 95.05 | 94.20 | 12.24 | 58.00 |
| production--dino_boost--local_events--evidence--budget-40 | source-group-012 | 91.86 | 100.00 | 95.75 | 91.18 | 95.24 | 95.24 | 95.24 | 8.87 | 34.67 |
| production--dino_boost--local_heads--chronological--budget-05 | source-group-005 | 72.53 | 99.13 | 83.77 | 67.04 | 65.85 | 73.33 | 69.39 | 1.93 | 6.00 |
| production--dino_boost--local_heads--chronological--budget-05 | source-group-007 | 60.88 | 98.95 | 75.38 | 56.52 | 52.57 | 60.44 | 56.23 | 2.07 | 4.67 |
| production--dino_boost--local_heads--chronological--budget-05 | source-group-009 | 91.09 | 100.00 | 95.34 | 90.67 | 88.16 | 90.54 | 89.33 | 1.48 | 4.00 |
| production--dino_boost--local_heads--chronological--budget-05 | source-group-012 | 79.65 | 98.70 | 88.16 | 65.00 | 67.19 | 76.79 | 71.67 | 1.02 | 4.00 |
| production--dino_boost--local_heads--chronological--budget-10 | source-group-005 | 73.73 | 99.31 | 84.63 | 68.88 | 68.07 | 76.08 | 71.85 | 3.79 | 12.33 |
| production--dino_boost--local_heads--chronological--budget-10 | source-group-007 | 62.02 | 98.95 | 76.25 | 58.62 | 56.45 | 65.42 | 60.61 | 4.34 | 12.67 |
| production--dino_boost--local_heads--chronological--budget-10 | source-group-009 | 91.67 | 100.00 | 95.65 | 92.62 | 90.67 | 91.89 | 91.28 | 2.97 | 9.00 |
| production--dino_boost--local_heads--chronological--budget-10 | source-group-012 | 80.82 | 98.70 | 88.87 | 66.67 | 67.19 | 76.79 | 71.67 | 2.13 | 8.00 |
| production--dino_boost--local_heads--chronological--budget-20 | source-group-005 | 75.78 | 99.41 | 86.00 | 72.96 | 72.14 | 79.22 | 75.51 | 7.76 | 24.67 |
| production--dino_boost--local_heads--chronological--budget-20 | source-group-007 | 65.95 | 98.95 | 79.15 | 64.66 | 61.39 | 71.34 | 65.99 | 8.90 | 27.33 |
| production--dino_boost--local_heads--chronological--budget-20 | source-group-009 | 92.79 | 100.00 | 96.26 | 93.24 | 94.59 | 94.59 | 94.59 | 5.98 | 17.33 |
| production--dino_boost--local_heads--chronological--budget-20 | source-group-012 | 82.78 | 99.41 | 90.34 | 68.72 | 69.48 | 78.57 | 73.74 | 4.37 | 15.33 |
| production--dino_boost--local_heads--chronological--budget-40 | source-group-005 | 81.85 | 99.41 | 89.78 | 83.89 | 81.93 | 87.06 | 84.41 | 15.63 | 49.67 |
| production--dino_boost--local_heads--chronological--budget-40 | source-group-007 | 77.96 | 99.45 | 87.40 | 77.88 | 75.35 | 82.87 | 78.93 | 18.00 | 51.33 |
| production--dino_boost--local_heads--chronological--budget-40 | source-group-009 | 95.03 | 100.00 | 97.45 | 94.59 | 95.95 | 95.95 | 95.95 | 12.17 | 39.00 |
| production--dino_boost--local_heads--chronological--budget-40 | source-group-012 | 86.23 | 99.41 | 92.36 | 77.14 | 76.67 | 82.14 | 79.31 | 8.82 | 29.33 |
| production--dino_boost--local_heads--evidence--budget-05 | source-group-005 | 72.37 | 99.28 | 83.72 | 66.91 | 67.49 | 74.90 | 71.00 | 1.96 | 6.00 |
| production--dino_boost--local_heads--evidence--budget-05 | source-group-007 | 60.29 | 99.01 | 74.95 | 57.14 | 51.30 | 61.37 | 55.89 | 2.21 | 8.33 |
| production--dino_boost--local_heads--evidence--budget-05 | source-group-009 | 89.75 | 100.00 | 94.60 | 91.19 | 86.21 | 90.09 | 88.11 | 1.49 | 7.67 |
| production--dino_boost--local_heads--evidence--budget-05 | source-group-012 | 79.81 | 99.80 | 88.69 | 66.49 | 69.10 | 78.57 | 73.53 | 1.11 | 3.67 |
| production--dino_boost--local_heads--evidence--budget-10 | source-group-005 | 73.26 | 99.44 | 84.37 | 70.00 | 68.08 | 76.08 | 71.85 | 3.95 | 13.33 |
| production--dino_boost--local_heads--evidence--budget-10 | source-group-007 | 62.34 | 99.18 | 76.55 | 60.56 | 54.92 | 66.04 | 59.97 | 4.50 | 17.00 |
| production--dino_boost--local_heads--evidence--budget-10 | source-group-009 | 90.52 | 100.00 | 95.02 | 92.31 | 87.55 | 91.89 | 89.67 | 3.03 | 16.00 |
| production--dino_boost--local_heads--evidence--budget-10 | source-group-012 | 80.78 | 100.00 | 89.37 | 72.91 | 73.21 | 79.76 | 76.33 | 2.22 | 7.33 |
| production--dino_boost--local_heads--evidence--budget-20 | source-group-005 | 75.00 | 99.71 | 85.60 | 73.10 | 70.42 | 78.43 | 74.21 | 7.91 | 28.67 |
| production--dino_boost--local_heads--evidence--budget-20 | source-group-007 | 65.78 | 99.35 | 79.14 | 66.66 | 60.41 | 73.21 | 66.20 | 9.07 | 36.33 |
| production--dino_boost--local_heads--evidence--budget-20 | source-group-009 | 91.75 | 100.00 | 95.69 | 93.00 | 88.99 | 94.59 | 91.70 | 6.10 | 33.00 |
| production--dino_boost--local_heads--evidence--budget-20 | source-group-012 | 81.78 | 100.00 | 89.98 | 75.57 | 76.39 | 80.95 | 78.61 | 4.44 | 17.67 |
| production--dino_boost--local_heads--evidence--budget-40 | source-group-005 | 79.41 | 99.90 | 88.48 | 80.43 | 81.62 | 90.59 | 85.87 | 15.82 | 60.67 |
| production--dino_boost--local_heads--evidence--budget-40 | source-group-007 | 70.81 | 99.45 | 82.70 | 73.32 | 70.91 | 86.60 | 77.97 | 18.16 | 74.00 |
| production--dino_boost--local_heads--evidence--budget-40 | source-group-009 | 93.66 | 100.00 | 96.72 | 93.68 | 93.19 | 98.65 | 95.84 | 12.24 | 60.00 |
| production--dino_boost--local_heads--evidence--budget-40 | source-group-012 | 84.60 | 100.00 | 91.65 | 81.89 | 81.11 | 86.90 | 83.91 | 8.88 | 40.67 |
| individual--compact_boost--legacy--chronological--budget-05 | source-group-005 | 82.78 | 88.89 | 85.70 | 54.94 | 63.67 | 64.31 | 63.95 | 1.90 | 5.67 |
| individual--compact_boost--legacy--chronological--budget-05 | source-group-007 | 82.89 | 86.36 | 84.53 | 61.29 | 68.59 | 64.49 | 66.43 | 2.08 | 6.33 |
| individual--compact_boost--legacy--chronological--budget-05 | source-group-009 | 98.36 | 99.56 | 98.96 | 93.82 | 93.93 | 97.30 | 95.58 | 1.45 | 4.33 |
| individual--compact_boost--legacy--chronological--budget-05 | source-group-012 | 92.05 | 98.24 | 95.04 | 85.91 | 84.94 | 90.48 | 87.60 | 1.07 | 5.00 |
| individual--compact_boost--legacy--chronological--budget-10 | source-group-005 | 83.12 | 89.17 | 86.02 | 56.53 | 65.74 | 66.67 | 66.15 | 3.86 | 12.33 |
| individual--compact_boost--legacy--chronological--budget-10 | source-group-007 | 83.62 | 87.89 | 85.63 | 64.82 | 70.40 | 67.29 | 68.77 | 4.29 | 13.67 |
| individual--compact_boost--legacy--chronological--budget-10 | source-group-009 | 98.51 | 99.56 | 99.03 | 94.90 | 95.59 | 97.30 | 96.43 | 2.91 | 11.33 |
| individual--compact_boost--legacy--chronological--budget-10 | source-group-012 | 92.88 | 98.24 | 95.48 | 88.22 | 84.94 | 90.48 | 87.60 | 2.17 | 9.00 |
| individual--compact_boost--legacy--chronological--budget-20 | source-group-005 | 85.18 | 90.44 | 87.72 | 61.26 | 71.93 | 71.37 | 71.62 | 7.79 | 24.00 |
| individual--compact_boost--legacy--chronological--budget-20 | source-group-007 | 85.57 | 89.53 | 87.45 | 70.32 | 74.89 | 71.34 | 73.02 | 8.87 | 27.33 |
| individual--compact_boost--legacy--chronological--budget-20 | source-group-009 | 98.63 | 99.64 | 99.13 | 95.99 | 96.03 | 97.75 | 96.88 | 5.96 | 23.33 |
| individual--compact_boost--legacy--chronological--budget-20 | source-group-012 | 93.28 | 98.24 | 95.69 | 89.29 | 85.89 | 90.48 | 88.11 | 4.40 | 18.67 |
| individual--compact_boost--legacy--chronological--budget-40 | source-group-005 | 88.36 | 93.59 | 90.90 | 73.81 | 84.20 | 81.57 | 82.86 | 15.72 | 46.33 |
| individual--compact_boost--legacy--chronological--budget-40 | source-group-007 | 89.60 | 93.42 | 91.43 | 79.51 | 83.93 | 78.19 | 80.94 | 18.04 | 52.00 |
| individual--compact_boost--legacy--chronological--budget-40 | source-group-009 | 98.82 | 99.80 | 99.31 | 98.20 | 97.75 | 97.75 | 97.75 | 12.13 | 48.00 |
| individual--compact_boost--legacy--chronological--budget-40 | source-group-012 | 95.48 | 99.94 | 97.66 | 92.19 | 91.45 | 95.24 | 93.30 | 8.82 | 34.67 |
| individual--compact_boost--legacy--evidence--budget-05 | source-group-005 | 82.69 | 88.43 | 85.44 | 51.85 | 61.10 | 62.75 | 61.89 | 1.95 | 6.00 |
| individual--compact_boost--legacy--evidence--budget-05 | source-group-007 | 83.02 | 87.31 | 85.01 | 62.16 | 69.65 | 65.11 | 67.23 | 2.19 | 9.67 |
| individual--compact_boost--legacy--evidence--budget-05 | source-group-009 | 98.00 | 99.62 | 98.80 | 93.61 | 93.43 | 95.95 | 94.67 | 1.49 | 6.67 |
| individual--compact_boost--legacy--evidence--budget-05 | source-group-012 | 92.18 | 98.22 | 95.10 | 85.90 | 84.82 | 89.29 | 86.95 | 1.08 | 4.00 |
| individual--compact_boost--legacy--evidence--budget-10 | source-group-005 | 83.46 | 90.40 | 86.76 | 57.19 | 66.89 | 68.63 | 67.72 | 3.93 | 13.67 |
| individual--compact_boost--legacy--evidence--budget-10 | source-group-007 | 83.86 | 88.32 | 85.94 | 66.03 | 73.50 | 68.22 | 70.71 | 4.50 | 17.67 |
| individual--compact_boost--legacy--evidence--budget-10 | source-group-009 | 98.08 | 99.62 | 98.84 | 95.13 | 94.30 | 96.85 | 95.56 | 2.99 | 13.67 |
| individual--compact_boost--legacy--evidence--budget-10 | source-group-012 | 92.65 | 98.22 | 95.35 | 88.72 | 86.36 | 89.88 | 88.04 | 2.22 | 9.33 |
| individual--compact_boost--legacy--evidence--budget-20 | source-group-005 | 85.48 | 93.49 | 89.29 | 64.80 | 73.85 | 72.94 | 73.38 | 7.89 | 27.00 |
| individual--compact_boost--legacy--evidence--budget-20 | source-group-007 | 85.27 | 90.28 | 87.62 | 71.07 | 79.45 | 73.21 | 76.17 | 9.01 | 36.00 |
| individual--compact_boost--legacy--evidence--budget-20 | source-group-009 | 98.66 | 99.64 | 99.15 | 96.20 | 96.88 | 97.75 | 97.31 | 6.09 | 28.00 |
| individual--compact_boost--legacy--evidence--budget-20 | source-group-012 | 93.99 | 98.24 | 96.07 | 92.64 | 90.00 | 91.07 | 90.53 | 4.44 | 20.00 |
| individual--compact_boost--legacy--evidence--budget-40 | source-group-005 | 89.21 | 96.53 | 92.72 | 78.23 | 92.44 | 86.27 | 89.25 | 15.79 | 56.33 |
| individual--compact_boost--legacy--evidence--budget-40 | source-group-007 | 89.17 | 93.53 | 91.26 | 81.88 | 89.71 | 81.62 | 85.47 | 18.13 | 69.00 |
| individual--compact_boost--legacy--evidence--budget-40 | source-group-009 | 99.03 | 99.91 | 99.47 | 99.10 | 99.54 | 98.20 | 98.87 | 12.17 | 52.67 |
| individual--compact_boost--legacy--evidence--budget-40 | source-group-012 | 95.83 | 99.17 | 97.47 | 95.83 | 97.55 | 95.24 | 96.37 | 8.87 | 35.67 |
| individual--compact_boost--local_events--chronological--budget-05 | source-group-005 | 82.76 | 88.76 | 85.63 | 55.98 | 64.17 | 65.10 | 64.59 | 1.89 | 7.67 |
| individual--compact_boost--local_events--chronological--budget-05 | source-group-007 | 83.08 | 87.38 | 85.10 | 64.07 | 69.33 | 65.42 | 67.26 | 2.14 | 7.33 |
| individual--compact_boost--local_events--chronological--budget-05 | source-group-009 | 98.44 | 99.56 | 99.00 | 95.76 | 96.01 | 97.30 | 96.65 | 1.33 | 6.00 |
| individual--compact_boost--local_events--chronological--budget-05 | source-group-012 | 92.20 | 98.24 | 95.12 | 88.12 | 85.89 | 90.48 | 88.11 | 1.02 | 5.00 |
| individual--compact_boost--local_events--chronological--budget-10 | source-group-005 | 83.88 | 90.30 | 86.96 | 60.25 | 67.45 | 69.02 | 68.21 | 3.83 | 16.00 |
| individual--compact_boost--local_events--chronological--budget-10 | source-group-007 | 84.95 | 88.66 | 86.72 | 68.12 | 72.70 | 68.85 | 70.68 | 4.29 | 14.33 |
| individual--compact_boost--local_events--chronological--budget-10 | source-group-009 | 98.55 | 99.58 | 99.06 | 96.63 | 96.86 | 97.30 | 97.08 | 2.50 | 12.00 |
| individual--compact_boost--local_events--chronological--budget-10 | source-group-012 | 93.23 | 98.58 | 95.83 | 88.90 | 88.45 | 91.07 | 89.72 | 2.19 | 9.33 |
| individual--compact_boost--local_events--chronological--budget-20 | source-group-005 | 85.53 | 93.38 | 89.27 | 65.49 | 74.14 | 74.12 | 74.12 | 7.74 | 30.00 |
| individual--compact_boost--local_events--chronological--budget-20 | source-group-007 | 88.01 | 91.85 | 89.84 | 76.83 | 79.07 | 76.64 | 77.81 | 8.93 | 31.67 |
| individual--compact_boost--local_events--chronological--budget-20 | source-group-009 | 98.55 | 99.69 | 99.11 | 97.08 | 97.32 | 97.75 | 97.53 | 2.69 | 13.00 |
| individual--compact_boost--local_events--chronological--budget-20 | source-group-012 | 94.93 | 98.64 | 96.74 | 92.60 | 94.03 | 93.45 | 93.71 | 3.99 | 16.33 |
| individual--compact_boost--local_events--chronological--budget-40 | source-group-005 | 89.75 | 97.16 | 93.31 | 79.61 | 86.45 | 82.75 | 84.55 | 13.28 | 48.67 |
| individual--compact_boost--local_events--chronological--budget-40 | source-group-007 | 91.59 | 95.52 | 93.48 | 88.28 | 88.47 | 83.49 | 85.90 | 15.65 | 53.00 |
| individual--compact_boost--local_events--chronological--budget-40 | source-group-009 | 98.55 | 99.69 | 99.11 | 97.08 | 97.32 | 97.75 | 97.53 | 2.69 | 13.00 |
| individual--compact_boost--local_events--chronological--budget-40 | source-group-012 | 94.93 | 98.64 | 96.75 | 92.60 | 94.03 | 93.45 | 93.71 | 4.12 | 16.67 |
| individual--compact_boost--local_events--evidence--budget-05 | source-group-005 | 83.10 | 87.87 | 85.41 | 53.83 | 64.30 | 63.53 | 63.90 | 1.94 | 6.00 |
| individual--compact_boost--local_events--evidence--budget-05 | source-group-007 | 83.99 | 86.29 | 85.04 | 62.37 | 70.73 | 64.80 | 67.57 | 2.14 | 6.67 |
| individual--compact_boost--local_events--evidence--budget-05 | source-group-009 | 98.43 | 99.67 | 99.05 | 94.26 | 94.35 | 97.75 | 96.02 | 1.41 | 6.00 |
| individual--compact_boost--local_events--evidence--budget-05 | source-group-012 | 92.23 | 98.15 | 95.10 | 86.40 | 85.86 | 90.48 | 88.09 | 1.05 | 5.00 |
| individual--compact_boost--local_events--evidence--budget-10 | source-group-005 | 84.56 | 88.61 | 86.52 | 59.40 | 68.66 | 65.10 | 66.82 | 3.94 | 13.33 |
| individual--compact_boost--local_events--evidence--budget-10 | source-group-007 | 86.21 | 87.26 | 86.68 | 66.67 | 74.62 | 66.67 | 70.37 | 4.43 | 14.33 |
| individual--compact_boost--local_events--evidence--budget-10 | source-group-009 | 98.55 | 99.69 | 99.11 | 97.08 | 97.32 | 97.75 | 97.53 | 2.51 | 12.00 |
| individual--compact_boost--local_events--evidence--budget-10 | source-group-012 | 93.81 | 98.19 | 95.95 | 89.13 | 90.01 | 91.07 | 90.49 | 2.18 | 9.33 |
| individual--compact_boost--local_events--evidence--budget-20 | source-group-005 | 87.14 | 93.68 | 90.28 | 70.50 | 77.65 | 74.90 | 76.25 | 7.88 | 30.67 |
| individual--compact_boost--local_events--evidence--budget-20 | source-group-007 | 89.27 | 88.55 | 88.86 | 73.82 | 81.03 | 70.72 | 75.49 | 9.01 | 30.67 |
| individual--compact_boost--local_events--evidence--budget-20 | source-group-009 | 98.55 | 99.69 | 99.11 | 97.08 | 97.32 | 97.75 | 97.53 | 2.69 | 13.00 |
| individual--compact_boost--local_events--evidence--budget-20 | source-group-012 | 94.91 | 98.64 | 96.73 | 92.60 | 94.03 | 93.45 | 93.71 | 4.00 | 16.00 |
| individual--compact_boost--local_events--evidence--budget-40 | source-group-005 | 90.18 | 97.12 | 93.52 | 79.69 | 87.19 | 82.75 | 84.90 | 13.29 | 48.00 |
| individual--compact_boost--local_events--evidence--budget-40 | source-group-007 | 91.85 | 95.49 | 93.60 | 88.24 | 88.43 | 83.18 | 85.71 | 15.66 | 54.00 |
| individual--compact_boost--local_events--evidence--budget-40 | source-group-009 | 98.55 | 99.69 | 99.11 | 97.08 | 97.32 | 97.75 | 97.53 | 2.69 | 13.00 |
| individual--compact_boost--local_events--evidence--budget-40 | source-group-012 | 94.93 | 98.64 | 96.75 | 92.60 | 94.03 | 93.45 | 93.71 | 4.12 | 16.67 |
| individual--compact_boost--local_heads--chronological--budget-05 | source-group-005 | 82.85 | 88.57 | 85.59 | 55.20 | 64.03 | 64.71 | 64.32 | 1.88 | 7.67 |
| individual--compact_boost--local_heads--chronological--budget-05 | source-group-007 | 82.93 | 87.11 | 84.89 | 62.92 | 69.26 | 65.73 | 67.40 | 2.17 | 9.67 |
| individual--compact_boost--local_heads--chronological--budget-05 | source-group-009 | 98.11 | 99.58 | 98.84 | 95.35 | 95.18 | 97.30 | 96.22 | 1.39 | 7.67 |
| individual--compact_boost--local_heads--chronological--budget-05 | source-group-012 | 92.33 | 98.43 | 95.28 | 88.14 | 85.89 | 90.48 | 88.11 | 1.09 | 5.67 |
| individual--compact_boost--local_heads--chronological--budget-10 | source-group-005 | 84.18 | 89.36 | 86.66 | 58.15 | 67.24 | 68.24 | 67.70 | 3.86 | 15.67 |
| individual--compact_boost--local_heads--chronological--budget-10 | source-group-007 | 84.18 | 88.50 | 86.22 | 67.09 | 73.11 | 70.09 | 71.53 | 4.31 | 18.67 |
| individual--compact_boost--local_heads--chronological--budget-10 | source-group-009 | 98.21 | 99.72 | 98.96 | 95.79 | 95.62 | 97.30 | 96.44 | 2.66 | 15.00 |
| individual--compact_boost--local_heads--chronological--budget-10 | source-group-012 | 93.27 | 98.73 | 95.92 | 89.48 | 87.92 | 91.07 | 89.46 | 2.18 | 10.33 |
| individual--compact_boost--local_heads--chronological--budget-20 | source-group-005 | 86.51 | 90.89 | 88.63 | 65.25 | 74.79 | 74.51 | 74.64 | 7.81 | 28.33 |
| individual--compact_boost--local_heads--chronological--budget-20 | source-group-007 | 87.05 | 91.59 | 89.22 | 73.35 | 78.07 | 76.32 | 77.15 | 8.91 | 35.67 |
| individual--compact_boost--local_heads--chronological--budget-20 | source-group-009 | 98.40 | 99.74 | 99.06 | 97.54 | 96.46 | 97.30 | 96.87 | 3.25 | 17.67 |
| individual--compact_boost--local_heads--chronological--budget-20 | source-group-012 | 94.98 | 98.94 | 96.92 | 92.90 | 94.05 | 94.05 | 94.05 | 4.14 | 19.00 |
| individual--compact_boost--local_heads--chronological--budget-40 | source-group-005 | 90.15 | 96.56 | 93.24 | 77.31 | 86.82 | 85.10 | 85.94 | 15.19 | 54.00 |
| individual--compact_boost--local_heads--chronological--budget-40 | source-group-007 | 91.47 | 95.46 | 93.38 | 87.93 | 90.23 | 86.29 | 88.21 | 17.83 | 71.67 |
| individual--compact_boost--local_heads--chronological--budget-40 | source-group-009 | 98.40 | 99.74 | 99.06 | 97.54 | 96.46 | 97.30 | 96.87 | 3.25 | 17.67 |
| individual--compact_boost--local_heads--chronological--budget-40 | source-group-012 | 95.87 | 98.94 | 97.38 | 94.33 | 97.56 | 95.24 | 96.37 | 4.87 | 21.33 |
| individual--compact_boost--local_heads--evidence--budget-05 | source-group-005 | 83.03 | 86.84 | 84.87 | 53.69 | 64.46 | 63.14 | 63.76 | 1.96 | 6.33 |
| individual--compact_boost--local_heads--evidence--budget-05 | source-group-007 | 84.06 | 86.87 | 85.36 | 62.98 | 71.20 | 65.42 | 68.12 | 2.11 | 7.00 |
| individual--compact_boost--local_heads--evidence--budget-05 | source-group-009 | 98.14 | 99.61 | 98.87 | 95.35 | 94.33 | 97.30 | 95.79 | 1.38 | 7.67 |
| individual--compact_boost--local_heads--evidence--budget-05 | source-group-012 | 92.99 | 98.17 | 95.51 | 87.85 | 88.58 | 92.26 | 90.36 | 1.10 | 5.33 |
| individual--compact_boost--local_heads--evidence--budget-10 | source-group-005 | 84.76 | 88.80 | 86.70 | 58.22 | 68.93 | 66.67 | 67.73 | 3.94 | 13.00 |
| individual--compact_boost--local_heads--evidence--budget-10 | source-group-007 | 85.77 | 87.76 | 86.70 | 68.64 | 75.53 | 67.29 | 71.13 | 4.46 | 15.33 |
| individual--compact_boost--local_heads--evidence--budget-10 | source-group-009 | 98.28 | 99.74 | 99.00 | 96.24 | 95.62 | 97.30 | 96.44 | 2.67 | 15.00 |
| individual--compact_boost--local_heads--evidence--budget-10 | source-group-012 | 93.74 | 98.47 | 96.05 | 90.60 | 91.25 | 92.86 | 92.04 | 2.21 | 11.33 |
| individual--compact_boost--local_heads--evidence--budget-20 | source-group-005 | 87.06 | 93.81 | 90.30 | 69.57 | 78.86 | 76.08 | 77.43 | 7.90 | 30.33 |
| individual--compact_boost--local_heads--evidence--budget-20 | source-group-007 | 88.40 | 89.31 | 88.80 | 75.66 | 82.98 | 72.59 | 77.39 | 9.00 | 33.67 |
| individual--compact_boost--local_heads--evidence--budget-20 | source-group-009 | 98.40 | 99.74 | 99.06 | 97.54 | 96.46 | 97.30 | 96.87 | 3.25 | 17.67 |
| individual--compact_boost--local_heads--evidence--budget-20 | source-group-012 | 95.77 | 98.79 | 97.25 | 94.33 | 96.35 | 94.64 | 95.48 | 4.14 | 18.33 |
| individual--compact_boost--local_heads--evidence--budget-40 | source-group-005 | 90.60 | 97.01 | 93.69 | 79.47 | 88.52 | 84.71 | 86.57 | 15.27 | 57.33 |
| individual--compact_boost--local_heads--evidence--budget-40 | source-group-007 | 91.70 | 95.13 | 93.32 | 87.57 | 92.06 | 86.29 | 89.06 | 17.93 | 75.00 |
| individual--compact_boost--local_heads--evidence--budget-40 | source-group-009 | 98.40 | 99.74 | 99.06 | 97.54 | 96.46 | 97.30 | 96.87 | 3.25 | 17.67 |
| individual--compact_boost--local_heads--evidence--budget-40 | source-group-012 | 95.87 | 98.94 | 97.38 | 94.33 | 97.56 | 95.24 | 96.37 | 4.87 | 21.33 |
| individual--dino_global--legacy--chronological--budget-05 | source-group-005 | 86.72 | 96.02 | 90.95 | 71.98 | 72.04 | 74.12 | 72.90 | 1.82 | 5.67 |
| individual--dino_global--legacy--chronological--budget-05 | source-group-007 | 86.04 | 94.05 | 89.85 | 74.16 | 76.57 | 76.95 | 76.73 | 2.10 | 8.33 |
| individual--dino_global--legacy--chronological--budget-05 | source-group-009 | 98.39 | 99.34 | 98.85 | 89.72 | 89.71 | 95.95 | 92.68 | 1.41 | 6.33 |
| individual--dino_global--legacy--chronological--budget-05 | source-group-012 | 88.85 | 100.00 | 94.09 | 85.57 | 89.43 | 94.64 | 91.95 | 1.04 | 4.67 |
| individual--dino_global--legacy--chronological--budget-10 | source-group-005 | 87.29 | 96.02 | 91.27 | 72.46 | 73.15 | 74.90 | 73.83 | 3.90 | 12.33 |
| individual--dino_global--legacy--chronological--budget-10 | source-group-007 | 86.83 | 94.71 | 90.59 | 75.58 | 77.83 | 78.50 | 78.15 | 4.38 | 17.33 |
| individual--dino_global--legacy--chronological--budget-10 | source-group-009 | 98.50 | 99.34 | 98.91 | 90.70 | 90.82 | 95.95 | 93.27 | 2.92 | 13.33 |
| individual--dino_global--legacy--chronological--budget-10 | source-group-012 | 89.12 | 100.00 | 94.24 | 85.32 | 89.43 | 94.64 | 91.95 | 2.15 | 9.33 |
| individual--dino_global--legacy--chronological--budget-20 | source-group-005 | 89.01 | 96.62 | 92.51 | 75.70 | 77.47 | 78.82 | 78.01 | 7.71 | 23.33 |
| individual--dino_global--legacy--chronological--budget-20 | source-group-007 | 88.84 | 95.83 | 92.20 | 80.20 | 81.98 | 80.69 | 81.32 | 8.94 | 31.67 |
| individual--dino_global--legacy--chronological--budget-20 | source-group-009 | 98.80 | 99.37 | 99.08 | 93.64 | 95.30 | 96.85 | 96.02 | 5.98 | 25.33 |
| individual--dino_global--legacy--chronological--budget-20 | source-group-012 | 89.76 | 100.00 | 94.59 | 86.81 | 89.97 | 94.64 | 92.23 | 4.39 | 18.00 |
| individual--dino_global--legacy--chronological--budget-40 | source-group-005 | 91.95 | 97.77 | 94.68 | 85.04 | 86.10 | 86.67 | 86.31 | 15.62 | 47.33 |
| individual--dino_global--legacy--chronological--budget-40 | source-group-007 | 92.74 | 97.27 | 94.95 | 86.08 | 89.64 | 85.98 | 87.76 | 18.00 | 61.00 |
| individual--dino_global--legacy--chronological--budget-40 | source-group-009 | 99.10 | 99.88 | 99.49 | 96.02 | 96.48 | 95.95 | 96.15 | 12.15 | 53.33 |
| individual--dino_global--legacy--chronological--budget-40 | source-group-012 | 92.92 | 100.00 | 96.32 | 91.63 | 95.32 | 95.83 | 95.57 | 8.88 | 34.67 |
| individual--dino_global--legacy--evidence--budget-05 | source-group-005 | 86.52 | 96.16 | 90.90 | 70.10 | 69.16 | 73.73 | 71.23 | 1.93 | 7.67 |
| individual--dino_global--legacy--evidence--budget-05 | source-group-007 | 86.24 | 94.06 | 89.97 | 73.66 | 76.33 | 76.01 | 76.15 | 2.16 | 9.33 |
| individual--dino_global--legacy--evidence--budget-05 | source-group-009 | 98.16 | 99.34 | 98.74 | 91.25 | 90.87 | 95.95 | 93.28 | 1.47 | 7.00 |
| individual--dino_global--legacy--evidence--budget-05 | source-group-012 | 89.11 | 100.00 | 94.23 | 86.23 | 89.89 | 94.64 | 92.20 | 1.10 | 5.67 |
| individual--dino_global--legacy--evidence--budget-10 | source-group-005 | 87.17 | 97.08 | 91.71 | 73.97 | 71.59 | 75.29 | 73.30 | 3.92 | 16.67 |
| individual--dino_global--legacy--evidence--budget-10 | source-group-007 | 86.84 | 94.33 | 90.42 | 75.99 | 78.39 | 77.88 | 78.13 | 4.48 | 19.00 |
| individual--dino_global--legacy--evidence--budget-10 | source-group-009 | 98.41 | 99.34 | 98.87 | 91.47 | 91.29 | 95.95 | 93.49 | 3.00 | 14.33 |
| individual--dino_global--legacy--evidence--budget-10 | source-group-012 | 89.36 | 100.00 | 94.37 | 87.38 | 90.45 | 95.24 | 92.78 | 2.16 | 11.33 |
| individual--dino_global--legacy--evidence--budget-20 | source-group-005 | 88.07 | 97.84 | 92.57 | 79.75 | 77.70 | 81.57 | 79.54 | 7.88 | 34.00 |
| individual--dino_global--legacy--evidence--budget-20 | source-group-007 | 88.43 | 95.11 | 91.64 | 79.76 | 83.93 | 81.31 | 82.59 | 9.02 | 38.00 |
| individual--dino_global--legacy--evidence--budget-20 | source-group-009 | 98.69 | 99.73 | 99.21 | 93.57 | 93.68 | 96.40 | 94.95 | 6.08 | 29.00 |
| individual--dino_global--legacy--evidence--budget-20 | source-group-012 | 90.48 | 100.00 | 95.00 | 88.29 | 91.47 | 95.24 | 93.31 | 4.43 | 22.33 |
| individual--dino_global--legacy--evidence--budget-40 | source-group-005 | 91.31 | 98.03 | 94.46 | 89.27 | 91.86 | 89.02 | 90.40 | 15.80 | 64.33 |
| individual--dino_global--legacy--evidence--budget-40 | source-group-007 | 91.86 | 96.48 | 94.11 | 86.23 | 91.99 | 85.98 | 88.88 | 18.11 | 73.33 |
| individual--dino_global--legacy--evidence--budget-40 | source-group-009 | 98.99 | 99.80 | 99.39 | 96.91 | 97.36 | 96.85 | 97.04 | 12.20 | 56.67 |
| individual--dino_global--legacy--evidence--budget-40 | source-group-012 | 93.16 | 100.00 | 96.45 | 92.84 | 95.91 | 96.43 | 96.16 | 8.88 | 40.33 |
| individual--dino_global--local_events--chronological--budget-05 | source-group-005 | 87.11 | 96.23 | 91.26 | 73.12 | 73.01 | 74.90 | 73.82 | 1.84 | 6.00 |
| individual--dino_global--local_events--chronological--budget-05 | source-group-007 | 86.66 | 94.92 | 90.59 | 76.22 | 77.55 | 78.19 | 77.84 | 2.07 | 10.33 |
| individual--dino_global--local_events--chronological--budget-05 | source-group-009 | 98.53 | 99.50 | 99.01 | 92.60 | 92.39 | 96.40 | 94.32 | 1.39 | 7.00 |
| individual--dino_global--local_events--chronological--budget-05 | source-group-012 | 90.03 | 100.00 | 94.75 | 87.46 | 91.38 | 94.64 | 92.98 | 1.04 | 4.00 |
| individual--dino_global--local_events--chronological--budget-10 | source-group-005 | 88.02 | 96.76 | 92.03 | 77.69 | 76.27 | 78.04 | 77.08 | 3.84 | 13.67 |
| individual--dino_global--local_events--chronological--budget-10 | source-group-007 | 88.30 | 95.51 | 91.75 | 80.66 | 81.71 | 79.13 | 80.37 | 4.19 | 16.33 |
| individual--dino_global--local_events--chronological--budget-10 | source-group-009 | 98.63 | 99.77 | 99.19 | 95.18 | 93.96 | 96.85 | 95.37 | 2.26 | 11.67 |
| individual--dino_global--local_events--chronological--budget-10 | source-group-012 | 91.27 | 100.00 | 95.43 | 89.99 | 94.64 | 95.24 | 94.92 | 1.96 | 7.00 |
| individual--dino_global--local_events--chronological--budget-20 | source-group-005 | 90.21 | 97.81 | 93.78 | 85.63 | 83.56 | 82.35 | 82.92 | 7.31 | 27.00 |
| individual--dino_global--local_events--chronological--budget-20 | source-group-007 | 90.37 | 96.64 | 93.39 | 86.13 | 85.33 | 81.62 | 83.43 | 7.16 | 27.00 |
| individual--dino_global--local_events--chronological--budget-20 | source-group-009 | 98.63 | 99.77 | 99.19 | 95.18 | 93.96 | 96.85 | 95.37 | 2.26 | 11.67 |
| individual--dino_global--local_events--chronological--budget-20 | source-group-012 | 91.47 | 100.00 | 95.54 | 90.25 | 95.19 | 95.24 | 95.20 | 2.13 | 7.33 |
| individual--dino_global--local_events--chronological--budget-40 | source-group-005 | 91.07 | 98.88 | 94.74 | 90.61 | 86.12 | 85.88 | 85.98 | 9.25 | 35.33 |
| individual--dino_global--local_events--chronological--budget-40 | source-group-007 | 90.69 | 97.27 | 93.86 | 88.50 | 86.08 | 83.18 | 84.58 | 8.60 | 32.67 |
| individual--dino_global--local_events--chronological--budget-40 | source-group-009 | 98.63 | 99.77 | 99.19 | 95.18 | 93.96 | 96.85 | 95.37 | 2.26 | 11.67 |
| individual--dino_global--local_events--chronological--budget-40 | source-group-012 | 91.47 | 100.00 | 95.54 | 90.25 | 95.19 | 95.24 | 95.20 | 2.13 | 7.33 |
| individual--dino_global--local_events--evidence--budget-05 | source-group-005 | 87.50 | 95.56 | 91.19 | 72.02 | 71.85 | 73.73 | 72.70 | 1.92 | 6.33 |
| individual--dino_global--local_events--evidence--budget-05 | source-group-007 | 87.13 | 94.53 | 90.65 | 76.68 | 78.54 | 76.95 | 77.70 | 2.15 | 8.33 |
| individual--dino_global--local_events--evidence--budget-05 | source-group-009 | 98.21 | 99.74 | 98.97 | 91.84 | 90.86 | 96.85 | 93.73 | 1.37 | 7.33 |
| individual--dino_global--local_events--evidence--budget-05 | source-group-012 | 90.06 | 100.00 | 94.76 | 86.78 | 92.48 | 95.24 | 93.83 | 1.07 | 4.67 |
| individual--dino_global--local_events--evidence--budget-10 | source-group-005 | 88.82 | 95.70 | 91.96 | 78.92 | 76.85 | 75.69 | 76.20 | 3.91 | 14.00 |
| individual--dino_global--local_events--evidence--budget-10 | source-group-007 | 88.67 | 95.99 | 92.17 | 81.03 | 81.81 | 79.13 | 80.42 | 4.18 | 15.67 |
| individual--dino_global--local_events--evidence--budget-10 | source-group-009 | 98.63 | 99.77 | 99.19 | 95.18 | 93.96 | 96.85 | 95.37 | 2.26 | 11.67 |
| individual--dino_global--local_events--evidence--budget-10 | source-group-012 | 91.47 | 100.00 | 95.54 | 90.25 | 95.19 | 95.24 | 95.20 | 1.96 | 7.00 |
| individual--dino_global--local_events--evidence--budget-20 | source-group-005 | 90.84 | 97.19 | 93.80 | 88.12 | 84.07 | 81.57 | 82.76 | 7.42 | 28.00 |
| individual--dino_global--local_events--evidence--budget-20 | source-group-007 | 90.33 | 96.48 | 93.30 | 86.22 | 85.19 | 80.69 | 82.87 | 7.24 | 29.00 |
| individual--dino_global--local_events--evidence--budget-20 | source-group-009 | 98.63 | 99.77 | 99.19 | 95.18 | 93.96 | 96.85 | 95.37 | 2.26 | 11.67 |
| individual--dino_global--local_events--evidence--budget-20 | source-group-012 | 91.47 | 100.00 | 95.54 | 90.25 | 95.19 | 95.24 | 95.20 | 2.13 | 7.33 |
| individual--dino_global--local_events--evidence--budget-40 | source-group-005 | 91.07 | 98.88 | 94.74 | 90.61 | 86.12 | 85.88 | 85.98 | 9.25 | 35.33 |
| individual--dino_global--local_events--evidence--budget-40 | source-group-007 | 90.69 | 97.27 | 93.86 | 88.50 | 86.08 | 83.18 | 84.58 | 8.60 | 32.67 |
| individual--dino_global--local_events--evidence--budget-40 | source-group-009 | 98.63 | 99.77 | 99.19 | 95.18 | 93.96 | 96.85 | 95.37 | 2.26 | 11.67 |
| individual--dino_global--local_events--evidence--budget-40 | source-group-012 | 91.47 | 100.00 | 95.54 | 90.25 | 95.19 | 95.24 | 95.20 | 2.13 | 7.33 |
| individual--dino_global--local_heads--chronological--budget-05 | source-group-005 | 87.00 | 95.95 | 91.07 | 72.08 | 71.75 | 73.73 | 72.54 | 1.84 | 7.00 |
| individual--dino_global--local_heads--chronological--budget-05 | source-group-007 | 86.45 | 94.51 | 90.29 | 74.66 | 77.00 | 77.88 | 77.42 | 2.08 | 11.33 |
| individual--dino_global--local_heads--chronological--budget-05 | source-group-009 | 98.55 | 99.37 | 98.95 | 90.39 | 90.56 | 96.40 | 93.33 | 1.31 | 6.00 |
| individual--dino_global--local_heads--chronological--budget-05 | source-group-012 | 89.80 | 100.00 | 94.62 | 86.74 | 89.43 | 94.64 | 91.95 | 1.02 | 4.67 |
| individual--dino_global--local_heads--chronological--budget-10 | source-group-005 | 87.89 | 96.25 | 91.72 | 73.93 | 73.93 | 76.08 | 74.84 | 3.83 | 15.33 |
| individual--dino_global--local_heads--chronological--budget-10 | source-group-007 | 88.07 | 95.09 | 91.44 | 77.55 | 80.84 | 79.75 | 80.27 | 4.44 | 20.67 |
| individual--dino_global--local_heads--chronological--budget-10 | source-group-009 | 98.88 | 99.51 | 99.19 | 91.88 | 92.62 | 97.30 | 94.83 | 2.93 | 15.00 |
| individual--dino_global--local_heads--chronological--budget-10 | source-group-012 | 91.16 | 100.00 | 95.37 | 89.53 | 92.59 | 95.83 | 94.18 | 2.19 | 10.67 |
| individual--dino_global--local_heads--chronological--budget-20 | source-group-005 | 89.89 | 96.84 | 93.12 | 81.01 | 79.86 | 81.57 | 80.62 | 7.83 | 31.67 |
| individual--dino_global--local_heads--chronological--budget-20 | source-group-007 | 90.99 | 96.17 | 93.49 | 83.50 | 88.28 | 83.80 | 85.95 | 8.65 | 36.67 |
| individual--dino_global--local_heads--chronological--budget-20 | source-group-009 | 99.06 | 99.61 | 99.33 | 94.40 | 94.16 | 98.20 | 96.09 | 4.64 | 24.00 |
| individual--dino_global--local_heads--chronological--budget-20 | source-group-012 | 93.05 | 100.00 | 96.40 | 92.45 | 94.85 | 98.21 | 96.50 | 3.70 | 18.33 |
| individual--dino_global--local_heads--chronological--budget-40 | source-group-005 | 93.79 | 98.26 | 95.95 | 89.88 | 91.10 | 88.24 | 89.63 | 15.53 | 64.00 |
| individual--dino_global--local_heads--chronological--budget-40 | source-group-007 | 92.22 | 97.32 | 94.69 | 87.76 | 91.48 | 87.23 | 89.28 | 12.61 | 55.33 |
| individual--dino_global--local_heads--chronological--budget-40 | source-group-009 | 99.10 | 99.92 | 99.51 | 96.94 | 95.69 | 98.65 | 97.13 | 6.71 | 33.67 |
| individual--dino_global--local_heads--chronological--budget-40 | source-group-012 | 94.09 | 100.00 | 96.95 | 93.84 | 97.66 | 99.40 | 98.53 | 4.60 | 21.00 |
| individual--dino_global--local_heads--evidence--budget-05 | source-group-005 | 87.66 | 95.39 | 91.19 | 73.78 | 74.04 | 73.73 | 73.79 | 1.96 | 6.33 |
| individual--dino_global--local_heads--evidence--budget-05 | source-group-007 | 87.58 | 94.40 | 90.84 | 76.52 | 79.09 | 76.01 | 77.49 | 2.11 | 7.33 |
| individual--dino_global--local_heads--evidence--budget-05 | source-group-009 | 98.26 | 99.85 | 99.05 | 91.89 | 91.36 | 97.75 | 94.40 | 1.41 | 8.00 |
| individual--dino_global--local_heads--evidence--budget-05 | source-group-012 | 89.78 | 100.00 | 94.61 | 86.08 | 91.55 | 95.83 | 93.64 | 1.04 | 4.67 |
| individual--dino_global--local_heads--evidence--budget-10 | source-group-005 | 88.89 | 96.07 | 92.20 | 79.04 | 77.67 | 76.47 | 77.00 | 3.94 | 15.67 |
| individual--dino_global--local_heads--evidence--budget-10 | source-group-007 | 89.15 | 95.25 | 92.09 | 80.79 | 81.96 | 77.57 | 79.69 | 4.39 | 17.67 |
| individual--dino_global--local_heads--evidence--budget-10 | source-group-009 | 98.63 | 99.85 | 99.23 | 94.00 | 93.00 | 98.65 | 95.69 | 2.91 | 16.33 |
| individual--dino_global--local_heads--evidence--budget-10 | source-group-012 | 90.86 | 100.00 | 95.20 | 88.67 | 93.77 | 97.62 | 95.65 | 2.22 | 11.33 |
| individual--dino_global--local_heads--evidence--budget-20 | source-group-005 | 89.68 | 96.82 | 93.01 | 84.55 | 81.17 | 81.18 | 81.12 | 7.88 | 37.33 |
| individual--dino_global--local_heads--evidence--budget-20 | source-group-007 | 90.71 | 96.25 | 93.38 | 85.68 | 88.25 | 83.80 | 85.96 | 8.75 | 41.00 |
| individual--dino_global--local_heads--evidence--budget-20 | source-group-009 | 99.02 | 99.85 | 99.43 | 95.92 | 94.56 | 98.65 | 96.52 | 4.66 | 26.00 |
| individual--dino_global--local_heads--evidence--budget-20 | source-group-012 | 92.58 | 100.00 | 96.14 | 91.32 | 95.44 | 98.81 | 97.08 | 3.75 | 19.00 |
| individual--dino_global--local_heads--evidence--budget-40 | source-group-005 | 93.96 | 97.71 | 95.77 | 91.69 | 91.21 | 89.41 | 90.28 | 15.55 | 69.67 |
| individual--dino_global--local_heads--evidence--budget-40 | source-group-007 | 92.01 | 97.56 | 94.68 | 87.80 | 90.91 | 87.23 | 89.00 | 12.67 | 57.67 |
| individual--dino_global--local_heads--evidence--budget-40 | source-group-009 | 99.09 | 99.88 | 99.48 | 96.31 | 95.72 | 99.10 | 97.36 | 6.67 | 35.00 |
| individual--dino_global--local_heads--evidence--budget-40 | source-group-012 | 94.09 | 100.00 | 96.95 | 93.84 | 97.66 | 99.40 | 98.53 | 4.60 | 21.00 |
| individual--dino_boost--legacy--chronological--budget-05 | source-group-005 | 79.13 | 97.92 | 87.43 | 63.57 | 72.04 | 80.39 | 75.75 | 1.78 | 4.33 |
| individual--dino_boost--legacy--chronological--budget-05 | source-group-007 | 90.84 | 91.45 | 91.08 | 69.63 | 77.05 | 76.64 | 76.83 | 2.10 | 7.33 |
| individual--dino_boost--legacy--chronological--budget-05 | source-group-009 | 99.08 | 99.20 | 99.13 | 90.55 | 91.84 | 96.40 | 94.06 | 1.46 | 4.67 |
| individual--dino_boost--legacy--chronological--budget-05 | source-group-012 | 88.27 | 100.00 | 93.73 | 85.89 | 84.47 | 92.86 | 88.44 | 1.01 | 4.33 |
| individual--dino_boost--legacy--chronological--budget-10 | source-group-005 | 80.08 | 97.92 | 88.01 | 65.29 | 73.70 | 81.18 | 77.05 | 3.87 | 10.33 |
| individual--dino_boost--legacy--chronological--budget-10 | source-group-007 | 91.20 | 91.69 | 91.38 | 71.81 | 77.97 | 77.57 | 77.75 | 4.35 | 15.67 |
| individual--dino_boost--legacy--chronological--budget-10 | source-group-009 | 99.19 | 99.20 | 99.19 | 91.20 | 92.24 | 96.40 | 94.27 | 2.93 | 11.00 |
| individual--dino_boost--legacy--chronological--budget-10 | source-group-012 | 88.42 | 100.00 | 93.81 | 85.66 | 84.47 | 92.86 | 88.44 | 2.13 | 9.00 |
| individual--dino_boost--legacy--chronological--budget-20 | source-group-005 | 82.30 | 98.12 | 89.44 | 69.92 | 78.76 | 83.14 | 80.80 | 7.76 | 22.00 |
| individual--dino_boost--legacy--chronological--budget-20 | source-group-007 | 92.42 | 93.07 | 92.71 | 75.91 | 82.08 | 81.62 | 81.84 | 8.88 | 30.00 |
| individual--dino_boost--legacy--chronological--budget-20 | source-group-009 | 99.25 | 99.24 | 99.24 | 93.31 | 95.15 | 96.85 | 95.99 | 6.06 | 22.00 |
| individual--dino_boost--legacy--chronological--budget-20 | source-group-012 | 89.05 | 100.00 | 94.17 | 86.54 | 84.91 | 92.86 | 88.68 | 4.36 | 18.00 |
| individual--dino_boost--legacy--chronological--budget-40 | source-group-005 | 85.15 | 98.88 | 91.43 | 79.67 | 83.88 | 87.06 | 85.41 | 15.66 | 44.33 |
| individual--dino_boost--legacy--chronological--budget-40 | source-group-007 | 96.00 | 95.41 | 95.69 | 85.26 | 91.20 | 87.23 | 89.16 | 18.01 | 59.33 |
| individual--dino_boost--legacy--chronological--budget-40 | source-group-009 | 99.37 | 99.76 | 99.56 | 95.07 | 96.44 | 97.30 | 96.86 | 12.17 | 45.33 |
| individual--dino_boost--legacy--chronological--budget-40 | source-group-012 | 91.56 | 100.00 | 95.55 | 89.32 | 89.27 | 93.45 | 91.30 | 8.83 | 36.00 |
| individual--dino_boost--legacy--evidence--budget-05 | source-group-005 | 78.96 | 98.01 | 87.37 | 62.73 | 70.23 | 79.61 | 74.41 | 1.94 | 6.67 |
| individual--dino_boost--legacy--evidence--budget-05 | source-group-007 | 91.15 | 91.30 | 91.16 | 68.99 | 78.91 | 77.26 | 78.06 | 2.19 | 9.33 |
| individual--dino_boost--legacy--evidence--budget-05 | source-group-009 | 98.80 | 99.20 | 99.00 | 90.53 | 92.68 | 96.85 | 94.71 | 1.50 | 6.67 |
| individual--dino_boost--legacy--evidence--budget-05 | source-group-012 | 88.33 | 100.00 | 93.77 | 86.02 | 85.37 | 92.86 | 88.93 | 1.06 | 4.67 |
| individual--dino_boost--legacy--evidence--budget-10 | source-group-005 | 79.63 | 98.18 | 87.84 | 65.90 | 73.05 | 82.35 | 77.20 | 3.92 | 14.00 |
| individual--dino_boost--legacy--evidence--budget-10 | source-group-007 | 91.54 | 91.81 | 91.62 | 71.29 | 80.18 | 78.50 | 79.32 | 4.47 | 20.00 |
| individual--dino_boost--legacy--evidence--budget-10 | source-group-009 | 98.83 | 99.20 | 99.01 | 90.94 | 93.52 | 96.85 | 95.14 | 2.98 | 14.00 |
| individual--dino_boost--legacy--evidence--budget-10 | source-group-012 | 88.66 | 100.00 | 93.96 | 86.57 | 86.41 | 93.45 | 89.76 | 2.19 | 10.33 |
| individual--dino_boost--legacy--evidence--budget-20 | source-group-005 | 81.61 | 98.67 | 89.28 | 69.59 | 78.92 | 84.71 | 81.61 | 7.90 | 29.00 |
| individual--dino_boost--legacy--evidence--budget-20 | source-group-007 | 92.54 | 92.52 | 92.47 | 74.77 | 85.77 | 82.55 | 84.07 | 9.04 | 42.67 |
| individual--dino_boost--legacy--evidence--budget-20 | source-group-009 | 98.88 | 99.58 | 99.23 | 92.72 | 94.37 | 97.75 | 96.02 | 6.10 | 29.67 |
| individual--dino_boost--legacy--evidence--budget-20 | source-group-012 | 89.53 | 100.00 | 94.45 | 87.46 | 88.95 | 94.05 | 91.38 | 4.42 | 20.67 |
| individual--dino_boost--legacy--evidence--budget-40 | source-group-005 | 84.96 | 99.32 | 91.55 | 81.36 | 94.06 | 91.37 | 92.68 | 15.79 | 60.67 |
| individual--dino_boost--legacy--evidence--budget-40 | source-group-007 | 94.87 | 94.77 | 94.80 | 85.49 | 94.59 | 89.10 | 91.68 | 18.09 | 79.33 |
| individual--dino_boost--legacy--evidence--budget-40 | source-group-009 | 99.38 | 99.68 | 99.53 | 96.00 | 98.26 | 99.55 | 98.89 | 12.18 | 59.33 |
| individual--dino_boost--legacy--evidence--budget-40 | source-group-012 | 91.87 | 100.00 | 95.74 | 90.45 | 94.25 | 95.83 | 95.00 | 8.87 | 43.00 |
| individual--dino_boost--local_events--chronological--budget-05 | source-group-005 | 79.62 | 97.92 | 87.73 | 65.49 | 71.98 | 80.00 | 75.59 | 1.84 | 4.00 |
| individual--dino_boost--local_events--chronological--budget-05 | source-group-007 | 91.46 | 92.18 | 91.75 | 72.53 | 79.39 | 78.19 | 78.76 | 2.11 | 9.33 |
| individual--dino_boost--local_events--chronological--budget-05 | source-group-009 | 99.08 | 99.61 | 99.34 | 93.54 | 94.28 | 96.40 | 95.33 | 1.37 | 6.67 |
| individual--dino_boost--local_events--chronological--budget-05 | source-group-012 | 89.39 | 100.00 | 94.36 | 88.56 | 86.88 | 92.86 | 89.74 | 1.07 | 5.00 |
| individual--dino_boost--local_events--chronological--budget-10 | source-group-005 | 80.58 | 98.29 | 88.47 | 69.59 | 74.75 | 81.96 | 78.08 | 3.81 | 11.67 |
| individual--dino_boost--local_events--chronological--budget-10 | source-group-007 | 92.38 | 93.33 | 92.80 | 76.78 | 82.10 | 78.82 | 80.39 | 4.31 | 19.00 |
| individual--dino_boost--local_events--chronological--budget-10 | source-group-009 | 99.26 | 99.73 | 99.49 | 96.22 | 95.17 | 97.30 | 96.22 | 2.73 | 14.00 |
| individual--dino_boost--local_events--chronological--budget-10 | source-group-012 | 90.66 | 100.00 | 95.07 | 90.40 | 90.88 | 94.05 | 92.42 | 1.98 | 7.00 |
| individual--dino_boost--local_events--chronological--budget-20 | source-group-005 | 83.04 | 99.11 | 90.31 | 75.40 | 82.84 | 85.49 | 84.09 | 7.76 | 24.33 |
| individual--dino_boost--local_events--chronological--budget-20 | source-group-007 | 94.20 | 94.83 | 94.49 | 83.50 | 87.15 | 82.87 | 84.94 | 7.06 | 30.00 |
| individual--dino_boost--local_events--chronological--budget-20 | source-group-009 | 99.28 | 99.73 | 99.50 | 96.67 | 95.17 | 97.30 | 96.22 | 2.94 | 15.00 |
| individual--dino_boost--local_events--chronological--budget-20 | source-group-012 | 90.79 | 100.00 | 95.15 | 90.65 | 91.38 | 94.05 | 92.68 | 2.03 | 7.00 |
| individual--dino_boost--local_events--chronological--budget-40 | source-group-005 | 85.01 | 99.74 | 91.74 | 82.19 | 88.29 | 88.24 | 88.25 | 10.69 | 33.33 |
| individual--dino_boost--local_events--chronological--budget-40 | source-group-007 | 94.51 | 95.26 | 94.85 | 86.31 | 87.84 | 84.11 | 85.92 | 8.72 | 37.00 |
| individual--dino_boost--local_events--chronological--budget-40 | source-group-009 | 99.28 | 99.73 | 99.50 | 96.67 | 95.17 | 97.30 | 96.22 | 2.94 | 15.00 |
| individual--dino_boost--local_events--chronological--budget-40 | source-group-012 | 90.79 | 100.00 | 95.15 | 90.65 | 91.38 | 94.05 | 92.68 | 2.03 | 7.00 |
| individual--dino_boost--local_events--evidence--budget-05 | source-group-005 | 79.99 | 97.95 | 87.98 | 64.55 | 71.89 | 80.78 | 75.85 | 1.90 | 6.00 |
| individual--dino_boost--local_events--evidence--budget-05 | source-group-007 | 91.75 | 91.85 | 91.75 | 73.17 | 79.61 | 76.95 | 78.24 | 2.14 | 8.67 |
| individual--dino_boost--local_events--evidence--budget-05 | source-group-009 | 98.83 | 99.59 | 99.20 | 92.93 | 93.48 | 96.85 | 95.13 | 1.44 | 7.00 |
| individual--dino_boost--local_events--evidence--budget-05 | source-group-012 | 90.02 | 100.00 | 94.72 | 89.25 | 89.28 | 92.86 | 91.01 | 1.11 | 3.67 |
| individual--dino_boost--local_events--evidence--budget-10 | source-group-005 | 81.48 | 98.13 | 88.95 | 69.91 | 77.26 | 82.75 | 79.82 | 3.93 | 13.00 |
| individual--dino_boost--local_events--evidence--budget-10 | source-group-007 | 92.60 | 93.52 | 93.02 | 78.55 | 83.68 | 80.37 | 81.96 | 4.39 | 19.67 |
| individual--dino_boost--local_events--evidence--budget-10 | source-group-009 | 99.28 | 99.67 | 99.48 | 95.99 | 95.14 | 96.85 | 95.99 | 2.74 | 14.00 |
| individual--dino_boost--local_events--evidence--budget-10 | source-group-012 | 90.79 | 100.00 | 95.15 | 90.91 | 91.90 | 94.05 | 92.94 | 1.98 | 6.67 |
| individual--dino_boost--local_events--evidence--budget-20 | source-group-005 | 83.86 | 98.72 | 90.62 | 78.40 | 84.98 | 85.49 | 85.21 | 7.80 | 25.67 |
| individual--dino_boost--local_events--evidence--budget-20 | source-group-007 | 94.34 | 94.65 | 94.47 | 84.85 | 87.83 | 83.80 | 85.75 | 7.13 | 31.00 |
| individual--dino_boost--local_events--evidence--budget-20 | source-group-009 | 99.28 | 99.73 | 99.50 | 96.67 | 95.17 | 97.30 | 96.22 | 2.94 | 15.00 |
| individual--dino_boost--local_events--evidence--budget-20 | source-group-012 | 90.79 | 100.00 | 95.15 | 90.65 | 91.38 | 94.05 | 92.68 | 2.03 | 7.00 |
| individual--dino_boost--local_events--evidence--budget-40 | source-group-005 | 85.01 | 99.74 | 91.74 | 82.19 | 88.29 | 88.24 | 88.25 | 10.69 | 33.33 |
| individual--dino_boost--local_events--evidence--budget-40 | source-group-007 | 94.51 | 95.26 | 94.85 | 86.31 | 87.84 | 84.11 | 85.92 | 8.72 | 37.00 |
| individual--dino_boost--local_events--evidence--budget-40 | source-group-009 | 99.28 | 99.73 | 99.50 | 96.67 | 95.17 | 97.30 | 96.22 | 2.94 | 15.00 |
| individual--dino_boost--local_events--evidence--budget-40 | source-group-012 | 90.79 | 100.00 | 95.15 | 90.65 | 91.38 | 94.05 | 92.68 | 2.03 | 7.00 |
| individual--dino_boost--local_heads--chronological--budget-05 | source-group-005 | 79.43 | 97.92 | 87.62 | 64.07 | 72.24 | 80.78 | 76.01 | 1.79 | 5.33 |
| individual--dino_boost--local_heads--chronological--budget-05 | source-group-007 | 91.00 | 91.65 | 91.26 | 70.34 | 76.56 | 76.64 | 76.58 | 2.03 | 10.33 |
| individual--dino_boost--local_heads--chronological--budget-05 | source-group-009 | 99.09 | 99.20 | 99.14 | 90.97 | 92.65 | 96.40 | 94.48 | 1.39 | 7.33 |
| individual--dino_boost--local_heads--chronological--budget-05 | source-group-012 | 88.68 | 100.00 | 93.96 | 86.77 | 85.12 | 92.86 | 88.76 | 1.06 | 6.00 |
| individual--dino_boost--local_heads--chronological--budget-10 | source-group-005 | 80.61 | 98.01 | 88.37 | 67.81 | 75.05 | 81.96 | 78.18 | 3.89 | 13.67 |
| individual--dino_boost--local_heads--chronological--budget-10 | source-group-007 | 91.98 | 92.43 | 92.15 | 72.56 | 78.94 | 77.57 | 78.22 | 4.30 | 20.00 |
| individual--dino_boost--local_heads--chronological--budget-10 | source-group-009 | 99.32 | 99.23 | 99.28 | 91.97 | 95.59 | 97.30 | 96.44 | 2.87 | 14.33 |
| individual--dino_boost--local_heads--chronological--budget-10 | source-group-012 | 89.53 | 100.00 | 94.44 | 88.20 | 86.65 | 93.45 | 89.85 | 2.13 | 11.33 |
| individual--dino_boost--local_heads--chronological--budget-20 | source-group-005 | 83.11 | 98.50 | 90.07 | 75.15 | 80.22 | 84.31 | 82.12 | 7.84 | 26.67 |
| individual--dino_boost--local_heads--chronological--budget-20 | source-group-007 | 93.98 | 94.25 | 94.06 | 79.39 | 85.12 | 80.37 | 82.65 | 8.90 | 40.00 |
| individual--dino_boost--local_heads--chronological--budget-20 | source-group-009 | 99.48 | 99.65 | 99.57 | 94.61 | 96.88 | 97.75 | 97.31 | 4.94 | 24.00 |
| individual--dino_boost--local_heads--chronological--budget-20 | source-group-012 | 91.00 | 100.00 | 95.26 | 90.41 | 90.12 | 95.24 | 92.57 | 3.87 | 19.33 |
| individual--dino_boost--local_heads--chronological--budget-40 | source-group-005 | 87.80 | 99.45 | 93.22 | 84.46 | 91.20 | 89.02 | 90.09 | 15.63 | 55.67 |
| individual--dino_boost--local_heads--chronological--budget-40 | source-group-007 | 96.57 | 95.72 | 96.12 | 88.68 | 91.92 | 85.67 | 88.66 | 16.34 | 76.33 |
| individual--dino_boost--local_heads--chronological--budget-40 | source-group-009 | 99.56 | 99.96 | 99.76 | 98.88 | 98.66 | 99.10 | 98.88 | 8.96 | 44.00 |
| individual--dino_boost--local_heads--chronological--budget-40 | source-group-012 | 93.58 | 100.00 | 96.68 | 93.03 | 97.07 | 98.21 | 97.63 | 6.19 | 28.67 |
| individual--dino_boost--local_heads--evidence--budget-05 | source-group-005 | 79.98 | 98.02 | 88.00 | 64.42 | 73.34 | 81.18 | 76.89 | 1.90 | 5.67 |
| individual--dino_boost--local_heads--evidence--budget-05 | source-group-007 | 91.67 | 91.77 | 91.67 | 71.46 | 78.54 | 75.70 | 77.06 | 2.17 | 8.67 |
| individual--dino_boost--local_heads--evidence--budget-05 | source-group-009 | 99.14 | 99.55 | 99.34 | 94.63 | 95.13 | 96.85 | 95.98 | 1.41 | 7.00 |
| individual--dino_boost--local_heads--evidence--budget-05 | source-group-012 | 89.37 | 100.00 | 94.36 | 87.89 | 87.70 | 92.86 | 90.20 | 1.06 | 4.67 |
| individual--dino_boost--local_heads--evidence--budget-10 | source-group-005 | 81.62 | 98.06 | 89.02 | 69.41 | 78.42 | 83.92 | 80.97 | 3.92 | 12.67 |
| individual--dino_boost--local_heads--evidence--budget-10 | source-group-007 | 92.76 | 92.41 | 92.54 | 75.72 | 82.42 | 77.88 | 80.05 | 4.42 | 19.33 |
| individual--dino_boost--local_heads--evidence--budget-10 | source-group-009 | 99.38 | 99.62 | 99.50 | 95.75 | 95.56 | 96.85 | 96.20 | 2.94 | 16.00 |
| individual--dino_boost--local_heads--evidence--budget-10 | source-group-012 | 90.25 | 100.00 | 94.85 | 88.99 | 89.28 | 93.45 | 91.31 | 2.18 | 10.67 |
| individual--dino_boost--local_heads--evidence--budget-20 | source-group-005 | 83.76 | 98.93 | 90.66 | 78.14 | 84.37 | 87.06 | 85.64 | 7.87 | 30.67 |
| individual--dino_boost--local_heads--evidence--budget-20 | source-group-007 | 94.34 | 93.63 | 93.96 | 82.03 | 85.28 | 80.06 | 82.57 | 8.98 | 44.00 |
| individual--dino_boost--local_heads--evidence--budget-20 | source-group-009 | 99.44 | 99.69 | 99.56 | 96.64 | 97.33 | 98.65 | 97.99 | 5.04 | 28.33 |
| individual--dino_boost--local_heads--evidence--budget-20 | source-group-012 | 91.29 | 100.00 | 95.43 | 89.96 | 91.51 | 94.64 | 93.03 | 3.90 | 20.00 |
| individual--dino_boost--local_heads--evidence--budget-40 | source-group-005 | 87.31 | 99.71 | 93.07 | 87.07 | 92.18 | 90.20 | 91.14 | 15.74 | 61.33 |
| individual--dino_boost--local_heads--evidence--budget-40 | source-group-007 | 96.13 | 95.85 | 95.96 | 89.36 | 91.02 | 85.98 | 88.38 | 16.37 | 80.00 |
| individual--dino_boost--local_heads--evidence--budget-40 | source-group-009 | 99.58 | 99.93 | 99.76 | 98.88 | 99.11 | 99.55 | 99.33 | 8.96 | 44.00 |
| individual--dino_boost--local_heads--evidence--budget-40 | source-group-012 | 93.58 | 100.00 | 96.68 | 93.03 | 97.07 | 98.21 | 97.63 | 6.19 | 28.67 |

## Provenance and limitations

Registered contract: `7386abe4befe96ba4c9607454f5f6936b312dd8459c5dae1a453d4cadc93440d`. Complete report SHA-256: `edc6e0fe63403460750cc204574b8cf750395b8c4325d76a34aec4ab5722799b`.

Every completed configuration/seed file was hash verified and required passed candidate, queue, editor, duration and identity audits before aggregation. The JSON summary retains seed mean/min/max/population standard deviation, all 0.25/0.5/1/2-second ordinary and observed boundary metrics, conditional matched-boundary errors, source-group details and references to every recording-level result.

Production weights historically saw these recordings, and neural decoder/model choices are development-exposed. The simulated perfect reviewer is an optimistic assumption. These results are not fresh protected-test quality, measured reviewer speed, device inference performance, serving-side accuracy or score reconstruction. Production remains unchanged.

Machine-readable summary: `private-reference-0170`
