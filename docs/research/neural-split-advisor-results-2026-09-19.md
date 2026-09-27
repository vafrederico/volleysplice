# Production-preserving compact split adviser: results

<!-- EXECUTIVE FINDINGS START -->
## Findings and product implications

**Compact is useful as a human-cleanup adviser, but the tested automatic split rules are not reliable enough to update a score.** The completed fixed experiment contains nine automatic split outcomes and 168 restricted human-review outcomes. All preserve production's export exactly at every required padding case, with zero new completely missed rallies and zero lost raw gold-core time. Production's three existing complete misses remain. No weights or product settings changed.

The useful low-effort result is **cleanup flags with evidence ordering**. At the 5% per-video cap, 4.74 minutes of playback covers 42.33 parent jobs and 14.67 real rallies; perfect review rejects 27.67 wholly false event records and raises rally-event F1 from **66.47% to 69.27%**. At the 10% cap, **8.01 minutes**, **62 jobs** and **28.33 real rallies** let the human reject **33.67 false event records**, giving **69.02% event precision / 70.81% recall / 69.90% F1**. Reviewing the whole available cleanup queue takes 12.02 minutes, removes 37 false event records and reaches 70.26% event F1. Mixed parents containing real play are retained. These removals affect score-tracking event records only; their video remains in the unchanged export.

Across the complete cleanup inventory, compact flags 74.67 parents: 37 wholly false and 37.67 real or mixed, for **49.57% mean wholly-false precision**. This is useful work prioritization for a human and insufficient evidence for an automatic veto. The model's scores are not calibrated correctness probabilities.

**Automatic splitting creates too many extra event records.** There are only **seven genuine additional-start targets** in the production parents on this scope. At one-second matching tolerance, compact event starts generate 22 proposals with 12.30% split precision and 38.10% recall; start/end-head evidence generates 34 proposals with 7.83% precision and 38.10% recall. Requiring event and head corroboration reduces the queue to **5.67 proposals**, of which **2.33 match** and **3.33 are spurious**, with **38.89% precision / 33.33% recall / 34.70% split F1**. Ratios are averaged after pooling within each seed, so dividing the mean counts need not reproduce mean precision. Corroboration's overall rally F1 is 66.51%, only 0.03 points above production, with a 66.18–66.86% seed range. The two broader automatic policies reduce event F1 to 64.60% and 64.73%. Preserving every second of footage does not prevent false points from incorrect splits.

**A small separate split-review queue remains useful.** Reviewing all corroborated proposals takes **3.25 minutes**, **5.67 jobs** and **nine real rallies**. The ideal reviewer accepts 2.33 proposed boundaries, rejects 3.33, and reduces material merged predictions from seven to 4.67 without increasing the four existing material split errors. Event F1 reaches 66.63% and observed-start recall 72.46% versus 71.74% originally. The gain is limited because this experiment only inserts starts: preceding synthetic ends remain at the next start, inter-rally dead time remains inside the partitions, original endpoints are not corrected, and unproposed missing starts are not discovered. These outputs are not complete serve-to-dead-ball corrections.

Combining corroborated split guidance and cleanup at the 40% cap uses **14.74 minutes**, **79.67 jobs** and **46.33 real rallies** for **69.64% event precision / 71.22% recall / 70.42% F1**. It rejects 37 false event records and confirms 2.33 splits. The broader head-plus-cleanup queue uses 20.03 minutes for 70.49% event F1, a small additional gain. At the 10% cap, the combined evidence queue spends 8.37 minutes but confirms zero genuine splits on average: its improvement comes entirely from cleanup. This supports exposing cleanup and split-review purposes separately rather than assuming one generic priority score serves both.

**Recommendation:** retain production export coverage; offer compact cleanup flags; present corroborated splits as provisional suggestions requiring confirmation; keep event existence, serve-start confidence and export inclusion separate. Do not enable these automatic splits for score increments. A subsequent boundary experiment should distinguish correcting the first start of an existing rally from adding a second rally, and should represent the preceding dead-ball end separately from the next serve start while leaving export coverage fixed. More independent, manually reviewed merged-rally examples are needed: seven positive split targets are too few to establish reliable deployment accuracy. Phone/browser latency and end-to-end serving-side/winner/score accuracy were not evaluated.

### Post-hoc error diagnosis

The frozen results explain why head-guided observed-start recall improves even while event F1 worsens. Of its **34.00 automatic proposals**, 2.67 match a genuine secondary start, **17.67 lie within one second of the first materially overlapping gold rally's start**, and 13.67 are away from any eligible gold start. Those 17.67 are candidates for correcting the first boundary, not evidence of an extra rally. Event-start proposals decompose into 2.67 secondary, 1.33 first-start and 18.00 away; corroborated proposals into 2.33 secondary, 1.33 first-start and 2.00 away. No unmatched proposals fall in the other/duplicate-start category on this dataset. This descriptive classification was performed after the fixed run and did not create or tune a new automatic policy.

A concrete example is `grass-source-03`, parent `R003`: production selects **50.875–92.358 seconds**, while the two gold rallies are **52.473–60.729** and **76.899–80.295**. Confirming the second start at 76.899 partitions the parent into **50.875–76.899** and **76.899–92.358**. Their best relevant IoUs are only **0.317** and **0.220**, so neither qualifies as an event match despite the exact second start. Retained dead time and inaccurate original endpoints explain this case. The future event timeline can refine both rally cores independently while the separate export remains unchanged; that behavior has not been evaluated in this split-only experiment. The bound diagnosis, per-proposal classifications and examples are saved in `split-error-diagnosis.json` beside the numerical results.

The automatic split figure (ledger `private-reference-0190`) shows the precision/recall tradeoff and false suggestions. The review workload figure (ledger `private-reference-0191`) separates cleanup gains from split-only gains. All review claims assume the registered restricted perfect-human operations, not observed human performance; the existing development-exposure limits remain.
<!-- EXECUTIVE FINDINGS END -->

This experiment keeps original production export footage fixed and evaluates compact neural guidance for separate rally identities. All arms tie on the required primary export metric `F1_padP_coreR`; the event diagnostics below do not replace that ranking.

Scope: 8 development recordings, 4 source groups, 322 eligible gold rallies; 9 automatic outcomes and 168 restricted ideal-human outcomes across 3 seeds. No new detector training, learned confidence calibration, protected test, or production change.

Values are means across seeds after pooling counts within each seed; brackets show the seed minimum and maximum, not confidence intervals. Source groups and recordings remain the same footage across seeds. Review minutes are union playback duration at 1x including context, not measured human labor.

There are 7 parent-specific genuine additional-start targets (7 distinct gold rallies), of which 7 satisfy the two-second edge guard and 0 do not. The full target count remains the split-recall denominator. This is a narrow split benchmark, not all-rally or score accuracy.

Automatic rows apply all suggested internal cuts. Restricted review accepts only proposed starts matched within one second and snaps those accepted starts to gold; it does not discover unproposed starts or adjust original endpoints. Cleanup can remove a wholly false parent from the event timeline only after perfect human classification; exported footage still stays selected. Mixed parents remain. Synthetic cut ends are not observed dead-ball boundaries.

## Export invariance and accounting

Every automatic and reviewed arm equals this production table exactly at all four padding cases. Padding applies symmetrically; overlapping/touching ranges merge and positive gaps strictly below three seconds join. Ignored intervals are subtracted without rejoining. Correct removed time means omitted time outside the wanted human export; incorrect removed time means wanted human export omitted. All duration columns are minutes.

| Padding each side (s) | P_pad % | R_core % | F1_padP_coreR % | Model export min | Human export min | Model-human min | Correct removed min | Incorrect removed min | Incorrect export min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 65.50 | 95.75 | 77.79 | 58.16 | 39.79 | 18.37 | 77.98 | 1.69 | 20.07 |
| 1 | 69.40 | 98.30 | 81.36 | 70.39 | 50.52 | 19.87 | 65.77 | 1.67 | 21.54 |
| 2 | 72.45 | 99.27 | 83.76 | 82.38 | 61.25 | 21.13 | 53.88 | 1.57 | 22.70 |
| 3 | 75.02 | 99.49 | 85.54 | 93.95 | 72.05 | 21.90 | 42.31 | 1.57 | 23.47 |

## Automatic split policies

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 62.64 | 70.81 | 66.47 | 71.74 | 67.45 | 3.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 62.22 [61.89, 62.40] | 71.43 [71.12, 72.05] | 66.51 [66.18, 66.86] | 72.88 [72.36, 73.29] | 67.95 [67.73, 68.11] | 3.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 59.24 [58.25, 60.00] | 71.01 [70.19, 71.74] | 64.60 [63.66, 65.35] | 72.98 [72.67, 73.29] | 66.48 [66.29, 66.86] | 3.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 58.57 [57.39, 60.72] | 72.36 [71.12, 72.98] | 64.73 [63.52, 66.29] | 77.74 [76.71, 79.19] | 69.63 [69.17, 69.96] | 3.00 | 0.00 | 0.00 | 0.00 |

Automatic split precision/recall/F1 below score proposed timestamps against secondary starts within the same original parent and valid component. A duplicate or unmatched proposal is spurious. A first rally start is not a split target. Parent partitions preserve raw occupancy exactly.

| Policy | Tolerance (s) | Proposals | Matched | Spurious | Missed targets | Split P % | Split R % | Split F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corroborated | 0.5 | 5.67 [3.00, 8.00] | 1.00 [0.00, 2.00] | 4.67 [3.00, 6.00] | 6.00 [5.00, 7.00] | 13.89 [0.00, 25.00] | 14.29 [0.00, 28.57] | 14.02 [0.00, 26.67] |
| corroborated | 1 | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 4.67 [3.00, 6.00] | 38.89 [33.33, 50.00] | 33.33 [14.29, 57.14] | 34.70 [20.00, 53.33] |
| corroborated | 2 | 5.67 [3.00, 8.00] | 2.67 [1.00, 5.00] | 3.00 [2.00, 4.00] | 4.33 [2.00, 6.00] | 43.06 [33.33, 62.50] | 38.10 [14.29, 71.43] | 39.15 [20.00, 66.67] |
| event_starts | 0.5 | 22.00 [21.00, 24.00] | 1.67 [0.00, 4.00] | 20.33 [17.00, 23.00] | 5.33 [3.00, 7.00] | 7.74 [0.00, 19.05] | 23.81 [0.00, 57.14] | 11.67 [0.00, 28.57] |
| event_starts | 1 | 22.00 [21.00, 24.00] | 2.67 [2.00, 4.00] | 19.33 [17.00, 22.00] | 4.33 [3.00, 5.00] | 12.30 [8.33, 19.05] | 38.10 [28.57, 57.14] | 18.59 [12.90, 28.57] |
| event_starts | 2 | 22.00 [21.00, 24.00] | 3.00 [2.00, 5.00] | 19.00 [16.00, 22.00] | 4.00 [2.00, 5.00] | 13.89 [8.33, 23.81] | 42.86 [28.57, 71.43] | 20.97 [12.90, 35.71] |
| head_evidence | 0.5 | 34.00 [23.00, 44.00] | 1.33 [1.00, 2.00] | 32.67 [22.00, 42.00] | 5.67 [5.00, 6.00] | 3.92 [2.86, 4.55] | 19.05 [14.29, 28.57] | 6.42 [4.76, 7.84] |
| head_evidence | 1 | 34.00 [23.00, 44.00] | 2.67 [2.00, 4.00] | 31.33 [21.00, 40.00] | 4.33 [3.00, 5.00] | 7.83 [5.71, 9.09] | 38.10 [28.57, 57.14] | 12.85 [9.52, 15.69] |
| head_evidence | 2 | 34.00 [23.00, 44.00] | 3.00 [2.00, 5.00] | 31.00 [21.00, 39.00] | 4.00 [2.00, 5.00] | 8.59 [5.71, 11.36] | 42.86 [28.57, 71.43] | 14.15 [9.52, 19.61] |

## Cleanup flag yield before review

Cleanup means compact supports less than half of an original parent after two-second neural dilation. These are heuristic flags, not calibrated probabilities. Wholly-false precision is the fraction whose event record can be removed with zero gold-core overlap; real/mixed flags must be retained in this restricted simulation.

| Scope | Flagged parents | Wholly false | Real/mixed | Wholly-false precision % | Real rallies touched |
| --- | --- | --- | --- | --- | --- |
| pooled | 74.67 [70.00, 77.00] | 37.00 [35.00, 39.00] | 37.67 [35.00, 40.00] | 49.57 [48.05, 50.65] | 38.33 [35.00, 41.00] |
| source-group-005 | 23.33 [22.00, 25.00] | 9.67 [9.00, 10.00] | 13.67 [12.00, 16.00] | 41.64 [36.00, 45.45] | 14.00 [13.00, 16.00] |
| source-group-007 | 44.33 [37.00, 49.00] | 22.33 [21.00, 24.00] | 22.00 [16.00, 25.00] | 50.85 [46.81, 56.76] | 22.33 [16.00, 26.00] |
| source-group-009 | 2.33 [2.00, 3.00] | 2.00 | 0.33 [0.00, 1.00] | 88.89 [66.67, 100.00] | 0.33 [0.00, 1.00] |
| source-group-012 | 4.67 [4.00, 5.00] | 3.00 | 1.67 [1.00, 2.00] | 65.00 [60.00, 75.00] | 1.67 [1.00, 2.00] |

## All restricted review configurations

Arm names give split policy, review inventory, ordering and per-video budget cap. The actual reviewed minutes can be below the cap because a whole parent does not fit or the queue is exhausted. `none--cleanup_only` isolates cleanup without split advice. All rows retain the original production export.

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corroborated--combined--chronological--budget-05 | 65.77 [65.71, 65.90] | 70.81 | 68.20 [68.16, 68.26] | 71.74 | 69.09 [69.06, 69.16] | 3.00 | 0.00 | 5.09 [4.78, 5.33] | 16.67 [15.00, 18.00] |
| corroborated--combined--chronological--budget-10 | 67.43 [66.67, 67.86] | 71.12 [70.81, 71.43] | 69.22 [68.97, 69.39] | 72.05 [71.74, 72.36] | 70.13 [69.87, 70.30] | 3.00 | 0.00 | 8.56 [8.28, 8.72] | 27.67 [27.00, 28.00] |
| corroborated--combined--chronological--budget-20 | 69.05 [68.86, 69.30] | 71.12 [70.81, 71.43] | 70.07 [70.03, 70.12] | 72.36 [72.05, 72.67] | 71.29 [71.25, 71.34] | 3.00 | 0.00 | 12.79 [12.02, 13.64] | 41.67 [40.00, 43.00] |
| corroborated--combined--chronological--budget-40 | 69.57 [69.18, 69.94] | 71.22 [70.81, 71.74] | 70.38 [70.14, 70.64] | 72.46 [72.05, 72.98] | 71.61 [71.36, 71.87] | 3.00 | 0.00 | 14.89 [13.79, 16.78] | 46.00 [43.00, 51.00] |
| corroborated--combined--evidence--budget-05 | 67.79 [67.66, 67.86] | 70.81 | 69.27 [69.20, 69.30] | 71.74 | 70.18 [70.11, 70.21] | 3.00 | 0.00 | 4.98 [4.68, 5.20] | 15.33 [14.00, 17.00] |
| corroborated--combined--evidence--budget-10 | 69.02 [68.88, 69.09] | 70.81 | 69.90 [69.83, 69.94] | 71.74 | 70.82 [70.75, 70.86] | 3.00 | 0.00 | 8.37 [8.02, 8.66] | 29.33 [28.00, 30.00] |
| corroborated--combined--evidence--budget-20 | 69.61 [69.18, 69.91] | 71.12 [70.81, 71.43] | 70.35 [70.14, 70.66] | 72.26 [72.05, 72.36] | 71.48 [71.36, 71.58] | 3.00 | 0.00 | 12.53 [11.70, 13.48] | 41.67 [39.00, 44.00] |
| corroborated--combined--evidence--budget-40 | 69.64 [69.18, 69.94] | 71.22 [70.81, 71.74] | 70.42 [70.14, 70.75] | 72.46 [72.05, 72.98] | 71.65 [71.36, 71.98] | 3.00 | 0.00 | 14.74 [13.79, 16.31] | 46.33 [43.00, 52.00] |
| corroborated--split_only--chronological--budget-05 | 62.74 [62.64, 62.84] | 71.12 [70.81, 71.43] | 66.67 [66.47, 66.86] | 72.05 [71.74, 72.36] | 67.64 [67.45, 67.83] | 3.00 | 0.00 | 1.40 [0.42, 2.49] | 4.33 [1.00, 8.00] |
| corroborated--split_only--chronological--budget-10 | 62.57 [62.47, 62.67] | 71.12 [70.81, 71.43] | 66.57 [66.38, 66.76] | 72.36 [72.05, 72.67] | 67.83 [67.64, 68.02] | 3.00 | 0.00 | 2.33 [1.18, 3.25] | 7.00 [3.00, 10.00] |
| corroborated--split_only--chronological--budget-20 | 62.60 [62.47, 62.77] | 71.22 [70.81, 71.74] | 66.63 [66.38, 66.96] | 72.46 [72.05, 72.98] | 67.89 [67.64, 68.21] | 3.00 | 0.00 | 3.25 [2.06, 4.28] | 9.00 [4.00, 14.00] |
| corroborated--split_only--chronological--budget-40 | 62.60 [62.47, 62.77] | 71.22 [70.81, 71.74] | 66.63 [66.38, 66.96] | 72.46 [72.05, 72.98] | 67.89 [67.64, 68.21] | 3.00 | 0.00 | 3.25 [2.06, 4.28] | 9.00 [4.00, 14.00] |
| corroborated--split_only--evidence--budget-05 | 62.74 [62.64, 62.84] | 71.12 [70.81, 71.43] | 66.67 [66.47, 66.86] | 72.05 [71.74, 72.36] | 67.64 [67.45, 67.83] | 3.00 | 0.00 | 1.40 [0.42, 2.49] | 4.33 [1.00, 8.00] |
| corroborated--split_only--evidence--budget-10 | 62.66 [62.47, 62.94] | 71.22 [70.81, 71.74] | 66.67 [66.38, 67.05] | 72.36 [72.05, 72.67] | 67.83 [67.64, 68.02] | 3.00 | 0.00 | 2.27 [1.18, 3.09] | 7.00 [3.00, 10.00] |
| corroborated--split_only--evidence--budget-20 | 62.60 [62.47, 62.77] | 71.22 [70.81, 71.74] | 66.63 [66.38, 66.96] | 72.46 [72.05, 72.98] | 67.89 [67.64, 68.21] | 3.00 | 0.00 | 3.25 [2.06, 4.28] | 9.00 [4.00, 14.00] |
| corroborated--split_only--evidence--budget-40 | 62.60 [62.47, 62.77] | 71.22 [70.81, 71.74] | 66.63 [66.38, 66.96] | 72.46 [72.05, 72.98] | 67.89 [67.64, 68.21] | 3.00 | 0.00 | 3.25 [2.06, 4.28] | 9.00 [4.00, 14.00] |
| event_starts--combined--chronological--budget-05 | 65.46 [65.14, 65.90] | 70.81 | 68.03 [67.86, 68.26] | 71.74 | 68.92 [68.75, 69.16] | 3.00 | 0.00 | 5.64 [5.59, 5.73] | 19.33 [19.00, 20.00] |
| event_starts--combined--chronological--budget-10 | 66.64 [66.09, 66.96] | 71.12 [70.81, 71.43] | 68.80 [68.37, 69.07] | 72.05 [71.74, 72.36] | 69.70 [69.27, 69.97] | 3.00 | 0.00 | 10.02 [9.90, 10.15] | 33.00 [32.00, 34.00] |
| event_starts--combined--chronological--budget-20 | 68.16 [67.75, 68.47] | 71.12 [70.81, 71.43] | 69.60 [69.39, 69.80] | 72.36 [72.05, 72.67] | 70.82 [70.61, 71.02] | 3.00 | 0.00 | 16.01 [15.59, 16.54] | 51.33 [50.00, 54.00] |
| event_starts--combined--chronological--budget-40 | 69.60 [69.18, 70.03] | 71.33 [71.12, 71.74] | 70.45 [70.14, 70.64] | 72.57 [72.36, 72.98] | 71.68 [71.36, 71.87] | 3.00 | 0.00 | 20.38 [19.56, 20.87] | 63.00 [61.00, 64.00] |
| event_starts--combined--evidence--budget-05 | 67.79 [67.66, 67.86] | 70.81 | 69.27 [69.20, 69.30] | 71.74 | 70.18 [70.11, 70.21] | 3.00 | 0.00 | 5.46 [5.27, 5.65] | 17.00 [16.00, 18.00] |
| event_starts--combined--evidence--budget-10 | 69.02 [68.88, 69.09] | 70.81 | 69.90 [69.83, 69.94] | 71.74 | 70.82 [70.75, 70.86] | 3.00 | 0.00 | 9.79 [9.59, 10.00] | 34.33 [34.00, 35.00] |
| event_starts--combined--evidence--budget-20 | 69.69 [69.30, 69.94] | 70.91 [70.81, 71.12] | 70.29 [70.05, 70.46] | 71.84 [71.74, 72.05] | 71.22 [70.97, 71.38] | 3.00 | 0.00 | 15.69 [14.89, 16.42] | 52.67 [50.00, 55.00] |
| event_starts--combined--evidence--budget-40 | 69.67 [69.18, 70.03] | 71.33 [71.12, 71.74] | 70.49 [70.14, 70.75] | 72.57 [72.36, 72.98] | 71.71 [71.36, 71.98] | 3.00 | 0.00 | 20.22 [19.56, 20.69] | 63.33 [61.00, 65.00] |
| event_starts--split_only--chronological--budget-05 | 62.74 [62.64, 62.84] | 71.12 [70.81, 71.43] | 66.67 [66.47, 66.86] | 72.05 [71.74, 72.36] | 67.64 [67.45, 67.83] | 3.00 | 0.00 | 4.35 [4.09, 4.78] | 14.00 [13.00, 15.00] |
| event_starts--split_only--chronological--budget-10 | 62.57 [62.47, 62.67] | 71.12 [70.81, 71.43] | 66.57 [66.38, 66.76] | 72.36 [72.05, 72.67] | 67.83 [67.64, 68.02] | 3.00 | 0.00 | 6.76 [6.64, 6.94] | 20.67 [20.00, 21.00] |
| event_starts--split_only--chronological--budget-20 | 62.57 [62.47, 62.67] | 71.12 [70.81, 71.43] | 66.57 [66.38, 66.76] | 72.36 [72.05, 72.67] | 67.83 [67.64, 68.02] | 3.00 | 0.00 | 8.42 [8.02, 8.65] | 24.33 [23.00, 26.00] |
| event_starts--split_only--chronological--budget-40 | 62.64 [62.57, 62.77] | 71.33 [71.12, 71.74] | 66.70 [66.57, 66.96] | 72.57 [72.36, 72.98] | 67.96 [67.83, 68.21] | 3.00 | 0.00 | 9.02 [8.62, 9.26] | 26.33 [25.00, 28.00] |
| event_starts--split_only--evidence--budget-05 | 62.64 | 70.81 | 66.47 | 71.74 | 67.45 | 3.00 | 0.00 | 4.07 [3.71, 4.42] | 13.67 [13.00, 15.00] |
| event_starts--split_only--evidence--budget-10 | 62.71 [62.64, 62.84] | 71.01 [70.81, 71.43] | 66.60 [66.47, 66.86] | 71.95 [71.74, 72.36] | 67.57 [67.45, 67.83] | 3.00 | 0.00 | 6.49 [6.12, 6.97] | 20.67 [20.00, 21.00] |
| event_starts--split_only--evidence--budget-20 | 62.64 [62.57, 62.77] | 71.33 [71.12, 71.74] | 66.70 [66.57, 66.96] | 72.57 [72.36, 72.98] | 67.96 [67.83, 68.21] | 3.00 | 0.00 | 8.15 [7.75, 8.38] | 25.33 [24.00, 27.00] |
| event_starts--split_only--evidence--budget-40 | 62.64 [62.57, 62.77] | 71.33 [71.12, 71.74] | 66.70 [66.57, 66.96] | 72.57 [72.36, 72.98] | 67.96 [67.83, 68.21] | 3.00 | 0.00 | 9.02 [8.62, 9.26] | 26.33 [25.00, 28.00] |
| head_evidence--combined--chronological--budget-05 | 65.36 [65.14, 65.52] | 70.91 [70.81, 71.12] | 68.02 [67.86, 68.15] | 71.84 [71.74, 72.05] | 68.92 [68.75, 69.05] | 3.00 | 0.00 | 5.98 [5.35, 6.36] | 20.33 [18.00, 23.00] |
| head_evidence--combined--chronological--budget-10 | 66.80 [66.47, 66.96] | 71.22 [71.12, 71.43] | 68.94 [68.86, 68.98] | 72.15 [72.05, 72.36] | 69.84 [69.76, 69.88] | 3.00 | 0.00 | 10.81 [10.08, 11.38] | 35.33 [32.00, 37.00] |
| head_evidence--combined--chronological--budget-20 | 68.39 [67.85, 68.77] | 71.22 [71.12, 71.43] | 69.78 [69.59, 69.92] | 72.46 [72.36, 72.67] | 70.99 [70.80, 71.15] | 3.00 | 0.00 | 16.69 [15.00, 18.03] | 55.33 [49.00, 60.00] |
| head_evidence--combined--chronological--budget-40 | 69.53 [69.18, 70.03] | 71.33 [71.12, 71.74] | 70.41 [70.14, 70.57] | 72.57 [72.36, 72.98] | 71.64 [71.36, 71.80] | 3.00 | 0.00 | 20.15 [18.52, 22.49] | 64.00 [57.00, 71.00] |
| head_evidence--combined--evidence--budget-05 | 67.79 [67.66, 67.86] | 70.81 | 69.27 [69.20, 69.30] | 71.74 | 70.18 [70.11, 70.21] | 3.00 | 0.00 | 5.82 [5.16, 6.16] | 18.67 [16.00, 21.00] |
| head_evidence--combined--evidence--budget-10 | 69.02 [68.88, 69.09] | 70.81 | 69.90 [69.83, 69.94] | 71.74 | 70.82 [70.75, 70.86] | 3.00 | 0.00 | 10.52 [9.99, 10.95] | 37.00 [34.00, 39.00] |
| head_evidence--combined--evidence--budget-20 | 69.68 [69.39, 69.82] | 71.12 | 70.39 [70.25, 70.46] | 72.15 [72.05, 72.36] | 71.41 [71.17, 71.69] | 3.00 | 0.00 | 16.53 [15.03, 17.58] | 56.00 [50.00, 59.00] |
| head_evidence--combined--evidence--budget-40 | 69.67 [69.18, 70.03] | 71.33 [71.12, 71.74] | 70.49 [70.14, 70.75] | 72.57 [72.36, 72.98] | 71.71 [71.36, 71.98] | 3.00 | 0.00 | 20.03 [18.52, 22.14] | 64.33 [57.00, 72.00] |
| head_evidence--split_only--chronological--budget-05 | 62.71 [62.64, 62.74] | 71.01 [70.81, 71.12] | 66.60 [66.47, 66.67] | 71.95 [71.74, 72.05] | 67.57 [67.45, 67.64] | 3.00 | 0.00 | 4.37 [3.31, 4.97] | 16.00 [13.00, 18.00] |
| head_evidence--split_only--chronological--budget-10 | 62.60 [62.57, 62.67] | 71.22 [71.12, 71.43] | 66.63 [66.57, 66.76] | 72.46 [72.36, 72.67] | 67.90 [67.83, 68.02] | 3.00 | 0.00 | 7.92 [6.32, 9.14] | 26.33 [21.00, 30.00] |
| head_evidence--split_only--chronological--budget-20 | 62.60 [62.57, 62.67] | 71.22 [71.12, 71.43] | 66.63 [66.57, 66.76] | 72.46 [72.36, 72.67] | 67.90 [67.83, 68.02] | 3.00 | 0.00 | 9.86 [7.92, 12.00] | 30.67 [24.00, 37.00] |
| head_evidence--split_only--chronological--budget-40 | 62.64 [62.57, 62.77] | 71.33 [71.12, 71.74] | 66.70 [66.57, 66.96] | 72.57 [72.36, 72.98] | 67.96 [67.83, 68.21] | 3.00 | 0.00 | 10.29 [7.92, 13.29] | 32.33 [24.00, 42.00] |
| head_evidence--split_only--evidence--budget-05 | 62.64 | 70.81 | 66.47 | 71.74 | 67.45 | 3.00 | 0.00 | 4.28 [3.28, 4.92] | 16.33 [13.00, 19.00] |
| head_evidence--split_only--evidence--budget-10 | 62.66 [62.57, 62.84] | 71.22 [71.12, 71.43] | 66.67 [66.57, 66.86] | 72.36 | 67.83 | 3.00 | 0.00 | 7.72 [6.11, 8.86] | 26.67 [21.00, 31.00] |
| head_evidence--split_only--evidence--budget-20 | 62.64 [62.57, 62.77] | 71.33 [71.12, 71.74] | 66.70 [66.57, 66.96] | 72.57 [72.36, 72.98] | 67.96 [67.83, 68.21] | 3.00 | 0.00 | 9.85 [7.92, 11.98] | 31.33 [24.00, 39.00] |
| head_evidence--split_only--evidence--budget-40 | 62.64 [62.57, 62.77] | 71.33 [71.12, 71.74] | 66.70 [66.57, 66.96] | 72.57 [72.36, 72.98] | 67.96 [67.83, 68.21] | 3.00 | 0.00 | 10.29 [7.92, 13.29] | 32.33 [24.00, 42.00] |
| none--cleanup_only--chronological--budget-05 | 65.77 [65.71, 65.90] | 70.81 | 68.20 [68.16, 68.26] | 71.74 | 69.09 [69.06, 69.16] | 3.00 | 0.00 | 4.85 [4.78, 4.96] | 16.00 [15.00, 17.00] |
| none--cleanup_only--chronological--budget-10 | 67.72 [67.66, 67.86] | 70.81 | 69.23 [69.20, 69.30] | 71.74 | 70.14 [70.11, 70.21] | 3.00 | 0.00 | 8.24 [8.04, 8.38] | 26.67 [26.00, 27.00] |
| none--cleanup_only--chronological--budget-20 | 69.51 [69.30, 69.72] | 70.81 | 70.15 [70.05, 70.26] | 71.74 | 71.08 [70.97, 71.19] | 3.00 | 0.00 | 11.29 [10.36, 12.11] | 36.67 [35.00, 38.00] |
| none--cleanup_only--chronological--budget-40 | 69.73 [69.30, 70.15] | 70.81 | 70.26 [70.05, 70.48] | 71.74 | 71.19 [70.97, 71.41] | 3.00 | 0.00 | 12.02 [10.36, 13.64] | 38.33 [35.00, 41.00] |
| none--cleanup_only--evidence--budget-05 | 67.79 [67.66, 67.86] | 70.81 | 69.27 [69.20, 69.30] | 71.74 | 70.18 [70.11, 70.21] | 3.00 | 0.00 | 4.74 [4.68, 4.83] | 14.67 [14.00, 16.00] |
| none--cleanup_only--evidence--budget-10 | 69.02 [68.88, 69.09] | 70.81 | 69.90 [69.83, 69.94] | 71.74 | 70.82 [70.75, 70.86] | 3.00 | 0.00 | 8.01 [7.95, 8.07] | 28.33 [28.00, 29.00] |
| none--cleanup_only--evidence--budget-20 | 69.65 [69.30, 69.94] | 70.81 | 70.23 [70.05, 70.37] | 71.74 | 71.15 [70.97, 71.30] | 3.00 | 0.00 | 11.28 [10.36, 12.03] | 37.00 [35.00, 38.00] |
| none--cleanup_only--evidence--budget-40 | 69.73 [69.30, 70.15] | 70.81 | 70.26 [70.05, 70.48] | 71.74 | 71.19 [70.97, 71.41] | 3.00 | 0.00 | 12.02 [10.36, 13.64] | 38.33 [35.00, 41.00] |

### Review actions and workload

| Arm | Parent jobs | Playback clips | Split proposals reviewed | Accepted | Rejected | False parents removed | Event timeline removed min | Unused budget min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corroborated--combined--chronological--budget-05 | 33.67 [31.00, 36.00] | 33.67 [31.00, 36.00] | 0.67 [0.00, 1.00] | 0.00 | 0.67 [0.00, 1.00] | 17.33 [17.00, 18.00] | 0.90 [0.87, 0.92] | 1.80 [1.56, 2.11] |
| corroborated--combined--chronological--budget-10 | 51.67 [47.00, 54.00] | 51.00 [47.00, 53.00] | 2.33 [0.00, 4.00] | 1.00 [0.00, 2.00] | 1.33 [0.00, 2.00] | 25.33 [21.00, 28.00] | 1.46 [1.17, 1.68] | 5.22 [5.06, 5.50] |
| corroborated--combined--chronological--budget-20 | 73.00 [72.00, 74.00] | 70.67 [70.00, 72.00] | 4.00 [1.00, 6.00] | 2.00 [1.00, 3.00] | 2.00 [0.00, 3.00] | 34.33 [33.00, 36.00] | 2.12 [1.96, 2.26] | 14.78 [13.93, 15.55] |
| corroborated--combined--chronological--budget-40 | 79.00 [76.00, 81.00] | 74.67 [71.00, 77.00] | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 36.67 [35.00, 39.00] | 2.30 [2.04, 2.54] | 40.24 [38.36, 41.34] |
| corroborated--combined--evidence--budget-05 | 43.00 [42.00, 44.00] | 42.00 [41.00, 43.00] | 0.67 [0.00, 1.00] | 0.00 | 0.67 [0.00, 1.00] | 27.67 [27.00, 28.00] | 1.34 [1.29, 1.38] | 1.91 [1.69, 2.21] |
| corroborated--combined--evidence--budget-10 | 63.00 [62.00, 64.00] | 61.00 [60.00, 62.00] | 1.00 [0.00, 2.00] | 0.00 | 1.00 [0.00, 2.00] | 33.67 [33.00, 34.00] | 1.81 [1.76, 1.85] | 5.41 [5.12, 5.77] |
| corroborated--combined--evidence--budget-20 | 75.67 [74.00, 78.00] | 72.33 [70.00, 74.00] | 3.33 [1.00, 5.00] | 1.67 [1.00, 2.00] | 1.67 [0.00, 3.00] | 36.67 [35.00, 38.00] | 2.26 [2.04, 2.39] | 15.04 [14.08, 15.87] |
| corroborated--combined--evidence--budget-40 | 79.67 [76.00, 83.00] | 75.33 [71.00, 79.00] | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 37.00 [35.00, 39.00] | 2.32 [2.04, 2.54] | 40.40 [38.82, 41.34] |
| corroborated--split_only--chronological--budget-05 | 3.00 [1.00, 5.00] | 3.00 [1.00, 5.00] | 3.00 [1.00, 5.00] | 1.00 [0.00, 2.00] | 2.00 [1.00, 3.00] | 0.00 | 0.00 | 5.49 [4.40, 6.47] |
| corroborated--split_only--chronological--budget-10 | 4.33 [2.00, 6.00] | 4.33 [2.00, 6.00] | 4.33 [2.00, 6.00] | 2.00 [1.00, 3.00] | 2.33 [1.00, 3.00] | 0.00 | 0.00 | 11.46 [10.53, 12.60] |
| corroborated--split_only--chronological--budget-20 | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 0.00 | 0.00 | 24.31 [23.29, 25.51] |
| corroborated--split_only--chronological--budget-40 | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 0.00 | 0.00 | 51.88 [50.85, 53.08] |
| corroborated--split_only--evidence--budget-05 | 3.00 [1.00, 5.00] | 3.00 [1.00, 5.00] | 3.00 [1.00, 5.00] | 1.00 [0.00, 2.00] | 2.00 [1.00, 3.00] | 0.00 | 0.00 | 5.49 [4.40, 6.47] |
| corroborated--split_only--evidence--budget-10 | 4.33 [2.00, 6.00] | 4.33 [2.00, 6.00] | 4.33 [2.00, 6.00] | 2.00 [1.00, 3.00] | 2.33 [1.00, 3.00] | 0.00 | 0.00 | 11.51 [10.69, 12.60] |
| corroborated--split_only--evidence--budget-20 | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 0.00 | 0.00 | 24.31 [23.29, 25.51] |
| corroborated--split_only--evidence--budget-40 | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 5.67 [3.00, 8.00] | 2.33 [1.00, 4.00] | 3.33 [2.00, 4.00] | 0.00 | 0.00 | 51.88 [50.85, 53.08] |
| event_starts--combined--chronological--budget-05 | 34.67 [33.00, 37.00] | 34.67 [33.00, 37.00] | 4.33 [4.00, 5.00] | 0.00 | 4.33 [4.00, 5.00] | 15.67 [14.00, 18.00] | 0.82 [0.76, 0.90] | 1.25 [1.17, 1.30] |
| event_starts--combined--chronological--budget-10 | 52.00 [50.00, 54.00] | 52.00 [50.00, 54.00] | 10.33 [8.00, 12.00] | 1.00 [0.00, 2.00] | 9.33 [6.00, 12.00] | 21.33 [19.00, 23.00] | 1.12 [1.00, 1.21] | 3.76 [3.64, 3.88] |
| event_starts--combined--chronological--budget-20 | 77.00 [74.00, 82.00] | 73.33 [70.00, 79.00] | 17.67 [16.00, 20.00] | 2.00 [1.00, 3.00] | 15.67 [13.00, 18.00] | 30.00 [28.00, 32.00] | 1.86 [1.59, 2.03] | 11.56 [11.03, 11.98] |
| event_starts--combined--chronological--budget-40 | 93.67 [91.00, 97.00] | 87.00 [84.00, 91.00] | 22.00 [21.00, 24.00] | 2.67 [2.00, 4.00] | 19.33 [17.00, 22.00] | 36.67 [35.00, 39.00] | 2.30 [2.04, 2.54] | 34.76 [34.26, 35.57] |
| event_starts--combined--evidence--budget-05 | 44.67 [44.00, 45.00] | 43.67 [43.00, 44.00] | 2.67 [2.00, 3.00] | 0.00 | 2.67 [2.00, 3.00] | 27.67 [27.00, 28.00] | 1.34 [1.29, 1.38] | 1.43 [1.24, 1.62] |
| event_starts--combined--evidence--budget-10 | 68.00 [67.00, 69.00] | 66.00 [65.00, 67.00] | 7.00 [6.00, 8.00] | 0.00 | 7.00 [6.00, 8.00] | 33.67 [33.00, 34.00] | 1.81 [1.76, 1.85] | 4.00 [3.79, 4.20] |
| event_starts--combined--evidence--budget-20 | 86.00 [83.00, 89.00] | 81.67 [78.00, 85.00] | 15.67 [14.00, 17.00] | 0.33 [0.00, 1.00] | 15.33 [13.00, 17.00] | 36.67 [35.00, 38.00] | 2.26 [2.04, 2.39] | 11.88 [11.15, 12.68] |
| event_starts--combined--evidence--budget-40 | 94.33 [91.00, 97.00] | 87.67 [84.00, 91.00] | 21.67 [20.00, 24.00] | 2.67 [2.00, 4.00] | 19.00 [16.00, 22.00] | 37.00 [35.00, 39.00] | 2.32 [2.04, 2.54] | 34.91 [34.44, 35.57] |
| event_starts--split_only--chronological--budget-05 | 11.67 [11.00, 13.00] | 11.67 [11.00, 13.00] | 12.33 [11.00, 13.00] | 1.00 [0.00, 2.00] | 11.33 [9.00, 13.00] | 0.00 | 0.00 | 2.54 [2.12, 2.80] |
| event_starts--split_only--chronological--budget-10 | 16.67 [16.00, 17.00] | 16.00 [15.00, 17.00] | 18.00 [16.00, 20.00] | 2.00 [1.00, 3.00] | 16.00 [13.00, 18.00] | 0.00 | 0.00 | 7.02 [6.84, 7.15] |
| event_starts--split_only--chronological--budget-20 | 19.67 [19.00, 20.00] | 19.00 | 21.00 [20.00, 23.00] | 2.00 [1.00, 3.00] | 19.00 [17.00, 21.00] | 0.00 | 0.00 | 19.15 [18.91, 19.55] |
| event_starts--split_only--chronological--budget-40 | 20.67 [20.00, 21.00] | 20.00 | 22.00 [21.00, 24.00] | 2.67 [2.00, 4.00] | 19.33 [17.00, 22.00] | 0.00 | 0.00 | 46.11 [45.88, 46.51] |
| event_starts--split_only--evidence--budget-05 | 12.00 [11.00, 13.00] | 12.00 [11.00, 13.00] | 13.00 [11.00, 14.00] | 0.00 | 13.00 [11.00, 14.00] | 0.00 | 0.00 | 2.83 [2.47, 3.18] |
| event_starts--split_only--evidence--budget-10 | 17.00 [16.00, 18.00] | 16.67 [16.00, 17.00] | 18.00 [16.00, 20.00] | 0.67 [0.00, 2.00] | 17.33 [14.00, 20.00] | 0.00 | 0.00 | 7.29 [6.82, 7.67] |
| event_starts--split_only--evidence--budget-20 | 19.67 [19.00, 20.00] | 19.00 | 21.00 [20.00, 23.00] | 2.67 [2.00, 4.00] | 18.33 [16.00, 21.00] | 0.00 | 0.00 | 19.42 [19.19, 19.82] |
| event_starts--split_only--evidence--budget-40 | 20.67 [20.00, 21.00] | 20.00 | 22.00 [21.00, 24.00] | 2.67 [2.00, 4.00] | 19.33 [17.00, 22.00] | 0.00 | 0.00 | 46.11 [45.88, 46.51] |
| head_evidence--combined--chronological--budget-05 | 35.00 [32.00, 39.00] | 35.00 [32.00, 39.00] | 10.00 [7.00, 13.00] | 0.33 [0.00, 1.00] | 9.67 [6.00, 13.00] | 15.00 [14.00, 16.00] | 0.78 [0.75, 0.82] | 0.92 [0.53, 1.54] |
| head_evidence--combined--chronological--budget-10 | 56.00 [54.00, 59.00] | 55.33 [53.00, 58.00] | 18.00 [13.00, 23.00] | 1.33 [1.00, 2.00] | 16.67 [12.00, 21.00] | 22.00 [20.00, 23.00] | 1.27 [1.09, 1.38] | 2.98 [2.40, 3.70] |
| head_evidence--combined--chronological--budget-20 | 83.33 [80.00, 86.00] | 76.67 [74.00, 79.00] | 26.33 [19.00, 35.00] | 2.33 [2.00, 3.00] | 24.00 [17.00, 32.00] | 31.00 [28.00, 33.00] | 1.80 [1.64, 1.95] | 10.88 [9.54, 12.57] |
| head_evidence--combined--chronological--budget-40 | 96.67 [94.00, 100.00] | 87.33 [85.00, 91.00] | 33.33 [23.00, 42.00] | 2.67 [2.00, 4.00] | 30.67 [21.00, 38.00] | 36.33 [35.00, 39.00] | 2.29 [2.04, 2.54] | 34.99 [32.64, 36.62] |
| head_evidence--combined--evidence--budget-05 | 46.33 [44.00, 48.00] | 45.33 [43.00, 47.00] | 4.33 [2.00, 6.00] | 0.00 | 4.33 [2.00, 6.00] | 27.67 [27.00, 28.00] | 1.34 [1.29, 1.38] | 1.07 [0.73, 1.73] |
| head_evidence--combined--evidence--budget-10 | 70.67 [68.00, 73.00] | 67.67 [64.00, 70.00] | 11.33 [8.00, 13.00] | 0.00 | 11.33 [8.00, 13.00] | 33.67 [33.00, 34.00] | 1.81 [1.76, 1.85] | 3.26 [2.84, 3.79] |
| head_evidence--combined--evidence--budget-20 | 90.67 [86.00, 94.00] | 82.67 [79.00, 86.00] | 25.67 [18.00, 33.00] | 1.33 [1.00, 2.00] | 24.33 [16.00, 32.00] | 36.67 [35.00, 38.00] | 2.26 [2.04, 2.39] | 11.03 [9.99, 12.53] |
| head_evidence--combined--evidence--budget-40 | 98.00 [94.00, 104.00] | 88.33 [85.00, 94.00] | 33.33 [23.00, 42.00] | 2.67 [2.00, 4.00] | 30.67 [21.00, 38.00] | 37.00 [35.00, 39.00] | 2.32 [2.04, 2.54] | 35.11 [32.99, 36.62] |
| head_evidence--split_only--chronological--budget-05 | 15.67 [12.00, 18.00] | 15.67 [12.00, 18.00] | 16.33 [12.00, 19.00] | 0.67 [0.00, 1.00] | 15.67 [11.00, 19.00] | 0.00 | 0.00 | 2.52 [1.93, 3.59] |
| head_evidence--split_only--chronological--budget-10 | 23.67 [19.00, 27.00] | 23.00 [18.00, 27.00] | 25.33 [19.00, 30.00] | 2.33 [2.00, 3.00] | 23.00 [17.00, 27.00] | 0.00 | 0.00 | 5.86 [4.64, 7.47] |
| head_evidence--split_only--chronological--budget-20 | 28.33 [22.00, 35.00] | 27.33 [21.00, 34.00] | 33.00 [23.00, 41.00] | 2.33 [2.00, 3.00] | 30.67 [21.00, 38.00] | 0.00 | 0.00 | 17.71 [15.56, 19.65] |
| head_evidence--split_only--chronological--budget-40 | 29.33 [22.00, 38.00] | 28.33 [21.00, 37.00] | 34.00 [23.00, 44.00] | 2.67 [2.00, 4.00] | 31.33 [21.00, 40.00] | 0.00 | 0.00 | 44.85 [41.85, 47.21] |
| head_evidence--split_only--evidence--budget-05 | 16.67 [13.00, 19.00] | 16.67 [13.00, 19.00] | 17.67 [13.00, 20.00] | 0.00 | 17.67 [13.00, 20.00] | 0.00 | 0.00 | 2.61 [1.97, 3.61] |
| head_evidence--split_only--evidence--budget-10 | 24.33 [19.00, 29.00] | 23.33 [18.00, 28.00] | 26.33 [20.00, 32.00] | 2.00 | 24.33 [18.00, 30.00] | 0.00 | 0.00 | 6.06 [4.92, 7.68] |
| head_evidence--split_only--evidence--budget-20 | 28.67 [22.00, 36.00] | 27.67 [21.00, 35.00] | 33.33 [23.00, 42.00] | 2.67 [2.00, 4.00] | 30.67 [21.00, 38.00] | 0.00 | 0.00 | 17.71 [15.59, 19.65] |
| head_evidence--split_only--evidence--budget-40 | 29.33 [22.00, 38.00] | 28.33 [21.00, 37.00] | 34.00 [23.00, 44.00] | 2.67 [2.00, 4.00] | 31.33 [21.00, 40.00] | 0.00 | 0.00 | 44.85 [41.85, 47.21] |
| none--cleanup_only--chronological--budget-05 | 33.00 [31.00, 35.00] | 33.00 [31.00, 35.00] | 0.00 | 0.00 | 0.00 | 17.33 [17.00, 18.00] | 0.90 [0.87, 0.92] | 2.05 [1.93, 2.11] |
| none--cleanup_only--chronological--budget-10 | 53.67 [53.00, 54.00] | 52.67 [52.00, 53.00] | 0.00 | 0.00 | 0.00 | 27.33 [27.00, 28.00] | 1.62 [1.54, 1.68] | 5.55 [5.40, 5.74] |
| none--cleanup_only--chronological--budget-20 | 72.00 [70.00, 74.00] | 70.00 [67.00, 72.00] | 0.00 | 0.00 | 0.00 | 36.00 [35.00, 37.00] | 2.23 [2.04, 2.33] | 16.28 [15.46, 17.20] |
| none--cleanup_only--chronological--budget-40 | 74.67 [70.00, 77.00] | 71.67 [67.00, 75.00] | 0.00 | 0.00 | 0.00 | 37.00 [35.00, 39.00] | 2.32 [2.04, 2.54] | 43.12 [41.49, 44.77] |
| none--cleanup_only--evidence--budget-05 | 42.33 [42.00, 43.00] | 41.33 [41.00, 42.00] | 0.00 | 0.00 | 0.00 | 27.67 [27.00, 28.00] | 1.34 [1.29, 1.38] | 2.15 [2.06, 2.21] |
| none--cleanup_only--evidence--budget-10 | 62.00 | 60.00 | 0.00 | 0.00 | 0.00 | 33.67 [33.00, 34.00] | 1.81 [1.76, 1.85] | 5.77 [5.71, 5.83] |
| none--cleanup_only--evidence--budget-20 | 73.33 [70.00, 75.00] | 70.67 [67.00, 73.00] | 0.00 | 0.00 | 0.00 | 36.67 [35.00, 38.00] | 2.26 [2.04, 2.39] | 16.28 [15.54, 17.20] |
| none--cleanup_only--evidence--budget-40 | 74.67 [70.00, 77.00] | 71.67 [67.00, 75.00] | 0.00 | 0.00 | 0.00 | 37.00 [35.00, 39.00] | 2.32 [2.04, 2.54] | 43.12 [41.49, 44.77] |

### Confirmed split diagnostics

These rows score accepted, gold-snapped splits only. Their perfect precision is an assumption of the human oracle, not model precision; automatic proposal precision appears above. Accepted snaps are exact, so 0.5/1/2-second scores coincide. Full sensitivity metrics are retained in summary JSON.

| Arm | Confirmed splits | Unrecovered split targets | Confirmed split R % | Confirmed split F1 % | Material merged events | Material split gold rallies | Raw core loss (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| corroborated--combined--chronological--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| corroborated--combined--chronological--budget-10 | 1.00 [0.00, 2.00] | 6.00 [5.00, 7.00] | 14.29 [0.00, 28.57] | 23.15 [0.00, 44.44] | 6.00 [5.00, 7.00] | 4.00 | 0.00 |
| corroborated--combined--chronological--budget-20 | 2.00 [1.00, 3.00] | 5.00 [4.00, 6.00] | 28.57 [14.29, 42.86] | 43.15 [25.00, 60.00] | 5.00 [4.00, 6.00] | 4.00 | 0.00 |
| corroborated--combined--chronological--budget-40 | 2.33 [1.00, 4.00] | 4.67 [3.00, 6.00] | 33.33 [14.29, 57.14] | 47.39 [25.00, 72.73] | 4.67 [3.00, 6.00] | 4.00 | 0.00 |
| corroborated--combined--evidence--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| corroborated--combined--evidence--budget-10 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| corroborated--combined--evidence--budget-20 | 1.67 [1.00, 2.00] | 5.33 [5.00, 6.00] | 23.81 [14.29, 28.57] | 37.96 [25.00, 44.44] | 5.33 [5.00, 6.00] | 4.00 | 0.00 |
| corroborated--combined--evidence--budget-40 | 2.33 [1.00, 4.00] | 4.67 [3.00, 6.00] | 33.33 [14.29, 57.14] | 47.39 [25.00, 72.73] | 4.67 [3.00, 6.00] | 4.00 | 0.00 |
| corroborated--split_only--chronological--budget-05 | 1.00 [0.00, 2.00] | 6.00 [5.00, 7.00] | 14.29 [0.00, 28.57] | 23.15 [0.00, 44.44] | 6.00 [5.00, 7.00] | 4.00 | 0.00 |
| corroborated--split_only--chronological--budget-10 | 2.00 [1.00, 3.00] | 5.00 [4.00, 6.00] | 28.57 [14.29, 42.86] | 43.15 [25.00, 60.00] | 5.00 [4.00, 6.00] | 4.00 | 0.00 |
| corroborated--split_only--chronological--budget-20 | 2.33 [1.00, 4.00] | 4.67 [3.00, 6.00] | 33.33 [14.29, 57.14] | 47.39 [25.00, 72.73] | 4.67 [3.00, 6.00] | 4.00 | 0.00 |
| corroborated--split_only--chronological--budget-40 | 2.33 [1.00, 4.00] | 4.67 [3.00, 6.00] | 33.33 [14.29, 57.14] | 47.39 [25.00, 72.73] | 4.67 [3.00, 6.00] | 4.00 | 0.00 |
| corroborated--split_only--evidence--budget-05 | 1.00 [0.00, 2.00] | 6.00 [5.00, 7.00] | 14.29 [0.00, 28.57] | 23.15 [0.00, 44.44] | 6.00 [5.00, 7.00] | 4.00 | 0.00 |
| corroborated--split_only--evidence--budget-10 | 2.00 [1.00, 3.00] | 5.00 [4.00, 6.00] | 28.57 [14.29, 42.86] | 43.15 [25.00, 60.00] | 5.00 [4.00, 6.00] | 4.00 | 0.00 |
| corroborated--split_only--evidence--budget-20 | 2.33 [1.00, 4.00] | 4.67 [3.00, 6.00] | 33.33 [14.29, 57.14] | 47.39 [25.00, 72.73] | 4.67 [3.00, 6.00] | 4.00 | 0.00 |
| corroborated--split_only--evidence--budget-40 | 2.33 [1.00, 4.00] | 4.67 [3.00, 6.00] | 33.33 [14.29, 57.14] | 47.39 [25.00, 72.73] | 4.67 [3.00, 6.00] | 4.00 | 0.00 |
| event_starts--combined--chronological--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| event_starts--combined--chronological--budget-10 | 1.00 [0.00, 2.00] | 6.00 [5.00, 7.00] | 14.29 [0.00, 28.57] | 23.15 [0.00, 44.44] | 6.00 [5.00, 7.00] | 4.00 | 0.00 |
| event_starts--combined--chronological--budget-20 | 2.00 [1.00, 3.00] | 5.00 [4.00, 6.00] | 28.57 [14.29, 42.86] | 43.15 [25.00, 60.00] | 5.00 [4.00, 6.00] | 4.00 | 0.00 |
| event_starts--combined--chronological--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| event_starts--combined--evidence--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| event_starts--combined--evidence--budget-10 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| event_starts--combined--evidence--budget-20 | 0.33 [0.00, 1.00] | 6.67 [6.00, 7.00] | 4.76 [0.00, 14.29] | 8.33 [0.00, 25.00] | 6.67 [6.00, 7.00] | 4.00 | 0.00 |
| event_starts--combined--evidence--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| event_starts--split_only--chronological--budget-05 | 1.00 [0.00, 2.00] | 6.00 [5.00, 7.00] | 14.29 [0.00, 28.57] | 23.15 [0.00, 44.44] | 6.00 [5.00, 7.00] | 4.00 | 0.00 |
| event_starts--split_only--chronological--budget-10 | 2.00 [1.00, 3.00] | 5.00 [4.00, 6.00] | 28.57 [14.29, 42.86] | 43.15 [25.00, 60.00] | 5.00 [4.00, 6.00] | 4.00 | 0.00 |
| event_starts--split_only--chronological--budget-20 | 2.00 [1.00, 3.00] | 5.00 [4.00, 6.00] | 28.57 [14.29, 42.86] | 43.15 [25.00, 60.00] | 5.00 [4.00, 6.00] | 4.00 | 0.00 |
| event_starts--split_only--chronological--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| event_starts--split_only--evidence--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| event_starts--split_only--evidence--budget-10 | 0.67 [0.00, 2.00] | 6.33 [5.00, 7.00] | 9.52 [0.00, 28.57] | 14.81 [0.00, 44.44] | 6.33 [5.00, 7.00] | 4.00 | 0.00 |
| event_starts--split_only--evidence--budget-20 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| event_starts--split_only--evidence--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| head_evidence--combined--chronological--budget-05 | 0.33 [0.00, 1.00] | 6.67 [6.00, 7.00] | 4.76 [0.00, 14.29] | 8.33 [0.00, 25.00] | 6.67 [6.00, 7.00] | 4.00 | 0.00 |
| head_evidence--combined--chronological--budget-10 | 1.33 [1.00, 2.00] | 5.67 [5.00, 6.00] | 19.05 [14.29, 28.57] | 31.48 [25.00, 44.44] | 5.67 [5.00, 6.00] | 4.00 | 0.00 |
| head_evidence--combined--chronological--budget-20 | 2.33 [2.00, 3.00] | 4.67 [4.00, 5.00] | 33.33 [28.57, 42.86] | 49.63 [44.44, 60.00] | 4.67 [4.00, 5.00] | 4.00 | 0.00 |
| head_evidence--combined--chronological--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| head_evidence--combined--evidence--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| head_evidence--combined--evidence--budget-10 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| head_evidence--combined--evidence--budget-20 | 1.33 [1.00, 2.00] | 5.67 [5.00, 6.00] | 19.05 [14.29, 28.57] | 31.48 [25.00, 44.44] | 5.67 [5.00, 6.00] | 4.00 | 0.00 |
| head_evidence--combined--evidence--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| head_evidence--split_only--chronological--budget-05 | 0.67 [0.00, 1.00] | 6.33 [6.00, 7.00] | 9.52 [0.00, 14.29] | 16.67 [0.00, 25.00] | 6.33 [6.00, 7.00] | 4.00 | 0.00 |
| head_evidence--split_only--chronological--budget-10 | 2.33 [2.00, 3.00] | 4.67 [4.00, 5.00] | 33.33 [28.57, 42.86] | 49.63 [44.44, 60.00] | 4.67 [4.00, 5.00] | 4.00 | 0.00 |
| head_evidence--split_only--chronological--budget-20 | 2.33 [2.00, 3.00] | 4.67 [4.00, 5.00] | 33.33 [28.57, 42.86] | 49.63 [44.44, 60.00] | 4.67 [4.00, 5.00] | 4.00 | 0.00 |
| head_evidence--split_only--chronological--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| head_evidence--split_only--evidence--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| head_evidence--split_only--evidence--budget-10 | 2.00 | 5.00 | 28.57 | 44.44 | 5.00 | 4.00 | 0.00 |
| head_evidence--split_only--evidence--budget-20 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| head_evidence--split_only--evidence--budget-40 | 2.67 [2.00, 4.00] | 4.33 [3.00, 5.00] | 38.10 [28.57, 57.14] | 53.87 [44.44, 72.73] | 4.33 [3.00, 5.00] | 4.00 | 0.00 |
| none--cleanup_only--chronological--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--chronological--budget-10 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--chronological--budget-20 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--chronological--budget-40 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--evidence--budget-05 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--evidence--budget-10 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--evidence--budget-20 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |
| none--cleanup_only--evidence--budget-40 | 0.00 | 7.00 | 0.00 | 0.00 | 7.00 | 4.00 | 0.00 |

## Source-group sensitivity

Group metrics pool recording counts before rates. The full JSON additionally retains every numeric field, every padding case, and seed ranges within each scope.

### source-group-005

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 62.11 | 69.41 | 65.56 | 71.76 | 67.78 | 2.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 57.82 [55.77, 60.40] | 69.80 [68.24, 71.76] | 63.24 [61.38, 65.59] | 72.55 [71.76, 72.94] | 65.72 [64.55, 66.67] | 2.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 60.14 [59.41, 61.00] | 70.98 [70.59, 71.76] | 65.11 [64.52, 65.95] | 76.08 [75.29, 76.47] | 69.78 [69.19, 70.27] | 2.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 67.05 | 69.41 | 68.21 | 71.76 | 70.52 | 2.00 | 0.00 | 1.84 [1.81, 1.88] | 6.00 |
| corroborated--combined--chronological--budget-10 | 68.34 [67.42, 69.41] | 70.20 [69.41, 70.59] | 69.25 [68.97, 69.41] | 72.55 [71.76, 72.94] | 71.57 [71.26, 71.76] | 2.00 | 0.00 | 3.16 [3.10, 3.27] | 10.67 [10.00, 11.00] |
| corroborated--combined--chronological--budget-20 | 69.38 [68.97, 69.77] | 70.20 [69.41, 70.59] | 69.78 [69.41, 70.18] | 72.55 [71.76, 72.94] | 72.12 [71.76, 72.51] | 2.00 | 0.00 | 4.18 [3.55, 4.61] | 15.33 [13.00, 18.00] |
| corroborated--combined--chronological--budget-40 | 69.38 [68.97, 69.77] | 70.20 [69.41, 70.59] | 69.78 [69.41, 70.18] | 72.55 [71.76, 72.94] | 72.12 [71.76, 72.51] | 2.00 | 0.00 | 4.18 [3.55, 4.61] | 15.33 [13.00, 18.00] |
| corroborated--combined--evidence--budget-05 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 1.79 [1.75, 1.86] | 5.67 [5.00, 6.00] |
| corroborated--combined--evidence--budget-10 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.02 [2.97, 3.10] | 11.67 [11.00, 13.00] |
| corroborated--combined--evidence--budget-20 | 69.38 [68.97, 69.77] | 70.20 [69.41, 70.59] | 69.78 [69.41, 70.18] | 72.55 [71.76, 72.94] | 72.12 [71.76, 72.51] | 2.00 | 0.00 | 4.18 [3.55, 4.61] | 15.33 [13.00, 18.00] |
| corroborated--combined--evidence--budget-40 | 69.38 [68.97, 69.77] | 70.20 [69.41, 70.59] | 69.78 [69.41, 70.18] | 72.55 [71.76, 72.94] | 72.12 [71.76, 72.51] | 2.00 | 0.00 | 4.18 [3.55, 4.61] | 15.33 [13.00, 18.00] |
| corroborated--split_only--chronological--budget-05 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-10 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-20 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-40 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-05 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-10 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-20 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-40 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| event_starts--combined--chronological--budget-05 | 66.05 [65.56, 67.05] | 69.41 | 67.69 [67.43, 68.21] | 71.76 | 69.98 [69.71, 70.52] | 2.00 | 0.00 | 1.85 [1.84, 1.85] | 6.67 [6.00, 7.00] |
| event_starts--combined--chronological--budget-10 | 67.80 [67.42, 68.18] | 70.20 [69.41, 70.59] | 68.98 [68.60, 69.36] | 72.55 [71.76, 72.94] | 71.29 [70.93, 71.68] | 2.00 | 0.00 | 3.71 [3.68, 3.75] | 12.33 [12.00, 13.00] |
| event_starts--combined--chronological--budget-20 | 68.60 [67.42, 69.41] | 70.20 [69.41, 70.59] | 69.38 [68.97, 69.77] | 72.55 [71.76, 72.94] | 71.71 [71.26, 72.09] | 2.00 | 0.00 | 5.91 [5.75, 6.21] | 20.00 [19.00, 21.00] |
| event_starts--combined--chronological--budget-40 | 69.38 [68.97, 69.77] | 70.20 [69.41, 70.59] | 69.78 [69.41, 70.18] | 72.55 [71.76, 72.94] | 72.12 [71.76, 72.51] | 2.00 | 0.00 | 6.48 [6.18, 7.04] | 22.33 [21.00, 25.00] |
| event_starts--combined--evidence--budget-05 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 1.79 [1.75, 1.86] | 5.67 [5.00, 6.00] |
| event_starts--combined--evidence--budget-10 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.59 [3.55, 3.62] | 14.00 [13.00, 15.00] |
| event_starts--combined--evidence--budget-20 | 69.26 [68.60, 69.77] | 69.80 [69.41, 70.59] | 69.53 [69.01, 70.18] | 72.16 [71.76, 72.94] | 71.87 [71.35, 72.51] | 2.00 | 0.00 | 5.86 [5.60, 6.21] | 20.67 [20.00, 21.00] |
| event_starts--combined--evidence--budget-40 | 69.38 [68.97, 69.77] | 70.20 [69.41, 70.59] | 69.78 [69.41, 70.18] | 72.55 [71.76, 72.94] | 72.12 [71.76, 72.51] | 2.00 | 0.00 | 6.48 [6.18, 7.04] | 22.33 [21.00, 25.00] |
| event_starts--split_only--chronological--budget-05 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 1.61 [1.43, 1.90] | 5.33 [5.00, 6.00] |
| event_starts--split_only--chronological--budget-10 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 2.43 [2.23, 2.66] | 7.67 [7.00, 8.00] |
| event_starts--split_only--chronological--budget-20 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 2.69 [2.40, 3.01] | 8.33 [8.00, 9.00] |
| event_starts--split_only--chronological--budget-40 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 2.69 [2.40, 3.01] | 8.33 [8.00, 9.00] |
| event_starts--split_only--evidence--budget-05 | 62.11 | 69.41 | 65.56 | 71.76 | 67.78 | 2.00 | 0.00 | 1.53 [1.28, 1.75] | 5.00 [4.00, 6.00] |
| event_starts--split_only--evidence--budget-10 | 62.24 [62.11, 62.50] | 69.80 [69.41, 70.59] | 65.80 [65.56, 66.30] | 72.16 [71.76, 72.94] | 68.02 [67.78, 68.51] | 2.00 | 0.00 | 2.49 [2.40, 2.66] | 7.67 [7.00, 8.00] |
| event_starts--split_only--evidence--budget-20 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 2.69 [2.40, 3.01] | 8.33 [8.00, 9.00] |
| event_starts--split_only--evidence--budget-40 | 62.37 [62.11, 62.50] | 70.20 [69.41, 70.59] | 66.05 [65.56, 66.30] | 72.55 [71.76, 72.94] | 68.26 [67.78, 68.51] | 2.00 | 0.00 | 2.69 [2.40, 3.01] | 8.33 [8.00, 9.00] |
| head_evidence--combined--chronological--budget-05 | 65.93 [65.56, 66.67] | 69.80 [69.41, 70.59] | 67.81 [67.43, 68.57] | 72.16 [71.76, 72.94] | 70.10 [69.71, 70.86] | 2.00 | 0.00 | 1.87 [1.81, 1.94] | 6.33 [6.00, 7.00] |
| head_evidence--combined--chronological--budget-10 | 68.45 [67.42, 68.97] | 70.59 | 69.50 [68.97, 69.77] | 72.94 | 71.82 [71.26, 72.09] | 2.00 | 0.00 | 3.48 [3.41, 3.59] | 12.00 [11.00, 13.00] |
| head_evidence--combined--chronological--budget-20 | 69.50 [68.97, 69.77] | 70.59 | 70.04 [69.77, 70.18] | 72.94 | 72.37 [72.09, 72.51] | 2.00 | 0.00 | 4.79 [4.26, 5.09] | 17.67 [16.00, 20.00] |
| head_evidence--combined--chronological--budget-40 | 69.50 [68.97, 69.77] | 70.59 | 70.04 [69.77, 70.18] | 72.94 | 72.37 [72.09, 72.51] | 2.00 | 0.00 | 4.86 [4.26, 5.22] | 18.00 [16.00, 20.00] |
| head_evidence--combined--evidence--budget-05 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 1.79 [1.75, 1.86] | 5.67 [5.00, 6.00] |
| head_evidence--combined--evidence--budget-10 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.43 [3.23, 3.57] | 13.33 [12.00, 15.00] |
| head_evidence--combined--evidence--budget-20 | 69.50 [68.97, 69.77] | 70.59 | 70.04 [69.77, 70.18] | 72.94 | 72.37 [72.09, 72.51] | 2.00 | 0.00 | 4.73 [4.26, 5.09] | 17.67 [16.00, 20.00] |
| head_evidence--combined--evidence--budget-40 | 69.50 [68.97, 69.77] | 70.59 | 70.04 [69.77, 70.18] | 72.94 | 72.37 [72.09, 72.51] | 2.00 | 0.00 | 4.86 [4.26, 5.22] | 18.00 [16.00, 20.00] |
| head_evidence--split_only--chronological--budget-05 | 62.24 [62.11, 62.50] | 69.80 [69.41, 70.59] | 65.80 [65.56, 66.30] | 72.16 [71.76, 72.94] | 68.02 [67.78, 68.51] | 2.00 | 0.00 | 0.97 [0.86, 1.07] | 3.67 [3.00, 4.00] |
| head_evidence--split_only--chronological--budget-10 | 62.50 | 70.59 | 66.30 | 72.94 | 68.51 | 2.00 | 0.00 | 1.65 [1.46, 1.83] | 6.00 |
| head_evidence--split_only--chronological--budget-20 | 62.50 | 70.59 | 66.30 | 72.94 | 68.51 | 2.00 | 0.00 | 1.65 [1.46, 1.83] | 6.00 |
| head_evidence--split_only--chronological--budget-40 | 62.50 | 70.59 | 66.30 | 72.94 | 68.51 | 2.00 | 0.00 | 1.65 [1.46, 1.83] | 6.00 |
| head_evidence--split_only--evidence--budget-05 | 62.11 | 69.41 | 65.56 | 71.76 | 67.78 | 2.00 | 0.00 | 0.94 [0.86, 1.07] | 3.67 [3.00, 4.00] |
| head_evidence--split_only--evidence--budget-10 | 62.50 | 70.59 | 66.30 | 72.94 | 68.51 | 2.00 | 0.00 | 1.65 [1.46, 1.83] | 6.00 |
| head_evidence--split_only--evidence--budget-20 | 62.50 | 70.59 | 66.30 | 72.94 | 68.51 | 2.00 | 0.00 | 1.65 [1.46, 1.83] | 6.00 |
| head_evidence--split_only--evidence--budget-40 | 62.50 | 70.59 | 66.30 | 72.94 | 68.51 | 2.00 | 0.00 | 1.65 [1.46, 1.83] | 6.00 |
| none--cleanup_only--chronological--budget-05 | 67.05 | 69.41 | 68.21 | 71.76 | 70.52 | 2.00 | 0.00 | 1.84 [1.81, 1.88] | 6.00 |
| none--cleanup_only--chronological--budget-10 | 68.35 [67.05, 69.41] | 69.41 | 68.88 [68.21, 69.41] | 71.76 | 71.21 [70.52, 71.76] | 2.00 | 0.00 | 3.19 [3.14, 3.27] | 11.00 [10.00, 12.00] |
| none--cleanup_only--chronological--budget-20 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.79 [3.55, 4.03] | 14.00 [13.00, 16.00] |
| none--cleanup_only--chronological--budget-40 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.79 [3.55, 4.03] | 14.00 [13.00, 16.00] |
| none--cleanup_only--evidence--budget-05 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 1.79 [1.75, 1.86] | 5.67 [5.00, 6.00] |
| none--cleanup_only--evidence--budget-10 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.02 [2.97, 3.10] | 11.67 [11.00, 13.00] |
| none--cleanup_only--evidence--budget-20 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.79 [3.55, 4.03] | 14.00 [13.00, 16.00] |
| none--cleanup_only--evidence--budget-40 | 69.14 [68.60, 69.41] | 69.41 | 69.28 [69.01, 69.41] | 71.76 | 71.62 [71.35, 71.76] | 2.00 | 0.00 | 3.79 [3.55, 4.03] | 14.00 [13.00, 16.00] |

### source-group-007

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 50.00 | 58.88 | 54.08 | 57.94 | 53.45 | 1.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 49.36 [49.23, 49.61] | 60.12 [59.81, 60.75] | 54.21 [54.01, 54.39] | 60.44 [59.81, 60.75] | 54.72 [54.47, 55.08] | 1.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 46.96 [45.99, 48.51] | 59.81 [58.88, 60.75] | 52.61 [51.64, 53.94] | 60.75 | 53.65 [53.28, 54.17] | 1.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 45.72 [44.44, 48.20] | 62.31 [60.75, 63.55] | 52.72 [51.38, 54.47] | 67.91 [66.36, 71.03] | 57.67 [56.35, 58.69] | 1.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 52.21 [52.07, 52.50] | 58.88 | 55.34 [55.26, 55.51] | 57.94 | 54.47 [54.39, 54.63] | 1.00 | 0.00 | 2.20 [2.18, 2.23] | 8.00 |
| corroborated--combined--chronological--budget-10 | 55.43 [53.78, 56.76] | 59.19 [58.88, 59.81] | 57.24 [56.64, 57.80] | 58.26 [57.94, 58.88] | 56.33 [55.75, 56.88] | 1.00 | 0.00 | 4.36 [4.31, 4.42] | 14.33 [13.00, 15.00] |
| corroborated--combined--chronological--budget-20 | 58.83 [58.18, 59.43] | 59.19 [58.88, 59.81] | 59.01 [58.88, 59.15] | 59.19 [58.88, 59.81] | 59.01 [58.88, 59.15] | 1.00 | 0.00 | 7.56 [6.79, 8.11] | 23.67 [20.00, 26.00] |
| corroborated--combined--chronological--budget-40 | 60.26 [59.43, 61.17] | 59.50 [58.88, 60.75] | 59.87 [59.15, 60.47] | 59.50 [58.88, 60.75] | 59.87 [59.15, 60.47] | 1.00 | 0.00 | 9.67 [7.88, 11.25] | 28.00 [22.00, 33.00] |
| corroborated--combined--evidence--budget-05 | 55.75 | 58.88 | 57.27 | 57.94 | 56.36 | 1.00 | 0.00 | 2.15 [2.14, 2.15] | 7.00 |
| corroborated--combined--evidence--budget-10 | 58.88 [58.33, 59.43] | 58.88 | 58.88 [58.60, 59.15] | 57.94 | 57.94 [57.67, 58.22] | 1.00 | 0.00 | 4.30 [4.26, 4.37] | 15.00 [13.00, 16.00] |
| corroborated--combined--evidence--budget-20 | 60.32 [59.43, 60.95] | 59.19 [58.88, 59.81] | 59.75 [59.15, 60.38] | 58.88 | 59.43 [59.15, 59.72] | 1.00 | 0.00 | 7.30 [6.49, 7.95] | 23.67 [20.00, 26.00] |
| corroborated--combined--evidence--budget-40 | 60.45 [59.43, 61.17] | 59.50 [58.88, 60.75] | 59.97 [59.15, 60.75] | 59.50 [58.88, 60.75] | 59.97 [59.15, 60.75] | 1.00 | 0.00 | 9.51 [7.88, 10.78] | 28.33 [22.00, 34.00] |
| corroborated--split_only--chronological--budget-05 | 50.13 [50.00, 50.39] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 58.26 [57.94, 58.88] | 53.66 [53.45, 54.08] | 1.00 | 0.00 | 0.77 [0.34, 1.55] | 2.33 [1.00, 5.00] |
| corroborated--split_only--chronological--budget-10 | 49.74 [49.61, 50.00] | 59.19 [58.88, 59.81] | 54.05 [53.85, 54.47] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 1.00 | 0.00 | 1.69 [1.18, 2.31] | 5.00 [3.00, 7.00] |
| corroborated--split_only--chronological--budget-20 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 2.62 [2.06, 3.34] | 7.00 [4.00, 11.00] |
| corroborated--split_only--chronological--budget-40 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 2.62 [2.06, 3.34] | 7.00 [4.00, 11.00] |
| corroborated--split_only--evidence--budget-05 | 50.13 [50.00, 50.39] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 58.26 [57.94, 58.88] | 53.66 [53.45, 54.08] | 1.00 | 0.00 | 0.77 [0.34, 1.55] | 2.33 [1.00, 5.00] |
| corroborated--split_only--evidence--budget-10 | 50.00 [49.61, 50.78] | 59.50 [58.88, 60.75] | 54.34 [53.85, 55.32] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 1.00 | 0.00 | 1.64 [1.18, 2.15] | 5.00 [3.00, 7.00] |
| corroborated--split_only--evidence--budget-20 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 2.62 [2.06, 3.34] | 7.00 [4.00, 11.00] |
| corroborated--split_only--evidence--budget-40 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 2.62 [2.06, 3.34] | 7.00 [4.00, 11.00] |
| event_starts--combined--chronological--budget-05 | 52.21 [52.07, 52.50] | 58.88 | 55.34 [55.26, 55.51] | 57.94 | 54.47 [54.39, 54.63] | 1.00 | 0.00 | 2.20 [2.18, 2.23] | 8.00 |
| event_starts--combined--chronological--budget-10 | 53.84 [52.50, 54.78] | 59.19 [58.88, 59.81] | 56.38 [55.51, 56.89] | 58.26 [57.94, 58.88] | 55.49 [54.63, 56.00] | 1.00 | 0.00 | 4.42 [4.38, 4.47] | 15.33 [14.00, 17.00] |
| event_starts--combined--chronological--budget-20 | 57.06 [56.76, 57.27] | 59.19 [58.88, 59.81] | 58.10 [57.80, 58.45] | 59.19 [58.88, 59.81] | 58.10 [57.80, 58.45] | 1.00 | 0.00 | 8.20 [7.87, 8.42] | 26.00 [24.00, 28.00] |
| event_starts--combined--chronological--budget-40 | 60.39 [59.43, 61.54] | 59.81 [58.88, 60.75] | 60.09 [59.15, 60.66] | 59.81 [58.88, 60.75] | 60.09 [59.15, 60.66] | 1.00 | 0.00 | 12.00 [10.57, 12.97] | 35.33 [30.00, 38.00] |
| event_starts--combined--evidence--budget-05 | 55.75 | 58.88 | 57.27 | 57.94 | 56.36 | 1.00 | 0.00 | 2.15 [2.14, 2.15] | 7.00 |
| event_starts--combined--evidence--budget-10 | 58.88 [58.33, 59.43] | 58.88 | 58.88 [58.60, 59.15] | 57.94 | 57.94 [57.67, 58.22] | 1.00 | 0.00 | 4.30 [4.26, 4.37] | 15.00 [13.00, 16.00] |
| event_starts--combined--evidence--budget-20 | 60.58 [60.00, 61.17] | 58.88 | 59.72 [59.43, 60.00] | 57.94 | 58.77 [58.49, 59.05] | 1.00 | 0.00 | 7.93 [7.33, 8.25] | 26.67 [23.00, 29.00] |
| event_starts--combined--evidence--budget-40 | 60.57 [59.43, 61.54] | 59.81 [58.88, 60.75] | 60.19 [59.15, 60.75] | 59.81 [58.88, 60.75] | 60.19 [59.15, 60.75] | 1.00 | 0.00 | 11.85 [10.57, 12.50] | 35.67 [30.00, 39.00] |
| event_starts--split_only--chronological--budget-05 | 50.13 [50.00, 50.39] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 58.26 [57.94, 58.88] | 53.66 [53.45, 54.08] | 1.00 | 0.00 | 1.65 [1.55, 1.78] | 5.33 [5.00, 6.00] |
| event_starts--split_only--chronological--budget-10 | 49.74 [49.61, 50.00] | 59.19 [58.88, 59.81] | 54.05 [53.85, 54.47] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 1.00 | 0.00 | 3.24 [2.73, 3.67] | 9.67 [8.00, 11.00] |
| event_starts--split_only--chronological--budget-20 | 49.74 [49.61, 50.00] | 59.19 [58.88, 59.81] | 54.05 [53.85, 54.47] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 1.00 | 0.00 | 4.64 [4.04, 5.34] | 12.67 [11.00, 15.00] |
| event_starts--split_only--chronological--budget-40 | 50.00 [49.61, 50.39] | 59.81 [58.88, 60.75] | 54.47 [53.85, 55.08] | 59.81 [58.88, 60.75] | 54.70 [54.08, 55.32] | 1.00 | 0.00 | 5.24 [4.64, 5.94] | 14.67 [13.00, 17.00] |
| event_starts--split_only--evidence--budget-05 | 50.00 | 58.88 | 54.08 | 57.94 | 53.45 | 1.00 | 0.00 | 1.45 [1.35, 1.52] | 5.33 [5.00, 6.00] |
| event_starts--split_only--evidence--budget-10 | 50.13 [50.00, 50.39] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 58.26 [57.94, 58.88] | 53.66 [53.45, 54.08] | 1.00 | 0.00 | 2.91 [2.40, 3.51] | 9.67 [8.00, 11.00] |
| event_starts--split_only--evidence--budget-20 | 50.00 [49.61, 50.39] | 59.81 [58.88, 60.75] | 54.47 [53.85, 55.08] | 59.81 [58.88, 60.75] | 54.70 [54.08, 55.32] | 1.00 | 0.00 | 4.37 [3.76, 5.07] | 13.67 [12.00, 16.00] |
| event_starts--split_only--evidence--budget-40 | 50.00 [49.61, 50.39] | 59.81 [58.88, 60.75] | 54.47 [53.85, 55.08] | 59.81 [58.88, 60.75] | 54.70 [54.08, 55.32] | 1.00 | 0.00 | 5.24 [4.64, 5.94] | 14.67 [13.00, 17.00] |
| head_evidence--combined--chronological--budget-05 | 52.21 [52.07, 52.50] | 58.88 | 55.34 [55.26, 55.51] | 57.94 | 54.47 [54.39, 54.63] | 1.00 | 0.00 | 2.21 [2.18, 2.23] | 7.67 [7.00, 8.00] |
| head_evidence--combined--chronological--budget-10 | 54.00 [52.89, 54.78] | 59.19 [58.88, 59.81] | 56.47 [56.14, 56.76] | 58.26 [57.94, 58.88] | 55.57 [55.26, 55.86] | 1.00 | 0.00 | 4.37 [4.36, 4.39] | 14.33 [13.00, 16.00] |
| head_evidence--combined--chronological--budget-20 | 57.08 [55.65, 57.80] | 59.19 [58.88, 59.81] | 58.11 [57.66, 58.33] | 59.19 [58.88, 59.81] | 58.11 [57.66, 58.33] | 1.00 | 0.00 | 8.43 [8.13, 8.86] | 27.67 [25.00, 31.00] |
| head_evidence--combined--chronological--budget-40 | 60.08 [59.43, 61.17] | 59.50 [58.88, 60.75] | 59.78 [59.15, 60.19] | 59.50 [58.88, 60.75] | 59.78 [59.15, 60.19] | 1.00 | 0.00 | 11.83 [10.52, 13.14] | 36.00 [32.00, 41.00] |
| head_evidence--combined--evidence--budget-05 | 55.75 | 58.88 | 57.27 | 57.94 | 56.36 | 1.00 | 0.00 | 2.15 [2.14, 2.15] | 7.00 |
| head_evidence--combined--evidence--budget-10 | 58.88 [58.33, 59.43] | 58.88 | 58.88 [58.60, 59.15] | 57.94 | 57.94 [57.67, 58.22] | 1.00 | 0.00 | 4.27 [4.18, 4.37] | 15.00 [13.00, 16.00] |
| head_evidence--combined--evidence--budget-20 | 60.38 [60.00, 60.58] | 58.88 | 59.62 [59.43, 59.72] | 58.26 [57.94, 58.88] | 58.99 [58.49, 59.72] | 1.00 | 0.00 | 8.34 [8.08, 8.60] | 28.33 [27.00, 30.00] |
| head_evidence--combined--evidence--budget-40 | 60.45 [59.43, 61.17] | 59.50 [58.88, 60.75] | 59.97 [59.15, 60.75] | 59.50 [58.88, 60.75] | 59.97 [59.15, 60.75] | 1.00 | 0.00 | 11.71 [10.52, 12.78] | 36.33 [32.00, 42.00] |
| head_evidence--split_only--chronological--budget-05 | 50.13 [50.00, 50.39] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 58.26 [57.94, 58.88] | 53.66 [53.45, 54.08] | 1.00 | 0.00 | 1.75 [1.54, 2.08] | 6.67 [6.00, 8.00] |
| head_evidence--split_only--chronological--budget-10 | 49.74 [49.61, 50.00] | 59.19 [58.88, 59.81] | 54.05 [53.85, 54.47] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 1.00 | 0.00 | 3.75 [3.10, 4.37] | 12.67 [10.00, 15.00] |
| head_evidence--split_only--chronological--budget-20 | 49.74 [49.61, 50.00] | 59.19 [58.88, 59.81] | 54.05 [53.85, 54.47] | 59.19 [58.88, 59.81] | 54.29 [54.08, 54.70] | 1.00 | 0.00 | 5.56 [4.71, 6.85] | 16.67 [13.00, 21.00] |
| head_evidence--split_only--chronological--budget-40 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 5.98 [4.71, 8.13] | 18.33 [13.00, 26.00] |
| head_evidence--split_only--evidence--budget-05 | 50.00 | 58.88 | 54.08 | 57.94 | 53.45 | 1.00 | 0.00 | 1.66 [1.57, 1.83] | 6.67 [6.00, 7.00] |
| head_evidence--split_only--evidence--budget-10 | 49.87 [49.61, 50.39] | 59.19 [58.88, 59.81] | 54.13 [53.85, 54.70] | 58.88 | 54.08 | 1.00 | 0.00 | 3.54 [2.89, 4.09] | 13.00 [10.00, 16.00] |
| head_evidence--split_only--evidence--budget-20 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 5.55 [4.71, 6.83] | 17.33 [13.00, 23.00] |
| head_evidence--split_only--evidence--budget-40 | 49.87 [49.61, 50.39] | 59.50 [58.88, 60.75] | 54.26 [53.85, 55.08] | 59.50 [58.88, 60.75] | 54.49 [54.08, 55.32] | 1.00 | 0.00 | 5.98 [4.71, 8.13] | 18.33 [13.00, 26.00] |
| none--cleanup_only--chronological--budget-05 | 52.21 [52.07, 52.50] | 58.88 | 55.34 [55.26, 55.51] | 57.94 | 54.47 [54.39, 54.63] | 1.00 | 0.00 | 2.20 [2.18, 2.23] | 8.00 |
| none--cleanup_only--chronological--budget-10 | 56.09 [55.75, 56.76] | 58.88 | 57.45 [57.27, 57.80] | 57.94 | 56.54 [56.36, 56.88] | 1.00 | 0.00 | 4.24 [3.97, 4.40] | 13.67 [12.00, 15.00] |
| none--cleanup_only--chronological--budget-20 | 60.19 [60.00, 60.58] | 58.88 | 59.53 [59.43, 59.72] | 57.94 | 58.58 [58.49, 58.77] | 1.00 | 0.00 | 6.70 [5.41, 7.52] | 20.67 [16.00, 24.00] |
| none--cleanup_only--chronological--budget-40 | 60.78 [60.00, 61.76] | 58.88 | 59.81 [59.43, 60.29] | 57.94 | 58.86 [58.49, 59.33] | 1.00 | 0.00 | 7.43 [5.41, 9.06] | 22.33 [16.00, 26.00] |
| none--cleanup_only--evidence--budget-05 | 55.75 | 58.88 | 57.27 | 57.94 | 56.36 | 1.00 | 0.00 | 2.15 [2.14, 2.15] | 7.00 |
| none--cleanup_only--evidence--budget-10 | 58.88 [58.33, 59.43] | 58.88 | 58.88 [58.60, 59.15] | 57.94 | 57.94 [57.67, 58.22] | 1.00 | 0.00 | 4.19 [3.92, 4.37] | 14.67 [12.00, 16.00] |
| none--cleanup_only--evidence--budget-20 | 60.58 [60.00, 61.17] | 58.88 | 59.72 [59.43, 60.00] | 57.94 | 58.77 [58.49, 59.05] | 1.00 | 0.00 | 6.69 [5.41, 7.44] | 21.00 [16.00, 24.00] |
| none--cleanup_only--evidence--budget-40 | 60.78 [60.00, 61.76] | 58.88 | 59.81 [59.43, 60.29] | 57.94 | 58.86 [58.49, 59.33] | 1.00 | 0.00 | 7.43 [5.41, 9.06] | 22.33 [16.00, 26.00] |

### source-group-009

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 85.30 [83.75, 86.08] | 91.44 [90.54, 91.89] | 88.26 [87.01, 88.89] | 87.84 | 84.78 [84.42, 84.97] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 82.76 [79.52, 85.00] | 90.54 [89.19, 91.89] | 86.47 [84.08, 88.31] | 87.84 | 83.88 [82.80, 84.42] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 81.72 [77.38, 86.08] | 90.09 [87.84, 91.89] | 85.69 [82.28, 88.89] | 88.74 [87.84, 89.19] | 84.38 [83.54, 84.97] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--chronological--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--chronological--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--chronological--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--evidence--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--evidence--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--evidence--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--combined--evidence--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.37 [0.20, 0.69] | 0.67 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-05 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-10 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-20 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-40 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-05 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-10 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-20 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-40 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.63 [0.57, 0.69] | 1.33 [1.00, 2.00] |
| event_starts--combined--chronological--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.76 [0.57, 1.10] | 1.67 [1.00, 3.00] |
| event_starts--combined--chronological--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.76 [0.57, 1.10] | 1.67 [1.00, 3.00] |
| event_starts--combined--chronological--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.76 [0.57, 1.10] | 1.67 [1.00, 3.00] |
| event_starts--combined--evidence--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.63 [0.57, 0.69] | 1.33 [1.00, 2.00] |
| event_starts--combined--evidence--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.76 [0.57, 1.10] | 1.67 [1.00, 3.00] |
| event_starts--combined--evidence--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.76 [0.57, 1.10] | 1.67 [1.00, 3.00] |
| event_starts--combined--evidence--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.76 [0.57, 1.10] | 1.67 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-05 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-10 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-20 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-40 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-05 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-10 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-20 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-40 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| head_evidence--combined--chronological--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.83 [0.20, 1.15] | 2.33 [0.00, 4.00] |
| head_evidence--combined--chronological--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.95 [0.20, 1.51] | 2.67 [0.00, 5.00] |
| head_evidence--combined--chronological--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.95 [0.20, 1.51] | 2.67 [0.00, 5.00] |
| head_evidence--combined--chronological--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.95 [0.20, 1.51] | 2.67 [0.00, 5.00] |
| head_evidence--combined--evidence--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.84 [0.20, 1.16] | 2.33 [0.00, 4.00] |
| head_evidence--combined--evidence--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.95 [0.20, 1.51] | 2.67 [0.00, 5.00] |
| head_evidence--combined--evidence--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.95 [0.20, 1.51] | 2.67 [0.00, 5.00] |
| head_evidence--combined--evidence--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.95 [0.20, 1.51] | 2.67 [0.00, 5.00] |
| head_evidence--split_only--chronological--budget-05 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--chronological--budget-10 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--chronological--budget-20 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--chronological--budget-40 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--evidence--budget-05 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--evidence--budget-10 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--evidence--budget-20 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| head_evidence--split_only--evidence--budget-40 | 86.08 | 91.89 | 88.89 | 87.84 | 84.97 | 0.00 | 0.00 | 0.71 [0.00, 1.18] | 2.33 [0.00, 4.00] |
| none--cleanup_only--chronological--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--chronological--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--chronological--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--chronological--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-05 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-10 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-20 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-40 | 88.31 | 91.89 | 90.07 | 87.84 | 86.09 | 0.00 | 0.00 | 0.24 [0.20, 0.32] | 0.33 [0.00, 1.00] |

### source-group-012

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 59.58 [59.38, 60.00] | 68.45 [67.86, 69.64] | 63.71 [63.33, 64.46] | 77.38 [76.79, 78.57] | 72.02 [71.67, 72.73] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 58.09 [56.72, 59.09] | 68.45 [67.86, 69.64] | 62.84 [61.79, 63.93] | 77.38 [76.79, 78.57] | 71.04 [69.92, 72.13] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 56.20 [55.56, 56.52] | 70.24 [69.64, 71.43] | 62.43 [62.40, 62.50] | 84.52 [83.93, 85.71] | 75.13 [75.00, 75.20] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--chronological--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--split_only--chronological--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-05 | 61.96 [61.29, 62.30] | 67.86 | 64.77 [64.41, 64.96] | 76.79 | 73.30 [72.88, 73.50] | 0.00 | 0.00 | 0.97 [0.86, 1.04] | 3.33 [3.00, 4.00] |
| event_starts--combined--chronological--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.90 [0.79, 1.04] | 3.00 |
| event_starts--combined--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--split_only--chronological--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| head_evidence--combined--chronological--budget-05 | 61.63 [61.29, 62.30] | 67.86 | 64.59 [64.41, 64.96] | 76.79 | 73.09 [72.88, 73.50] | 0.00 | 0.00 | 1.07 [1.05, 1.11] | 4.00 |
| head_evidence--combined--chronological--budget-10 | 61.96 [61.29, 62.30] | 67.86 | 64.77 [64.41, 64.96] | 76.79 | 73.30 [72.88, 73.50] | 0.00 | 0.00 | 2.01 [1.93, 2.08] | 6.33 [6.00, 7.00] |
| head_evidence--combined--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--combined--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--combined--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.05 [0.96, 1.09] | 3.67 [3.00, 4.00] |
| head_evidence--combined--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.87 [1.69, 1.99] | 6.00 [5.00, 7.00] |
| head_evidence--combined--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--combined--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--split_only--chronological--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.94 [0.78, 1.08] | 3.33 [3.00, 4.00] |
| head_evidence--split_only--chronological--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.82 [1.70, 2.00] | 5.33 [5.00, 6.00] |
| head_evidence--split_only--chronological--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| head_evidence--split_only--chronological--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| head_evidence--split_only--evidence--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.98 [0.84, 1.08] | 3.67 [3.00, 4.00] |
| head_evidence--split_only--evidence--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.82 [1.70, 2.01] | 5.33 [5.00, 6.00] |
| head_evidence--split_only--evidence--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| head_evidence--split_only--evidence--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| none--cleanup_only--chronological--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |

## Recording sensitivity

The same automatic and restricted-review comparisons are retained per recording; these are diagnostic slices, not extra independent test sets.

### grass-source-03

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 41.67 | 48.39 | 44.78 | 64.52 | 60.61 | 1.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 40.48 [39.47, 42.50] | 50.54 [48.39, 54.84] | 44.95 [43.48, 47.89] | 72.04 [70.97, 74.19] | 65.04 [64.71, 65.71] | 1.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 39.02 [36.59, 41.46] | 51.61 [48.39, 54.84] | 44.44 [41.67, 47.22] | 73.12 [70.97, 74.19] | 63.85 [61.97, 64.79] | 1.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 34.94 [32.56, 36.36] | 47.31 [45.16, 51.61] | 40.17 [37.84, 42.67] | 74.19 [70.97, 77.42] | 63.88 [63.01, 64.86] | 1.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 44.12 | 48.39 | 46.15 | 64.52 | 61.54 | 1.00 | 0.00 | 0.65 | 1.00 |
| corroborated--combined--chronological--budget-10 | 49.02 [47.06, 50.00] | 49.46 [48.39, 51.61] | 49.20 [49.18, 49.23] | 65.59 [64.52, 67.74] | 65.25 [64.62, 65.57] | 1.00 | 0.00 | 1.30 [1.30, 1.30] | 1.67 [1.00, 3.00] |
| corroborated--combined--chronological--budget-20 | 52.38 [50.00, 53.57] | 49.46 [48.39, 51.61] | 50.83 [50.79, 50.85] | 68.82 [67.74, 70.97] | 70.74 [69.84, 71.19] | 1.00 | 0.00 | 2.70 [2.68, 2.71] | 5.33 [5.00, 6.00] |
| corroborated--combined--chronological--budget-40 | 55.93 [55.56, 56.67] | 50.54 [48.39, 54.84] | 53.06 [51.72, 55.74] | 69.89 [67.74, 74.19] | 73.41 [72.41, 75.41] | 1.00 | 0.00 | 4.30 [3.68, 5.42] | 8.67 [6.00, 13.00] |
| corroborated--combined--evidence--budget-05 | 48.92 [48.39, 50.00] | 48.39 | 48.65 [48.39, 49.18] | 64.52 | 64.87 [64.52, 65.57] | 1.00 | 0.00 | 0.66 [0.66, 0.67] | 1.67 [1.00, 2.00] |
| corroborated--combined--evidence--budget-10 | 54.89 [53.57, 55.56] | 48.39 | 51.43 [50.85, 51.72] | 64.52 | 68.58 [67.80, 68.97] | 1.00 | 0.00 | 1.25 [1.23, 1.29] | 2.33 [2.00, 3.00] |
| corroborated--combined--evidence--budget-20 | 56.79 [55.56, 59.26] | 49.46 [48.39, 51.61] | 52.87 [51.72, 55.17] | 67.74 | 72.41 | 1.00 | 0.00 | 2.45 [2.29, 2.64] | 5.33 [4.00, 7.00] |
| corroborated--combined--evidence--budget-40 | 56.58 [55.56, 58.62] | 50.54 [48.39, 54.84] | 53.37 [51.72, 56.67] | 69.89 [67.74, 74.19] | 73.83 [72.41, 76.67] | 1.00 | 0.00 | 4.14 [3.68, 4.95] | 9.00 [6.00, 14.00] |
| corroborated--split_only--chronological--budget-05 | 42.19 [41.67, 43.24] | 49.46 [48.39, 51.61] | 45.54 [44.78, 47.06] | 65.59 [64.52, 67.74] | 61.30 [60.61, 62.69] | 1.00 | 0.00 | 0.18 [0.00, 0.54] | 0.67 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-10 | 41.06 [40.54, 42.11] | 49.46 [48.39, 51.61] | 44.87 [44.12, 46.38] | 68.82 [67.74, 70.97] | 63.36 [62.69, 64.71] | 1.00 | 0.00 | 0.94 [0.76, 1.30] | 2.67 [2.00, 4.00] |
| corroborated--split_only--chronological--budget-20 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 1.86 [1.63, 2.33] | 4.67 [3.00, 8.00] |
| corroborated--split_only--chronological--budget-40 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 1.86 [1.63, 2.33] | 4.67 [3.00, 8.00] |
| corroborated--split_only--evidence--budget-05 | 42.19 [41.67, 43.24] | 49.46 [48.39, 51.61] | 45.54 [44.78, 47.06] | 65.59 [64.52, 67.74] | 61.30 [60.61, 62.69] | 1.00 | 0.00 | 0.18 [0.00, 0.54] | 0.67 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-10 | 41.94 [40.54, 44.74] | 50.54 [48.39, 54.84] | 45.84 [44.12, 49.28] | 68.82 [67.74, 70.97] | 63.36 [62.69, 64.71] | 1.00 | 0.00 | 0.89 [0.76, 1.14] | 2.67 [2.00, 4.00] |
| corroborated--split_only--evidence--budget-20 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 1.86 [1.63, 2.33] | 4.67 [3.00, 8.00] |
| corroborated--split_only--evidence--budget-40 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 1.86 [1.63, 2.33] | 4.67 [3.00, 8.00] |
| event_starts--combined--chronological--budget-05 | 44.12 | 48.39 | 46.15 | 64.52 | 61.54 | 1.00 | 0.00 | 0.65 | 1.00 |
| event_starts--combined--chronological--budget-10 | 45.99 [45.45, 47.06] | 49.46 [48.39, 51.61] | 47.66 [46.88, 49.23] | 65.59 [64.52, 67.74] | 63.21 [62.50, 64.62] | 1.00 | 0.00 | 1.30 | 3.00 |
| event_starts--combined--chronological--budget-20 | 48.92 [48.39, 50.00] | 49.46 [48.39, 51.61] | 49.19 [48.39, 50.79] | 68.82 [67.74, 70.97] | 68.44 [67.74, 69.84] | 1.00 | 0.00 | 2.67 [2.60, 2.71] | 5.67 [5.00, 6.00] |
| event_starts--combined--chronological--budget-40 | 56.46 [55.56, 57.14] | 51.61 [48.39, 54.84] | 53.90 [51.72, 55.74] | 70.97 [67.74, 74.19] | 74.13 [72.41, 75.41] | 1.00 | 0.00 | 5.21 [4.94, 5.42] | 12.00 [11.00, 13.00] |
| event_starts--combined--evidence--budget-05 | 48.92 [48.39, 50.00] | 48.39 | 48.65 [48.39, 49.18] | 64.52 | 64.87 [64.52, 65.57] | 1.00 | 0.00 | 0.66 [0.66, 0.67] | 1.67 [1.00, 2.00] |
| event_starts--combined--evidence--budget-10 | 54.89 [53.57, 55.56] | 48.39 | 51.43 [50.85, 51.72] | 64.52 | 68.58 [67.80, 68.97] | 1.00 | 0.00 | 1.25 [1.23, 1.29] | 2.33 [2.00, 3.00] |
| event_starts--combined--evidence--budget-20 | 57.69 | 48.39 | 52.63 | 64.52 | 70.18 | 1.00 | 0.00 | 2.41 [2.20, 2.52] | 6.00 [5.00, 7.00] |
| event_starts--combined--evidence--budget-40 | 57.11 [55.56, 58.62] | 51.61 [48.39, 54.84] | 54.21 [51.72, 56.67] | 70.97 [67.74, 74.19] | 74.55 [72.41, 76.67] | 1.00 | 0.00 | 5.05 [4.94, 5.26] | 12.33 [11.00, 14.00] |
| event_starts--split_only--chronological--budget-05 | 42.19 [41.67, 43.24] | 49.46 [48.39, 51.61] | 45.54 [44.78, 47.06] | 65.59 [64.52, 67.74] | 61.30 [60.61, 62.69] | 1.00 | 0.00 | 0.54 | 2.00 |
| event_starts--split_only--chronological--budget-10 | 41.06 [40.54, 42.11] | 49.46 [48.39, 51.61] | 44.87 [44.12, 46.38] | 68.82 [67.74, 70.97] | 63.36 [62.69, 64.71] | 1.00 | 0.00 | 1.30 | 4.00 |
| event_starts--split_only--chronological--budget-20 | 41.06 [40.54, 42.11] | 49.46 [48.39, 51.61] | 44.87 [44.12, 46.38] | 68.82 [67.74, 70.97] | 63.36 [62.69, 64.71] | 1.00 | 0.00 | 2.46 [2.17, 2.60] | 6.33 [5.00, 7.00] |
| event_starts--split_only--chronological--budget-40 | 42.08 [40.54, 43.59] | 51.61 [48.39, 54.84] | 46.36 [44.12, 48.57] | 70.97 [67.74, 74.19] | 64.69 [62.69, 66.67] | 1.00 | 0.00 | 3.06 [2.77, 3.20] | 8.33 [7.00, 9.00] |
| event_starts--split_only--evidence--budget-05 | 41.67 | 48.39 | 44.78 | 64.52 | 60.61 | 1.00 | 0.00 | 0.47 [0.43, 0.54] | 2.00 |
| event_starts--split_only--evidence--budget-10 | 42.19 [41.67, 43.24] | 49.46 [48.39, 51.61] | 45.54 [44.78, 47.06] | 65.59 [64.52, 67.74] | 61.30 [60.61, 62.69] | 1.00 | 0.00 | 1.03 [0.97, 1.14] | 4.00 |
| event_starts--split_only--evidence--budget-20 | 42.08 [40.54, 43.59] | 51.61 [48.39, 54.84] | 46.36 [44.12, 48.57] | 70.97 [67.74, 74.19] | 64.69 [62.69, 66.67] | 1.00 | 0.00 | 2.19 [1.90, 2.33] | 7.33 [6.00, 8.00] |
| event_starts--split_only--evidence--budget-40 | 42.08 [40.54, 43.59] | 51.61 [48.39, 54.84] | 46.36 [44.12, 48.57] | 70.97 [67.74, 74.19] | 64.69 [62.69, 66.67] | 1.00 | 0.00 | 3.06 [2.77, 3.20] | 8.33 [7.00, 9.00] |
| head_evidence--combined--chronological--budget-05 | 44.12 | 48.39 | 46.15 | 64.52 | 61.54 | 1.00 | 0.00 | 0.65 | 1.00 |
| head_evidence--combined--chronological--budget-10 | 49.02 [47.06, 50.00] | 49.46 [48.39, 51.61] | 49.20 [49.18, 49.23] | 65.59 [64.52, 67.74] | 65.25 [64.62, 65.57] | 1.00 | 0.00 | 1.30 [1.30, 1.30] | 1.67 [1.00, 3.00] |
| head_evidence--combined--chronological--budget-20 | 50.78 [47.06, 53.57] | 49.46 [48.39, 51.61] | 50.03 [49.23, 50.85] | 68.82 [67.74, 70.97] | 69.63 [67.69, 71.19] | 1.00 | 0.00 | 2.65 [2.60, 2.70] | 5.00 [4.00, 6.00] |
| head_evidence--combined--chronological--budget-40 | 55.32 [54.84, 55.56] | 50.54 [48.39, 54.84] | 52.76 [51.72, 54.84] | 69.89 [67.74, 74.19] | 73.01 [72.41, 74.19] | 1.00 | 0.00 | 4.52 [3.96, 5.37] | 9.67 [7.00, 13.00] |
| head_evidence--combined--evidence--budget-05 | 48.92 [48.39, 50.00] | 48.39 | 48.65 [48.39, 49.18] | 64.52 | 64.87 [64.52, 65.57] | 1.00 | 0.00 | 0.66 [0.66, 0.67] | 1.67 [1.00, 2.00] |
| head_evidence--combined--evidence--budget-10 | 54.89 [53.57, 55.56] | 48.39 | 51.43 [50.85, 51.72] | 64.52 | 68.58 [67.80, 68.97] | 1.00 | 0.00 | 1.25 [1.23, 1.29] | 2.33 [2.00, 3.00] |
| head_evidence--combined--evidence--budget-20 | 56.98 [55.56, 57.69] | 48.39 | 52.33 [51.72, 52.63] | 65.59 [64.52, 67.74] | 70.92 [70.18, 72.41] | 1.00 | 0.00 | 2.59 [2.58, 2.61] | 6.00 [5.00, 7.00] |
| head_evidence--combined--evidence--budget-40 | 56.58 [55.56, 58.62] | 50.54 [48.39, 54.84] | 53.37 [51.72, 56.67] | 69.89 [67.74, 74.19] | 73.83 [72.41, 76.67] | 1.00 | 0.00 | 4.40 [3.96, 5.02] | 10.00 [7.00, 14.00] |
| head_evidence--split_only--chronological--budget-05 | 42.19 [41.67, 43.24] | 49.46 [48.39, 51.61] | 45.54 [44.78, 47.06] | 65.59 [64.52, 67.74] | 61.30 [60.61, 62.69] | 1.00 | 0.00 | 0.40 [0.25, 0.54] | 1.67 [1.00, 2.00] |
| head_evidence--split_only--chronological--budget-10 | 41.06 [40.54, 42.11] | 49.46 [48.39, 51.61] | 44.87 [44.12, 46.38] | 68.82 [67.74, 70.97] | 63.36 [62.69, 64.71] | 1.00 | 0.00 | 1.16 [1.01, 1.30] | 3.67 [3.00, 4.00] |
| head_evidence--split_only--chronological--budget-20 | 41.06 [40.54, 42.11] | 49.46 [48.39, 51.61] | 44.87 [44.12, 46.38] | 68.82 [67.74, 70.97] | 63.36 [62.69, 64.71] | 1.00 | 0.00 | 2.18 [1.88, 2.61] | 5.33 [4.00, 7.00] |
| head_evidence--split_only--chronological--budget-40 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 2.60 [1.88, 3.90] | 7.00 [4.00, 12.00] |
| head_evidence--split_only--evidence--budget-05 | 41.67 | 48.39 | 44.78 | 64.52 | 60.61 | 1.00 | 0.00 | 0.36 [0.25, 0.44] | 1.67 [1.00, 2.00] |
| head_evidence--split_only--evidence--budget-10 | 41.44 [40.54, 43.24] | 49.46 [48.39, 51.61] | 45.10 [44.12, 47.06] | 67.74 | 62.69 | 1.00 | 0.00 | 1.13 [1.01, 1.23] | 4.00 [3.00, 5.00] |
| head_evidence--split_only--evidence--budget-20 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 2.17 [1.88, 2.59] | 6.00 [4.00, 9.00] |
| head_evidence--split_only--evidence--budget-40 | 41.56 [40.54, 43.59] | 50.54 [48.39, 54.84] | 45.60 [44.12, 48.57] | 69.89 [67.74, 74.19] | 64.01 [62.69, 66.67] | 1.00 | 0.00 | 2.60 [1.88, 3.90] | 7.00 [4.00, 12.00] |
| none--cleanup_only--chronological--budget-05 | 44.12 | 48.39 | 46.15 | 64.52 | 61.54 | 1.00 | 0.00 | 0.65 | 1.00 |
| none--cleanup_only--chronological--budget-10 | 50.00 | 48.39 | 49.18 | 64.52 | 65.57 | 1.00 | 0.00 | 1.30 | 1.00 |
| none--cleanup_only--chronological--budget-20 | 56.98 [55.56, 57.69] | 48.39 | 52.33 [51.72, 52.63] | 64.52 | 69.77 [68.97, 70.18] | 1.00 | 0.00 | 2.32 [2.06, 2.70] | 4.00 [3.00, 5.00] |
| none--cleanup_only--chronological--budget-40 | 57.69 | 48.39 | 52.63 | 64.52 | 70.18 | 1.00 | 0.00 | 2.83 [2.06, 4.24] | 5.33 [3.00, 9.00] |
| none--cleanup_only--evidence--budget-05 | 48.92 [48.39, 50.00] | 48.39 | 48.65 [48.39, 49.18] | 64.52 | 64.87 [64.52, 65.57] | 1.00 | 0.00 | 0.66 [0.66, 0.67] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-10 | 54.89 [53.57, 55.56] | 48.39 | 51.43 [50.85, 51.72] | 64.52 | 68.58 [67.80, 68.97] | 1.00 | 0.00 | 1.25 [1.23, 1.29] | 2.33 [2.00, 3.00] |
| none--cleanup_only--evidence--budget-20 | 57.69 | 48.39 | 52.63 | 64.52 | 70.18 | 1.00 | 0.00 | 2.29 [2.06, 2.62] | 4.33 [3.00, 6.00] |
| none--cleanup_only--evidence--budget-40 | 57.69 | 48.39 | 52.63 | 64.52 | 70.18 | 1.00 | 0.00 | 2.83 [2.06, 4.24] | 5.33 [3.00, 9.00] |

### grass-source-04

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 58.02 [56.82, 59.09] | 70.37 [69.44, 72.22] | 63.60 [62.50, 65.00] | 58.33 | 52.72 [52.50, 53.16] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 52.61 [48.98, 57.78] | 68.52 [66.67, 72.22] | 59.50 [56.47, 64.20] | 58.33 | 50.62 [49.41, 51.85] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 52.01 [48.21, 56.86] | 76.85 [75.00, 80.56] | 62.01 [58.70, 66.67] | 74.07 [69.44, 77.78] | 59.71 [56.18, 62.07] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.78 [0.77, 0.82] | 5.00 |
| corroborated--combined--chronological--budget-10 | 60.52 [59.52, 62.50] | 69.44 | 64.66 [64.10, 65.79] | 58.33 | 54.32 [53.85, 55.26] | 0.00 | 0.00 | 1.56 [1.53, 1.59] | 8.00 [7.00, 9.00] |
| corroborated--combined--chronological--budget-20 | 66.38 [65.79, 67.57] | 69.44 | 67.88 [67.57, 68.49] | 58.33 | 57.02 [56.76, 57.53] | 0.00 | 0.00 | 2.73 [2.12, 3.17] | 11.00 [9.00, 14.00] |
| corroborated--combined--chronological--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 3.24 [2.12, 4.26] | 12.00 [9.00, 16.00] |
| corroborated--combined--evidence--budget-05 | 62.50 | 69.44 | 65.79 | 58.33 | 55.26 | 0.00 | 0.00 | 0.80 | 4.00 |
| corroborated--combined--evidence--budget-10 | 65.23 [64.10, 65.79] | 69.44 | 67.27 [66.67, 67.57] | 58.33 | 56.50 [56.00, 56.76] | 0.00 | 0.00 | 1.57 [1.56, 1.58] | 7.33 [7.00, 8.00] |
| corroborated--combined--evidence--budget-20 | 66.97 [65.79, 67.57] | 69.44 | 68.18 [67.57, 68.49] | 58.33 | 57.28 [56.76, 57.53] | 0.00 | 0.00 | 2.73 [2.12, 3.24] | 11.00 [9.00, 14.00] |
| corroborated--combined--evidence--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 3.24 [2.12, 4.26] | 12.00 [9.00, 16.00] |
| corroborated--split_only--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-20 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-40 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-20 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-40 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.31 [0.00, 0.51] | 0.67 [0.00, 1.00] |
| event_starts--combined--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.78 [0.77, 0.82] | 5.00 |
| event_starts--combined--chronological--budget-10 | 60.03 [58.14, 60.98] | 69.44 | 64.39 [63.29, 64.94] | 58.33 | 54.09 [53.16, 54.55] | 0.00 | 0.00 | 1.62 [1.59, 1.64] | 7.33 [7.00, 8.00] |
| event_starts--combined--chronological--budget-20 | 64.10 | 69.44 | 66.67 | 58.33 | 56.00 | 0.00 | 0.00 | 3.23 [3.20, 3.27] | 12.33 [11.00, 14.00] |
| event_starts--combined--chronological--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 4.49 [3.66, 5.05] | 15.33 [13.00, 17.00] |
| event_starts--combined--evidence--budget-05 | 62.50 | 69.44 | 65.79 | 58.33 | 55.26 | 0.00 | 0.00 | 0.80 | 4.00 |
| event_starts--combined--evidence--budget-10 | 65.23 [64.10, 65.79] | 69.44 | 67.27 [66.67, 67.57] | 58.33 | 56.50 [56.00, 56.76] | 0.00 | 0.00 | 1.57 [1.56, 1.58] | 7.33 [7.00, 8.00] |
| event_starts--combined--evidence--budget-20 | 66.97 [65.79, 67.57] | 69.44 | 68.18 [67.57, 68.49] | 58.33 | 57.28 [56.76, 57.53] | 0.00 | 0.00 | 3.21 [3.17, 3.24] | 12.67 [12.00, 14.00] |
| event_starts--combined--evidence--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 4.49 [3.66, 5.05] | 15.33 [13.00, 17.00] |
| event_starts--split_only--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.67 [0.51, 0.75] | 1.67 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.33 [0.94, 1.54] | 3.33 [2.00, 4.00] |
| event_starts--split_only--chronological--budget-20 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.57 [0.94, 2.24] | 4.00 [2.00, 6.00] |
| event_starts--split_only--chronological--budget-40 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.57 [0.94, 2.24] | 4.00 [2.00, 6.00] |
| event_starts--split_only--evidence--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.54 [0.42, 0.60] | 1.67 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.27 [0.94, 1.54] | 3.33 [2.00, 4.00] |
| event_starts--split_only--evidence--budget-20 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.57 [0.94, 2.24] | 4.00 [2.00, 6.00] |
| event_starts--split_only--evidence--budget-40 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.57 [0.94, 2.24] | 4.00 [2.00, 6.00] |
| head_evidence--combined--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.80 [0.76, 0.82] | 4.67 [4.00, 5.00] |
| head_evidence--combined--chronological--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.57 [1.53, 1.60] | 7.67 [7.00, 8.00] |
| head_evidence--combined--chronological--budget-20 | 62.50 | 69.44 | 65.79 | 58.33 | 55.26 | 0.00 | 0.00 | 3.22 [3.19, 3.26] | 13.67 [13.00, 14.00] |
| head_evidence--combined--chronological--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 4.74 [4.07, 5.34] | 17.33 [16.00, 19.00] |
| head_evidence--combined--evidence--budget-05 | 62.50 | 69.44 | 65.79 | 58.33 | 55.26 | 0.00 | 0.00 | 0.80 | 4.00 |
| head_evidence--combined--evidence--budget-10 | 65.23 [64.10, 65.79] | 69.44 | 67.27 [66.67, 67.57] | 58.33 | 56.50 [56.00, 56.76] | 0.00 | 0.00 | 1.57 [1.56, 1.58] | 7.33 [7.00, 8.00] |
| head_evidence--combined--evidence--budget-20 | 66.97 [65.79, 67.57] | 69.44 | 68.18 [67.57, 68.49] | 58.33 | 57.28 [56.76, 57.53] | 0.00 | 0.00 | 3.19 [3.06, 3.26] | 13.33 [12.00, 14.00] |
| head_evidence--combined--evidence--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 4.74 [4.07, 5.34] | 17.33 [16.00, 19.00] |
| head_evidence--split_only--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.72 [0.66, 0.79] | 3.00 [2.00, 4.00] |
| head_evidence--split_only--chronological--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.56 [1.53, 1.59] | 5.33 [5.00, 6.00] |
| head_evidence--split_only--chronological--budget-20 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 2.34 [2.02, 2.76] | 7.67 [7.00, 9.00] |
| head_evidence--split_only--chronological--budget-40 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 2.34 [2.02, 2.76] | 7.67 [7.00, 9.00] |
| head_evidence--split_only--evidence--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.71 [0.62, 0.75] | 3.00 |
| head_evidence--split_only--evidence--budget-10 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 1.37 [1.32, 1.42] | 5.33 [5.00, 6.00] |
| head_evidence--split_only--evidence--budget-20 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 2.34 [2.02, 2.76] | 7.67 [7.00, 9.00] |
| head_evidence--split_only--evidence--budget-40 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 2.34 [2.02, 2.76] | 7.67 [7.00, 9.00] |
| none--cleanup_only--chronological--budget-05 | 58.14 | 69.44 | 63.29 | 58.33 | 53.16 | 0.00 | 0.00 | 0.78 [0.77, 0.82] | 5.00 |
| none--cleanup_only--chronological--budget-10 | 61.51 [59.52, 62.50] | 69.44 | 65.23 [64.10, 65.79] | 58.33 | 54.79 [53.85, 55.26] | 0.00 | 0.00 | 1.56 [1.53, 1.57] | 8.33 [8.00, 9.00] |
| none--cleanup_only--chronological--budget-20 | 66.38 [65.79, 67.57] | 69.44 | 67.88 [67.57, 68.49] | 58.33 | 57.02 [56.76, 57.53] | 0.00 | 0.00 | 2.70 [2.12, 3.17] | 11.00 [9.00, 14.00] |
| none--cleanup_only--chronological--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 2.92 [2.12, 3.82] | 11.33 [9.00, 15.00] |
| none--cleanup_only--evidence--budget-05 | 62.50 | 69.44 | 65.79 | 58.33 | 55.26 | 0.00 | 0.00 | 0.80 | 4.00 |
| none--cleanup_only--evidence--budget-10 | 65.23 [64.10, 65.79] | 69.44 | 67.27 [66.67, 67.57] | 58.33 | 56.50 [56.00, 56.76] | 0.00 | 0.00 | 1.57 [1.56, 1.58] | 7.33 [7.00, 8.00] |
| none--cleanup_only--evidence--budget-20 | 66.97 [65.79, 67.57] | 69.44 | 68.18 [67.57, 68.49] | 58.33 | 57.28 [56.76, 57.53] | 0.00 | 0.00 | 2.73 [2.12, 3.24] | 11.00 [9.00, 14.00] |
| none--cleanup_only--evidence--budget-40 | 67.60 [65.79, 69.44] | 69.44 | 68.50 [67.57, 69.44] | 58.33 | 57.54 [56.76, 58.33] | 0.00 | 0.00 | 2.92 [2.12, 3.82] | 11.33 [9.00, 15.00] |

### grass-source-01

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 51.09 [50.00, 51.67] | 68.89 [66.67, 71.11] | 58.67 [57.14, 59.81] | 84.44 | 71.93 [71.03, 72.38] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 51.71 [50.85, 53.45] | 67.41 [66.67, 68.89] | 58.53 [57.69, 60.19] | 85.93 [84.44, 86.67] | 74.60 [73.08, 75.73] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 62.00 | 68.89 | 65.26 | 84.44 | 80.00 | 0.00 | 0.00 | 1.01 [0.99, 1.07] | 1.33 [1.00, 2.00] |
| corroborated--combined--chronological--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| corroborated--combined--chronological--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| corroborated--combined--chronological--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| corroborated--combined--evidence--budget-05 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.00 [0.97, 1.05] | 0.67 [0.00, 1.00] |
| corroborated--combined--evidence--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| corroborated--combined--evidence--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| corroborated--combined--evidence--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| corroborated--split_only--chronological--budget-05 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--chronological--budget-10 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--chronological--budget-20 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--chronological--budget-40 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-05 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-10 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-20 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-40 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--combined--chronological--budget-05 | 60.41 [59.62, 62.00] | 68.89 | 64.37 [63.92, 65.26] | 84.44 | 78.90 [78.35, 80.00] | 0.00 | 0.00 | 1.02 [0.99, 1.04] | 2.00 [1.00, 3.00] |
| event_starts--combined--chronological--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 2.04 [2.01, 2.06] | 4.67 [4.00, 5.00] |
| event_starts--combined--chronological--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 2.56 [2.29, 3.02] | 6.33 [5.00, 8.00] |
| event_starts--combined--chronological--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 2.56 [2.29, 3.02] | 6.33 [5.00, 8.00] |
| event_starts--combined--evidence--budget-05 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.00 [0.97, 1.05] | 0.67 [0.00, 1.00] |
| event_starts--combined--evidence--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 2.04 [1.90, 2.18] | 5.00 [4.00, 6.00] |
| event_starts--combined--evidence--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 2.56 [2.29, 3.02] | 6.33 [5.00, 8.00] |
| event_starts--combined--evidence--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 2.56 [2.29, 3.02] | 6.33 [5.00, 8.00] |
| event_starts--split_only--chronological--budget-05 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.95 [0.84, 1.10] | 3.33 [3.00, 4.00] |
| event_starts--split_only--chronological--budget-10 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 1.08 [0.84, 1.49] | 3.67 [3.00, 5.00] |
| event_starts--split_only--chronological--budget-20 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 1.08 [0.84, 1.49] | 3.67 [3.00, 5.00] |
| event_starts--split_only--chronological--budget-40 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 1.08 [0.84, 1.49] | 3.67 [3.00, 5.00] |
| event_starts--split_only--evidence--budget-05 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.93 [0.84, 1.05] | 3.33 [3.00, 4.00] |
| event_starts--split_only--evidence--budget-10 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 1.08 [0.84, 1.49] | 3.67 [3.00, 5.00] |
| event_starts--split_only--evidence--budget-20 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 1.08 [0.84, 1.49] | 3.67 [3.00, 5.00] |
| event_starts--split_only--evidence--budget-40 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 1.08 [0.84, 1.49] | 3.67 [3.00, 5.00] |
| head_evidence--combined--chronological--budget-05 | 60.01 [59.62, 60.78] | 68.89 | 64.14 [63.92, 64.58] | 84.44 | 78.62 [78.35, 79.17] | 0.00 | 0.00 | 1.04 [1.00, 1.08] | 2.00 |
| head_evidence--combined--chronological--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.79 [1.71, 1.93] | 4.00 [3.00, 5.00] |
| head_evidence--combined--chronological--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.79 [1.71, 1.93] | 4.00 [3.00, 5.00] |
| head_evidence--combined--chronological--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.79 [1.71, 1.93] | 4.00 [3.00, 5.00] |
| head_evidence--combined--evidence--budget-05 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.00 [0.97, 1.05] | 0.67 [0.00, 1.00] |
| head_evidence--combined--evidence--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.79 [1.71, 1.93] | 4.00 [3.00, 5.00] |
| head_evidence--combined--evidence--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.79 [1.71, 1.93] | 4.00 [3.00, 5.00] |
| head_evidence--combined--evidence--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.79 [1.71, 1.93] | 4.00 [3.00, 5.00] |
| head_evidence--split_only--chronological--budget-05 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--chronological--budget-10 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--chronological--budget-20 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--chronological--budget-40 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--evidence--budget-05 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--evidence--budget-10 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--evidence--budget-20 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| head_evidence--split_only--evidence--budget-40 | 54.39 | 68.89 | 60.78 | 84.44 | 74.51 | 0.00 | 0.00 | 0.32 [0.21, 0.48] | 1.33 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-05 | 62.00 | 68.89 | 65.26 | 84.44 | 80.00 | 0.00 | 0.00 | 1.01 [0.99, 1.07] | 1.33 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| none--cleanup_only--chronological--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| none--cleanup_only--chronological--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| none--cleanup_only--evidence--budget-05 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.00 [0.97, 1.05] | 0.67 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-10 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| none--cleanup_only--evidence--budget-20 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |
| none--cleanup_only--evidence--budget-40 | 62.84 [62.00, 63.27] | 68.89 | 65.73 [65.26, 65.96] | 84.44 | 80.57 [80.00, 80.85] | 0.00 | 0.00 | 1.48 [1.45, 1.53] | 2.67 [2.00, 3.00] |

### grass-source-09

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 73.68 | 70.00 | 71.79 | 57.50 | 58.97 | 2.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 67.51 [61.90, 73.17] | 70.83 [65.00, 75.00] | 69.12 [63.41, 74.07] | 59.17 [57.50, 60.00] | 57.73 [56.10, 59.26] | 2.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 72.01 [71.43, 73.17] | 75.00 | 73.47 [73.17, 74.07] | 65.00 [62.50, 67.50] | 63.68 [60.98, 65.85] | 2.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 73.68 | 70.00 | 71.79 | 57.50 | 58.97 | 2.00 | 0.00 | 0.83 [0.81, 0.86] | 4.67 [4.00, 5.00] |
| corroborated--combined--chronological--budget-10 | 75.50 [74.36, 77.78] | 71.67 [70.00, 72.50] | 73.51 [73.42, 73.68] | 59.17 [57.50, 60.00] | 60.68 [60.53, 60.76] | 2.00 | 0.00 | 1.68 [1.65, 1.74] | 8.00 |
| corroborated--combined--chronological--budget-20 | 78.18 [77.78, 78.38] | 71.67 [70.00, 72.50] | 74.78 [73.68, 75.32] | 59.17 [57.50, 60.00] | 61.73 [60.53, 62.34] | 2.00 | 0.00 | 2.70 [2.03, 3.17] | 12.67 [10.00, 15.00] |
| corroborated--combined--chronological--budget-40 | 78.18 [77.78, 78.38] | 71.67 [70.00, 72.50] | 74.78 [73.68, 75.32] | 59.17 [57.50, 60.00] | 61.73 [60.53, 62.34] | 2.00 | 0.00 | 2.70 [2.03, 3.17] | 12.67 [10.00, 15.00] |
| corroborated--combined--evidence--budget-05 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 0.79 [0.78, 0.81] | 5.00 |
| corroborated--combined--evidence--budget-10 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 1.54 [1.44, 1.65] | 9.00 [8.00, 10.00] |
| corroborated--combined--evidence--budget-20 | 78.18 [77.78, 78.38] | 71.67 [70.00, 72.50] | 74.78 [73.68, 75.32] | 59.17 [57.50, 60.00] | 61.73 [60.53, 62.34] | 2.00 | 0.00 | 2.70 [2.03, 3.17] | 12.67 [10.00, 15.00] |
| corroborated--combined--evidence--budget-40 | 78.18 [77.78, 78.38] | 71.67 [70.00, 72.50] | 74.78 [73.68, 75.32] | 59.17 [57.50, 60.00] | 61.73 [60.53, 62.34] | 2.00 | 0.00 | 2.70 [2.03, 3.17] | 12.67 [10.00, 15.00] |
| corroborated--split_only--chronological--budget-05 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-10 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-20 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-40 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-05 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-10 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-20 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-40 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.39 [0.00, 0.59] | 1.33 [0.00, 2.00] |
| event_starts--combined--chronological--budget-05 | 73.68 | 70.00 | 71.79 | 57.50 | 58.97 | 2.00 | 0.00 | 0.83 [0.81, 0.86] | 4.67 [4.00, 5.00] |
| event_starts--combined--chronological--budget-10 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 1.68 [1.65, 1.72] | 7.67 [7.00, 8.00] |
| event_starts--combined--chronological--budget-20 | 76.15 [74.36, 77.78] | 71.67 [70.00, 72.50] | 73.82 [73.42, 74.36] | 59.17 [57.50, 60.00] | 60.94 [60.53, 61.54] | 2.00 | 0.00 | 3.35 [3.20, 3.46] | 13.67 [13.00, 14.00] |
| event_starts--combined--chronological--budget-40 | 78.18 [77.78, 78.38] | 71.67 [70.00, 72.50] | 74.78 [73.68, 75.32] | 59.17 [57.50, 60.00] | 61.73 [60.53, 62.34] | 2.00 | 0.00 | 3.92 [3.20, 4.74] | 16.00 [13.00, 19.00] |
| event_starts--combined--evidence--budget-05 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 0.79 [0.78, 0.81] | 5.00 |
| event_starts--combined--evidence--budget-10 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 1.54 [1.44, 1.65] | 9.00 [8.00, 10.00] |
| event_starts--combined--evidence--budget-20 | 77.98 [77.78, 78.38] | 70.83 [70.00, 72.50] | 74.23 [73.68, 75.32] | 58.33 [57.50, 60.00] | 61.13 [60.53, 62.34] | 2.00 | 0.00 | 3.31 [3.20, 3.43] | 14.33 [13.00, 15.00] |
| event_starts--combined--evidence--budget-40 | 78.18 [77.78, 78.38] | 71.67 [70.00, 72.50] | 74.78 [73.68, 75.32] | 59.17 [57.50, 60.00] | 61.73 [60.53, 62.34] | 2.00 | 0.00 | 3.92 [3.20, 4.74] | 16.00 [13.00, 19.00] |
| event_starts--split_only--chronological--budget-05 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 0.66 [0.59, 0.80] | 2.00 |
| event_starts--split_only--chronological--budget-10 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 1.35 [1.17, 1.50] | 4.00 [3.00, 5.00] |
| event_starts--split_only--chronological--budget-20 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 1.61 [1.17, 2.17] | 4.67 [3.00, 6.00] |
| event_starts--split_only--chronological--budget-40 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 1.61 [1.17, 2.17] | 4.67 [3.00, 6.00] |
| event_starts--split_only--evidence--budget-05 | 73.68 | 70.00 | 71.79 | 57.50 | 58.97 | 2.00 | 0.00 | 0.60 [0.38, 0.71] | 1.67 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-10 | 73.91 [73.68, 74.36] | 70.83 [70.00, 72.50] | 72.34 [71.79, 73.42] | 58.33 [57.50, 60.00] | 59.57 [58.97, 60.76] | 2.00 | 0.00 | 1.42 [1.17, 1.58] | 4.00 [3.00, 5.00] |
| event_starts--split_only--evidence--budget-20 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 1.61 [1.17, 2.17] | 4.67 [3.00, 6.00] |
| event_starts--split_only--evidence--budget-40 | 74.13 [73.68, 74.36] | 71.67 [70.00, 72.50] | 72.88 [71.79, 73.42] | 59.17 [57.50, 60.00] | 60.16 [58.97, 60.76] | 2.00 | 0.00 | 1.61 [1.17, 2.17] | 4.67 [3.00, 6.00] |
| head_evidence--combined--chronological--budget-05 | 73.91 [73.68, 74.36] | 70.83 [70.00, 72.50] | 72.34 [71.79, 73.42] | 58.33 [57.50, 60.00] | 59.57 [58.97, 60.76] | 2.00 | 0.00 | 0.83 [0.81, 0.86] | 4.33 [4.00, 5.00] |
| head_evidence--combined--chronological--budget-10 | 75.66 [74.36, 76.32] | 72.50 | 74.05 [73.42, 74.36] | 60.00 | 61.28 [60.76, 61.54] | 2.00 | 0.00 | 1.69 [1.66, 1.72] | 8.00 |
| head_evidence--combined--chronological--budget-20 | 78.38 | 72.50 | 75.32 | 60.00 | 62.34 | 2.00 | 0.00 | 3.00 [2.52, 3.32] | 13.67 [12.00, 15.00] |
| head_evidence--combined--chronological--budget-40 | 78.38 | 72.50 | 75.32 | 60.00 | 62.34 | 2.00 | 0.00 | 3.06 [2.52, 3.50] | 14.00 [12.00, 15.00] |
| head_evidence--combined--evidence--budget-05 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 0.79 [0.78, 0.81] | 5.00 |
| head_evidence--combined--evidence--budget-10 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 1.63 [1.51, 1.74] | 9.33 [9.00, 10.00] |
| head_evidence--combined--evidence--budget-20 | 78.38 | 72.50 | 75.32 | 60.00 | 62.34 | 2.00 | 0.00 | 2.94 [2.52, 3.17] | 13.67 [12.00, 15.00] |
| head_evidence--combined--evidence--budget-40 | 78.38 | 72.50 | 75.32 | 60.00 | 62.34 | 2.00 | 0.00 | 3.06 [2.52, 3.50] | 14.00 [12.00, 15.00] |
| head_evidence--split_only--chronological--budget-05 | 73.91 [73.68, 74.36] | 70.83 [70.00, 72.50] | 72.34 [71.79, 73.42] | 58.33 [57.50, 60.00] | 59.57 [58.97, 60.76] | 2.00 | 0.00 | 0.66 [0.60, 0.78] | 2.33 [2.00, 3.00] |
| head_evidence--split_only--chronological--budget-10 | 74.36 | 72.50 | 73.42 | 60.00 | 60.76 | 2.00 | 0.00 | 1.33 [1.18, 1.56] | 4.67 [4.00, 5.00] |
| head_evidence--split_only--chronological--budget-20 | 74.36 | 72.50 | 73.42 | 60.00 | 60.76 | 2.00 | 0.00 | 1.33 [1.18, 1.56] | 4.67 [4.00, 5.00] |
| head_evidence--split_only--chronological--budget-40 | 74.36 | 72.50 | 73.42 | 60.00 | 60.76 | 2.00 | 0.00 | 1.33 [1.18, 1.56] | 4.67 [4.00, 5.00] |
| head_evidence--split_only--evidence--budget-05 | 73.68 | 70.00 | 71.79 | 57.50 | 58.97 | 2.00 | 0.00 | 0.62 [0.60, 0.66] | 2.33 [2.00, 3.00] |
| head_evidence--split_only--evidence--budget-10 | 74.36 | 72.50 | 73.42 | 60.00 | 60.76 | 2.00 | 0.00 | 1.33 [1.18, 1.56] | 4.67 [4.00, 5.00] |
| head_evidence--split_only--evidence--budget-20 | 74.36 | 72.50 | 73.42 | 60.00 | 60.76 | 2.00 | 0.00 | 1.33 [1.18, 1.56] | 4.67 [4.00, 5.00] |
| head_evidence--split_only--evidence--budget-40 | 74.36 | 72.50 | 73.42 | 60.00 | 60.76 | 2.00 | 0.00 | 1.33 [1.18, 1.56] | 4.67 [4.00, 5.00] |
| none--cleanup_only--chronological--budget-05 | 73.68 | 70.00 | 71.79 | 57.50 | 58.97 | 2.00 | 0.00 | 0.83 [0.81, 0.86] | 4.67 [4.00, 5.00] |
| none--cleanup_only--chronological--budget-10 | 75.71 [73.68, 77.78] | 70.00 | 72.74 [71.79, 73.68] | 57.50 | 59.75 [58.97, 60.53] | 2.00 | 0.00 | 1.72 [1.69, 1.74] | 8.33 [8.00, 9.00] |
| none--cleanup_only--chronological--budget-20 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 2.31 [2.03, 2.58] | 11.33 [10.00, 13.00] |
| none--cleanup_only--chronological--budget-40 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 2.31 [2.03, 2.58] | 11.33 [10.00, 13.00] |
| none--cleanup_only--evidence--budget-05 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 0.79 [0.78, 0.81] | 5.00 |
| none--cleanup_only--evidence--budget-10 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 1.54 [1.44, 1.65] | 9.00 [8.00, 10.00] |
| none--cleanup_only--evidence--budget-20 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 2.31 [2.03, 2.58] | 11.33 [10.00, 13.00] |
| none--cleanup_only--evidence--budget-40 | 77.78 | 70.00 | 73.68 | 57.50 | 60.53 | 2.00 | 0.00 | 2.31 [2.03, 2.58] | 11.33 [10.00, 13.00] |

### grass-source-10

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 48.61 [47.92, 48.98] | 58.33 [57.50, 60.00] | 53.03 [52.27, 53.93] | 53.33 [52.50, 55.00] | 48.48 [47.73, 49.44] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 48.27 [47.92, 48.98] | 58.33 [57.50, 60.00] | 52.83 [52.27, 53.93] | 53.33 [52.50, 55.00] | 48.30 [47.73, 49.44] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 48.05 [47.17, 48.98] | 60.83 [60.00, 62.50] | 53.68 [53.33, 53.93] | 57.50 [55.00, 60.00] | 50.72 [49.44, 51.61] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 52.68 [52.27, 53.49] | 57.50 | 54.98 [54.76, 55.42] | 52.50 | 50.20 [50.00, 50.60] | 0.00 | 0.00 | 0.76 [0.76, 0.76] | 2.00 |
| corroborated--combined--chronological--budget-10 | 55.23 [53.49, 56.10] | 57.50 | 56.33 [55.42, 56.79] | 52.50 | 51.44 [50.60, 51.85] | 0.00 | 0.00 | 1.49 [1.44, 1.53] | 4.67 [4.00, 5.00] |
| corroborated--combined--chronological--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.13 [1.93, 2.50] | 7.33 [6.00, 9.00] |
| corroborated--combined--chronological--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.13 [1.93, 2.50] | 7.33 [6.00, 9.00] |
| corroborated--combined--evidence--budget-05 | 54.34 [53.49, 54.76] | 57.50 | 55.87 [55.42, 56.10] | 52.50 | 51.01 [50.60, 51.22] | 0.00 | 0.00 | 0.68 [0.67, 0.69] | 1.33 [1.00, 2.00] |
| corroborated--combined--evidence--budget-10 | 55.65 [54.76, 56.10] | 57.50 | 56.56 [56.10, 56.79] | 52.50 | 51.64 [51.22, 51.85] | 0.00 | 0.00 | 1.48 [1.44, 1.53] | 5.33 [4.00, 6.00] |
| corroborated--combined--evidence--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.13 [1.93, 2.50] | 7.33 [6.00, 9.00] |
| corroborated--combined--evidence--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.13 [1.93, 2.50] | 7.33 [6.00, 9.00] |
| corroborated--split_only--chronological--budget-05 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.28 [0.00, 0.50] | 1.00 [0.00, 2.00] |
| corroborated--split_only--chronological--budget-10 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.00, 0.83] | 1.67 [0.00, 3.00] |
| corroborated--split_only--chronological--budget-20 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.00, 0.83] | 1.67 [0.00, 3.00] |
| corroborated--split_only--chronological--budget-40 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.00, 0.83] | 1.67 [0.00, 3.00] |
| corroborated--split_only--evidence--budget-05 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.28 [0.00, 0.50] | 1.00 [0.00, 2.00] |
| corroborated--split_only--evidence--budget-10 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.00, 0.83] | 1.67 [0.00, 3.00] |
| corroborated--split_only--evidence--budget-20 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.00, 0.83] | 1.67 [0.00, 3.00] |
| corroborated--split_only--evidence--budget-40 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.00, 0.83] | 1.67 [0.00, 3.00] |
| event_starts--combined--chronological--budget-05 | 52.68 [52.27, 53.49] | 57.50 | 54.98 [54.76, 55.42] | 52.50 | 50.20 [50.00, 50.60] | 0.00 | 0.00 | 0.76 [0.76, 0.76] | 2.00 |
| event_starts--combined--chronological--budget-10 | 53.95 [52.27, 56.10] | 57.50 | 55.66 [54.76, 56.79] | 52.50 | 50.82 [50.00, 51.85] | 0.00 | 0.00 | 1.50 [1.44, 1.53] | 5.00 [4.00, 6.00] |
| event_starts--combined--chronological--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.30 [1.96, 2.50] | 8.00 [6.00, 9.00] |
| event_starts--combined--chronological--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.30 [1.96, 2.50] | 8.00 [6.00, 9.00] |
| event_starts--combined--evidence--budget-05 | 54.34 [53.49, 54.76] | 57.50 | 55.87 [55.42, 56.10] | 52.50 | 51.01 [50.60, 51.22] | 0.00 | 0.00 | 0.68 [0.67, 0.69] | 1.33 [1.00, 2.00] |
| event_starts--combined--evidence--budget-10 | 55.65 [54.76, 56.10] | 57.50 | 56.56 [56.10, 56.79] | 52.50 | 51.64 [51.22, 51.85] | 0.00 | 0.00 | 1.48 [1.44, 1.53] | 5.33 [4.00, 6.00] |
| event_starts--combined--evidence--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.30 [1.96, 2.50] | 8.00 [6.00, 9.00] |
| event_starts--combined--evidence--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.30 [1.96, 2.50] | 8.00 [6.00, 9.00] |
| event_starts--split_only--chronological--budget-05 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.34, 0.50] | 1.67 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-10 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.61 [0.50, 0.83] | 2.33 [2.00, 3.00] |
| event_starts--split_only--chronological--budget-20 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.61 [0.50, 0.83] | 2.33 [2.00, 3.00] |
| event_starts--split_only--chronological--budget-40 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.61 [0.50, 0.83] | 2.33 [2.00, 3.00] |
| event_starts--split_only--evidence--budget-05 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.44 [0.34, 0.50] | 1.67 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-10 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.61 [0.50, 0.83] | 2.33 [2.00, 3.00] |
| event_starts--split_only--evidence--budget-20 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.61 [0.50, 0.83] | 2.33 [2.00, 3.00] |
| event_starts--split_only--evidence--budget-40 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.61 [0.50, 0.83] | 2.33 [2.00, 3.00] |
| head_evidence--combined--chronological--budget-05 | 52.68 [52.27, 53.49] | 57.50 | 54.98 [54.76, 55.42] | 52.50 | 50.20 [50.00, 50.60] | 0.00 | 0.00 | 0.76 [0.76, 0.76] | 2.00 |
| head_evidence--combined--chronological--budget-10 | 53.51 [52.27, 54.76] | 57.50 | 55.43 [54.76, 56.10] | 52.50 | 50.61 [50.00, 51.22] | 0.00 | 0.00 | 1.50 [1.46, 1.52] | 5.00 [4.00, 6.00] |
| head_evidence--combined--chronological--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.57 [2.22, 2.96] | 9.00 [7.00, 11.00] |
| head_evidence--combined--chronological--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.57 [2.22, 2.96] | 9.00 [7.00, 11.00] |
| head_evidence--combined--evidence--budget-05 | 54.34 [53.49, 54.76] | 57.50 | 55.87 [55.42, 56.10] | 52.50 | 51.01 [50.60, 51.22] | 0.00 | 0.00 | 0.68 [0.67, 0.69] | 1.33 [1.00, 2.00] |
| head_evidence--combined--evidence--budget-10 | 55.65 [54.76, 56.10] | 57.50 | 56.56 [56.10, 56.79] | 52.50 | 51.64 [51.22, 51.85] | 0.00 | 0.00 | 1.45 [1.36, 1.53] | 5.33 [4.00, 6.00] |
| head_evidence--combined--evidence--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.57 [2.22, 2.96] | 9.00 [7.00, 11.00] |
| head_evidence--combined--evidence--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 2.57 [2.22, 2.96] | 9.00 [7.00, 11.00] |
| head_evidence--split_only--chronological--budget-05 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.63 [0.57, 0.75] | 2.00 |
| head_evidence--split_only--chronological--budget-10 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 1.04 [0.57, 1.48] | 3.67 [2.00, 5.00] |
| head_evidence--split_only--chronological--budget-20 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 1.04 [0.57, 1.48] | 3.67 [2.00, 5.00] |
| head_evidence--split_only--chronological--budget-40 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 1.04 [0.57, 1.48] | 3.67 [2.00, 5.00] |
| head_evidence--split_only--evidence--budget-05 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 0.59 [0.57, 0.64] | 2.00 |
| head_evidence--split_only--evidence--budget-10 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 1.04 [0.57, 1.48] | 3.67 [2.00, 5.00] |
| head_evidence--split_only--evidence--budget-20 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 1.04 [0.57, 1.48] | 3.67 [2.00, 5.00] |
| head_evidence--split_only--evidence--budget-40 | 48.94 | 57.50 | 52.87 | 52.50 | 48.28 | 0.00 | 0.00 | 1.04 [0.57, 1.48] | 3.67 [2.00, 5.00] |
| none--cleanup_only--chronological--budget-05 | 52.68 [52.27, 53.49] | 57.50 | 54.98 [54.76, 55.42] | 52.50 | 50.20 [50.00, 50.60] | 0.00 | 0.00 | 0.76 [0.76, 0.76] | 2.00 |
| none--cleanup_only--chronological--budget-10 | 55.23 [53.49, 56.10] | 57.50 | 56.33 [55.42, 56.79] | 52.50 | 51.44 [50.60, 51.85] | 0.00 | 0.00 | 1.38 [1.11, 1.53] | 4.33 [3.00, 5.00] |
| none--cleanup_only--chronological--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 1.68 [1.11, 2.00] | 5.67 [3.00, 7.00] |
| none--cleanup_only--chronological--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 1.68 [1.11, 2.00] | 5.67 [3.00, 7.00] |
| none--cleanup_only--evidence--budget-05 | 54.34 [53.49, 54.76] | 57.50 | 55.87 [55.42, 56.10] | 52.50 | 51.01 [50.60, 51.22] | 0.00 | 0.00 | 0.68 [0.67, 0.69] | 1.33 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-10 | 55.65 [54.76, 56.10] | 57.50 | 56.56 [56.10, 56.79] | 52.50 | 51.64 [51.22, 51.85] | 0.00 | 0.00 | 1.37 [1.11, 1.53] | 5.00 [3.00, 6.00] |
| none--cleanup_only--evidence--budget-20 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 1.68 [1.11, 2.00] | 5.67 [3.00, 7.00] |
| none--cleanup_only--evidence--budget-40 | 56.57 [56.10, 57.50] | 57.50 | 57.03 [56.79, 57.50] | 52.50 | 52.07 [51.85, 52.50] | 0.00 | 0.00 | 1.68 [1.11, 2.00] | 5.67 [3.00, 7.00] |

### indoor-source-01

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 84.27 [81.40, 85.71] | 93.86 [92.11, 94.74] | 88.81 [86.42, 90.00] | 94.74 | 89.63 [88.89, 90.00] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 79.68 [73.91, 83.72] | 92.11 [89.47, 94.74] | 85.42 [80.95, 88.89] | 94.74 | 87.83 [85.71, 88.89] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 81.46 [77.27, 85.71] | 92.11 [89.47, 94.74] | 86.45 [82.93, 90.00] | 94.74 | 88.90 [87.80, 90.00] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--chronological--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--chronological--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--chronological--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.23 [0.11, 0.48] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-05 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-10 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-20 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-40 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-05 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-10 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-20 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-40 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.12 [0.00, 0.37] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.49 [0.48, 0.51] | 1.00 |
| event_starts--combined--chronological--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.62 [0.48, 0.88] | 1.33 [1.00, 2.00] |
| event_starts--combined--chronological--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.62 [0.48, 0.88] | 1.33 [1.00, 2.00] |
| event_starts--combined--chronological--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.62 [0.48, 0.88] | 1.33 [1.00, 2.00] |
| event_starts--combined--evidence--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.49 [0.48, 0.51] | 1.00 |
| event_starts--combined--evidence--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.62 [0.48, 0.88] | 1.33 [1.00, 2.00] |
| event_starts--combined--evidence--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.62 [0.48, 0.88] | 1.33 [1.00, 2.00] |
| event_starts--combined--evidence--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.62 [0.48, 0.88] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-05 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-10 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-20 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--chronological--budget-40 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-05 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-10 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-20 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| event_starts--split_only--evidence--budget-40 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.52 [0.37, 0.77] | 1.33 [1.00, 2.00] |
| head_evidence--combined--chronological--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.35 [0.11, 0.48] | 0.67 [0.00, 1.00] |
| head_evidence--combined--chronological--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.47 [0.11, 0.82] | 1.00 [0.00, 2.00] |
| head_evidence--combined--chronological--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.47 [0.11, 0.82] | 1.00 [0.00, 2.00] |
| head_evidence--combined--chronological--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.47 [0.11, 0.82] | 1.00 [0.00, 2.00] |
| head_evidence--combined--evidence--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.36 [0.11, 0.48] | 0.67 [0.00, 1.00] |
| head_evidence--combined--evidence--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.47 [0.11, 0.82] | 1.00 [0.00, 2.00] |
| head_evidence--combined--evidence--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.47 [0.11, 0.82] | 1.00 [0.00, 2.00] |
| head_evidence--combined--evidence--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.47 [0.11, 0.82] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-05 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-10 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-20 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-40 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-05 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-10 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-20 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-40 | 85.71 | 94.74 | 90.00 | 94.74 | 90.00 | 0.00 | 0.00 | 0.36 [0.00, 0.71] | 1.00 [0.00, 2.00] |
| none--cleanup_only--chronological--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--chronological--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--chronological--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--chronological--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--evidence--budget-05 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--evidence--budget-10 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--evidence--budget-20 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |
| none--cleanup_only--evidence--budget-40 | 87.80 | 94.74 | 91.14 | 94.74 | 91.14 | 0.00 | 0.00 | 0.11 | 0.00 |

### indoor-source-07

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 82.01 [77.50, 86.49] | 87.96 [86.11, 88.89] | 84.86 [81.58, 87.67] | 82.41 [80.56, 83.33] | 79.47 [78.95, 80.00] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--chronological--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--chronological--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--chronological--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--combined--evidence--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-05 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--chronological--budget-10 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--chronological--budget-20 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--chronological--budget-40 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-05 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-10 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-20 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--split_only--evidence--budget-40 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--combined--chronological--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--evidence--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--evidence--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--evidence--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--combined--evidence--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| event_starts--split_only--chronological--budget-05 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--chronological--budget-10 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--chronological--budget-20 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--chronological--budget-40 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--evidence--budget-05 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--evidence--budget-10 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--evidence--budget-20 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| event_starts--split_only--evidence--budget-40 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.00 | 0.00 |
| head_evidence--combined--chronological--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--chronological--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--chronological--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--chronological--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--evidence--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--evidence--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--evidence--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--combined--evidence--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.48 [0.10, 0.69] | 1.67 [0.00, 3.00] |
| head_evidence--split_only--chronological--budget-05 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-10 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-20 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--chronological--budget-40 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-05 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-10 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-20 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| head_evidence--split_only--evidence--budget-40 | 86.49 | 88.89 | 87.67 | 80.56 | 79.45 | 0.00 | 0.00 | 0.35 [0.00, 0.57] | 1.33 [0.00, 2.00] |
| none--cleanup_only--chronological--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--chronological--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--chronological--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--chronological--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-05 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-10 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-20 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |
| none--cleanup_only--evidence--budget-40 | 88.89 | 88.89 | 88.89 | 80.56 | 80.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.33 [0.00, 1.00] |

### indoor-source-08

| Arm | Rally P % | Rally R % | Rally F1 % | Observed start R @1s % | Observed start F1 @1s % | Complete misses | Additional misses | Review min | Real rallies reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--corroborated | 59.58 [59.38, 60.00] | 68.45 [67.86, 69.64] | 63.71 [63.33, 64.46] | 77.38 [76.79, 78.57] | 72.02 [71.67, 72.73] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--event_starts | 58.09 [56.72, 59.09] | 68.45 [67.86, 69.64] | 62.84 [61.79, 63.93] | 77.38 [76.79, 78.57] | 71.04 [69.92, 72.13] | 0.00 | 0.00 | 0.00 | 0.00 |
| automatic--head_evidence | 56.20 [55.56, 56.52] | 70.24 [69.64, 71.43] | 62.43 [62.40, 62.50] | 84.52 [83.93, 85.71] | 75.13 [75.00, 75.20] | 0.00 | 0.00 | 0.00 | 0.00 |
| corroborated--combined--chronological--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--chronological--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--combined--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.68 [0.47, 0.96] | 2.00 [1.00, 3.00] |
| corroborated--split_only--chronological--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--chronological--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| corroborated--split_only--evidence--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.12 [0.00, 0.35] | 0.33 [0.00, 1.00] |
| event_starts--combined--chronological--budget-05 | 61.96 [61.29, 62.30] | 67.86 | 64.77 [64.41, 64.96] | 76.79 | 73.30 [72.88, 73.50] | 0.00 | 0.00 | 0.97 [0.86, 1.04] | 3.33 [3.00, 4.00] |
| event_starts--combined--chronological--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.90 [0.79, 1.04] | 3.00 |
| event_starts--combined--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--combined--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.14 [0.86, 1.39] | 3.67 [3.00, 4.00] |
| event_starts--split_only--chronological--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--chronological--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| event_starts--split_only--evidence--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.57 [0.26, 0.92] | 2.00 [1.00, 3.00] |
| head_evidence--combined--chronological--budget-05 | 61.63 [61.29, 62.30] | 67.86 | 64.59 [64.41, 64.96] | 76.79 | 73.09 [72.88, 73.50] | 0.00 | 0.00 | 1.07 [1.05, 1.11] | 4.00 |
| head_evidence--combined--chronological--budget-10 | 61.96 [61.29, 62.30] | 67.86 | 64.77 [64.41, 64.96] | 76.79 | 73.30 [72.88, 73.50] | 0.00 | 0.00 | 2.01 [1.93, 2.08] | 6.33 [6.00, 7.00] |
| head_evidence--combined--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--combined--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--combined--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.05 [0.96, 1.09] | 3.67 [3.00, 4.00] |
| head_evidence--combined--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 1.87 [1.69, 1.99] | 6.00 [5.00, 7.00] |
| head_evidence--combined--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--combined--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 2.51 [2.23, 2.99] | 7.33 [6.00, 9.00] |
| head_evidence--split_only--chronological--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.94 [0.78, 1.08] | 3.33 [3.00, 4.00] |
| head_evidence--split_only--chronological--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.82 [1.70, 2.00] | 5.33 [5.00, 6.00] |
| head_evidence--split_only--chronological--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| head_evidence--split_only--chronological--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| head_evidence--split_only--evidence--budget-05 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 0.98 [0.84, 1.08] | 3.67 [3.00, 4.00] |
| head_evidence--split_only--evidence--budget-10 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.82 [1.70, 2.01] | 5.33 [5.00, 6.00] |
| head_evidence--split_only--evidence--budget-20 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| head_evidence--split_only--evidence--budget-40 | 59.38 | 67.86 | 63.33 | 76.79 | 71.67 | 0.00 | 0.00 | 1.95 [1.70, 2.39] | 5.67 [5.00, 7.00] |
| none--cleanup_only--chronological--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--chronological--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-05 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-10 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-20 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |
| none--cleanup_only--evidence--budget-40 | 62.30 | 67.86 | 64.96 | 76.79 | 73.50 | 0.00 | 0.00 | 0.56 [0.47, 0.61] | 1.67 [1.00, 2.00] |

## Interpretation limits and provenance

Unchanged export recall does not imply correct rally counts or safe automatic score updates. A false split can create a false scoring event despite retaining every frame. Observed start localization is a serve-contact timing proxy, not serving-side, point-winner or reconstructed-score accuracy. Dead time remains inside partitioned event records because no automatic endpoint trimming is performed.

These development recordings informed earlier model/decoder choices. Results require fresh-footage validation and actual human review measurements before a product decision. Low-confidence cleanup flags alone leave both export and event metrics unchanged until a human acts.

The summary validates exact four-padding export equality, zero additional complete misses, zero raw-core loss, and automatic raw-occupancy preservation for every arm. Upstream candidate, partition, queue, human-action, identity and duration audits remain bound in the original result files. Complete numeric means/ranges, scope summaries and input references are in `summary.json` alongside the registered experiment.
