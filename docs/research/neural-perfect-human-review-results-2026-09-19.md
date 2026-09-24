# Perfect-human rally review results — 19 September 2026

Completed 120 fixed review policies across three neural seeds (360 cells), with 30 prior export-disagreement policies retained as a separately named reference. These are simulations under explicit perfect-human assumptions, not measured reviewer outcomes. No fitting or production changes occurred.

## Findings and practical choices

**Yes, existing neural outputs can produce a useful needs-review decision, but the available human action matters.** A correct keep/remove decision classifies a whole proposed candidate. It cannot trim unwanted footage inside a candidate that contains genuine play. Perfect boundary editing makes much larger quality gains from the same flags. The tables below preserve both interpretations rather than attributing those gains to binary review.

The very restrictive guarded compact policy discussed before this study flags just **one false candidate**, containing **zero real rallies**, with **10.5 seconds of contextual playback**. Correctly dropping it gives 72.55% precision, 99.27% recall and 83.83% F1_padP_coreR. It removes about 6.98 seconds of unwanted export; the slightly different 6.5-second result from correction limited to the decision export reflects export-join effects. This remains a tiny change from production's 83.76% F1. Broadening flags is useful when a human, rather than an automatic veto, resolves disagreement.

For phone/browser-oriented review assistance, production plus **compact short boost with zero tolerant neural support** is a reasonable development candidate: **12.86 minutes** of playback (9.33% of the footage), **65 proposed candidates**, and **29 distinct real rallies** involved. Perfect whole-candidate decisions give **76.85% P / 99.27% R / 86.63% F1**, a 2.87-point F1 gain over production. Actual missed core is 17.54 seconds versus production's 17.47 seconds, so the rounded recall equality must not be called exact coverage preservation. Allowing boundary edits in the same queue gives 87.01% F1.

For desktop review, production plus **DINO short boost with less than half tolerant support** gives **77.88% P / 99.26% R / 87.28% F1**, reviewing **14.22 minutes**, about **60 candidates**, and **20 real rallies**. Bidirectional half-support flags add possible missed-rally proposals: **15.92 minutes**, **72 decisions** (59.67 production candidates and 12.33 neural-only candidates), and **21.67 real rallies**, for **77.92% P / 99.42% R / 87.37% F1**. These are useful low-effort production-review choices from this development matrix, not newly validated deployment policies.

The stronger high-coverage workflow includes boundary correction. Production plus **DINO global, bidirectional any unsupported time**, reaches **92.36% P / 99.68% R / 95.88% F1** after perfect editing of flagged regions. It requires **54.80 minutes** of playback (39.76% of footage), **188.67 candidate decisions**, and **143 real rallies** in raw flagged regions. The same flags with keep/remove alone give only **87.57% F1**, because mixed candidates retain false tails. Compact short boost is close in the boundary-editing workflow: **92.25% P / 99.53% R / 95.75% F1**, **57.76 minutes**, **198.67 decisions**, and **158 real rallies**. This suggests a compact model can provide useful on-device review assistance even though DINO's standalone detector is stronger. Device latency was not measured here.

Individual neural models can also flag review without production. Reviewing only positive candidates with mean live score below 0.8 gives:

- **DINO global:** 4.12 minutes, 26.67 proposed candidates and 14.67 true rallies; binary **90.66% P / 96.94% R / 93.69% F1**, compared with 93.07% automatic F1. Boundary editing gives 93.85% F1. This is a small queue, but it does not search for missing rallies and retains much less core play than production-based review.
- **DINO short boost:** 2.54 minutes, 18.33 candidates and 6.33 true rallies; binary **89.39% P / 96.93% R / 93.00% F1**, compared with 92.51% automatic F1.
- **Compact short boost:** 4.84 minutes, 27.33 candidates and 21.33 true rallies; binary **88.81% P / 92.39% R / 90.56% F1**, compared with 90.31% automatic F1. Its missed-rally limitation remains.

Searching uncertain negative sections recovers many omissions, but the current fixed score rules are broad. DINO global's medium rule needs **86.54 minutes**, **624.33 decisions** (26.67 positive candidates plus 597.67 negative search sections), and **221 real rallies** involved. With perfect boundary editing it reaches **97.88% P / 99.51% R / 98.69% F1**. The wide rule reaches **98.82% P / 99.73% R / 99.27% F1**, but needs **96.94 minutes**, **742 decisions** (65.67 positive plus 676.33 negative), and **242.67 real rallies**. These review 62.78% and 70.33% of the footage respectively; the search is not yet selective enough to describe as a lightweight review feature.

The same medium/wide negative-section queues give only **87.92% / 87.64% binary F1**. This is not a simulated human error: correctly keeping an entire five-second search section that contains some real play also retains its dead time. A practical missed-rally search UI needs boundary selection or better event-sized proposals, not just accept/reject of fixed windows. Reviewing every predicted DINO-global positive candidate is also expensive (83.08 minutes, 336.67 candidates, 304 real rallies): binary F1 is 94.39%, while boundary editing reaches 98.46% at the original 96.96% recall. Reviewing every candidate and negative section reaches 100% only in the perfect-editing reference and requires all 137.83 minutes; it is a ceiling, not a selective-flagger success.

**Recommendation:** test production plus broader neural disagreement flags as a human-review feature, retaining explicit keep/remove controls and allowing boundary edits on mixed or recovered candidates. Compact is a credible phone/browser flagger; DINO provides a desktop comparison. For an individual neural model, positive-only uncertainty is inexpensive but does not solve omissions. The next bounded study should improve the proposal boundaries and ranking of negative search sections under a review budget. A learned needs-review head could use separately cross-fitted candidate and boundary errors as targets, with source-group-held validation; it has not been trained in this study.

The fixed uncertainty thresholds are uncalibrated research rules. Candidate decisions, true rallies and playback clips are different workload measures; raw flagged regions and playback context also touch different numbers of rallies. All table counts average three seeds. Differences between the two human assumptions are explicit throughout the artifacts. The independent audit found no numerical grid slivers (the shortest negative candidate was 0.1083 seconds).

The quality-versus-workload plot (ledger `private-reference-0162`) compares both human assumptions; an SVG version (ledger `private-reference-0163`) is also available.

## Scope, units and definitions

The same eight indoor/grass recordings contain 322 labeled rallies and 137.83 evaluable minutes. Target product padding is ±2 s with positive gaps joined only when strictly below 3 s; ignored time is excluded and never rejoined. P_pad measures acceptable padded export, R_core actual play retained, and F1_padP_coreR their harmonic mean. Each seed pools recordings first; tables average three seed scores/counts. Human export at target padding is 61.25 minutes.

A **decision** is a flagged proposed raw rally or negative search section. **True rallies** counts distinct gold rally IDs overlapping raw decision regions; it excludes false-positive candidates and deduplicates repeated fragments. **Playback clips** merge padded decision regions plus two seconds of viewing context using strict <3 s joins. True rallies appearing only in context are counted separately. Viewing minutes mean footage at 1×, not observed wall-clock labor. Negative sections are split on an absolute five-second grid, so they must not be called rally detections.

**Binary keep/remove:** keep an entire flagged raw candidate if any evaluable human core play overlaps it; otherwise remove it. Unflagged automatic output remains. Ordinary export padding/joining follows. Mixed candidates keep their incorrect tails. A false raw candidate can have export padding that happens to cover nearby play; dropping that candidate can lose such coverage. Correct raw-candidate classification therefore does not imply perfect boundaries or monotonic export recall.

**Perfect boundary editing:** replace only the flagged candidates’ padded decision-export regions with the human export inside those regions. This can trim false tails and recover only play within flagged regions. Playback context does not grant additional edit permission, and no second padding/joining is applied. It is a distinct optimistic editing scenario. Neither scenario fixes unflagged errors automatically.

The current neural networks have live, serve/start, end and keep heads; **no learned needs-review head**. This study adds fixed review rules using model disagreement or uncalibrated live scores. A score of 0.8 is not established to mean 80% correctness. A dedicated review model or tuned review budget needs separate validation.

## Policy key

- `guarded_trim`: existing guarded compact/DINO trimming acts only as a flag trigger; review its entire containing production candidate. This expansion can remove more than the automatically trimmed tail.
- `suppression_zero/half/any`: flag production candidates with zero / less than half / any missing support from neural intervals expanded ±2 s.
- `bidirectional_half/any`: use both directions of that support test, permitting neural-only missed-rally proposals; review containing raw union components.
- `positive_uncertain`: review neural positive candidates with mean live score below 0.8; no search in omitted sections.
- `uncertain_narrow/medium/wide`: positive mean below 0.7/0.8/0.9, or negative-section maximum above 0.3/0.2/0.1. No-score sections are flagged by applicable uncertainty rules.
- `all_positive`: review every predicted positive candidate, without negative search.
- `all_candidates`: review every positive and negative section, a full-footage reference.

Compact boost means reviewed-export compact short boost; compact keep means reviewed-export keep rescue. DINO global is the draft/global control; DINO boost and keep use reviewed-export supervision. ProductionDefault is the checked-in fresh-project aggressive suppression setting, not a query of installed/deployed apps. ShippedUnion has suppression disabled; refitUnion is a source-group-held sensitivity baseline. Shipped weights historically saw these recordings; refit decoder choices and neural candidates remain development-exposed.

## Complete human-assisted quality and workload matrix

Sorted by binary keep/remove F1_padP_coreR at the predeclared target padding. The same review workload applies to both human assumptions. Counts can be fractional because they average seeds. All fixed policies are retained; this is descriptive development ranking, not a validated deployment selection.

| Policy | Automatic F1 % | Binary P % | Binary R % | Binary F1 % | Boundary P % | Boundary R % | Boundary F1 % | Playback min | Decisions | True rallies | Playback clips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dino_global--all_positive` | 93.07 | 92.01 | 96.92 | 94.39 | 100.00 | 96.96 | 98.46 | 83.08 | 336.67 | 304.00 | 277.67 |
| `dino_boost--all_positive` | 92.51 | 90.98 | 96.87 | 93.83 | 100.00 | 96.95 | 98.45 | 83.71 | 345.00 | 307.00 | 279.00 |
| `dino_global--positive_uncertain` | 93.07 | 90.66 | 96.94 | 93.69 | 90.96 | 96.96 | 93.85 | 4.12 | 26.67 | 14.67 | 25.00 |
| `dino_keep--all_positive` | 91.94 | 89.98 | 96.44 | 93.07 | 100.00 | 96.47 | 98.20 | 83.32 | 345.00 | 306.33 | 276.67 |
| `dino_boost--positive_uncertain` | 92.51 | 89.39 | 96.93 | 93.00 | 89.61 | 96.95 | 93.13 | 2.54 | 18.33 | 6.33 | 15.33 |
| `dino_keep--positive_uncertain` | 91.94 | 88.97 | 96.47 | 92.54 | 89.92 | 96.47 | 93.06 | 7.23 | 41.67 | 29.67 | 36.00 |
| `compact_keep--all_positive` | 89.97 | 91.22 | 93.34 | 92.25 | 100.00 | 93.42 | 96.60 | 81.83 | 341.67 | 292.67 | 272.67 |
| `compact_boost--all_positive` | 90.31 | 91.72 | 92.36 | 92.03 | 100.00 | 92.41 | 96.05 | 79.40 | 324.00 | 288.00 | 268.33 |
| `compact_keep--positive_uncertain` | 89.97 | 88.09 | 93.37 | 90.64 | 88.72 | 93.42 | 90.99 | 8.18 | 47.67 | 34.33 | 44.00 |
| `compact_boost--positive_uncertain` | 90.31 | 88.81 | 92.39 | 90.56 | 89.10 | 92.41 | 90.72 | 4.84 | 27.33 | 21.33 | 26.33 |
| `dino_global--uncertain_narrow` | 93.07 | 79.07 | 99.41 | 88.08 | 97.23 | 99.41 | 98.31 | 81.09 | 562.33 | 211.00 | 206.33 |
| `dino_global--uncertain_medium` | 93.07 | 78.75 | 99.51 | 87.92 | 97.88 | 99.51 | 98.69 | 86.54 | 624.33 | 221.00 | 200.00 |
| `dino_global--uncertain_wide` | 93.07 | 78.16 | 99.73 | 87.64 | 98.82 | 99.73 | 99.27 | 96.94 | 742.00 | 242.67 | 191.00 |
| `shippedUnion--dino_boost--suppression_any` | 80.15 | 78.20 | 99.63 | 87.63 | 90.48 | 99.64 | 94.84 | 56.32 | 219.67 | 124.67 | 131.67 |
| `productionDefault--dino_global--suppression_any` | 83.76 | 78.44 | 99.18 | 87.60 | 91.71 | 99.27 | 95.34 | 48.24 | 165.67 | 127.67 | 117.00 |
| `productionDefault--dino_global--bidirectional_any` | 83.76 | 78.09 | 99.68 | 87.57 | 92.36 | 99.68 | 95.88 | 54.80 | 188.67 | 143.00 | 126.67 |
| `productionDefault--dino_boost--suppression_any` | 83.76 | 78.34 | 99.26 | 87.57 | 90.86 | 99.27 | 94.88 | 46.66 | 161.00 | 123.33 | 113.33 |
| `shippedUnion--dino_keep--suppression_any` | 80.15 | 78.03 | 99.63 | 87.52 | 90.45 | 99.64 | 94.81 | 60.14 | 230.00 | 135.67 | 139.33 |
| `shippedUnion--dino_global--suppression_any` | 80.15 | 78.05 | 99.56 | 87.50 | 91.01 | 99.64 | 95.13 | 57.83 | 223.33 | 129.67 | 133.33 |
| `compact_boost--uncertain_narrow` | 90.31 | 78.51 | 98.73 | 87.47 | 97.53 | 98.81 | 98.16 | 84.22 | 597.00 | 218.00 | 200.00 |
| `productionDefault--dino_keep--suppression_any` | 83.76 | 78.18 | 99.26 | 87.47 | 90.81 | 99.27 | 94.84 | 50.34 | 171.00 | 134.00 | 119.33 |
| `shippedUnion--compact_boost--suppression_any` | 80.15 | 77.95 | 99.63 | 87.47 | 91.18 | 99.64 | 95.22 | 62.35 | 243.00 | 149.33 | 135.33 |
| `productionDefault--compact_boost--suppression_any` | 83.76 | 78.11 | 99.26 | 87.43 | 91.51 | 99.27 | 95.23 | 52.75 | 182.33 | 146.33 | 124.67 |
| `shippedUnion--dino_global--bidirectional_any` | 80.15 | 77.74 | 99.82 | 87.40 | 91.64 | 99.83 | 95.56 | 62.90 | 240.33 | 142.33 | 137.67 |
| `compact_keep--uncertain_narrow` | 89.97 | 78.54 | 98.47 | 87.38 | 97.10 | 98.55 | 97.82 | 82.49 | 608.67 | 212.00 | 194.33 |
| `productionDefault--dino_boost--bidirectional_half` | 83.76 | 77.92 | 99.42 | 87.37 | 80.32 | 99.43 | 88.86 | 15.92 | 72.00 | 21.67 | 62.67 |
| `productionDefault--dino_boost--bidirectional_any` | 83.76 | 77.69 | 99.72 | 87.33 | 91.77 | 99.73 | 95.58 | 54.92 | 191.00 | 143.00 | 128.67 |
| `productionDefault--compact_keep--suppression_any` | 83.76 | 77.96 | 99.27 | 87.33 | 90.81 | 99.27 | 94.85 | 49.91 | 172.67 | 137.33 | 117.33 |
| `productionDefault--compact_boost--bidirectional_any` | 83.76 | 77.80 | 99.52 | 87.33 | 92.25 | 99.53 | 95.75 | 57.76 | 198.67 | 158.00 | 129.67 |
| `shippedUnion--compact_boost--bidirectional_any` | 80.15 | 77.62 | 99.76 | 87.31 | 91.83 | 99.77 | 95.63 | 66.36 | 254.33 | 159.00 | 135.33 |
| `shippedUnion--compact_keep--suppression_any` | 80.15 | 77.68 | 99.64 | 87.30 | 90.28 | 99.64 | 94.73 | 59.45 | 231.67 | 140.00 | 131.67 |
| `productionDefault--dino_boost--suppression_half` | 83.76 | 77.88 | 99.26 | 87.28 | 80.23 | 99.27 | 88.74 | 14.22 | 59.67 | 20.00 | 53.33 |
| `shippedUnion--dino_boost--bidirectional_any` | 80.15 | 77.53 | 99.80 | 87.27 | 91.40 | 99.81 | 95.42 | 63.16 | 242.33 | 141.33 | 142.00 |
| `productionDefault--dino_keep--bidirectional_half` | 83.76 | 77.66 | 99.38 | 87.19 | 79.88 | 99.39 | 88.57 | 15.19 | 67.00 | 22.00 | 59.67 |
| `productionDefault--dino_global--bidirectional_half` | 83.76 | 77.67 | 99.28 | 87.15 | 79.59 | 99.37 | 88.38 | 15.06 | 69.67 | 22.33 | 62.00 |
| `shippedUnion--dino_boost--bidirectional_half` | 80.15 | 77.44 | 99.63 | 87.14 | 79.71 | 99.64 | 88.57 | 25.13 | 122.00 | 21.33 | 99.33 |
| `shippedUnion--dino_boost--suppression_half` | 80.15 | 77.44 | 99.63 | 87.14 | 79.67 | 99.64 | 88.54 | 24.20 | 115.00 | 21.33 | 93.67 |
| `productionDefault--dino_keep--suppression_half` | 83.76 | 77.66 | 99.26 | 87.14 | 79.80 | 99.27 | 88.47 | 13.96 | 59.33 | 20.67 | 53.00 |
| `compact_keep--uncertain_medium` | 89.97 | 77.96 | 98.72 | 87.11 | 97.83 | 98.82 | 98.32 | 90.62 | 692.00 | 229.00 | 195.67 |
| `productionDefault--dino_boost--suppression_zero` | 83.76 | 77.61 | 99.26 | 87.11 | 77.91 | 99.27 | 87.30 | 10.27 | 50.67 | 11.67 | 47.67 |
| `productionDefault--dino_global--suppression_half` | 83.76 | 77.64 | 99.18 | 87.10 | 79.51 | 99.27 | 88.30 | 13.78 | 61.00 | 21.33 | 55.00 |
| `productionDefault--compact_keep--bidirectional_any` | 83.76 | 77.41 | 99.53 | 87.09 | 91.82 | 99.54 | 95.52 | 56.45 | 196.00 | 151.67 | 125.33 |
| `productionDefault--dino_keep--suppression_zero` | 83.76 | 77.54 | 99.26 | 87.07 | 77.75 | 99.27 | 87.20 | 10.28 | 51.33 | 13.00 | 48.67 |
| `compact_boost--uncertain_medium` | 90.31 | 77.62 | 99.12 | 87.06 | 98.25 | 99.21 | 98.73 | 93.89 | 689.33 | 238.67 | 196.67 |
| `shippedUnion--dino_global--suppression_half` | 80.15 | 77.28 | 99.56 | 87.01 | 79.05 | 99.64 | 88.16 | 24.00 | 118.00 | 23.33 | 95.33 |
| `productionDefault--dino_keep--bidirectional_any` | 83.76 | 77.19 | 99.70 | 87.01 | 92.07 | 99.71 | 95.73 | 60.02 | 201.00 | 158.00 | 135.67 |
| `shippedUnion--compact_keep--bidirectional_any` | 80.15 | 77.13 | 99.76 | 86.99 | 91.11 | 99.76 | 95.24 | 64.45 | 248.33 | 151.00 | 134.33 |
| `shippedUnion--dino_keep--bidirectional_any` | 80.15 | 77.09 | 99.82 | 86.99 | 91.66 | 99.83 | 95.56 | 68.47 | 254.67 | 157.00 | 148.33 |
| `shippedUnion--dino_keep--suppression_half` | 80.15 | 77.19 | 99.63 | 86.99 | 79.20 | 99.64 | 88.25 | 23.99 | 115.00 | 22.33 | 93.67 |
| `dino_keep--uncertain_narrow` | 91.94 | 77.43 | 99.30 | 86.99 | 96.08 | 99.30 | 97.64 | 87.45 | 627.33 | 224.67 | 202.33 |
| `shippedUnion--dino_keep--bidirectional_half` | 80.15 | 77.19 | 99.63 | 86.98 | 79.23 | 99.64 | 88.26 | 24.64 | 119.33 | 22.33 | 97.67 |
| `shippedUnion--dino_global--bidirectional_half` | 80.15 | 77.18 | 99.56 | 86.95 | 79.10 | 99.64 | 88.19 | 24.69 | 123.00 | 24.00 | 98.00 |
| `productionDefault--dino_global--suppression_zero` | 83.76 | 77.19 | 99.22 | 86.83 | 77.69 | 99.27 | 87.16 | 10.49 | 52.67 | 14.67 | 49.33 |
| `compact_keep--uncertain_wide` | 89.97 | 77.01 | 99.40 | 86.78 | 98.44 | 99.41 | 98.92 | 102.65 | 844.67 | 257.67 | 170.33 |
| `productionDefault--compact_boost--suppression_half` | 83.76 | 77.03 | 99.26 | 86.74 | 79.28 | 99.27 | 88.15 | 16.82 | 74.67 | 38.33 | 65.33 |
| `productionDefault--compact_boost--bidirectional_half` | 83.76 | 77.01 | 99.26 | 86.73 | 79.31 | 99.27 | 88.17 | 17.43 | 79.67 | 38.33 | 68.00 |
| `dino_keep--uncertain_medium` | 91.94 | 76.93 | 99.41 | 86.72 | 97.09 | 99.41 | 98.23 | 94.72 | 699.33 | 235.33 | 191.00 |
| `shippedUnion--dino_keep--suppression_zero` | 80.15 | 76.75 | 99.63 | 86.70 | 76.89 | 99.64 | 86.80 | 19.69 | 103.67 | 14.67 | 88.67 |
| `shippedUnion--dino_boost--suppression_zero` | 80.15 | 76.71 | 99.63 | 86.68 | 76.94 | 99.64 | 86.83 | 19.49 | 102.33 | 13.00 | 87.67 |
| `compact_boost--uncertain_wide` | 90.31 | 76.78 | 99.42 | 86.65 | 98.65 | 99.43 | 99.04 | 103.86 | 837.67 | 261.33 | 169.67 |
| `shippedUnion--compact_boost--suppression_half` | 80.15 | 76.63 | 99.63 | 86.63 | 78.80 | 99.64 | 88.00 | 26.91 | 133.00 | 41.33 | 99.00 |
| `productionDefault--compact_boost--suppression_zero` | 83.76 | 76.85 | 99.27 | 86.63 | 77.45 | 99.27 | 87.01 | 12.86 | 65.00 | 29.00 | 57.67 |
| `shippedUnion--compact_boost--bidirectional_half` | 80.15 | 76.56 | 99.63 | 86.59 | 78.82 | 99.64 | 88.01 | 27.36 | 136.00 | 41.67 | 100.33 |
| `dino_boost--uncertain_narrow` | 92.51 | 76.86 | 99.10 | 86.54 | 97.08 | 99.10 | 98.08 | 89.18 | 642.00 | 228.67 | 214.33 |
| `dino_keep--uncertain_wide` | 91.94 | 76.53 | 99.59 | 86.54 | 98.76 | 99.59 | 99.17 | 104.63 | 818.33 | 260.00 | 171.67 |
| `productionDefault--compact_keep--suppression_half` | 83.76 | 76.54 | 99.27 | 86.43 | 78.48 | 99.27 | 87.65 | 14.97 | 66.67 | 33.00 | 58.00 |
| `productionDefault--compact_keep--bidirectional_half` | 83.76 | 76.45 | 99.32 | 86.40 | 78.61 | 99.33 | 87.76 | 16.44 | 76.00 | 34.67 | 64.33 |
| `shippedUnion--compact_boost--suppression_zero` | 80.15 | 76.24 | 99.64 | 86.38 | 76.75 | 99.64 | 86.71 | 22.51 | 121.33 | 32.00 | 91.67 |
| `shippedUnion--dino_global--suppression_zero` | 80.15 | 76.18 | 99.60 | 86.32 | 76.61 | 99.64 | 86.62 | 19.59 | 104.00 | 16.67 | 87.33 |
| `productionDefault--compact_keep--suppression_zero` | 83.76 | 76.35 | 99.27 | 86.31 | 76.90 | 99.27 | 86.66 | 11.47 | 58.33 | 25.33 | 51.33 |
| `shippedUnion--compact_keep--suppression_half` | 80.15 | 76.02 | 99.64 | 86.24 | 77.87 | 99.64 | 87.42 | 24.86 | 123.00 | 35.67 | 95.00 |
| `dino_boost--uncertain_medium` | 92.51 | 76.30 | 99.20 | 86.23 | 97.75 | 99.22 | 98.48 | 95.69 | 712.67 | 235.67 | 197.00 |
| `shippedUnion--compact_keep--bidirectional_half` | 80.15 | 75.83 | 99.67 | 86.13 | 77.96 | 99.68 | 87.49 | 25.94 | 129.67 | 37.33 | 98.33 |
| `dino_boost--uncertain_wide` | 92.51 | 75.88 | 99.59 | 86.11 | 98.65 | 99.59 | 99.12 | 103.27 | 821.33 | 248.67 | 177.67 |
| `shippedUnion--compact_keep--suppression_zero` | 80.15 | 75.41 | 99.64 | 85.84 | 75.87 | 99.64 | 86.15 | 20.54 | 111.00 | 28.00 | 87.33 |
| `dino_global--all_candidates` | 93.07 | 74.84 | 100.00 | 85.61 | 100.00 | 100.00 | 100.00 | 137.83 | 1851.67 | 322.00 | 11.00 |
| `compact_keep--all_candidates` | 89.97 | 74.56 | 100.00 | 85.43 | 100.00 | 100.00 | 100.00 | 137.83 | 1869.00 | 322.00 | 11.00 |
| `compact_boost--all_candidates` | 90.31 | 74.41 | 100.00 | 85.33 | 100.00 | 100.00 | 100.00 | 137.83 | 1844.00 | 322.00 | 11.00 |
| `dino_boost--all_candidates` | 92.51 | 73.70 | 100.00 | 84.85 | 100.00 | 100.00 | 100.00 | 137.83 | 1872.67 | 322.00 | 11.00 |
| `dino_keep--all_candidates` | 91.94 | 73.21 | 100.00 | 84.53 | 100.00 | 100.00 | 100.00 | 137.83 | 1872.00 | 322.00 | 11.00 |
| `refitUnion--dino_boost--bidirectional_any` | 73.52 | 72.84 | 99.70 | 84.18 | 92.80 | 99.77 | 96.16 | 81.85 | 293.00 | 169.67 | 131.00 |
| `refitUnion--dino_global--bidirectional_any` | 73.52 | 72.67 | 99.60 | 84.03 | 92.77 | 99.71 | 96.11 | 81.99 | 294.67 | 171.33 | 131.00 |
| `refitUnion--compact_boost--bidirectional_any` | 73.52 | 72.85 | 99.01 | 83.94 | 93.56 | 99.37 | 96.38 | 84.20 | 309.00 | 181.33 | 137.00 |
| `refitUnion--dino_keep--bidirectional_any` | 73.52 | 72.45 | 99.48 | 83.84 | 92.65 | 99.65 | 96.02 | 83.24 | 298.33 | 176.00 | 134.67 |
| `productionDefault--compact_boost--guarded_trim` | 83.76 | 72.55 | 99.27 | 83.83 | 72.54 | 99.27 | 83.83 | 0.17 | 1.00 | 0.00 | 1.00 |
| `productionDefault--compact_keep--guarded_trim` | 83.76 | 72.55 | 99.27 | 83.83 | 72.54 | 99.27 | 83.83 | 0.17 | 1.00 | 0.00 | 1.00 |
| `productionDefault--dino_boost--guarded_trim` | 83.76 | 72.55 | 99.27 | 83.83 | 72.54 | 99.27 | 83.83 | 0.17 | 1.00 | 0.00 | 1.00 |
| `productionDefault--dino_keep--guarded_trim` | 83.76 | 72.55 | 99.27 | 83.83 | 72.54 | 99.27 | 83.83 | 0.17 | 1.00 | 0.00 | 1.00 |
| `productionDefault--dino_global--guarded_trim` | 83.76 | 72.57 | 99.19 | 83.82 | 72.58 | 99.27 | 83.85 | 0.29 | 1.67 | 0.00 | 1.67 |
| `refitUnion--compact_keep--bidirectional_any` | 73.52 | 72.59 | 99.12 | 83.81 | 93.02 | 99.46 | 96.14 | 82.54 | 301.33 | 175.00 | 134.00 |
| `refitUnion--compact_boost--suppression_any` | 73.52 | 73.00 | 97.95 | 83.66 | 93.10 | 98.32 | 95.64 | 79.43 | 296.00 | 170.67 | 137.00 |
| `refitUnion--dino_boost--suppression_any` | 73.52 | 72.88 | 97.95 | 83.57 | 91.85 | 98.32 | 94.97 | 74.09 | 271.67 | 147.67 | 131.33 |
| `refitUnion--dino_global--suppression_any` | 73.52 | 72.79 | 97.99 | 83.53 | 91.98 | 98.32 | 95.04 | 74.57 | 274.67 | 150.67 | 131.33 |
| `refitUnion--compact_keep--suppression_any` | 73.52 | 72.80 | 97.95 | 83.52 | 92.30 | 98.32 | 95.21 | 77.02 | 285.67 | 161.67 | 134.33 |
| `refitUnion--dino_keep--suppression_any` | 73.52 | 72.72 | 98.01 | 83.49 | 91.75 | 98.32 | 94.91 | 76.15 | 277.67 | 155.00 | 135.33 |
| `refitUnion--dino_keep--bidirectional_half` | 73.52 | 72.10 | 98.62 | 83.30 | 75.40 | 98.80 | 85.51 | 36.53 | 163.00 | 33.33 | 123.00 |
| `refitUnion--dino_boost--bidirectional_half` | 73.52 | 71.99 | 98.61 | 83.22 | 75.54 | 98.76 | 85.60 | 36.76 | 162.00 | 34.33 | 120.67 |
| `refitUnion--dino_keep--suppression_half` | 73.52 | 72.20 | 98.07 | 83.17 | 75.23 | 98.32 | 85.22 | 35.55 | 157.67 | 29.33 | 121.67 |
| `refitUnion--dino_boost--suppression_half` | 73.52 | 72.06 | 98.09 | 83.08 | 75.41 | 98.32 | 85.35 | 35.93 | 157.00 | 30.33 | 120.67 |
| `refitUnion--dino_global--bidirectional_half` | 73.52 | 71.83 | 98.48 | 83.07 | 75.80 | 98.68 | 85.73 | 37.51 | 165.67 | 37.67 | 120.33 |
| `refitUnion--dino_global--suppression_half` | 73.52 | 71.92 | 98.06 | 82.98 | 75.69 | 98.32 | 85.52 | 36.81 | 161.67 | 34.67 | 121.33 |
| `refitUnion--compact_boost--bidirectional_half` | 73.52 | 71.44 | 98.38 | 82.77 | 75.56 | 98.74 | 85.60 | 40.04 | 178.00 | 50.33 | 127.33 |
| `refitUnion--compact_boost--suppression_half` | 73.52 | 71.63 | 97.95 | 82.75 | 75.44 | 98.32 | 85.37 | 39.28 | 174.33 | 48.33 | 127.33 |
| `refitUnion--compact_keep--bidirectional_half` | 73.52 | 70.96 | 98.46 | 82.48 | 74.74 | 98.81 | 85.10 | 38.38 | 170.67 | 47.33 | 122.67 |
| `refitUnion--compact_keep--suppression_half` | 73.52 | 71.04 | 97.95 | 82.35 | 74.62 | 98.32 | 84.84 | 37.57 | 166.00 | 44.67 | 122.33 |
| `refitUnion--dino_global--suppression_zero` | 73.52 | 70.33 | 98.19 | 81.96 | 70.37 | 98.32 | 82.03 | 26.27 | 134.00 | 15.00 | 110.00 |
| `refitUnion--dino_keep--suppression_zero` | 73.52 | 70.19 | 98.15 | 81.85 | 70.10 | 98.32 | 81.84 | 25.77 | 132.00 | 13.00 | 111.00 |
| `refitUnion--dino_boost--suppression_zero` | 73.52 | 70.06 | 98.19 | 81.78 | 70.06 | 98.32 | 81.82 | 25.73 | 131.67 | 12.33 | 110.67 |
| `shippedUnion--compact_boost--guarded_trim` | 80.15 | 69.29 | 99.64 | 81.74 | 69.20 | 99.64 | 81.68 | 4.46 | 26.00 | 1.00 | 23.00 |
| `shippedUnion--dino_keep--guarded_trim` | 80.15 | 69.29 | 99.64 | 81.74 | 69.20 | 99.64 | 81.68 | 4.46 | 26.00 | 1.00 | 23.00 |
| `shippedUnion--dino_boost--guarded_trim` | 80.15 | 69.26 | 99.64 | 81.72 | 69.17 | 99.64 | 81.66 | 4.40 | 25.67 | 1.00 | 22.67 |
| `shippedUnion--dino_global--guarded_trim` | 80.15 | 69.28 | 99.56 | 81.70 | 69.21 | 99.64 | 81.69 | 4.52 | 26.33 | 1.00 | 23.33 |
| `shippedUnion--compact_keep--guarded_trim` | 80.15 | 69.23 | 99.64 | 81.70 | 69.15 | 99.64 | 81.64 | 4.34 | 25.33 | 1.00 | 22.33 |
| `refitUnion--compact_boost--suppression_zero` | 73.52 | 69.77 | 97.95 | 81.49 | 70.17 | 98.32 | 81.89 | 28.66 | 151.67 | 28.67 | 118.33 |
| `refitUnion--compact_keep--suppression_zero` | 73.52 | 68.90 | 97.95 | 80.89 | 69.22 | 98.32 | 81.24 | 26.37 | 140.33 | 24.33 | 111.67 |
| `refitUnion--dino_keep--guarded_trim` | 73.52 | 61.11 | 98.09 | 75.30 | 61.02 | 98.32 | 75.30 | 6.20 | 34.67 | 3.33 | 28.67 |
| `refitUnion--dino_global--guarded_trim` | 73.52 | 61.11 | 98.05 | 75.30 | 61.06 | 98.32 | 75.34 | 6.45 | 36.00 | 4.33 | 28.67 |
| `refitUnion--compact_boost--guarded_trim` | 73.52 | 61.11 | 98.00 | 75.28 | 61.13 | 98.32 | 75.39 | 7.87 | 42.33 | 10.33 | 33.00 |
| `refitUnion--compact_keep--guarded_trim` | 73.52 | 61.11 | 98.00 | 75.28 | 61.13 | 98.32 | 75.39 | 7.69 | 42.00 | 9.67 | 32.67 |
| `refitUnion--dino_boost--guarded_trim` | 73.52 | 61.11 | 98.00 | 75.28 | 61.06 | 98.32 | 75.33 | 6.40 | 36.00 | 4.00 | 28.33 |

## Decision composition and missed-rally guardrails

| Policy | Positive decisions | Negative decisions | Kept | Dropped | Mixed | True rallies in playback | Binary complete losses | Binary partial losses | Boundary complete losses | Boundary partial losses | Binary event F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dino_global--all_positive` | 336.67 | 0.00 | 315.00 | 21.67 | 205.33 | 304.67 | 18.00 | 22.00 | 17.33 | 22.33 | 79.76 |
| `dino_boost--all_positive` | 345.00 | 0.00 | 318.00 | 27.00 | 194.00 | 308.33 | 15.00 | 28.67 | 13.67 | 29.33 | 77.29 |
| `dino_global--positive_uncertain` | 26.67 | 0.00 | 15.00 | 11.67 | 8.00 | 19.00 | 17.67 | 22.00 | 17.33 | 22.33 | 78.53 |
| `dino_keep--all_positive` | 345.00 | 0.00 | 324.67 | 20.33 | 195.67 | 307.00 | 15.67 | 34.00 | 15.00 | 34.00 | 76.08 |
| `dino_boost--positive_uncertain` | 18.33 | 0.00 | 7.00 | 11.33 | 2.33 | 9.00 | 14.00 | 29.00 | 13.67 | 29.33 | 75.44 |
| `dino_keep--positive_uncertain` | 41.67 | 0.00 | 30.00 | 11.67 | 15.67 | 33.67 | 15.00 | 34.00 | 15.00 | 34.00 | 75.07 |
| `compact_keep--all_positive` | 341.67 | 0.00 | 308.67 | 33.00 | 198.00 | 295.00 | 29.33 | 36.67 | 28.33 | 37.33 | 72.21 |
| `compact_boost--all_positive` | 324.00 | 0.00 | 299.67 | 24.33 | 192.00 | 290.33 | 34.00 | 41.67 | 33.00 | 42.00 | 72.17 |
| `compact_keep--positive_uncertain` | 47.67 | 0.00 | 36.00 | 11.67 | 19.00 | 38.00 | 28.67 | 37.00 | 28.33 | 37.33 | 69.85 |
| `compact_boost--positive_uncertain` | 27.33 | 0.00 | 22.00 | 5.33 | 12.33 | 23.67 | 33.33 | 42.00 | 33.00 | 42.00 | 70.03 |
| `dino_global--uncertain_narrow` | 13.00 | 549.33 | 304.00 | 258.33 | 239.00 | 289.33 | 4.67 | 5.67 | 4.67 | 5.67 | 76.70 |
| `dino_global--uncertain_medium` | 26.67 | 597.67 | 331.33 | 293.00 | 262.33 | 299.33 | 4.00 | 4.67 | 4.00 | 4.67 | 76.94 |
| `dino_global--uncertain_wide` | 65.67 | 676.33 | 389.33 | 352.67 | 301.33 | 314.67 | 2.00 | 3.33 | 2.00 | 3.33 | 77.08 |
| `shippedUnion--dino_boost--suppression_any` | 219.67 | 0.00 | 118.67 | 101.00 | 116.67 | 143.33 | 0.00 | 7.00 | 0.00 | 6.00 | 70.99 |
| `productionDefault--dino_global--suppression_any` | 165.67 | 0.00 | 121.00 | 44.67 | 120.33 | 131.67 | 3.00 | 8.33 | 3.00 | 7.00 | 71.10 |
| `productionDefault--dino_global--bidirectional_any` | 180.00 | 8.67 | 136.33 | 52.33 | 135.00 | 148.00 | 2.00 | 4.00 | 2.00 | 3.33 | 71.88 |
| `productionDefault--dino_boost--suppression_any` | 161.00 | 0.00 | 117.33 | 43.67 | 115.33 | 127.33 | 3.00 | 8.00 | 3.00 | 7.00 | 70.99 |
| `shippedUnion--dino_keep--suppression_any` | 230.00 | 0.00 | 130.33 | 99.67 | 128.67 | 153.00 | 0.00 | 6.67 | 0.00 | 6.00 | 70.84 |
| `shippedUnion--dino_global--suppression_any` | 223.33 | 0.00 | 123.00 | 100.33 | 122.33 | 146.67 | 0.00 | 7.33 | 0.00 | 6.00 | 70.92 |
| `compact_boost--uncertain_narrow` | 7.33 | 589.67 | 327.33 | 269.67 | 243.67 | 287.00 | 9.67 | 8.00 | 9.33 | 7.00 | 73.54 |
| `productionDefault--dino_keep--suppression_any` | 171.00 | 0.00 | 128.67 | 42.33 | 127.00 | 137.67 | 3.00 | 7.67 | 3.00 | 7.00 | 70.85 |
| `shippedUnion--compact_boost--suppression_any` | 243.00 | 0.00 | 143.00 | 100.00 | 138.33 | 163.33 | 0.00 | 7.00 | 0.00 | 6.00 | 70.88 |
| `productionDefault--compact_boost--suppression_any` | 182.33 | 0.00 | 140.00 | 42.33 | 135.33 | 149.00 | 3.00 | 8.00 | 3.00 | 7.00 | 70.84 |
| `shippedUnion--dino_global--bidirectional_any` | 235.33 | 5.00 | 135.67 | 104.67 | 135.00 | 158.33 | 0.00 | 4.00 | 0.00 | 3.33 | 71.70 |
| `compact_keep--uncertain_narrow` | 27.33 | 581.33 | 328.00 | 280.67 | 237.00 | 281.67 | 10.00 | 8.33 | 9.67 | 7.33 | 73.07 |
| `productionDefault--dino_boost--bidirectional_half` | 59.67 | 12.33 | 20.67 | 51.33 | 18.67 | 30.00 | 1.33 | 8.00 | 1.33 | 7.00 | 70.79 |
| `productionDefault--dino_boost--bidirectional_any` | 178.00 | 13.00 | 136.33 | 54.67 | 133.33 | 149.00 | 1.33 | 5.00 | 1.33 | 4.00 | 71.15 |
| `productionDefault--compact_keep--suppression_any` | 172.67 | 0.00 | 131.67 | 41.00 | 127.67 | 140.00 | 3.00 | 7.33 | 3.00 | 7.00 | 70.70 |
| `productionDefault--compact_boost--bidirectional_any` | 193.33 | 5.33 | 150.67 | 48.00 | 147.00 | 161.00 | 3.00 | 5.33 | 3.00 | 4.33 | 70.03 |
| `shippedUnion--compact_boost--bidirectional_any` | 251.00 | 3.33 | 151.67 | 102.67 | 148.00 | 172.33 | 0.00 | 5.33 | 0.00 | 4.33 | 70.28 |
| `shippedUnion--compact_keep--suppression_any` | 231.67 | 0.00 | 134.33 | 97.33 | 130.33 | 155.00 | 0.00 | 6.33 | 0.00 | 6.00 | 70.59 |
| `productionDefault--dino_boost--suppression_half` | 59.67 | 0.00 | 19.00 | 40.67 | 18.00 | 25.33 | 3.00 | 8.00 | 3.00 | 7.00 | 70.66 |
| `shippedUnion--dino_boost--bidirectional_any` | 234.67 | 7.67 | 134.67 | 107.67 | 132.67 | 160.33 | 0.00 | 5.00 | 0.00 | 4.00 | 71.12 |
| `productionDefault--dino_keep--bidirectional_half` | 59.33 | 7.67 | 21.33 | 45.67 | 19.67 | 28.00 | 1.67 | 7.67 | 1.67 | 7.00 | 70.68 |
| `productionDefault--dino_global--bidirectional_half` | 61.00 | 8.67 | 21.67 | 48.00 | 20.67 | 29.67 | 2.00 | 8.33 | 2.00 | 7.00 | 70.72 |
| `shippedUnion--dino_boost--bidirectional_half` | 115.00 | 7.00 | 20.33 | 101.67 | 19.33 | 45.33 | 0.00 | 7.00 | 0.00 | 6.00 | 70.30 |
| `shippedUnion--dino_boost--suppression_half` | 115.00 | 0.00 | 20.33 | 94.67 | 19.33 | 43.33 | 0.00 | 7.00 | 0.00 | 6.00 | 70.30 |
| `productionDefault--dino_keep--suppression_half` | 59.33 | 0.00 | 20.00 | 39.33 | 19.00 | 24.67 | 3.00 | 7.67 | 3.00 | 7.00 | 70.52 |
| `compact_keep--uncertain_medium` | 47.67 | 644.33 | 368.00 | 324.00 | 266.67 | 299.67 | 8.67 | 6.67 | 8.00 | 6.33 | 73.30 |
| `productionDefault--dino_boost--suppression_zero` | 50.67 | 0.00 | 11.67 | 39.00 | 11.00 | 16.00 | 3.00 | 8.00 | 3.00 | 7.00 | 70.48 |
| `productionDefault--dino_global--suppression_half` | 61.00 | 0.00 | 20.67 | 40.33 | 20.33 | 25.67 | 3.00 | 8.33 | 3.00 | 7.00 | 70.63 |
| `productionDefault--compact_keep--bidirectional_any` | 185.67 | 10.33 | 145.00 | 51.00 | 141.33 | 155.67 | 2.67 | 5.33 | 2.67 | 4.67 | 69.64 |
| `productionDefault--dino_keep--suppression_zero` | 51.33 | 0.00 | 13.00 | 38.33 | 12.00 | 16.67 | 3.00 | 7.67 | 3.00 | 7.00 | 70.41 |
| `compact_boost--uncertain_medium` | 27.33 | 662.00 | 371.33 | 318.00 | 279.00 | 306.00 | 6.67 | 5.00 | 6.00 | 4.67 | 73.62 |
| `shippedUnion--dino_global--suppression_half` | 118.00 | 0.00 | 22.67 | 95.33 | 22.33 | 45.67 | 0.00 | 7.33 | 0.00 | 6.00 | 70.37 |
| `productionDefault--dino_keep--bidirectional_any` | 192.67 | 8.33 | 151.67 | 49.33 | 149.67 | 162.67 | 1.67 | 4.00 | 1.67 | 3.33 | 70.94 |
| `shippedUnion--compact_keep--bidirectional_any` | 240.33 | 8.00 | 144.33 | 104.00 | 140.67 | 167.00 | 0.00 | 5.00 | 0.00 | 4.67 | 69.85 |
| `shippedUnion--dino_keep--bidirectional_any` | 249.67 | 5.00 | 150.67 | 104.00 | 149.33 | 173.33 | 0.00 | 4.00 | 0.00 | 3.33 | 70.85 |
| `shippedUnion--dino_keep--suppression_half` | 115.00 | 0.00 | 21.67 | 93.33 | 20.67 | 43.00 | 0.00 | 6.67 | 0.00 | 6.00 | 70.16 |
| `dino_keep--uncertain_narrow` | 26.67 | 600.67 | 357.67 | 269.67 | 265.33 | 295.33 | 4.33 | 8.00 | 4.33 | 8.00 | 75.83 |
| `shippedUnion--dino_keep--bidirectional_half` | 115.00 | 4.33 | 21.67 | 97.67 | 20.67 | 44.67 | 0.00 | 6.67 | 0.00 | 6.00 | 70.16 |
| `shippedUnion--dino_global--bidirectional_half` | 118.00 | 5.00 | 23.33 | 99.67 | 23.00 | 48.00 | 0.00 | 7.33 | 0.00 | 6.00 | 70.17 |
| `productionDefault--dino_global--suppression_zero` | 52.67 | 0.00 | 14.67 | 38.00 | 14.33 | 18.67 | 3.00 | 8.00 | 3.00 | 7.00 | 70.37 |
| `compact_keep--uncertain_wide` | 107.33 | 737.33 | 460.67 | 384.00 | 335.33 | 313.67 | 4.67 | 2.33 | 4.67 | 2.00 | 73.69 |
| `productionDefault--compact_boost--suppression_half` | 74.67 | 0.00 | 37.67 | 37.00 | 35.67 | 41.67 | 3.00 | 8.00 | 3.00 | 7.00 | 70.26 |
| `productionDefault--compact_boost--bidirectional_half` | 74.67 | 5.00 | 37.67 | 42.00 | 35.67 | 42.33 | 3.00 | 8.00 | 3.00 | 7.00 | 70.26 |
| `dino_keep--uncertain_medium` | 41.67 | 657.67 | 387.00 | 312.33 | 288.67 | 305.33 | 3.67 | 6.33 | 3.67 | 6.33 | 75.80 |
| `shippedUnion--dino_keep--suppression_zero` | 103.67 | 0.00 | 14.67 | 89.00 | 13.67 | 32.33 | 0.00 | 6.67 | 0.00 | 6.00 | 69.70 |
| `shippedUnion--dino_boost--suppression_zero` | 102.33 | 0.00 | 13.00 | 89.33 | 12.33 | 31.33 | 0.00 | 7.00 | 0.00 | 6.00 | 69.73 |
| `compact_boost--uncertain_wide` | 88.00 | 749.67 | 456.00 | 381.67 | 340.33 | 315.00 | 4.33 | 2.67 | 4.33 | 2.33 | 72.76 |
| `shippedUnion--compact_boost--suppression_half` | 133.00 | 0.00 | 40.67 | 92.33 | 38.67 | 58.00 | 0.00 | 7.00 | 0.00 | 6.00 | 70.05 |
| `productionDefault--compact_boost--suppression_zero` | 65.00 | 0.00 | 29.00 | 36.00 | 27.00 | 32.33 | 3.00 | 7.33 | 3.00 | 7.00 | 70.15 |
| `shippedUnion--compact_boost--bidirectional_half` | 133.00 | 3.00 | 41.00 | 95.00 | 39.00 | 58.67 | 0.00 | 7.00 | 0.00 | 6.00 | 69.95 |
| `dino_boost--uncertain_narrow` | 9.00 | 633.00 | 345.67 | 296.33 | 274.00 | 308.00 | 4.33 | 9.00 | 4.33 | 9.00 | 72.06 |
| `dino_keep--uncertain_wide` | 84.00 | 734.33 | 445.00 | 373.33 | 332.33 | 318.00 | 2.00 | 4.33 | 2.00 | 4.33 | 75.70 |
| `productionDefault--compact_keep--suppression_half` | 66.67 | 0.00 | 32.67 | 34.00 | 30.33 | 36.33 | 3.00 | 7.33 | 3.00 | 7.00 | 69.94 |
| `productionDefault--compact_keep--bidirectional_half` | 67.33 | 8.67 | 34.33 | 41.67 | 31.33 | 38.33 | 2.67 | 7.33 | 2.67 | 6.67 | 69.77 |
| `shippedUnion--compact_boost--suppression_zero` | 121.33 | 0.00 | 32.00 | 89.33 | 30.00 | 47.33 | 0.00 | 6.33 | 0.00 | 6.00 | 69.73 |
| `shippedUnion--dino_global--suppression_zero` | 104.00 | 0.00 | 16.67 | 87.33 | 16.33 | 35.33 | 0.00 | 7.00 | 0.00 | 6.00 | 69.52 |
| `productionDefault--compact_keep--suppression_zero` | 58.33 | 0.00 | 25.33 | 33.00 | 23.00 | 28.67 | 3.00 | 7.00 | 3.00 | 7.00 | 69.83 |
| `shippedUnion--compact_keep--suppression_half` | 123.00 | 0.00 | 35.33 | 87.67 | 33.00 | 53.67 | 0.00 | 6.33 | 0.00 | 6.00 | 69.56 |
| `dino_boost--uncertain_medium` | 18.33 | 694.33 | 370.33 | 342.33 | 294.33 | 313.00 | 3.67 | 6.67 | 3.33 | 7.00 | 72.24 |
| `shippedUnion--compact_keep--bidirectional_half` | 123.33 | 6.33 | 37.00 | 92.67 | 34.00 | 55.67 | 0.00 | 6.00 | 0.00 | 5.67 | 69.29 |
| `dino_boost--uncertain_wide` | 53.67 | 767.67 | 425.33 | 396.00 | 329.33 | 318.67 | 2.33 | 4.00 | 2.33 | 4.00 | 73.70 |
| `shippedUnion--compact_keep--suppression_zero` | 111.00 | 0.00 | 28.00 | 83.00 | 25.67 | 44.00 | 0.00 | 6.00 | 0.00 | 6.00 | 69.07 |
| `dino_global--all_candidates` | 336.67 | 1515.00 | 732.67 | 1119.00 | 556.33 | 322.00 | 0.00 | 0.00 | 0.00 | 0.00 | 74.09 |
| `compact_keep--all_candidates` | 341.67 | 1527.33 | 758.00 | 1111.00 | 555.67 | 322.00 | 0.00 | 0.00 | 0.00 | 0.00 | 71.41 |
| `compact_boost--all_candidates` | 324.00 | 1520.00 | 754.67 | 1089.33 | 560.00 | 322.00 | 0.00 | 0.00 | 0.00 | 0.00 | 70.93 |
| `dino_boost--all_candidates` | 345.00 | 1527.67 | 762.67 | 1110.00 | 560.67 | 322.00 | 0.00 | 0.00 | 0.00 | 0.00 | 72.19 |
| `dino_keep--all_candidates` | 345.00 | 1527.00 | 774.67 | 1097.33 | 554.00 | 322.00 | 0.00 | 0.00 | 0.00 | 0.00 | 72.61 |
| `refitUnion--dino_boost--bidirectional_any` | 286.67 | 6.33 | 153.67 | 139.33 | 148.33 | 194.67 | 1.33 | 8.00 | 0.33 | 6.67 | 63.09 |
| `refitUnion--dino_global--bidirectional_any` | 289.67 | 5.00 | 156.67 | 138.00 | 152.00 | 197.00 | 2.33 | 8.33 | 1.00 | 7.33 | 62.75 |
| `refitUnion--compact_boost--bidirectional_any` | 304.67 | 4.33 | 168.67 | 140.33 | 156.67 | 209.00 | 4.67 | 11.33 | 1.00 | 11.00 | 61.36 |
| `refitUnion--dino_keep--bidirectional_any` | 291.67 | 6.67 | 161.00 | 137.33 | 156.33 | 200.00 | 2.33 | 9.67 | 0.67 | 8.33 | 62.54 |
| `productionDefault--compact_boost--guarded_trim` | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 3.00 | 7.00 | 3.00 | 7.00 | 66.57 |
| `productionDefault--compact_keep--guarded_trim` | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 3.00 | 7.00 | 3.00 | 7.00 | 66.57 |
| `productionDefault--dino_boost--guarded_trim` | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 3.00 | 7.00 | 3.00 | 7.00 | 66.57 |
| `productionDefault--dino_keep--guarded_trim` | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 3.00 | 7.00 | 3.00 | 7.00 | 66.57 |
| `productionDefault--dino_global--guarded_trim` | 1.67 | 0.00 | 0.00 | 1.67 | 0.00 | 0.67 | 3.00 | 7.67 | 3.00 | 7.00 | 66.63 |
| `refitUnion--compact_keep--bidirectional_any` | 295.33 | 6.00 | 162.33 | 139.00 | 151.33 | 203.00 | 4.00 | 11.67 | 1.00 | 10.67 | 60.82 |
| `refitUnion--compact_boost--suppression_any` | 296.00 | 0.00 | 156.00 | 140.00 | 147.00 | 199.00 | 5.00 | 21.00 | 1.00 | 21.00 | 59.44 |
| `refitUnion--dino_boost--suppression_any` | 271.67 | 0.00 | 133.33 | 138.33 | 131.00 | 178.00 | 5.00 | 21.00 | 1.00 | 21.00 | 59.29 |
| `refitUnion--dino_global--suppression_any` | 274.67 | 0.00 | 136.67 | 138.00 | 134.33 | 180.00 | 5.00 | 20.67 | 1.00 | 21.00 | 59.26 |
| `refitUnion--compact_keep--suppression_any` | 285.67 | 0.00 | 147.33 | 138.33 | 139.67 | 192.00 | 5.00 | 21.00 | 1.00 | 21.00 | 59.29 |
| `refitUnion--dino_keep--suppression_any` | 277.67 | 0.00 | 141.33 | 136.33 | 139.00 | 184.00 | 4.67 | 21.00 | 1.00 | 21.00 | 59.11 |
| `refitUnion--dino_keep--bidirectional_half` | 158.33 | 4.67 | 29.67 | 133.33 | 26.67 | 74.67 | 2.67 | 19.67 | 0.67 | 18.67 | 59.02 |
| `refitUnion--dino_boost--bidirectional_half` | 157.33 | 4.67 | 29.33 | 132.67 | 27.33 | 72.67 | 2.33 | 19.33 | 0.33 | 18.67 | 58.85 |
| `refitUnion--dino_keep--suppression_half` | 157.67 | 0.00 | 25.67 | 132.00 | 25.00 | 71.67 | 4.00 | 21.33 | 1.00 | 21.00 | 58.72 |
| `refitUnion--dino_boost--suppression_half` | 157.00 | 0.00 | 25.33 | 131.67 | 25.00 | 71.33 | 5.00 | 20.00 | 1.00 | 21.00 | 58.69 |
| `refitUnion--dino_global--bidirectional_half` | 162.00 | 3.67 | 33.67 | 132.00 | 32.33 | 77.67 | 2.67 | 20.67 | 1.00 | 19.33 | 58.95 |
| `refitUnion--dino_global--suppression_half` | 161.67 | 0.00 | 30.67 | 131.00 | 30.33 | 75.67 | 4.00 | 21.33 | 1.00 | 21.00 | 58.63 |
| `refitUnion--compact_boost--bidirectional_half` | 175.00 | 3.00 | 45.67 | 132.33 | 37.00 | 90.00 | 4.67 | 19.67 | 1.00 | 19.33 | 58.55 |
| `refitUnion--compact_boost--suppression_half` | 174.33 | 0.00 | 42.67 | 131.67 | 36.00 | 89.00 | 5.00 | 21.00 | 1.00 | 21.00 | 58.69 |
| `refitUnion--compact_keep--bidirectional_half` | 166.33 | 4.33 | 42.33 | 128.33 | 34.33 | 86.00 | 4.00 | 19.67 | 1.00 | 18.67 | 58.15 |
| `refitUnion--compact_keep--suppression_half` | 166.00 | 0.00 | 38.67 | 127.33 | 33.33 | 85.00 | 5.00 | 21.00 | 1.00 | 21.00 | 58.30 |
| `refitUnion--dino_global--suppression_zero` | 134.00 | 0.00 | 15.00 | 119.00 | 14.67 | 47.33 | 2.33 | 22.33 | 1.00 | 21.00 | 57.57 |
| `refitUnion--dino_keep--suppression_zero` | 132.00 | 0.00 | 13.00 | 119.00 | 12.33 | 46.33 | 2.33 | 22.67 | 1.00 | 21.00 | 57.58 |
| `refitUnion--dino_boost--suppression_zero` | 131.67 | 0.00 | 12.33 | 119.33 | 12.00 | 46.67 | 2.00 | 23.00 | 1.00 | 21.00 | 57.60 |
| `shippedUnion--compact_boost--guarded_trim` | 26.00 | 0.00 | 1.00 | 25.00 | 1.00 | 2.00 | 0.00 | 6.00 | 0.00 | 6.00 | 63.54 |
| `shippedUnion--dino_keep--guarded_trim` | 26.00 | 0.00 | 1.00 | 25.00 | 1.00 | 2.00 | 0.00 | 6.00 | 0.00 | 6.00 | 63.54 |
| `shippedUnion--dino_boost--guarded_trim` | 25.67 | 0.00 | 1.00 | 24.67 | 1.00 | 2.00 | 0.00 | 6.00 | 0.00 | 6.00 | 63.51 |
| `shippedUnion--dino_global--guarded_trim` | 26.33 | 0.00 | 1.00 | 25.33 | 1.00 | 2.67 | 0.00 | 6.67 | 0.00 | 6.00 | 63.57 |
| `shippedUnion--compact_keep--guarded_trim` | 25.33 | 0.00 | 1.00 | 24.33 | 1.00 | 2.00 | 0.00 | 6.00 | 0.00 | 6.00 | 63.48 |
| `refitUnion--compact_boost--suppression_zero` | 151.67 | 0.00 | 28.67 | 123.00 | 23.00 | 63.67 | 5.00 | 21.00 | 1.00 | 21.00 | 57.92 |
| `refitUnion--compact_keep--suppression_zero` | 140.33 | 0.00 | 24.33 | 116.00 | 20.33 | 58.33 | 5.00 | 21.00 | 1.00 | 21.00 | 57.32 |
| `refitUnion--dino_keep--guarded_trim` | 34.67 | 0.00 | 3.33 | 31.33 | 3.33 | 5.67 | 2.67 | 21.67 | 1.00 | 21.00 | 50.88 |
| `refitUnion--dino_global--guarded_trim` | 36.00 | 0.00 | 4.33 | 31.67 | 4.33 | 6.33 | 3.00 | 21.67 | 1.00 | 21.00 | 50.91 |
| `refitUnion--compact_boost--guarded_trim` | 42.33 | 0.00 | 10.33 | 32.00 | 9.33 | 12.33 | 3.00 | 22.00 | 1.00 | 21.00 | 50.93 |
| `refitUnion--compact_keep--guarded_trim` | 42.00 | 0.00 | 10.00 | 32.00 | 8.67 | 11.67 | 3.00 | 22.00 | 1.00 | 21.00 | 50.93 |
| `refitUnion--dino_boost--guarded_trim` | 36.00 | 0.00 | 4.00 | 32.00 | 4.00 | 6.00 | 3.00 | 22.00 | 1.00 | 21.00 | 50.93 |

Mixed means a kept flagged raw candidate contains both human rally core and other time. That other time is not necessarily incorrect export under the padded metric. Binary raw-event F1 is a secondary guardrail; boundary-editing output has no invented raw-event match score.

## Export time accounting at target padding

| Policy | Human action | Export min | Correctly removed min | Incorrectly removed min | Incorrect export min | Missed core s |
| --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--compact_boost--guarded_trim` | binary | 82.27 | 54.00 | 1.57 | 22.58 | 17.47 |
| `productionDefault--compact_boost--guarded_trim` | boundary | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--compact_boost--suppression_zero` | binary | 77.59 | 58.62 | 1.62 | 17.96 | 17.54 |
| `productionDefault--compact_boost--suppression_zero` | boundary | 77.06 | 59.20 | 1.57 | 17.38 | 17.47 |
| `productionDefault--compact_boost--suppression_half` | binary | 77.39 | 58.80 | 1.65 | 17.78 | 17.67 |
| `productionDefault--compact_boost--suppression_half` | boundary | 75.29 | 60.97 | 1.57 | 15.61 | 17.47 |
| `productionDefault--compact_boost--suppression_any` | binary | 76.31 | 59.88 | 1.65 | 16.70 | 17.67 |
| `productionDefault--compact_boost--suppression_any` | boundary | 65.22 | 71.05 | 1.57 | 5.54 | 17.47 |
| `productionDefault--compact_boost--bidirectional_half` | binary | 77.41 | 58.79 | 1.64 | 17.80 | 17.67 |
| `productionDefault--compact_boost--bidirectional_half` | boundary | 75.29 | 60.99 | 1.55 | 15.59 | 17.47 |
| `productionDefault--compact_boost--bidirectional_any` | binary | 76.89 | 59.51 | 1.42 | 17.07 | 11.48 |
| `productionDefault--compact_boost--bidirectional_any` | boundary | 64.94 | 71.55 | 1.34 | 5.03 | 11.28 |
| `productionDefault--compact_keep--guarded_trim` | binary | 82.27 | 54.00 | 1.57 | 22.58 | 17.47 |
| `productionDefault--compact_keep--guarded_trim` | boundary | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--compact_keep--suppression_zero` | binary | 78.12 | 58.10 | 1.61 | 18.48 | 17.47 |
| `productionDefault--compact_keep--suppression_zero` | boundary | 77.62 | 58.65 | 1.57 | 17.94 | 17.47 |
| `productionDefault--compact_keep--suppression_half` | binary | 77.91 | 58.30 | 1.62 | 18.28 | 17.54 |
| `productionDefault--compact_keep--suppression_half` | boundary | 76.07 | 60.20 | 1.57 | 16.38 | 17.47 |
| `productionDefault--compact_keep--suppression_any` | binary | 76.48 | 59.73 | 1.63 | 16.86 | 17.54 |
| `productionDefault--compact_keep--suppression_any` | boundary | 65.73 | 70.54 | 1.57 | 6.05 | 17.47 |
| `productionDefault--compact_keep--bidirectional_half` | binary | 78.05 | 58.20 | 1.58 | 18.38 | 16.20 |
| `productionDefault--compact_keep--bidirectional_half` | boundary | 76.00 | 60.32 | 1.52 | 16.26 | 16.07 |
| `productionDefault--compact_keep--bidirectional_any` | binary | 77.34 | 59.11 | 1.39 | 17.47 | 11.12 |
| `productionDefault--compact_keep--bidirectional_any` | boundary | 65.27 | 71.24 | 1.32 | 5.34 | 10.99 |
| `productionDefault--dino_global--guarded_trim` | binary | 82.17 | 54.05 | 1.62 | 22.54 | 19.37 |
| `productionDefault--dino_global--guarded_trim` | boundary | 82.23 | 54.04 | 1.57 | 22.54 | 17.47 |
| `productionDefault--dino_global--suppression_zero` | binary | 77.22 | 58.96 | 1.65 | 17.62 | 18.55 |
| `productionDefault--dino_global--suppression_zero` | boundary | 76.83 | 59.44 | 1.57 | 17.14 | 17.47 |
| `productionDefault--dino_global--suppression_half` | binary | 76.73 | 59.43 | 1.68 | 17.16 | 19.51 |
| `productionDefault--dino_global--suppression_half` | boundary | 75.07 | 61.20 | 1.57 | 15.38 | 17.47 |
| `productionDefault--dino_global--suppression_any` | binary | 75.93 | 60.21 | 1.69 | 16.37 | 19.51 |
| `productionDefault--dino_global--suppression_any` | boundary | 65.08 | 71.18 | 1.57 | 5.40 | 17.47 |
| `productionDefault--dino_global--bidirectional_half` | binary | 76.82 | 59.42 | 1.59 | 17.16 | 17.15 |
| `productionDefault--dino_global--bidirectional_half` | boundary | 75.14 | 61.24 | 1.46 | 15.34 | 15.11 |
| `productionDefault--dino_global--bidirectional_any` | binary | 76.89 | 59.74 | 1.21 | 16.85 | 7.72 |
| `productionDefault--dino_global--bidirectional_any` | boundary | 65.08 | 71.60 | 1.15 | 4.98 | 7.58 |
| `productionDefault--dino_boost--guarded_trim` | binary | 82.27 | 54.00 | 1.57 | 22.58 | 17.47 |
| `productionDefault--dino_boost--guarded_trim` | boundary | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_boost--suppression_zero` | binary | 76.82 | 59.38 | 1.63 | 17.20 | 17.67 |
| `productionDefault--dino_boost--suppression_zero` | boundary | 76.61 | 59.66 | 1.57 | 16.92 | 17.47 |
| `productionDefault--dino_boost--suppression_half` | binary | 76.54 | 59.65 | 1.64 | 16.93 | 17.67 |
| `productionDefault--dino_boost--suppression_half` | boundary | 74.39 | 61.87 | 1.57 | 14.71 | 17.47 |
| `productionDefault--dino_boost--suppression_any` | binary | 76.08 | 60.11 | 1.65 | 16.48 | 17.67 |
| `productionDefault--dino_boost--suppression_any` | boundary | 65.69 | 70.57 | 1.57 | 6.01 | 17.47 |
| `productionDefault--dino_boost--bidirectional_half` | binary | 76.69 | 59.65 | 1.49 | 16.93 | 13.86 |
| `productionDefault--dino_boost--bidirectional_half` | boundary | 74.50 | 61.92 | 1.41 | 14.66 | 13.66 |
| `productionDefault--dino_boost--bidirectional_any` | binary | 77.32 | 59.33 | 1.19 | 17.25 | 6.68 |
| `productionDefault--dino_boost--bidirectional_any` | boundary | 65.55 | 71.18 | 1.10 | 5.40 | 6.48 |
| `productionDefault--dino_keep--guarded_trim` | binary | 82.27 | 54.00 | 1.57 | 22.58 | 17.47 |
| `productionDefault--dino_keep--guarded_trim` | boundary | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_keep--suppression_zero` | binary | 76.91 | 59.30 | 1.62 | 17.28 | 17.60 |
| `productionDefault--dino_keep--suppression_zero` | boundary | 76.77 | 59.49 | 1.57 | 17.09 | 17.47 |
| `productionDefault--dino_keep--suppression_half` | binary | 76.78 | 59.43 | 1.62 | 17.16 | 17.60 |
| `productionDefault--dino_keep--suppression_half` | boundary | 74.81 | 61.45 | 1.57 | 15.13 | 17.47 |
| `productionDefault--dino_keep--suppression_any` | binary | 76.26 | 59.94 | 1.64 | 16.64 | 17.60 |
| `productionDefault--dino_keep--suppression_any` | boundary | 65.76 | 70.51 | 1.57 | 6.07 | 17.47 |
| `productionDefault--dino_keep--bidirectional_half` | binary | 76.96 | 59.38 | 1.49 | 17.20 | 14.70 |
| `productionDefault--dino_keep--bidirectional_half` | boundary | 74.91 | 61.49 | 1.43 | 15.09 | 14.57 |
| `productionDefault--dino_keep--bidirectional_any` | binary | 77.84 | 58.81 | 1.18 | 17.77 | 7.17 |
| `productionDefault--dino_keep--bidirectional_any` | boundary | 65.33 | 71.39 | 1.12 | 5.19 | 7.04 |
| `shippedUnion--compact_boost--guarded_trim` | binary | 86.73 | 49.95 | 1.16 | 26.63 | 8.58 |
| `shippedUnion--compact_boost--guarded_trim` | boundary | 86.87 | 49.83 | 1.13 | 26.75 | 8.58 |
| `shippedUnion--compact_boost--suppression_zero` | binary | 78.73 | 57.87 | 1.23 | 18.71 | 8.65 |
| `shippedUnion--compact_boost--suppression_zero` | boundary | 78.34 | 58.37 | 1.13 | 18.22 | 8.58 |
| `shippedUnion--compact_boost--suppression_half` | binary | 78.28 | 58.29 | 1.26 | 18.29 | 8.78 |
| `shippedUnion--compact_boost--suppression_half` | boundary | 76.31 | 60.40 | 1.13 | 16.18 | 8.58 |
| `shippedUnion--compact_boost--suppression_any` | binary | 76.96 | 59.61 | 1.26 | 16.97 | 8.78 |
| `shippedUnion--compact_boost--suppression_any` | boundary | 65.94 | 70.76 | 1.13 | 5.82 | 8.58 |
| `shippedUnion--compact_boost--bidirectional_half` | binary | 78.37 | 58.21 | 1.25 | 18.37 | 8.78 |
| `shippedUnion--compact_boost--bidirectional_half` | boundary | 76.30 | 60.41 | 1.12 | 16.17 | 8.58 |
| `shippedUnion--compact_boost--bidirectional_any` | binary | 77.45 | 59.25 | 1.13 | 17.33 | 5.71 |
| `shippedUnion--compact_boost--bidirectional_any` | boundary | 65.63 | 71.22 | 0.99 | 5.37 | 5.52 |
| `shippedUnion--compact_keep--guarded_trim` | binary | 86.81 | 49.87 | 1.16 | 26.71 | 8.58 |
| `shippedUnion--compact_keep--guarded_trim` | boundary | 86.95 | 49.75 | 1.13 | 26.83 | 8.58 |
| `shippedUnion--compact_keep--suppression_zero` | binary | 79.63 | 56.99 | 1.21 | 19.59 | 8.58 |
| `shippedUnion--compact_keep--suppression_zero` | boundary | 79.25 | 57.46 | 1.13 | 19.12 | 8.58 |
| `shippedUnion--compact_keep--suppression_half` | binary | 78.96 | 57.64 | 1.23 | 18.94 | 8.65 |
| `shippedUnion--compact_keep--suppression_half` | boundary | 77.22 | 59.48 | 1.13 | 17.10 | 8.58 |
| `shippedUnion--compact_keep--suppression_any` | binary | 77.27 | 59.33 | 1.24 | 17.25 | 8.65 |
| `shippedUnion--compact_keep--suppression_any` | boundary | 66.60 | 70.10 | 1.13 | 6.48 | 8.58 |
| `shippedUnion--compact_keep--bidirectional_half` | binary | 79.20 | 57.44 | 1.20 | 19.14 | 7.79 |
| `shippedUnion--compact_keep--bidirectional_half` | boundary | 77.16 | 59.57 | 1.10 | 17.02 | 7.73 |
| `shippedUnion--compact_keep--bidirectional_any` | binary | 77.99 | 58.74 | 1.11 | 17.84 | 5.83 |
| `shippedUnion--compact_keep--bidirectional_any` | boundary | 66.13 | 70.70 | 1.01 | 5.88 | 5.76 |
| `shippedUnion--dino_global--guarded_trim` | binary | 86.67 | 49.95 | 1.21 | 26.63 | 10.49 |
| `shippedUnion--dino_global--guarded_trim` | boundary | 86.86 | 49.84 | 1.13 | 26.74 | 8.58 |
| `shippedUnion--dino_global--suppression_zero` | binary | 78.78 | 57.81 | 1.25 | 18.77 | 9.67 |
| `shippedUnion--dino_global--suppression_zero` | boundary | 78.48 | 58.22 | 1.13 | 18.36 | 8.58 |
| `shippedUnion--dino_global--suppression_half` | binary | 77.60 | 58.95 | 1.28 | 17.63 | 10.62 |
| `shippedUnion--dino_global--suppression_half` | boundary | 76.06 | 60.64 | 1.13 | 15.94 | 8.58 |
| `shippedUnion--dino_global--suppression_any` | binary | 76.81 | 59.72 | 1.30 | 16.86 | 10.62 |
| `shippedUnion--dino_global--suppression_any` | boundary | 66.07 | 70.64 | 1.13 | 5.94 | 8.58 |
| `shippedUnion--dino_global--bidirectional_half` | binary | 77.71 | 58.84 | 1.28 | 17.74 | 10.62 |
| `shippedUnion--dino_global--bidirectional_half` | boundary | 76.04 | 60.68 | 1.11 | 15.90 | 8.58 |
| `shippedUnion--dino_global--bidirectional_any` | binary | 77.50 | 59.33 | 1.00 | 17.26 | 4.31 |
| `shippedUnion--dino_global--bidirectional_any` | boundary | 65.85 | 71.07 | 0.91 | 5.51 | 4.18 |
| `shippedUnion--dino_boost--guarded_trim` | binary | 86.76 | 49.91 | 1.16 | 26.67 | 8.58 |
| `shippedUnion--dino_boost--guarded_trim` | boundary | 86.91 | 49.79 | 1.13 | 26.79 | 8.58 |
| `shippedUnion--dino_boost--suppression_zero` | binary | 78.24 | 58.36 | 1.24 | 18.23 | 8.78 |
| `shippedUnion--dino_boost--suppression_zero` | boundary | 78.14 | 58.56 | 1.13 | 18.02 | 8.58 |
| `shippedUnion--dino_boost--suppression_half` | binary | 77.49 | 59.10 | 1.25 | 17.49 | 8.78 |
| `shippedUnion--dino_boost--suppression_half` | boundary | 75.47 | 61.23 | 1.13 | 15.35 | 8.58 |
| `shippedUnion--dino_boost--suppression_any` | binary | 76.71 | 59.86 | 1.26 | 16.72 | 8.78 |
| `shippedUnion--dino_boost--suppression_any` | boundary | 66.45 | 70.25 | 1.13 | 6.33 | 8.58 |
| `shippedUnion--dino_boost--bidirectional_half` | binary | 77.49 | 59.10 | 1.24 | 17.49 | 8.78 |
| `shippedUnion--dino_boost--bidirectional_half` | boundary | 75.44 | 61.27 | 1.12 | 15.31 | 8.58 |
| `shippedUnion--dino_boost--bidirectional_any` | binary | 77.65 | 59.13 | 1.04 | 17.45 | 4.72 |
| `shippedUnion--dino_boost--bidirectional_any` | boundary | 66.00 | 70.90 | 0.93 | 5.68 | 4.53 |
| `shippedUnion--dino_keep--guarded_trim` | binary | 86.73 | 49.95 | 1.16 | 26.63 | 8.58 |
| `shippedUnion--dino_keep--guarded_trim` | boundary | 86.87 | 49.83 | 1.13 | 26.75 | 8.58 |
| `shippedUnion--dino_keep--suppression_zero` | binary | 78.23 | 58.39 | 1.22 | 18.20 | 8.72 |
| `shippedUnion--dino_keep--suppression_zero` | boundary | 78.20 | 58.50 | 1.13 | 18.08 | 8.58 |
| `shippedUnion--dino_keep--suppression_half` | binary | 77.77 | 58.84 | 1.23 | 17.74 | 8.72 |
| `shippedUnion--dino_keep--suppression_half` | boundary | 75.93 | 60.77 | 1.13 | 15.81 | 8.58 |
| `shippedUnion--dino_keep--suppression_any` | binary | 76.90 | 59.68 | 1.25 | 16.90 | 8.72 |
| `shippedUnion--dino_keep--suppression_any` | boundary | 66.50 | 70.20 | 1.13 | 6.38 | 8.58 |
| `shippedUnion--dino_keep--bidirectional_half` | binary | 77.77 | 58.84 | 1.23 | 17.75 | 8.72 |
| `shippedUnion--dino_keep--bidirectional_half` | boundary | 75.91 | 60.79 | 1.13 | 15.79 | 8.58 |
| `shippedUnion--dino_keep--bidirectional_any` | binary | 78.15 | 58.66 | 1.01 | 17.92 | 4.31 |
| `shippedUnion--dino_keep--bidirectional_any` | boundary | 65.84 | 71.08 | 0.91 | 5.50 | 4.18 |
| `refitUnion--compact_boost--guarded_trim` | binary | 96.19 | 39.18 | 2.47 | 37.41 | 47.63 |
| `refitUnion--compact_boost--guarded_trim` | boundary | 96.58 | 39.04 | 2.21 | 37.54 | 40.17 |
| `refitUnion--compact_boost--suppression_zero` | binary | 83.82 | 51.24 | 2.77 | 25.34 | 48.99 |
| `refitUnion--compact_boost--suppression_zero` | boundary | 84.14 | 51.48 | 2.21 | 25.10 | 40.17 |
| `refitUnion--compact_boost--suppression_half` | binary | 81.59 | 53.43 | 2.81 | 23.15 | 48.99 |
| `refitUnion--compact_boost--suppression_half` | boundary | 78.27 | 57.35 | 2.21 | 19.23 | 40.17 |
| `refitUnion--compact_boost--suppression_any` | binary | 80.04 | 54.97 | 2.82 | 21.61 | 48.99 |
| `refitUnion--compact_boost--suppression_any` | boundary | 63.42 | 72.20 | 2.21 | 4.38 | 40.17 |
| `refitUnion--compact_boost--bidirectional_half` | binary | 82.08 | 53.14 | 2.61 | 23.44 | 38.77 |
| `refitUnion--compact_boost--bidirectional_half` | boundary | 78.40 | 57.41 | 2.02 | 19.17 | 30.18 |
| `refitUnion--compact_boost--bidirectional_any` | binary | 81.21 | 54.54 | 2.09 | 22.04 | 23.53 |
| `refitUnion--compact_boost--bidirectional_any` | boundary | 63.87 | 72.47 | 1.50 | 4.11 | 14.94 |
| `refitUnion--compact_keep--guarded_trim` | binary | 96.19 | 39.18 | 2.47 | 37.41 | 47.63 |
| `refitUnion--compact_keep--guarded_trim` | boundary | 96.59 | 39.04 | 2.21 | 37.54 | 40.17 |
| `refitUnion--compact_keep--suppression_zero` | binary | 84.92 | 50.16 | 2.75 | 26.42 | 48.99 |
| `refitUnion--compact_keep--suppression_zero` | boundary | 85.30 | 50.32 | 2.21 | 26.26 | 40.17 |
| `refitUnion--compact_keep--suppression_half` | binary | 82.29 | 52.75 | 2.80 | 23.84 | 48.99 |
| `refitUnion--compact_keep--suppression_half` | boundary | 79.14 | 56.48 | 2.21 | 20.10 | 40.17 |
| `refitUnion--compact_keep--suppression_any` | binary | 80.26 | 54.75 | 2.82 | 21.83 | 48.99 |
| `refitUnion--compact_keep--suppression_any` | boundary | 63.97 | 71.66 | 2.21 | 4.93 | 40.17 |
| `refitUnion--compact_keep--bidirectional_half` | binary | 82.75 | 52.55 | 2.54 | 24.04 | 36.65 |
| `refitUnion--compact_keep--bidirectional_half` | boundary | 79.34 | 56.53 | 1.97 | 20.05 | 28.47 |
| `refitUnion--compact_keep--bidirectional_any` | binary | 81.63 | 54.21 | 1.99 | 22.37 | 20.96 |
| `refitUnion--compact_keep--bidirectional_any` | boundary | 64.30 | 72.10 | 1.44 | 4.49 | 12.78 |
| `refitUnion--dino_global--guarded_trim` | binary | 96.23 | 39.16 | 2.44 | 37.42 | 46.53 |
| `refitUnion--dino_global--guarded_trim` | boundary | 96.69 | 38.94 | 2.21 | 37.65 | 40.17 |
| `refitUnion--dino_global--suppression_zero` | binary | 83.51 | 51.80 | 2.52 | 24.78 | 43.15 |
| `refitUnion--dino_global--suppression_zero` | boundary | 83.91 | 51.71 | 2.21 | 24.87 | 40.17 |
| `refitUnion--dino_global--suppression_half` | binary | 81.43 | 53.71 | 2.69 | 22.87 | 46.20 |
| `refitUnion--dino_global--suppression_half` | boundary | 78.05 | 57.57 | 2.21 | 19.01 | 40.17 |
| `refitUnion--dino_global--suppression_any` | binary | 80.32 | 54.73 | 2.78 | 21.85 | 47.89 |
| `refitUnion--dino_global--suppression_any` | boundary | 64.19 | 71.43 | 2.21 | 5.15 | 40.17 |
| `refitUnion--dino_global--bidirectional_half` | binary | 81.90 | 53.51 | 2.43 | 23.08 | 36.22 |
| `refitUnion--dino_global--bidirectional_half` | boundary | 78.19 | 57.63 | 2.01 | 18.95 | 31.46 |
| `refitUnion--dino_global--bidirectional_any` | binary | 82.14 | 54.13 | 1.56 | 22.45 | 9.48 |
| `refitUnion--dino_global--bidirectional_any` | boundary | 64.72 | 71.90 | 1.21 | 4.68 | 7.01 |
| `refitUnion--dino_boost--guarded_trim` | binary | 96.19 | 39.18 | 2.47 | 37.41 | 47.63 |
| `refitUnion--dino_boost--guarded_trim` | boundary | 96.70 | 38.92 | 2.21 | 37.66 | 40.17 |
| `refitUnion--dino_boost--suppression_zero` | binary | 83.83 | 51.49 | 2.52 | 25.10 | 43.25 |
| `refitUnion--dino_boost--suppression_zero` | boundary | 84.27 | 51.35 | 2.21 | 25.23 | 40.17 |
| `refitUnion--dino_boost--suppression_half` | binary | 81.26 | 53.88 | 2.70 | 22.71 | 45.67 |
| `refitUnion--dino_boost--suppression_half` | boundary | 78.32 | 57.31 | 2.21 | 19.28 | 40.17 |
| `refitUnion--dino_boost--suppression_any` | binary | 80.21 | 54.83 | 2.80 | 21.76 | 48.99 |
| `refitUnion--dino_boost--suppression_any` | boundary | 64.28 | 71.34 | 2.21 | 5.24 | 40.17 |
| `refitUnion--dino_boost--bidirectional_half` | binary | 81.82 | 53.66 | 2.35 | 22.92 | 33.18 |
| `refitUnion--dino_boost--bidirectional_half` | boundary | 78.52 | 57.36 | 1.95 | 19.22 | 29.63 |
| `refitUnion--dino_boost--bidirectional_any` | binary | 82.06 | 54.30 | 1.48 | 22.28 | 7.25 |
| `refitUnion--dino_boost--bidirectional_any` | boundary | 64.76 | 71.92 | 1.15 | 4.66 | 5.43 |
| `refitUnion--dino_keep--guarded_trim` | binary | 96.28 | 39.14 | 2.42 | 37.44 | 45.62 |
| `refitUnion--dino_keep--guarded_trim` | boundary | 96.76 | 38.86 | 2.21 | 37.72 | 40.17 |
| `refitUnion--dino_keep--suppression_zero` | binary | 83.65 | 51.64 | 2.55 | 24.95 | 44.25 |
| `refitUnion--dino_keep--suppression_zero` | boundary | 84.25 | 51.38 | 2.21 | 25.20 | 40.17 |
| `refitUnion--dino_keep--suppression_half` | binary | 81.14 | 54.02 | 2.68 | 22.56 | 46.03 |
| `refitUnion--dino_keep--suppression_half` | boundary | 78.56 | 57.06 | 2.21 | 19.52 | 40.17 |
| `refitUnion--dino_keep--suppression_any` | binary | 80.46 | 54.63 | 2.74 | 21.95 | 47.48 |
| `refitUnion--dino_keep--suppression_any` | boundary | 64.37 | 71.26 | 2.21 | 5.33 | 40.17 |
| `refitUnion--dino_keep--bidirectional_half` | binary | 81.70 | 53.78 | 2.35 | 22.80 | 32.94 |
| `refitUnion--dino_keep--bidirectional_half` | boundary | 78.72 | 57.16 | 1.95 | 19.42 | 28.74 |
| `refitUnion--dino_keep--bidirectional_any` | binary | 82.31 | 53.90 | 1.63 | 22.68 | 12.32 |
| `refitUnion--dino_keep--bidirectional_any` | boundary | 64.76 | 71.82 | 1.25 | 4.76 | 8.29 |
| `compact_boost--positive_uncertain` | binary | 58.70 | 70.01 | 9.13 | 6.58 | 181.64 |
| `compact_boost--positive_uncertain` | boundary | 58.54 | 70.19 | 9.11 | 6.39 | 181.20 |
| `compact_boost--uncertain_narrow` | binary | 75.29 | 60.40 | 2.14 | 16.18 | 30.20 |
| `compact_boost--uncertain_narrow` | boundary | 60.75 | 75.08 | 2.00 | 1.50 | 28.49 |
| `compact_boost--uncertain_medium` | binary | 76.99 | 59.35 | 1.50 | 17.23 | 21.00 |
| `compact_boost--uncertain_medium` | boundary | 60.97 | 75.51 | 1.35 | 1.07 | 18.88 |
| `compact_boost--uncertain_wide` | binary | 78.56 | 58.34 | 0.93 | 18.24 | 13.85 |
| `compact_boost--uncertain_wide` | boundary | 61.23 | 75.75 | 0.85 | 0.83 | 13.60 |
| `compact_boost--all_positive` | binary | 56.71 | 71.88 | 9.25 | 4.70 | 182.40 |
| `compact_boost--all_positive` | boundary | 52.14 | 76.58 | 9.11 | 0.00 | 181.20 |
| `compact_boost--all_candidates` | binary | 82.32 | 55.51 | 0.00 | 21.07 | 0.00 |
| `compact_boost--all_candidates` | boundary | 61.25 | 76.58 | 0.00 | 0.00 | 0.00 |
| `compact_keep--positive_uncertain` | binary | 60.09 | 69.40 | 8.35 | 7.18 | 158.22 |
| `compact_keep--positive_uncertain` | boundary | 59.73 | 69.81 | 8.29 | 6.77 | 157.02 |
| `compact_keep--uncertain_narrow` | binary | 75.01 | 60.47 | 2.36 | 16.11 | 36.64 |
| `compact_keep--uncertain_narrow` | boundary | 60.82 | 74.82 | 2.20 | 1.76 | 34.64 |
| `compact_keep--uncertain_medium` | binary | 76.15 | 59.78 | 1.90 | 16.80 | 30.46 |
| `compact_keep--uncertain_medium` | boundary | 60.84 | 75.26 | 1.74 | 1.32 | 28.14 |
| `compact_keep--uncertain_wide` | binary | 78.26 | 58.58 | 0.99 | 18.00 | 14.35 |
| `compact_keep--uncertain_wide` | boundary | 61.30 | 75.62 | 0.91 | 0.96 | 14.10 |
| `compact_keep--all_positive` | binary | 57.87 | 71.48 | 8.49 | 5.10 | 159.04 |
| `compact_keep--all_positive` | boundary | 52.96 | 76.58 | 8.29 | 0.00 | 157.02 |
| `compact_keep--all_candidates` | binary | 82.15 | 55.68 | 0.00 | 20.90 | 0.00 |
| `compact_keep--all_candidates` | boundary | 61.25 | 76.58 | 0.00 | 0.00 | 0.00 |
| `dino_global--positive_uncertain` | binary | 60.96 | 70.87 | 6.01 | 5.71 | 73.06 |
| `dino_global--positive_uncertain` | boundary | 60.79 | 71.07 | 5.98 | 5.51 | 72.57 |
| `dino_global--uncertain_narrow` | binary | 75.59 | 60.74 | 1.50 | 15.84 | 14.04 |
| `dino_global--uncertain_narrow` | boundary | 61.47 | 74.87 | 1.49 | 1.71 | 14.04 |
| `dino_global--uncertain_medium` | binary | 76.28 | 60.36 | 1.20 | 16.22 | 11.64 |
| `dino_global--uncertain_medium` | boundary | 61.38 | 75.28 | 1.18 | 1.30 | 11.64 |
| `dino_global--uncertain_wide` | binary | 77.36 | 59.68 | 0.79 | 16.90 | 6.48 |
| `dino_global--uncertain_wide` | boundary | 61.22 | 75.86 | 0.75 | 0.72 | 6.48 |
| `dino_global--all_positive` | binary | 60.02 | 71.77 | 6.04 | 4.81 | 73.56 |
| `dino_global--all_positive` | boundary | 55.27 | 76.58 | 5.98 | 0.00 | 72.57 |
| `dino_global--all_candidates` | binary | 81.84 | 55.99 | 0.00 | 20.59 | 0.00 |
| `dino_global--all_candidates` | boundary | 61.25 | 76.58 | 0.00 | 0.00 | 0.00 |
| `dino_boost--positive_uncertain` | binary | 61.64 | 70.03 | 6.16 | 6.55 | 73.25 |
| `dino_boost--positive_uncertain` | boundary | 61.53 | 70.18 | 6.13 | 6.41 | 72.70 |
| `dino_boost--uncertain_narrow` | binary | 77.72 | 58.51 | 1.60 | 18.07 | 21.53 |
| `dino_boost--uncertain_narrow` | boundary | 61.46 | 74.79 | 1.59 | 1.79 | 21.53 |
| `dino_boost--uncertain_medium` | binary | 78.68 | 57.87 | 1.28 | 18.71 | 19.17 |
| `dino_boost--uncertain_medium` | boundary | 61.38 | 75.20 | 1.25 | 1.38 | 18.62 |
| `dino_boost--uncertain_wide` | binary | 79.79 | 57.28 | 0.76 | 19.30 | 9.79 |
| `dino_boost--uncertain_wide` | boundary | 61.33 | 75.76 | 0.75 | 0.83 | 9.79 |
| `dino_boost--all_positive` | binary | 60.46 | 71.12 | 6.25 | 5.46 | 74.78 |
| `dino_boost--all_positive` | boundary | 55.13 | 76.58 | 6.13 | 0.00 | 72.70 |
| `dino_boost--all_candidates` | binary | 83.13 | 54.71 | 0.00 | 21.87 | 0.00 |
| `dino_boost--all_candidates` | boundary | 61.25 | 76.58 | 0.00 | 0.00 | 0.00 |
| `dino_keep--positive_uncertain` | binary | 61.47 | 69.73 | 6.64 | 6.86 | 84.18 |
| `dino_keep--positive_uncertain` | boundary | 60.80 | 70.41 | 6.62 | 6.17 | 84.18 |
| `dino_keep--uncertain_narrow` | binary | 77.20 | 59.08 | 1.55 | 17.50 | 16.68 |
| `dino_keep--uncertain_narrow` | boundary | 62.21 | 74.09 | 1.53 | 2.49 | 16.68 |
| `dino_keep--uncertain_medium` | binary | 78.06 | 58.51 | 1.26 | 18.07 | 14.14 |
| `dino_keep--uncertain_medium` | boundary | 61.83 | 74.76 | 1.25 | 1.82 | 14.14 |
| `dino_keep--uncertain_wide` | binary | 78.96 | 58.01 | 0.86 | 18.57 | 9.76 |
| `dino_keep--uncertain_wide` | boundary | 61.17 | 75.82 | 0.84 | 0.76 | 9.76 |
| `dino_keep--all_positive` | binary | 60.73 | 70.41 | 6.69 | 6.17 | 85.07 |
| `dino_keep--all_positive` | boundary | 54.63 | 76.58 | 6.62 | 0.00 | 84.18 |
| `dino_keep--all_candidates` | binary | 83.67 | 54.16 | 0.00 | 22.42 | 0.00 |
| `dino_keep--all_candidates` | boundary | 61.25 | 76.58 | 0.00 | 0.00 | 0.00 |

Correctly removed is unwanted omitted time (TN); incorrectly removed is wanted padded human export omitted (FNpad); incorrect export is unwanted retained time (FP). Export + TN + FNpad partitions evaluable footage. Missed core separately measures actual play loss.

## Prior fine-grained export-disagreement reference

These previously tested queues flag disputed export seconds rather than whole candidates. Human editing is restricted to those disputed seconds, with two seconds of playback context. They have no whole-rally keep/remove estimate. Decisions below count disjoint disputed segments, not predicted rallies. The distinct true-rally counts are new post-hoc descriptions; flags and original quality metrics are unchanged.

| Policy | P_pad % | R_core % | F1_padP_coreR % | Playback min | Disputed segments | True rallies | Playback clips |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--compact_boost--bidirectional_review` | 90.95 | 99.54 | 95.05 | 69.39 | 602.33 | 74.33 | 445.33 |
| `productionDefault--compact_boost--suppression_review` | 90.91 | 99.27 | 94.90 | 63.36 | 539.00 | 71.00 | 410.67 |
| `productionDefault--compact_keep--bidirectional_review` | 90.13 | 99.58 | 94.62 | 69.03 | 607.00 | 66.00 | 446.00 |
| `productionDefault--compact_keep--suppression_review` | 90.08 | 99.27 | 94.45 | 61.85 | 536.67 | 62.00 | 408.67 |
| `productionDefault--dino_boost--bidirectional_review` | 92.02 | 99.76 | 95.73 | 69.03 | 605.67 | 46.33 | 445.67 |
| `productionDefault--dino_boost--suppression_review` | 91.95 | 99.27 | 95.47 | 59.66 | 513.67 | 40.33 | 390.33 |
| `productionDefault--dino_global--bidirectional_review` | 92.16 | 99.69 | 95.77 | 68.35 | 608.67 | 42.33 | 452.33 |
| `productionDefault--dino_global--suppression_review` | 92.10 | 99.27 | 95.54 | 60.02 | 521.00 | 37.33 | 396.33 |
| `productionDefault--dino_keep--bidirectional_review` | 91.62 | 99.71 | 95.48 | 68.74 | 599.33 | 52.33 | 445.33 |
| `productionDefault--dino_keep--suppression_review` | 91.54 | 99.27 | 95.24 | 59.19 | 506.67 | 47.00 | 390.33 |
| `refitUnion--compact_boost--bidirectional_review` | 90.05 | 99.41 | 94.50 | 84.16 | 553.67 | 83.67 | 376.33 |
| `refitUnion--compact_boost--suppression_review` | 89.92 | 98.32 | 93.93 | 78.21 | 491.00 | 73.00 | 349.67 |
| `refitUnion--compact_keep--bidirectional_review` | 88.72 | 99.52 | 93.80 | 83.21 | 561.33 | 75.33 | 379.67 |
| `refitUnion--compact_keep--suppression_review` | 88.56 | 98.32 | 93.18 | 76.49 | 491.00 | 63.00 | 350.33 |
| `refitUnion--dino_boost--bidirectional_review` | 90.11 | 99.88 | 94.74 | 82.60 | 561.00 | 60.33 | 381.00 |
| `refitUnion--dino_boost--suppression_review` | 89.92 | 98.32 | 93.93 | 75.32 | 487.33 | 42.00 | 350.00 |
| `refitUnion--dino_global--bidirectional_review` | 90.93 | 99.76 | 95.14 | 82.98 | 564.67 | 54.00 | 385.67 |
| `refitUnion--dino_global--suppression_review` | 90.77 | 98.32 | 94.38 | 75.61 | 487.33 | 37.00 | 352.33 |
| `refitUnion--dino_keep--bidirectional_review` | 89.65 | 99.79 | 94.42 | 82.27 | 557.67 | 63.33 | 381.33 |
| `refitUnion--dino_keep--suppression_review` | 89.48 | 98.32 | 93.66 | 74.97 | 484.00 | 46.33 | 353.67 |
| `shippedUnion--compact_boost--bidirectional_review` | 90.55 | 99.79 | 94.94 | 76.67 | 610.67 | 76.33 | 421.33 |
| `shippedUnion--compact_boost--suppression_review` | 90.53 | 99.64 | 94.86 | 71.65 | 556.00 | 74.00 | 394.00 |
| `shippedUnion--compact_keep--bidirectional_review` | 89.57 | 99.80 | 94.40 | 76.03 | 616.00 | 67.33 | 422.67 |
| `shippedUnion--compact_keep--suppression_review` | 89.53 | 99.64 | 94.31 | 70.25 | 557.00 | 64.67 | 394.67 |
| `shippedUnion--dino_boost--bidirectional_review` | 91.45 | 99.84 | 95.46 | 75.78 | 616.00 | 45.00 | 424.33 |
| `shippedUnion--dino_boost--suppression_review` | 91.41 | 99.64 | 95.34 | 68.17 | 538.33 | 41.67 | 384.00 |
| `shippedUnion--dino_global--bidirectional_review` | 91.49 | 99.83 | 95.48 | 74.90 | 617.33 | 42.33 | 430.67 |
| `shippedUnion--dino_global--suppression_review` | 91.45 | 99.64 | 95.37 | 68.47 | 545.33 | 39.33 | 391.67 |
| `shippedUnion--dino_keep--bidirectional_review` | 91.12 | 99.83 | 95.26 | 75.61 | 610.00 | 51.67 | 425.00 |
| `shippedUnion--dino_keep--suppression_review` | 91.08 | 99.64 | 95.15 | 67.51 | 528.33 | 48.67 | 384.00 |

## Required 0/1/2/3-second padding sensitivity

Candidate inventories and flags remain fixed across these padding cases. Playback duration varies with the decision export’s padding. No model or policy chooses its own best padding.

| Policy | Padding s | Human action | P_pad % | R_core % | F1_padP_coreR % | Model export s | Human export s | Difference s | Playback min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--compact_boost--guarded_trim` | 0 | binary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--compact_boost--guarded_trim` | 0 | boundary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--compact_boost--guarded_trim` | 1 | binary | 69.50 | 98.30 | 81.43 | 4217.53 | 3031.15 | 1186.37 | 0.14 |
| `productionDefault--compact_boost--guarded_trim` | 1 | boundary | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 | 0.14 |
| `productionDefault--compact_boost--guarded_trim` | 2 | binary | 72.55 | 99.27 | 83.83 | 4936.04 | 3675.15 | 1260.89 | 0.17 |
| `productionDefault--compact_boost--guarded_trim` | 2 | boundary | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 | 0.17 |
| `productionDefault--compact_boost--guarded_trim` | 3 | binary | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 | 0.21 |
| `productionDefault--compact_boost--guarded_trim` | 3 | boundary | 75.13 | 99.49 | 85.61 | 5628.70 | 4323.26 | 1305.43 | 0.21 |
| `productionDefault--compact_boost--suppression_zero` | 0 | binary | 68.28 | 95.71 | 79.70 | 3345.98 | 2387.15 | 958.83 | 8.64 |
| `productionDefault--compact_boost--suppression_zero` | 0 | boundary | 69.21 | 95.75 | 80.34 | 3302.52 | 2387.15 | 915.37 | 8.64 |
| `productionDefault--compact_boost--suppression_zero` | 1 | binary | 73.16 | 98.30 | 83.89 | 4004.21 | 3031.15 | 973.06 | 10.70 |
| `productionDefault--compact_boost--suppression_zero` | 1 | boundary | 73.89 | 98.30 | 84.37 | 3966.90 | 3031.15 | 935.75 | 10.70 |
| `productionDefault--compact_boost--suppression_zero` | 2 | binary | 76.85 | 99.27 | 86.63 | 4655.67 | 3675.15 | 980.52 | 12.86 |
| `productionDefault--compact_boost--suppression_zero` | 2 | boundary | 77.45 | 99.27 | 87.01 | 4623.87 | 3675.15 | 948.72 | 12.86 |
| `productionDefault--compact_boost--suppression_zero` | 3 | binary | 79.69 | 99.49 | 88.49 | 5299.19 | 4323.26 | 975.93 | 14.76 |
| `productionDefault--compact_boost--suppression_zero` | 3 | boundary | 80.34 | 99.49 | 88.90 | 5264.04 | 4323.26 | 940.77 | 14.76 |
| `productionDefault--compact_boost--suppression_half` | 0 | binary | 68.40 | 95.71 | 79.79 | 3340.16 | 2387.15 | 953.01 | 12.02 |
| `productionDefault--compact_boost--suppression_half` | 0 | boundary | 71.47 | 95.75 | 81.84 | 3198.65 | 2387.15 | 811.50 | 12.02 |
| `productionDefault--compact_boost--suppression_half` | 1 | binary | 73.32 | 98.30 | 83.99 | 3995.46 | 3031.15 | 964.31 | 14.36 |
| `productionDefault--compact_boost--suppression_half` | 1 | boundary | 75.91 | 98.30 | 85.66 | 3861.99 | 3031.15 | 830.84 | 14.36 |
| `productionDefault--compact_boost--suppression_half` | 2 | binary | 77.03 | 99.26 | 86.74 | 4643.18 | 3675.15 | 968.03 | 16.82 |
| `productionDefault--compact_boost--suppression_half` | 2 | boundary | 79.28 | 99.27 | 88.15 | 4517.46 | 3675.15 | 842.31 | 16.82 |
| `productionDefault--compact_boost--suppression_half` | 3 | binary | 79.84 | 99.49 | 88.59 | 5287.36 | 4323.26 | 964.10 | 18.99 |
| `productionDefault--compact_boost--suppression_half` | 3 | boundary | 81.95 | 99.49 | 89.87 | 5161.03 | 4323.26 | 837.77 | 18.99 |
| `productionDefault--compact_boost--suppression_any` | 0 | binary | 69.28 | 95.71 | 80.38 | 3297.78 | 2387.15 | 910.62 | 42.18 |
| `productionDefault--compact_boost--suppression_any` | 0 | boundary | 87.79 | 95.75 | 91.60 | 2603.50 | 2387.15 | 216.35 | 42.18 |
| `productionDefault--compact_boost--suppression_any` | 1 | binary | 74.34 | 98.30 | 84.66 | 3940.62 | 3031.15 | 909.47 | 47.64 |
| `productionDefault--compact_boost--suppression_any` | 1 | boundary | 89.86 | 98.30 | 93.89 | 3261.74 | 3031.15 | 230.58 | 47.64 |
| `productionDefault--compact_boost--suppression_any` | 2 | binary | 78.11 | 99.26 | 87.43 | 4578.34 | 3675.15 | 903.19 | 52.75 |
| `productionDefault--compact_boost--suppression_any` | 2 | boundary | 91.51 | 99.27 | 95.23 | 3913.22 | 3675.15 | 238.06 | 52.75 |
| `productionDefault--compact_boost--suppression_any` | 3 | binary | 80.91 | 99.49 | 89.24 | 5216.94 | 4323.26 | 893.68 | 57.20 |
| `productionDefault--compact_boost--suppression_any` | 3 | boundary | 92.18 | 99.49 | 95.70 | 4587.95 | 4323.26 | 264.69 | 57.20 |
| `productionDefault--compact_boost--bidirectional_half` | 0 | binary | 68.39 | 95.74 | 79.79 | 3341.65 | 2387.15 | 954.50 | 12.45 |
| `productionDefault--compact_boost--bidirectional_half` | 0 | boundary | 71.48 | 95.77 | 81.86 | 3199.23 | 2387.15 | 812.08 | 12.45 |
| `productionDefault--compact_boost--bidirectional_half` | 1 | binary | 73.31 | 98.30 | 83.99 | 3996.96 | 3031.15 | 965.81 | 14.88 |
| `productionDefault--compact_boost--bidirectional_half` | 1 | boundary | 75.91 | 98.30 | 85.67 | 3862.57 | 3031.15 | 831.42 | 14.88 |
| `productionDefault--compact_boost--bidirectional_half` | 2 | binary | 77.01 | 99.26 | 86.73 | 4644.67 | 3675.15 | 969.52 | 17.43 |
| `productionDefault--compact_boost--bidirectional_half` | 2 | boundary | 79.31 | 99.27 | 88.17 | 4517.15 | 3675.15 | 842.00 | 17.43 |
| `productionDefault--compact_boost--bidirectional_half` | 3 | binary | 79.83 | 99.49 | 88.58 | 5288.86 | 4323.26 | 965.59 | 19.69 |
| `productionDefault--compact_boost--bidirectional_half` | 3 | boundary | 82.01 | 99.49 | 89.90 | 5158.73 | 4323.26 | 835.47 | 19.69 |
| `productionDefault--compact_boost--bidirectional_any` | 0 | binary | 68.89 | 96.78 | 80.48 | 3353.69 | 2387.15 | 966.54 | 46.32 |
| `productionDefault--compact_boost--bidirectional_any` | 0 | boundary | 88.83 | 96.76 | 92.62 | 2600.26 | 2387.15 | 213.11 | 46.32 |
| `productionDefault--compact_boost--bidirectional_any` | 1 | binary | 74.01 | 99.00 | 84.70 | 3987.74 | 3031.15 | 956.59 | 52.25 |
| `productionDefault--compact_boost--bidirectional_any` | 1 | boundary | 90.71 | 98.99 | 94.67 | 3255.19 | 3031.15 | 224.04 | 52.25 |
| `productionDefault--compact_boost--bidirectional_any` | 2 | binary | 77.80 | 99.52 | 87.33 | 4613.70 | 3675.15 | 938.55 | 57.76 |
| `productionDefault--compact_boost--bidirectional_any` | 2 | boundary | 92.25 | 99.53 | 95.75 | 3896.63 | 3675.15 | 221.48 | 57.76 |
| `productionDefault--compact_boost--bidirectional_any` | 3 | binary | 80.61 | 99.64 | 89.12 | 5253.05 | 4323.26 | 929.79 | 62.37 |
| `productionDefault--compact_boost--bidirectional_any` | 3 | boundary | 92.84 | 99.64 | 96.12 | 4570.34 | 4323.26 | 247.08 | 62.37 |
| `productionDefault--compact_keep--guarded_trim` | 0 | binary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--compact_keep--guarded_trim` | 0 | boundary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--compact_keep--guarded_trim` | 1 | binary | 69.50 | 98.30 | 81.43 | 4217.53 | 3031.15 | 1186.37 | 0.14 |
| `productionDefault--compact_keep--guarded_trim` | 1 | boundary | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 | 0.14 |
| `productionDefault--compact_keep--guarded_trim` | 2 | binary | 72.55 | 99.27 | 83.83 | 4936.04 | 3675.15 | 1260.89 | 0.17 |
| `productionDefault--compact_keep--guarded_trim` | 2 | boundary | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 | 0.17 |
| `productionDefault--compact_keep--guarded_trim` | 3 | binary | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 | 0.21 |
| `productionDefault--compact_keep--guarded_trim` | 3 | boundary | 75.13 | 99.49 | 85.61 | 5628.70 | 4323.26 | 1305.43 | 0.21 |
| `productionDefault--compact_keep--suppression_zero` | 0 | binary | 67.94 | 95.71 | 79.47 | 3362.80 | 2387.15 | 975.65 | 7.69 |
| `productionDefault--compact_keep--suppression_zero` | 0 | boundary | 68.74 | 95.75 | 80.03 | 3325.16 | 2387.15 | 938.01 | 7.69 |
| `productionDefault--compact_keep--suppression_zero` | 1 | binary | 72.70 | 98.30 | 83.58 | 4029.99 | 3031.15 | 998.84 | 9.51 |
| `productionDefault--compact_keep--suppression_zero` | 1 | boundary | 73.37 | 98.30 | 84.02 | 3995.13 | 3031.15 | 963.98 | 9.51 |
| `productionDefault--compact_keep--suppression_zero` | 2 | binary | 76.35 | 99.27 | 86.31 | 4687.37 | 3675.15 | 1012.22 | 11.47 |
| `productionDefault--compact_keep--suppression_zero` | 2 | boundary | 76.90 | 99.27 | 86.66 | 4657.20 | 3675.15 | 982.05 | 11.47 |
| `productionDefault--compact_keep--suppression_zero` | 3 | binary | 79.21 | 99.49 | 88.20 | 5332.14 | 4323.26 | 1008.88 | 13.13 |
| `productionDefault--compact_keep--suppression_zero` | 3 | boundary | 79.76 | 99.49 | 88.54 | 5302.72 | 4323.26 | 979.46 | 13.13 |
| `productionDefault--compact_keep--suppression_half` | 0 | binary | 68.07 | 95.71 | 79.56 | 3356.68 | 2387.15 | 969.53 | 10.70 |
| `productionDefault--compact_keep--suppression_half` | 0 | boundary | 70.65 | 95.75 | 81.30 | 3235.95 | 2387.15 | 848.80 | 10.70 |
| `productionDefault--compact_keep--suppression_half` | 1 | binary | 72.86 | 98.30 | 83.69 | 4021.12 | 3031.15 | 989.97 | 12.76 |
| `productionDefault--compact_keep--suppression_half` | 1 | boundary | 75.09 | 98.30 | 85.14 | 3904.21 | 3031.15 | 873.06 | 12.76 |
| `productionDefault--compact_keep--suppression_half` | 2 | binary | 76.54 | 99.27 | 86.43 | 4674.58 | 3675.15 | 999.43 | 14.97 |
| `productionDefault--compact_keep--suppression_half` | 2 | boundary | 78.48 | 99.27 | 87.65 | 4564.11 | 3675.15 | 888.96 | 14.97 |
| `productionDefault--compact_keep--suppression_half` | 3 | binary | 79.37 | 99.49 | 88.30 | 5320.69 | 4323.26 | 997.42 | 16.86 |
| `productionDefault--compact_keep--suppression_half` | 3 | boundary | 81.16 | 99.49 | 89.39 | 5211.41 | 4323.26 | 888.14 | 16.86 |
| `productionDefault--compact_keep--suppression_any` | 0 | binary | 69.19 | 95.71 | 80.32 | 3302.16 | 2387.15 | 915.01 | 39.95 |
| `productionDefault--compact_keep--suppression_any` | 0 | boundary | 86.86 | 95.75 | 91.09 | 2631.68 | 2387.15 | 244.52 | 39.95 |
| `productionDefault--compact_keep--suppression_any` | 1 | binary | 74.19 | 98.30 | 84.56 | 3948.37 | 3031.15 | 917.22 | 45.13 |
| `productionDefault--compact_keep--suppression_any` | 1 | boundary | 89.05 | 98.30 | 93.45 | 3291.58 | 3031.15 | 260.43 | 45.13 |
| `productionDefault--compact_keep--suppression_any` | 2 | binary | 77.96 | 99.27 | 87.33 | 4589.00 | 3675.15 | 913.85 | 49.91 |
| `productionDefault--compact_keep--suppression_any` | 2 | boundary | 90.81 | 99.27 | 94.85 | 3943.78 | 3675.15 | 268.63 | 49.91 |
| `productionDefault--compact_keep--suppression_any` | 3 | binary | 80.74 | 99.49 | 89.14 | 5229.44 | 4323.26 | 906.17 | 54.10 |
| `productionDefault--compact_keep--suppression_any` | 3 | boundary | 91.60 | 99.49 | 95.38 | 4616.98 | 4323.26 | 293.71 | 54.10 |
| `productionDefault--compact_keep--bidirectional_half` | 0 | binary | 67.94 | 95.84 | 79.52 | 3367.28 | 2387.15 | 980.12 | 11.73 |
| `productionDefault--compact_keep--bidirectional_half` | 0 | boundary | 70.73 | 95.78 | 81.37 | 3233.40 | 2387.15 | 846.25 | 11.73 |
| `productionDefault--compact_keep--bidirectional_half` | 1 | binary | 72.72 | 98.37 | 83.62 | 4031.92 | 3031.15 | 1000.77 | 14.01 |
| `productionDefault--compact_keep--bidirectional_half` | 1 | boundary | 75.19 | 98.36 | 85.22 | 3901.99 | 3031.15 | 870.84 | 14.01 |
| `productionDefault--compact_keep--bidirectional_half` | 2 | binary | 76.45 | 99.32 | 86.40 | 4683.26 | 3675.15 | 1008.11 | 16.44 |
| `productionDefault--compact_keep--bidirectional_half` | 2 | boundary | 78.61 | 99.33 | 87.76 | 4559.96 | 3675.15 | 884.81 | 16.44 |
| `productionDefault--compact_keep--bidirectional_half` | 3 | binary | 79.30 | 99.53 | 88.27 | 5328.70 | 4323.26 | 1005.44 | 18.56 |
| `productionDefault--compact_keep--bidirectional_half` | 3 | boundary | 81.37 | 99.53 | 89.54 | 5202.80 | 4323.26 | 879.53 | 18.56 |
| `productionDefault--compact_keep--bidirectional_any` | 0 | binary | 68.47 | 96.84 | 80.22 | 3376.56 | 2387.15 | 989.41 | 45.24 |
| `productionDefault--compact_keep--bidirectional_any` | 0 | boundary | 88.07 | 96.73 | 92.20 | 2622.04 | 2387.15 | 234.89 | 45.24 |
| `productionDefault--compact_keep--bidirectional_any` | 1 | binary | 73.57 | 99.00 | 84.41 | 4012.82 | 3031.15 | 981.67 | 51.09 |
| `productionDefault--compact_keep--bidirectional_any` | 1 | boundary | 90.17 | 98.99 | 94.38 | 3275.25 | 3031.15 | 244.10 | 51.09 |
| `productionDefault--compact_keep--bidirectional_any` | 2 | binary | 77.41 | 99.53 | 87.09 | 4640.32 | 3675.15 | 965.17 | 56.45 |
| `productionDefault--compact_keep--bidirectional_any` | 2 | boundary | 91.82 | 99.54 | 95.52 | 3916.29 | 3675.15 | 241.14 | 56.45 |
| `productionDefault--compact_keep--bidirectional_any` | 3 | binary | 80.24 | 99.67 | 88.90 | 5281.13 | 4323.26 | 957.86 | 60.93 |
| `productionDefault--compact_keep--bidirectional_any` | 3 | boundary | 92.56 | 99.67 | 95.98 | 4586.29 | 4323.26 | 263.02 | 60.93 |
| `productionDefault--dino_global--guarded_trim` | 0 | binary | 65.58 | 95.75 | 77.84 | 3485.44 | 2387.15 | 1098.29 | 0.18 |
| `productionDefault--dino_global--guarded_trim` | 0 | boundary | 65.58 | 95.75 | 77.84 | 3485.44 | 2387.15 | 1098.29 | 0.18 |
| `productionDefault--dino_global--guarded_trim` | 1 | binary | 69.54 | 98.30 | 81.46 | 4214.53 | 3031.15 | 1183.37 | 0.24 |
| `productionDefault--dino_global--guarded_trim` | 1 | boundary | 69.52 | 98.30 | 81.44 | 4216.16 | 3031.15 | 1185.01 | 0.24 |
| `productionDefault--dino_global--guarded_trim` | 2 | binary | 72.57 | 99.19 | 83.82 | 4929.96 | 3675.15 | 1254.81 | 0.29 |
| `productionDefault--dino_global--guarded_trim` | 2 | boundary | 72.58 | 99.27 | 83.85 | 4933.68 | 3675.15 | 1258.53 | 0.29 |
| `productionDefault--dino_global--guarded_trim` | 3 | binary | 75.19 | 99.44 | 85.63 | 5620.26 | 4323.26 | 1296.99 | 0.35 |
| `productionDefault--dino_global--guarded_trim` | 3 | boundary | 75.17 | 99.49 | 85.64 | 5625.85 | 4323.26 | 1302.59 | 0.35 |
| `productionDefault--dino_global--suppression_zero` | 0 | binary | 68.61 | 95.75 | 79.94 | 3331.33 | 2387.15 | 944.17 | 7.10 |
| `productionDefault--dino_global--suppression_zero` | 0 | boundary | 69.30 | 95.75 | 80.40 | 3298.50 | 2387.15 | 911.34 | 7.10 |
| `productionDefault--dino_global--suppression_zero` | 1 | binary | 73.51 | 98.30 | 84.12 | 3986.35 | 3031.15 | 955.19 | 8.78 |
| `productionDefault--dino_global--suppression_zero` | 1 | boundary | 74.06 | 98.30 | 84.48 | 3957.71 | 3031.15 | 926.56 | 8.78 |
| `productionDefault--dino_global--suppression_zero` | 2 | binary | 77.19 | 99.22 | 86.83 | 4633.46 | 3675.15 | 958.31 | 10.49 |
| `productionDefault--dino_global--suppression_zero` | 2 | boundary | 77.69 | 99.27 | 87.16 | 4609.74 | 3675.15 | 934.59 | 10.49 |
| `productionDefault--dino_global--suppression_zero` | 3 | binary | 80.08 | 99.46 | 88.72 | 5271.58 | 4323.26 | 948.32 | 12.12 |
| `productionDefault--dino_global--suppression_zero` | 3 | boundary | 80.60 | 99.49 | 89.05 | 5247.35 | 4323.26 | 924.09 | 12.12 |
| `productionDefault--dino_global--suppression_half` | 0 | binary | 68.98 | 95.74 | 80.18 | 3313.31 | 2387.15 | 926.16 | 9.86 |
| `productionDefault--dino_global--suppression_half` | 0 | boundary | 71.47 | 95.75 | 81.85 | 3198.35 | 2387.15 | 811.20 | 9.86 |
| `productionDefault--dino_global--suppression_half` | 1 | binary | 73.94 | 98.30 | 84.40 | 3962.61 | 3031.15 | 931.46 | 11.78 |
| `productionDefault--dino_global--suppression_half` | 1 | boundary | 76.06 | 98.30 | 85.76 | 3854.18 | 3031.15 | 823.03 | 11.78 |
| `productionDefault--dino_global--suppression_half` | 2 | binary | 77.64 | 99.18 | 87.10 | 4603.81 | 3675.15 | 928.66 | 13.78 |
| `productionDefault--dino_global--suppression_half` | 2 | boundary | 79.51 | 99.27 | 88.30 | 4504.07 | 3675.15 | 828.92 | 13.78 |
| `productionDefault--dino_global--suppression_half` | 3 | binary | 80.45 | 99.44 | 88.94 | 5244.60 | 4323.26 | 921.33 | 15.57 |
| `productionDefault--dino_global--suppression_half` | 3 | boundary | 82.23 | 99.49 | 90.04 | 5143.50 | 4323.26 | 820.24 | 15.57 |
| `productionDefault--dino_global--suppression_any` | 0 | binary | 69.56 | 95.71 | 80.57 | 3284.74 | 2387.15 | 897.59 | 38.77 |
| `productionDefault--dino_global--suppression_any` | 0 | boundary | 88.08 | 95.75 | 91.75 | 2595.24 | 2387.15 | 208.09 | 38.77 |
| `productionDefault--dino_global--suppression_any` | 1 | binary | 74.67 | 98.30 | 84.87 | 3922.79 | 3031.15 | 891.64 | 43.59 |
| `productionDefault--dino_global--suppression_any` | 1 | boundary | 90.24 | 98.30 | 94.10 | 3248.35 | 3031.15 | 217.20 | 43.59 |
| `productionDefault--dino_global--suppression_any` | 2 | binary | 78.44 | 99.18 | 87.60 | 4555.59 | 3675.15 | 880.44 | 48.24 |
| `productionDefault--dino_global--suppression_any` | 2 | boundary | 91.71 | 99.27 | 95.34 | 3904.94 | 3675.15 | 229.79 | 48.24 |
| `productionDefault--dino_global--suppression_any` | 3 | binary | 81.21 | 99.44 | 89.41 | 5194.20 | 4323.26 | 870.93 | 52.40 |
| `productionDefault--dino_global--suppression_any` | 3 | boundary | 92.28 | 99.49 | 95.75 | 4582.96 | 4323.26 | 259.70 | 52.40 |
| `productionDefault--dino_global--bidirectional_half` | 0 | binary | 68.99 | 95.79 | 80.21 | 3314.73 | 2387.15 | 927.57 | 10.65 |
| `productionDefault--dino_global--bidirectional_half` | 0 | boundary | 71.49 | 95.80 | 81.87 | 3199.68 | 2387.15 | 812.53 | 10.65 |
| `productionDefault--dino_global--bidirectional_half` | 1 | binary | 73.96 | 98.40 | 84.45 | 3966.03 | 3031.15 | 934.88 | 12.81 |
| `productionDefault--dino_global--bidirectional_half` | 1 | boundary | 76.09 | 98.40 | 85.81 | 3857.46 | 3031.15 | 826.31 | 12.81 |
| `productionDefault--dino_global--bidirectional_half` | 2 | binary | 77.67 | 99.28 | 87.15 | 4609.23 | 3675.15 | 934.07 | 15.06 |
| `productionDefault--dino_global--bidirectional_half` | 2 | boundary | 79.59 | 99.37 | 88.38 | 4508.21 | 3675.15 | 833.05 | 15.06 |
| `productionDefault--dino_global--bidirectional_half` | 3 | binary | 80.48 | 99.54 | 89.00 | 5252.01 | 4323.26 | 928.75 | 17.10 |
| `productionDefault--dino_global--bidirectional_half` | 3 | boundary | 82.34 | 99.59 | 90.15 | 5147.77 | 4323.26 | 824.51 | 17.10 |
| `productionDefault--dino_global--bidirectional_any` | 0 | binary | 69.21 | 97.25 | 80.87 | 3354.13 | 2387.15 | 966.97 | 44.30 |
| `productionDefault--dino_global--bidirectional_any` | 0 | boundary | 89.03 | 97.25 | 92.96 | 2607.84 | 2387.15 | 220.69 | 44.30 |
| `productionDefault--dino_global--bidirectional_any` | 1 | binary | 74.25 | 99.21 | 84.94 | 3988.43 | 3031.15 | 957.27 | 49.61 |
| `productionDefault--dino_global--bidirectional_any` | 1 | boundary | 90.95 | 99.21 | 94.90 | 3257.48 | 3031.15 | 226.33 | 49.61 |
| `productionDefault--dino_global--bidirectional_any` | 2 | binary | 78.09 | 99.68 | 87.57 | 4613.30 | 3675.15 | 938.15 | 54.80 |
| `productionDefault--dino_global--bidirectional_any` | 2 | boundary | 92.36 | 99.68 | 95.88 | 3904.54 | 3675.15 | 229.39 | 54.80 |
| `productionDefault--dino_global--bidirectional_any` | 3 | binary | 80.89 | 99.76 | 89.34 | 5253.15 | 4323.26 | 929.89 | 59.33 |
| `productionDefault--dino_global--bidirectional_any` | 3 | boundary | 92.83 | 99.76 | 96.17 | 4584.72 | 4323.26 | 261.45 | 59.33 |
| `productionDefault--dino_boost--guarded_trim` | 0 | binary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--dino_boost--guarded_trim` | 0 | boundary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--dino_boost--guarded_trim` | 1 | binary | 69.50 | 98.30 | 81.43 | 4217.53 | 3031.15 | 1186.37 | 0.14 |
| `productionDefault--dino_boost--guarded_trim` | 1 | boundary | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 | 0.14 |
| `productionDefault--dino_boost--guarded_trim` | 2 | binary | 72.55 | 99.27 | 83.83 | 4936.04 | 3675.15 | 1260.89 | 0.17 |
| `productionDefault--dino_boost--guarded_trim` | 2 | boundary | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 | 0.17 |
| `productionDefault--dino_boost--guarded_trim` | 3 | binary | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 | 0.21 |
| `productionDefault--dino_boost--guarded_trim` | 3 | boundary | 75.13 | 99.49 | 85.61 | 5628.70 | 4323.26 | 1305.43 | 0.21 |
| `productionDefault--dino_boost--suppression_zero` | 0 | binary | 68.92 | 95.75 | 80.15 | 3316.61 | 2387.15 | 929.46 | 7.01 |
| `productionDefault--dino_boost--suppression_zero` | 0 | boundary | 69.47 | 95.75 | 80.52 | 3290.38 | 2387.15 | 903.23 | 7.01 |
| `productionDefault--dino_boost--suppression_zero` | 1 | binary | 73.88 | 98.30 | 84.36 | 3966.50 | 3031.15 | 935.34 | 8.62 |
| `productionDefault--dino_boost--suppression_zero` | 1 | boundary | 74.25 | 98.30 | 84.60 | 3947.52 | 3031.15 | 916.37 | 8.62 |
| `productionDefault--dino_boost--suppression_zero` | 2 | binary | 77.61 | 99.26 | 87.11 | 4609.06 | 3675.15 | 933.91 | 10.27 |
| `productionDefault--dino_boost--suppression_zero` | 2 | boundary | 77.91 | 99.27 | 87.30 | 4596.57 | 3675.15 | 921.42 | 10.27 |
| `productionDefault--dino_boost--suppression_zero` | 3 | binary | 80.43 | 99.49 | 88.95 | 5249.41 | 4323.26 | 926.15 | 11.83 |
| `productionDefault--dino_boost--suppression_zero` | 3 | boundary | 80.82 | 99.49 | 89.19 | 5232.60 | 4323.26 | 909.33 | 11.83 |
| `productionDefault--dino_boost--suppression_half` | 0 | binary | 69.13 | 95.72 | 80.28 | 3305.46 | 2387.15 | 918.31 | 10.50 |
| `productionDefault--dino_boost--suppression_half` | 0 | boundary | 72.35 | 95.75 | 82.42 | 3159.12 | 2387.15 | 771.97 | 10.50 |
| `productionDefault--dino_boost--suppression_half` | 1 | binary | 74.13 | 98.30 | 84.52 | 3952.38 | 3031.15 | 921.23 | 12.32 |
| `productionDefault--dino_boost--suppression_half` | 1 | boundary | 76.85 | 98.30 | 86.26 | 3814.31 | 3031.15 | 783.16 | 12.32 |
| `productionDefault--dino_boost--suppression_half` | 2 | binary | 77.88 | 99.26 | 87.28 | 4592.35 | 3675.15 | 917.20 | 14.22 |
| `productionDefault--dino_boost--suppression_half` | 2 | boundary | 80.23 | 99.27 | 88.74 | 4463.60 | 3675.15 | 788.45 | 14.22 |
| `productionDefault--dino_boost--suppression_half` | 3 | binary | 80.65 | 99.49 | 89.08 | 5234.21 | 4323.26 | 910.95 | 15.98 |
| `productionDefault--dino_boost--suppression_half` | 3 | boundary | 82.82 | 99.49 | 90.39 | 5106.47 | 4323.26 | 783.20 | 15.98 |
| `productionDefault--dino_boost--suppression_any` | 0 | binary | 69.47 | 95.71 | 80.51 | 3288.78 | 2387.15 | 901.63 | 37.47 |
| `productionDefault--dino_boost--suppression_any` | 0 | boundary | 86.77 | 95.75 | 91.03 | 2634.67 | 2387.15 | 247.51 | 37.47 |
| `productionDefault--dino_boost--suppression_any` | 1 | binary | 74.56 | 98.30 | 84.80 | 3928.83 | 3031.15 | 897.68 | 42.27 |
| `productionDefault--dino_boost--suppression_any` | 1 | boundary | 89.11 | 98.30 | 93.48 | 3289.62 | 3031.15 | 258.47 | 42.27 |
| `productionDefault--dino_boost--suppression_any` | 2 | binary | 78.34 | 99.26 | 87.57 | 4564.60 | 3675.15 | 889.44 | 46.66 |
| `productionDefault--dino_boost--suppression_any` | 2 | boundary | 90.86 | 99.27 | 94.88 | 3941.48 | 3675.15 | 266.33 | 46.66 |
| `productionDefault--dino_boost--suppression_any` | 3 | binary | 81.10 | 99.49 | 89.36 | 5204.53 | 4323.26 | 881.27 | 50.68 |
| `productionDefault--dino_boost--suppression_any` | 3 | boundary | 91.68 | 99.49 | 95.42 | 4613.33 | 4323.26 | 290.07 | 50.68 |
| `productionDefault--dino_boost--bidirectional_half` | 0 | binary | 69.15 | 95.82 | 80.33 | 3308.04 | 2387.15 | 920.89 | 11.50 |
| `productionDefault--dino_boost--bidirectional_half` | 0 | boundary | 72.38 | 95.85 | 82.47 | 3161.53 | 2387.15 | 774.38 | 11.50 |
| `productionDefault--dino_boost--bidirectional_half` | 1 | binary | 74.16 | 98.46 | 84.60 | 3958.30 | 3031.15 | 927.15 | 13.66 |
| `productionDefault--dino_boost--bidirectional_half` | 1 | boundary | 76.89 | 98.46 | 86.35 | 3819.77 | 3031.15 | 788.62 | 13.66 |
| `productionDefault--dino_boost--bidirectional_half` | 2 | binary | 77.92 | 99.42 | 87.37 | 4601.60 | 3675.15 | 926.45 | 15.92 |
| `productionDefault--dino_boost--bidirectional_half` | 2 | boundary | 80.32 | 99.43 | 88.86 | 4470.24 | 3675.15 | 795.09 | 15.92 |
| `productionDefault--dino_boost--bidirectional_half` | 3 | binary | 80.69 | 99.65 | 89.18 | 5246.80 | 4323.26 | 923.53 | 18.00 |
| `productionDefault--dino_boost--bidirectional_half` | 3 | boundary | 82.96 | 99.65 | 90.54 | 5114.41 | 4323.26 | 791.15 | 18.00 |
| `productionDefault--dino_boost--bidirectional_any` | 0 | binary | 68.66 | 97.22 | 80.48 | 3380.45 | 2387.15 | 993.29 | 44.22 |
| `productionDefault--dino_boost--bidirectional_any` | 0 | boundary | 88.03 | 97.24 | 92.40 | 2637.30 | 2387.15 | 250.15 | 44.22 |
| `productionDefault--dino_boost--bidirectional_any` | 1 | binary | 73.78 | 99.26 | 84.65 | 4014.83 | 3031.15 | 983.67 | 49.78 |
| `productionDefault--dino_boost--bidirectional_any` | 1 | boundary | 90.12 | 99.26 | 94.47 | 3288.86 | 3031.15 | 257.71 | 49.78 |
| `productionDefault--dino_boost--bidirectional_any` | 2 | binary | 77.69 | 99.72 | 87.33 | 4639.17 | 3675.15 | 964.02 | 54.92 |
| `productionDefault--dino_boost--bidirectional_any` | 2 | boundary | 91.77 | 99.73 | 95.58 | 3932.78 | 3675.15 | 257.63 | 54.92 |
| `productionDefault--dino_boost--bidirectional_any` | 3 | binary | 80.52 | 99.83 | 89.14 | 5280.35 | 4323.26 | 957.09 | 59.44 |
| `productionDefault--dino_boost--bidirectional_any` | 3 | boundary | 92.44 | 99.83 | 96.00 | 4608.83 | 4323.26 | 285.57 | 59.44 |
| `productionDefault--dino_keep--guarded_trim` | 0 | binary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--dino_keep--guarded_trim` | 0 | boundary | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 | 0.11 |
| `productionDefault--dino_keep--guarded_trim` | 1 | binary | 69.50 | 98.30 | 81.43 | 4217.53 | 3031.15 | 1186.37 | 0.14 |
| `productionDefault--dino_keep--guarded_trim` | 1 | boundary | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 | 0.14 |
| `productionDefault--dino_keep--guarded_trim` | 2 | binary | 72.55 | 99.27 | 83.83 | 4936.04 | 3675.15 | 1260.89 | 0.17 |
| `productionDefault--dino_keep--guarded_trim` | 2 | boundary | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 | 0.17 |
| `productionDefault--dino_keep--guarded_trim` | 3 | binary | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 | 0.21 |
| `productionDefault--dino_keep--guarded_trim` | 3 | boundary | 75.13 | 99.49 | 85.61 | 5628.70 | 4323.26 | 1305.43 | 0.21 |
| `productionDefault--dino_keep--suppression_zero` | 0 | binary | 68.88 | 95.75 | 80.12 | 3318.63 | 2387.15 | 931.47 | 6.93 |
| `productionDefault--dino_keep--suppression_zero` | 0 | boundary | 69.32 | 95.75 | 80.42 | 3297.63 | 2387.15 | 910.48 | 6.93 |
| `productionDefault--dino_keep--suppression_zero` | 1 | binary | 73.82 | 98.30 | 84.32 | 3969.93 | 3031.15 | 938.78 | 8.59 |
| `productionDefault--dino_keep--suppression_zero` | 1 | boundary | 74.07 | 98.30 | 84.48 | 3957.61 | 3031.15 | 926.46 | 8.59 |
| `productionDefault--dino_keep--suppression_zero` | 2 | binary | 77.54 | 99.26 | 87.07 | 4614.70 | 3675.15 | 939.55 | 10.28 |
| `productionDefault--dino_keep--suppression_zero` | 2 | boundary | 77.75 | 99.27 | 87.20 | 4606.37 | 3675.15 | 931.22 | 10.28 |
| `productionDefault--dino_keep--suppression_zero` | 3 | binary | 80.38 | 99.49 | 88.92 | 5253.64 | 4323.26 | 930.38 | 11.89 |
| `productionDefault--dino_keep--suppression_zero` | 3 | boundary | 80.70 | 99.49 | 89.11 | 5240.97 | 4323.26 | 917.71 | 11.89 |
| `productionDefault--dino_keep--suppression_half` | 0 | binary | 68.98 | 95.74 | 80.18 | 3313.25 | 2387.15 | 926.10 | 10.21 |
| `productionDefault--dino_keep--suppression_half` | 0 | boundary | 71.88 | 95.75 | 82.11 | 3180.81 | 2387.15 | 793.66 | 10.21 |
| `productionDefault--dino_keep--suppression_half` | 1 | binary | 73.93 | 98.30 | 84.39 | 3963.18 | 3031.15 | 932.03 | 12.05 |
| `productionDefault--dino_keep--suppression_half` | 1 | boundary | 76.38 | 98.30 | 85.96 | 3838.80 | 3031.15 | 807.65 | 12.05 |
| `productionDefault--dino_keep--suppression_half` | 2 | binary | 77.66 | 99.26 | 87.14 | 4607.07 | 3675.15 | 931.92 | 13.96 |
| `productionDefault--dino_keep--suppression_half` | 2 | boundary | 79.80 | 99.27 | 88.47 | 4488.69 | 3675.15 | 813.54 | 13.96 |
| `productionDefault--dino_keep--suppression_half` | 3 | binary | 80.49 | 99.49 | 88.99 | 5246.19 | 4323.26 | 922.93 | 15.70 |
| `productionDefault--dino_keep--suppression_half` | 3 | boundary | 82.48 | 99.49 | 90.19 | 5128.26 | 4323.26 | 805.00 | 15.70 |
| `productionDefault--dino_keep--suppression_any` | 0 | binary | 69.36 | 95.71 | 80.43 | 3294.08 | 2387.15 | 906.93 | 40.38 |
| `productionDefault--dino_keep--suppression_any` | 0 | boundary | 86.94 | 95.75 | 91.11 | 2631.08 | 2387.15 | 243.93 | 40.38 |
| `productionDefault--dino_keep--suppression_any` | 1 | binary | 74.42 | 98.30 | 84.71 | 3936.17 | 3031.15 | 905.02 | 45.54 |
| `productionDefault--dino_keep--suppression_any` | 1 | boundary | 89.11 | 98.30 | 93.47 | 3291.19 | 3031.15 | 260.04 | 45.54 |
| `productionDefault--dino_keep--suppression_any` | 2 | binary | 78.18 | 99.26 | 87.47 | 4575.47 | 3675.15 | 900.32 | 50.34 |
| `productionDefault--dino_keep--suppression_any` | 2 | boundary | 90.81 | 99.27 | 94.84 | 3945.33 | 3675.15 | 270.18 | 50.34 |
| `productionDefault--dino_keep--suppression_any` | 3 | binary | 80.98 | 99.49 | 89.29 | 5213.41 | 4323.26 | 890.15 | 54.59 |
| `productionDefault--dino_keep--suppression_any` | 3 | boundary | 91.68 | 99.49 | 95.42 | 4614.35 | 4323.26 | 291.08 | 54.59 |
| `productionDefault--dino_keep--bidirectional_half` | 0 | binary | 68.95 | 95.84 | 80.20 | 3318.35 | 2387.15 | 931.19 | 10.99 |
| `productionDefault--dino_keep--bidirectional_half` | 0 | boundary | 71.91 | 95.85 | 82.16 | 3183.35 | 2387.15 | 796.20 | 10.99 |
| `productionDefault--dino_keep--bidirectional_half` | 1 | binary | 73.92 | 98.42 | 84.43 | 3970.94 | 3031.15 | 939.79 | 13.06 |
| `productionDefault--dino_keep--bidirectional_half` | 1 | boundary | 76.41 | 98.42 | 86.03 | 3843.86 | 3031.15 | 812.71 | 13.06 |
| `productionDefault--dino_keep--bidirectional_half` | 2 | binary | 77.66 | 99.38 | 87.19 | 4617.50 | 3675.15 | 942.35 | 15.19 |
| `productionDefault--dino_keep--bidirectional_half` | 2 | boundary | 79.88 | 99.39 | 88.57 | 4494.45 | 3675.15 | 819.29 | 15.19 |
| `productionDefault--dino_keep--bidirectional_half` | 3 | binary | 80.48 | 99.61 | 89.03 | 5259.87 | 4323.26 | 936.61 | 17.16 |
| `productionDefault--dino_keep--bidirectional_half` | 3 | boundary | 82.63 | 99.61 | 90.32 | 5132.87 | 4323.26 | 809.60 | 17.16 |
| `productionDefault--dino_keep--bidirectional_any` | 0 | binary | 68.16 | 97.27 | 80.14 | 3408.14 | 2387.15 | 1020.99 | 48.52 |
| `productionDefault--dino_keep--bidirectional_any` | 0 | boundary | 88.65 | 97.28 | 92.76 | 2620.38 | 2387.15 | 233.23 | 48.52 |
| `productionDefault--dino_keep--bidirectional_any` | 1 | binary | 73.31 | 99.23 | 84.31 | 4042.59 | 3031.15 | 1011.44 | 54.47 |
| `productionDefault--dino_keep--bidirectional_any` | 1 | boundary | 90.51 | 99.23 | 94.66 | 3275.46 | 3031.15 | 244.31 | 54.47 |
| `productionDefault--dino_keep--bidirectional_any` | 2 | binary | 77.19 | 99.70 | 87.01 | 4670.67 | 3675.15 | 995.52 | 60.02 |
| `productionDefault--dino_keep--bidirectional_any` | 2 | boundary | 92.07 | 99.71 | 95.73 | 3919.56 | 3675.15 | 244.41 | 60.02 |
| `productionDefault--dino_keep--bidirectional_any` | 3 | binary | 80.08 | 99.78 | 88.85 | 5310.41 | 4323.26 | 987.14 | 64.85 |
| `productionDefault--dino_keep--bidirectional_any` | 3 | boundary | 92.78 | 99.78 | 96.15 | 4590.78 | 4323.26 | 267.51 | 64.85 |
| `shippedUnion--compact_boost--guarded_trim` | 0 | binary | 62.41 | 96.26 | 75.72 | 3681.74 | 2387.15 | 1294.59 | 2.90 |
| `shippedUnion--compact_boost--guarded_trim` | 0 | boundary | 62.49 | 96.26 | 75.78 | 3677.17 | 2387.15 | 1290.02 | 2.90 |
| `shippedUnion--compact_boost--guarded_trim` | 1 | binary | 65.93 | 98.72 | 79.06 | 4474.03 | 3031.15 | 1442.88 | 3.71 |
| `shippedUnion--compact_boost--guarded_trim` | 1 | boundary | 65.92 | 98.72 | 79.06 | 4474.31 | 3031.15 | 1443.16 | 3.71 |
| `shippedUnion--compact_boost--guarded_trim` | 2 | binary | 69.29 | 99.64 | 81.74 | 5203.53 | 3675.15 | 1528.38 | 4.46 |
| `shippedUnion--compact_boost--guarded_trim` | 2 | boundary | 69.20 | 99.64 | 81.68 | 5212.49 | 3675.15 | 1537.34 | 4.46 |
| `shippedUnion--compact_boost--guarded_trim` | 3 | binary | 72.32 | 99.82 | 83.88 | 5889.88 | 4323.26 | 1566.62 | 5.25 |
| `shippedUnion--compact_boost--guarded_trim` | 3 | boundary | 72.22 | 99.85 | 83.81 | 5909.89 | 4323.26 | 1586.63 | 5.25 |
| `shippedUnion--compact_boost--suppression_zero` | 0 | binary | 67.62 | 96.19 | 79.42 | 3395.56 | 2387.15 | 1008.41 | 15.14 |
| `shippedUnion--compact_boost--suppression_zero` | 0 | boundary | 68.10 | 96.26 | 79.77 | 3373.93 | 2387.15 | 986.78 | 15.14 |
| `shippedUnion--compact_boost--suppression_zero` | 1 | binary | 72.49 | 98.72 | 83.60 | 4065.08 | 3031.15 | 1033.92 | 18.92 |
| `shippedUnion--compact_boost--suppression_zero` | 1 | boundary | 72.76 | 98.72 | 83.77 | 4053.99 | 3031.15 | 1022.84 | 18.92 |
| `shippedUnion--compact_boost--suppression_zero` | 2 | binary | 76.24 | 99.64 | 86.38 | 4723.71 | 3675.15 | 1048.56 | 22.51 |
| `shippedUnion--compact_boost--suppression_zero` | 2 | boundary | 76.75 | 99.64 | 86.71 | 4700.19 | 3675.15 | 1025.04 | 22.51 |
| `shippedUnion--compact_boost--suppression_zero` | 3 | binary | 79.10 | 99.82 | 88.26 | 5375.53 | 4323.26 | 1052.27 | 25.59 |
| `shippedUnion--compact_boost--suppression_zero` | 3 | boundary | 80.09 | 99.85 | 88.88 | 5329.46 | 4323.26 | 1006.19 | 25.59 |
| `shippedUnion--compact_boost--suppression_half` | 0 | binary | 68.01 | 96.18 | 79.67 | 3376.02 | 2387.15 | 988.87 | 18.82 |
| `shippedUnion--compact_boost--suppression_half` | 0 | boundary | 70.53 | 96.26 | 81.40 | 3258.81 | 2387.15 | 871.66 | 18.82 |
| `shippedUnion--compact_boost--suppression_half` | 1 | binary | 72.92 | 98.72 | 83.88 | 4040.62 | 3031.15 | 1009.47 | 22.94 |
| `shippedUnion--compact_boost--suppression_half` | 1 | boundary | 74.97 | 98.72 | 85.22 | 3935.05 | 3031.15 | 903.90 | 22.94 |
| `shippedUnion--compact_boost--suppression_half` | 2 | binary | 76.63 | 99.63 | 86.63 | 4696.92 | 3675.15 | 1021.77 | 26.91 |
| `shippedUnion--compact_boost--suppression_half` | 2 | boundary | 78.80 | 99.64 | 88.00 | 4578.38 | 3675.15 | 903.23 | 26.91 |
| `shippedUnion--compact_boost--suppression_half` | 3 | binary | 79.43 | 99.82 | 88.46 | 5351.07 | 4323.26 | 1027.80 | 30.26 |
| `shippedUnion--compact_boost--suppression_half` | 3 | boundary | 81.92 | 99.85 | 90.00 | 5210.72 | 4323.26 | 887.46 | 30.26 |
| `shippedUnion--compact_boost--suppression_any` | 0 | binary | 69.11 | 96.18 | 80.42 | 3322.25 | 2387.15 | 935.10 | 49.29 |
| `shippedUnion--compact_boost--suppression_any` | 0 | boundary | 86.85 | 96.26 | 91.31 | 2645.68 | 2387.15 | 258.53 | 49.29 |
| `shippedUnion--compact_boost--suppression_any` | 1 | binary | 74.17 | 98.72 | 84.70 | 3972.43 | 3031.15 | 941.28 | 56.21 |
| `shippedUnion--compact_boost--suppression_any` | 1 | boundary | 89.11 | 98.72 | 93.67 | 3310.29 | 3031.15 | 279.14 | 56.21 |
| `shippedUnion--compact_boost--suppression_any` | 2 | binary | 77.95 | 99.63 | 87.47 | 4617.48 | 3675.15 | 942.33 | 62.35 |
| `shippedUnion--compact_boost--suppression_any` | 2 | boundary | 91.18 | 99.64 | 95.22 | 3956.38 | 3675.15 | 281.23 | 62.35 |
| `shippedUnion--compact_boost--suppression_any` | 3 | binary | 80.71 | 99.82 | 89.25 | 5266.05 | 4323.26 | 942.79 | 67.24 |
| `shippedUnion--compact_boost--suppression_any` | 3 | boundary | 92.18 | 99.85 | 95.86 | 4630.20 | 4323.26 | 306.94 | 67.24 |
| `shippedUnion--compact_boost--bidirectional_half` | 0 | binary | 67.92 | 96.20 | 79.62 | 3381.26 | 2387.15 | 994.11 | 19.16 |
| `shippedUnion--compact_boost--bidirectional_half` | 0 | boundary | 70.54 | 96.28 | 81.42 | 3259.03 | 2387.15 | 871.87 | 19.16 |
| `shippedUnion--compact_boost--bidirectional_half` | 1 | binary | 72.85 | 98.72 | 83.83 | 4045.86 | 3031.15 | 1014.71 | 23.35 |
| `shippedUnion--compact_boost--bidirectional_half` | 1 | boundary | 74.99 | 98.72 | 85.23 | 3934.60 | 3031.15 | 903.44 | 23.35 |
| `shippedUnion--compact_boost--bidirectional_half` | 2 | binary | 76.56 | 99.63 | 86.59 | 4702.16 | 3675.15 | 1027.01 | 27.36 |
| `shippedUnion--compact_boost--bidirectional_half` | 2 | boundary | 78.82 | 99.64 | 88.01 | 4578.17 | 3675.15 | 903.02 | 27.36 |
| `shippedUnion--compact_boost--bidirectional_half` | 3 | binary | 79.35 | 99.82 | 88.42 | 5356.81 | 4323.26 | 1033.55 | 30.76 |
| `shippedUnion--compact_boost--bidirectional_half` | 3 | boundary | 81.96 | 99.85 | 90.02 | 5208.78 | 4323.26 | 885.51 | 30.76 |
| `shippedUnion--compact_boost--bidirectional_any` | 0 | binary | 68.67 | 97.01 | 80.42 | 3372.54 | 2387.15 | 985.39 | 52.63 |
| `shippedUnion--compact_boost--bidirectional_any` | 0 | boundary | 87.83 | 97.04 | 92.20 | 2637.57 | 2387.15 | 250.42 | 52.63 |
| `shippedUnion--compact_boost--bidirectional_any` | 1 | binary | 73.81 | 99.24 | 84.66 | 4013.93 | 3031.15 | 982.77 | 59.94 |
| `shippedUnion--compact_boost--bidirectional_any` | 1 | boundary | 89.89 | 99.23 | 94.33 | 3299.55 | 3031.15 | 268.40 | 59.94 |
| `shippedUnion--compact_boost--bidirectional_any` | 2 | binary | 77.62 | 99.76 | 87.31 | 4647.22 | 3675.15 | 972.07 | 66.36 |
| `shippedUnion--compact_boost--bidirectional_any` | 2 | boundary | 91.83 | 99.77 | 95.63 | 3937.53 | 3675.15 | 262.38 | 66.36 |
| `shippedUnion--compact_boost--bidirectional_any` | 3 | binary | 80.39 | 99.89 | 89.08 | 5297.05 | 4323.26 | 973.78 | 71.25 |
| `shippedUnion--compact_boost--bidirectional_any` | 3 | boundary | 92.80 | 99.91 | 96.22 | 4608.74 | 4323.26 | 285.48 | 71.25 |
| `shippedUnion--compact_keep--guarded_trim` | 0 | binary | 62.38 | 96.26 | 75.70 | 3683.49 | 2387.15 | 1296.34 | 2.83 |
| `shippedUnion--compact_keep--guarded_trim` | 0 | boundary | 62.46 | 96.26 | 75.76 | 3678.92 | 2387.15 | 1291.77 | 2.83 |
| `shippedUnion--compact_keep--guarded_trim` | 1 | binary | 65.88 | 98.72 | 79.03 | 4477.12 | 3031.15 | 1445.97 | 3.62 |
| `shippedUnion--compact_keep--guarded_trim` | 1 | boundary | 65.88 | 98.72 | 79.02 | 4477.39 | 3031.15 | 1446.24 | 3.62 |
| `shippedUnion--compact_keep--guarded_trim` | 2 | binary | 69.23 | 99.64 | 81.70 | 5208.58 | 3675.15 | 1533.43 | 4.34 |
| `shippedUnion--compact_keep--guarded_trim` | 2 | boundary | 69.15 | 99.64 | 81.64 | 5216.91 | 3675.15 | 1541.75 | 4.34 |
| `shippedUnion--compact_keep--guarded_trim` | 3 | binary | 72.25 | 99.82 | 83.83 | 5895.59 | 4323.26 | 1572.33 | 5.11 |
| `shippedUnion--compact_keep--guarded_trim` | 3 | boundary | 72.15 | 99.85 | 83.77 | 5915.64 | 4323.26 | 1592.38 | 5.11 |
| `shippedUnion--compact_keep--suppression_zero` | 0 | binary | 66.99 | 96.21 | 78.98 | 3428.68 | 2387.15 | 1041.53 | 13.82 |
| `shippedUnion--compact_keep--suppression_zero` | 0 | boundary | 67.35 | 96.26 | 79.25 | 3411.65 | 2387.15 | 1024.50 | 13.82 |
| `shippedUnion--compact_keep--suppression_zero` | 1 | binary | 71.72 | 98.72 | 83.08 | 4109.82 | 3031.15 | 1078.67 | 17.28 |
| `shippedUnion--compact_keep--suppression_zero` | 1 | boundary | 71.95 | 98.72 | 83.23 | 4100.03 | 3031.15 | 1068.88 | 17.28 |
| `shippedUnion--compact_keep--suppression_zero` | 2 | binary | 75.41 | 99.64 | 85.84 | 4777.74 | 3675.15 | 1102.59 | 20.54 |
| `shippedUnion--compact_keep--suppression_zero` | 2 | boundary | 75.87 | 99.64 | 86.15 | 4754.75 | 3675.15 | 1079.60 | 20.54 |
| `shippedUnion--compact_keep--suppression_zero` | 3 | binary | 78.32 | 99.82 | 87.77 | 5431.57 | 4323.26 | 1108.31 | 23.48 |
| `shippedUnion--compact_keep--suppression_zero` | 3 | boundary | 79.13 | 99.85 | 88.29 | 5393.74 | 4323.26 | 1070.48 | 23.48 |
| `shippedUnion--compact_keep--suppression_half` | 0 | binary | 67.56 | 96.20 | 79.37 | 3399.26 | 2387.15 | 1012.11 | 17.37 |
| `shippedUnion--compact_keep--suppression_half` | 0 | boundary | 69.58 | 96.26 | 80.77 | 3303.36 | 2387.15 | 916.21 | 17.37 |
| `shippedUnion--compact_keep--suppression_half` | 1 | binary | 72.35 | 98.72 | 83.50 | 4073.66 | 3031.15 | 1042.51 | 21.19 |
| `shippedUnion--compact_keep--suppression_half` | 1 | boundary | 74.05 | 98.72 | 84.62 | 3984.23 | 3031.15 | 953.08 | 21.19 |
| `shippedUnion--compact_keep--suppression_half` | 2 | binary | 76.02 | 99.64 | 86.24 | 4737.83 | 3675.15 | 1062.68 | 24.86 |
| `shippedUnion--compact_keep--suppression_half` | 2 | boundary | 77.87 | 99.64 | 87.42 | 4633.16 | 3675.15 | 958.01 | 24.86 |
| `shippedUnion--compact_keep--suppression_half` | 3 | binary | 78.84 | 99.82 | 88.10 | 5394.15 | 4323.26 | 1070.89 | 28.05 |
| `shippedUnion--compact_keep--suppression_half` | 3 | boundary | 80.97 | 99.85 | 89.42 | 5272.19 | 4323.26 | 948.93 | 28.05 |
| `shippedUnion--compact_keep--suppression_any` | 0 | binary | 68.90 | 96.19 | 80.29 | 3332.43 | 2387.15 | 945.27 | 46.90 |
| `shippedUnion--compact_keep--suppression_any` | 0 | boundary | 85.74 | 96.26 | 90.69 | 2680.23 | 2387.15 | 293.08 | 46.90 |
| `shippedUnion--compact_keep--suppression_any` | 1 | binary | 73.92 | 98.72 | 84.53 | 3986.64 | 3031.15 | 955.49 | 53.57 |
| `shippedUnion--compact_keep--suppression_any` | 1 | boundary | 88.12 | 98.72 | 93.12 | 3347.64 | 3031.15 | 316.49 | 53.57 |
| `shippedUnion--compact_keep--suppression_any` | 2 | binary | 77.68 | 99.64 | 87.30 | 4635.90 | 3675.15 | 960.75 | 59.45 |
| `shippedUnion--compact_keep--suppression_any` | 2 | boundary | 90.28 | 99.64 | 94.73 | 3995.98 | 3675.15 | 320.83 | 59.45 |
| `shippedUnion--compact_keep--suppression_any` | 3 | binary | 80.43 | 99.82 | 89.08 | 5286.30 | 4323.26 | 963.04 | 64.23 |
| `shippedUnion--compact_keep--suppression_any` | 3 | boundary | 91.43 | 99.85 | 95.45 | 4668.27 | 4323.26 | 345.01 | 64.23 |
| `shippedUnion--compact_keep--bidirectional_half` | 0 | binary | 67.35 | 96.32 | 79.27 | 3414.26 | 2387.15 | 1027.11 | 18.18 |
| `shippedUnion--compact_keep--bidirectional_half` | 0 | boundary | 69.65 | 96.29 | 80.82 | 3301.18 | 2387.15 | 914.03 | 18.18 |
| `shippedUnion--compact_keep--bidirectional_half` | 1 | binary | 72.12 | 98.77 | 83.37 | 4088.87 | 3031.15 | 1057.72 | 22.13 |
| `shippedUnion--compact_keep--bidirectional_half` | 1 | boundary | 74.13 | 98.76 | 84.69 | 3981.72 | 3031.15 | 950.57 | 22.13 |
| `shippedUnion--compact_keep--bidirectional_half` | 2 | binary | 75.83 | 99.67 | 86.13 | 4751.71 | 3675.15 | 1076.56 | 25.94 |
| `shippedUnion--compact_keep--bidirectional_half` | 2 | boundary | 77.96 | 99.68 | 87.49 | 4629.81 | 3675.15 | 954.66 | 25.94 |
| `shippedUnion--compact_keep--bidirectional_half` | 3 | binary | 78.68 | 99.84 | 88.01 | 5407.20 | 4323.26 | 1083.93 | 29.27 |
| `shippedUnion--compact_keep--bidirectional_half` | 3 | boundary | 81.12 | 99.87 | 89.52 | 5264.68 | 4323.26 | 941.42 | 29.27 |
| `shippedUnion--compact_keep--bidirectional_any` | 0 | binary | 68.20 | 97.07 | 80.11 | 3397.95 | 2387.15 | 1010.80 | 51.07 |
| `shippedUnion--compact_keep--bidirectional_any` | 0 | boundary | 86.76 | 96.98 | 91.59 | 2668.54 | 2387.15 | 281.39 | 51.07 |
| `shippedUnion--compact_keep--bidirectional_any` | 1 | binary | 73.30 | 99.22 | 84.31 | 4042.21 | 3031.15 | 1011.06 | 58.22 |
| `shippedUnion--compact_keep--bidirectional_any` | 1 | boundary | 89.09 | 99.21 | 93.87 | 3328.58 | 3031.15 | 297.43 | 58.22 |
| `shippedUnion--compact_keep--bidirectional_any` | 2 | binary | 77.13 | 99.76 | 86.99 | 4679.13 | 3675.15 | 1003.98 | 64.45 |
| `shippedUnion--compact_keep--bidirectional_any` | 2 | boundary | 91.11 | 99.76 | 95.24 | 3967.51 | 3675.15 | 292.35 | 64.45 |
| `shippedUnion--compact_keep--bidirectional_any` | 3 | binary | 79.92 | 99.89 | 88.80 | 5329.74 | 4323.26 | 1006.48 | 69.34 |
| `shippedUnion--compact_keep--bidirectional_any` | 3 | boundary | 92.24 | 99.91 | 95.92 | 4635.76 | 4323.26 | 312.50 | 69.34 |
| `shippedUnion--dino_global--guarded_trim` | 0 | binary | 62.42 | 96.26 | 75.73 | 3680.91 | 2387.15 | 1293.76 | 2.94 |
| `shippedUnion--dino_global--guarded_trim` | 0 | boundary | 62.50 | 96.26 | 75.79 | 3676.34 | 2387.15 | 1289.19 | 2.94 |
| `shippedUnion--dino_global--guarded_trim` | 1 | binary | 65.95 | 98.72 | 79.07 | 4472.53 | 3031.15 | 1441.38 | 3.76 |
| `shippedUnion--dino_global--guarded_trim` | 1 | boundary | 65.94 | 98.72 | 79.07 | 4472.96 | 3031.15 | 1441.81 | 3.76 |
| `shippedUnion--dino_global--guarded_trim` | 2 | binary | 69.28 | 99.56 | 81.70 | 5200.25 | 3675.15 | 1525.10 | 4.52 |
| `shippedUnion--dino_global--guarded_trim` | 2 | boundary | 69.21 | 99.64 | 81.69 | 5211.81 | 3675.15 | 1536.66 | 4.52 |
| `shippedUnion--dino_global--guarded_trim` | 3 | binary | 72.31 | 99.77 | 83.85 | 5886.59 | 4323.26 | 1563.33 | 5.32 |
| `shippedUnion--dino_global--guarded_trim` | 3 | boundary | 72.22 | 99.85 | 83.81 | 5909.88 | 4323.26 | 1586.62 | 5.32 |
| `shippedUnion--dino_global--suppression_zero` | 0 | binary | 67.69 | 96.24 | 79.48 | 3394.22 | 2387.15 | 1007.07 | 13.15 |
| `shippedUnion--dino_global--suppression_zero` | 0 | boundary | 67.83 | 96.26 | 79.58 | 3387.83 | 2387.15 | 1000.68 | 13.15 |
| `shippedUnion--dino_global--suppression_zero` | 1 | binary | 72.47 | 98.72 | 83.58 | 4068.52 | 3031.15 | 1037.37 | 16.53 |
| `shippedUnion--dino_global--suppression_zero` | 1 | boundary | 72.54 | 98.72 | 83.63 | 4066.27 | 3031.15 | 1035.12 | 16.53 |
| `shippedUnion--dino_global--suppression_zero` | 2 | binary | 76.18 | 99.60 | 86.32 | 4726.51 | 3675.15 | 1051.36 | 19.59 |
| `shippedUnion--dino_global--suppression_zero` | 2 | boundary | 76.61 | 99.64 | 86.62 | 4709.06 | 3675.15 | 1033.91 | 19.59 |
| `shippedUnion--dino_global--suppression_zero` | 3 | binary | 79.11 | 99.79 | 88.25 | 5373.98 | 4323.26 | 1050.72 | 22.57 |
| `shippedUnion--dino_global--suppression_zero` | 3 | boundary | 79.98 | 99.85 | 88.81 | 5336.87 | 4323.26 | 1013.61 | 22.57 |
| `shippedUnion--dino_global--suppression_half` | 0 | binary | 68.65 | 96.22 | 80.13 | 3345.93 | 2387.15 | 958.77 | 16.68 |
| `shippedUnion--dino_global--suppression_half` | 0 | boundary | 70.46 | 96.26 | 81.36 | 3261.78 | 2387.15 | 874.63 | 16.68 |
| `shippedUnion--dino_global--suppression_half` | 1 | binary | 73.59 | 98.72 | 84.32 | 4005.79 | 3031.15 | 974.64 | 20.46 |
| `shippedUnion--dino_global--suppression_half` | 1 | boundary | 75.11 | 98.72 | 85.31 | 3927.67 | 3031.15 | 896.52 | 20.46 |
| `shippedUnion--dino_global--suppression_half` | 2 | binary | 77.28 | 99.56 | 87.01 | 4656.03 | 3675.15 | 980.88 | 24.00 |
| `shippedUnion--dino_global--suppression_half` | 2 | boundary | 79.05 | 99.64 | 88.16 | 4563.64 | 3675.15 | 888.49 | 24.00 |
| `shippedUnion--dino_global--suppression_half` | 3 | binary | 80.11 | 99.77 | 88.86 | 5304.07 | 4323.26 | 980.81 | 27.25 |
| `shippedUnion--dino_global--suppression_half` | 3 | boundary | 82.23 | 99.85 | 90.18 | 5191.04 | 4323.26 | 867.77 | 27.25 |
| `shippedUnion--dino_global--suppression_any` | 0 | binary | 69.28 | 96.19 | 80.55 | 3314.08 | 2387.15 | 926.93 | 45.53 |
| `shippedUnion--dino_global--suppression_any` | 0 | boundary | 86.87 | 96.26 | 91.32 | 2645.57 | 2387.15 | 258.42 | 45.53 |
| `shippedUnion--dino_global--suppression_any` | 1 | binary | 74.32 | 98.72 | 84.80 | 3964.69 | 3031.15 | 933.54 | 51.97 |
| `shippedUnion--dino_global--suppression_any` | 1 | boundary | 89.09 | 98.72 | 93.65 | 3311.41 | 3031.15 | 280.25 | 51.97 |
| `shippedUnion--dino_global--suppression_any` | 2 | binary | 78.05 | 99.56 | 87.50 | 4608.54 | 3675.15 | 933.39 | 57.83 |
| `shippedUnion--dino_global--suppression_any` | 2 | boundary | 91.01 | 99.64 | 95.13 | 3963.93 | 3675.15 | 288.78 | 57.83 |
| `shippedUnion--dino_global--suppression_any` | 3 | binary | 80.82 | 99.77 | 89.30 | 5255.78 | 4323.26 | 932.51 | 62.73 |
| `shippedUnion--dino_global--suppression_any` | 3 | boundary | 92.09 | 99.85 | 95.81 | 4634.77 | 4323.26 | 311.50 | 62.73 |
| `shippedUnion--dino_global--bidirectional_half` | 0 | binary | 68.52 | 96.22 | 80.04 | 3352.25 | 2387.15 | 965.10 | 17.17 |
| `shippedUnion--dino_global--bidirectional_half` | 0 | boundary | 70.49 | 96.26 | 81.38 | 3259.97 | 2387.15 | 872.82 | 17.17 |
| `shippedUnion--dino_global--bidirectional_half` | 1 | binary | 73.47 | 98.72 | 84.24 | 4012.11 | 3031.15 | 980.96 | 21.06 |
| `shippedUnion--dino_global--bidirectional_half` | 1 | boundary | 75.16 | 98.72 | 85.34 | 3925.49 | 3031.15 | 894.34 | 21.06 |
| `shippedUnion--dino_global--bidirectional_half` | 2 | binary | 77.18 | 99.56 | 86.95 | 4662.36 | 3675.15 | 987.21 | 24.69 |
| `shippedUnion--dino_global--bidirectional_half` | 2 | boundary | 79.10 | 99.64 | 88.19 | 4562.57 | 3675.15 | 887.42 | 24.69 |
| `shippedUnion--dino_global--bidirectional_half` | 3 | binary | 80.01 | 99.77 | 88.80 | 5310.90 | 4323.26 | 987.64 | 28.03 |
| `shippedUnion--dino_global--bidirectional_half` | 3 | boundary | 82.29 | 99.85 | 90.22 | 5189.61 | 4323.26 | 866.34 | 28.03 |
| `shippedUnion--dino_global--bidirectional_any` | 0 | binary | 68.90 | 97.43 | 80.72 | 3375.47 | 2387.15 | 988.32 | 50.16 |
| `shippedUnion--dino_global--bidirectional_any` | 0 | boundary | 87.83 | 97.46 | 92.39 | 2649.29 | 2387.15 | 262.14 | 50.16 |
| `shippedUnion--dino_global--bidirectional_any` | 1 | binary | 73.91 | 99.35 | 84.76 | 4018.33 | 3031.15 | 987.18 | 56.78 |
| `shippedUnion--dino_global--bidirectional_any` | 1 | boundary | 89.84 | 99.36 | 94.36 | 3308.90 | 3031.15 | 277.74 | 56.78 |
| `shippedUnion--dino_global--bidirectional_any` | 2 | binary | 77.74 | 99.82 | 87.40 | 4650.25 | 3675.15 | 975.10 | 62.90 |
| `shippedUnion--dino_global--bidirectional_any` | 2 | boundary | 91.64 | 99.83 | 95.56 | 3951.10 | 3675.15 | 275.95 | 62.90 |
| `shippedUnion--dino_global--bidirectional_any` | 3 | binary | 80.56 | 99.90 | 89.19 | 5295.24 | 4323.26 | 971.98 | 67.96 |
| `shippedUnion--dino_global--bidirectional_any` | 3 | boundary | 92.58 | 99.91 | 96.11 | 4621.88 | 4323.26 | 298.61 | 67.96 |
| `shippedUnion--dino_boost--guarded_trim` | 0 | binary | 62.39 | 96.26 | 75.71 | 3682.66 | 2387.15 | 1295.51 | 2.87 |
| `shippedUnion--dino_boost--guarded_trim` | 0 | boundary | 62.47 | 96.26 | 75.77 | 3678.09 | 2387.15 | 1290.94 | 2.87 |
| `shippedUnion--dino_boost--guarded_trim` | 1 | binary | 65.91 | 98.72 | 79.04 | 4475.62 | 3031.15 | 1444.47 | 3.66 |
| `shippedUnion--dino_boost--guarded_trim` | 1 | boundary | 65.90 | 98.72 | 79.04 | 4475.89 | 3031.15 | 1444.74 | 3.66 |
| `shippedUnion--dino_boost--guarded_trim` | 2 | binary | 69.26 | 99.64 | 81.72 | 5205.78 | 3675.15 | 1530.63 | 4.40 |
| `shippedUnion--dino_boost--guarded_trim` | 2 | boundary | 69.17 | 99.64 | 81.66 | 5214.74 | 3675.15 | 1539.59 | 4.40 |
| `shippedUnion--dino_boost--guarded_trim` | 3 | binary | 72.29 | 99.82 | 83.85 | 5892.80 | 4323.26 | 1569.53 | 5.18 |
| `shippedUnion--dino_boost--guarded_trim` | 3 | boundary | 72.18 | 99.85 | 83.79 | 5912.81 | 4323.26 | 1589.55 | 5.18 |
| `shippedUnion--dino_boost--suppression_zero` | 0 | binary | 68.00 | 96.24 | 79.69 | 3378.71 | 2387.15 | 991.56 | 13.13 |
| `shippedUnion--dino_boost--suppression_zero` | 0 | boundary | 67.97 | 96.26 | 79.68 | 3380.72 | 2387.15 | 993.57 | 13.13 |
| `shippedUnion--dino_boost--suppression_zero` | 1 | binary | 72.93 | 98.72 | 83.89 | 4042.09 | 3031.15 | 1010.94 | 16.43 |
| `shippedUnion--dino_boost--suppression_zero` | 1 | boundary | 72.81 | 98.72 | 83.81 | 4051.38 | 3031.15 | 1020.23 | 16.43 |
| `shippedUnion--dino_boost--suppression_zero` | 2 | binary | 76.71 | 99.63 | 86.68 | 4694.49 | 3675.15 | 1019.34 | 19.49 |
| `shippedUnion--dino_boost--suppression_zero` | 2 | boundary | 76.94 | 99.64 | 86.83 | 4688.61 | 3675.15 | 1013.46 | 19.49 |
| `shippedUnion--dino_boost--suppression_zero` | 3 | binary | 79.51 | 99.82 | 88.51 | 5348.39 | 4323.26 | 1025.12 | 22.48 |
| `shippedUnion--dino_boost--suppression_zero` | 3 | boundary | 80.28 | 99.85 | 89.00 | 5316.31 | 4323.26 | 993.05 | 22.48 |
| `shippedUnion--dino_boost--suppression_half` | 0 | binary | 68.69 | 96.22 | 80.16 | 3343.92 | 2387.15 | 956.77 | 17.15 |
| `shippedUnion--dino_boost--suppression_half` | 0 | boundary | 71.18 | 96.26 | 81.84 | 3228.46 | 2387.15 | 841.31 | 17.15 |
| `shippedUnion--dino_boost--suppression_half` | 1 | binary | 73.69 | 98.72 | 84.38 | 4000.18 | 3031.15 | 969.03 | 20.76 |
| `shippedUnion--dino_boost--suppression_half` | 1 | boundary | 75.78 | 98.72 | 85.74 | 3892.53 | 3031.15 | 861.38 | 20.76 |
| `shippedUnion--dino_boost--suppression_half` | 2 | binary | 77.44 | 99.63 | 87.14 | 4649.48 | 3675.15 | 974.33 | 24.20 |
| `shippedUnion--dino_boost--suppression_half` | 2 | boundary | 79.67 | 99.64 | 88.54 | 4528.28 | 3675.15 | 853.13 | 24.20 |
| `shippedUnion--dino_boost--suppression_half` | 3 | binary | 80.16 | 99.82 | 88.92 | 5304.02 | 4323.26 | 980.75 | 27.40 |
| `shippedUnion--dino_boost--suppression_half` | 3 | boundary | 82.68 | 99.85 | 90.45 | 5162.55 | 4323.26 | 839.29 | 27.40 |
| `shippedUnion--dino_boost--suppression_any` | 0 | binary | 69.33 | 96.19 | 80.58 | 3312.14 | 2387.15 | 924.99 | 44.43 |
| `shippedUnion--dino_boost--suppression_any` | 0 | boundary | 85.72 | 96.26 | 90.68 | 2680.96 | 2387.15 | 293.81 | 44.43 |
| `shippedUnion--dino_boost--suppression_any` | 1 | binary | 74.42 | 98.72 | 84.86 | 3959.52 | 3031.15 | 928.37 | 50.71 |
| `shippedUnion--dino_boost--suppression_any` | 1 | boundary | 88.27 | 98.72 | 93.20 | 3341.93 | 3031.15 | 310.78 | 50.71 |
| `shippedUnion--dino_boost--suppression_any` | 2 | binary | 78.20 | 99.63 | 87.63 | 4602.62 | 3675.15 | 927.47 | 56.32 |
| `shippedUnion--dino_boost--suppression_any` | 2 | boundary | 90.48 | 99.64 | 94.84 | 3987.19 | 3675.15 | 312.04 | 56.32 |
| `shippedUnion--dino_boost--suppression_any` | 3 | binary | 80.92 | 99.82 | 89.38 | 5252.53 | 4323.26 | 929.26 | 61.16 |
| `shippedUnion--dino_boost--suppression_any` | 3 | boundary | 91.58 | 99.85 | 95.54 | 4660.57 | 4323.26 | 337.30 | 61.16 |
| `shippedUnion--dino_boost--bidirectional_half` | 0 | binary | 68.69 | 96.22 | 80.16 | 3344.09 | 2387.15 | 956.94 | 17.70 |
| `shippedUnion--dino_boost--bidirectional_half` | 0 | boundary | 71.18 | 96.26 | 81.84 | 3228.63 | 2387.15 | 841.48 | 17.70 |
| `shippedUnion--dino_boost--bidirectional_half` | 1 | binary | 73.69 | 98.72 | 84.38 | 4000.35 | 3031.15 | 969.19 | 21.50 |
| `shippedUnion--dino_boost--bidirectional_half` | 1 | boundary | 75.80 | 98.72 | 85.75 | 3891.92 | 3031.15 | 860.77 | 21.50 |
| `shippedUnion--dino_boost--bidirectional_half` | 2 | binary | 77.44 | 99.63 | 87.14 | 4649.65 | 3675.15 | 974.50 | 25.13 |
| `shippedUnion--dino_boost--bidirectional_half` | 2 | boundary | 79.71 | 99.64 | 88.57 | 4526.16 | 3675.15 | 851.01 | 25.13 |
| `shippedUnion--dino_boost--bidirectional_half` | 3 | binary | 80.16 | 99.82 | 88.92 | 5304.18 | 4323.26 | 980.92 | 28.52 |
| `shippedUnion--dino_boost--bidirectional_half` | 3 | boundary | 82.76 | 99.85 | 90.50 | 5159.03 | 4323.26 | 835.77 | 28.52 |
| `shippedUnion--dino_boost--bidirectional_any` | 0 | binary | 68.48 | 97.36 | 80.40 | 3393.89 | 2387.15 | 1006.74 | 50.25 |
| `shippedUnion--dino_boost--bidirectional_any` | 0 | boundary | 87.05 | 97.40 | 91.93 | 2671.33 | 2387.15 | 284.18 | 50.25 |
| `shippedUnion--dino_boost--bidirectional_any` | 1 | binary | 73.62 | 99.34 | 84.57 | 4031.60 | 3031.15 | 1000.45 | 57.07 |
| `shippedUnion--dino_boost--bidirectional_any` | 1 | boundary | 89.31 | 99.34 | 94.06 | 3326.40 | 3031.15 | 295.25 | 57.07 |
| `shippedUnion--dino_boost--bidirectional_any` | 2 | binary | 77.53 | 99.80 | 87.27 | 4659.28 | 3675.15 | 984.13 | 63.16 |
| `shippedUnion--dino_boost--bidirectional_any` | 2 | boundary | 91.40 | 99.81 | 95.42 | 3959.72 | 3675.15 | 284.56 | 63.16 |
| `shippedUnion--dino_boost--bidirectional_any` | 3 | binary | 80.34 | 99.92 | 89.06 | 5306.43 | 4323.26 | 983.17 | 68.32 |
| `shippedUnion--dino_boost--bidirectional_any` | 3 | boundary | 92.33 | 99.94 | 95.99 | 4635.49 | 4323.26 | 312.23 | 68.32 |
| `shippedUnion--dino_keep--guarded_trim` | 0 | binary | 62.41 | 96.26 | 75.72 | 3681.74 | 2387.15 | 1294.59 | 2.90 |
| `shippedUnion--dino_keep--guarded_trim` | 0 | boundary | 62.49 | 96.26 | 75.78 | 3677.17 | 2387.15 | 1290.02 | 2.90 |
| `shippedUnion--dino_keep--guarded_trim` | 1 | binary | 65.93 | 98.72 | 79.06 | 4474.03 | 3031.15 | 1442.88 | 3.71 |
| `shippedUnion--dino_keep--guarded_trim` | 1 | boundary | 65.92 | 98.72 | 79.06 | 4474.31 | 3031.15 | 1443.16 | 3.71 |
| `shippedUnion--dino_keep--guarded_trim` | 2 | binary | 69.29 | 99.64 | 81.74 | 5203.53 | 3675.15 | 1528.38 | 4.46 |
| `shippedUnion--dino_keep--guarded_trim` | 2 | boundary | 69.20 | 99.64 | 81.68 | 5212.49 | 3675.15 | 1537.34 | 4.46 |
| `shippedUnion--dino_keep--guarded_trim` | 3 | binary | 72.32 | 99.82 | 83.88 | 5889.88 | 4323.26 | 1566.62 | 5.25 |
| `shippedUnion--dino_keep--guarded_trim` | 3 | boundary | 72.22 | 99.85 | 83.81 | 5909.89 | 4323.26 | 1586.63 | 5.25 |
| `shippedUnion--dino_keep--suppression_zero` | 0 | binary | 68.05 | 96.24 | 79.73 | 3376.40 | 2387.15 | 989.25 | 13.18 |
| `shippedUnion--dino_keep--suppression_zero` | 0 | boundary | 67.96 | 96.26 | 79.66 | 3381.82 | 2387.15 | 994.67 | 13.18 |
| `shippedUnion--dino_keep--suppression_zero` | 1 | binary | 72.99 | 98.72 | 83.92 | 4039.61 | 3031.15 | 1008.46 | 16.54 |
| `shippedUnion--dino_keep--suppression_zero` | 1 | boundary | 72.72 | 98.72 | 83.74 | 4057.03 | 3031.15 | 1025.88 | 16.54 |
| `shippedUnion--dino_keep--suppression_zero` | 2 | binary | 76.75 | 99.63 | 86.70 | 4693.55 | 3675.15 | 1018.39 | 19.69 |
| `shippedUnion--dino_keep--suppression_zero` | 2 | boundary | 76.89 | 99.64 | 86.80 | 4692.15 | 3675.15 | 1017.00 | 19.69 |
| `shippedUnion--dino_keep--suppression_zero` | 3 | binary | 79.58 | 99.82 | 88.56 | 5344.15 | 4323.26 | 1020.89 | 22.72 |
| `shippedUnion--dino_keep--suppression_zero` | 3 | boundary | 80.28 | 99.85 | 89.00 | 5317.44 | 4323.26 | 994.18 | 22.72 |
| `shippedUnion--dino_keep--suppression_half` | 0 | binary | 68.47 | 96.23 | 80.01 | 3355.17 | 2387.15 | 968.02 | 16.85 |
| `shippedUnion--dino_keep--suppression_half` | 0 | boundary | 70.70 | 96.26 | 81.51 | 3251.76 | 2387.15 | 864.61 | 16.85 |
| `shippedUnion--dino_keep--suppression_half` | 1 | binary | 73.44 | 98.72 | 84.22 | 4014.19 | 3031.15 | 983.04 | 20.49 |
| `shippedUnion--dino_keep--suppression_half` | 1 | boundary | 75.26 | 98.72 | 85.40 | 3920.77 | 3031.15 | 889.62 | 20.49 |
| `shippedUnion--dino_keep--suppression_half` | 2 | binary | 77.19 | 99.63 | 86.99 | 4666.08 | 3675.15 | 990.93 | 23.99 |
| `shippedUnion--dino_keep--suppression_half` | 2 | boundary | 79.20 | 99.64 | 88.25 | 4556.00 | 3675.15 | 880.84 | 23.99 |
| `shippedUnion--dino_keep--suppression_half` | 3 | binary | 80.00 | 99.82 | 88.82 | 5315.83 | 4323.26 | 992.57 | 27.19 |
| `shippedUnion--dino_keep--suppression_half` | 3 | boundary | 82.31 | 99.85 | 90.23 | 5186.46 | 4323.26 | 863.20 | 27.19 |
| `shippedUnion--dino_keep--suppression_any` | 0 | binary | 69.17 | 96.19 | 80.47 | 3319.52 | 2387.15 | 932.37 | 47.36 |
| `shippedUnion--dino_keep--suppression_any` | 0 | boundary | 85.80 | 96.26 | 90.71 | 2680.52 | 2387.15 | 293.37 | 47.36 |
| `shippedUnion--dino_keep--suppression_any` | 1 | binary | 74.26 | 98.72 | 84.76 | 3968.28 | 3031.15 | 937.12 | 54.05 |
| `shippedUnion--dino_keep--suppression_any` | 1 | boundary | 88.21 | 98.72 | 93.15 | 3346.21 | 3031.15 | 315.06 | 54.05 |
| `shippedUnion--dino_keep--suppression_any` | 2 | binary | 78.03 | 99.63 | 87.52 | 4614.24 | 3675.15 | 939.09 | 60.14 |
| `shippedUnion--dino_keep--suppression_any` | 2 | boundary | 90.45 | 99.64 | 94.81 | 3990.02 | 3675.15 | 314.87 | 60.14 |
| `shippedUnion--dino_keep--suppression_any` | 3 | binary | 80.80 | 99.82 | 89.31 | 5261.48 | 4323.26 | 938.22 | 65.25 |
| `shippedUnion--dino_keep--suppression_any` | 3 | boundary | 91.63 | 99.85 | 95.55 | 4659.48 | 4323.26 | 336.21 | 65.25 |
| `shippedUnion--dino_keep--bidirectional_half` | 0 | binary | 68.46 | 96.23 | 80.00 | 3355.35 | 2387.15 | 968.19 | 17.23 |
| `shippedUnion--dino_keep--bidirectional_half` | 0 | boundary | 70.70 | 96.26 | 81.51 | 3251.76 | 2387.15 | 864.61 | 17.23 |
| `shippedUnion--dino_keep--bidirectional_half` | 1 | binary | 73.44 | 98.72 | 84.22 | 4014.36 | 3031.15 | 983.21 | 21.01 |
| `shippedUnion--dino_keep--bidirectional_half` | 1 | boundary | 75.26 | 98.72 | 85.40 | 3920.81 | 3031.15 | 889.66 | 21.01 |
| `shippedUnion--dino_keep--bidirectional_half` | 2 | binary | 77.19 | 99.63 | 86.98 | 4666.25 | 3675.15 | 991.10 | 24.64 |
| `shippedUnion--dino_keep--bidirectional_half` | 2 | boundary | 79.23 | 99.64 | 88.26 | 4554.73 | 3675.15 | 879.58 | 24.64 |
| `shippedUnion--dino_keep--bidirectional_half` | 3 | binary | 80.00 | 99.82 | 88.81 | 5316.00 | 4323.26 | 992.74 | 27.97 |
| `shippedUnion--dino_keep--bidirectional_half` | 3 | boundary | 82.39 | 99.85 | 90.28 | 5182.05 | 4323.26 | 858.79 | 27.97 |
| `shippedUnion--dino_keep--bidirectional_any` | 0 | binary | 68.01 | 97.40 | 80.09 | 3420.09 | 2387.15 | 1032.94 | 54.65 |
| `shippedUnion--dino_keep--bidirectional_any` | 0 | boundary | 87.49 | 97.44 | 92.19 | 2659.47 | 2387.15 | 272.32 | 54.65 |
| `shippedUnion--dino_keep--bidirectional_any` | 1 | binary | 73.19 | 99.35 | 84.28 | 4057.88 | 3031.15 | 1026.73 | 61.87 |
| `shippedUnion--dino_keep--bidirectional_any` | 1 | boundary | 89.59 | 99.35 | 94.21 | 3317.85 | 3031.15 | 286.70 | 61.87 |
| `shippedUnion--dino_keep--bidirectional_any` | 2 | binary | 77.09 | 99.82 | 86.99 | 4689.30 | 3675.15 | 1014.14 | 68.47 |
| `shippedUnion--dino_keep--bidirectional_any` | 2 | boundary | 91.66 | 99.83 | 95.56 | 3950.45 | 3675.15 | 275.30 | 68.47 |
| `shippedUnion--dino_keep--bidirectional_any` | 3 | binary | 79.96 | 99.90 | 88.82 | 5335.00 | 4323.26 | 1011.73 | 73.93 |
| `shippedUnion--dino_keep--bidirectional_any` | 3 | boundary | 92.66 | 99.91 | 96.15 | 4618.06 | 4323.26 | 294.79 | 73.93 |
| `refitUnion--compact_boost--guarded_trim` | 0 | binary | 52.58 | 92.76 | 67.12 | 4211.11 | 2387.15 | 1823.96 | 5.51 |
| `refitUnion--compact_boost--guarded_trim` | 0 | boundary | 52.81 | 92.76 | 67.31 | 4192.61 | 2387.15 | 1805.46 | 5.51 |
| `refitUnion--compact_boost--guarded_trim` | 1 | binary | 56.59 | 96.74 | 71.41 | 5084.24 | 3031.15 | 2053.09 | 6.71 |
| `refitUnion--compact_boost--guarded_trim` | 1 | boundary | 56.75 | 96.81 | 71.55 | 5078.95 | 3031.15 | 2047.80 | 6.71 |
| `refitUnion--compact_boost--guarded_trim` | 2 | binary | 61.11 | 98.00 | 75.28 | 5771.12 | 3675.15 | 2095.97 | 7.87 |
| `refitUnion--compact_boost--guarded_trim` | 2 | boundary | 61.13 | 98.32 | 75.39 | 5795.02 | 3675.15 | 2119.87 | 7.87 |
| `refitUnion--compact_boost--guarded_trim` | 3 | binary | 65.84 | 98.89 | 79.05 | 6347.69 | 4323.26 | 2024.42 | 8.95 |
| `refitUnion--compact_boost--guarded_trim` | 3 | boundary | 65.94 | 99.16 | 79.21 | 6376.46 | 4323.26 | 2053.19 | 8.95 |
| `refitUnion--compact_boost--suppression_zero` | 0 | binary | 59.32 | 92.70 | 72.34 | 3730.61 | 2387.15 | 1343.46 | 20.29 |
| `refitUnion--compact_boost--suppression_zero` | 0 | boundary | 59.16 | 92.76 | 72.24 | 3743.18 | 2387.15 | 1356.03 | 20.29 |
| `refitUnion--compact_boost--suppression_zero` | 1 | binary | 65.09 | 96.73 | 77.82 | 4409.86 | 3031.15 | 1378.71 | 24.58 |
| `refitUnion--compact_boost--suppression_zero` | 1 | boundary | 64.56 | 96.81 | 77.46 | 4464.13 | 3031.15 | 1432.97 | 24.58 |
| `refitUnion--compact_boost--suppression_zero` | 2 | binary | 69.77 | 97.95 | 81.49 | 5029.11 | 3675.15 | 1353.96 | 28.66 |
| `refitUnion--compact_boost--suppression_zero` | 2 | boundary | 70.17 | 98.32 | 81.89 | 5048.58 | 3675.15 | 1373.43 | 28.66 |
| `refitUnion--compact_boost--suppression_zero` | 3 | binary | 73.94 | 98.68 | 84.54 | 5609.94 | 4323.26 | 1286.68 | 32.56 |
| `refitUnion--compact_boost--suppression_zero` | 3 | boundary | 75.47 | 99.16 | 85.70 | 5571.68 | 4323.26 | 1248.42 | 32.56 |
| `refitUnion--compact_boost--suppression_half` | 0 | binary | 61.19 | 92.67 | 73.71 | 3615.23 | 2387.15 | 1228.08 | 29.76 |
| `refitUnion--compact_boost--suppression_half` | 0 | boundary | 65.51 | 92.76 | 76.78 | 3381.03 | 2387.15 | 993.88 | 29.76 |
| `refitUnion--compact_boost--suppression_half` | 1 | binary | 67.03 | 96.72 | 79.18 | 4279.81 | 3031.15 | 1248.66 | 34.60 |
| `refitUnion--compact_boost--suppression_half` | 1 | boundary | 70.32 | 96.81 | 81.46 | 4099.26 | 3031.15 | 1068.11 | 34.60 |
| `refitUnion--compact_boost--suppression_half` | 2 | binary | 71.63 | 97.95 | 82.75 | 4895.44 | 3675.15 | 1220.29 | 39.28 |
| `refitUnion--compact_boost--suppression_half` | 2 | boundary | 75.44 | 98.32 | 85.37 | 4696.32 | 3675.15 | 1021.16 | 39.28 |
| `refitUnion--compact_boost--suppression_half` | 3 | binary | 75.61 | 98.68 | 85.62 | 5482.96 | 4323.26 | 1159.70 | 43.48 |
| `refitUnion--compact_boost--suppression_half` | 3 | boundary | 80.20 | 99.16 | 88.67 | 5243.22 | 4323.26 | 919.95 | 43.48 |
| `refitUnion--compact_boost--suppression_any` | 0 | binary | 62.34 | 92.65 | 74.53 | 3547.69 | 2387.15 | 1160.54 | 66.44 |
| `refitUnion--compact_boost--suppression_any` | 0 | boundary | 89.43 | 92.76 | 91.06 | 2476.12 | 2387.15 | 88.97 | 66.44 |
| `refitUnion--compact_boost--suppression_any` | 1 | binary | 68.32 | 96.72 | 80.07 | 4198.01 | 3031.15 | 1166.86 | 73.21 |
| `refitUnion--compact_boost--suppression_any` | 1 | boundary | 91.10 | 96.81 | 93.87 | 3163.82 | 3031.15 | 132.67 | 73.21 |
| `refitUnion--compact_boost--suppression_any` | 2 | binary | 73.00 | 97.95 | 83.66 | 4802.18 | 3675.15 | 1127.02 | 79.43 |
| `refitUnion--compact_boost--suppression_any` | 2 | boundary | 93.10 | 98.32 | 95.64 | 3805.20 | 3675.15 | 130.04 | 79.43 |
| `refitUnion--compact_boost--suppression_any` | 3 | binary | 76.94 | 98.68 | 86.47 | 5387.02 | 4323.26 | 1063.76 | 84.31 |
| `refitUnion--compact_boost--suppression_any` | 3 | boundary | 94.47 | 99.16 | 96.76 | 4450.88 | 4323.26 | 127.62 | 84.31 |
| `refitUnion--compact_boost--bidirectional_half` | 0 | binary | 61.02 | 93.16 | 73.74 | 3644.46 | 2387.15 | 1257.31 | 30.49 |
| `refitUnion--compact_boost--bidirectional_half` | 0 | boundary | 65.72 | 93.21 | 77.08 | 3386.64 | 2387.15 | 999.48 | 30.49 |
| `refitUnion--compact_boost--bidirectional_half` | 1 | binary | 66.86 | 97.22 | 79.23 | 4309.75 | 3031.15 | 1278.60 | 35.34 |
| `refitUnion--compact_boost--bidirectional_half` | 1 | boundary | 70.50 | 97.32 | 81.76 | 4106.54 | 3031.15 | 1075.39 | 35.34 |
| `refitUnion--compact_boost--bidirectional_half` | 2 | binary | 71.44 | 98.38 | 82.77 | 4924.71 | 3675.15 | 1249.56 | 40.04 |
| `refitUnion--compact_boost--bidirectional_half` | 2 | boundary | 75.56 | 98.74 | 85.60 | 4704.22 | 3675.15 | 1029.07 | 40.04 |
| `refitUnion--compact_boost--bidirectional_half` | 3 | binary | 75.42 | 99.03 | 85.63 | 5511.57 | 4323.26 | 1188.31 | 44.26 |
| `refitUnion--compact_boost--bidirectional_half` | 3 | boundary | 80.27 | 99.48 | 88.85 | 5250.96 | 4323.26 | 927.70 | 44.26 |
| `refitUnion--compact_boost--bidirectional_any` | 0 | binary | 62.42 | 94.94 | 75.32 | 3630.76 | 2387.15 | 1243.60 | 70.71 |
| `refitUnion--compact_boost--bidirectional_any` | 0 | boundary | 90.26 | 94.94 | 92.54 | 2511.03 | 2387.15 | 123.88 | 70.71 |
| `refitUnion--compact_boost--bidirectional_any` | 1 | binary | 68.24 | 98.26 | 80.54 | 4270.47 | 3031.15 | 1239.32 | 77.84 |
| `refitUnion--compact_boost--bidirectional_any` | 1 | boundary | 91.79 | 98.36 | 94.96 | 3189.94 | 3031.15 | 158.79 | 77.84 |
| `refitUnion--compact_boost--bidirectional_any` | 2 | binary | 72.85 | 99.01 | 83.94 | 4872.33 | 3675.15 | 1197.18 | 84.20 |
| `refitUnion--compact_boost--bidirectional_any` | 2 | boundary | 93.56 | 99.37 | 96.38 | 3832.13 | 3675.15 | 156.98 | 84.20 |
| `refitUnion--compact_boost--bidirectional_any` | 3 | binary | 76.74 | 99.32 | 86.58 | 5456.98 | 4323.26 | 1133.72 | 89.17 |
| `refitUnion--compact_boost--bidirectional_any` | 3 | boundary | 94.95 | 99.73 | 97.28 | 4467.89 | 4323.26 | 144.63 | 89.17 |
| `refitUnion--compact_keep--guarded_trim` | 0 | binary | 52.58 | 92.76 | 67.12 | 4211.11 | 2387.15 | 1823.96 | 5.38 |
| `refitUnion--compact_keep--guarded_trim` | 0 | boundary | 52.81 | 92.76 | 67.30 | 4192.79 | 2387.15 | 1805.64 | 5.38 |
| `refitUnion--compact_keep--guarded_trim` | 1 | binary | 56.59 | 96.74 | 71.41 | 5084.24 | 3031.15 | 2053.09 | 6.54 |
| `refitUnion--compact_keep--guarded_trim` | 1 | boundary | 56.74 | 96.81 | 71.55 | 5079.13 | 3031.15 | 2047.97 | 6.54 |
| `refitUnion--compact_keep--guarded_trim` | 2 | binary | 61.11 | 98.00 | 75.28 | 5771.12 | 3675.15 | 2095.97 | 7.69 |
| `refitUnion--compact_keep--guarded_trim` | 2 | boundary | 61.13 | 98.32 | 75.39 | 5795.19 | 3675.15 | 2120.04 | 7.69 |
| `refitUnion--compact_keep--guarded_trim` | 3 | binary | 65.84 | 98.89 | 79.05 | 6347.69 | 4323.26 | 2024.42 | 8.76 |
| `refitUnion--compact_keep--guarded_trim` | 3 | boundary | 65.93 | 99.16 | 79.20 | 6377.38 | 4323.26 | 2054.12 | 8.76 |
| `refitUnion--compact_keep--suppression_zero` | 0 | binary | 58.68 | 92.72 | 71.87 | 3772.57 | 2387.15 | 1385.42 | 18.66 |
| `refitUnion--compact_keep--suppression_zero` | 0 | boundary | 58.43 | 92.76 | 71.70 | 3789.95 | 2387.15 | 1402.79 | 18.66 |
| `refitUnion--compact_keep--suppression_zero` | 1 | binary | 64.29 | 96.73 | 77.24 | 4466.19 | 3031.15 | 1435.04 | 22.63 |
| `refitUnion--compact_keep--suppression_zero` | 1 | boundary | 63.72 | 96.81 | 76.85 | 4523.42 | 3031.15 | 1492.27 | 22.63 |
| `refitUnion--compact_keep--suppression_zero` | 2 | binary | 68.90 | 97.95 | 80.89 | 5095.37 | 3675.15 | 1420.22 | 26.37 |
| `refitUnion--compact_keep--suppression_zero` | 2 | boundary | 69.22 | 98.32 | 81.24 | 5118.10 | 3675.15 | 1442.95 | 26.37 |
| `refitUnion--compact_keep--suppression_zero` | 3 | binary | 73.09 | 98.68 | 83.98 | 5677.83 | 4323.26 | 1354.57 | 30.04 |
| `refitUnion--compact_keep--suppression_zero` | 3 | boundary | 74.43 | 99.16 | 85.03 | 5649.32 | 4323.26 | 1326.06 | 30.04 |
| `refitUnion--compact_keep--suppression_half` | 0 | binary | 60.74 | 92.69 | 73.38 | 3643.22 | 2387.15 | 1256.07 | 28.50 |
| `refitUnion--compact_keep--suppression_half` | 0 | boundary | 64.73 | 92.76 | 76.25 | 3422.11 | 2387.15 | 1034.96 | 28.50 |
| `refitUnion--compact_keep--suppression_half` | 1 | binary | 66.47 | 96.72 | 78.79 | 4317.42 | 3031.15 | 1286.27 | 33.10 |
| `refitUnion--compact_keep--suppression_half` | 1 | boundary | 69.55 | 96.81 | 80.94 | 4144.90 | 3031.15 | 1113.74 | 33.10 |
| `refitUnion--compact_keep--suppression_half` | 2 | binary | 71.04 | 97.95 | 82.35 | 4937.38 | 3675.15 | 1262.23 | 37.57 |
| `refitUnion--compact_keep--suppression_half` | 2 | boundary | 74.62 | 98.32 | 84.84 | 4748.38 | 3675.15 | 1073.22 | 37.57 |
| `refitUnion--compact_keep--suppression_half` | 3 | binary | 75.03 | 98.68 | 85.25 | 5526.46 | 4323.26 | 1203.20 | 41.59 |
| `refitUnion--compact_keep--suppression_half` | 3 | boundary | 79.34 | 99.16 | 88.15 | 5300.17 | 4323.26 | 976.90 | 41.59 |
| `refitUnion--compact_keep--suppression_any` | 0 | binary | 62.22 | 92.65 | 74.44 | 3554.85 | 2387.15 | 1167.70 | 64.39 |
| `refitUnion--compact_keep--suppression_any` | 0 | boundary | 88.61 | 92.76 | 90.64 | 2498.91 | 2387.15 | 111.76 | 64.39 |
| `refitUnion--compact_keep--suppression_any` | 1 | binary | 68.14 | 96.72 | 79.95 | 4209.33 | 3031.15 | 1178.18 | 70.92 |
| `refitUnion--compact_keep--suppression_any` | 1 | boundary | 90.17 | 96.81 | 93.37 | 3196.47 | 3031.15 | 165.32 | 70.92 |
| `refitUnion--compact_keep--suppression_any` | 2 | binary | 72.80 | 97.95 | 83.52 | 4815.59 | 3675.15 | 1140.44 | 77.02 |
| `refitUnion--compact_keep--suppression_any` | 2 | boundary | 92.30 | 98.32 | 95.21 | 3838.08 | 3675.15 | 162.93 | 77.02 |
| `refitUnion--compact_keep--suppression_any` | 3 | binary | 76.75 | 98.68 | 86.34 | 5400.97 | 4323.26 | 1077.71 | 81.77 |
| `refitUnion--compact_keep--suppression_any` | 3 | boundary | 93.75 | 99.16 | 96.38 | 4484.83 | 4323.26 | 161.57 | 81.77 |
| `refitUnion--compact_keep--bidirectional_half` | 0 | binary | 60.66 | 93.17 | 73.48 | 3666.78 | 2387.15 | 1279.63 | 29.24 |
| `refitUnion--compact_keep--bidirectional_half` | 0 | boundary | 64.91 | 93.19 | 76.51 | 3428.89 | 2387.15 | 1041.74 | 29.24 |
| `refitUnion--compact_keep--bidirectional_half` | 1 | binary | 66.39 | 97.28 | 78.92 | 4343.69 | 3031.15 | 1312.54 | 33.91 |
| `refitUnion--compact_keep--bidirectional_half` | 1 | boundary | 69.73 | 97.38 | 81.26 | 4155.71 | 3031.15 | 1124.56 | 33.91 |
| `refitUnion--compact_keep--bidirectional_half` | 2 | binary | 70.96 | 98.46 | 82.48 | 4964.98 | 3675.15 | 1289.83 | 38.38 |
| `refitUnion--compact_keep--bidirectional_half` | 2 | boundary | 74.74 | 98.81 | 85.10 | 4760.17 | 3675.15 | 1085.02 | 38.38 |
| `refitUnion--compact_keep--bidirectional_half` | 3 | binary | 74.94 | 99.13 | 85.35 | 5556.39 | 4323.26 | 1233.13 | 42.43 |
| `refitUnion--compact_keep--bidirectional_half` | 3 | boundary | 79.43 | 99.53 | 88.35 | 5311.84 | 4323.26 | 988.57 | 42.43 |
| `refitUnion--compact_keep--bidirectional_any` | 0 | binary | 62.16 | 95.00 | 75.15 | 3648.32 | 2387.15 | 1261.17 | 69.35 |
| `refitUnion--compact_keep--bidirectional_any` | 0 | boundary | 89.71 | 95.01 | 92.28 | 2528.43 | 2387.15 | 141.28 | 69.35 |
| `refitUnion--compact_keep--bidirectional_any` | 1 | binary | 67.98 | 98.36 | 80.40 | 4293.24 | 3031.15 | 1262.09 | 76.30 |
| `refitUnion--compact_keep--bidirectional_any` | 1 | boundary | 91.16 | 98.45 | 94.67 | 3215.66 | 3031.15 | 184.51 | 76.30 |
| `refitUnion--compact_keep--bidirectional_any` | 2 | binary | 72.59 | 99.12 | 83.81 | 4897.90 | 3675.15 | 1222.75 | 82.54 |
| `refitUnion--compact_keep--bidirectional_any` | 2 | boundary | 93.02 | 99.46 | 96.14 | 3857.75 | 3675.15 | 182.60 | 82.54 |
| `refitUnion--compact_keep--bidirectional_any` | 3 | binary | 76.49 | 99.44 | 86.47 | 5484.49 | 4323.26 | 1161.23 | 87.36 |
| `refitUnion--compact_keep--bidirectional_any` | 3 | boundary | 94.40 | 99.81 | 97.03 | 4497.29 | 4323.26 | 174.02 | 87.36 |
| `refitUnion--dino_global--guarded_trim` | 0 | binary | 52.57 | 92.76 | 67.11 | 4211.95 | 2387.15 | 1824.80 | 4.48 |
| `refitUnion--dino_global--guarded_trim` | 0 | boundary | 52.74 | 92.76 | 67.25 | 4198.17 | 2387.15 | 1811.02 | 4.48 |
| `refitUnion--dino_global--guarded_trim` | 1 | binary | 56.59 | 96.75 | 71.41 | 5085.74 | 3031.15 | 2054.59 | 5.46 |
| `refitUnion--dino_global--guarded_trim` | 1 | boundary | 56.68 | 96.81 | 71.50 | 5084.52 | 3031.15 | 2053.37 | 5.46 |
| `refitUnion--dino_global--guarded_trim` | 2 | binary | 61.11 | 98.05 | 75.30 | 5773.74 | 3675.15 | 2098.59 | 6.45 |
| `refitUnion--dino_global--guarded_trim` | 2 | boundary | 61.06 | 98.32 | 75.34 | 5801.26 | 3675.15 | 2126.11 | 6.45 |
| `refitUnion--dino_global--guarded_trim` | 3 | binary | 65.84 | 98.92 | 79.06 | 6350.31 | 4323.26 | 2027.04 | 7.38 |
| `refitUnion--dino_global--guarded_trim` | 3 | boundary | 65.86 | 99.16 | 79.15 | 6384.28 | 4323.26 | 2061.02 | 7.38 |
| `refitUnion--dino_global--suppression_zero` | 0 | binary | 59.91 | 92.72 | 72.78 | 3695.12 | 2387.15 | 1307.97 | 18.70 |
| `refitUnion--dino_global--suppression_zero` | 0 | boundary | 59.48 | 92.76 | 72.48 | 3723.51 | 2387.15 | 1336.35 | 18.70 |
| `refitUnion--dino_global--suppression_zero` | 1 | binary | 65.60 | 96.78 | 78.20 | 4383.73 | 3031.15 | 1352.58 | 22.53 |
| `refitUnion--dino_global--suppression_zero` | 1 | boundary | 64.83 | 96.81 | 77.66 | 4445.88 | 3031.15 | 1414.73 | 22.53 |
| `refitUnion--dino_global--suppression_zero` | 2 | binary | 70.33 | 98.19 | 81.96 | 5010.87 | 3675.15 | 1335.72 | 26.27 |
| `refitUnion--dino_global--suppression_zero` | 2 | boundary | 70.37 | 98.32 | 82.03 | 5034.70 | 3675.15 | 1359.55 | 26.27 |
| `refitUnion--dino_global--suppression_zero` | 3 | binary | 74.46 | 98.98 | 84.98 | 5599.32 | 4323.26 | 1276.05 | 29.88 |
| `refitUnion--dino_global--suppression_zero` | 3 | boundary | 75.71 | 99.16 | 85.86 | 5554.19 | 4323.26 | 1230.93 | 29.88 |
| `refitUnion--dino_global--suppression_half` | 0 | binary | 61.49 | 92.68 | 73.93 | 3598.31 | 2387.15 | 1211.16 | 27.87 |
| `refitUnion--dino_global--suppression_half` | 0 | boundary | 65.74 | 92.76 | 76.93 | 3371.90 | 2387.15 | 984.75 | 27.87 |
| `refitUnion--dino_global--suppression_half` | 1 | binary | 67.22 | 96.75 | 79.33 | 4270.75 | 3031.15 | 1239.60 | 32.41 |
| `refitUnion--dino_global--suppression_half` | 1 | boundary | 70.62 | 96.81 | 81.65 | 4084.17 | 3031.15 | 1053.02 | 32.41 |
| `refitUnion--dino_global--suppression_half` | 2 | binary | 71.92 | 98.06 | 82.98 | 4885.69 | 3675.15 | 1210.54 | 36.81 |
| `refitUnion--dino_global--suppression_half` | 2 | boundary | 75.69 | 98.32 | 85.52 | 4683.03 | 3675.15 | 1007.88 | 36.81 |
| `refitUnion--dino_global--suppression_half` | 3 | binary | 75.90 | 98.81 | 85.85 | 5474.27 | 4323.26 | 1151.00 | 40.83 |
| `refitUnion--dino_global--suppression_half` | 3 | boundary | 80.57 | 99.16 | 88.89 | 5220.60 | 4323.26 | 897.33 | 40.83 |
| `refitUnion--dino_global--suppression_any` | 0 | binary | 62.20 | 92.66 | 74.43 | 3556.60 | 2387.15 | 1169.45 | 62.44 |
| `refitUnion--dino_global--suppression_any` | 0 | boundary | 88.38 | 92.76 | 90.52 | 2505.57 | 2387.15 | 118.42 | 62.44 |
| `refitUnion--dino_global--suppression_any` | 1 | binary | 68.13 | 96.73 | 79.95 | 4210.69 | 3031.15 | 1179.54 | 68.76 |
| `refitUnion--dino_global--suppression_any` | 1 | boundary | 89.93 | 96.81 | 93.24 | 3204.91 | 3031.15 | 173.76 | 68.76 |
| `refitUnion--dino_global--suppression_any` | 2 | binary | 72.79 | 97.99 | 83.53 | 4819.30 | 3675.15 | 1144.15 | 74.57 |
| `refitUnion--dino_global--suppression_any` | 2 | boundary | 91.98 | 98.32 | 95.04 | 3851.32 | 3675.15 | 176.17 | 74.57 |
| `refitUnion--dino_global--suppression_any` | 3 | binary | 76.76 | 98.71 | 86.36 | 5403.18 | 4323.26 | 1079.92 | 79.28 |
| `refitUnion--dino_global--suppression_any` | 3 | boundary | 93.64 | 99.16 | 96.32 | 4490.48 | 4323.26 | 167.22 | 79.28 |
| `refitUnion--dino_global--bidirectional_half` | 0 | binary | 61.36 | 93.11 | 73.98 | 3622.12 | 2387.15 | 1234.97 | 28.55 |
| `refitUnion--dino_global--bidirectional_half` | 0 | boundary | 65.96 | 93.21 | 77.23 | 3376.95 | 2387.15 | 989.80 | 28.55 |
| `refitUnion--dino_global--bidirectional_half` | 1 | binary | 67.13 | 97.20 | 79.41 | 4297.69 | 3031.15 | 1266.54 | 33.08 |
| `refitUnion--dino_global--bidirectional_half` | 1 | boundary | 70.81 | 97.26 | 81.94 | 4092.64 | 3031.15 | 1061.49 | 33.08 |
| `refitUnion--dino_global--bidirectional_half` | 2 | binary | 71.83 | 98.48 | 83.07 | 4913.92 | 3675.15 | 1238.77 | 37.51 |
| `refitUnion--dino_global--bidirectional_half` | 2 | boundary | 75.80 | 98.68 | 85.73 | 4691.39 | 3675.15 | 1016.24 | 37.51 |
| `refitUnion--dino_global--bidirectional_half` | 3 | binary | 75.80 | 99.20 | 85.93 | 5506.08 | 4323.26 | 1182.82 | 41.51 |
| `refitUnion--dino_global--bidirectional_half` | 3 | boundary | 80.64 | 99.44 | 89.05 | 5230.83 | 4323.26 | 907.56 | 41.51 |
| `refitUnion--dino_global--bidirectional_any` | 0 | binary | 62.33 | 96.09 | 75.61 | 3680.52 | 2387.15 | 1293.37 | 69.09 |
| `refitUnion--dino_global--bidirectional_any` | 0 | boundary | 89.68 | 96.21 | 92.83 | 2560.91 | 2387.15 | 173.76 | 69.09 |
| `refitUnion--dino_global--bidirectional_any` | 1 | binary | 68.04 | 99.00 | 80.65 | 4322.36 | 3031.15 | 1291.21 | 75.86 |
| `refitUnion--dino_global--bidirectional_any` | 1 | boundary | 90.89 | 99.05 | 94.79 | 3247.17 | 3031.15 | 216.02 | 75.86 |
| `refitUnion--dino_global--bidirectional_any` | 2 | binary | 72.67 | 99.60 | 84.03 | 4928.23 | 3675.15 | 1253.08 | 81.99 |
| `refitUnion--dino_global--bidirectional_any` | 2 | boundary | 92.77 | 99.71 | 96.11 | 3883.12 | 3675.15 | 207.97 | 81.99 |
| `refitUnion--dino_global--bidirectional_any` | 3 | binary | 76.56 | 99.79 | 86.65 | 5520.88 | 4323.26 | 1197.61 | 86.71 |
| `refitUnion--dino_global--bidirectional_any` | 3 | boundary | 94.29 | 99.89 | 97.01 | 4513.80 | 4323.26 | 190.53 | 86.71 |
| `refitUnion--dino_boost--guarded_trim` | 0 | binary | 52.58 | 92.76 | 67.12 | 4211.11 | 2387.15 | 1823.96 | 4.45 |
| `refitUnion--dino_boost--guarded_trim` | 0 | boundary | 52.73 | 92.76 | 67.24 | 4199.34 | 2387.15 | 1812.19 | 4.45 |
| `refitUnion--dino_boost--guarded_trim` | 1 | binary | 56.59 | 96.74 | 71.41 | 5084.24 | 3031.15 | 2053.09 | 5.42 |
| `refitUnion--dino_boost--guarded_trim` | 1 | boundary | 56.67 | 96.81 | 71.49 | 5085.68 | 3031.15 | 2054.53 | 5.42 |
| `refitUnion--dino_boost--guarded_trim` | 2 | binary | 61.11 | 98.00 | 75.28 | 5771.12 | 3675.15 | 2095.97 | 6.40 |
| `refitUnion--dino_boost--guarded_trim` | 2 | boundary | 61.06 | 98.32 | 75.33 | 5801.94 | 3675.15 | 2126.79 | 6.40 |
| `refitUnion--dino_boost--guarded_trim` | 3 | binary | 65.84 | 98.89 | 79.05 | 6347.69 | 4323.26 | 2024.42 | 7.32 |
| `refitUnion--dino_boost--guarded_trim` | 3 | boundary | 65.85 | 99.16 | 79.14 | 6385.44 | 4323.26 | 2062.17 | 7.32 |
| `refitUnion--dino_boost--suppression_zero` | 0 | binary | 59.50 | 92.74 | 72.49 | 3720.82 | 2387.15 | 1333.67 | 18.12 |
| `refitUnion--dino_boost--suppression_zero` | 0 | boundary | 59.00 | 92.76 | 72.12 | 3753.23 | 2387.15 | 1366.08 | 18.12 |
| `refitUnion--dino_boost--suppression_zero` | 1 | binary | 65.24 | 96.79 | 77.94 | 4409.02 | 3031.15 | 1377.87 | 21.99 |
| `refitUnion--dino_boost--suppression_zero` | 1 | boundary | 64.47 | 96.81 | 77.40 | 4470.29 | 3031.15 | 1439.14 | 21.99 |
| `refitUnion--dino_boost--suppression_zero` | 2 | binary | 70.06 | 98.19 | 81.78 | 5029.81 | 3675.15 | 1354.65 | 25.73 |
| `refitUnion--dino_boost--suppression_zero` | 2 | boundary | 70.06 | 98.32 | 81.82 | 5056.11 | 3675.15 | 1380.96 | 25.73 |
| `refitUnion--dino_boost--suppression_zero` | 3 | binary | 74.20 | 99.01 | 84.83 | 5620.17 | 4323.26 | 1296.91 | 29.36 |
| `refitUnion--dino_boost--suppression_zero` | 3 | boundary | 75.46 | 99.16 | 85.70 | 5572.17 | 4323.26 | 1248.90 | 29.36 |
| `refitUnion--dino_boost--suppression_half` | 0 | binary | 61.47 | 92.70 | 73.92 | 3600.30 | 2387.15 | 1213.15 | 27.29 |
| `refitUnion--dino_boost--suppression_half` | 0 | boundary | 65.28 | 92.76 | 76.62 | 3393.93 | 2387.15 | 1006.78 | 27.29 |
| `refitUnion--dino_boost--suppression_half` | 1 | binary | 67.32 | 96.76 | 79.40 | 4264.99 | 3031.15 | 1233.84 | 31.66 |
| `refitUnion--dino_boost--suppression_half` | 1 | boundary | 70.31 | 96.81 | 81.46 | 4100.45 | 3031.15 | 1069.30 | 31.66 |
| `refitUnion--dino_boost--suppression_half` | 2 | binary | 72.06 | 98.09 | 83.08 | 4875.34 | 3675.15 | 1200.19 | 35.93 |
| `refitUnion--dino_boost--suppression_half` | 2 | boundary | 75.41 | 98.32 | 85.35 | 4699.03 | 3675.15 | 1023.88 | 35.93 |
| `refitUnion--dino_boost--suppression_half` | 3 | binary | 76.01 | 98.78 | 85.91 | 5463.61 | 4323.26 | 1140.35 | 39.93 |
| `refitUnion--dino_boost--suppression_half` | 3 | boundary | 80.28 | 99.16 | 88.72 | 5238.50 | 4323.26 | 915.24 | 39.93 |
| `refitUnion--dino_boost--suppression_any` | 0 | binary | 62.22 | 92.70 | 74.46 | 3556.44 | 2387.15 | 1169.29 | 61.97 |
| `refitUnion--dino_boost--suppression_any` | 0 | boundary | 87.74 | 92.76 | 90.18 | 2524.00 | 2387.15 | 136.85 | 61.97 |
| `refitUnion--dino_boost--suppression_any` | 1 | binary | 68.19 | 96.72 | 79.99 | 4206.97 | 3031.15 | 1175.82 | 68.31 |
| `refitUnion--dino_boost--suppression_any` | 1 | boundary | 89.58 | 96.81 | 93.06 | 3217.37 | 3031.15 | 186.22 | 68.31 |
| `refitUnion--dino_boost--suppression_any` | 2 | binary | 72.88 | 97.95 | 83.57 | 4812.62 | 3675.15 | 1137.47 | 74.09 |
| `refitUnion--dino_boost--suppression_any` | 2 | boundary | 91.85 | 98.32 | 94.97 | 3856.96 | 3675.15 | 181.81 | 74.09 |
| `refitUnion--dino_boost--suppression_any` | 3 | binary | 76.82 | 98.68 | 86.39 | 5397.83 | 4323.26 | 1074.57 | 78.78 |
| `refitUnion--dino_boost--suppression_any` | 3 | boundary | 93.61 | 99.16 | 96.30 | 4491.73 | 4323.26 | 168.47 | 78.78 |
| `refitUnion--dino_boost--bidirectional_half` | 0 | binary | 61.36 | 93.14 | 73.98 | 3623.65 | 2387.15 | 1236.50 | 27.93 |
| `refitUnion--dino_boost--bidirectional_half` | 0 | boundary | 65.46 | 93.21 | 76.90 | 3400.99 | 2387.15 | 1013.84 | 27.93 |
| `refitUnion--dino_boost--bidirectional_half` | 1 | binary | 67.24 | 97.30 | 79.52 | 4293.67 | 3031.15 | 1262.52 | 32.41 |
| `refitUnion--dino_boost--bidirectional_half` | 1 | boundary | 70.50 | 97.35 | 81.77 | 4111.61 | 3031.15 | 1080.46 | 32.41 |
| `refitUnion--dino_boost--bidirectional_half` | 2 | binary | 71.99 | 98.61 | 83.22 | 4909.36 | 3675.15 | 1234.21 | 36.76 |
| `refitUnion--dino_boost--bidirectional_half` | 2 | boundary | 75.54 | 98.76 | 85.60 | 4711.17 | 3675.15 | 1036.02 | 36.76 |
| `refitUnion--dino_boost--bidirectional_half` | 3 | binary | 75.94 | 99.27 | 86.05 | 5503.96 | 4323.26 | 1180.70 | 40.75 |
| `refitUnion--dino_boost--bidirectional_half` | 3 | boundary | 80.38 | 99.50 | 88.92 | 5253.50 | 4323.26 | 930.24 | 40.75 |
| `refitUnion--dino_boost--bidirectional_any` | 0 | binary | 62.40 | 96.22 | 75.70 | 3681.04 | 2387.15 | 1293.89 | 69.05 |
| `refitUnion--dino_boost--bidirectional_any` | 0 | boundary | 89.10 | 96.34 | 92.58 | 2581.31 | 2387.15 | 194.16 | 69.05 |
| `refitUnion--dino_boost--bidirectional_any` | 1 | binary | 68.19 | 99.10 | 80.79 | 4317.19 | 3031.15 | 1286.04 | 75.82 |
| `refitUnion--dino_boost--bidirectional_any` | 1 | boundary | 90.67 | 99.16 | 94.73 | 3258.69 | 3031.15 | 227.54 | 75.82 |
| `refitUnion--dino_boost--bidirectional_any` | 2 | binary | 72.84 | 99.70 | 84.18 | 4923.31 | 3675.15 | 1248.15 | 81.85 |
| `refitUnion--dino_boost--bidirectional_any` | 2 | boundary | 92.80 | 99.77 | 96.16 | 3885.69 | 3675.15 | 210.54 | 81.85 |
| `refitUnion--dino_boost--bidirectional_any` | 3 | binary | 76.73 | 99.89 | 86.79 | 5517.37 | 4323.26 | 1194.10 | 86.53 |
| `refitUnion--dino_boost--bidirectional_any` | 3 | boundary | 94.28 | 99.95 | 97.03 | 4520.66 | 4323.26 | 197.40 | 86.53 |
| `refitUnion--dino_keep--guarded_trim` | 0 | binary | 52.56 | 92.76 | 67.10 | 4213.10 | 2387.15 | 1825.95 | 4.23 |
| `refitUnion--dino_keep--guarded_trim` | 0 | boundary | 52.70 | 92.76 | 67.21 | 4201.97 | 2387.15 | 1814.82 | 4.23 |
| `refitUnion--dino_keep--guarded_trim` | 1 | binary | 56.58 | 96.76 | 71.40 | 5087.56 | 3031.15 | 2056.41 | 5.20 |
| `refitUnion--dino_keep--guarded_trim` | 1 | boundary | 56.64 | 96.81 | 71.47 | 5088.59 | 3031.15 | 2057.44 | 5.20 |
| `refitUnion--dino_keep--guarded_trim` | 2 | binary | 61.11 | 98.09 | 75.30 | 5776.78 | 3675.15 | 2101.63 | 6.20 |
| `refitUnion--dino_keep--guarded_trim` | 2 | boundary | 61.02 | 98.32 | 75.30 | 5805.71 | 3675.15 | 2130.56 | 6.20 |
| `refitUnion--dino_keep--guarded_trim` | 3 | binary | 65.83 | 98.94 | 79.06 | 6354.02 | 4323.26 | 2030.76 | 7.13 |
| `refitUnion--dino_keep--guarded_trim` | 3 | boundary | 65.82 | 99.16 | 79.12 | 6388.35 | 4323.26 | 2065.08 | 7.13 |
| `refitUnion--dino_keep--suppression_zero` | 0 | binary | 59.63 | 92.74 | 72.58 | 3713.36 | 2387.15 | 1326.21 | 18.15 |
| `refitUnion--dino_keep--suppression_zero` | 0 | boundary | 59.03 | 92.76 | 72.15 | 3752.01 | 2387.15 | 1364.86 | 18.15 |
| `refitUnion--dino_keep--suppression_zero` | 1 | binary | 65.41 | 96.77 | 78.05 | 4397.14 | 3031.15 | 1365.99 | 22.05 |
| `refitUnion--dino_keep--suppression_zero` | 1 | boundary | 64.50 | 96.81 | 77.41 | 4469.81 | 3031.15 | 1438.66 | 22.05 |
| `refitUnion--dino_keep--suppression_zero` | 2 | binary | 70.19 | 98.15 | 81.85 | 5019.18 | 3675.15 | 1344.03 | 25.77 |
| `refitUnion--dino_keep--suppression_zero` | 2 | boundary | 70.10 | 98.32 | 81.84 | 5054.80 | 3675.15 | 1379.65 | 25.77 |
| `refitUnion--dino_keep--suppression_zero` | 3 | binary | 74.34 | 98.95 | 84.90 | 5606.66 | 4323.26 | 1283.40 | 29.40 |
| `refitUnion--dino_keep--suppression_zero` | 3 | boundary | 75.51 | 99.16 | 85.73 | 5569.65 | 4323.26 | 1246.38 | 29.40 |
| `refitUnion--dino_keep--suppression_half` | 0 | binary | 61.65 | 92.70 | 74.05 | 3589.63 | 2387.15 | 1202.48 | 26.72 |
| `refitUnion--dino_keep--suppression_half` | 0 | boundary | 65.03 | 92.76 | 76.42 | 3412.55 | 2387.15 | 1025.40 | 26.72 |
| `refitUnion--dino_keep--suppression_half` | 1 | binary | 67.50 | 96.76 | 79.52 | 4255.32 | 3031.15 | 1224.17 | 31.18 |
| `refitUnion--dino_keep--suppression_half` | 1 | boundary | 70.09 | 96.81 | 81.29 | 4117.81 | 3031.15 | 1086.66 | 31.18 |
| `refitUnion--dino_keep--suppression_half` | 2 | binary | 72.20 | 98.07 | 83.17 | 4868.31 | 3675.15 | 1193.15 | 35.55 |
| `refitUnion--dino_keep--suppression_half` | 2 | boundary | 75.23 | 98.32 | 85.22 | 4713.55 | 3675.15 | 1038.40 | 35.55 |
| `refitUnion--dino_keep--suppression_half` | 3 | binary | 76.17 | 98.82 | 86.02 | 5456.89 | 4323.26 | 1133.62 | 39.57 |
| `refitUnion--dino_keep--suppression_half` | 3 | boundary | 80.17 | 99.16 | 88.64 | 5248.10 | 4323.26 | 924.84 | 39.57 |
| `refitUnion--dino_keep--suppression_any` | 0 | binary | 62.07 | 92.70 | 74.35 | 3565.14 | 2387.15 | 1177.99 | 63.76 |
| `refitUnion--dino_keep--suppression_any` | 0 | boundary | 87.48 | 92.76 | 90.03 | 2532.69 | 2387.15 | 145.54 | 63.76 |
| `refitUnion--dino_keep--suppression_any` | 1 | binary | 68.04 | 96.74 | 79.89 | 4218.96 | 3031.15 | 1187.81 | 70.21 |
| `refitUnion--dino_keep--suppression_any` | 1 | boundary | 89.32 | 96.81 | 92.91 | 3227.89 | 3031.15 | 196.73 | 70.21 |
| `refitUnion--dino_keep--suppression_any` | 2 | binary | 72.72 | 98.01 | 83.49 | 4827.49 | 3675.15 | 1152.34 | 76.15 |
| `refitUnion--dino_keep--suppression_any` | 2 | boundary | 91.75 | 98.32 | 94.91 | 3862.04 | 3675.15 | 186.89 | 76.15 |
| `refitUnion--dino_keep--suppression_any` | 3 | binary | 76.70 | 98.75 | 86.34 | 5411.50 | 4323.26 | 1088.24 | 81.05 |
| `refitUnion--dino_keep--suppression_any` | 3 | boundary | 93.40 | 99.16 | 96.19 | 4502.67 | 4323.26 | 179.40 | 81.05 |
| `refitUnion--dino_keep--bidirectional_half` | 0 | binary | 61.54 | 93.27 | 74.15 | 3618.16 | 2387.15 | 1231.01 | 27.46 |
| `refitUnion--dino_keep--bidirectional_half` | 0 | boundary | 65.24 | 93.30 | 76.75 | 3420.87 | 2387.15 | 1033.71 | 27.46 |
| `refitUnion--dino_keep--bidirectional_half` | 1 | binary | 67.40 | 97.35 | 79.65 | 4286.52 | 3031.15 | 1255.37 | 32.03 |
| `refitUnion--dino_keep--bidirectional_half` | 1 | boundary | 70.32 | 97.41 | 81.65 | 4128.67 | 3031.15 | 1097.52 | 32.03 |
| `refitUnion--dino_keep--bidirectional_half` | 2 | binary | 72.10 | 98.62 | 83.30 | 4902.17 | 3675.15 | 1227.02 | 36.53 |
| `refitUnion--dino_keep--bidirectional_half` | 2 | boundary | 75.40 | 98.80 | 85.51 | 4723.01 | 3675.15 | 1047.86 | 36.53 |
| `refitUnion--dino_keep--bidirectional_half` | 3 | binary | 76.05 | 99.29 | 86.13 | 5494.24 | 4323.26 | 1170.98 | 40.60 |
| `refitUnion--dino_keep--bidirectional_half` | 3 | boundary | 80.34 | 99.52 | 88.89 | 5257.02 | 4323.26 | 933.76 | 40.60 |
| `refitUnion--dino_keep--bidirectional_any` | 0 | binary | 61.99 | 96.00 | 75.33 | 3696.99 | 2387.15 | 1309.84 | 70.46 |
| `refitUnion--dino_keep--bidirectional_any` | 0 | boundary | 88.96 | 96.02 | 92.34 | 2577.49 | 2387.15 | 190.34 | 70.46 |
| `refitUnion--dino_keep--bidirectional_any` | 1 | binary | 67.78 | 98.82 | 80.41 | 4333.52 | 3031.15 | 1302.37 | 77.16 |
| `refitUnion--dino_keep--bidirectional_any` | 1 | boundary | 90.49 | 98.88 | 94.50 | 3257.41 | 3031.15 | 226.26 | 77.16 |
| `refitUnion--dino_keep--bidirectional_any` | 2 | binary | 72.45 | 99.48 | 83.84 | 4938.44 | 3675.15 | 1263.29 | 83.24 |
| `refitUnion--dino_keep--bidirectional_any` | 2 | boundary | 92.65 | 99.65 | 96.02 | 3885.79 | 3675.15 | 210.64 | 83.24 |
| `refitUnion--dino_keep--bidirectional_any` | 3 | binary | 76.44 | 99.74 | 86.55 | 5524.25 | 4323.26 | 1200.99 | 88.13 |
| `refitUnion--dino_keep--bidirectional_any` | 3 | boundary | 94.04 | 99.90 | 96.88 | 4526.11 | 4323.26 | 202.85 | 88.13 |
| `compact_boost--positive_uncertain` | 0 | binary | 84.60 | 81.99 | 83.27 | 2314.24 | 2387.15 | -72.91 | 3.09 |
| `compact_boost--positive_uncertain` | 0 | boundary | 84.84 | 82.00 | 83.38 | 2308.04 | 2387.15 | -79.11 | 3.09 |
| `compact_boost--positive_uncertain` | 1 | binary | 86.92 | 89.59 | 88.23 | 2927.81 | 3031.15 | -103.34 | 3.97 |
| `compact_boost--positive_uncertain` | 1 | boundary | 87.22 | 89.60 | 88.39 | 2918.47 | 3031.15 | -112.68 | 3.97 |
| `compact_boost--positive_uncertain` | 2 | binary | 88.81 | 92.39 | 90.56 | 3522.03 | 3675.15 | -153.13 | 4.84 |
| `compact_boost--positive_uncertain` | 2 | boundary | 89.10 | 92.41 | 90.72 | 3512.22 | 3675.15 | -162.93 | 4.84 |
| `compact_boost--positive_uncertain` | 3 | binary | 90.23 | 93.86 | 92.01 | 4119.38 | 4323.26 | -203.89 | 5.71 |
| `compact_boost--positive_uncertain` | 3 | boundary | 90.50 | 93.88 | 92.16 | 4109.19 | 4323.26 | -214.07 | 5.71 |
| `compact_boost--uncertain_narrow` | 0 | binary | 71.38 | 96.23 | 81.96 | 3218.53 | 2387.15 | 831.38 | 61.09 |
| `compact_boost--uncertain_narrow` | 0 | boundary | 87.86 | 96.22 | 91.84 | 2615.00 | 2387.15 | 227.85 | 61.09 |
| `compact_boost--uncertain_narrow` | 1 | binary | 75.62 | 98.14 | 85.42 | 3863.98 | 3031.15 | 832.83 | 72.13 |
| `compact_boost--uncertain_narrow` | 1 | boundary | 95.08 | 98.15 | 96.59 | 3074.25 | 3031.15 | 43.10 | 72.13 |
| `compact_boost--uncertain_narrow` | 2 | binary | 78.51 | 98.73 | 87.47 | 4517.65 | 3675.15 | 842.50 | 84.22 |
| `compact_boost--uncertain_narrow` | 2 | boundary | 97.53 | 98.81 | 98.16 | 3645.18 | 3675.15 | -29.97 | 84.22 |
| `compact_boost--uncertain_narrow` | 3 | binary | 80.95 | 98.98 | 89.06 | 5160.20 | 4323.26 | 836.93 | 91.18 |
| `compact_boost--uncertain_narrow` | 3 | boundary | 98.40 | 99.01 | 98.71 | 4253.60 | 4323.26 | -69.66 | 91.18 |
| `compact_boost--uncertain_medium` | 0 | binary | 70.19 | 97.35 | 81.57 | 3311.06 | 2387.15 | 923.90 | 69.39 |
| `compact_boost--uncertain_medium` | 0 | boundary | 88.39 | 97.34 | 92.64 | 2629.76 | 2387.15 | 242.61 | 69.39 |
| `compact_boost--uncertain_medium` | 1 | binary | 74.65 | 98.73 | 85.02 | 3957.92 | 3031.15 | 926.77 | 80.94 |
| `compact_boost--uncertain_medium` | 1 | boundary | 95.86 | 98.75 | 97.28 | 3083.93 | 3031.15 | 52.78 | 80.94 |
| `compact_boost--uncertain_medium` | 2 | binary | 77.62 | 99.12 | 87.06 | 4619.45 | 3675.15 | 944.30 | 93.89 |
| `compact_boost--uncertain_medium` | 2 | boundary | 98.25 | 99.21 | 98.73 | 3658.43 | 3675.15 | -16.73 | 93.89 |
| `compact_boost--uncertain_medium` | 3 | binary | 80.23 | 99.27 | 88.74 | 5261.83 | 4323.26 | 938.56 | 100.79 |
| `compact_boost--uncertain_medium` | 3 | boundary | 99.08 | 99.34 | 99.21 | 4270.72 | 4323.26 | -52.55 | 100.79 |
| `compact_boost--uncertain_wide` | 0 | binary | 68.99 | 98.34 | 81.09 | 3402.83 | 2387.15 | 1015.67 | 80.41 |
| `compact_boost--uncertain_wide` | 0 | boundary | 90.00 | 98.35 | 93.98 | 2609.15 | 2387.15 | 222.00 | 80.41 |
| `compact_boost--uncertain_wide` | 1 | binary | 73.72 | 99.19 | 84.58 | 4046.52 | 3031.15 | 1015.37 | 91.38 |
| `compact_boost--uncertain_wide` | 1 | boundary | 96.64 | 99.19 | 97.89 | 3087.10 | 3031.15 | 55.95 | 91.38 |
| `compact_boost--uncertain_wide` | 2 | binary | 76.78 | 99.42 | 86.65 | 4713.68 | 3675.15 | 1038.52 | 103.86 |
| `compact_boost--uncertain_wide` | 2 | boundary | 98.65 | 99.43 | 99.04 | 3673.96 | 3675.15 | -1.19 | 103.86 |
| `compact_boost--uncertain_wide` | 3 | binary | 79.74 | 99.51 | 88.54 | 5342.26 | 4323.26 | 1018.99 | 109.99 |
| `compact_boost--uncertain_wide` | 3 | boundary | 99.37 | 99.52 | 99.44 | 4293.06 | 4323.26 | -30.20 | 109.99 |
| `compact_boost--all_positive` | 0 | binary | 86.96 | 81.96 | 84.37 | 2250.63 | 2387.15 | -136.52 | 59.06 |
| `compact_boost--all_positive` | 0 | boundary | 100.00 | 82.00 | 90.11 | 1957.36 | 2387.15 | -429.79 | 59.06 |
| `compact_boost--all_positive` | 1 | binary | 89.82 | 89.56 | 89.69 | 2829.58 | 3031.15 | -201.57 | 69.09 |
| `compact_boost--all_positive` | 1 | boundary | 100.00 | 89.60 | 94.51 | 2544.77 | 3031.15 | -486.38 | 69.09 |
| `compact_boost--all_positive` | 2 | binary | 91.72 | 92.36 | 92.03 | 3402.37 | 3675.15 | -272.78 | 79.40 |
| `compact_boost--all_positive` | 2 | boundary | 100.00 | 92.41 | 96.05 | 3128.65 | 3675.15 | -546.51 | 79.40 |
| `compact_boost--all_positive` | 3 | binary | 93.03 | 93.78 | 93.40 | 3979.13 | 4323.26 | -344.14 | 89.71 |
| `compact_boost--all_positive` | 3 | boundary | 100.00 | 93.88 | 96.84 | 3718.40 | 4323.26 | -604.86 | 89.71 |
| `compact_boost--all_candidates` | 0 | binary | 65.88 | 100.00 | 79.43 | 3623.43 | 2387.15 | 1236.28 | 137.83 |
| `compact_boost--all_candidates` | 0 | boundary | 100.00 | 100.00 | 100.00 | 2387.15 | 2387.15 | 0.00 | 137.83 |
| `compact_boost--all_candidates` | 1 | binary | 71.13 | 100.00 | 83.13 | 4261.28 | 3031.15 | 1230.13 | 137.83 |
| `compact_boost--all_candidates` | 1 | boundary | 100.00 | 100.00 | 100.00 | 3031.15 | 3031.15 | 0.00 | 137.83 |
| `compact_boost--all_candidates` | 2 | binary | 74.41 | 100.00 | 85.33 | 4939.13 | 3675.15 | 1263.98 | 137.83 |
| `compact_boost--all_candidates` | 2 | boundary | 100.00 | 100.00 | 100.00 | 3675.15 | 3675.15 | 0.00 | 137.83 |
| `compact_boost--all_candidates` | 3 | binary | 78.22 | 100.00 | 87.78 | 5527.35 | 4323.26 | 1204.09 | 137.83 |
| `compact_boost--all_candidates` | 3 | boundary | 100.00 | 100.00 | 100.00 | 4323.26 | 4323.26 | 0.00 | 137.83 |
| `compact_keep--positive_uncertain` | 0 | binary | 83.55 | 82.80 | 83.14 | 2367.44 | 2387.15 | -19.71 | 5.20 |
| `compact_keep--positive_uncertain` | 0 | boundary | 84.10 | 82.81 | 83.42 | 2352.81 | 2387.15 | -34.35 | 5.20 |
| `compact_keep--positive_uncertain` | 1 | binary | 86.13 | 90.78 | 88.38 | 2998.13 | 3031.15 | -33.02 | 6.71 |
| `compact_keep--positive_uncertain` | 1 | boundary | 86.73 | 90.80 | 88.70 | 2979.58 | 3031.15 | -51.58 | 6.71 |
| `compact_keep--positive_uncertain` | 2 | binary | 88.09 | 93.37 | 90.64 | 3605.16 | 3675.15 | -70.00 | 8.18 |
| `compact_keep--positive_uncertain` | 2 | boundary | 88.72 | 93.42 | 90.99 | 3583.67 | 3675.15 | -91.48 | 8.18 |
| `compact_keep--positive_uncertain` | 3 | binary | 89.70 | 94.64 | 92.09 | 4204.04 | 4323.26 | -119.22 | 9.68 |
| `compact_keep--positive_uncertain` | 3 | boundary | 90.25 | 94.73 | 92.42 | 4188.36 | 4323.26 | -134.90 | 9.68 |
| `compact_keep--uncertain_narrow` | 0 | binary | 71.31 | 95.73 | 81.73 | 3205.94 | 2387.15 | 818.79 | 60.33 |
| `compact_keep--uncertain_narrow` | 0 | boundary | 86.93 | 95.71 | 91.10 | 2629.78 | 2387.15 | 242.63 | 60.33 |
| `compact_keep--uncertain_narrow` | 1 | binary | 75.54 | 97.83 | 85.25 | 3852.18 | 3031.15 | 821.03 | 71.01 |
| `compact_keep--uncertain_narrow` | 1 | boundary | 94.41 | 97.84 | 96.09 | 3083.06 | 3031.15 | 51.91 | 71.01 |
| `compact_keep--uncertain_narrow` | 2 | binary | 78.54 | 98.47 | 87.38 | 4500.47 | 3675.15 | 825.32 | 82.49 |
| `compact_keep--uncertain_narrow` | 2 | boundary | 97.10 | 98.55 | 97.82 | 3649.02 | 3675.15 | -26.13 | 82.49 |
| `compact_keep--uncertain_narrow` | 3 | binary | 81.13 | 98.77 | 89.08 | 5130.82 | 4323.26 | 807.56 | 89.31 |
| `compact_keep--uncertain_narrow` | 3 | boundary | 98.16 | 98.81 | 98.48 | 4250.93 | 4323.26 | -72.34 | 89.31 |
| `compact_keep--uncertain_medium` | 0 | binary | 70.53 | 96.64 | 81.54 | 3272.40 | 2387.15 | 885.25 | 67.43 |
| `compact_keep--uncertain_medium` | 0 | boundary | 87.61 | 96.63 | 91.88 | 2634.78 | 2387.15 | 247.63 | 67.43 |
| `compact_keep--uncertain_medium` | 1 | binary | 74.91 | 98.24 | 85.00 | 3917.55 | 3031.15 | 886.40 | 78.53 |
| `compact_keep--uncertain_medium` | 1 | boundary | 95.24 | 98.26 | 96.72 | 3082.63 | 3031.15 | 51.48 | 78.53 |
| `compact_keep--uncertain_medium` | 2 | binary | 77.96 | 98.72 | 87.11 | 4569.04 | 3675.15 | 893.89 | 90.62 |
| `compact_keep--uncertain_medium` | 2 | boundary | 97.83 | 98.82 | 98.32 | 3650.22 | 3675.15 | -24.93 | 90.62 |
| `compact_keep--uncertain_medium` | 3 | binary | 80.59 | 98.96 | 88.83 | 5202.38 | 4323.26 | 879.11 | 97.56 |
| `compact_keep--uncertain_medium` | 3 | boundary | 98.77 | 99.03 | 98.90 | 4256.94 | 4323.26 | -66.32 | 97.56 |
| `compact_keep--uncertain_wide` | 0 | binary | 69.20 | 98.25 | 81.20 | 3390.67 | 2387.15 | 1003.52 | 79.04 |
| `compact_keep--uncertain_wide` | 0 | boundary | 89.10 | 98.26 | 93.44 | 2634.53 | 2387.15 | 247.38 | 79.04 |
| `compact_keep--uncertain_wide` | 1 | binary | 73.86 | 99.17 | 84.66 | 4035.98 | 3031.15 | 1004.82 | 90.21 |
| `compact_keep--uncertain_wide` | 1 | boundary | 96.06 | 99.17 | 97.59 | 3103.20 | 3031.15 | 72.05 | 90.21 |
| `compact_keep--uncertain_wide` | 2 | binary | 77.01 | 99.40 | 86.78 | 4695.88 | 3675.15 | 1020.72 | 102.65 |
| `compact_keep--uncertain_wide` | 2 | boundary | 98.44 | 99.41 | 98.92 | 3677.99 | 3675.15 | 2.84 | 102.65 |
| `compact_keep--uncertain_wide` | 3 | binary | 79.93 | 99.47 | 88.63 | 5324.44 | 4323.26 | 1001.18 | 108.81 |
| `compact_keep--uncertain_wide` | 3 | boundary | 99.28 | 99.47 | 99.37 | 4293.64 | 4323.26 | -29.62 | 108.81 |
| `compact_keep--all_positive` | 0 | binary | 86.16 | 82.73 | 84.38 | 2293.88 | 2387.15 | -93.27 | 61.05 |
| `compact_keep--all_positive` | 0 | boundary | 100.00 | 82.81 | 90.59 | 1976.70 | 2387.15 | -410.45 | 61.05 |
| `compact_keep--all_positive` | 1 | binary | 89.23 | 90.74 | 89.96 | 2889.66 | 3031.15 | -141.49 | 71.31 |
| `compact_keep--all_positive` | 1 | boundary | 100.00 | 90.80 | 95.18 | 2582.26 | 3031.15 | -448.89 | 71.31 |
| `compact_keep--all_positive` | 2 | binary | 91.22 | 93.34 | 92.25 | 3472.20 | 3675.15 | -202.95 | 81.83 |
| `compact_keep--all_positive` | 2 | boundary | 100.00 | 93.42 | 96.60 | 3177.51 | 3675.15 | -497.64 | 81.83 |
| `compact_keep--all_positive` | 3 | binary | 92.66 | 94.55 | 93.59 | 4053.66 | 4323.26 | -269.60 | 92.49 |
| `compact_keep--all_positive` | 3 | boundary | 100.00 | 94.73 | 97.29 | 3778.15 | 4323.26 | -545.11 | 92.49 |
| `compact_keep--all_candidates` | 0 | binary | 65.96 | 100.00 | 79.48 | 3619.58 | 2387.15 | 1232.43 | 137.83 |
| `compact_keep--all_candidates` | 0 | boundary | 100.00 | 100.00 | 100.00 | 2387.15 | 2387.15 | 0.00 | 137.83 |
| `compact_keep--all_candidates` | 1 | binary | 71.24 | 100.00 | 83.20 | 4255.10 | 3031.15 | 1223.95 | 137.83 |
| `compact_keep--all_candidates` | 1 | boundary | 100.00 | 100.00 | 100.00 | 3031.15 | 3031.15 | 0.00 | 137.83 |
| `compact_keep--all_candidates` | 2 | binary | 74.56 | 100.00 | 85.43 | 4929.28 | 3675.15 | 1254.13 | 137.83 |
| `compact_keep--all_candidates` | 2 | boundary | 100.00 | 100.00 | 100.00 | 3675.15 | 3675.15 | 0.00 | 137.83 |
| `compact_keep--all_candidates` | 3 | binary | 78.30 | 100.00 | 87.83 | 5521.91 | 4323.26 | 1198.64 | 137.83 |
| `compact_keep--all_candidates` | 3 | boundary | 100.00 | 100.00 | 100.00 | 4323.26 | 4323.26 | 0.00 | 137.83 |
| `dino_global--positive_uncertain` | 0 | binary | 86.73 | 87.32 | 87.01 | 2405.04 | 2387.15 | 17.89 | 2.46 |
| `dino_global--positive_uncertain` | 0 | boundary | 86.73 | 87.32 | 87.01 | 2405.08 | 2387.15 | 17.93 | 2.46 |
| `dino_global--positive_uncertain` | 1 | binary | 89.08 | 94.96 | 91.91 | 3036.32 | 3031.15 | 5.17 | 3.30 |
| `dino_global--positive_uncertain` | 1 | boundary | 89.27 | 94.96 | 92.02 | 3030.75 | 3031.15 | -0.40 | 3.30 |
| `dino_global--positive_uncertain` | 2 | binary | 90.66 | 96.94 | 93.69 | 3657.32 | 3675.15 | -17.83 | 4.12 |
| `dino_global--positive_uncertain` | 2 | boundary | 90.96 | 96.96 | 93.85 | 3647.12 | 3675.15 | -28.03 | 4.12 |
| `dino_global--positive_uncertain` | 3 | binary | 91.89 | 97.61 | 94.66 | 4278.24 | 4323.26 | -45.02 | 4.96 |
| `dino_global--positive_uncertain` | 3 | boundary | 92.16 | 97.64 | 94.81 | 4270.18 | 4323.26 | -53.08 | 4.96 |
| `dino_global--uncertain_narrow` | 0 | binary | 72.13 | 96.92 | 82.70 | 3209.09 | 2387.15 | 821.94 | 57.90 |
| `dino_global--uncertain_narrow` | 0 | boundary | 88.53 | 96.92 | 92.52 | 2614.54 | 2387.15 | 227.39 | 57.90 |
| `dino_global--uncertain_narrow` | 1 | binary | 76.33 | 98.89 | 86.15 | 3864.75 | 3031.15 | 833.60 | 69.01 |
| `dino_global--uncertain_narrow` | 1 | boundary | 95.34 | 98.89 | 97.08 | 3093.56 | 3031.15 | 62.41 | 69.01 |
| `dino_global--uncertain_narrow` | 2 | binary | 79.07 | 99.41 | 88.08 | 4535.41 | 3675.15 | 860.26 | 81.09 |
| `dino_global--uncertain_narrow` | 2 | boundary | 97.23 | 99.41 | 98.31 | 3688.45 | 3675.15 | 13.30 | 81.09 |
| `dino_global--uncertain_narrow` | 3 | binary | 81.80 | 99.59 | 89.81 | 5167.58 | 4323.26 | 844.31 | 88.48 |
| `dino_global--uncertain_narrow` | 3 | boundary | 97.98 | 99.59 | 98.78 | 4314.58 | 4323.26 | -8.69 | 88.48 |
| `dino_global--uncertain_medium` | 0 | binary | 71.46 | 97.60 | 82.50 | 3261.45 | 2387.15 | 874.30 | 62.79 |
| `dino_global--uncertain_medium` | 0 | boundary | 88.98 | 97.59 | 93.08 | 2619.34 | 2387.15 | 232.19 | 62.79 |
| `dino_global--uncertain_medium` | 1 | binary | 75.94 | 99.13 | 85.99 | 3906.20 | 3031.15 | 875.05 | 74.05 |
| `dino_global--uncertain_medium` | 1 | boundary | 96.00 | 99.13 | 97.53 | 3089.91 | 3031.15 | 58.76 | 74.05 |
| `dino_global--uncertain_medium` | 2 | binary | 78.75 | 99.51 | 87.92 | 4576.87 | 3675.15 | 901.72 | 86.54 |
| `dino_global--uncertain_medium` | 2 | boundary | 97.88 | 99.51 | 98.69 | 3682.54 | 3675.15 | 7.38 | 86.54 |
| `dino_global--uncertain_medium` | 3 | binary | 81.58 | 99.65 | 89.71 | 5203.55 | 4323.26 | 880.28 | 93.67 |
| `dino_global--uncertain_medium` | 3 | boundary | 98.50 | 99.65 | 99.07 | 4310.59 | 4323.26 | -12.67 | 93.67 |
| `dino_global--uncertain_wide` | 0 | binary | 70.51 | 98.32 | 82.12 | 3329.41 | 2387.15 | 942.26 | 72.37 |
| `dino_global--uncertain_wide` | 0 | boundary | 90.15 | 98.33 | 94.06 | 2604.60 | 2387.15 | 217.45 | 72.37 |
| `dino_global--uncertain_wide` | 1 | binary | 75.19 | 99.45 | 85.63 | 3973.37 | 3031.15 | 942.22 | 83.99 |
| `dino_global--uncertain_wide` | 1 | boundary | 97.19 | 99.46 | 98.31 | 3074.68 | 3031.15 | 43.53 | 83.99 |
| `dino_global--uncertain_wide` | 2 | binary | 78.16 | 99.73 | 87.64 | 4641.90 | 3675.15 | 966.75 | 96.94 |
| `dino_global--uncertain_wide` | 2 | boundary | 98.82 | 99.73 | 99.27 | 3673.34 | 3675.15 | -1.81 | 96.94 |
| `dino_global--uncertain_wide` | 3 | binary | 81.10 | 99.82 | 89.49 | 5268.70 | 4323.26 | 945.44 | 103.85 |
| `dino_global--uncertain_wide` | 3 | boundary | 99.30 | 99.82 | 99.56 | 4304.91 | 4323.26 | -18.35 | 103.85 |
| `dino_global--all_positive` | 0 | binary | 87.79 | 87.31 | 87.53 | 2375.48 | 2387.15 | -11.67 | 61.78 |
| `dino_global--all_positive` | 0 | boundary | 100.00 | 87.32 | 93.23 | 2084.55 | 2387.15 | -302.60 | 61.78 |
| `dino_global--all_positive` | 1 | binary | 90.37 | 94.95 | 92.60 | 2991.25 | 3031.15 | -39.90 | 72.30 |
| `dino_global--all_positive` | 1 | boundary | 100.00 | 94.96 | 97.42 | 2704.37 | 3031.15 | -326.78 | 72.30 |
| `dino_global--all_positive` | 2 | binary | 92.01 | 96.92 | 94.39 | 3601.20 | 3675.15 | -73.95 | 83.08 |
| `dino_global--all_positive` | 2 | boundary | 100.00 | 96.96 | 98.46 | 3316.23 | 3675.15 | -358.92 | 83.08 |
| `dino_global--all_positive` | 3 | binary | 93.24 | 97.59 | 95.36 | 4212.00 | 4323.26 | -111.26 | 94.15 |
| `dino_global--all_positive` | 3 | boundary | 100.00 | 97.64 | 98.80 | 3934.17 | 4323.26 | -389.09 | 94.15 |
| `dino_global--all_candidates` | 0 | binary | 66.60 | 100.00 | 79.95 | 3584.40 | 2387.15 | 1197.25 | 137.83 |
| `dino_global--all_candidates` | 0 | boundary | 100.00 | 100.00 | 100.00 | 2387.15 | 2387.15 | 0.00 | 137.83 |
| `dino_global--all_candidates` | 1 | binary | 71.73 | 100.00 | 83.54 | 4226.02 | 3031.15 | 1194.87 | 137.83 |
| `dino_global--all_candidates` | 1 | boundary | 100.00 | 100.00 | 100.00 | 3031.15 | 3031.15 | 0.00 | 137.83 |
| `dino_global--all_candidates` | 2 | binary | 74.84 | 100.00 | 85.61 | 4910.51 | 3675.15 | 1235.36 | 137.83 |
| `dino_global--all_candidates` | 2 | boundary | 100.00 | 100.00 | 100.00 | 3675.15 | 3675.15 | 0.00 | 137.83 |
| `dino_global--all_candidates` | 3 | binary | 78.46 | 100.00 | 87.93 | 5510.49 | 4323.26 | 1187.22 | 137.83 |
| `dino_global--all_candidates` | 3 | boundary | 100.00 | 100.00 | 100.00 | 4323.26 | 4323.26 | 0.00 | 137.83 |
| `dino_boost--positive_uncertain` | 0 | binary | 84.94 | 86.20 | 85.55 | 2423.92 | 2387.15 | 36.77 | 1.46 |
| `dino_boost--positive_uncertain` | 0 | boundary | 84.76 | 86.22 | 85.47 | 2429.57 | 2387.15 | 42.42 | 1.46 |
| `dino_boost--positive_uncertain` | 1 | binary | 87.57 | 94.63 | 90.95 | 3068.78 | 3031.15 | 37.63 | 2.01 |
| `dino_boost--positive_uncertain` | 1 | boundary | 87.58 | 94.63 | 90.96 | 3069.60 | 3031.15 | 38.45 | 2.01 |
| `dino_boost--positive_uncertain` | 2 | binary | 89.39 | 96.93 | 93.00 | 3698.62 | 3675.15 | 23.47 | 2.54 |
| `dino_boost--positive_uncertain` | 2 | boundary | 89.61 | 96.95 | 93.13 | 3691.99 | 3675.15 | 16.84 | 2.54 |
| `dino_boost--positive_uncertain` | 3 | binary | 90.96 | 97.80 | 94.25 | 4318.49 | 4323.26 | -4.77 | 3.05 |
| `dino_boost--positive_uncertain` | 3 | boundary | 91.27 | 97.84 | 94.44 | 4306.73 | 4323.26 | -16.53 | 3.05 |
| `dino_boost--uncertain_narrow` | 0 | binary | 69.22 | 96.62 | 80.59 | 3339.41 | 2387.15 | 952.26 | 64.94 |
| `dino_boost--uncertain_narrow` | 0 | boundary | 87.48 | 96.59 | 91.79 | 2636.89 | 2387.15 | 249.73 | 64.94 |
| `dino_boost--uncertain_narrow` | 1 | binary | 73.81 | 98.49 | 84.35 | 3992.21 | 3031.15 | 961.06 | 76.49 |
| `dino_boost--uncertain_narrow` | 1 | boundary | 94.73 | 98.49 | 96.57 | 3106.33 | 3031.15 | 75.18 | 76.49 |
| `dino_boost--uncertain_narrow` | 2 | binary | 76.86 | 99.10 | 86.54 | 4663.40 | 3675.15 | 988.25 | 89.18 |
| `dino_boost--uncertain_narrow` | 2 | boundary | 97.08 | 99.10 | 98.08 | 3687.37 | 3675.15 | 12.22 | 89.18 |
| `dino_boost--uncertain_narrow` | 3 | binary | 79.90 | 99.35 | 88.55 | 5287.23 | 4323.26 | 963.97 | 97.00 |
| `dino_boost--uncertain_narrow` | 3 | boundary | 97.98 | 99.35 | 98.66 | 4306.46 | 4323.26 | -16.81 | 97.00 |
| `dino_boost--uncertain_medium` | 0 | binary | 68.51 | 97.37 | 80.38 | 3398.76 | 2387.15 | 1011.61 | 71.06 |
| `dino_boost--uncertain_medium` | 0 | boundary | 87.75 | 97.33 | 92.28 | 2648.68 | 2387.15 | 261.53 | 71.06 |
| `dino_boost--uncertain_medium` | 1 | binary | 73.28 | 98.76 | 84.11 | 4045.52 | 3031.15 | 1014.37 | 82.67 |
| `dino_boost--uncertain_medium` | 1 | boundary | 95.38 | 98.77 | 97.04 | 3105.40 | 3031.15 | 74.25 | 82.67 |
| `dino_boost--uncertain_medium` | 2 | binary | 76.30 | 99.20 | 86.23 | 4720.87 | 3675.15 | 1045.72 | 95.69 |
| `dino_boost--uncertain_medium` | 2 | boundary | 97.75 | 99.22 | 98.48 | 3682.93 | 3675.15 | 7.78 | 95.69 |
| `dino_boost--uncertain_medium` | 3 | binary | 79.53 | 99.39 | 88.34 | 5334.97 | 4323.26 | 1011.70 | 102.98 |
| `dino_boost--uncertain_medium` | 3 | boundary | 98.56 | 99.43 | 98.99 | 4303.78 | 4323.26 | -19.48 | 102.98 |
| `dino_boost--uncertain_wide` | 0 | binary | 67.92 | 98.47 | 80.35 | 3466.05 | 2387.15 | 1078.90 | 78.39 |
| `dino_boost--uncertain_wide` | 0 | boundary | 88.42 | 98.47 | 93.17 | 2659.12 | 2387.15 | 271.97 | 78.39 |
| `dino_boost--uncertain_wide` | 1 | binary | 72.87 | 99.33 | 84.05 | 4107.30 | 3031.15 | 1076.14 | 90.00 |
| `dino_boost--uncertain_wide` | 1 | boundary | 96.28 | 99.33 | 97.78 | 3105.59 | 3031.15 | 74.44 | 90.00 |
| `dino_boost--uncertain_wide` | 2 | binary | 75.88 | 99.59 | 86.11 | 4787.40 | 3675.15 | 1112.25 | 103.27 |
| `dino_boost--uncertain_wide` | 2 | boundary | 98.65 | 99.59 | 99.12 | 3679.87 | 3675.15 | 4.72 | 103.27 |
| `dino_boost--uncertain_wide` | 3 | binary | 79.23 | 99.70 | 88.28 | 5396.60 | 4323.26 | 1073.33 | 109.92 |
| `dino_boost--uncertain_wide` | 3 | boundary | 99.36 | 99.70 | 99.53 | 4301.69 | 4323.26 | -21.58 | 109.92 |
| `dino_boost--all_positive` | 0 | binary | 86.08 | 86.20 | 86.12 | 2391.92 | 2387.15 | 4.77 | 62.33 |
| `dino_boost--all_positive` | 0 | boundary | 100.00 | 86.22 | 92.60 | 2058.26 | 2387.15 | -328.89 | 62.33 |
| `dino_boost--all_positive` | 1 | binary | 89.08 | 94.58 | 91.74 | 3013.17 | 3031.15 | -17.98 | 72.74 |
| `dino_boost--all_positive` | 1 | boundary | 100.00 | 94.63 | 97.24 | 2687.48 | 3031.15 | -343.68 | 72.74 |
| `dino_boost--all_positive` | 2 | binary | 90.98 | 96.87 | 93.83 | 3627.75 | 3675.15 | -47.40 | 83.71 |
| `dino_boost--all_positive` | 2 | boundary | 100.00 | 96.95 | 98.45 | 3307.64 | 3675.15 | -367.51 | 83.71 |
| `dino_boost--all_positive` | 3 | binary | 92.51 | 97.72 | 95.04 | 4237.58 | 4323.26 | -85.69 | 94.75 |
| `dino_boost--all_positive` | 3 | boundary | 100.00 | 97.84 | 98.90 | 3930.43 | 4323.26 | -392.83 | 94.75 |
| `dino_boost--all_candidates` | 0 | binary | 65.23 | 100.00 | 78.95 | 3660.81 | 2387.15 | 1273.65 | 137.83 |
| `dino_boost--all_candidates` | 0 | boundary | 100.00 | 100.00 | 100.00 | 2387.15 | 2387.15 | 0.00 | 137.83 |
| `dino_boost--all_candidates` | 1 | binary | 70.50 | 100.00 | 82.70 | 4300.05 | 3031.15 | 1268.90 | 137.83 |
| `dino_boost--all_candidates` | 1 | boundary | 100.00 | 100.00 | 100.00 | 3031.15 | 3031.15 | 0.00 | 137.83 |
| `dino_boost--all_candidates` | 2 | binary | 73.70 | 100.00 | 84.85 | 4987.65 | 3675.15 | 1312.50 | 137.83 |
| `dino_boost--all_candidates` | 2 | boundary | 100.00 | 100.00 | 100.00 | 3675.15 | 3675.15 | 0.00 | 137.83 |
| `dino_boost--all_candidates` | 3 | binary | 77.62 | 100.00 | 87.40 | 5570.37 | 4323.26 | 1247.11 | 137.83 |
| `dino_boost--all_candidates` | 3 | boundary | 100.00 | 100.00 | 100.00 | 4323.26 | 4323.26 | 0.00 | 137.83 |
| `dino_keep--positive_uncertain` | 0 | binary | 83.81 | 84.89 | 84.22 | 2426.11 | 2387.15 | 38.96 | 4.71 |
| `dino_keep--positive_uncertain` | 0 | boundary | 84.67 | 84.91 | 84.70 | 2399.16 | 2387.15 | 12.01 | 4.71 |
| `dino_keep--positive_uncertain` | 1 | binary | 86.87 | 93.75 | 90.13 | 3066.20 | 3031.15 | 35.05 | 5.99 |
| `dino_keep--positive_uncertain` | 1 | boundary | 87.81 | 93.75 | 90.65 | 3031.53 | 3031.15 | 0.38 | 5.99 |
| `dino_keep--positive_uncertain` | 2 | binary | 88.97 | 96.47 | 92.54 | 3688.06 | 3675.15 | 12.91 | 7.23 |
| `dino_keep--positive_uncertain` | 2 | boundary | 89.92 | 96.47 | 93.06 | 3648.09 | 3675.15 | -27.06 | 7.23 |
| `dino_keep--positive_uncertain` | 3 | binary | 90.61 | 97.51 | 93.91 | 4307.25 | 4323.26 | -16.01 | 8.44 |
| `dino_keep--positive_uncertain` | 3 | boundary | 91.53 | 97.51 | 94.42 | 4263.81 | 4323.26 | -59.46 | 8.44 |
| `dino_keep--uncertain_narrow` | 0 | binary | 69.47 | 96.75 | 80.82 | 3330.02 | 2387.15 | 942.87 | 63.82 |
| `dino_keep--uncertain_narrow` | 0 | boundary | 86.42 | 96.75 | 91.27 | 2676.42 | 2387.15 | 289.27 | 63.82 |
| `dino_keep--uncertain_narrow` | 1 | binary | 74.29 | 98.70 | 84.74 | 3970.39 | 3031.15 | 939.23 | 74.93 |
| `dino_keep--uncertain_narrow` | 1 | boundary | 93.71 | 98.70 | 96.11 | 3147.72 | 3031.15 | 116.57 | 74.93 |
| `dino_keep--uncertain_narrow` | 2 | binary | 77.43 | 99.30 | 86.99 | 4632.10 | 3675.15 | 956.95 | 87.45 |
| `dino_keep--uncertain_narrow` | 2 | boundary | 96.08 | 99.30 | 97.64 | 3732.66 | 3675.15 | 57.51 | 87.45 |
| `dino_keep--uncertain_narrow` | 3 | binary | 80.37 | 99.56 | 88.93 | 5258.41 | 4323.26 | 935.15 | 94.83 |
| `dino_keep--uncertain_narrow` | 3 | boundary | 97.12 | 99.56 | 98.31 | 4351.46 | 4323.26 | 28.19 | 94.83 |
| `dino_keep--uncertain_medium` | 0 | binary | 68.93 | 97.36 | 80.67 | 3376.56 | 2387.15 | 989.41 | 70.49 |
| `dino_keep--uncertain_medium` | 0 | boundary | 87.32 | 97.36 | 92.05 | 2663.48 | 2387.15 | 276.33 | 70.49 |
| `dino_keep--uncertain_medium` | 1 | binary | 73.78 | 98.92 | 84.49 | 4018.59 | 3031.15 | 987.44 | 81.80 |
| `dino_keep--uncertain_medium` | 1 | boundary | 94.85 | 98.92 | 96.83 | 3124.32 | 3031.15 | 93.17 | 81.80 |
| `dino_keep--uncertain_medium` | 2 | binary | 76.93 | 99.41 | 86.72 | 4683.51 | 3675.15 | 1008.36 | 94.72 |
| `dino_keep--uncertain_medium` | 2 | boundary | 97.09 | 99.41 | 98.23 | 3709.61 | 3675.15 | 34.46 | 94.72 |
| `dino_keep--uncertain_medium` | 3 | binary | 80.01 | 99.61 | 88.73 | 5304.53 | 4323.26 | 981.27 | 101.80 |
| `dino_keep--uncertain_medium` | 3 | boundary | 97.98 | 99.61 | 98.78 | 4331.79 | 4323.26 | 8.53 | 101.80 |
| `dino_keep--uncertain_wide` | 0 | binary | 68.32 | 98.09 | 80.51 | 3430.64 | 2387.15 | 1043.49 | 80.38 |
| `dino_keep--uncertain_wide` | 0 | boundary | 89.86 | 98.09 | 93.78 | 2606.97 | 2387.15 | 219.82 | 80.38 |
| `dino_keep--uncertain_wide` | 1 | binary | 73.30 | 99.27 | 84.31 | 4071.51 | 3031.15 | 1040.36 | 91.55 |
| `dino_keep--uncertain_wide` | 1 | boundary | 96.93 | 99.27 | 98.09 | 3076.94 | 3031.15 | 45.79 | 91.55 |
| `dino_keep--uncertain_wide` | 2 | binary | 76.53 | 99.59 | 86.54 | 4737.85 | 3675.15 | 1062.70 | 104.63 |
| `dino_keep--uncertain_wide` | 2 | boundary | 98.76 | 99.59 | 99.17 | 3670.29 | 3675.15 | -4.87 | 104.63 |
| `dino_keep--uncertain_wide` | 3 | binary | 79.63 | 99.73 | 88.55 | 5362.50 | 4323.26 | 1039.24 | 111.04 |
| `dino_keep--uncertain_wide` | 3 | boundary | 99.31 | 99.73 | 99.52 | 4299.93 | 4323.26 | -23.33 | 111.04 |
| `dino_keep--all_positive` | 0 | binary | 84.58 | 84.88 | 84.60 | 2404.33 | 2387.15 | 17.18 | 62.30 |
| `dino_keep--all_positive` | 0 | boundary | 100.00 | 84.91 | 91.82 | 2026.91 | 2387.15 | -360.24 | 62.30 |
| `dino_keep--all_positive` | 1 | binary | 87.86 | 93.74 | 90.65 | 3030.59 | 3031.15 | -0.56 | 72.69 |
| `dino_keep--all_positive` | 1 | boundary | 100.00 | 93.75 | 96.77 | 2658.66 | 3031.15 | -372.50 | 72.69 |
| `dino_keep--all_positive` | 2 | binary | 89.98 | 96.44 | 93.07 | 3643.70 | 3675.15 | -31.45 | 83.32 |
| `dino_keep--all_positive` | 2 | boundary | 100.00 | 96.47 | 98.20 | 3277.91 | 3675.15 | -397.24 | 83.32 |
| `dino_keep--all_positive` | 3 | binary | 91.59 | 97.47 | 94.42 | 4255.96 | 4323.26 | -67.30 | 94.14 |
| `dino_keep--all_positive` | 3 | boundary | 100.00 | 97.51 | 98.74 | 3900.94 | 4323.26 | -422.32 | 94.14 |
| `dino_keep--all_candidates` | 0 | binary | 64.70 | 100.00 | 78.56 | 3689.93 | 2387.15 | 1302.77 | 137.83 |
| `dino_keep--all_candidates` | 0 | boundary | 100.00 | 100.00 | 100.00 | 2387.15 | 2387.15 | 0.00 | 137.83 |
| `dino_keep--all_candidates` | 1 | binary | 70.01 | 100.00 | 82.36 | 4329.75 | 3031.15 | 1298.60 | 137.83 |
| `dino_keep--all_candidates` | 1 | boundary | 100.00 | 100.00 | 100.00 | 3031.15 | 3031.15 | 0.00 | 137.83 |
| `dino_keep--all_candidates` | 2 | binary | 73.21 | 100.00 | 84.53 | 5020.31 | 3675.15 | 1345.16 | 137.83 |
| `dino_keep--all_candidates` | 2 | boundary | 100.00 | 100.00 | 100.00 | 3675.15 | 3675.15 | 0.00 | 137.83 |
| `dino_keep--all_candidates` | 3 | binary | 77.21 | 100.00 | 87.14 | 5599.57 | 4323.26 | 1276.31 | 137.83 |
| `dino_keep--all_candidates` | 3 | boundary | 100.00 | 100.00 | 100.00 | 4323.26 | 4323.26 | 0.00 | 137.83 |

All source-group sensitivities, automatic baselines, seed statistics and per-recording decisions remain in the machine-readable summary and immutable cell files. Group scores are diagnostic and are not averaged to obtain dataset F1.

## Evidence

Contract SHA-256: `7f47932b9a18e030840449b7e4eceec489ac474d8eb5fa9dee13ee0a32a71b56`.

NAS root: `private-reference-0091`.

Summary: `private-reference-0164` (SHA-256 `6e0f3664d8695fbb9203f2b094972899a4464d1d8fdd0eb7e55738bf71fb25b4`).
Completed result index: `private-reference-0165` (SHA-256 `a18a93f1b0a09297b285a639bac7ebd7b35d7159bfd82ff6889dccb55b5862e5`).

Every one of 2,880 recording-level candidate plans and human outcomes was independently reconstructed without shared interval/model imports. Each binary and boundary outcome also passed an independent duration oracle across 13 scopes and all four paddings. Thirty focused tests passed before registration, plus 480 randomized plan/outcome cross-checks. Sources and inputs were hash-checked before and after execution. A separate summary audit and final reproducible archive are stored alongside the run.
