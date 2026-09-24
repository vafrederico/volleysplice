# Production and neural combination results — 19 September 2026

Completed 107 fixed automatic configurations (303 seed/configuration cells) and 30 review policies (90 cells), with independent interval construction and duration-accounting checks. No new training, production change, or protected-test evaluation occurred.

## Findings and recommendation

**Neural predictions offer a strong precision signal, but none of the tested automatic combinations passes the predeclared coverage-preserving improvement screen.** The highest development F1 comes from strict intersection of current production default with DINO+TCN global control: P_pad 91.67%, R_core 96.46%, F1_padP_coreR 94.00%, compared with production default 72.45% / 99.27% / 83.76%. Incorrect export falls from 22.70 to 4.98 minutes. However, omitted wanted human export rises from 1.57 to 6.58 minutes; actual missed play rises from 17.47 to 84.44 seconds. Completely lost rallies increase from 3 to a seed mean of 18.33, with up to 20 newly lost rallies and 41 rallies with worse coverage in a seed. This is a precision/recall tradeoff, not a coverage-preserving replacement.

The best standalone neural F1 remains DINO+TCN global control at 93.07%, with 96.96% core recall. Reviewed-export DINO short boost has similar recall (96.95%) and lower F1 (92.51%), but fewer completely lost rallies: 13.67 versus 17.33 on average. The distinction matters: time recall alone can hide the loss of several short points. Compact short boost reaches 90.31% F1 and 92.41% recall; compact keep rescue raises recall to 93.42% while F1 is 89.97%.

Two-of-three majority among production, compact boost and DINO global retains more play than strict intersection: 98.01% recall, 92.61% F1, and 7.93 minutes of incorrect export. It still misses 13.33 whole rallies on average and worsens up to 27 rallies per seed versus production default. Union of the two neural candidates reaches 98.18% recall and 91.31% F1, also below production's recall.

Protected-component trimming with compact boost preserves every rally's measured coverage versus production default in all three seeds, but removes only **6.5 additional seconds** of unwanted export over the entire dataset. F1 moves from 83.763% to 83.827%, well below the required two-point improvement. Using DINO global for the same guarded trimming saves 8.72 seconds of incorrect export but also loses 0.95 seconds of core play on average. Replacing the existing production suppression with a neural veto restricted to the unsuppressed ensemble's eligible components likewise provides no passing candidate. The current agreement protection leaves little room for a large additional automatic improvement.

Adding DINO global by union to production default recovers 9.97 of its 17.47 missed core seconds and reaches 99.69% recall. It adds 1.50 minutes of incorrect export, however, and lowers F1 to 83.16%. This is a recall-oriented option, not an F1 improvement. These unions cannot worsen existing export coverage by construction; the meaningful tradeoff is recovered play against added unwanted footage.

**The useful next product experiment is disagreement-assisted review**, keeping the automatic export unchanged while marking disputed regions for review. With production default and DINO global, suppression-only flags contain 17.57 of the 22.70 minutes of incorrect export (77.39%), but also 4.93 minutes of wanted export. Playback with context totals 60.02 minutes across about 396 clips. Bidirectional flags add possible missed play: they contain 9.97 seconds (57.05%) of production's missed core, with 68.35 minutes of playback across about 452 clips. DINO short boost flags slightly more missed core (11.73 seconds, 67.14%) with 69.03 minutes of playback. These are substantial workloads on 137.83 minutes of footage, so disagreement alone is not yet a compact review queue or a calibrated low-confidence score.

For phone/browser work, compact boost is the lighter review-model candidate: its bidirectional queue captures 73.68% of incorrect export and 37.77% of missed core, requiring 69.39 minutes of playback. For desktop RTX 3080 work, DINO offers the stronger measured error signal. This experiment measures detection quality only; it does not establish end-to-end phone/browser or desktop inference latency. A next bounded study could prioritize/coalesce disagreement clips and measure real reviewer time and corrections on fresh source groups; any tuned ranking must be evaluated separately. The existing DINO feature path should remain a desktop candidate until deployment cost is measured or a smaller model is trained to reproduce its useful signal.

Perfect correction of all DINO-global bidirectional disputed seconds would reach 95.77% F1, but that is a ground-truth-assisted upper bound, not a tested automatic configuration or observed human outcome. Merely showing review flags leaves the production metrics and exported footage unchanged. These queue estimates use canonical exports; exact editor workload was not replayed. Actual production app exports differ from canonical default exports by only 0.119 seconds of symmetric difference at the target padding, so the headline production percentages round identically here.

## Reading the tables

**Target: symmetric ±2 seconds; positive gaps are joined only when strictly below 3 seconds.** All tables use the same 8 indoor/grass recordings, 4 source groups and 322 labeled rallies. Ignored time is removed from all interval unions and never rejoined. Neural values average three seed results; each seed pools duration counts across recordings before computing ratios. Seconds are totals over the eight recordings, averaged across seeds, not three concatenated copies of the dataset.

Evaluable footage: **137.83 min**. Human export at ±2 s: **61.25 min**. Actual rally core: **39.79 min**.

- **P_pad**: share of exported time within the equally padded human export.
- **R_core**: share of actual rally core retained in the model export.
- **F1_padP_coreR**: harmonic mean of those two values; the fixed primary ranking metric.
- **Correctly removed**: unwanted footage excluded from export (TN).
- **Incorrectly removed**: wanted padded human export omitted (FNpad), including desired context.
- **Incorrect export**: unwanted footage retained (FP).
- **Missed core**: actual play omitted; this is the loss reflected by R_core.

Export + correctly removed + incorrectly removed partitions evaluable video time. Incorrect export is a subset of export. Neither export duration alone nor model-minus-human duration measures error.

## Production identity and scope limits

“Production default” means the current checked-in browser/Android fresh-project default: aggressive whole-rally suppression. Saved projects can retain different settings. This study did not query the deployed website or an installed APK. The unsuppressed production union, balanced and conservative suppression options are listed separately.

This observed default differs from the canonical vault decision dated 18 August, which requested suppression disabled by default. The comparison records the checked-in behavior without changing either production or the approved decision. A pending draft records the discrepancy for future reconciliation.

The adapter replayed the actual checked-in browser TypeScript runtime against audited feature caches; both shipped raw heads matched cached endpoints exactly. Browser and Android core weight assets match. This does not revalidate native feature extraction on devices. The separate app-fidelity table preserves materialized exports and suppression barriers, while primary ranking uses the repository’s canonical export contract.

Shipped production weights have historical exposure to these videos. Source-group-held refit heads are a separate sensitivity comparison; their historical decoder/settings remain development-exposed and no refitted suppression head was introduced. Neural candidates and recipes are adaptive development choices. Rankings are descriptive on this scope, not an untouched estimate of generalization or permission to deploy.

## Standalone models and production

| Configuration | P_pad % | R_core % | F1_padP_coreR % | Export min | Correctly removed min | Incorrectly removed min | Incorrect export min | Missed core s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `shippedPrevious` | 71.98 | 98.51 | 83.18 | 81.65 | 53.70 | 2.48 | 22.88 | 35.61 |
| `shippedV2` | 73.99 | 97.88 | 84.28 | 78.87 | 56.07 | 2.89 | 20.51 | 50.51 |
| `shippedUnion` | 67.04 | 99.64 | 80.15 | 89.69 | 47.02 | 1.13 | 29.56 | 8.58 |
| `refitPrevious` | 59.99 | 87.90 | 71.31 | 86.41 | 42.01 | 9.42 | 34.57 | 288.84 |
| `refitV2` | 66.88 | 94.62 | 78.37 | 83.69 | 48.86 | 5.28 | 27.72 | 128.38 |
| `refitUnion` | 58.71 | 98.32 | 73.52 | 100.56 | 35.07 | 2.21 | 41.52 | 40.17 |
| `productionDefault` | 72.45 | 99.27 | 83.76 | 82.38 | 53.88 | 1.57 | 22.70 | 17.47 |
| `productionBalanced` | 70.34 | 99.58 | 82.44 | 85.31 | 51.28 | 1.25 | 25.31 | 10.00 |
| `productionConservative` | 69.23 | 99.58 | 81.68 | 86.68 | 49.91 | 1.25 | 26.67 | 10.00 |
| `compact_boost` | 88.30 | 92.41 | 90.31 | 59.06 | 69.67 | 9.11 | 6.91 | 181.20 |
| `compact_keep` | 86.79 | 93.42 | 89.97 | 61.05 | 68.49 | 8.29 | 8.09 | 157.02 |
| `dino_global` | 89.50 | 96.96 | 93.07 | 61.78 | 70.07 | 5.98 | 6.51 | 72.57 |
| `dino_boost` | 88.47 | 96.95 | 92.51 | 62.33 | 69.38 | 6.13 | 7.20 | 72.70 |
| `dino_keep` | 87.91 | 96.47 | 91.94 | 62.30 | 68.91 | 6.62 | 7.67 | 84.18 |
| `compact_baseline` | 87.34 | 92.49 | 89.82 | 59.79 | 68.99 | 9.06 | 7.60 | 179.23 |
| `dino_baseline` | 87.92 | 95.80 | 91.63 | 61.53 | 69.00 | 7.31 | 7.59 | 100.18 |

## Configuration key

| Key | Meaning |
| --- | --- |
| `productionDefault` | Production default (aggressive suppression) |
| `productionBalanced` | Production balanced suppression |
| `productionConservative` | Production conservative suppression |
| `shippedPrevious` | Shipped previous head |
| `shippedV2` | Shipped v2 head |
| `shippedUnion` | Production unsuppressed union |
| `refitPrevious` | Source-held refit previous head |
| `refitV2` | Source-held refit v2 head |
| `refitUnion` | Source-held refit union |
| `compact_boost` | Compact TCN, short boost |
| `compact_keep` | Compact TCN, keep rescue |
| `compact_baseline` | Compact TCN, baseline |
| `dino_global` | DINO+TCN, global control |
| `dino_boost` | DINO+TCN, short boost |
| `dino_keep` | DINO+TCN, keep rescue |
| `dino_baseline` | DINO+TCN, baseline |
| `best_f1_pair` | Compact boost + DINO global |
| `recovery_pair` | Compact keep + DINO boost |
| `union` | union |
| `intersection` | strict intersection |
| `tolerant_intersection` | 2 s tolerant intersection |
| `guarded_trim` | protected-component trimming |
| `guarded_component_rejection` | protected-component rejection |
| `nn_union` | NN union |
| `nn_intersection` | NN intersection |
| `three_union` | three-model union |
| `majority` | two-of-three majority |

All combinations operate on raw core intervals before final padding/joining. Two-second tolerant support is dilation only, without short-gap joining. Protected-component policies preserve retained raw time in components supported by both production models, including original heads and tails; agreement uses ±2 s and strictly <0.5 s joins. One-model components remain eligible for suppression, and previously suppressed time is not restored. Source-held refit recipes use refit source tags. Matched neural seeds are paired, never all nine seed combinations.

## Complete automatic matrix at target padding

Sorted by mean development F1_padP_coreR at the predeclared ±2 s. All fixed candidates are retained, including aggressive veto controls and candidates failing coverage guardrails. The source-held refit rows and shipped-exposure rows are distinct evidence regimes.

| Configuration | P_pad % | R_core % | F1_padP_coreR % | Export min | Correctly removed min | Incorrectly removed min | Incorrect export min | Missed core s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--dino_global--intersection` | 91.67 | 96.46 | 94.00 | 59.66 | 71.60 | 6.58 | 4.98 | 84.44 |
| `productionDefault--dino_boost--intersection` | 91.47 | 96.43 | 93.88 | 59.59 | 71.49 | 6.76 | 5.09 | 85.16 |
| `shippedUnion--dino_global--intersection` | 91.19 | 96.69 | 93.86 | 60.18 | 71.27 | 6.38 | 5.31 | 78.96 |
| `shippedUnion--dino_boost--intersection` | 91.01 | 96.72 | 93.78 | 60.16 | 71.17 | 6.51 | 5.42 | 78.23 |
| `productionDefault--dino_keep--intersection` | 91.03 | 95.95 | 93.41 | 59.39 | 71.21 | 7.24 | 5.37 | 96.67 |
| `shippedUnion--dino_keep--intersection` | 90.68 | 96.20 | 93.34 | 59.88 | 70.95 | 7.01 | 5.64 | 90.65 |
| `dino_global` | 89.50 | 96.96 | 93.07 | 61.78 | 70.07 | 5.98 | 6.51 | 72.57 |
| `refitUnion--dino_global--intersection` | 90.37 | 95.26 | 92.74 | 59.54 | 70.82 | 7.47 | 5.76 | 113.21 |
| `productionDefault--best_f1_pair--majority` | 87.77 | 98.01 | 92.61 | 64.79 | 68.65 | 4.39 | 7.93 | 47.46 |
| `dino_boost` | 88.47 | 96.95 | 92.51 | 62.33 | 69.38 | 6.13 | 7.20 | 72.70 |
| `recovery_pair--nn_intersection` | 93.04 | 91.77 | 92.40 | 54.31 | 72.80 | 10.72 | 3.78 | 196.40 |
| `shippedUnion--best_f1_pair--majority` | 87.28 | 98.11 | 92.38 | 65.27 | 68.28 | 4.29 | 8.30 | 45.10 |
| `best_f1_pair--nn_intersection` | 93.51 | 91.07 | 92.27 | 53.62 | 73.09 | 11.12 | 3.49 | 213.25 |
| `refitUnion--dino_boost--intersection` | 89.44 | 95.03 | 92.14 | 59.83 | 70.25 | 7.75 | 6.33 | 118.61 |
| `productionDefault--recovery_pair--majority` | 86.79 | 98.11 | 92.10 | 65.62 | 67.91 | 4.30 | 8.67 | 45.11 |
| `shippedUnion--recovery_pair--majority` | 86.47 | 98.29 | 92.00 | 66.07 | 67.64 | 4.12 | 8.94 | 40.76 |
| `dino_keep` | 87.91 | 96.47 | 91.94 | 62.30 | 68.91 | 6.62 | 7.67 | 84.18 |
| `refitUnion--dino_keep--intersection` | 88.92 | 94.78 | 91.72 | 59.97 | 69.84 | 8.02 | 6.74 | 124.62 |
| `dino_baseline` | 87.92 | 95.80 | 91.63 | 61.53 | 69.00 | 7.31 | 7.59 | 100.18 |
| `refitUnion--best_f1_pair--majority` | 86.33 | 97.57 | 91.61 | 65.40 | 67.64 | 4.79 | 8.94 | 57.96 |
| `best_f1_pair--nn_union` | 85.34 | 98.18 | 91.31 | 67.05 | 66.74 | 4.03 | 9.84 | 43.42 |
| `productionDefault--compact_keep--intersection` | 89.16 | 93.07 | 91.06 | 58.99 | 70.17 | 8.67 | 6.41 | 165.44 |
| `productionDefault--compact_boost--intersection` | 89.82 | 92.09 | 90.94 | 57.70 | 70.70 | 9.43 | 5.88 | 188.75 |
| `shippedUnion--compact_keep--intersection` | 88.65 | 93.22 | 90.87 | 59.47 | 69.82 | 8.55 | 6.76 | 161.78 |
| `shippedUnion--compact_boost--intersection` | 89.43 | 92.22 | 90.80 | 58.06 | 70.44 | 9.34 | 6.14 | 185.63 |
| `refitUnion--recovery_pair--majority` | 84.69 | 97.73 | 90.74 | 66.78 | 66.36 | 4.70 | 10.23 | 54.15 |
| `recovery_pair--nn_union` | 83.45 | 98.42 | 90.32 | 68.81 | 65.19 | 3.83 | 11.39 | 37.79 |
| `compact_boost` | 88.30 | 92.41 | 90.31 | 59.06 | 69.67 | 9.11 | 6.91 | 181.20 |
| `compact_keep` | 86.79 | 93.42 | 89.97 | 61.05 | 68.49 | 8.29 | 8.09 | 157.02 |
| `refitUnion--compact_boost--intersection` | 88.72 | 91.07 | 89.88 | 57.55 | 70.08 | 10.20 | 6.50 | 213.24 |
| `compact_baseline` | 87.34 | 92.49 | 89.82 | 59.79 | 68.99 | 9.06 | 7.60 | 179.23 |
| `productionDefault--dino_boost--tolerant_intersection` | 83.09 | 97.58 | 89.75 | 69.72 | 64.79 | 3.32 | 11.79 | 57.67 |
| `refitUnion--compact_keep--intersection` | 87.60 | 92.00 | 89.73 | 59.18 | 69.22 | 9.44 | 7.37 | 190.96 |
| `productionDefault--dino_keep--tolerant_intersection` | 82.93 | 97.40 | 89.57 | 69.64 | 64.67 | 3.52 | 11.91 | 62.08 |
| `shippedUnion--dino_boost--tolerant_intersection` | 82.35 | 97.87 | 89.44 | 70.68 | 64.11 | 3.05 | 12.48 | 50.74 |
| `productionDefault--dino_global--tolerant_intersection` | 82.61 | 97.46 | 89.42 | 69.91 | 64.42 | 3.50 | 12.16 | 60.53 |
| `shippedUnion--dino_keep--tolerant_intersection` | 82.29 | 97.65 | 89.30 | 70.48 | 64.07 | 3.28 | 12.52 | 56.06 |
| `shippedUnion--dino_global--tolerant_intersection` | 81.84 | 97.69 | 89.06 | 70.84 | 63.70 | 3.29 | 12.88 | 55.05 |
| `refitUnion--dino_global--tolerant_intersection` | 80.30 | 96.34 | 87.58 | 70.78 | 62.61 | 4.44 | 13.97 | 87.39 |
| `productionDefault--compact_boost--tolerant_intersection` | 81.32 | 94.32 | 87.34 | 68.13 | 63.85 | 5.85 | 12.73 | 135.69 |
| `refitUnion--dino_boost--tolerant_intersection` | 79.78 | 96.40 | 87.30 | 71.36 | 62.14 | 4.33 | 14.44 | 85.89 |
| `productionDefault--compact_keep--tolerant_intersection` | 80.63 | 94.91 | 87.18 | 69.45 | 63.12 | 5.26 | 13.46 | 121.58 |
| `refitUnion--dino_keep--tolerant_intersection` | 79.63 | 96.22 | 87.11 | 71.36 | 61.97 | 4.50 | 14.61 | 90.27 |
| `shippedUnion--compact_boost--tolerant_intersection` | 80.81 | 94.45 | 87.10 | 68.68 | 63.40 | 5.75 | 13.18 | 132.57 |
| `shippedUnion--compact_keep--tolerant_intersection` | 79.88 | 95.06 | 86.81 | 70.27 | 62.44 | 5.13 | 14.15 | 117.92 |
| `refitUnion--compact_boost--tolerant_intersection` | 79.47 | 93.25 | 85.81 | 68.52 | 62.51 | 6.80 | 14.07 | 161.04 |
| `refitUnion--compact_keep--tolerant_intersection` | 78.22 | 93.84 | 85.31 | 70.37 | 61.23 | 6.23 | 15.35 | 147.07 |
| `shippedV2` | 73.99 | 97.88 | 84.28 | 78.87 | 56.07 | 2.89 | 20.51 | 50.51 |
| `productionDefault--dino_global--guarded_trim` | 72.57 | 99.23 | 83.83 | 82.21 | 54.03 | 1.59 | 22.55 | 18.42 |
| `productionDefault--compact_boost--guarded_component_rejection` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--compact_boost--guarded_trim` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--compact_keep--guarded_component_rejection` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--compact_keep--guarded_trim` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_boost--guarded_component_rejection` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_boost--guarded_trim` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_keep--guarded_component_rejection` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_keep--guarded_trim` | 72.54 | 99.27 | 83.83 | 82.28 | 53.99 | 1.57 | 22.59 | 17.47 |
| `productionDefault--dino_global--guarded_component_rejection` | 72.55 | 99.23 | 83.82 | 82.22 | 54.01 | 1.59 | 22.57 | 18.42 |
| `productionDefault` | 72.45 | 99.27 | 83.76 | 82.38 | 53.88 | 1.57 | 22.70 | 17.47 |
| `productionDefault--compact_boost--union` | 71.70 | 99.54 | 83.36 | 83.64 | 52.91 | 1.28 | 23.67 | 10.87 |
| `shippedPrevious` | 71.98 | 98.51 | 83.18 | 81.65 | 53.70 | 2.48 | 22.88 | 35.61 |
| `productionDefault--dino_global--union` | 71.33 | 99.69 | 83.16 | 84.40 | 52.38 | 1.05 | 24.20 | 7.50 |
| `productionDefault--compact_keep--union` | 71.21 | 99.58 | 83.04 | 84.29 | 52.31 | 1.24 | 24.27 | 10.01 |
| `productionDefault--dino_boost--union` | 70.90 | 99.76 | 82.89 | 85.00 | 51.85 | 0.98 | 24.73 | 5.74 |
| `productionDefault--best_f1_pair--three_union` | 70.78 | 99.70 | 82.78 | 85.13 | 51.70 | 1.00 | 24.88 | 7.17 |
| `productionDefault--dino_keep--union` | 70.74 | 99.71 | 82.76 | 85.17 | 51.65 | 1.02 | 24.93 | 6.87 |
| `productionBalanced` | 70.34 | 99.58 | 82.44 | 85.31 | 51.28 | 1.25 | 25.31 | 10.00 |
| `productionDefault--recovery_pair--three_union` | 70.06 | 99.81 | 82.33 | 86.15 | 50.79 | 0.90 | 25.80 | 4.54 |
| `shippedUnion--compact_boost--guarded_trim` | 69.29 | 99.58 | 81.72 | 86.61 | 49.98 | 1.25 | 26.60 | 10.00 |
| `shippedUnion--dino_keep--guarded_trim` | 69.28 | 99.58 | 81.71 | 86.61 | 49.98 | 1.25 | 26.60 | 10.00 |
| `shippedUnion--compact_boost--guarded_component_rejection` | 69.28 | 99.58 | 81.71 | 86.61 | 49.97 | 1.25 | 26.61 | 10.00 |
| `shippedUnion--dino_keep--guarded_component_rejection` | 69.28 | 99.58 | 81.71 | 86.61 | 49.97 | 1.25 | 26.61 | 10.00 |
| `shippedUnion--dino_global--guarded_trim` | 69.28 | 99.54 | 81.70 | 86.58 | 49.98 | 1.27 | 26.60 | 10.95 |
| `shippedUnion--dino_boost--guarded_trim` | 69.24 | 99.58 | 81.68 | 86.66 | 49.92 | 1.25 | 26.66 | 10.00 |
| `shippedUnion--dino_global--guarded_component_rejection` | 69.25 | 99.54 | 81.68 | 86.61 | 49.95 | 1.27 | 26.63 | 10.95 |
| `productionConservative` | 69.23 | 99.58 | 81.68 | 86.68 | 49.91 | 1.25 | 26.67 | 10.00 |
| `shippedUnion--compact_keep--guarded_trim` | 69.22 | 99.58 | 81.67 | 86.69 | 49.90 | 1.25 | 26.68 | 10.00 |
| `shippedUnion--compact_keep--guarded_component_rejection` | 69.22 | 99.58 | 81.67 | 86.69 | 49.90 | 1.25 | 26.68 | 10.00 |
| `shippedUnion--dino_boost--guarded_component_rejection` | 69.22 | 99.58 | 81.67 | 86.69 | 49.90 | 1.25 | 26.68 | 10.00 |
| `shippedUnion` | 67.04 | 99.64 | 80.15 | 89.69 | 47.02 | 1.13 | 29.56 | 8.58 |
| `shippedUnion--compact_boost--union` | 66.62 | 99.79 | 79.90 | 90.52 | 46.37 | 0.94 | 30.21 | 5.11 |
| `shippedUnion--dino_global--union` | 66.39 | 99.83 | 79.74 | 91.04 | 45.98 | 0.82 | 30.60 | 4.10 |
| `shippedUnion--compact_keep--union` | 66.30 | 99.80 | 79.67 | 90.98 | 45.92 | 0.93 | 30.66 | 4.78 |
| `shippedUnion--best_f1_pair--three_union` | 66.03 | 99.84 | 79.49 | 91.59 | 45.47 | 0.77 | 31.11 | 3.76 |
| `shippedUnion--dino_boost--union` | 65.96 | 99.86 | 79.45 | 91.65 | 45.38 | 0.80 | 31.20 | 3.35 |
| `shippedUnion--dino_keep--union` | 65.79 | 99.85 | 79.32 | 91.88 | 45.14 | 0.81 | 31.44 | 3.63 |
| `shippedUnion--recovery_pair--three_union` | 65.39 | 99.89 | 79.03 | 92.55 | 44.55 | 0.74 | 32.04 | 2.70 |
| `refitV2` | 66.88 | 94.62 | 78.37 | 83.69 | 48.86 | 5.28 | 27.72 | 128.38 |
| `refitUnion--dino_boost--guarded_trim` | 61.17 | 98.17 | 75.38 | 96.18 | 39.24 | 2.42 | 37.35 | 43.69 |
| `refitUnion--dino_global--guarded_trim` | 61.16 | 98.11 | 75.35 | 96.18 | 39.22 | 2.43 | 37.36 | 45.04 |
| `refitUnion--dino_keep--guarded_trim` | 61.14 | 98.07 | 75.32 | 96.10 | 39.23 | 2.50 | 37.35 | 45.96 |
| `refitUnion--dino_boost--guarded_component_rejection` | 61.10 | 98.17 | 75.32 | 96.30 | 39.12 | 2.42 | 37.46 | 43.69 |
| `refitUnion--dino_global--guarded_component_rejection` | 61.10 | 98.11 | 75.31 | 96.26 | 39.14 | 2.43 | 37.44 | 45.04 |
| `refitUnion--dino_keep--guarded_component_rejection` | 61.10 | 98.07 | 75.30 | 96.15 | 39.18 | 2.50 | 37.40 | 45.96 |
| `refitUnion--compact_keep--guarded_component_rejection` | 60.99 | 97.42 | 75.02 | 95.62 | 39.28 | 2.93 | 37.30 | 61.59 |
| `refitUnion--compact_boost--guarded_component_rejection` | 61.00 | 97.28 | 74.99 | 95.41 | 39.38 | 3.05 | 37.21 | 64.86 |
| `refitUnion--compact_keep--guarded_trim` | 61.02 | 97.25 | 74.98 | 95.42 | 39.38 | 3.03 | 37.20 | 65.73 |
| `refitUnion--compact_boost--guarded_trim` | 61.01 | 97.08 | 74.93 | 95.21 | 39.45 | 3.17 | 37.13 | 69.77 |
| `refitUnion--dino_boost--union` | 58.83 | 99.88 | 74.04 | 102.47 | 34.39 | 0.97 | 42.19 | 2.98 |
| `refitUnion--dino_global--union` | 58.83 | 99.76 | 74.01 | 102.36 | 34.44 | 1.03 | 42.14 | 5.64 |
| `refitUnion--dino_keep--union` | 58.75 | 99.79 | 73.96 | 102.43 | 34.33 | 1.08 | 42.25 | 5.09 |
| `refitUnion--compact_boost--union` | 58.87 | 99.41 | 73.95 | 101.79 | 34.71 | 1.33 | 41.87 | 14.01 |
| `refitUnion--best_f1_pair--three_union` | 58.73 | 99.80 | 73.94 | 102.74 | 34.17 | 0.92 | 42.41 | 4.85 |
| `refitUnion--compact_keep--union` | 58.81 | 99.52 | 73.93 | 101.98 | 34.58 | 1.28 | 42.00 | 11.44 |
| `refitUnion--recovery_pair--three_union` | 58.65 | 99.89 | 73.91 | 102.97 | 34.00 | 0.86 | 42.58 | 2.69 |
| `refitUnion` | 58.71 | 98.32 | 73.52 | 100.56 | 35.07 | 2.21 | 41.52 | 40.17 |
| `refitPrevious` | 59.99 | 87.90 | 71.31 | 86.41 | 42.01 | 9.42 | 34.57 | 288.84 |

## Coverage and event guardrails

Loss counts are out of 322 rallies and can be fractional because of the seed average. Long means rally core longer than 3 s. Event F1 uses the evaluator’s raw-event IoU matching and is a secondary guardrail. “Worsened max” counts any lower retained rally-core duration than that candidate’s paired anchor in the worst seed, including partially cut rallies. Standalone neural/pair rows use production default as their paired anchor.

| Configuration | Event F1 % | Long R % | Complete losses | Partial losses | Short complete losses | Paired anchor | New complete max | Worsened max | Conservative screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--dino_global--intersection` | 77.81 | 97.51 | 18.33 | 26.67 | 15.67 | productionDefault | 20 | 41 | fail |
| `productionDefault--dino_boost--intersection` | 75.78 | 97.27 | 15.67 | 32.00 | 12.67 | productionDefault | 16 | 46 | fail |
| `shippedUnion--dino_global--intersection` | 77.19 | 97.64 | 17.33 | 25.67 | 14.67 | shippedUnion | 23 | 44 | fail |
| `shippedUnion--dino_boost--intersection` | 75.25 | 97.40 | 14.00 | 31.00 | 11.00 | shippedUnion | 18 | 47 | fail |
| `productionDefault--dino_keep--intersection` | 75.08 | 96.82 | 16.33 | 38.33 | 14.00 | productionDefault | 17 | 52 | fail |
| `shippedUnion--dino_keep--intersection` | 74.80 | 96.96 | 15.00 | 37.33 | 12.67 | shippedUnion | 20 | 55 | fail |
| `dino_global` | 77.16 | 97.93 | 17.33 | 22.33 | 14.67 | productionDefault | 20 | 40 | fail |
| `refitUnion--dino_global--intersection` | 70.65 | 96.39 | 20.67 | 35.33 | 17.00 | refitUnion | 25 | 45 | fail |
| `productionDefault--best_f1_pair--majority` | 77.67 | 98.82 | 13.33 | 15.67 | 11.67 | productionDefault | 13 | 27 | fail |
| `dino_boost` | 74.17 | 97.65 | 13.67 | 29.33 | 11.00 | productionDefault | 16 | 46 | fail |
| `recovery_pair--nn_intersection` | 69.46 | 93.17 | 34.33 | 48.67 | 23.00 | productionDefault | 36 | 84 | fail |
| `shippedUnion--best_f1_pair--majority` | 77.09 | 98.82 | 12.33 | 15.67 | 10.67 | shippedUnion | 16 | 30 | fail |
| `best_f1_pair--nn_intersection` | 70.78 | 92.76 | 39.33 | 48.67 | 26.33 | productionDefault | 40 | 88 | fail |
| `refitUnion--dino_boost--intersection` | 68.56 | 95.98 | 18.33 | 44.67 | 14.67 | refitUnion | 21 | 55 | fail |
| `productionDefault--recovery_pair--majority` | 76.56 | 98.83 | 11.00 | 18.33 | 9.33 | productionDefault | 10 | 30 | fail |
| `shippedUnion--recovery_pair--majority` | 76.26 | 98.83 | 9.00 | 18.33 | 7.33 | shippedUnion | 11 | 31 | fail |
| `dino_keep` | 73.77 | 97.25 | 15.00 | 34.00 | 12.67 | productionDefault | 17 | 52 | fail |
| `refitUnion--dino_keep--intersection` | 67.92 | 95.73 | 18.67 | 45.67 | 15.33 | refitUnion | 22 | 56 | fail |
| `dino_baseline` | 71.45 | 96.91 | 21.00 | 33.00 | 17.33 | productionDefault | 30 | 63 | fail |
| `refitUnion--best_f1_pair--majority` | 72.67 | 98.43 | 14.67 | 19.00 | 12.33 | refitUnion | 17 | 32 | fail |
| `best_f1_pair--nn_union` | 75.14 | 98.89 | 12.33 | 14.67 | 10.67 | productionDefault | 13 | 27 | fail |
| `productionDefault--compact_keep--intersection` | 70.19 | 94.25 | 28.67 | 40.33 | 18.67 | productionDefault | 30 | 66 | fail |
| `productionDefault--compact_boost--intersection` | 70.42 | 93.48 | 33.00 | 45.00 | 21.67 | productionDefault | 33 | 74 | fail |
| `shippedUnion--compact_keep--intersection` | 69.56 | 94.39 | 28.33 | 39.33 | 18.33 | shippedUnion | 33 | 69 | fail |
| `shippedUnion--compact_boost--intersection` | 69.95 | 93.62 | 33.00 | 44.00 | 21.67 | shippedUnion | 36 | 77 | fail |
| `refitUnion--recovery_pair--majority` | 72.16 | 98.54 | 12.00 | 24.33 | 10.33 | refitUnion | 14 | 38 | fail |
| `recovery_pair--nn_union` | 73.05 | 98.96 | 8.67 | 18.33 | 7.33 | productionDefault | 10 | 30 | fail |
| `compact_boost` | 69.45 | 93.82 | 33.00 | 42.00 | 21.67 | productionDefault | 33 | 74 | fail |
| `compact_keep` | 68.62 | 94.60 | 28.33 | 37.33 | 18.33 | productionDefault | 30 | 66 | fail |
| `refitUnion--compact_boost--intersection` | 66.12 | 92.40 | 35.00 | 50.00 | 21.67 | refitUnion | 36 | 76 | fail |
| `compact_baseline` | 68.62 | 94.03 | 35.67 | 36.33 | 23.00 | productionDefault | 39 | 76 | fail |
| `productionDefault--dino_boost--tolerant_intersection` | 74.78 | 98.46 | 15.33 | 12.00 | 12.67 | productionDefault | 16 | 23 | fail |
| `refitUnion--compact_keep--intersection` | 65.99 | 93.14 | 30.67 | 45.67 | 18.67 | refitUnion | 33 | 68 | fail |
| `productionDefault--dino_keep--tolerant_intersection` | 74.69 | 98.35 | 16.33 | 13.00 | 14.00 | productionDefault | 17 | 27 | fail |
| `shippedUnion--dino_boost--tolerant_intersection` | 74.00 | 98.60 | 13.67 | 11.00 | 11.00 | shippedUnion | 18 | 25 | fail |
| `productionDefault--dino_global--tolerant_intersection` | 75.08 | 98.56 | 18.33 | 11.67 | 15.67 | productionDefault | 20 | 24 | fail |
| `shippedUnion--dino_keep--tolerant_intersection` | 73.88 | 98.49 | 15.00 | 12.00 | 12.67 | shippedUnion | 20 | 30 | fail |
| `shippedUnion--dino_global--tolerant_intersection` | 74.05 | 98.70 | 17.33 | 10.67 | 14.67 | shippedUnion | 23 | 27 | fail |
| `refitUnion--dino_global--tolerant_intersection` | 66.28 | 97.38 | 17.33 | 25.67 | 14.67 | refitUnion | 22 | 27 | fail |
| `productionDefault--compact_boost--tolerant_intersection` | 71.05 | 95.71 | 31.67 | 20.67 | 20.33 | productionDefault | 32 | 46 | fail |
| `refitUnion--dino_boost--tolerant_intersection` | 65.71 | 97.24 | 14.33 | 27.33 | 11.67 | refitUnion | 17 | 25 | fail |
| `productionDefault--compact_keep--tolerant_intersection` | 71.01 | 96.11 | 27.67 | 15.33 | 18.00 | productionDefault | 27 | 38 | fail |
| `refitUnion--dino_keep--tolerant_intersection` | 65.14 | 97.10 | 15.33 | 28.00 | 13.00 | refitUnion | 19 | 30 | fail |
| `shippedUnion--compact_boost--tolerant_intersection` | 70.31 | 95.85 | 31.67 | 19.67 | 20.33 | shippedUnion | 35 | 49 | fail |
| `shippedUnion--compact_keep--tolerant_intersection` | 69.98 | 96.25 | 27.33 | 14.33 | 17.67 | shippedUnion | 30 | 41 | fail |
| `refitUnion--compact_boost--tolerant_intersection` | 65.26 | 94.62 | 33.00 | 29.33 | 20.67 | refitUnion | 35 | 49 | fail |
| `refitUnion--compact_keep--tolerant_intersection` | 64.72 | 95.00 | 29.00 | 25.00 | 18.00 | refitUnion | 30 | 40 | fail |
| `shippedV2` | 61.41 | 97.95 | 3.00 | 21.00 | 2.00 | — | — | — | — |
| `productionDefault--dino_global--guarded_trim` | 66.60 | 99.44 | 3.00 | 7.33 | 3.00 | productionDefault | 0 | 1 | fail |
| `productionDefault--compact_boost--guarded_component_rejection` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--compact_boost--guarded_trim` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--compact_keep--guarded_component_rejection` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--compact_keep--guarded_trim` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_boost--guarded_component_rejection` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_boost--guarded_trim` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_keep--guarded_component_rejection` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_keep--guarded_trim` | 66.57 | 99.48 | 3.00 | 7.00 | 3.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_global--guarded_component_rejection` | 66.60 | 99.44 | 3.00 | 7.33 | 3.00 | productionDefault | 0 | 1 | fail |
| `productionDefault` | 66.47 | 99.48 | 3.00 | 7.00 | 3.00 | — | — | — | — |
| `productionDefault--compact_boost--union` | 65.57 | 99.77 | 3.00 | 4.33 | 3.00 | productionDefault | 0 | 0 | fail |
| `shippedPrevious` | 66.58 | 98.58 | 1.00 | 15.00 | 1.00 | — | — | — | — |
| `productionDefault--dino_global--union` | 66.48 | 99.82 | 2.00 | 3.33 | 2.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--compact_keep--union` | 65.00 | 99.79 | 2.67 | 4.33 | 2.67 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_boost--union` | 65.13 | 99.83 | 1.33 | 4.00 | 1.33 | productionDefault | 0 | 0 | fail |
| `productionDefault--best_f1_pair--three_union` | 64.87 | 99.83 | 2.00 | 3.33 | 2.00 | productionDefault | 0 | 0 | fail |
| `productionDefault--dino_keep--union` | 65.22 | 99.82 | 1.67 | 3.33 | 1.67 | productionDefault | 0 | 0 | fail |
| `productionBalanced` | 64.43 | 99.62 | 1.00 | 6.00 | 1.00 | — | — | — | — |
| `productionDefault--recovery_pair--three_union` | 63.77 | 99.86 | 1.00 | 4.00 | 1.00 | productionDefault | 0 | 0 | fail |
| `shippedUnion--compact_boost--guarded_trim` | 63.59 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--dino_keep--guarded_trim` | 63.59 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--compact_boost--guarded_component_rejection` | 63.59 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--dino_keep--guarded_component_rejection` | 63.59 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--dino_global--guarded_trim` | 63.59 | 99.58 | 1.00 | 6.33 | 1.00 | shippedUnion | 1 | 2 | fail |
| `shippedUnion--dino_boost--guarded_trim` | 63.54 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--dino_global--guarded_component_rejection` | 63.59 | 99.58 | 1.00 | 6.33 | 1.00 | shippedUnion | 1 | 2 | fail |
| `productionConservative` | 63.54 | 99.62 | 1.00 | 6.00 | 1.00 | — | — | — | — |
| `shippedUnion--compact_keep--guarded_trim` | 63.54 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--compact_keep--guarded_component_rejection` | 63.54 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion--dino_boost--guarded_component_rejection` | 63.54 | 99.62 | 1.00 | 6.00 | 1.00 | shippedUnion | 1 | 1 | fail |
| `shippedUnion` | 61.42 | 99.62 | 0.00 | 6.00 | 0.00 | — | — | — | — |
| `shippedUnion--compact_boost--union` | 60.95 | 99.77 | 0.00 | 4.33 | 0.00 | shippedUnion | 0 | 0 | fail |
| `shippedUnion--dino_global--union` | 61.84 | 99.82 | 0.00 | 3.33 | 0.00 | shippedUnion | 0 | 0 | fail |
| `shippedUnion--compact_keep--union` | 60.58 | 99.79 | 0.00 | 4.33 | 0.00 | shippedUnion | 0 | 0 | fail |
| `shippedUnion--best_f1_pair--three_union` | 60.54 | 99.83 | 0.00 | 3.33 | 0.00 | shippedUnion | 0 | 0 | fail |
| `shippedUnion--dino_boost--union` | 60.57 | 99.85 | 0.00 | 3.33 | 0.00 | shippedUnion | 0 | 0 | fail |
| `shippedUnion--dino_keep--union` | 60.32 | 99.84 | 0.00 | 3.00 | 0.00 | shippedUnion | 0 | 0 | fail |
| `shippedUnion--recovery_pair--three_union` | 59.49 | 99.88 | 0.00 | 3.33 | 0.00 | shippedUnion | 0 | 0 | fail |
| `refitV2` | 51.37 | 94.99 | 11.00 | 36.00 | 7.00 | — | — | — | — |
| `refitUnion--dino_boost--guarded_trim` | 50.88 | 98.37 | 3.33 | 21.00 | 3.33 | refitUnion | 4 | 4 | fail |
| `refitUnion--dino_global--guarded_trim` | 50.91 | 98.37 | 3.67 | 21.00 | 3.67 | refitUnion | 4 | 4 | fail |
| `refitUnion--dino_keep--guarded_trim` | 51.04 | 98.32 | 4.00 | 21.33 | 4.00 | refitUnion | 5 | 6 | fail |
| `refitUnion--dino_boost--guarded_component_rejection` | 50.88 | 98.37 | 3.33 | 21.00 | 3.33 | refitUnion | 4 | 4 | fail |
| `refitUnion--dino_global--guarded_component_rejection` | 50.91 | 98.37 | 3.67 | 21.00 | 3.67 | refitUnion | 4 | 4 | fail |
| `refitUnion--dino_keep--guarded_component_rejection` | 50.95 | 98.32 | 4.00 | 21.33 | 4.00 | refitUnion | 5 | 6 | fail |
| `refitUnion--compact_keep--guarded_component_rejection` | 51.04 | 97.92 | 7.67 | 21.00 | 6.33 | refitUnion | 8 | 8 | fail |
| `refitUnion--compact_boost--guarded_component_rejection` | 50.96 | 97.84 | 8.67 | 21.00 | 7.00 | refitUnion | 8 | 8 | fail |
| `refitUnion--compact_keep--guarded_trim` | 51.09 | 97.74 | 7.67 | 21.00 | 6.33 | refitUnion | 8 | 9 | fail |
| `refitUnion--compact_boost--guarded_trim` | 51.03 | 97.62 | 8.67 | 21.33 | 7.00 | refitUnion | 8 | 9 | fail |
| `refitUnion--dino_boost--union` | 52.77 | 99.89 | 0.33 | 4.00 | 0.33 | refitUnion | 0 | 0 | fail |
| `refitUnion--dino_global--union` | 53.48 | 99.81 | 1.00 | 5.67 | 1.00 | refitUnion | 0 | 0 | fail |
| `refitUnion--dino_keep--union` | 52.75 | 99.82 | 0.67 | 6.00 | 0.67 | refitUnion | 0 | 0 | fail |
| `refitUnion--compact_boost--union` | 51.33 | 99.52 | 1.00 | 9.33 | 1.00 | refitUnion | 0 | 0 | fail |
| `refitUnion--best_f1_pair--three_union` | 53.09 | 99.85 | 1.00 | 4.67 | 1.00 | refitUnion | 0 | 0 | fail |
| `refitUnion--compact_keep--union` | 50.75 | 99.61 | 1.00 | 9.00 | 1.00 | refitUnion | 0 | 0 | fail |
| `refitUnion--recovery_pair--three_union` | 52.13 | 99.90 | 0.33 | 3.33 | 0.33 | refitUnion | 0 | 0 | fail |
| `refitUnion` | 48.85 | 98.37 | 1.00 | 21.00 | 1.00 | — | — | — | — |
| `refitPrevious` | 43.20 | 87.92 | 24.00 | 51.00 | 8.00 | — | — | — | — |

The screen requires at least +2 percentage points mean F1, recall/long-recall losses at most 0.5 points, event-F1 loss at most 1 point, and zero new complete or worsened rally coverage in every seed. It is stricter than F1 ranking and does not authorize deployment.

## Review queues: unchanged automatic output

Suppression review flags production export absent from the neural export. Bidirectional review also flags neural export missing from production. Automatic P/R/F1 and export durations remain those of the production anchor until a person edits the result. Queue playback includes ±2 s context around disputed spans and <3 s joining; context counts toward workload but not the hypothetical correction area. Clip counts are seed means. These are label-blind disagreement flags, not calibrated confidence probabilities.

| Policy | Review min | Clips | Disputed min | Incorrect export flagged min | FP capture % | Wanted export flagged min | Omitted human export flagged min | Missed core flagged s | Missed core capture % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--compact_boost--bidirectional_review` | 69.39 | 445.33 | 25.78 | 16.72 | 73.68 | 7.83 | 0.29 | 6.60 | 37.77 |
| `productionDefault--compact_boost--suppression_review` | 63.36 | 410.67 | 24.55 | 16.72 | 73.68 | 7.83 | 0.00 | 0.00 | 0.00 |
| `productionDefault--compact_keep--bidirectional_review` | 69.03 | 446.00 | 25.02 | 16.12 | 71.01 | 7.06 | 0.33 | 7.46 | 42.72 |
| `productionDefault--compact_keep--suppression_review` | 61.85 | 408.67 | 23.18 | 16.12 | 71.01 | 7.06 | 0.00 | 0.00 | 0.00 |
| `productionDefault--dino_boost--bidirectional_review` | 69.03 | 445.67 | 25.16 | 17.47 | 76.95 | 5.14 | 0.58 | 11.73 | 67.14 |
| `productionDefault--dino_boost--suppression_review` | 59.66 | 390.33 | 22.61 | 17.47 | 76.95 | 5.14 | 0.00 | 0.00 | 0.00 |
| `productionDefault--dino_global--bidirectional_review` | 68.35 | 452.33 | 24.39 | 17.57 | 77.39 | 4.93 | 0.51 | 9.97 | 57.05 |
| `productionDefault--dino_global--suppression_review` | 60.02 | 396.33 | 22.50 | 17.57 | 77.39 | 4.93 | 0.00 | 0.00 | 0.00 |
| `productionDefault--dino_keep--bidirectional_review` | 68.74 | 445.33 | 25.42 | 17.15 | 75.56 | 5.60 | 0.55 | 10.60 | 60.67 |
| `productionDefault--dino_keep--suppression_review` | 59.19 | 390.33 | 22.75 | 17.15 | 75.56 | 5.60 | 0.00 | 0.00 | 0.00 |
| `refitUnion--compact_boost--bidirectional_review` | 84.16 | 376.33 | 43.82 | 34.89 | 84.04 | 7.77 | 0.87 | 26.12 | 65.01 |
| `refitUnion--compact_boost--suppression_review` | 78.21 | 349.67 | 42.66 | 34.89 | 84.04 | 7.77 | 0.00 | 0.00 | 0.00 |
| `refitUnion--compact_keep--bidirectional_review` | 83.21 | 379.67 | 42.25 | 33.87 | 81.59 | 7.01 | 0.93 | 28.69 | 71.42 |
| `refitUnion--compact_keep--suppression_review` | 76.49 | 350.33 | 40.88 | 33.87 | 81.59 | 7.01 | 0.00 | 0.00 | 0.00 |
| `refitUnion--dino_boost--bidirectional_review` | 82.60 | 381.00 | 41.83 | 34.88 | 84.03 | 5.14 | 1.23 | 37.20 | 92.59 |
| `refitUnion--dino_boost--suppression_review` | 75.32 | 350.00 | 40.03 | 34.88 | 84.03 | 5.14 | 0.00 | 0.00 | 0.00 |
| `refitUnion--dino_global--bidirectional_review` | 82.98 | 385.67 | 42.09 | 35.49 | 85.48 | 4.94 | 1.17 | 34.54 | 85.97 |
| `refitUnion--dino_global--suppression_review` | 75.61 | 352.33 | 40.43 | 35.49 | 85.48 | 4.94 | 0.00 | 0.00 | 0.00 |
| `refitUnion--dino_keep--bidirectional_review` | 82.27 | 381.33 | 41.79 | 34.49 | 83.07 | 5.54 | 1.13 | 35.09 | 87.34 |
| `refitUnion--dino_keep--suppression_review` | 74.97 | 353.67 | 40.02 | 34.49 | 83.07 | 5.54 | 0.00 | 0.00 | 0.00 |
| `shippedUnion--compact_boost--bidirectional_review` | 76.67 | 421.33 | 32.24 | 23.27 | 78.70 | 8.16 | 0.19 | 3.48 | 40.52 |
| `shippedUnion--compact_boost--suppression_review` | 71.65 | 394.00 | 31.43 | 23.27 | 78.70 | 8.16 | 0.00 | 0.00 | 0.00 |
| `shippedUnion--compact_keep--bidirectional_review` | 76.03 | 422.67 | 31.14 | 22.53 | 76.20 | 7.36 | 0.20 | 3.80 | 44.27 |
| `shippedUnion--compact_keep--suppression_review` | 70.25 | 394.67 | 29.89 | 22.53 | 76.20 | 7.36 | 0.00 | 0.00 | 0.00 |
| `shippedUnion--dino_boost--bidirectional_review` | 75.78 | 424.33 | 31.06 | 23.91 | 80.87 | 5.30 | 0.30 | 4.80 | 55.92 |
| `shippedUnion--dino_boost--suppression_review` | 68.17 | 384.00 | 29.21 | 23.91 | 80.87 | 5.30 | 0.00 | 0.00 | 0.00 |
| `shippedUnion--dino_global--bidirectional_review` | 74.90 | 430.67 | 30.26 | 23.93 | 80.96 | 5.14 | 0.29 | 4.49 | 52.29 |
| `shippedUnion--dino_global--suppression_review` | 68.47 | 391.67 | 29.08 | 23.93 | 80.96 | 5.14 | 0.00 | 0.00 | 0.00 |
| `shippedUnion--dino_keep--bidirectional_review` | 75.61 | 425.00 | 31.44 | 23.64 | 79.95 | 5.78 | 0.29 | 4.58 | 53.33 |
| `shippedUnion--dino_keep--suppression_review` | 67.51 | 384.00 | 29.41 | 23.64 | 79.95 | 5.78 | 0.00 | 0.00 | 0.00 |

## Perfect disputed-region review: optimistic upper bound only

The following uses ground truth to correct only disputed export seconds. It is not an automatic model, not a measured human result, and is excluded from the automatic ranking. It cannot fix errors shared by both models. No padding or joining is applied again after the hypothetical edits.

| Configuration | P_pad % | R_core % | F1_padP_coreR % | Export min | Correctly removed min | Incorrectly removed min | Incorrect export min | Missed core s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `productionDefault--compact_boost--bidirectional_review` | 90.95 | 99.54 | 95.05 | 65.95 | 70.61 | 1.28 | 5.97 | 10.87 |
| `productionDefault--compact_boost--suppression_review` | 90.91 | 99.27 | 94.90 | 65.66 | 70.61 | 1.57 | 5.97 | 17.47 |
| `productionDefault--compact_keep--bidirectional_review` | 90.13 | 99.58 | 94.62 | 66.60 | 70.00 | 1.24 | 6.58 | 10.01 |
| `productionDefault--compact_keep--suppression_review` | 90.08 | 99.27 | 94.45 | 66.26 | 70.00 | 1.57 | 6.58 | 17.47 |
| `productionDefault--dino_boost--bidirectional_review` | 92.02 | 99.76 | 95.73 | 65.50 | 71.35 | 0.98 | 5.23 | 5.74 |
| `productionDefault--dino_boost--suppression_review` | 91.95 | 99.27 | 95.47 | 64.92 | 71.35 | 1.57 | 5.23 | 17.47 |
| `productionDefault--dino_global--bidirectional_review` | 92.16 | 99.69 | 95.77 | 65.33 | 71.45 | 1.05 | 5.13 | 7.50 |
| `productionDefault--dino_global--suppression_review` | 92.10 | 99.27 | 95.54 | 64.82 | 71.45 | 1.57 | 5.13 | 17.47 |
| `productionDefault--dino_keep--bidirectional_review` | 91.62 | 99.71 | 95.48 | 65.78 | 71.03 | 1.02 | 5.55 | 6.87 |
| `productionDefault--dino_keep--suppression_review` | 91.54 | 99.27 | 95.24 | 65.23 | 71.03 | 1.57 | 5.55 | 17.47 |
| `refitUnion--compact_boost--bidirectional_review` | 90.05 | 99.41 | 94.50 | 66.54 | 69.96 | 1.34 | 6.63 | 14.06 |
| `refitUnion--compact_boost--suppression_review` | 89.92 | 98.32 | 93.93 | 65.67 | 69.96 | 2.21 | 6.63 | 40.17 |
| `refitUnion--compact_keep--bidirectional_review` | 88.72 | 99.52 | 93.80 | 67.61 | 68.94 | 1.28 | 7.64 | 11.48 |
| `refitUnion--compact_keep--suppression_review` | 88.56 | 98.32 | 93.18 | 66.69 | 68.94 | 2.21 | 7.64 | 40.17 |
| `refitUnion--dino_boost--bidirectional_review` | 90.11 | 99.88 | 94.74 | 66.90 | 69.95 | 0.98 | 6.63 | 2.98 |
| `refitUnion--dino_boost--suppression_review` | 89.92 | 98.32 | 93.93 | 65.67 | 69.95 | 2.21 | 6.63 | 40.17 |
| `refitUnion--dino_global--bidirectional_review` | 90.93 | 99.76 | 95.14 | 66.24 | 70.55 | 1.04 | 6.03 | 5.64 |
| `refitUnion--dino_global--suppression_review` | 90.77 | 98.32 | 94.38 | 65.07 | 70.55 | 2.21 | 6.03 | 40.17 |
| `refitUnion--dino_keep--bidirectional_review` | 89.65 | 99.79 | 94.42 | 67.20 | 69.55 | 1.08 | 7.03 | 5.09 |
| `refitUnion--dino_keep--suppression_review` | 89.48 | 98.32 | 93.66 | 66.07 | 69.55 | 2.21 | 7.03 | 40.17 |
| `shippedUnion--compact_boost--bidirectional_review` | 90.55 | 99.79 | 94.94 | 66.61 | 70.29 | 0.94 | 6.30 | 5.11 |
| `shippedUnion--compact_boost--suppression_review` | 90.53 | 99.64 | 94.86 | 66.42 | 70.29 | 1.13 | 6.30 | 8.58 |
| `shippedUnion--compact_keep--bidirectional_review` | 89.57 | 99.80 | 94.40 | 67.36 | 69.55 | 0.93 | 7.04 | 4.78 |
| `shippedUnion--compact_keep--suppression_review` | 89.53 | 99.64 | 94.31 | 67.16 | 69.55 | 1.13 | 7.04 | 8.58 |
| `shippedUnion--dino_boost--bidirectional_review` | 91.45 | 99.84 | 95.46 | 66.08 | 70.92 | 0.83 | 5.66 | 3.78 |
| `shippedUnion--dino_boost--suppression_review` | 91.41 | 99.64 | 95.34 | 65.78 | 70.92 | 1.13 | 5.66 | 8.58 |
| `shippedUnion--dino_global--bidirectional_review` | 91.49 | 99.83 | 95.48 | 66.04 | 70.95 | 0.84 | 5.63 | 4.10 |
| `shippedUnion--dino_global--suppression_review` | 91.45 | 99.64 | 95.37 | 65.75 | 70.95 | 1.13 | 5.63 | 8.58 |
| `shippedUnion--dino_keep--bidirectional_review` | 91.12 | 99.83 | 95.26 | 66.34 | 70.65 | 0.84 | 5.93 | 4.01 |
| `shippedUnion--dino_keep--suppression_review` | 91.08 | 99.64 | 95.15 | 66.05 | 70.65 | 1.13 | 5.93 | 8.58 |

## Exact app-export fidelity (separate from primary ranking)

| Production variant | Pad s | P_pad % | R_core % | F1_padP_coreR % | Export s | Human export s | Export minus human s | App-only s | Canonical-only s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `shippedUnion` | 0 | 61.28 | 96.26 | 74.89 | 3749.50 | 2387.15 | 1362.35 | 0.09 | 0.06 |
| `shippedUnion` | 1 | 64.21 | 98.72 | 77.81 | 4593.88 | 3031.15 | 1562.73 | 0.08 | 0.05 |
| `shippedUnion` | 2 | 67.04 | 99.64 | 80.15 | 5381.16 | 3675.15 | 1706.00 | 0.08 | 0.05 |
| `shippedUnion` | 3 | 69.72 | 99.85 | 82.11 | 6121.48 | 4323.26 | 1798.22 | 0.06 | 0.04 |
| `productionDefault` | 0 | 65.50 | 95.75 | 77.79 | 3489.64 | 2387.15 | 1102.49 | 0.08 | 0.05 |
| `productionDefault` | 1 | 69.40 | 98.30 | 81.36 | 4223.54 | 3031.15 | 1192.39 | 0.08 | 0.05 |
| `productionDefault` | 2 | 72.45 | 99.27 | 83.76 | 4943.06 | 3675.15 | 1267.91 | 0.08 | 0.04 |
| `productionDefault` | 3 | 75.02 | 99.49 | 85.54 | 5637.23 | 4323.26 | 1313.96 | 0.07 | 0.04 |
| `productionBalanced` | 0 | 63.10 | 96.20 | 76.21 | 3639.34 | 2387.15 | 1252.19 | 0.09 | 0.05 |
| `productionBalanced` | 1 | 67.10 | 98.66 | 79.87 | 4391.01 | 3031.15 | 1359.86 | 0.08 | 0.05 |
| `productionBalanced` | 2 | 70.34 | 99.58 | 82.44 | 5118.74 | 3675.15 | 1443.59 | 0.08 | 0.04 |
| `productionBalanced` | 3 | 73.20 | 99.76 | 84.44 | 5808.91 | 4323.26 | 1485.65 | 0.07 | 0.03 |
| `productionConservative` | 0 | 62.43 | 96.20 | 75.72 | 3678.29 | 2387.15 | 1291.14 | 0.09 | 0.05 |
| `productionConservative` | 1 | 65.88 | 98.66 | 79.01 | 4472.07 | 3031.15 | 1440.92 | 0.08 | 0.05 |
| `productionConservative` | 2 | 69.23 | 99.58 | 81.68 | 5200.57 | 3675.15 | 1525.41 | 0.07 | 0.04 |
| `productionConservative` | 3 | 72.21 | 99.76 | 83.78 | 5888.78 | 4323.26 | 1565.52 | 0.06 | 0.03 |

## Required automatic padding sensitivity

The remaining padding cases are sensitivities; no candidate chooses its own best padding. Duration difference is model export minus human export. JSON retains every other duration field, per-seed statistics, source-group results, paired loss identities, and all review/oracle padding cases.

| Configuration | Pad s | P_pad % | R_core % | F1_padP_coreR % | Model export s | Human export s | Difference s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `shippedPrevious` | 0 | 67.28 | 92.54 | 77.92 | 3283.40 | 2387.15 | 896.25 |
| `shippedPrevious` | 1 | 69.56 | 96.92 | 80.99 | 4120.21 | 3031.15 | 1089.06 |
| `shippedPrevious` | 2 | 71.98 | 98.51 | 83.18 | 4899.10 | 3675.15 | 1223.95 |
| `shippedPrevious` | 3 | 74.60 | 98.99 | 85.08 | 5601.91 | 4323.26 | 1278.65 |
| `shippedV2` | 0 | 68.75 | 90.81 | 78.26 | 3153.06 | 2387.15 | 765.91 |
| `shippedV2` | 1 | 71.74 | 96.02 | 82.12 | 3962.63 | 3031.15 | 931.48 |
| `shippedV2` | 2 | 73.99 | 97.88 | 84.28 | 4732.34 | 3675.15 | 1057.19 |
| `shippedV2` | 3 | 75.74 | 98.94 | 85.80 | 5512.10 | 4323.26 | 1188.83 |
| `shippedUnion` | 0 | 61.28 | 96.26 | 74.89 | 3749.47 | 2387.15 | 1362.32 |
| `shippedUnion` | 1 | 64.21 | 98.72 | 77.81 | 4593.85 | 3031.15 | 1562.70 |
| `shippedUnion` | 2 | 67.04 | 99.64 | 80.15 | 5381.12 | 3675.15 | 1705.97 |
| `shippedUnion` | 3 | 69.72 | 99.85 | 82.11 | 6121.46 | 4323.26 | 1798.19 |
| `refitPrevious` | 0 | 51.97 | 76.79 | 61.99 | 3527.33 | 2387.15 | 1140.18 |
| `refitPrevious` | 1 | 55.88 | 84.26 | 67.20 | 4425.15 | 3031.15 | 1394.00 |
| `refitPrevious` | 2 | 59.99 | 87.90 | 71.31 | 5184.60 | 3675.15 | 1509.45 |
| `refitPrevious` | 3 | 64.65 | 90.36 | 75.38 | 5811.24 | 4323.26 | 1487.98 |
| `refitV2` | 0 | 61.20 | 85.65 | 71.39 | 3340.75 | 2387.15 | 953.60 |
| `refitV2` | 1 | 63.93 | 92.16 | 75.50 | 4236.71 | 3031.15 | 1205.56 |
| `refitV2` | 2 | 66.88 | 94.62 | 78.37 | 5021.32 | 3675.15 | 1346.17 |
| `refitV2` | 3 | 69.40 | 96.42 | 80.71 | 5796.70 | 4323.26 | 1473.44 |
| `refitUnion` | 0 | 51.33 | 92.76 | 66.09 | 4314.21 | 2387.15 | 1927.06 |
| `refitUnion` | 1 | 54.77 | 96.81 | 69.96 | 5261.85 | 3031.15 | 2230.69 |
| `refitUnion` | 2 | 58.71 | 98.32 | 73.52 | 6033.48 | 3675.15 | 2358.33 |
| `refitUnion` | 3 | 63.16 | 99.16 | 77.17 | 6656.78 | 4323.26 | 2333.52 |
| `productionDefault` | 0 | 65.50 | 95.75 | 77.79 | 3489.61 | 2387.15 | 1102.46 |
| `productionDefault` | 1 | 69.40 | 98.30 | 81.36 | 4223.51 | 3031.15 | 1192.36 |
| `productionDefault` | 2 | 72.45 | 99.27 | 83.76 | 4943.03 | 3675.15 | 1267.87 |
| `productionDefault` | 3 | 75.02 | 99.49 | 85.54 | 5637.20 | 4323.26 | 1313.93 |
| `productionBalanced` | 0 | 63.10 | 96.20 | 76.21 | 3639.31 | 2387.15 | 1252.16 |
| `productionBalanced` | 1 | 67.10 | 98.66 | 79.87 | 4390.98 | 3031.15 | 1359.82 |
| `productionBalanced` | 2 | 70.34 | 99.58 | 82.44 | 5118.71 | 3675.15 | 1443.56 |
| `productionBalanced` | 3 | 73.21 | 99.76 | 84.44 | 5808.88 | 4323.26 | 1485.62 |
| `productionConservative` | 0 | 62.43 | 96.20 | 75.72 | 3678.26 | 2387.15 | 1291.11 |
| `productionConservative` | 1 | 65.88 | 98.66 | 79.01 | 4472.03 | 3031.15 | 1440.88 |
| `productionConservative` | 2 | 69.23 | 99.58 | 81.68 | 5200.53 | 3675.15 | 1525.38 |
| `productionConservative` | 3 | 72.21 | 99.76 | 83.78 | 5888.76 | 4323.26 | 1565.49 |
| `compact_boost` | 0 | 84.24 | 82.00 | 83.10 | 2324.05 | 2387.15 | -63.10 |
| `compact_boost` | 1 | 86.47 | 89.60 | 88.00 | 2943.53 | 3031.15 | -87.62 |
| `compact_boost` | 2 | 88.30 | 92.41 | 90.31 | 3543.47 | 3675.15 | -131.68 |
| `compact_boost` | 3 | 89.71 | 93.88 | 91.75 | 4145.11 | 4323.26 | -178.15 |
| `compact_keep` | 0 | 82.64 | 82.81 | 82.70 | 2393.67 | 2387.15 | 6.52 |
| `compact_keep` | 1 | 84.95 | 90.80 | 87.76 | 3041.36 | 3031.15 | 10.21 |
| `compact_keep` | 2 | 86.79 | 93.42 | 89.97 | 3662.73 | 3675.15 | -12.43 |
| `compact_keep` | 3 | 88.34 | 94.73 | 91.41 | 4278.61 | 4323.26 | -44.65 |
| `dino_global` | 0 | 85.81 | 87.32 | 86.53 | 2431.40 | 2387.15 | 44.25 |
| `dino_global` | 1 | 87.96 | 94.96 | 91.31 | 3076.71 | 3031.15 | 45.56 |
| `dino_global` | 2 | 89.50 | 96.96 | 93.07 | 3707.08 | 3675.15 | 31.93 |
| `dino_global` | 3 | 90.73 | 97.64 | 94.05 | 4338.01 | 4323.26 | 14.75 |
| `dino_boost` | 0 | 84.34 | 86.22 | 85.25 | 2441.89 | 2387.15 | 54.74 |
| `dino_boost` | 1 | 86.69 | 94.63 | 90.48 | 3101.64 | 3031.15 | 70.49 |
| `dino_boost` | 2 | 88.47 | 96.95 | 92.51 | 3739.83 | 3675.15 | 64.68 |
| `dino_boost` | 3 | 90.08 | 97.84 | 93.79 | 4364.46 | 4323.26 | 41.19 |
| `dino_keep` | 0 | 82.85 | 84.91 | 83.70 | 2458.73 | 2387.15 | 71.58 |
| `dino_keep` | 1 | 85.83 | 93.75 | 89.54 | 3108.59 | 3031.15 | 77.44 |
| `dino_keep` | 2 | 87.91 | 96.47 | 91.94 | 3737.96 | 3675.15 | 62.80 |
| `dino_keep` | 3 | 89.61 | 97.51 | 93.36 | 4361.15 | 4323.26 | 37.89 |
| `compact_baseline` | 0 | 83.05 | 82.16 | 82.57 | 2363.35 | 2387.15 | -23.80 |
| `compact_baseline` | 1 | 85.42 | 89.95 | 87.61 | 2988.46 | 3031.15 | -42.69 |
| `compact_baseline` | 2 | 87.34 | 92.49 | 89.82 | 3587.16 | 3675.15 | -87.99 |
| `compact_baseline` | 3 | 88.90 | 93.76 | 91.25 | 4180.14 | 4323.26 | -143.12 |
| `dino_baseline` | 0 | 82.91 | 84.16 | 83.35 | 2436.06 | 2387.15 | 48.91 |
| `dino_baseline` | 1 | 85.86 | 93.14 | 89.26 | 3075.00 | 3031.15 | 43.85 |
| `dino_baseline` | 2 | 87.92 | 95.80 | 91.63 | 3691.87 | 3675.15 | 16.72 |
| `dino_baseline` | 3 | 89.63 | 96.88 | 93.07 | 4303.23 | 4323.26 | -20.03 |
| `best_f1_pair--nn_union` | 0 | 80.64 | 91.45 | 85.70 | 2707.56 | 2387.15 | 320.41 |
| `best_f1_pair--nn_union` | 1 | 83.20 | 96.85 | 89.50 | 3377.67 | 3031.15 | 346.52 |
| `best_f1_pair--nn_union` | 2 | 85.34 | 98.18 | 91.31 | 4023.26 | 3675.15 | 348.11 |
| `best_f1_pair--nn_union` | 3 | 87.01 | 98.71 | 92.49 | 4673.69 | 4323.26 | 350.43 |
| `best_f1_pair--nn_intersection` | 0 | 90.91 | 77.89 | 83.89 | 2045.39 | 2387.15 | -341.76 |
| `best_f1_pair--nn_intersection` | 1 | 92.49 | 87.61 | 89.98 | 2634.54 | 3031.15 | -396.61 |
| `best_f1_pair--nn_intersection` | 2 | 93.51 | 91.07 | 92.27 | 3217.25 | 3675.15 | -457.90 |
| `best_f1_pair--nn_intersection` | 3 | 94.31 | 92.71 | 93.50 | 3795.38 | 4323.26 | -527.88 |
| `recovery_pair--nn_union` | 0 | 78.48 | 91.39 | 84.44 | 2780.03 | 2387.15 | 392.88 |
| `recovery_pair--nn_union` | 1 | 81.19 | 97.00 | 88.39 | 3466.70 | 3031.15 | 435.55 |
| `recovery_pair--nn_union` | 2 | 83.45 | 98.42 | 90.32 | 4128.84 | 3675.15 | 453.69 |
| `recovery_pair--nn_union` | 3 | 85.48 | 99.03 | 91.76 | 4776.86 | 4323.26 | 453.60 |
| `recovery_pair--nn_intersection` | 0 | 90.16 | 77.65 | 83.43 | 2056.21 | 2387.15 | -330.94 |
| `recovery_pair--nn_intersection` | 1 | 91.96 | 88.29 | 90.09 | 2665.41 | 3031.15 | -365.74 |
| `recovery_pair--nn_intersection` | 2 | 93.04 | 91.77 | 92.40 | 3258.62 | 3675.15 | -416.53 |
| `recovery_pair--nn_intersection` | 3 | 93.95 | 93.37 | 93.66 | 3845.31 | 4323.26 | -477.95 |
| `productionDefault--compact_boost--union` | 0 | 64.68 | 97.09 | 77.64 | 3583.18 | 2387.15 | 1196.03 |
| `productionDefault--compact_boost--union` | 1 | 68.57 | 99.02 | 81.03 | 4311.67 | 3031.15 | 1280.52 |
| `productionDefault--compact_boost--union` | 2 | 71.70 | 99.54 | 83.36 | 5018.36 | 3675.15 | 1343.21 |
| `productionDefault--compact_boost--union` | 3 | 74.29 | 99.66 | 85.13 | 5716.21 | 4323.26 | 1392.95 |
| `productionDefault--compact_boost--intersection` | 0 | 86.24 | 80.64 | 83.34 | 2232.71 | 2387.15 | -154.45 |
| `productionDefault--compact_boost--intersection` | 1 | 88.24 | 88.88 | 88.55 | 2854.52 | 3031.15 | -176.63 |
| `productionDefault--compact_boost--intersection` | 2 | 89.82 | 92.09 | 90.94 | 3461.86 | 3675.15 | -213.29 |
| `productionDefault--compact_boost--intersection` | 3 | 91.17 | 93.71 | 92.42 | 4058.02 | 4323.26 | -265.24 |
| `productionDefault--compact_boost--tolerant_intersection` | 0 | 74.66 | 89.76 | 81.52 | 2870.46 | 2387.15 | 483.31 |
| `productionDefault--compact_boost--tolerant_intersection` | 1 | 78.51 | 92.81 | 85.06 | 3481.18 | 3031.15 | 450.02 |
| `productionDefault--compact_boost--tolerant_intersection` | 2 | 81.32 | 94.32 | 87.34 | 4087.94 | 3675.15 | 412.79 |
| `productionDefault--compact_boost--tolerant_intersection` | 3 | 83.24 | 94.95 | 88.71 | 4697.68 | 4323.26 | 374.42 |
| `productionDefault--compact_boost--guarded_trim` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--compact_boost--guarded_trim` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--compact_boost--guarded_trim` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--compact_boost--guarded_trim` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--compact_boost--guarded_component_rejection` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--compact_boost--guarded_component_rejection` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--compact_boost--guarded_component_rejection` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--compact_boost--guarded_component_rejection` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--compact_keep--union` | 0 | 64.26 | 97.11 | 77.34 | 3607.67 | 2387.15 | 1220.52 |
| `productionDefault--compact_keep--union` | 1 | 68.04 | 99.07 | 80.67 | 4348.82 | 3031.15 | 1317.67 |
| `productionDefault--compact_keep--union` | 2 | 71.21 | 99.58 | 83.04 | 5057.18 | 3675.15 | 1382.02 |
| `productionDefault--compact_keep--union` | 3 | 73.85 | 99.69 | 84.85 | 5755.48 | 4323.26 | 1432.22 |
| `productionDefault--compact_keep--intersection` | 0 | 85.46 | 81.42 | 83.37 | 2275.09 | 2387.15 | -112.06 |
| `productionDefault--compact_keep--intersection` | 1 | 87.58 | 90.03 | 88.78 | 2916.45 | 3031.15 | -114.70 |
| `productionDefault--compact_keep--intersection` | 2 | 89.16 | 93.07 | 91.06 | 3539.26 | 3675.15 | -135.89 |
| `productionDefault--compact_keep--intersection` | 3 | 90.51 | 94.52 | 92.47 | 4148.82 | 4323.26 | -174.45 |
| `productionDefault--compact_keep--tolerant_intersection` | 0 | 74.01 | 90.59 | 81.46 | 2922.33 | 2387.15 | 535.18 |
| `productionDefault--compact_keep--tolerant_intersection` | 1 | 77.80 | 93.54 | 84.94 | 3548.59 | 3031.15 | 517.44 |
| `productionDefault--compact_keep--tolerant_intersection` | 2 | 80.63 | 94.91 | 87.18 | 4166.95 | 3675.15 | 491.79 |
| `productionDefault--compact_keep--tolerant_intersection` | 3 | 82.43 | 95.42 | 88.45 | 4794.61 | 4323.26 | 471.34 |
| `productionDefault--compact_keep--guarded_trim` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--compact_keep--guarded_trim` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--compact_keep--guarded_trim` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--compact_keep--guarded_trim` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--compact_keep--guarded_component_rejection` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--compact_keep--guarded_component_rejection` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--compact_keep--guarded_component_rejection` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--compact_keep--guarded_component_rejection` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--dino_global--union` | 0 | 64.39 | 97.51 | 77.56 | 3615.43 | 2387.15 | 1228.28 |
| `productionDefault--dino_global--union` | 1 | 68.24 | 99.27 | 80.88 | 4350.83 | 3031.15 | 1319.68 |
| `productionDefault--dino_global--union` | 2 | 71.33 | 99.69 | 83.16 | 5064.28 | 3675.15 | 1389.13 |
| `productionDefault--dino_global--union` | 3 | 74.09 | 99.76 | 85.03 | 5753.31 | 4323.26 | 1430.05 |
| `productionDefault--dino_global--intersection` | 0 | 88.51 | 85.54 | 86.99 | 2307.58 | 2387.15 | -79.57 |
| `productionDefault--dino_global--intersection` | 1 | 90.44 | 94.00 | 92.18 | 2948.02 | 3031.15 | -83.13 |
| `productionDefault--dino_global--intersection` | 2 | 91.67 | 96.46 | 94.00 | 3579.54 | 3675.15 | -95.61 |
| `productionDefault--dino_global--intersection` | 3 | 92.59 | 97.31 | 94.89 | 4207.01 | 4323.26 | -116.25 |
| `productionDefault--dino_global--tolerant_intersection` | 0 | 76.25 | 93.70 | 84.07 | 2934.13 | 2387.15 | 546.97 |
| `productionDefault--dino_global--tolerant_intersection` | 1 | 79.94 | 96.40 | 87.40 | 3566.47 | 3031.15 | 535.32 |
| `productionDefault--dino_global--tolerant_intersection` | 2 | 82.61 | 97.46 | 89.42 | 4194.80 | 3675.15 | 519.64 |
| `productionDefault--dino_global--tolerant_intersection` | 3 | 84.21 | 97.83 | 90.51 | 4844.28 | 4323.26 | 521.02 |
| `productionDefault--dino_global--guarded_trim` | 0 | 65.58 | 95.75 | 77.84 | 3485.48 | 2387.15 | 1098.33 |
| `productionDefault--dino_global--guarded_trim` | 1 | 69.51 | 98.30 | 81.44 | 4216.71 | 3031.15 | 1185.56 |
| `productionDefault--dino_global--guarded_trim` | 2 | 72.57 | 99.23 | 83.83 | 4932.69 | 3675.15 | 1257.54 |
| `productionDefault--dino_global--guarded_trim` | 3 | 75.19 | 99.46 | 85.64 | 5622.50 | 4323.26 | 1299.24 |
| `productionDefault--dino_global--guarded_component_rejection` | 0 | 65.56 | 95.75 | 77.83 | 3486.28 | 2387.15 | 1099.12 |
| `productionDefault--dino_global--guarded_component_rejection` | 1 | 69.50 | 98.30 | 81.43 | 4217.51 | 3031.15 | 1186.36 |
| `productionDefault--dino_global--guarded_component_rejection` | 2 | 72.55 | 99.23 | 83.82 | 4933.48 | 3675.15 | 1258.33 |
| `productionDefault--dino_global--guarded_component_rejection` | 3 | 75.18 | 99.46 | 85.63 | 5623.30 | 4323.26 | 1300.03 |
| `productionDefault--dino_boost--union` | 0 | 63.97 | 97.54 | 77.27 | 3640.11 | 2387.15 | 1252.95 |
| `productionDefault--dino_boost--union` | 1 | 67.78 | 99.34 | 80.58 | 4383.28 | 3031.15 | 1352.13 |
| `productionDefault--dino_boost--union` | 2 | 70.90 | 99.76 | 82.89 | 5100.17 | 3675.15 | 1425.02 |
| `productionDefault--dino_boost--union` | 3 | 73.65 | 99.85 | 84.77 | 5794.16 | 4323.26 | 1470.90 |
| `productionDefault--dino_boost--intersection` | 0 | 87.86 | 84.38 | 86.08 | 2293.15 | 2387.15 | -94.00 |
| `productionDefault--dino_boost--intersection` | 1 | 90.05 | 93.58 | 91.77 | 2938.66 | 3031.15 | -92.49 |
| `productionDefault--dino_boost--intersection` | 2 | 91.47 | 96.43 | 93.88 | 3575.12 | 3675.15 | -100.03 |
| `productionDefault--dino_boost--intersection` | 3 | 92.67 | 97.43 | 94.99 | 4195.58 | 4323.26 | -127.69 |
| `productionDefault--dino_boost--tolerant_intersection` | 0 | 76.60 | 93.58 | 84.24 | 2916.61 | 2387.15 | 529.46 |
| `productionDefault--dino_boost--tolerant_intersection` | 1 | 80.44 | 96.44 | 87.72 | 3547.78 | 3031.15 | 516.63 |
| `productionDefault--dino_boost--tolerant_intersection` | 2 | 83.09 | 97.58 | 89.75 | 4183.31 | 3675.15 | 508.15 |
| `productionDefault--dino_boost--tolerant_intersection` | 3 | 84.77 | 97.96 | 90.89 | 4830.37 | 4323.26 | 507.10 |
| `productionDefault--dino_boost--guarded_trim` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--dino_boost--guarded_trim` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--dino_boost--guarded_trim` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--dino_boost--guarded_trim` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--dino_boost--guarded_component_rejection` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--dino_boost--guarded_component_rejection` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--dino_boost--guarded_component_rejection` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--dino_boost--guarded_component_rejection` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--dino_keep--union` | 0 | 63.60 | 97.55 | 76.99 | 3663.13 | 2387.15 | 1275.98 |
| `productionDefault--dino_keep--union` | 1 | 67.52 | 99.29 | 80.37 | 4400.48 | 3031.15 | 1369.32 |
| `productionDefault--dino_keep--union` | 2 | 70.74 | 99.71 | 82.76 | 5109.94 | 3675.15 | 1434.79 |
| `productionDefault--dino_keep--union` | 3 | 73.56 | 99.79 | 84.68 | 5798.36 | 4323.26 | 1475.09 |
| `productionDefault--dino_keep--intersection` | 0 | 86.98 | 83.07 | 84.90 | 2283.66 | 2387.15 | -103.50 |
| `productionDefault--dino_keep--intersection` | 1 | 89.41 | 92.66 | 90.98 | 2929.48 | 3031.15 | -101.68 |
| `productionDefault--dino_keep--intersection` | 2 | 91.03 | 95.95 | 93.41 | 3563.40 | 3675.15 | -111.75 |
| `productionDefault--dino_keep--intersection` | 3 | 92.34 | 97.16 | 94.67 | 4185.33 | 4323.26 | -137.93 |
| `productionDefault--dino_keep--tolerant_intersection` | 0 | 76.35 | 93.14 | 83.90 | 2914.08 | 2387.15 | 526.93 |
| `productionDefault--dino_keep--tolerant_intersection` | 1 | 80.26 | 96.18 | 87.49 | 3544.12 | 3031.15 | 512.97 |
| `productionDefault--dino_keep--tolerant_intersection` | 2 | 82.93 | 97.40 | 89.57 | 4178.50 | 3675.15 | 503.34 |
| `productionDefault--dino_keep--tolerant_intersection` | 3 | 84.65 | 97.79 | 90.74 | 4822.38 | 4323.26 | 499.12 |
| `productionDefault--dino_keep--guarded_trim` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--dino_keep--guarded_trim` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--dino_keep--guarded_trim` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--dino_keep--guarded_trim` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--dino_keep--guarded_component_rejection` | 0 | 65.55 | 95.75 | 77.82 | 3487.11 | 2387.15 | 1099.96 |
| `productionDefault--dino_keep--guarded_component_rejection` | 1 | 69.47 | 98.30 | 81.41 | 4219.01 | 3031.15 | 1187.86 |
| `productionDefault--dino_keep--guarded_component_rejection` | 2 | 72.54 | 99.27 | 83.83 | 4936.53 | 3675.15 | 1261.37 |
| `productionDefault--dino_keep--guarded_component_rejection` | 3 | 75.17 | 99.49 | 85.63 | 5626.34 | 4323.26 | 1303.08 |
| `productionDefault--best_f1_pair--three_union` | 0 | 63.72 | 97.63 | 77.11 | 3657.56 | 2387.15 | 1270.41 |
| `productionDefault--best_f1_pair--three_union` | 1 | 67.63 | 99.29 | 80.45 | 4394.13 | 3031.15 | 1362.97 |
| `productionDefault--best_f1_pair--three_union` | 2 | 70.78 | 99.70 | 82.78 | 5107.86 | 3675.15 | 1432.70 |
| `productionDefault--best_f1_pair--three_union` | 3 | 73.56 | 99.78 | 84.68 | 5799.07 | 4323.26 | 1475.80 |
| `productionDefault--best_f1_pair--majority` | 0 | 83.67 | 90.74 | 87.06 | 2589.12 | 2387.15 | 201.97 |
| `productionDefault--best_f1_pair--majority` | 1 | 85.97 | 96.56 | 90.96 | 3245.81 | 3031.15 | 214.66 |
| `productionDefault--best_f1_pair--majority` | 2 | 87.77 | 98.01 | 92.61 | 3887.61 | 3675.15 | 212.46 |
| `productionDefault--best_f1_pair--majority` | 3 | 89.25 | 98.57 | 93.68 | 4527.41 | 4323.26 | 204.15 |
| `productionDefault--recovery_pair--three_union` | 0 | 63.09 | 97.70 | 76.67 | 3696.73 | 2387.15 | 1309.58 |
| `productionDefault--recovery_pair--three_union` | 1 | 66.82 | 99.40 | 79.92 | 4452.37 | 3031.15 | 1421.22 |
| `productionDefault--recovery_pair--three_union` | 2 | 70.06 | 99.81 | 82.33 | 5168.92 | 3675.15 | 1493.77 |
| `productionDefault--recovery_pair--three_union` | 3 | 72.85 | 99.90 | 84.26 | 5866.36 | 4323.26 | 1543.09 |
| `productionDefault--recovery_pair--majority` | 0 | 82.09 | 90.60 | 86.13 | 2634.64 | 2387.15 | 247.49 |
| `productionDefault--recovery_pair--majority` | 1 | 84.78 | 96.60 | 90.30 | 3293.03 | 3031.15 | 261.88 |
| `productionDefault--recovery_pair--majority` | 2 | 86.79 | 98.11 | 92.10 | 3937.07 | 3675.15 | 261.92 |
| `productionDefault--recovery_pair--majority` | 3 | 88.42 | 98.73 | 93.29 | 4579.86 | 4323.26 | 256.60 |
| `shippedUnion--compact_boost--union` | 0 | 60.71 | 97.37 | 74.79 | 3828.30 | 2387.15 | 1441.15 |
| `shippedUnion--compact_boost--union` | 1 | 63.76 | 99.26 | 77.64 | 4657.49 | 3031.15 | 1626.33 |
| `shippedUnion--compact_boost--union` | 2 | 66.62 | 99.79 | 79.90 | 5431.27 | 3675.15 | 1756.12 |
| `shippedUnion--compact_boost--union` | 3 | 69.31 | 99.93 | 81.85 | 6174.66 | 4323.26 | 1851.39 |
| `shippedUnion--compact_boost--intersection` | 0 | 85.93 | 80.86 | 83.31 | 2247.04 | 2387.15 | -140.11 |
| `shippedUnion--compact_boost--intersection` | 1 | 87.86 | 89.05 | 88.45 | 2872.85 | 3031.15 | -158.30 |
| `shippedUnion--compact_boost--intersection` | 2 | 89.43 | 92.22 | 90.80 | 3483.31 | 3675.15 | -191.84 |
| `shippedUnion--compact_boost--intersection` | 3 | 90.79 | 93.80 | 92.27 | 4081.48 | 4323.26 | -241.79 |
| `shippedUnion--compact_boost--tolerant_intersection` | 0 | 74.20 | 90.01 | 81.34 | 2895.96 | 2387.15 | 508.80 |
| `shippedUnion--compact_boost--tolerant_intersection` | 1 | 77.97 | 92.98 | 84.81 | 3512.63 | 3031.15 | 481.48 |
| `shippedUnion--compact_boost--tolerant_intersection` | 2 | 80.81 | 94.45 | 87.10 | 4120.81 | 3675.15 | 445.66 |
| `shippedUnion--compact_boost--tolerant_intersection` | 3 | 82.78 | 95.04 | 88.49 | 4730.55 | 4323.26 | 407.29 |
| `shippedUnion--compact_boost--guarded_trim` | 0 | 62.46 | 96.20 | 75.74 | 3676.76 | 2387.15 | 1289.61 |
| `shippedUnion--compact_boost--guarded_trim` | 1 | 65.92 | 98.66 | 79.04 | 4469.21 | 3031.15 | 1438.05 |
| `shippedUnion--compact_boost--guarded_trim` | 2 | 69.29 | 99.58 | 81.72 | 5196.37 | 3675.15 | 1521.22 |
| `shippedUnion--compact_boost--guarded_trim` | 3 | 72.31 | 99.76 | 83.84 | 5880.90 | 4323.26 | 1557.64 |
| `shippedUnion--compact_boost--guarded_component_rejection` | 0 | 62.45 | 96.20 | 75.73 | 3677.26 | 2387.15 | 1290.11 |
| `shippedUnion--compact_boost--guarded_component_rejection` | 1 | 65.92 | 98.66 | 79.03 | 4469.71 | 3031.15 | 1438.55 |
| `shippedUnion--compact_boost--guarded_component_rejection` | 2 | 69.28 | 99.58 | 81.71 | 5196.87 | 3675.15 | 1521.72 |
| `shippedUnion--compact_boost--guarded_component_rejection` | 3 | 72.30 | 99.76 | 83.84 | 5881.40 | 4323.26 | 1558.14 |
| `shippedUnion--compact_keep--union` | 0 | 60.38 | 97.36 | 74.53 | 3849.27 | 2387.15 | 1462.12 |
| `shippedUnion--compact_keep--union` | 1 | 63.37 | 99.29 | 77.36 | 4687.16 | 3031.15 | 1656.01 |
| `shippedUnion--compact_keep--union` | 2 | 66.30 | 99.80 | 79.67 | 5459.03 | 3675.15 | 1783.88 |
| `shippedUnion--compact_keep--union` | 3 | 69.04 | 99.94 | 81.66 | 6199.94 | 4323.26 | 1876.68 |
| `shippedUnion--compact_keep--intersection` | 0 | 85.06 | 81.67 | 83.32 | 2292.98 | 2387.15 | -94.17 |
| `shippedUnion--compact_keep--intersection` | 1 | 87.10 | 90.23 | 88.62 | 2940.33 | 3031.15 | -90.82 |
| `shippedUnion--compact_keep--intersection` | 2 | 88.65 | 93.22 | 90.87 | 3568.27 | 3675.15 | -106.88 |
| `shippedUnion--compact_keep--intersection` | 3 | 89.97 | 94.64 | 92.24 | 4182.65 | 4323.26 | -140.61 |
| `shippedUnion--compact_keep--tolerant_intersection` | 0 | 73.36 | 90.87 | 81.17 | 2957.71 | 2387.15 | 570.55 |
| `shippedUnion--compact_keep--tolerant_intersection` | 1 | 77.03 | 93.73 | 84.56 | 3593.92 | 3031.15 | 562.77 |
| `shippedUnion--compact_keep--tolerant_intersection` | 2 | 79.88 | 95.06 | 86.81 | 4216.23 | 3675.15 | 541.08 |
| `shippedUnion--compact_keep--tolerant_intersection` | 3 | 81.69 | 95.53 | 88.07 | 4848.61 | 4323.26 | 525.34 |
| `shippedUnion--compact_keep--guarded_trim` | 0 | 62.43 | 96.20 | 75.72 | 3678.21 | 2387.15 | 1291.06 |
| `shippedUnion--compact_keep--guarded_trim` | 1 | 65.88 | 98.66 | 79.01 | 4471.99 | 3031.15 | 1440.84 |
| `shippedUnion--compact_keep--guarded_trim` | 2 | 69.22 | 99.58 | 81.67 | 5201.12 | 3675.15 | 1525.97 |
| `shippedUnion--compact_keep--guarded_trim` | 3 | 72.24 | 99.76 | 83.80 | 5886.31 | 4323.26 | 1563.05 |
| `shippedUnion--compact_keep--guarded_component_rejection` | 0 | 62.43 | 96.20 | 75.72 | 3678.34 | 2387.15 | 1291.19 |
| `shippedUnion--compact_keep--guarded_component_rejection` | 1 | 65.88 | 98.66 | 79.00 | 4472.12 | 3031.15 | 1440.97 |
| `shippedUnion--compact_keep--guarded_component_rejection` | 2 | 69.22 | 99.58 | 81.67 | 5201.25 | 3675.15 | 1526.10 |
| `shippedUnion--compact_keep--guarded_component_rejection` | 3 | 72.23 | 99.76 | 83.79 | 5887.36 | 4323.26 | 1564.09 |
| `shippedUnion--dino_global--union` | 0 | 60.44 | 97.70 | 74.68 | 3859.17 | 2387.15 | 1472.02 |
| `shippedUnion--dino_global--union` | 1 | 63.48 | 99.41 | 77.48 | 4690.95 | 3031.15 | 1659.80 |
| `shippedUnion--dino_global--union` | 2 | 66.39 | 99.83 | 79.74 | 5462.15 | 3675.15 | 1787.00 |
| `shippedUnion--dino_global--union` | 3 | 69.24 | 99.91 | 81.79 | 6187.34 | 4323.26 | 1864.08 |
| `shippedUnion--dino_global--intersection` | 0 | 88.16 | 85.83 | 86.97 | 2324.94 | 2387.15 | -62.21 |
| `shippedUnion--dino_global--intersection` | 1 | 89.92 | 94.27 | 92.04 | 2975.66 | 3031.15 | -55.50 |
| `shippedUnion--dino_global--intersection` | 2 | 91.19 | 96.69 | 93.86 | 3611.01 | 3675.15 | -64.14 |
| `shippedUnion--dino_global--intersection` | 3 | 92.15 | 97.50 | 94.74 | 4242.06 | 4323.26 | -81.20 |
| `shippedUnion--dino_global--tolerant_intersection` | 0 | 75.65 | 94.05 | 83.85 | 2968.65 | 2387.15 | 581.50 |
| `shippedUnion--dino_global--tolerant_intersection` | 1 | 79.12 | 96.67 | 87.01 | 3616.94 | 3031.15 | 585.79 |
| `shippedUnion--dino_global--tolerant_intersection` | 2 | 81.84 | 97.69 | 89.06 | 4250.19 | 3675.15 | 575.04 |
| `shippedUnion--dino_global--tolerant_intersection` | 3 | 83.45 | 98.02 | 90.14 | 4905.96 | 4323.26 | 582.70 |
| `shippedUnion--dino_global--guarded_trim` | 0 | 62.48 | 96.20 | 75.76 | 3675.30 | 2387.15 | 1288.15 |
| `shippedUnion--dino_global--guarded_trim` | 1 | 65.94 | 98.66 | 79.05 | 4467.74 | 3031.15 | 1436.59 |
| `shippedUnion--dino_global--guarded_trim` | 2 | 69.28 | 99.54 | 81.70 | 5194.66 | 3675.15 | 1519.51 |
| `shippedUnion--dino_global--guarded_trim` | 3 | 72.30 | 99.74 | 83.83 | 5879.19 | 4323.26 | 1555.93 |
| `shippedUnion--dino_global--guarded_component_rejection` | 0 | 62.45 | 96.20 | 75.73 | 3677.26 | 2387.15 | 1290.11 |
| `shippedUnion--dino_global--guarded_component_rejection` | 1 | 65.91 | 98.66 | 79.03 | 4469.71 | 3031.15 | 1438.55 |
| `shippedUnion--dino_global--guarded_component_rejection` | 2 | 69.25 | 99.54 | 81.68 | 5196.63 | 3675.15 | 1521.48 |
| `shippedUnion--dino_global--guarded_component_rejection` | 3 | 72.28 | 99.74 | 83.82 | 5881.16 | 4323.26 | 1557.90 |
| `shippedUnion--dino_boost--union` | 0 | 60.13 | 97.70 | 74.44 | 3879.06 | 2387.15 | 1491.91 |
| `shippedUnion--dino_boost--union` | 1 | 63.04 | 99.43 | 77.16 | 4722.74 | 3031.15 | 1691.59 |
| `shippedUnion--dino_boost--union` | 2 | 65.96 | 99.86 | 79.45 | 5499.15 | 3675.15 | 1824.00 |
| `shippedUnion--dino_boost--union` | 3 | 68.79 | 99.94 | 81.49 | 6229.16 | 4323.26 | 1905.90 |
| `shippedUnion--dino_boost--intersection` | 0 | 87.52 | 84.71 | 86.09 | 2311.24 | 2387.15 | -75.91 |
| `shippedUnion--dino_boost--intersection` | 1 | 89.58 | 93.91 | 91.69 | 2967.16 | 3031.15 | -63.99 |
| `shippedUnion--dino_boost--intersection` | 2 | 91.01 | 96.72 | 93.78 | 3609.61 | 3675.15 | -65.54 |
| `shippedUnion--dino_boost--intersection` | 3 | 92.22 | 97.67 | 94.86 | 4236.07 | 4323.26 | -87.19 |
| `shippedUnion--dino_boost--tolerant_intersection` | 0 | 75.97 | 93.99 | 84.02 | 2953.68 | 2387.15 | 566.52 |
| `shippedUnion--dino_boost--tolerant_intersection` | 1 | 79.71 | 96.77 | 87.41 | 3597.00 | 3031.15 | 565.85 |
| `shippedUnion--dino_boost--tolerant_intersection` | 2 | 82.35 | 97.87 | 89.44 | 4240.78 | 3675.15 | 565.62 |
| `shippedUnion--dino_boost--tolerant_intersection` | 3 | 84.02 | 98.21 | 90.56 | 4897.79 | 4323.26 | 574.53 |
| `shippedUnion--dino_boost--guarded_trim` | 0 | 62.44 | 96.20 | 75.73 | 3677.55 | 2387.15 | 1290.40 |
| `shippedUnion--dino_boost--guarded_trim` | 1 | 65.89 | 98.66 | 79.01 | 4471.33 | 3031.15 | 1440.17 |
| `shippedUnion--dino_boost--guarded_trim` | 2 | 69.24 | 99.58 | 81.68 | 5199.83 | 3675.15 | 1524.67 |
| `shippedUnion--dino_boost--guarded_trim` | 3 | 72.25 | 99.76 | 83.81 | 5886.40 | 4323.26 | 1563.13 |
| `shippedUnion--dino_boost--guarded_component_rejection` | 0 | 62.42 | 96.20 | 75.71 | 3679.01 | 2387.15 | 1291.86 |
| `shippedUnion--dino_boost--guarded_component_rejection` | 1 | 65.87 | 98.66 | 79.00 | 4472.79 | 3031.15 | 1441.64 |
| `shippedUnion--dino_boost--guarded_component_rejection` | 2 | 69.22 | 99.58 | 81.67 | 5201.29 | 3675.15 | 1526.14 |
| `shippedUnion--dino_boost--guarded_component_rejection` | 3 | 72.23 | 99.76 | 83.79 | 5887.86 | 4323.26 | 1564.60 |
| `shippedUnion--dino_keep--union` | 0 | 59.77 | 97.70 | 74.16 | 3903.01 | 2387.15 | 1515.86 |
| `shippedUnion--dino_keep--union` | 1 | 62.82 | 99.41 | 76.98 | 4740.58 | 3031.15 | 1709.43 |
| `shippedUnion--dino_keep--union` | 2 | 65.79 | 99.85 | 79.32 | 5513.02 | 3675.15 | 1837.87 |
| `shippedUnion--dino_keep--union` | 3 | 68.68 | 99.91 | 81.40 | 6236.67 | 4323.26 | 1913.41 |
| `shippedUnion--dino_keep--intersection` | 0 | 86.65 | 83.41 | 84.92 | 2302.50 | 2387.15 | -84.65 |
| `shippedUnion--dino_keep--intersection` | 1 | 89.05 | 92.95 | 90.93 | 2954.24 | 3031.15 | -76.91 |
| `shippedUnion--dino_keep--intersection` | 2 | 90.68 | 96.20 | 93.34 | 3592.83 | 3675.15 | -82.32 |
| `shippedUnion--dino_keep--intersection` | 3 | 91.99 | 97.37 | 94.59 | 4219.43 | 4323.26 | -103.84 |
| `shippedUnion--dino_keep--tolerant_intersection` | 0 | 75.75 | 93.52 | 83.68 | 2949.34 | 2387.15 | 562.19 |
| `shippedUnion--dino_keep--tolerant_intersection` | 1 | 79.60 | 96.48 | 87.21 | 3588.62 | 3031.15 | 557.47 |
| `shippedUnion--dino_keep--tolerant_intersection` | 2 | 82.29 | 97.65 | 89.30 | 4228.99 | 3675.15 | 553.84 |
| `shippedUnion--dino_keep--tolerant_intersection` | 3 | 84.02 | 98.00 | 90.46 | 4879.29 | 4323.26 | 556.03 |
| `shippedUnion--dino_keep--guarded_trim` | 0 | 62.45 | 96.20 | 75.74 | 3677.01 | 2387.15 | 1289.86 |
| `shippedUnion--dino_keep--guarded_trim` | 1 | 65.92 | 98.66 | 79.03 | 4469.45 | 3031.15 | 1438.30 |
| `shippedUnion--dino_keep--guarded_trim` | 2 | 69.28 | 99.58 | 81.71 | 5196.62 | 3675.15 | 1521.47 |
| `shippedUnion--dino_keep--guarded_trim` | 3 | 72.31 | 99.76 | 83.84 | 5881.15 | 4323.26 | 1557.88 |
| `shippedUnion--dino_keep--guarded_component_rejection` | 0 | 62.45 | 96.20 | 75.73 | 3677.26 | 2387.15 | 1290.11 |
| `shippedUnion--dino_keep--guarded_component_rejection` | 1 | 65.92 | 98.66 | 79.03 | 4469.71 | 3031.15 | 1438.55 |
| `shippedUnion--dino_keep--guarded_component_rejection` | 2 | 69.28 | 99.58 | 81.71 | 5196.87 | 3675.15 | 1521.72 |
| `shippedUnion--dino_keep--guarded_component_rejection` | 3 | 72.30 | 99.76 | 83.84 | 5881.40 | 4323.26 | 1558.14 |
| `shippedUnion--best_f1_pair--three_union` | 0 | 59.95 | 97.82 | 74.34 | 3895.40 | 2387.15 | 1508.25 |
| `shippedUnion--best_f1_pair--three_union` | 1 | 63.09 | 99.43 | 77.19 | 4724.80 | 3031.15 | 1693.65 |
| `shippedUnion--best_f1_pair--three_union` | 2 | 66.03 | 99.84 | 79.49 | 5495.62 | 3675.15 | 1820.47 |
| `shippedUnion--best_f1_pair--three_union` | 3 | 68.89 | 99.93 | 81.56 | 6223.03 | 4323.26 | 1899.76 |
| `shippedUnion--best_f1_pair--majority` | 0 | 83.26 | 90.81 | 86.87 | 2603.93 | 2387.15 | 216.78 |
| `shippedUnion--best_f1_pair--majority` | 1 | 85.47 | 96.66 | 90.72 | 3269.21 | 3031.15 | 238.06 |
| `shippedUnion--best_f1_pair--majority` | 2 | 87.28 | 98.11 | 92.38 | 3916.17 | 3675.15 | 241.02 |
| `shippedUnion--best_f1_pair--majority` | 3 | 88.78 | 98.67 | 93.46 | 4560.64 | 4323.26 | 237.37 |
| `shippedUnion--recovery_pair--three_union` | 0 | 59.40 | 97.85 | 73.92 | 3932.74 | 2387.15 | 1545.59 |
| `shippedUnion--recovery_pair--three_union` | 1 | 62.36 | 99.46 | 76.66 | 4780.30 | 3031.15 | 1749.15 |
| `shippedUnion--recovery_pair--three_union` | 2 | 65.39 | 99.89 | 79.03 | 5552.78 | 3675.15 | 1877.63 |
| `shippedUnion--recovery_pair--three_union` | 3 | 68.26 | 99.97 | 81.13 | 6282.48 | 4323.26 | 1959.22 |
| `shippedUnion--recovery_pair--majority` | 0 | 81.90 | 90.70 | 86.07 | 2643.70 | 2387.15 | 256.55 |
| `shippedUnion--recovery_pair--majority` | 1 | 84.49 | 96.77 | 90.21 | 3312.17 | 3031.15 | 281.02 |
| `shippedUnion--recovery_pair--majority` | 2 | 86.47 | 98.29 | 92.00 | 3964.42 | 3675.15 | 289.27 |
| `shippedUnion--recovery_pair--majority` | 3 | 88.07 | 98.91 | 93.18 | 4615.00 | 4323.26 | 291.74 |
| `refitUnion--compact_boost--union` | 0 | 51.78 | 95.86 | 67.24 | 4418.88 | 2387.15 | 2031.73 |
| `refitUnion--compact_boost--union` | 1 | 55.05 | 98.49 | 70.63 | 5337.35 | 3031.15 | 2306.20 |
| `refitUnion--compact_boost--union` | 2 | 58.87 | 99.41 | 73.95 | 6107.32 | 3675.15 | 2432.17 |
| `refitUnion--compact_boost--union` | 3 | 63.20 | 99.73 | 77.37 | 6727.36 | 4323.26 | 2404.10 |
| `refitUnion--compact_boost--intersection` | 0 | 84.65 | 78.81 | 81.62 | 2223.23 | 2387.15 | -163.92 |
| `refitUnion--compact_boost--intersection` | 1 | 86.95 | 87.79 | 87.36 | 2854.54 | 3031.15 | -176.61 |
| `refitUnion--compact_boost--intersection` | 2 | 88.72 | 91.07 | 89.88 | 3453.22 | 3675.15 | -221.93 |
| `refitUnion--compact_boost--intersection` | 3 | 90.12 | 92.88 | 91.48 | 4051.40 | 4323.26 | -271.87 |
| `refitUnion--compact_boost--tolerant_intersection` | 0 | 72.27 | 87.30 | 79.07 | 2884.23 | 2387.15 | 497.07 |
| `refitUnion--compact_boost--tolerant_intersection` | 1 | 76.45 | 91.65 | 83.36 | 3510.56 | 3031.15 | 479.41 |
| `refitUnion--compact_boost--tolerant_intersection` | 2 | 79.47 | 93.25 | 85.81 | 4111.21 | 3675.15 | 436.06 |
| `refitUnion--compact_boost--tolerant_intersection` | 3 | 81.79 | 94.28 | 87.59 | 4709.91 | 4323.26 | 386.65 |
| `refitUnion--compact_boost--guarded_trim` | 0 | 52.56 | 91.96 | 66.89 | 4176.99 | 2387.15 | 1789.84 |
| `refitUnion--compact_boost--guarded_trim` | 1 | 56.51 | 95.88 | 71.11 | 5037.94 | 3031.15 | 2006.78 |
| `refitUnion--compact_boost--guarded_trim` | 2 | 61.01 | 97.08 | 74.93 | 5712.75 | 3675.15 | 2037.60 |
| `refitUnion--compact_boost--guarded_trim` | 3 | 65.74 | 97.95 | 78.68 | 6278.28 | 4323.26 | 1955.02 |
| `refitUnion--compact_boost--guarded_component_rejection` | 0 | 52.58 | 92.25 | 66.98 | 4188.61 | 2387.15 | 1801.46 |
| `refitUnion--compact_boost--guarded_component_rejection` | 1 | 56.52 | 96.10 | 71.18 | 5049.55 | 3031.15 | 2018.40 |
| `refitUnion--compact_boost--guarded_component_rejection` | 2 | 61.00 | 97.28 | 74.99 | 5724.37 | 3675.15 | 2049.22 |
| `refitUnion--compact_boost--guarded_component_rejection` | 3 | 65.73 | 98.12 | 78.73 | 6288.67 | 4323.26 | 1965.41 |
| `refitUnion--compact_keep--union` | 0 | 51.67 | 96.03 | 67.19 | 4436.24 | 2387.15 | 2049.09 |
| `refitUnion--compact_keep--union` | 1 | 54.93 | 98.62 | 70.56 | 5355.70 | 3031.15 | 2324.55 |
| `refitUnion--compact_keep--union` | 2 | 58.81 | 99.52 | 73.93 | 6118.59 | 3675.15 | 2443.44 |
| `refitUnion--compact_keep--union` | 3 | 63.13 | 99.81 | 77.34 | 6739.99 | 4323.26 | 2416.72 |
| `refitUnion--compact_keep--intersection` | 0 | 83.56 | 79.49 | 81.45 | 2272.48 | 2387.15 | -114.67 |
| `refitUnion--compact_keep--intersection` | 1 | 85.78 | 88.88 | 87.29 | 2934.59 | 3031.15 | -96.56 |
| `refitUnion--compact_keep--intersection` | 2 | 87.60 | 92.00 | 89.73 | 3550.94 | 3675.15 | -124.21 |
| `refitUnion--compact_keep--intersection` | 3 | 88.97 | 93.65 | 91.24 | 4167.26 | 4323.26 | -156.00 |
| `refitUnion--compact_keep--tolerant_intersection` | 0 | 71.22 | 88.08 | 78.75 | 2953.70 | 2387.15 | 566.55 |
| `refitUnion--compact_keep--tolerant_intersection` | 1 | 75.21 | 92.34 | 82.89 | 3604.76 | 3031.15 | 573.61 |
| `refitUnion--compact_keep--tolerant_intersection` | 2 | 78.22 | 93.84 | 85.31 | 4222.19 | 3675.15 | 547.04 |
| `refitUnion--compact_keep--tolerant_intersection` | 3 | 80.55 | 94.74 | 87.06 | 4836.76 | 4323.26 | 513.50 |
| `refitUnion--compact_keep--guarded_trim` | 0 | 52.57 | 92.14 | 66.94 | 4184.27 | 2387.15 | 1797.12 |
| `refitUnion--compact_keep--guarded_trim` | 1 | 56.54 | 96.05 | 71.18 | 5047.04 | 3031.15 | 2015.89 |
| `refitUnion--compact_keep--guarded_trim` | 2 | 61.02 | 97.25 | 74.98 | 5725.48 | 3675.15 | 2050.33 |
| `refitUnion--compact_keep--guarded_trim` | 3 | 65.76 | 98.12 | 78.74 | 6292.73 | 4323.26 | 1969.47 |
| `refitUnion--compact_keep--guarded_component_rejection` | 0 | 52.56 | 92.38 | 67.00 | 4196.02 | 2387.15 | 1808.87 |
| `refitUnion--compact_keep--guarded_component_rejection` | 1 | 56.52 | 96.24 | 71.21 | 5058.79 | 3031.15 | 2027.64 |
| `refitUnion--compact_keep--guarded_component_rejection` | 2 | 60.99 | 97.42 | 75.02 | 5737.24 | 3675.15 | 2062.09 |
| `refitUnion--compact_keep--guarded_component_rejection` | 3 | 65.72 | 98.26 | 78.76 | 6305.54 | 4323.26 | 1982.27 |
| `refitUnion--dino_global--union` | 0 | 51.77 | 97.01 | 67.51 | 4472.94 | 2387.15 | 2085.78 |
| `refitUnion--dino_global--union` | 1 | 55.01 | 99.23 | 70.78 | 5386.28 | 3031.15 | 2355.13 |
| `refitUnion--dino_global--union` | 2 | 58.83 | 99.76 | 74.01 | 6141.69 | 3675.15 | 2466.54 |
| `refitUnion--dino_global--union` | 3 | 63.22 | 99.91 | 77.44 | 6750.27 | 4323.26 | 2427.01 |
| `refitUnion--dino_global--intersection` | 0 | 87.07 | 83.06 | 85.00 | 2278.91 | 2387.15 | -108.25 |
| `refitUnion--dino_global--intersection` | 1 | 89.03 | 92.46 | 90.70 | 2940.48 | 3031.15 | -90.67 |
| `refitUnion--dino_global--intersection` | 2 | 90.37 | 95.26 | 92.74 | 3572.68 | 3675.15 | -102.47 |
| `refitUnion--dino_global--intersection` | 3 | 91.47 | 96.48 | 93.90 | 4196.85 | 4323.26 | -126.41 |
| `refitUnion--dino_global--tolerant_intersection` | 0 | 73.53 | 90.55 | 81.14 | 2941.66 | 2387.15 | 554.51 |
| `refitUnion--dino_global--tolerant_intersection` | 1 | 77.38 | 94.79 | 85.19 | 3608.88 | 3031.15 | 577.73 |
| `refitUnion--dino_global--tolerant_intersection` | 2 | 80.30 | 96.34 | 87.58 | 4246.93 | 3675.15 | 571.78 |
| `refitUnion--dino_global--tolerant_intersection` | 3 | 82.36 | 97.27 | 89.18 | 4888.89 | 4323.26 | 565.63 |
| `refitUnion--dino_global--guarded_trim` | 0 | 52.62 | 92.73 | 67.14 | 4207.01 | 2387.15 | 1819.86 |
| `refitUnion--dino_global--guarded_trim` | 1 | 56.62 | 96.71 | 71.42 | 5080.80 | 3031.15 | 2049.65 |
| `refitUnion--dino_global--guarded_trim` | 2 | 61.16 | 98.11 | 75.35 | 5770.81 | 3675.15 | 2095.66 |
| `refitUnion--dino_global--guarded_trim` | 3 | 65.89 | 98.93 | 79.10 | 6342.49 | 4323.26 | 2019.22 |
| `refitUnion--dino_global--guarded_component_rejection` | 0 | 52.55 | 92.73 | 67.09 | 4212.01 | 2387.15 | 1824.86 |
| `refitUnion--dino_global--guarded_component_rejection` | 1 | 56.56 | 96.71 | 71.38 | 5085.80 | 3031.15 | 2054.65 |
| `refitUnion--dino_global--guarded_component_rejection` | 2 | 61.10 | 98.11 | 75.31 | 5775.80 | 3675.15 | 2100.65 |
| `refitUnion--dino_global--guarded_component_rejection` | 3 | 65.83 | 98.93 | 79.05 | 6349.64 | 4323.26 | 2026.37 |
| `refitUnion--dino_boost--union` | 0 | 51.71 | 97.08 | 67.48 | 4481.30 | 2387.15 | 2094.15 |
| `refitUnion--dino_boost--union` | 1 | 54.97 | 99.39 | 70.79 | 5396.25 | 3031.15 | 2365.10 |
| `refitUnion--dino_boost--union` | 2 | 58.83 | 99.88 | 74.04 | 6148.22 | 3675.15 | 2473.07 |
| `refitUnion--dino_boost--union` | 3 | 63.15 | 99.97 | 77.40 | 6768.98 | 4323.26 | 2445.71 |
| `refitUnion--dino_boost--intersection` | 0 | 85.68 | 81.89 | 83.73 | 2282.86 | 2387.15 | -104.29 |
| `refitUnion--dino_boost--intersection` | 1 | 87.85 | 91.92 | 89.83 | 2954.54 | 3031.15 | -76.61 |
| `refitUnion--dino_boost--intersection` | 2 | 89.44 | 95.03 | 92.14 | 3589.91 | 3675.15 | -85.24 |
| `refitUnion--dino_boost--intersection` | 3 | 90.86 | 96.54 | 93.61 | 4209.08 | 4323.26 | -114.19 |
| `refitUnion--dino_boost--tolerant_intersection` | 0 | 72.90 | 90.38 | 80.70 | 2960.18 | 2387.15 | 573.03 |
| `refitUnion--dino_boost--tolerant_intersection` | 1 | 76.90 | 94.79 | 84.90 | 3632.45 | 3031.15 | 601.30 |
| `refitUnion--dino_boost--tolerant_intersection` | 2 | 79.78 | 96.40 | 87.30 | 4281.48 | 3675.15 | 606.33 |
| `refitUnion--dino_boost--tolerant_intersection` | 3 | 82.04 | 97.39 | 89.06 | 4921.10 | 4323.26 | 597.84 |
| `refitUnion--dino_boost--guarded_trim` | 0 | 52.62 | 92.70 | 67.13 | 4205.33 | 2387.15 | 1818.18 |
| `refitUnion--dino_boost--guarded_trim` | 1 | 56.63 | 96.71 | 71.43 | 5078.95 | 3031.15 | 2047.80 |
| `refitUnion--dino_boost--guarded_trim` | 2 | 61.17 | 98.17 | 75.38 | 5770.83 | 3675.15 | 2095.68 |
| `refitUnion--dino_boost--guarded_trim` | 3 | 65.92 | 98.97 | 79.13 | 6341.79 | 4323.26 | 2018.53 |
| `refitUnion--dino_boost--guarded_component_rejection` | 0 | 52.53 | 92.70 | 67.06 | 4212.50 | 2387.15 | 1825.35 |
| `refitUnion--dino_boost--guarded_component_rejection` | 1 | 56.55 | 96.71 | 71.37 | 5086.12 | 3031.15 | 2054.96 |
| `refitUnion--dino_boost--guarded_component_rejection` | 2 | 61.10 | 98.17 | 75.32 | 5778.00 | 3675.15 | 2102.85 |
| `refitUnion--dino_boost--guarded_component_rejection` | 3 | 65.83 | 98.97 | 79.07 | 6351.00 | 4323.26 | 2027.73 |
| `refitUnion--dino_keep--union` | 0 | 51.57 | 96.80 | 67.29 | 4480.94 | 2387.15 | 2093.79 |
| `refitUnion--dino_keep--union` | 1 | 54.82 | 99.14 | 70.60 | 5397.81 | 3031.15 | 2366.66 |
| `refitUnion--dino_keep--union` | 2 | 58.75 | 99.79 | 73.96 | 6145.55 | 3675.15 | 2470.39 |
| `refitUnion--dino_keep--union` | 3 | 63.13 | 99.95 | 77.39 | 6759.07 | 4323.26 | 2435.80 |
| `refitUnion--dino_keep--intersection` | 0 | 84.28 | 80.83 | 82.41 | 2297.19 | 2387.15 | -89.96 |
| `refitUnion--dino_keep--intersection` | 1 | 87.00 | 91.32 | 89.05 | 2967.91 | 3031.15 | -63.24 |
| `refitUnion--dino_keep--intersection` | 2 | 88.92 | 94.78 | 91.72 | 3598.36 | 3675.15 | -76.79 |
| `refitUnion--dino_keep--intersection` | 3 | 90.45 | 96.34 | 93.28 | 4216.62 | 4323.26 | -106.64 |
| `refitUnion--dino_keep--tolerant_intersection` | 0 | 72.25 | 90.09 | 80.15 | 2981.98 | 2387.15 | 594.83 |
| `refitUnion--dino_keep--tolerant_intersection` | 1 | 76.48 | 94.60 | 84.55 | 3646.88 | 3031.15 | 615.72 |
| `refitUnion--dino_keep--tolerant_intersection` | 2 | 79.63 | 96.22 | 87.11 | 4281.82 | 3675.15 | 606.67 |
| `refitUnion--dino_keep--tolerant_intersection` | 3 | 82.01 | 97.19 | 88.93 | 4912.28 | 4323.26 | 589.01 |
| `refitUnion--dino_keep--guarded_trim` | 0 | 52.60 | 92.69 | 67.12 | 4206.09 | 2387.15 | 1818.94 |
| `refitUnion--dino_keep--guarded_trim` | 1 | 56.63 | 96.69 | 71.42 | 5076.02 | 3031.15 | 2044.86 |
| `refitUnion--dino_keep--guarded_trim` | 2 | 61.14 | 98.07 | 75.32 | 5766.23 | 3675.15 | 2091.08 |
| `refitUnion--dino_keep--guarded_trim` | 3 | 65.89 | 98.90 | 79.09 | 6337.36 | 4323.26 | 2014.09 |
| `refitUnion--dino_keep--guarded_component_rejection` | 0 | 52.57 | 92.69 | 67.09 | 4209.11 | 2387.15 | 1821.96 |
| `refitUnion--dino_keep--guarded_component_rejection` | 1 | 56.59 | 96.69 | 71.40 | 5079.03 | 3031.15 | 2047.88 |
| `refitUnion--dino_keep--guarded_component_rejection` | 2 | 61.10 | 98.07 | 75.30 | 5769.25 | 3675.15 | 2094.10 |
| `refitUnion--dino_keep--guarded_component_rejection` | 3 | 65.83 | 98.90 | 79.05 | 6343.57 | 4323.26 | 2020.31 |
| `refitUnion--best_f1_pair--three_union` | 0 | 51.68 | 97.33 | 67.51 | 4495.86 | 2387.15 | 2108.71 |
| `refitUnion--best_f1_pair--three_union` | 1 | 54.94 | 99.30 | 70.74 | 5406.65 | 3031.15 | 2375.50 |
| `refitUnion--best_f1_pair--three_union` | 2 | 58.73 | 99.80 | 73.94 | 6164.57 | 3675.15 | 2489.42 |
| `refitUnion--best_f1_pair--three_union` | 3 | 63.09 | 99.91 | 77.34 | 6775.81 | 4323.26 | 2452.55 |
| `refitUnion--best_f1_pair--majority` | 0 | 81.88 | 89.63 | 85.57 | 2613.27 | 2387.15 | 226.12 |
| `refitUnion--best_f1_pair--majority` | 1 | 84.35 | 95.91 | 89.76 | 3279.39 | 3031.15 | 248.24 |
| `refitUnion--best_f1_pair--majority` | 2 | 86.33 | 97.57 | 91.61 | 3924.26 | 3675.15 | 249.11 |
| `refitUnion--best_f1_pair--majority` | 3 | 87.86 | 98.25 | 92.76 | 4568.00 | 4323.26 | 244.74 |
| `refitUnion--recovery_pair--three_union` | 0 | 51.52 | 97.42 | 67.40 | 4513.87 | 2387.15 | 2126.72 |
| `refitUnion--recovery_pair--three_union` | 1 | 54.78 | 99.45 | 70.65 | 5427.82 | 3031.15 | 2396.67 |
| `refitUnion--recovery_pair--three_union` | 2 | 58.65 | 99.89 | 73.91 | 6178.32 | 3675.15 | 2503.17 |
| `refitUnion--recovery_pair--three_union` | 3 | 62.96 | 99.97 | 77.26 | 6799.77 | 4323.26 | 2476.50 |
| `refitUnion--recovery_pair--majority` | 0 | 80.10 | 89.61 | 84.58 | 2670.89 | 2387.15 | 283.74 |
| `refitUnion--recovery_pair--majority` | 1 | 82.57 | 96.00 | 88.77 | 3353.09 | 3031.15 | 321.94 |
| `refitUnion--recovery_pair--majority` | 2 | 84.69 | 97.73 | 90.74 | 4006.84 | 3675.15 | 331.69 |
| `refitUnion--recovery_pair--majority` | 3 | 86.47 | 98.53 | 92.10 | 4651.99 | 4323.26 | 328.73 |

## Per-source-group sensitivity at target padding

These group scores are diagnostics, not an average used to calculate the dataset ranking.

| Configuration | Source group | P_pad % | R_core % | F1_padP_coreR % | Export s | Incorrect export s | Missed core s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `shippedPrevious` | source-group-005 | 76.95 | 99.38 | 86.74 | 1159.80 | 267.28 | 3.59 |
| `shippedPrevious` | source-group-007 | 56.82 | 96.41 | 71.50 | 1722.93 | 743.88 | 22.36 |
| `shippedPrevious` | source-group-009 | 92.60 | 99.96 | 96.14 | 1099.58 | 81.31 | 0.29 |
| `shippedPrevious` | source-group-012 | 69.42 | 97.87 | 81.22 | 916.79 | 280.38 | 9.37 |
| `shippedV2` | source-group-005 | 68.40 | 97.04 | 80.24 | 1264.38 | 399.53 | 17.24 |
| `shippedV2` | source-group-007 | 63.13 | 97.61 | 76.67 | 1559.93 | 575.09 | 14.90 |
| `shippedV2` | source-group-009 | 88.69 | 97.87 | 93.06 | 1130.45 | 127.81 | 15.79 |
| `shippedV2` | source-group-012 | 83.50 | 99.41 | 90.76 | 777.58 | 128.32 | 2.57 |
| `shippedUnion` | source-group-005 | 67.52 | 99.87 | 80.57 | 1340.25 | 435.27 | 0.74 |
| `shippedUnion` | source-group-007 | 53.07 | 99.15 | 69.13 | 1919.56 | 900.88 | 5.27 |
| `shippedUnion` | source-group-009 | 87.67 | 100.00 | 93.43 | 1174.87 | 144.91 | 0.00 |
| `shippedUnion` | source-group-012 | 69.07 | 99.41 | 81.51 | 946.45 | 292.78 | 2.57 |
| `refitPrevious` | source-group-005 | 44.08 | 64.42 | 52.34 | 1317.42 | 736.72 | 207.39 |
| `refitPrevious` | source-group-007 | 54.83 | 97.08 | 70.08 | 1806.47 | 815.90 | 18.20 |
| `refitPrevious` | source-group-009 | 78.53 | 92.56 | 84.97 | 1144.30 | 245.72 | 55.25 |
| `refitPrevious` | source-group-012 | 69.87 | 98.18 | 81.64 | 916.41 | 276.10 | 7.99 |
| `refitV2` | source-group-005 | 57.33 | 88.92 | 69.71 | 1379.55 | 588.72 | 64.58 |
| `refitV2` | source-group-007 | 55.61 | 93.46 | 69.73 | 1692.86 | 751.39 | 40.70 |
| `refitV2` | source-group-009 | 89.30 | 98.04 | 93.46 | 1115.28 | 119.36 | 14.58 |
| `refitV2` | source-group-012 | 75.57 | 98.06 | 85.36 | 833.62 | 203.65 | 8.52 |
| `refitUnion` | source-group-005 | 47.80 | 98.39 | 64.34 | 1864.33 | 973.25 | 9.36 |
| `refitUnion` | source-group-007 | 52.03 | 97.55 | 67.86 | 1922.15 | 922.06 | 15.25 |
| `refitUnion` | source-group-009 | 76.63 | 98.83 | 86.32 | 1316.62 | 307.72 | 8.68 |
| `refitUnion` | source-group-012 | 69.05 | 98.43 | 81.16 | 930.39 | 287.94 | 6.88 |
| `productionDefault` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault` | source-group-007 | 58.88 | 98.93 | 73.82 | 1720.37 | 707.46 | 6.69 |
| `productionDefault` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionBalanced` | source-group-005 | 69.63 | 99.87 | 82.05 | 1299.73 | 394.75 | 0.74 |
| `productionBalanced` | source-group-007 | 56.91 | 98.93 | 72.25 | 1780.24 | 767.12 | 6.69 |
| `productionBalanced` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `productionBalanced` | source-group-012 | 74.37 | 99.41 | 85.09 | 877.12 | 224.82 | 2.57 |
| `productionConservative` | source-group-005 | 68.96 | 99.87 | 81.59 | 1312.23 | 407.25 | 0.74 |
| `productionConservative` | source-group-007 | 55.72 | 98.93 | 71.28 | 1818.33 | 805.21 | 6.69 |
| `productionConservative` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `productionConservative` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `compact_boost` | source-group-005 | 81.96 | 86.23 | 84.02 | 875.57 | 158.24 | 80.28 |
| `compact_boost` | source-group-007 | 82.03 | 85.68 | 83.73 | 970.14 | 176.43 | 89.18 |
| `compact_boost` | source-group-009 | 97.68 | 99.56 | 98.61 | 1013.18 | 23.54 | 3.26 |
| `compact_boost` | source-group-012 | 91.73 | 98.07 | 94.79 | 684.57 | 56.61 | 8.48 |
| `compact_keep` | source-group-005 | 80.07 | 87.68 | 83.61 | 910.16 | 183.73 | 71.80 |
| `compact_keep` | source-group-007 | 80.38 | 87.88 | 83.84 | 1032.45 | 205.58 | 75.44 |
| `compact_keep` | source-group-009 | 97.02 | 99.68 | 98.33 | 1022.95 | 30.67 | 2.41 |
| `compact_keep` | source-group-012 | 90.64 | 98.32 | 94.32 | 697.17 | 65.22 | 7.38 |
| `dino_global` | source-group-005 | 86.09 | 95.20 | 90.20 | 944.12 | 139.13 | 28.00 |
| `dino_global` | source-group-007 | 85.70 | 93.66 | 89.49 | 1030.54 | 148.22 | 39.49 |
| `dino_global` | source-group-009 | 98.09 | 99.32 | 98.70 | 995.56 | 19.27 | 5.06 |
| `dino_global` | source-group-012 | 88.60 | 100.00 | 93.95 | 736.86 | 84.23 | 0.02 |
| `dino_boost` | source-group-005 | 78.49 | 97.92 | 87.03 | 1091.24 | 239.32 | 12.15 |
| `dino_boost` | source-group-007 | 90.50 | 91.23 | 90.80 | 931.17 | 89.21 | 54.60 |
| `dino_boost` | source-group-009 | 98.62 | 99.20 | 98.90 | 974.59 | 13.68 | 5.95 |
| `dino_boost` | source-group-012 | 88.04 | 100.00 | 93.60 | 742.82 | 89.98 | 0.00 |
| `dino_keep` | source-group-005 | 76.68 | 96.64 | 85.16 | 1118.23 | 275.81 | 19.60 |
| `dino_keep` | source-group-007 | 89.49 | 93.93 | 91.58 | 988.25 | 106.30 | 37.78 |
| `dino_keep` | source-group-009 | 99.57 | 96.39 | 97.95 | 904.83 | 3.93 | 26.79 |
| `dino_keep` | source-group-012 | 89.88 | 100.00 | 94.65 | 726.64 | 74.00 | 0.00 |
| `compact_baseline` | source-group-005 | 80.46 | 86.03 | 82.96 | 885.58 | 176.52 | 81.46 |
| `compact_baseline` | source-group-007 | 81.31 | 85.87 | 83.43 | 986.14 | 186.75 | 87.99 |
| `compact_baseline` | source-group-009 | 97.11 | 99.68 | 98.37 | 1021.34 | 29.74 | 2.41 |
| `compact_baseline` | source-group-012 | 90.97 | 98.32 | 94.50 | 694.10 | 62.70 | 7.38 |
| `dino_baseline` | source-group-005 | 76.67 | 96.53 | 85.11 | 1115.21 | 275.30 | 20.21 |
| `dino_baseline` | source-group-007 | 89.59 | 92.44 | 90.86 | 960.11 | 103.05 | 47.09 |
| `dino_baseline` | source-group-009 | 99.62 | 95.57 | 97.55 | 890.80 | 3.35 | 32.87 |
| `dino_baseline` | source-group-012 | 89.94 | 100.00 | 94.69 | 725.76 | 73.45 | 0.00 |
| `best_f1_pair--nn_union` | source-group-005 | 79.30 | 97.68 | 87.47 | 1075.66 | 225.23 | 13.52 |
| `best_f1_pair--nn_union` | source-group-007 | 79.90 | 95.34 | 86.91 | 1152.44 | 232.64 | 29.03 |
| `best_f1_pair--nn_union` | source-group-009 | 96.83 | 99.88 | 98.33 | 1040.34 | 33.08 | 0.87 |
| `best_f1_pair--nn_union` | source-group-012 | 86.87 | 100.00 | 92.97 | 754.83 | 99.25 | 0.00 |
| `best_f1_pair--nn_intersection` | source-group-005 | 90.71 | 83.70 | 87.02 | 740.62 | 69.90 | 95.00 |
| `best_f1_pair--nn_intersection` | source-group-007 | 89.55 | 83.78 | 86.52 | 844.66 | 89.57 | 101.01 |
| `best_f1_pair--nn_intersection` | source-group-009 | 99.00 | 98.83 | 98.91 | 967.03 | 9.73 | 8.72 |
| `best_f1_pair--nn_intersection` | source-group-012 | 93.98 | 98.06 | 95.98 | 664.94 | 40.00 | 8.51 |
| `recovery_pair--nn_union` | source-group-005 | 73.43 | 98.48 | 84.10 | 1180.89 | 314.96 | 8.86 |
| `recovery_pair--nn_union` | source-group-007 | 80.33 | 95.54 | 87.21 | 1152.19 | 229.11 | 27.79 |
| `recovery_pair--nn_union` | source-group-009 | 96.89 | 99.85 | 98.34 | 1032.57 | 32.33 | 1.15 |
| `recovery_pair--nn_union` | source-group-012 | 86.09 | 100.00 | 92.50 | 763.19 | 106.90 | 0.00 |
| `recovery_pair--nn_intersection` | source-group-005 | 87.40 | 87.00 | 87.15 | 813.47 | 103.64 | 75.81 |
| `recovery_pair--nn_intersection` | source-group-007 | 91.98 | 82.98 | 87.12 | 806.38 | 65.80 | 106.00 |
| `recovery_pair--nn_intersection` | source-group-009 | 98.78 | 99.03 | 98.90 | 964.96 | 12.03 | 7.21 |
| `recovery_pair--nn_intersection` | source-group-012 | 93.27 | 98.32 | 95.72 | 673.81 | 45.45 | 7.38 |
| `productionDefault--compact_boost--union` | source-group-005 | 69.53 | 99.22 | 81.77 | 1287.18 | 392.23 | 4.52 |
| `productionDefault--compact_boost--union` | source-group-007 | 58.46 | 98.98 | 73.50 | 1735.97 | 721.19 | 6.35 |
| `productionDefault--compact_boost--union` | source-group-009 | 88.92 | 100.00 | 94.14 | 1159.87 | 128.46 | 0.00 |
| `productionDefault--compact_boost--union` | source-group-012 | 78.66 | 100.00 | 88.06 | 835.35 | 178.24 | 0.00 |
| `productionDefault--compact_boost--intersection` | source-group-005 | 85.53 | 85.97 | 85.74 | 832.49 | 120.55 | 81.80 |
| `productionDefault--compact_boost--intersection` | source-group-007 | 83.34 | 85.62 | 84.39 | 951.70 | 160.11 | 89.51 |
| `productionDefault--compact_boost--intersection` | source-group-009 | 97.83 | 99.56 | 98.69 | 1010.01 | 21.96 | 3.26 |
| `productionDefault--compact_boost--intersection` | source-group-012 | 92.50 | 96.77 | 94.58 | 667.66 | 50.08 | 14.18 |
| `productionDefault--compact_boost--tolerant_intersection` | source-group-005 | 78.63 | 89.32 | 83.63 | 993.60 | 212.28 | 62.27 |
| `productionDefault--compact_boost--tolerant_intersection` | source-group-007 | 73.86 | 90.04 | 81.12 | 1194.36 | 313.01 | 62.04 |
| `productionDefault--compact_boost--tolerant_intersection` | source-group-009 | 90.52 | 99.93 | 94.99 | 1134.09 | 107.56 | 0.53 |
| `productionDefault--compact_boost--tolerant_intersection` | source-group-012 | 82.90 | 97.53 | 89.62 | 765.89 | 131.00 | 10.85 |
| `productionDefault--compact_boost--guarded_trim` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--compact_boost--guarded_trim` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--compact_boost--guarded_trim` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--compact_boost--guarded_trim` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--compact_boost--guarded_component_rejection` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--compact_boost--guarded_component_rejection` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--compact_boost--guarded_component_rejection` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--compact_boost--guarded_component_rejection` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--compact_keep--union` | source-group-005 | 68.42 | 99.32 | 81.01 | 1311.62 | 414.73 | 3.98 |
| `productionDefault--compact_keep--union` | source-group-007 | 58.20 | 99.03 | 73.32 | 1744.48 | 729.17 | 6.03 |
| `productionDefault--compact_keep--union` | source-group-009 | 88.61 | 100.00 | 93.96 | 1164.19 | 132.58 | 0.00 |
| `productionDefault--compact_keep--union` | source-group-012 | 78.53 | 100.00 | 87.98 | 836.89 | 179.66 | 0.00 |
| `productionDefault--compact_keep--intersection` | source-group-005 | 85.52 | 87.33 | 86.38 | 840.80 | 122.02 | 73.86 |
| `productionDefault--compact_keep--intersection` | source-group-007 | 82.03 | 87.78 | 84.70 | 1007.66 | 183.50 | 76.10 |
| `productionDefault--compact_keep--intersection` | source-group-009 | 97.54 | 99.68 | 98.60 | 1015.49 | 25.01 | 2.41 |
| `productionDefault--compact_keep--intersection` | source-group-012 | 92.00 | 97.02 | 94.44 | 675.30 | 54.03 | 13.07 |
| `productionDefault--compact_keep--tolerant_intersection` | source-group-005 | 78.32 | 90.14 | 83.80 | 1011.73 | 219.37 | 57.47 |
| `productionDefault--compact_keep--tolerant_intersection` | source-group-007 | 72.43 | 91.55 | 80.82 | 1251.77 | 346.29 | 52.59 |
| `productionDefault--compact_keep--tolerant_intersection` | source-group-009 | 90.46 | 99.89 | 94.94 | 1133.89 | 108.17 | 0.83 |
| `productionDefault--compact_keep--tolerant_intersection` | source-group-012 | 82.63 | 97.56 | 89.48 | 769.56 | 133.70 | 10.69 |
| `productionDefault--compact_keep--guarded_trim` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--compact_keep--guarded_trim` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--compact_keep--guarded_trim` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--compact_keep--guarded_trim` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--compact_keep--guarded_component_rejection` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--compact_keep--guarded_component_rejection` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--compact_keep--guarded_component_rejection` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--compact_keep--guarded_component_rejection` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--dino_global--union` | source-group-005 | 69.71 | 99.66 | 82.01 | 1299.13 | 394.58 | 1.99 |
| `productionDefault--dino_global--union` | source-group-007 | 58.14 | 99.12 | 73.29 | 1748.24 | 731.77 | 5.51 |
| `productionDefault--dino_global--union` | source-group-009 | 88.60 | 100.00 | 93.96 | 1164.14 | 132.69 | 0.00 |
| `productionDefault--dino_global--union` | source-group-012 | 77.36 | 100.00 | 87.23 | 852.76 | 193.10 | 0.00 |
| `productionDefault--dino_global--intersection` | source-group-005 | 89.18 | 94.34 | 91.61 | 887.29 | 98.34 | 33.00 |
| `productionDefault--dino_global--intersection` | source-group-007 | 87.69 | 93.47 | 90.48 | 1001.24 | 123.94 | 40.66 |
| `productionDefault--dino_global--intersection` | source-group-009 | 98.64 | 99.32 | 98.97 | 988.27 | 13.62 | 5.06 |
| `productionDefault--dino_global--intersection` | source-group-012 | 91.05 | 98.70 | 94.71 | 702.74 | 63.08 | 5.72 |
| `productionDefault--dino_global--tolerant_intersection` | source-group-005 | 81.19 | 95.89 | 87.90 | 1043.14 | 196.93 | 23.97 |
| `productionDefault--dino_global--tolerant_intersection` | source-group-007 | 76.14 | 95.45 | 84.70 | 1247.03 | 298.20 | 28.31 |
| `productionDefault--dino_global--tolerant_intersection` | source-group-009 | 91.37 | 99.66 | 95.33 | 1120.01 | 96.74 | 2.56 |
| `productionDefault--dino_global--tolerant_intersection` | source-group-012 | 82.43 | 98.70 | 89.84 | 784.61 | 137.87 | 5.69 |
| `productionDefault--dino_global--guarded_trim` | source-group-005 | 71.58 | 98.96 | 83.07 | 1243.39 | 353.39 | 6.04 |
| `productionDefault--dino_global--guarded_trim` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--dino_global--guarded_trim` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--dino_global--guarded_trim` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--dino_global--guarded_component_rejection` | source-group-005 | 71.53 | 98.96 | 83.04 | 1244.19 | 354.19 | 6.04 |
| `productionDefault--dino_global--guarded_component_rejection` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--dino_global--guarded_component_rejection` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--dino_global--guarded_component_rejection` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--dino_boost--union` | source-group-005 | 67.20 | 99.91 | 80.34 | 1354.14 | 444.76 | 0.54 |
| `productionDefault--dino_boost--union` | source-group-007 | 58.72 | 99.17 | 73.77 | 1730.58 | 714.31 | 5.20 |
| `productionDefault--dino_boost--union` | source-group-009 | 88.86 | 100.00 | 94.10 | 1160.11 | 129.24 | 0.00 |
| `productionDefault--dino_boost--union` | source-group-012 | 77.15 | 100.00 | 87.09 | 855.35 | 195.74 | 0.00 |
| `productionDefault--dino_boost--intersection` | source-group-005 | 85.31 | 97.04 | 90.76 | 977.99 | 145.20 | 17.27 |
| `productionDefault--dino_boost--intersection` | source-group-007 | 91.16 | 90.97 | 91.00 | 918.96 | 81.82 | 56.25 |
| `productionDefault--dino_boost--intersection` | source-group-009 | 98.82 | 99.20 | 99.01 | 971.46 | 11.62 | 5.95 |
| `productionDefault--dino_boost--intersection` | source-group-012 | 90.60 | 98.70 | 94.47 | 706.71 | 66.75 | 5.69 |
| `productionDefault--dino_boost--tolerant_intersection` | source-group-005 | 79.21 | 98.02 | 87.59 | 1101.77 | 229.86 | 11.57 |
| `productionDefault--dino_boost--tolerant_intersection` | source-group-007 | 79.47 | 93.85 | 86.02 | 1175.63 | 242.13 | 38.26 |
| `productionDefault--dino_boost--tolerant_intersection` | source-group-009 | 91.32 | 99.71 | 95.33 | 1121.00 | 97.46 | 2.14 |
| `productionDefault--dino_boost--tolerant_intersection` | source-group-012 | 82.40 | 98.70 | 89.82 | 784.90 | 138.17 | 5.69 |
| `productionDefault--dino_boost--guarded_trim` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--dino_boost--guarded_trim` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--dino_boost--guarded_trim` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--dino_boost--guarded_trim` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--dino_boost--guarded_component_rejection` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--dino_boost--guarded_component_rejection` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--dino_boost--guarded_component_rejection` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--dino_boost--guarded_component_rejection` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--dino_keep--union` | source-group-005 | 66.27 | 99.75 | 79.58 | 1375.62 | 466.92 | 1.45 |
| `productionDefault--dino_keep--union` | source-group-007 | 58.66 | 99.13 | 73.70 | 1731.63 | 715.92 | 5.42 |
| `productionDefault--dino_keep--union` | source-group-009 | 89.00 | 100.00 | 94.18 | 1157.29 | 127.31 | 0.00 |
| `productionDefault--dino_keep--union` | source-group-012 | 78.02 | 100.00 | 87.65 | 845.40 | 185.82 | 0.00 |
| `productionDefault--dino_keep--intersection` | source-group-005 | 84.09 | 96.00 | 89.52 | 985.55 | 161.38 | 23.31 |
| `productionDefault--dino_keep--intersection` | source-group-007 | 90.33 | 93.73 | 91.93 | 974.27 | 96.13 | 39.05 |
| `productionDefault--dino_keep--intersection` | source-group-009 | 99.58 | 96.15 | 97.83 | 902.70 | 3.80 | 28.61 |
| `productionDefault--dino_keep--intersection` | source-group-012 | 91.31 | 98.70 | 94.86 | 700.88 | 61.11 | 5.69 |
| `productionDefault--dino_keep--tolerant_intersection` | source-group-005 | 78.53 | 96.69 | 86.61 | 1097.08 | 237.36 | 19.27 |
| `productionDefault--dino_keep--tolerant_intersection` | source-group-007 | 78.38 | 95.82 | 86.17 | 1228.77 | 267.47 | 26.05 |
| `productionDefault--dino_keep--tolerant_intersection` | source-group-009 | 92.86 | 98.51 | 95.60 | 1072.62 | 76.58 | 11.06 |
| `productionDefault--dino_keep--tolerant_intersection` | source-group-012 | 82.93 | 98.70 | 90.13 | 780.02 | 133.29 | 5.69 |
| `productionDefault--dino_keep--guarded_trim` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--dino_keep--guarded_trim` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--dino_keep--guarded_trim` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--dino_keep--guarded_trim` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--dino_keep--guarded_component_rejection` | source-group-005 | 71.49 | 99.13 | 83.07 | 1247.23 | 355.61 | 5.09 |
| `productionDefault--dino_keep--guarded_component_rejection` | source-group-007 | 59.10 | 98.93 | 74.00 | 1713.87 | 700.96 | 6.69 |
| `productionDefault--dino_keep--guarded_component_rejection` | source-group-009 | 89.01 | 100.00 | 94.18 | 1156.98 | 127.18 | 0.00 |
| `productionDefault--dino_keep--guarded_component_rejection` | source-group-012 | 79.02 | 98.70 | 87.77 | 818.44 | 171.71 | 5.69 |
| `productionDefault--best_f1_pair--three_union` | source-group-005 | 68.22 | 99.66 | 80.98 | 1328.05 | 422.77 | 1.99 |
| `productionDefault--best_f1_pair--three_union` | source-group-007 | 57.88 | 99.17 | 73.10 | 1757.63 | 740.30 | 5.18 |
| `productionDefault--best_f1_pair--three_union` | source-group-009 | 88.53 | 100.00 | 93.91 | 1166.13 | 133.80 | 0.00 |
| `productionDefault--best_f1_pair--three_union` | source-group-012 | 77.11 | 100.00 | 87.07 | 856.04 | 195.99 | 0.00 |
| `productionDefault--best_f1_pair--majority` | source-group-005 | 83.60 | 97.25 | 89.88 | 1004.69 | 165.60 | 16.05 |
| `productionDefault--best_f1_pair--majority` | source-group-007 | 81.76 | 95.10 | 87.89 | 1120.54 | 205.55 | 30.53 |
| `productionDefault--best_f1_pair--majority` | source-group-009 | 97.46 | 99.88 | 98.66 | 1031.66 | 26.22 | 0.87 |
| `productionDefault--best_f1_pair--majority` | source-group-012 | 89.28 | 100.00 | 94.33 | 730.72 | 78.48 | 0.00 |
| `productionDefault--recovery_pair--three_union` | source-group-005 | 65.47 | 100.00 | 79.12 | 1392.79 | 481.37 | 0.00 |
| `productionDefault--recovery_pair--three_union` | source-group-007 | 58.11 | 99.27 | 73.30 | 1751.78 | 733.92 | 4.54 |
| `productionDefault--recovery_pair--three_union` | source-group-009 | 88.60 | 100.00 | 93.95 | 1164.55 | 132.82 | 0.00 |
| `productionDefault--recovery_pair--three_union` | source-group-012 | 76.81 | 100.00 | 86.87 | 859.80 | 199.67 | 0.00 |
| `productionDefault--recovery_pair--majority` | source-group-005 | 80.61 | 97.61 | 88.29 | 1051.89 | 204.40 | 13.94 |
| `productionDefault--recovery_pair--majority` | source-group-007 | 81.94 | 95.18 | 88.02 | 1121.53 | 204.26 | 30.02 |
| `productionDefault--recovery_pair--majority` | source-group-009 | 97.26 | 99.85 | 98.53 | 1027.52 | 28.25 | 1.15 |
| `productionDefault--recovery_pair--majority` | source-group-012 | 88.72 | 100.00 | 94.02 | 736.13 | 83.28 | 0.00 |
| `shippedUnion--compact_boost--union` | source-group-005 | 66.31 | 99.97 | 79.73 | 1369.35 | 461.34 | 0.17 |
| `shippedUnion--compact_boost--union` | source-group-007 | 52.81 | 99.21 | 68.93 | 1932.38 | 911.83 | 4.94 |
| `shippedUnion--compact_boost--union` | source-group-009 | 87.59 | 100.00 | 93.38 | 1177.70 | 146.20 | 0.00 |
| `shippedUnion--compact_boost--union` | source-group-012 | 69.18 | 100.00 | 81.78 | 951.84 | 293.35 | 0.00 |
| `shippedUnion--compact_boost--intersection` | source-group-005 | 84.33 | 85.97 | 85.13 | 844.83 | 132.59 | 81.80 |
| `shippedUnion--compact_boost--intersection` | source-group-007 | 83.20 | 85.62 | 84.31 | 953.91 | 162.07 | 89.51 |
| `shippedUnion--compact_boost--intersection` | source-group-009 | 97.83 | 99.56 | 98.69 | 1010.01 | 21.96 | 3.26 |
| `shippedUnion--compact_boost--intersection` | source-group-012 | 92.31 | 97.48 | 94.82 | 674.57 | 51.87 | 11.06 |
| `shippedUnion--compact_boost--tolerant_intersection` | source-group-005 | 77.58 | 89.32 | 83.03 | 1007.94 | 226.03 | 62.27 |
| `shippedUnion--compact_boost--tolerant_intersection` | source-group-007 | 73.67 | 90.04 | 81.00 | 1197.66 | 316.31 | 62.04 |
| `shippedUnion--compact_boost--tolerant_intersection` | source-group-009 | 90.46 | 99.93 | 94.96 | 1134.81 | 108.22 | 0.53 |
| `shippedUnion--compact_boost--tolerant_intersection` | source-group-012 | 82.01 | 98.24 | 89.39 | 780.40 | 140.39 | 7.73 |
| `shippedUnion--compact_boost--guarded_trim` | source-group-005 | 68.84 | 99.87 | 81.50 | 1314.57 | 409.59 | 0.74 |
| `shippedUnion--compact_boost--guarded_trim` | source-group-007 | 55.92 | 98.93 | 71.45 | 1811.83 | 798.71 | 6.69 |
| `shippedUnion--compact_boost--guarded_trim` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `shippedUnion--compact_boost--guarded_trim` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--compact_boost--guarded_component_rejection` | source-group-005 | 68.82 | 99.87 | 81.49 | 1315.07 | 410.09 | 0.74 |
| `shippedUnion--compact_boost--guarded_component_rejection` | source-group-007 | 55.92 | 98.93 | 71.45 | 1811.83 | 798.71 | 6.69 |
| `shippedUnion--compact_boost--guarded_component_rejection` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `shippedUnion--compact_boost--guarded_component_rejection` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--compact_keep--union` | source-group-005 | 65.32 | 99.97 | 79.01 | 1390.36 | 482.44 | 0.17 |
| `shippedUnion--compact_keep--union` | source-group-007 | 52.74 | 99.26 | 68.88 | 1936.17 | 915.10 | 4.61 |
| `shippedUnion--compact_keep--union` | source-group-009 | 87.41 | 100.00 | 93.28 | 1180.36 | 148.65 | 0.00 |
| `shippedUnion--compact_keep--union` | source-group-012 | 69.17 | 100.00 | 81.78 | 952.13 | 293.52 | 0.00 |
| `shippedUnion--compact_keep--intersection` | source-group-005 | 84.39 | 87.42 | 85.84 | 855.19 | 134.09 | 73.32 |
| `shippedUnion--compact_keep--intersection` | source-group-007 | 81.60 | 87.78 | 84.46 | 1013.67 | 189.25 | 76.10 |
| `shippedUnion--compact_keep--intersection` | source-group-009 | 97.39 | 99.68 | 98.52 | 1017.15 | 26.67 | 2.41 |
| `shippedUnion--compact_keep--intersection` | source-group-012 | 91.81 | 97.73 | 94.68 | 682.26 | 55.87 | 9.95 |
| `shippedUnion--compact_keep--tolerant_intersection` | source-group-005 | 77.17 | 90.23 | 83.18 | 1030.52 | 235.39 | 56.93 |
| `shippedUnion--compact_keep--tolerant_intersection` | source-group-007 | 71.88 | 91.55 | 80.46 | 1262.03 | 356.43 | 52.59 |
| `shippedUnion--compact_keep--tolerant_intersection` | source-group-009 | 90.23 | 99.89 | 94.81 | 1136.86 | 111.09 | 0.83 |
| `shippedUnion--compact_keep--tolerant_intersection` | source-group-012 | 81.46 | 98.28 | 89.08 | 786.83 | 145.85 | 7.57 |
| `shippedUnion--compact_keep--guarded_trim` | source-group-005 | 68.86 | 99.87 | 81.52 | 1314.27 | 409.29 | 0.74 |
| `shippedUnion--compact_keep--guarded_trim` | source-group-007 | 55.83 | 98.93 | 71.38 | 1814.63 | 801.51 | 6.69 |
| `shippedUnion--compact_keep--guarded_trim` | source-group-009 | 88.49 | 100.00 | 93.90 | 1163.87 | 133.91 | 0.00 |
| `shippedUnion--compact_keep--guarded_trim` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--compact_keep--guarded_component_rejection` | source-group-005 | 68.85 | 99.87 | 81.51 | 1314.40 | 409.42 | 0.74 |
| `shippedUnion--compact_keep--guarded_component_rejection` | source-group-007 | 55.83 | 98.93 | 71.38 | 1814.63 | 801.51 | 6.69 |
| `shippedUnion--compact_keep--guarded_component_rejection` | source-group-009 | 88.49 | 100.00 | 93.90 | 1163.87 | 133.91 | 0.00 |
| `shippedUnion--compact_keep--guarded_component_rejection` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--dino_global--union` | source-group-005 | 66.10 | 100.00 | 79.58 | 1380.18 | 468.52 | 0.00 |
| `shippedUnion--dino_global--union` | source-group-007 | 52.84 | 99.34 | 68.99 | 1936.00 | 913.01 | 4.10 |
| `shippedUnion--dino_global--union` | source-group-009 | 87.28 | 100.00 | 93.20 | 1182.03 | 150.43 | 0.00 |
| `shippedUnion--dino_global--union` | source-group-012 | 68.45 | 100.00 | 81.27 | 963.95 | 304.17 | 0.00 |
| `shippedUnion--dino_global--intersection` | source-group-005 | 88.95 | 94.74 | 91.66 | 896.72 | 101.87 | 30.65 |
| `shippedUnion--dino_global--intersection` | source-group-007 | 86.83 | 93.47 | 90.02 | 1011.92 | 134.01 | 40.66 |
| `shippedUnion--dino_global--intersection` | source-group-009 | 98.64 | 99.32 | 98.97 | 988.27 | 13.62 | 5.06 |
| `shippedUnion--dino_global--intersection` | source-group-012 | 90.31 | 99.41 | 94.64 | 714.11 | 69.33 | 2.59 |
| `shippedUnion--dino_global--tolerant_intersection` | source-group-005 | 80.59 | 96.29 | 87.70 | 1060.53 | 207.15 | 21.61 |
| `shippedUnion--dino_global--tolerant_intersection` | source-group-007 | 74.93 | 95.45 | 83.93 | 1267.82 | 318.86 | 28.31 |
| `shippedUnion--dino_global--tolerant_intersection` | source-group-009 | 91.37 | 99.66 | 95.33 | 1120.01 | 96.74 | 2.56 |
| `shippedUnion--dino_global--tolerant_intersection` | source-group-012 | 81.30 | 99.41 | 89.45 | 801.82 | 149.97 | 2.57 |
| `shippedUnion--dino_global--guarded_trim` | source-group-005 | 68.96 | 99.71 | 81.53 | 1310.07 | 406.70 | 1.69 |
| `shippedUnion--dino_global--guarded_trim` | source-group-007 | 55.83 | 98.93 | 71.38 | 1814.63 | 801.51 | 6.69 |
| `shippedUnion--dino_global--guarded_trim` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `shippedUnion--dino_global--guarded_trim` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--dino_global--guarded_component_rejection` | source-group-005 | 68.85 | 99.71 | 81.46 | 1312.03 | 408.67 | 1.69 |
| `shippedUnion--dino_global--guarded_component_rejection` | source-group-007 | 55.83 | 98.93 | 71.38 | 1814.63 | 801.51 | 6.69 |
| `shippedUnion--dino_global--guarded_component_rejection` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `shippedUnion--dino_global--guarded_component_rejection` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--dino_boost--union` | source-group-005 | 64.01 | 100.00 | 78.04 | 1427.18 | 514.25 | 0.00 |
| `shippedUnion--dino_boost--union` | source-group-007 | 53.10 | 99.46 | 69.24 | 1927.51 | 903.94 | 3.35 |
| `shippedUnion--dino_boost--union` | source-group-009 | 87.64 | 100.00 | 93.41 | 1176.41 | 145.40 | 0.00 |
| `shippedUnion--dino_boost--union` | source-group-012 | 68.16 | 100.00 | 81.06 | 968.05 | 308.33 | 0.00 |
| `shippedUnion--dino_boost--intersection` | source-group-005 | 84.48 | 97.69 | 90.57 | 998.87 | 156.58 | 13.46 |
| `shippedUnion--dino_boost--intersection` | source-group-007 | 91.03 | 90.97 | 90.93 | 920.80 | 83.30 | 56.25 |
| `shippedUnion--dino_boost--intersection` | source-group-009 | 98.67 | 99.20 | 98.93 | 973.04 | 13.20 | 5.95 |
| `shippedUnion--dino_boost--intersection` | source-group-012 | 90.04 | 99.41 | 94.48 | 716.91 | 71.84 | 2.57 |
| `shippedUnion--dino_boost--tolerant_intersection` | source-group-005 | 78.07 | 98.67 | 87.15 | 1132.40 | 249.24 | 7.76 |
| `shippedUnion--dino_boost--tolerant_intersection` | source-group-007 | 78.99 | 93.85 | 85.73 | 1183.24 | 249.60 | 38.26 |
| `shippedUnion--dino_boost--tolerant_intersection` | source-group-009 | 91.14 | 99.71 | 95.23 | 1123.25 | 99.71 | 2.14 |
| `shippedUnion--dino_boost--tolerant_intersection` | source-group-012 | 81.31 | 99.41 | 89.45 | 801.89 | 150.03 | 2.57 |
| `shippedUnion--dino_boost--guarded_trim` | source-group-005 | 68.88 | 99.87 | 81.53 | 1313.82 | 408.84 | 0.74 |
| `shippedUnion--dino_boost--guarded_trim` | source-group-007 | 55.92 | 98.93 | 71.45 | 1811.83 | 798.71 | 6.69 |
| `shippedUnion--dino_boost--guarded_trim` | source-group-009 | 88.49 | 100.00 | 93.90 | 1163.87 | 133.91 | 0.00 |
| `shippedUnion--dino_boost--guarded_trim` | source-group-012 | 71.66 | 99.41 | 83.28 | 910.31 | 258.01 | 2.57 |
| `shippedUnion--dino_boost--guarded_component_rejection` | source-group-005 | 68.82 | 99.87 | 81.49 | 1315.07 | 410.09 | 0.74 |
| `shippedUnion--dino_boost--guarded_component_rejection` | source-group-007 | 55.92 | 98.93 | 71.45 | 1811.83 | 798.71 | 6.69 |
| `shippedUnion--dino_boost--guarded_component_rejection` | source-group-009 | 88.49 | 100.00 | 93.90 | 1163.87 | 133.91 | 0.00 |
| `shippedUnion--dino_boost--guarded_component_rejection` | source-group-012 | 71.64 | 99.41 | 83.27 | 910.52 | 258.22 | 2.57 |
| `shippedUnion--dino_keep--union` | source-group-005 | 63.28 | 100.00 | 77.47 | 1446.57 | 533.24 | 0.00 |
| `shippedUnion--dino_keep--union` | source-group-007 | 53.05 | 99.42 | 69.18 | 1929.32 | 905.85 | 3.63 |
| `shippedUnion--dino_keep--union` | source-group-009 | 87.66 | 100.00 | 93.42 | 1175.17 | 145.05 | 0.00 |
| `shippedUnion--dino_keep--union` | source-group-012 | 68.57 | 100.00 | 81.35 | 961.96 | 302.38 | 0.00 |
| `shippedUnion--dino_keep--intersection` | source-group-005 | 83.25 | 96.50 | 89.24 | 1006.47 | 173.87 | 20.41 |
| `shippedUnion--dino_keep--intersection` | source-group-007 | 90.18 | 93.73 | 91.85 | 976.19 | 97.87 | 39.05 |
| `shippedUnion--dino_keep--intersection` | source-group-009 | 99.58 | 96.15 | 97.83 | 902.70 | 3.80 | 28.61 |
| `shippedUnion--dino_keep--intersection` | source-group-012 | 91.19 | 99.41 | 95.12 | 707.47 | 62.58 | 2.57 |
| `shippedUnion--dino_keep--tolerant_intersection` | source-group-005 | 77.25 | 97.19 | 85.99 | 1128.18 | 259.37 | 16.37 |
| `shippedUnion--dino_keep--tolerant_intersection` | source-group-007 | 77.93 | 95.82 | 85.90 | 1236.20 | 274.83 | 26.05 |
| `shippedUnion--dino_keep--tolerant_intersection` | source-group-009 | 92.86 | 98.51 | 95.60 | 1072.62 | 76.58 | 11.06 |
| `shippedUnion--dino_keep--tolerant_intersection` | source-group-012 | 82.33 | 99.41 | 90.06 | 791.99 | 140.13 | 2.57 |
| `shippedUnion--dino_keep--guarded_trim` | source-group-005 | 68.83 | 99.87 | 81.50 | 1314.82 | 409.84 | 0.74 |
| `shippedUnion--dino_keep--guarded_trim` | source-group-007 | 55.92 | 98.93 | 71.45 | 1811.83 | 798.71 | 6.69 |
| `shippedUnion--dino_keep--guarded_trim` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `shippedUnion--dino_keep--guarded_trim` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--dino_keep--guarded_component_rejection` | source-group-005 | 68.82 | 99.87 | 81.49 | 1315.07 | 410.09 | 0.74 |
| `shippedUnion--dino_keep--guarded_component_rejection` | source-group-007 | 55.92 | 98.93 | 71.45 | 1811.83 | 798.71 | 6.69 |
| `shippedUnion--dino_keep--guarded_component_rejection` | source-group-009 | 88.67 | 100.00 | 93.99 | 1161.62 | 131.66 | 0.00 |
| `shippedUnion--dino_keep--guarded_component_rejection` | source-group-012 | 71.81 | 99.41 | 83.39 | 908.35 | 256.05 | 2.57 |
| `shippedUnion--best_f1_pair--three_union` | source-group-005 | 65.13 | 100.00 | 78.87 | 1401.66 | 489.30 | 0.00 |
| `shippedUnion--best_f1_pair--three_union` | source-group-007 | 52.63 | 99.40 | 68.82 | 1945.40 | 921.54 | 3.76 |
| `shippedUnion--best_f1_pair--three_union` | source-group-009 | 87.20 | 100.00 | 93.16 | 1183.97 | 151.53 | 0.00 |
| `shippedUnion--best_f1_pair--three_union` | source-group-012 | 68.44 | 100.00 | 81.26 | 964.59 | 304.43 | 0.00 |
| `shippedUnion--best_f1_pair--majority` | source-group-005 | 82.97 | 97.65 | 89.70 | 1018.82 | 174.15 | 13.69 |
| `shippedUnion--best_f1_pair--majority` | source-group-007 | 81.16 | 95.10 | 87.55 | 1129.26 | 213.75 | 30.53 |
| `shippedUnion--best_f1_pair--majority` | source-group-009 | 97.46 | 99.88 | 98.66 | 1031.66 | 26.22 | 0.87 |
| `shippedUnion--best_f1_pair--majority` | source-group-012 | 88.58 | 100.00 | 93.94 | 736.43 | 84.18 | 0.00 |
| `shippedUnion--recovery_pair--three_union` | source-group-005 | 62.57 | 100.00 | 76.97 | 1460.71 | 547.15 | 0.00 |
| `shippedUnion--recovery_pair--three_union` | source-group-007 | 52.78 | 99.57 | 68.99 | 1942.27 | 917.12 | 2.70 |
| `shippedUnion--recovery_pair--three_union` | source-group-009 | 87.39 | 100.00 | 93.27 | 1180.72 | 148.90 | 0.00 |
| `shippedUnion--recovery_pair--three_union` | source-group-012 | 68.13 | 100.00 | 81.04 | 969.07 | 308.95 | 0.00 |
| `shippedUnion--recovery_pair--majority` | source-group-005 | 80.31 | 98.35 | 88.41 | 1068.56 | 210.73 | 9.59 |
| `shippedUnion--recovery_pair--majority` | source-group-007 | 81.46 | 95.18 | 87.73 | 1129.36 | 211.51 | 30.02 |
| `shippedUnion--recovery_pair--majority` | source-group-009 | 97.25 | 99.85 | 98.53 | 1027.59 | 28.32 | 1.15 |
| `shippedUnion--recovery_pair--majority` | source-group-012 | 88.39 | 100.00 | 93.83 | 738.90 | 86.03 | 0.00 |
| `refitUnion--compact_boost--union` | source-group-005 | 47.94 | 99.36 | 64.68 | 1877.70 | 977.46 | 3.73 |
| `refitUnion--compact_boost--union` | source-group-007 | 51.96 | 98.63 | 68.06 | 1945.09 | 934.40 | 8.56 |
| `refitUnion--compact_boost--union` | source-group-009 | 76.77 | 100.00 | 86.86 | 1341.46 | 311.68 | 0.00 |
| `refitUnion--compact_boost--union` | source-group-012 | 69.39 | 99.61 | 81.80 | 943.07 | 288.63 | 1.73 |
| `refitUnion--compact_boost--intersection` | source-group-005 | 82.11 | 84.71 | 83.37 | 852.73 | 152.91 | 89.12 |
| `refitUnion--compact_boost--intersection` | source-group-007 | 82.90 | 84.17 | 83.47 | 942.57 | 163.12 | 98.54 |
| `refitUnion--compact_boost--intersection` | source-group-009 | 97.88 | 98.39 | 98.13 | 989.16 | 21.01 | 11.94 |
| `refitUnion--compact_boost--intersection` | source-group-012 | 92.08 | 96.89 | 94.42 | 668.77 | 52.98 | 13.64 |
| `refitUnion--compact_boost--tolerant_intersection` | source-group-005 | 71.71 | 87.74 | 78.91 | 1069.88 | 302.90 | 71.47 |
| `refitUnion--compact_boost--tolerant_intersection` | source-group-007 | 73.70 | 88.91 | 80.55 | 1176.61 | 310.63 | 69.07 |
| `refitUnion--compact_boost--tolerant_intersection` | source-group-009 | 91.62 | 98.76 | 95.05 | 1094.10 | 91.73 | 9.21 |
| `refitUnion--compact_boost--tolerant_intersection` | source-group-012 | 81.97 | 97.43 | 89.03 | 770.63 | 138.94 | 11.28 |
| `refitUnion--compact_boost--guarded_trim` | source-group-005 | 48.89 | 94.38 | 64.41 | 1739.35 | 889.02 | 32.74 |
| `refitUnion--compact_boost--guarded_trim` | source-group-007 | 54.43 | 96.62 | 69.63 | 1810.32 | 824.92 | 21.04 |
| `refitUnion--compact_boost--guarded_trim` | source-group-009 | 80.43 | 98.77 | 88.66 | 1251.91 | 244.98 | 9.10 |
| `refitUnion--compact_boost--guarded_trim` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--compact_boost--guarded_component_rejection` | source-group-005 | 49.01 | 95.22 | 64.71 | 1749.10 | 891.85 | 27.84 |
| `refitUnion--compact_boost--guarded_component_rejection` | source-group-007 | 54.38 | 96.62 | 69.59 | 1812.20 | 826.80 | 21.04 |
| `refitUnion--compact_boost--guarded_component_rejection` | source-group-009 | 80.43 | 98.77 | 88.66 | 1251.91 | 244.98 | 9.10 |
| `refitUnion--compact_boost--guarded_component_rejection` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--compact_keep--union` | source-group-005 | 47.87 | 99.42 | 64.62 | 1880.82 | 980.54 | 3.41 |
| `refitUnion--compact_keep--union` | source-group-007 | 51.96 | 98.97 | 68.14 | 1951.24 | 937.39 | 6.39 |
| `refitUnion--compact_keep--union` | source-group-009 | 76.67 | 100.00 | 86.80 | 1343.28 | 313.35 | 0.00 |
| `refitUnion--compact_keep--union` | source-group-012 | 69.39 | 99.63 | 81.80 | 943.25 | 288.72 | 1.64 |
| `refitUnion--compact_keep--intersection` | source-group-005 | 80.73 | 86.30 | 83.36 | 881.76 | 171.28 | 79.88 |
| `refitUnion--compact_keep--intersection` | source-group-007 | 81.29 | 85.97 | 83.46 | 998.24 | 189.60 | 87.37 |
| `refitUnion--compact_keep--intersection` | source-group-009 | 97.61 | 98.51 | 98.06 | 994.18 | 23.78 | 11.09 |
| `refitUnion--compact_keep--intersection` | source-group-012 | 91.54 | 97.12 | 94.25 | 676.77 | 57.26 | 12.62 |
| `refitUnion--compact_keep--tolerant_intersection` | source-group-005 | 70.41 | 88.94 | 78.54 | 1112.57 | 331.08 | 64.46 |
| `refitUnion--compact_keep--tolerant_intersection` | source-group-007 | 71.92 | 90.07 | 79.91 | 1234.34 | 348.37 | 61.82 |
| `refitUnion--compact_keep--tolerant_intersection` | source-group-009 | 91.27 | 98.72 | 94.85 | 1097.41 | 95.87 | 9.52 |
| `refitUnion--compact_keep--tolerant_intersection` | source-group-012 | 81.29 | 97.43 | 88.63 | 777.86 | 145.50 | 11.28 |
| `refitUnion--compact_keep--guarded_trim` | source-group-005 | 48.99 | 94.99 | 64.64 | 1748.92 | 892.04 | 29.22 |
| `refitUnion--compact_keep--guarded_trim` | source-group-007 | 54.46 | 96.77 | 69.70 | 1815.91 | 826.98 | 20.11 |
| `refitUnion--compact_keep--guarded_trim` | source-group-009 | 80.45 | 98.72 | 88.65 | 1249.49 | 244.30 | 9.52 |
| `refitUnion--compact_keep--guarded_trim` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--compact_keep--guarded_component_rejection` | source-group-005 | 49.07 | 95.70 | 64.87 | 1757.89 | 895.31 | 25.08 |
| `refitUnion--compact_keep--guarded_component_rejection` | source-group-007 | 54.38 | 96.77 | 69.63 | 1818.70 | 829.76 | 20.11 |
| `refitUnion--compact_keep--guarded_component_rejection` | source-group-009 | 80.45 | 98.72 | 88.65 | 1249.49 | 244.30 | 9.52 |
| `refitUnion--compact_keep--guarded_component_rejection` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--dino_global--union` | source-group-005 | 47.84 | 99.51 | 64.61 | 1888.04 | 984.84 | 2.85 |
| `refitUnion--dino_global--union` | source-group-007 | 52.28 | 99.58 | 68.56 | 1951.83 | 931.47 | 2.62 |
| `refitUnion--dino_global--union` | source-group-009 | 76.59 | 99.98 | 86.74 | 1345.05 | 314.86 | 0.16 |
| `refitUnion--dino_global--union` | source-group-012 | 68.93 | 100.00 | 81.60 | 956.77 | 297.32 | 0.00 |
| `refitUnion--dino_global--intersection` | source-group-005 | 86.98 | 93.58 | 89.98 | 908.50 | 124.35 | 37.44 |
| `refitUnion--dino_global--intersection` | source-group-007 | 86.40 | 91.43 | 88.83 | 991.70 | 135.63 | 53.33 |
| `refitUnion--dino_global--intersection` | source-group-009 | 98.66 | 98.13 | 98.39 | 966.92 | 13.05 | 13.91 |
| `refitUnion--dino_global--intersection` | source-group-012 | 89.75 | 98.06 | 93.71 | 705.56 | 72.48 | 8.53 |
| `refitUnion--dino_global--tolerant_intersection` | source-group-005 | 74.49 | 95.04 | 83.39 | 1136.57 | 294.73 | 28.91 |
| `refitUnion--dino_global--tolerant_intersection` | source-group-007 | 74.98 | 93.52 | 83.21 | 1236.19 | 310.24 | 40.36 |
| `refitUnion--dino_global--tolerant_intersection` | source-group-009 | 92.51 | 98.49 | 95.40 | 1079.38 | 80.89 | 11.24 |
| `refitUnion--dino_global--tolerant_intersection` | source-group-012 | 80.82 | 98.43 | 88.76 | 794.79 | 152.52 | 6.88 |
| `refitUnion--dino_global--guarded_trim` | source-group-005 | 49.62 | 97.89 | 65.86 | 1780.41 | 896.98 | 12.30 |
| `refitUnion--dino_global--guarded_trim` | source-group-007 | 54.51 | 97.24 | 69.86 | 1824.90 | 830.18 | 17.17 |
| `refitUnion--dino_global--guarded_trim` | source-group-009 | 80.42 | 98.83 | 88.68 | 1254.33 | 245.65 | 8.68 |
| `refitUnion--dino_global--guarded_trim` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--dino_global--guarded_component_rejection` | source-group-005 | 49.53 | 97.89 | 65.78 | 1783.71 | 900.28 | 12.30 |
| `refitUnion--dino_global--guarded_component_rejection` | source-group-007 | 54.46 | 97.24 | 69.82 | 1826.60 | 831.87 | 17.17 |
| `refitUnion--dino_global--guarded_component_rejection` | source-group-009 | 80.42 | 98.83 | 88.68 | 1254.33 | 245.65 | 8.68 |
| `refitUnion--dino_global--guarded_component_rejection` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--dino_boost--union` | source-group-005 | 47.72 | 99.84 | 64.57 | 1907.42 | 997.20 | 0.96 |
| `refitUnion--dino_boost--union` | source-group-007 | 52.44 | 99.70 | 68.73 | 1944.91 | 924.98 | 1.86 |
| `refitUnion--dino_boost--union` | source-group-009 | 76.86 | 99.98 | 86.91 | 1336.86 | 309.33 | 0.16 |
| `refitUnion--dino_boost--union` | source-group-012 | 68.75 | 100.00 | 81.48 | 959.02 | 299.78 | 0.00 |
| `refitUnion--dino_boost--intersection` | source-group-005 | 80.08 | 95.33 | 86.96 | 1024.07 | 207.54 | 27.23 |
| `refitUnion--dino_boost--intersection` | source-group-007 | 90.74 | 88.88 | 89.73 | 900.84 | 84.19 | 69.21 |
| `refitUnion--dino_boost--intersection` | source-group-009 | 98.83 | 98.05 | 98.44 | 953.47 | 11.30 | 14.47 |
| `refitUnion--dino_boost--intersection` | source-group-012 | 89.27 | 98.25 | 93.52 | 711.54 | 76.89 | 7.70 |
| `refitUnion--dino_boost--tolerant_intersection` | source-group-005 | 69.08 | 96.83 | 80.58 | 1251.51 | 389.23 | 18.47 |
| `refitUnion--dino_boost--tolerant_intersection` | source-group-007 | 79.34 | 92.02 | 85.16 | 1150.23 | 238.65 | 49.72 |
| `refitUnion--dino_boost--tolerant_intersection` | source-group-009 | 92.18 | 98.54 | 95.25 | 1084.01 | 84.88 | 10.82 |
| `refitUnion--dino_boost--tolerant_intersection` | source-group-012 | 80.74 | 98.43 | 88.71 | 795.73 | 153.46 | 6.88 |
| `refitUnion--dino_boost--guarded_trim` | source-group-005 | 49.69 | 98.28 | 66.00 | 1786.94 | 899.06 | 10.04 |
| `refitUnion--dino_boost--guarded_trim` | source-group-007 | 54.53 | 97.16 | 69.85 | 1820.82 | 827.95 | 17.67 |
| `refitUnion--dino_boost--guarded_trim` | source-group-009 | 80.43 | 98.77 | 88.66 | 1251.91 | 244.98 | 9.10 |
| `refitUnion--dino_boost--guarded_trim` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--dino_boost--guarded_component_rejection` | source-group-005 | 49.57 | 98.28 | 65.90 | 1791.08 | 903.20 | 10.04 |
| `refitUnion--dino_boost--guarded_component_rejection` | source-group-007 | 54.44 | 97.16 | 69.78 | 1823.85 | 830.94 | 17.67 |
| `refitUnion--dino_boost--guarded_component_rejection` | source-group-009 | 80.43 | 98.77 | 88.66 | 1251.91 | 244.98 | 9.10 |
| `refitUnion--dino_boost--guarded_component_rejection` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--dino_keep--union` | source-group-005 | 47.55 | 99.76 | 64.40 | 1911.43 | 1002.66 | 1.42 |
| `refitUnion--dino_keep--union` | source-group-007 | 52.38 | 99.68 | 68.67 | 1947.74 | 927.59 | 1.96 |
| `refitUnion--dino_keep--union` | source-group-009 | 76.82 | 99.77 | 86.80 | 1331.10 | 308.60 | 1.71 |
| `refitUnion--dino_keep--union` | source-group-012 | 69.01 | 100.00 | 81.66 | 955.28 | 296.05 | 0.00 |
| `refitUnion--dino_keep--intersection` | source-group-005 | 78.57 | 95.06 | 85.74 | 1055.34 | 237.42 | 28.82 |
| `refitUnion--dino_keep--intersection` | source-group-007 | 89.91 | 91.60 | 90.68 | 955.06 | 98.44 | 52.29 |
| `refitUnion--dino_keep--intersection` | source-group-009 | 99.61 | 95.40 | 97.45 | 890.76 | 3.51 | 34.18 |
| `refitUnion--dino_keep--intersection` | source-group-012 | 90.67 | 97.87 | 94.13 | 697.21 | 65.31 | 9.33 |
| `refitUnion--dino_keep--tolerant_intersection` | source-group-005 | 68.27 | 95.45 | 79.41 | 1256.18 | 405.78 | 26.50 |
| `refitUnion--dino_keep--tolerant_intersection` | source-group-007 | 78.24 | 93.94 | 85.31 | 1202.49 | 263.90 | 37.76 |
| `refitUnion--dino_keep--tolerant_intersection` | source-group-009 | 93.83 | 97.43 | 95.59 | 1038.19 | 64.09 | 19.13 |
| `refitUnion--dino_keep--tolerant_intersection` | source-group-012 | 81.85 | 98.43 | 89.37 | 784.96 | 142.68 | 6.88 |
| `refitUnion--dino_keep--guarded_trim` | source-group-005 | 49.56 | 97.93 | 65.82 | 1784.55 | 900.06 | 12.05 |
| `refitUnion--dino_keep--guarded_trim` | source-group-007 | 54.56 | 97.25 | 69.90 | 1823.45 | 828.64 | 17.10 |
| `refitUnion--dino_keep--guarded_trim` | source-group-009 | 80.46 | 98.66 | 88.64 | 1247.07 | 243.63 | 9.93 |
| `refitUnion--dino_keep--guarded_trim` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--dino_keep--guarded_component_rejection` | source-group-005 | 49.54 | 97.93 | 65.80 | 1785.42 | 900.93 | 12.05 |
| `refitUnion--dino_keep--guarded_component_rejection` | source-group-007 | 54.49 | 97.25 | 69.85 | 1825.60 | 830.78 | 17.10 |
| `refitUnion--dino_keep--guarded_component_rejection` | source-group-009 | 80.46 | 98.66 | 88.64 | 1247.07 | 243.63 | 9.93 |
| `refitUnion--dino_keep--guarded_component_rejection` | source-group-012 | 70.51 | 98.43 | 82.16 | 911.17 | 268.72 | 6.88 |
| `refitUnion--best_f1_pair--three_union` | source-group-005 | 47.84 | 99.60 | 64.63 | 1894.15 | 988.03 | 2.35 |
| `refitUnion--best_f1_pair--three_union` | source-group-007 | 52.04 | 99.60 | 68.36 | 1962.47 | 941.14 | 2.50 |
| `refitUnion--best_f1_pair--three_union` | source-group-009 | 76.48 | 100.00 | 86.67 | 1350.06 | 317.52 | 0.00 |
| `refitUnion--best_f1_pair--three_union` | source-group-012 | 68.92 | 100.00 | 81.60 | 957.90 | 297.72 | 0.00 |
| `refitUnion--best_f1_pair--majority` | source-group-005 | 79.99 | 96.74 | 87.52 | 1045.16 | 211.06 | 19.00 |
| `refitUnion--best_f1_pair--majority` | source-group-007 | 81.01 | 94.19 | 87.08 | 1115.70 | 212.76 | 36.19 |
| `refitUnion--best_f1_pair--majority` | source-group-009 | 97.52 | 99.86 | 98.67 | 1026.89 | 25.50 | 1.04 |
| `refitUnion--best_f1_pair--majority` | source-group-012 | 88.16 | 99.61 | 93.53 | 736.51 | 87.29 | 1.73 |
| `refitUnion--recovery_pair--three_union` | source-group-005 | 47.64 | 99.84 | 64.50 | 1912.59 | 1001.43 | 0.96 |
| `refitUnion--recovery_pair--three_union` | source-group-007 | 52.10 | 99.72 | 68.45 | 1960.89 | 939.19 | 1.73 |
| `refitUnion--recovery_pair--three_union` | source-group-009 | 76.67 | 100.00 | 86.79 | 1344.56 | 313.72 | 0.00 |
| `refitUnion--recovery_pair--three_union` | source-group-012 | 68.73 | 100.00 | 81.46 | 960.28 | 300.38 | 0.00 |
| `refitUnion--recovery_pair--majority` | source-group-005 | 74.78 | 97.04 | 84.46 | 1125.19 | 284.52 | 17.26 |
| `refitUnion--recovery_pair--majority` | source-group-007 | 81.37 | 94.55 | 87.39 | 1118.16 | 210.65 | 33.95 |
| `refitUnion--recovery_pair--majority` | source-group-009 | 97.39 | 99.82 | 98.59 | 1021.68 | 26.75 | 1.31 |
| `refitUnion--recovery_pair--majority` | source-group-012 | 87.70 | 99.63 | 93.27 | 741.80 | 91.59 | 1.64 |

## Evidence

Contract SHA-256: `c0656f85658e6eade8f4e8feacbeb88453186c3f1c3b7d38ab332eb105f47b27`.

NAS run root: `private-reference-0083`.

Full machine-readable summary: `private-reference-0166` (SHA-256 `e777cbae2f8cb5d8942e561e4678e292f844b816901ad69c9afb0c1f4a8b8007`).
Full immutable result index: `private-reference-0167` (SHA-256 `556f44e55b56425ddd0693cfe7f997819fd68365aa29a18653296699062462e4`).

Every result includes an independent endpoint-sweep duration check on pooled, four group and eight individual-recording scopes, at all four paddings. The separate recipe auditor reconstructs all 2,424 automatic recording outputs, 2,880 review recording/padding rows and 128 app-export overrides without shared interval/model/application imports. Qualification passed 44 focused tests. Registered source/input hashes were checked before and after execution. The final source archive and manifest are stored alongside the run.
