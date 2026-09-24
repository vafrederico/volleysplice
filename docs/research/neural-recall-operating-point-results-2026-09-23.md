# Strict 99% recall operating-point experiment - 2026-09-23

Raising the inner recall floor from 95% to 99% substantially recovers play for the compact AV TCN, with more unwanted footage and a lower primary F1. The AV TCN is the only tested family with a feasible selection for every source fold and seed. None of the fully evaluated neural seeds reaches 99% recall on held sources at the target two-second padding. This is a development comparison, not a production promotion.

The complete DINO seed retains 98.84% of core play with 86.09% export precision, but two other DINO seeds have infeasible folds. Frozen Mobile TCN has one infeasible fold and materially lower held recall in its two complete seeds. Neither family receives a three-seed mean from only its favorable subset.

## Frozen experiment and interpretation

- Same eight exact-label recordings, four source groups, 322 original rallies; beach is excluded. The evaluation universe after ignored intervals is 8270.032 seconds, with 2387.151 seconds of human core play.
- Nested source-held development evaluation. The outer source group is excluded from training, scaler fitting and inner selection. Inner scores hold out an additional source group. Existing source-group rules also exclude related auxiliary recordings. No protected test split is opened.
- Primary selection: the original 48 decoders at epochs 5, 15, 30 and 60, giving 192 candidates per fold. Require pooled inner `R_core >= 0.99`; among eligible candidates maximize `F1_padP_coreR`. Preserve original candidate order for ties. No eligible candidate means an explicit infeasible fold, with no relaxed fallback.
- The decoder grid remains unchanged: smoothing 0.5/1 second, entry 0.20/0.35/0.50/0.65/0.80/0.90, exit 0.10 below entry, minimum duration 0.25/1 second, boundary adjustment on/off. The frozen bridge, short-rally and boundary settings are unchanged. This experiment does not test lower thresholds or a broader grid.
- Target product padding is two seconds before and after. Apply identical padding to model and human ranges, merge overlap/touch, then retain positive gaps strictly below three seconds. A three-second gap remains a cut. Subtract ignored intervals afterward without rejoining across them. Report all four symmetric padding cases; do not select padding using held results.
- `P_pad` measures exported time within padded human exports; `R_core` measures core human play retained by the padded model export; `F1_padP_coreR` is their harmonic mean. Pool duration numerators/denominators across recordings within each seed, then average complete seed metrics. Production is a fixed descriptive comparator with historical label exposure.
- This is operating-point selection, not probability calibration. It changes checkpoint/decoder selection, not the existing architecture or loss. Sixteen missing selected outer checkpoints were refitted with their original numerical recipe, seed and source exclusions. Every refit exactly reproduced its saved old-epoch weights, probabilities and numerical history.

## Feasibility and per-seed held results

All P/R/F1 triplets below use the declared two-second padding. A full-scope result needs all four feasible outer folds. A dash means no full-scope metric, not zero recall. The 99% floor applies to inner selection and is not a guarantee for a new source. Source abbreviations: source-group-005 = `source-group-005`, source-group-007 = `source-group-007`, indoor source = `source-group-009`, source-group-012 = `source-group-012`.

| Model | Seed | Joint feasible | Fixed-epoch feasible | Old95 P / R / F1 | Strict99 P / R / F1 | Infeasible held source: maximum inner R |
|---|---|---|---|---|---|---|
| AV TCN | 3407 | 4/4 | 0/4 | 89.35% / 92.30% / 90.80% | 75.52% / 98.41% / 85.46% | None |
| AV TCN | 1729 | 4/4 | 4/4 | 88.14% / 92.21% / 90.13% | 77.62% / 98.27% / 86.73% | None |
| AV TCN | 20260918 | 4/4 | 2/4 | 87.43% / 92.72% / 89.99% | 80.31% / 97.20% / 87.95% | None |
| DINO TCN | 3407 | 3/4 | 0/4 | 86.97% / 96.21% / 91.36% | - | source-group-012 98.631% |
| DINO TCN | 1729 | 3/4 | 3/4 | 90.45% / 96.57% / 93.41% | - | source-group-007 98.832% |
| DINO TCN | 20260918 | 4/4 | 4/4 | 88.00% / 98.08% / 92.77% | 86.09% / 98.84% / 92.03% | None |
| AV transformer | 3407 | 0/4 | 0/4 | 72.28% / 92.12% / 81.00% | - | source-group-005 96.798%; source-group-007 98.081%; indoor source 95.617%; source-group-012 97.512% |
| AV transformer | 1729 | 0/4 | 0/4 | 64.63% / 95.05% / 76.94% | - | source-group-005 96.697%; source-group-007 97.279%; indoor source 96.321%; source-group-012 96.631% |
| AV transformer | 20260918 | 0/4 | 0/4 | 65.20% / 93.71% / 76.89% | - | source-group-005 97.113%; source-group-007 97.090%; indoor source 97.332%; source-group-012 95.744% |
| DINO transformer | 3407 | 0/4 | 0/4 | 82.46% / 94.87% / 88.23% | - | source-group-005 98.488%; source-group-007 98.425%; indoor source 96.716%; source-group-012 98.124% |
| DINO transformer | 1729 | 1/4 | 1/4 | 82.64% / 94.38% / 88.12% | - | source-group-005 97.277%; indoor source 96.074%; source-group-012 98.777% |
| DINO transformer | 20260918 | 0/4 | 0/4 | 80.97% / 92.20% / 86.22% | - | source-group-005 97.476%; source-group-007 97.767%; indoor source 98.366%; source-group-012 98.989% |
| Frozen Mobile TCN | 3407 | 4/4 | 4/4 | 92.32% / 90.57% / 91.44% | 85.79% / 96.00% / 90.61% | None |
| Frozen Mobile TCN | 1729 | 4/4 | 2/4 | 90.32% / 88.15% / 89.22% | 83.33% / 96.93% / 89.62% | None |
| Frozen Mobile TCN | 20260918 | 3/4 | 2/4 | 87.06% / 93.53% / 90.18% | - | source-group-012 98.269% |

Overall feasible-fold counts are AV TCN 12/12, DINO TCN 10/12, AV transformer 0/12, DINO transformer 1/12, and frozen Mobile TCN 11/12. The transformer result applies to these registered configurations and this fixed search space, not all possible transformers.

## Full-scope results and duration accounting

Rows marked mean use all three seeds; single-seed rows are descriptive and are not a replacement model ranking population. All times are seconds over the same evaluated video universe. Correct removal is non-human-export time excluded by the model. Incorrect export is retained time outside the padded human export. Wanted export omitted is missing padded-human footage; missed core is the smaller subset of omitted actual play. At two seconds the padded human export is 3675.151 seconds.

| Model / scope | P_pad | R_core | F1_padP_coreR | Export | Export minus human | Correct removal | Incorrect export | Wanted omitted | Missed core |
|---|---|---|---|---|---|---|---|---|---|
| Production | 72.45% | 99.27% | 83.76% | 4943.03 | 1267.87 | 3232.93 | 1361.95 | 94.08 | 17.47 |
| AV TCN old95 mean | 88.30% | 92.41% | 90.31% | 3543.47 | -131.68 | 4180.06 | 414.82 | 546.51 | 181.20 |
| AV TCN strict99 mean | 77.82% | 97.96% | 86.71% | 4437.94 | 762.79 | 3607.53 | 987.35 | 224.57 | 48.70 |
| AV TCN strict99 3407 | 75.52% | 98.41% | 85.46% | 4596.47 | 921.32 | 3469.85 | 1125.03 | 203.72 | 38.02 |
| AV TCN strict99 1729 | 77.62% | 98.27% | 86.73% | 4478.67 | 803.52 | 3592.43 | 1002.45 | 198.94 | 41.19 |
| AV TCN strict99 20260918 | 80.31% | 97.20% | 87.95% | 4238.68 | 563.52 | 3760.30 | 834.58 | 271.05 | 66.89 |
| DINO TCN old95 mean | 88.47% | 96.95% | 92.51% | 3739.83 | 64.68 | 4162.69 | 432.19 | 367.51 | 72.70 |
| DINO TCN strict99 20260918 | 86.09% | 98.84% | 92.03% | 4022.35 | 347.20 | 4035.31 | 559.57 | 212.37 | 27.66 |
| AV transformer old95 mean | 67.37% | 93.63% | 78.28% | 4849.02 | 1173.87 | 3001.32 | 1593.56 | 419.69 | 152.16 |
| DINO transformer old95 mean | 82.02% | 93.82% | 87.53% | 3887.77 | 212.62 | 3896.13 | 698.75 | 486.14 | 147.54 |
| Frozen Mobile TCN old95 mean | 89.90% | 90.75% | 90.28% | 3423.30 | -251.85 | 4244.82 | 350.07 | 601.92 | 220.83 |
| Frozen Mobile TCN strict99 3407 | 85.79% | 96.00% | 90.61% | 3903.30 | 228.15 | 4040.39 | 554.49 | 326.34 | 95.48 |
| Frozen Mobile TCN strict99 1729 | 83.33% | 96.93% | 89.62% | 4061.18 | 386.02 | 3917.92 | 676.96 | 290.94 | 73.40 |

Accounting identities: `export - human_export = incorrect_export - wanted_export_omitted`; `correct_removal = evaluable_universe - human_export - incorrect_export`. Ignored time is in neither side of these accounts.

## Event, original-rally and boundary guardrails

Event P/R/F1 uses one-to-one rally matching at IoU >= 0.5 on unpadded decoded intervals; original rallies touched by ignored time are excluded from event diagnostics. Complete loss means no evaluable core retained after product padding; partial loss means some but not all core retained. These two counts preserve original rally identities. Short means original duration <= 3 seconds, long means > 3 seconds. Boundary MAE is conditional on matched, uncensored events and does not measure boundaries of missed rallies.

| Model / scope | Event P / R / F1 | Complete loss | Partial loss | Short complete loss | Long core R | Start / end MAE (s) |
|---|---|---|---|---|---|---|
| Production | 62.64% / 70.81% / 66.47% | 3 | 7 | 3 | 99.48% | 0.71 / 1.54 |
| AV TCN old95 mean | 69.23% / 69.67% / 69.45% | 33 | 42 | 21.67 | 93.82% | 0.44 / 1.15 |
| AV TCN strict99 mean | 61.19% / 70.39% / 65.45% | 9.67 | 19.67 | 7.33 | 98.42% | 0.57 / 1.37 |
| AV TCN strict99 3407 | 56.01% / 68.01% / 61.43% | 10 | 17 | 8 | 98.95% | 0.66 / 1.56 |
| AV TCN strict99 1729 | 64.89% / 71.74% / 68.14% | 7 | 18 | 5 | 98.59% | 0.49 / 1.25 |
| AV TCN strict99 20260918 | 62.67% / 71.43% / 66.76% | 12 | 24 | 9 | 97.73% | 0.57 / 1.29 |
| DINO TCN old95 mean | 71.71% / 76.81% / 74.17% | 13.67 | 29.33 | 11 | 97.65% | 0.52 / 1.12 |
| DINO TCN strict99 20260918 | 76.02% / 80.75% / 78.31% | 5 | 14 | 4 | 99.13% | 0.45 / 1.10 |
| AV transformer old95 mean | 46.26% / 52.80% / 49.29% | 38 | 22.33 | 25.33 | 95.44% | 0.83 / 2.10 |
| DINO transformer old95 mean | 61.10% / 63.77% / 62.37% | 34.67 | 29 | 23 | 95.28% | 0.70 / 1.34 |
| Frozen Mobile TCN old95 mean | 68.69% / 69.36% / 68.93% | 32.67 | 47.00 | 21 | 91.91% | 0.49 / 1.06 |
| Frozen Mobile TCN strict99 3407 | 74.71% / 79.81% / 77.18% | 14 | 33 | 12 | 96.74% | 0.46 / 1.16 |
| Frozen Mobile TCN strict99 1729 | 70.03% / 75.47% / 72.65% | 12 | 32 | 12 | 97.69% | 0.46 / 1.39 |

## Fixed-old-epoch sensitivity

This secondary analysis keeps the previously selected 95%-floor epoch and searches only the same 48 decoders at the 99% floor. It does not replace the joint primary experiment. Only the following three seed cells have full source scope; every other cell is infeasible in at least one fold. No family has a three-seed mean.

| Model / seed | P / R / F1 | Event P / R / F1 | Complete / partial losses | Export (s) | Incorrect export (s) | Missed core (s) |
|---|---|---|---|---|---|---|
| AV TCN fixed99 1729 | 78.22% / 98.84% / 87.33% | 66.29% / 72.67% / 69.33% | 8 / 17 | 4454.74 | 970.45 | 27.74 |
| DINO TCN fixed99 20260918 | 84.52% / 98.84% / 91.12% | 72.99% / 78.88% / 75.82% | 5 / 14 | 4100.86 | 634.90 | 27.66 |
| Frozen Mobile TCN fixed99 3407 | 85.36% / 96.37% / 90.53% | 74.49% / 78.88% / 76.62% | 16 / 26 | 3940.62 | 577.08 | 86.72 |

## Padding sensitivity

Each row group retains its selected two-second operating point. Padding alone changes for this required sensitivity. All rows use a strict gap threshold of three seconds. Human export and duration differences are shown explicitly; means are averages of the three complete seed-level pooled results.

| Model / scope | Padding each side (s) | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---|---|---|---|---|---|---|
| Production | 0 | 65.50% | 95.75% | 77.79% | 3489.61 | 2387.15 | 1102.46 |
| Production | 1 | 69.40% | 98.30% | 81.36% | 4223.51 | 3031.15 | 1192.36 |
| Production | 2 | 72.45% | 99.27% | 83.76% | 4943.03 | 3675.15 | 1267.87 |
| Production | 3 | 75.02% | 99.49% | 85.54% | 5637.20 | 4323.26 | 1313.93 |
| AV TCN old95 mean | 0 | 84.24% | 82.00% | 83.10% | 2324.05 | 2387.15 | -63.10 |
| AV TCN old95 mean | 1 | 86.47% | 89.60% | 88.00% | 2943.53 | 3031.15 | -87.62 |
| AV TCN old95 mean | 2 | 88.30% | 92.41% | 90.31% | 3543.47 | 3675.15 | -131.68 |
| AV TCN old95 mean | 3 | 89.71% | 93.88% | 91.75% | 4145.11 | 4323.26 | -178.15 |
| AV TCN strict99 mean | 0 | 71.23% | 91.32% | 80.01% | 3063.41 | 2387.15 | 676.26 |
| AV TCN strict99 mean | 1 | 74.76% | 96.42% | 84.20% | 3770.32 | 3031.15 | 739.17 |
| AV TCN strict99 mean | 2 | 77.82% | 97.96% | 86.71% | 4437.94 | 3675.15 | 762.79 |
| AV TCN strict99 mean | 3 | 80.20% | 98.51% | 88.40% | 5094.59 | 4323.26 | 771.33 |
| AV TCN strict99 3407 | 0 | 69.07% | 91.55% | 78.74% | 3163.93 | 2387.15 | 776.78 |
| AV TCN strict99 3407 | 1 | 72.55% | 96.98% | 83.01% | 3905.68 | 3031.15 | 874.53 |
| AV TCN strict99 3407 | 2 | 75.52% | 98.41% | 85.46% | 4596.47 | 3675.15 | 921.32 |
| AV TCN strict99 3407 | 3 | 77.90% | 98.80% | 87.11% | 5262.61 | 4323.26 | 939.35 |
| AV TCN strict99 1729 | 0 | 70.71% | 92.37% | 80.10% | 3118.26 | 2387.15 | 731.11 |
| AV TCN strict99 1729 | 1 | 74.48% | 96.90% | 84.22% | 3810.56 | 3031.15 | 779.41 |
| AV TCN strict99 1729 | 2 | 77.62% | 98.27% | 86.73% | 4478.67 | 3675.15 | 803.52 |
| AV TCN strict99 1729 | 3 | 80.04% | 98.72% | 88.41% | 5137.46 | 4323.26 | 814.20 |
| AV TCN strict99 20260918 | 0 | 73.92% | 90.05% | 81.19% | 2908.05 | 2387.15 | 520.90 |
| AV TCN strict99 20260918 | 1 | 77.26% | 95.38% | 85.37% | 3594.72 | 3031.15 | 563.57 |
| AV TCN strict99 20260918 | 2 | 80.31% | 97.20% | 87.95% | 4238.68 | 3675.15 | 563.52 |
| AV TCN strict99 20260918 | 3 | 82.66% | 98.02% | 89.69% | 4883.71 | 4323.26 | 560.45 |
| DINO TCN old95 mean | 0 | 84.34% | 86.22% | 85.25% | 2441.89 | 2387.15 | 54.74 |
| DINO TCN old95 mean | 1 | 86.69% | 94.63% | 90.48% | 3101.64 | 3031.15 | 70.49 |
| DINO TCN old95 mean | 2 | 88.47% | 96.95% | 92.51% | 3739.83 | 3675.15 | 64.68 |
| DINO TCN old95 mean | 3 | 90.08% | 97.84% | 93.79% | 4364.46 | 4323.26 | 41.19 |
| DINO TCN strict99 20260918 | 0 | 81.14% | 91.61% | 86.05% | 2695.15 | 2387.15 | 308.00 |
| DINO TCN strict99 20260918 | 1 | 83.91% | 97.71% | 90.29% | 3370.37 | 3031.15 | 339.22 |
| DINO TCN strict99 20260918 | 2 | 86.09% | 98.84% | 92.03% | 4022.35 | 3675.15 | 347.20 |
| DINO TCN strict99 20260918 | 3 | 87.57% | 99.28% | 93.06% | 4685.87 | 4323.26 | 362.60 |
| AV transformer old95 mean | 0 | 57.79% | 88.30% | 69.69% | 3674.19 | 2387.15 | 1287.04 |
| AV transformer old95 mean | 1 | 62.62% | 92.15% | 74.45% | 4303.61 | 3031.15 | 1272.46 |
| AV transformer old95 mean | 2 | 67.37% | 93.63% | 78.28% | 4849.02 | 3675.15 | 1173.87 |
| AV transformer old95 mean | 3 | 71.91% | 94.44% | 81.60% | 5356.74 | 4323.26 | 1033.48 |
| DINO transformer old95 mean | 0 | 75.51% | 84.95% | 79.95% | 2685.81 | 2387.15 | 298.66 |
| DINO transformer old95 mean | 1 | 79.20% | 91.56% | 84.93% | 3296.43 | 3031.15 | 265.28 |
| DINO transformer old95 mean | 2 | 82.02% | 93.82% | 87.53% | 3887.77 | 3675.15 | 212.62 |
| DINO transformer old95 mean | 3 | 84.41% | 94.90% | 89.35% | 4472.80 | 4323.26 | 149.54 |
| Frozen Mobile TCN old95 mean | 0 | 86.43% | 79.45% | 82.67% | 2199.24 | 2387.15 | -187.91 |
| Frozen Mobile TCN old95 mean | 1 | 88.49% | 87.54% | 87.94% | 2815.64 | 3031.15 | -215.51 |
| Frozen Mobile TCN old95 mean | 2 | 89.90% | 90.75% | 90.28% | 3423.30 | 3675.15 | -251.85 |
| Frozen Mobile TCN old95 mean | 3 | 91.01% | 92.46% | 91.70% | 4027.49 | 4323.26 | -295.78 |
| Frozen Mobile TCN strict99 3407 | 0 | 81.96% | 88.07% | 84.91% | 2565.23 | 2387.15 | 178.07 |
| Frozen Mobile TCN strict99 3407 | 1 | 84.11% | 93.87% | 88.72% | 3237.08 | 3031.15 | 205.93 |
| Frozen Mobile TCN strict99 3407 | 2 | 85.79% | 96.00% | 90.61% | 3903.30 | 3675.15 | 228.15 |
| Frozen Mobile TCN strict99 3407 | 3 | 87.16% | 97.11% | 91.87% | 4567.48 | 4323.26 | 244.22 |
| Frozen Mobile TCN strict99 1729 | 0 | 78.25% | 89.09% | 83.32% | 2717.82 | 2387.15 | 330.67 |
| Frozen Mobile TCN strict99 1729 | 1 | 81.15% | 95.01% | 87.53% | 3402.89 | 3031.15 | 371.74 |
| Frozen Mobile TCN strict99 1729 | 2 | 83.33% | 96.93% | 89.62% | 4061.18 | 3675.15 | 386.02 |
| Frozen Mobile TCN strict99 1729 | 3 | 84.69% | 97.90% | 90.82% | 4747.64 | 4323.26 | 424.38 |
| AV TCN fixed99 1729 | 0 | 71.51% | 92.71% | 80.74% | 3094.55 | 2387.15 | 707.40 |
| AV TCN fixed99 1729 | 1 | 75.39% | 97.51% | 85.03% | 3777.47 | 3031.15 | 746.32 |
| AV TCN fixed99 1729 | 2 | 78.22% | 98.84% | 87.33% | 4454.74 | 3675.15 | 779.59 |
| AV TCN fixed99 1729 | 3 | 80.62% | 99.22% | 88.96% | 5108.35 | 4323.26 | 785.09 |
| DINO TCN fixed99 20260918 | 0 | 79.22% | 91.76% | 85.03% | 2765.12 | 2387.15 | 377.97 |
| DINO TCN fixed99 20260918 | 1 | 81.96% | 97.71% | 89.15% | 3455.12 | 3031.15 | 423.97 |
| DINO TCN fixed99 20260918 | 2 | 84.52% | 98.84% | 91.12% | 4100.86 | 3675.15 | 425.71 |
| DINO TCN fixed99 20260918 | 3 | 85.93% | 99.28% | 92.12% | 4779.20 | 4323.26 | 455.94 |
| Frozen Mobile TCN fixed99 3407 | 0 | 80.92% | 88.97% | 84.76% | 2624.78 | 2387.15 | 237.62 |
| Frozen Mobile TCN fixed99 3407 | 1 | 83.41% | 94.40% | 88.57% | 3285.02 | 3031.15 | 253.87 |
| Frozen Mobile TCN fixed99 3407 | 2 | 85.36% | 96.37% | 90.53% | 3940.62 | 3675.15 | 265.47 |
| Frozen Mobile TCN fixed99 3407 | 3 | 86.78% | 97.26% | 91.72% | 4600.24 | 4323.26 | 276.98 |

## Conclusions and limitations

- AV TCN: mean core recall rises 5.55 percentage points, from 92.41% to 97.96%; complete rally losses fall from 33 to 9.67 and missed play from 181.20 to 48.70 seconds. The cost is precision falling 10.49 points and F1 falling 3.59 points, with 572.53 more seconds of unwanted export and 894.47 more seconds of total export. This demonstrates a useful recall/cleanup trade-off, not an improvement in the declared primary ranking metric.
- Production still retains more core play (99.27%) and completely loses fewer rallies (3) than any fully evaluated strict99 neural seed. AV strict99 improves export precision relative to production, but does not meet the desired 99% held-play bar. Historical production label exposure prevents treating this as a clean held-training comparison.
- DINO seed 20260918 is promising at 98.84% core recall, 92.03% F1 and five complete losses. It must remain a single-seed observation: two other DINO seeds cannot satisfy the frozen inner constraint in one source fold each. Frozen Mobile TCN is similarly incomplete and retains only 96.00% and 96.93% of core play in its complete seeds.
- Export recall is not rally separation accuracy. AV strict99 event F1 falls from 69.45% to 65.45% despite recovering more play. Preserve event and original-rally guardrails for score tracking; broad retention by itself does not establish reliable serve/rally boundaries.
- The source count is only four, related recordings stay grouped, and the three seeds do not provide independent new matches. No confidence interval or unseen-phone performance claim is inferred. This study evaluates existing features and checkpoint/decoder choices; it does not establish gains from a new detector, encoder, distillation or mobile runtime.
- Future broader decoder searches or explicit review/composite policies need a new frozen experiment. Infeasibility here is limited to the unchanged grid; it is not proof that no lower-threshold policy could attain the inner floor. This experiment did not silently substitute production or relax the floor.

## Validation and artifact identities

Five focused selection/checkpoint-status tests passed. The first independent audit passed all 60 outer selections and 11,520 candidate metric computations, reproduced the original 95% selections and saved predictions, and independently checked available held interval metrics. The final audit passed all 16 supplemental fits, validating exact original weights/probabilities/history, artifact and source memberships, decoded intervals, original-rally coverage and all four padding metrics.

The supplemental audit validates new checkpoint score hashes, shapes, finite values, interval decoding and independent metric arithmetic; it does not perform a second independent neural forward pass for every newly selected checkpoint. Precision replay has its own additional DINO checkpoint-forward validation. No protected test or production model was changed.

One refit worker ran with the frozen torch thread count. Its process affinity was increased from two to four logical CPUs under an approved resource adjustment. Brief parent engineering and sporadic low-memory FP16 work were allowed to overlap, so this report makes no exclusive GPU timing claim. Generated checkpoints, predictions, logs, caches and temporary report-generation data were written to the NAS.

NAS root: `private-reference-0171` (WSL `private-reference-0172`). Registered source snapshots and all original sources remain available through the protocol/reference manifests.

| Artifact | SHA-256 |
|---|---|
| report-complete.json | `7443fa8e5409a84c2f1708ebaaef13b0333f0e850403a55608c70d6a5e28237f` |
| audit-refits.json | `f04015c21ed010a61bdf41dc70e6627d50804d0d5a17ba3c4fc5354780bdb4df` |
| audit-selection.json | `42fba53d9d5ed66f49b35e6801069e023e172bba6395219bd21ba7a55ed1546b` |
| summary-complete.json | `820ea0c28ba8e78b6e6892adc329d04917d4a780f36920e0cb5fb155fdc4aae4` |
| protocol.json | `d851be8ce83f2931b3cdc1f74a4994c1431aff48b2b46a78129eb94956e6a493` |

The parent [combined protocol](neural-recall-distillation-protocol-2026-09-23.md) is bound to NAS `../protocol.md`, SHA-256 `1b2c764fde6553e0f19ac3bf6379aa02b9a12f1a19dec86fe61ab934a888eea4`. The final JSON preserves every feasible held-fold result and each infeasible fold; this repository note reports full-scope comparisons. `summary-complete.json` is descriptive and binds the audited final report.
