# DINO-distilled MobileNetV3-Large + TCN

The student uses four 960-value regional MobileNetV3-Large embeddings at 224px/2Hz, AV104 features and eight quality/age/availability values. The FP32 TCN consumes 3,952 values per tick. Cached frozen DINO features supervise student training; DINO and the training projection are absent at inference.

15 of 24 registered fits achieved the strict 99% calibration bar. The target refers to retained human-core play after 2s export padding. It is not a guarantee of 99% recall on another video.

Strict 99% calibration only; maximum common-unseen exact-label F1_padP_coreR; highest-recall draw within the winning variant, breaking ties with F1_padP_coreR. Common-unseen is already selection data, not an untouched test set. Its exact-label panel contains 1 recording(s) from 1 source group(s); this small panel limits generalization claims.

Target product padding is fixed at 2s before and after. Positive gaps strictly under 3s join; ignored intervals are subtracted afterward without rejoining. Metrics pool intersections and durations.

| Model | Selection | Fit | Draw | Epoch | P_pad | R_core | F1_padP_coreR | Wholly missed human rallies |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Distilled Large | Highest f1 | fit-002 | 20260918 | 60 | 97.76% | 97.68% | 97.72% | 1 |
| Distilled Large | Highest recall | fit-001 | 3407 | 15 | 70.88% | 100.00% | 82.96% | 0 |
| Frozen Large | Highest f1 | fit-014 | 20260918 | 30 | 98.52% | 94.22% | 96.32% | 1 |
| Frozen Large | Highest recall | fit-013 | 3407 | 15 | 81.51% | 99.51% | 89.61% | 1 |

Wholly missed means no retained nonignored core from a saved-human rally after padding and gap joining. This event count differs from duration-weighted retained-play recall.

| Model | Selection | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Distilled Large | f1 | 0 | 97.94% | 83.84% | 90.34% | 276.000 | 322.419 | -46.419 |
| Distilled Large | f1 | 1 | 97.83% | 95.32% | 96.56% | 354.000 | 400.419 | -46.419 |
| Distilled Large | f1 | 2 | 97.76% | 97.68% | 97.72% | 432.000 | 478.419 | -46.419 |
| Distilled Large | f1 | 3 | 97.71% | 98.77% | 98.24% | 510.000 | 556.419 | -46.419 |
| Distilled Large | recall | 0 | 65.29% | 98.51% | 78.53% | 486.500 | 322.419 | +164.081 |
| Distilled Large | recall | 1 | 67.85% | 100.00% | 80.84% | 584.125 | 400.419 | +183.706 |
| Distilled Large | recall | 2 | 70.88% | 100.00% | 82.96% | 669.250 | 478.419 | +190.831 |
| Distilled Large | recall | 3 | 73.49% | 100.00% | 84.72% | 751.625 | 556.419 | +195.206 |
| Frozen Large | f1 | 0 | 97.59% | 77.56% | 86.43% | 256.250 | 322.419 | -66.169 |
| Frozen Large | f1 | 1 | 98.16% | 90.08% | 93.95% | 336.250 | 400.419 | -64.169 |
| Frozen Large | f1 | 2 | 98.52% | 94.22% | 96.32% | 416.250 | 478.419 | -62.169 |
| Frozen Large | f1 | 3 | 98.76% | 96.84% | 97.79% | 497.250 | 556.419 | -59.169 |
| Frozen Large | recall | 0 | 76.92% | 95.30% | 85.13% | 399.500 | 322.419 | +77.081 |
| Frozen Large | recall | 1 | 79.60% | 99.12% | 88.30% | 481.500 | 400.419 | +81.081 |
| Frozen Large | recall | 2 | 81.51% | 99.51% | 89.61% | 563.500 | 478.419 | +85.081 |
| Frozen Large | recall | 3 | 82.32% | 99.51% | 90.10% | 650.750 | 556.419 | +94.331 |

Frozen-Large comparisons use the same common exact-label recordings, human rally boundaries, ignored intervals and duration revision, verified before reporting. Each family retains its own frozen selections.

## Frozen selections across all videos

Exact, draft and export panels are separate; all and no-production-training scopes exclude beach. These panels include fitting/calibration recordings and are not an untouched generalization test. No-production-training excludes known production fitting exposure; production calibration exposure can remain.

| Selection | Videos with predictions | Videos with scored labels | Videos with counts only |
|---|---:|---:|---:|
| f1 | 44 | 36 | 8 |
| recall | 44 | 36 | 8 |

Videos without an eligible scoring-label policy still receive predictions; their counts do not establish accuracy.

P/R/F1 below use padded-precision/core-recall for exact labels, reviewed-live overlap for drafts, and fixed human-export overlap for reviewed exports. Their different denominators are not pooled together.

| Selection | Labels | Scope | Videos | Found rallies | P | R | F1 | Wholly missed human rallies |
|---|---|---|---:|---:|---:|---:|---:|---:|
| f1 | exact-rallies | all | 9 | 371 | 98.84% | 99.60% | 99.22% | 2 |
| f1 | exact-rallies | no-production-training | 1 | 40 | 97.76% | 97.68% | 97.72% | 1 |
| f1 | reviewed-draft | all | 6 | 231 | 96.84% | 99.27% | 98.04% | N/A |
| f1 | reviewed-export | all | 19 | 711 | 98.23% | 78.94% | 87.54% | N/A |
| f1 | reviewed-export | no-production-training | 8 | 298 | 98.37% | 78.58% | 87.37% | N/A |
| f1 | exact-rallies | beach | 2 | 9 | 97.90% | 7.63% | 14.16% | 62 |
| recall | exact-rallies | all | 9 | 411 | 78.40% | 99.81% | 87.82% | 1 |
| recall | exact-rallies | no-production-training | 1 | 47 | 70.88% | 100.00% | 82.96% | 0 |
| recall | reviewed-draft | all | 6 | 261 | 76.72% | 99.79% | 86.75% | N/A |
| recall | reviewed-export | all | 19 | 798 | 86.32% | 92.01% | 89.08% | N/A |
| recall | reviewed-export | no-production-training | 8 | 340 | 83.75% | 91.75% | 87.57% | N/A |
| recall | exact-rallies | beach | 2 | 68 | 89.34% | 78.01% | 83.29% | 16 |

| Selection | Labels | Scope | Padding | P | R | F1 | Model export (s) | Human export (s) | Difference (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| f1 | exact-rallies | all | 0s | 98.31% | 88.48% | 93.14% | 2438.467 | 2709.570 | -271.103 |
| f1 | exact-rallies | all | 1s | 98.64% | 98.60% | 98.62% | 3160.467 | 3431.570 | -271.103 |
| f1 | exact-rallies | all | 2s | 98.84% | 99.60% | 99.22% | 3884.950 | 4153.570 | -268.620 |
| f1 | exact-rallies | all | 3s | 98.99% | 99.81% | 99.39% | 4606.562 | 4879.682 | -273.120 |
| f1 | exact-rallies | no-production-training | 0s | 97.94% | 83.84% | 90.34% | 276.000 | 322.419 | -46.419 |
| f1 | exact-rallies | no-production-training | 1s | 97.83% | 95.32% | 96.56% | 354.000 | 400.419 | -46.419 |
| f1 | exact-rallies | no-production-training | 2s | 97.76% | 97.68% | 97.72% | 432.000 | 478.419 | -46.419 |
| f1 | exact-rallies | no-production-training | 3s | 97.71% | 98.77% | 98.24% | 510.000 | 556.419 | -46.419 |
| f1 | reviewed-draft | all | 0s | 94.74% | 85.28% | 89.76% | 1452.233 | 1613.050 | -160.817 |
| f1 | reviewed-draft | all | 1s | 96.02% | 97.76% | 96.89% | 1911.887 | 2085.150 | -173.263 |
| f1 | reviewed-draft | all | 2s | 96.84% | 99.27% | 98.04% | 2370.887 | 2556.700 | -185.813 |
| f1 | reviewed-draft | all | 3s | 97.36% | 99.40% | 98.37% | 2829.062 | 3035.755 | -206.693 |
| f1 | reviewed-export | all | 0s | 99.22% | 54.20% | 70.10% | 5766.375 | 10555.696 | -4789.321 |
| f1 | reviewed-export | all | 1s | 98.90% | 66.75% | 79.70% | 7123.875 | 10555.696 | -3431.821 |
| f1 | reviewed-export | all | 2s | 98.23% | 78.94% | 87.54% | 8482.963 | 10555.696 | -2072.733 |
| f1 | reviewed-export | all | 3s | 95.26% | 88.59% | 91.80% | 9817.581 | 10555.696 | -738.115 |
| f1 | reviewed-export | no-production-training | 0s | 99.45% | 51.91% | 68.22% | 2212.000 | 4237.325 | -2025.325 |
| f1 | reviewed-export | no-production-training | 1s | 99.14% | 65.46% | 78.86% | 2798.000 | 4237.325 | -1439.325 |
| f1 | reviewed-export | no-production-training | 2s | 98.37% | 78.58% | 87.37% | 3385.000 | 4237.325 | -852.325 |
| f1 | reviewed-export | no-production-training | 3s | 94.94% | 88.93% | 91.84% | 3969.000 | 4237.325 | -268.325 |
| f1 | exact-rallies | beach | 0s | 93.11% | 2.85% | 5.52% | 15.783 | 516.485 | -500.702 |
| f1 | exact-rallies | beach | 1s | 96.78% | 5.74% | 10.84% | 33.783 | 658.485 | -624.702 |
| f1 | exact-rallies | beach | 2s | 97.90% | 7.63% | 14.16% | 51.783 | 800.485 | -748.702 |
| f1 | exact-rallies | beach | 3s | 98.44% | 8.99% | 16.47% | 69.783 | 948.299 | -878.516 |
| recall | exact-rallies | all | 0s | 72.93% | 96.72% | 83.16% | 3593.670 | 2709.570 | +884.100 |
| recall | exact-rallies | all | 1s | 76.07% | 99.46% | 86.20% | 4390.845 | 3431.570 | +959.275 |
| recall | exact-rallies | all | 2s | 78.40% | 99.81% | 87.82% | 5181.079 | 4153.570 | +1027.509 |
| recall | exact-rallies | all | 3s | 80.50% | 99.91% | 89.16% | 5951.704 | 4879.682 | +1072.022 |
| recall | exact-rallies | no-production-training | 0s | 65.29% | 98.51% | 78.53% | 486.500 | 322.419 | +164.081 |
| recall | exact-rallies | no-production-training | 1s | 67.85% | 100.00% | 80.84% | 584.125 | 400.419 | +183.706 |
| recall | exact-rallies | no-production-training | 2s | 70.88% | 100.00% | 82.96% | 669.250 | 478.419 | +190.831 |
| recall | exact-rallies | no-production-training | 3s | 73.49% | 100.00% | 84.72% | 751.625 | 556.419 | +195.206 |
| recall | reviewed-draft | all | 0s | 68.37% | 95.25% | 79.60% | 2247.416 | 1613.050 | +634.366 |
| recall | reviewed-draft | all | 1s | 73.08% | 98.98% | 84.08% | 2747.083 | 2085.150 | +661.933 |
| recall | reviewed-draft | all | 2s | 76.72% | 99.79% | 86.75% | 3229.491 | 2556.700 | +672.791 |
| recall | reviewed-draft | all | 3s | 79.69% | 99.86% | 88.64% | 3710.866 | 3035.755 | +675.111 |
| recall | reviewed-export | all | 0s | 92.19% | 73.11% | 81.55% | 8371.805 | 10555.696 | -2183.892 |
| recall | reviewed-export | all | 1s | 89.86% | 83.75% | 86.70% | 9838.473 | 10555.696 | -717.224 |
| recall | reviewed-export | all | 2s | 86.32% | 92.01% | 89.08% | 11250.960 | 10555.696 | +695.263 |
| recall | reviewed-export | all | 3s | 80.23% | 96.61% | 87.66% | 12710.709 | 10555.696 | +2155.012 |
| recall | reviewed-export | no-production-training | 0s | 90.86% | 71.99% | 80.33% | 3357.605 | 4237.325 | -879.721 |
| recall | reviewed-export | no-production-training | 1s | 87.82% | 83.15% | 85.42% | 4012.066 | 4237.325 | -225.260 |
| recall | reviewed-export | no-production-training | 2s | 83.75% | 91.75% | 87.57% | 4642.254 | 4237.325 | +404.928 |
| recall | reviewed-export | no-production-training | 3s | 77.17% | 96.62% | 85.80% | 5305.379 | 4237.325 | +1068.053 |
| recall | exact-rallies | beach | 0s | 84.52% | 59.98% | 70.17% | 366.525 | 516.485 | -149.960 |
| recall | exact-rallies | beach | 1s | 86.73% | 71.29% | 78.25% | 490.792 | 658.485 | -167.693 |
| recall | exact-rallies | beach | 2s | 89.34% | 78.01% | 83.29% | 603.842 | 800.485 | -196.643 |
| recall | exact-rallies | beach | 3s | 92.13% | 82.17% | 86.87% | 710.592 | 948.299 | -237.707 |

The JSON includes every feasible target-99 fit with all four padding cases and aggregate event/coverage diagnostics. Fit and variant indexes are report-local and disclose no recording identifiers. Exact human rallies, reviewed drafts and reviewed export coverage must remain separate evaluation panels; export coverage does not establish rally boundaries or wholly missed rally counts.
