# Balanced-encoder serving-side precision: results

The centered serving-side head trained on **Balanced embeddings** achieved **97.99% automatic precision with 16.36% reviewed**, averaged across three seeds at the 98% selection target. Its three-seed agreement policy achieved **98.53% precision**, with **220 of 968 serves reviewed (22.73%)** and **11 wrong automatic answers among 748 accepted**. Production has not changed.

This matched comparison changes only the frozen image encoder. It uses the same 968 reviewed true non-beach serves from 26 recordings and eight source groups, human labels, anchors, temporal masks, frame timestamps, 237 specialist ranks, and 12 phase-specific absolute motion-quality values. All 108 nested neural fits and three final refits are newly trained; the previous serving weights are not reused.

The Balanced encoder weights match `high-f1` in the production manifest. Both encoders are DINO-distilled MobileNetV3-Large but have different weights and embeddings. Inputs retain FP16-rounded four-region 960-dimensional tokens over 13 frames spanning -3 to +3 seconds. Centering subtracts each window's valid-frame mean embedding. The serving head architecture, training settings, seeds, and nested precision selector match the Maximum coverage centered experiment.

Outer source groups are excluded from fitting and inner epoch/threshold selection. Precision is correct automatic side answers divided by accepted side answers; review rate uses all 968 known serves. A target is an inner-validation condition and does not guarantee unseen-source precision. This evaluation does not rerun rally detection or the separate serve/no-serve gate.

## Matched encoder comparison

Mean-seed counts may be fractional. Agreement requires all three seed policies to accept and agree.

| Encoder | Policy | Selection target | Measured precision | Reviews / 968 | Review rate | Wrong automatic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Maximum coverage | Mean seeds | 95% | 95.25% | 19.7 | 2.03% | 45 |
| Maximum coverage | Three-seed agreement | 95% | 96.19% | 49 | 5.06% | 35 |
| Balanced | Mean seeds | 95% | 95.72% | 17.7 | 1.83% | 40.7 |
| Balanced | Three-seed agreement | 95% | 96.75% | 45 | 4.65% | 30 |
| Maximum coverage | Mean seeds | 98% | 98.19% | 160 | 16.53% | 14.7 |
| Maximum coverage | Three-seed agreement | 98% | 98.79% | 223 | 23.04% | 9 |
| Balanced | Mean seeds | 98% | 97.99% | 158.3 | 16.36% | 16.3 |
| Balanced | Three-seed agreement | 98% | 98.53% | 220 | 22.73% | 11 |
| Maximum coverage | Mean seeds | 99% | 98.13% | 518.7 | 53.58% | 8.3 |
| Maximum coverage | Three-seed agreement | 99% | 98.52% | 631 | 65.19% | 5 |
| Balanced | Mean seeds | 99% | 98.58% | 476.3 | 49.21% | 7 |
| Balanced | Three-seed agreement | 99% | 98.69% | 585 | 60.43% | 5 |

## Precision guardrails at the 98% target

| Encoder | Policy | Near precision | Far precision | Source-average precision | All four 98% checks |
| --- | --- | ---: | ---: | ---: | --- |
| Maximum coverage | Mean seeds | 98.17% | 98.22% | 97.97% | Fail |
| Maximum coverage | Three-seed agreement | 98.89% | 98.70% | 98.49% | Pass |
| Balanced | Mean seeds | 98.01% | 97.97% | 97.73% | Fail |
| Balanced | Three-seed agreement | 98.65% | 98.41% | 98.24% | Pass |

The Balanced agreement policy **cleared** the registered pooled, both-side, and average-source 98% guardrails. Source-average precision excludes groups with no accepted answers; those groups remain in review/coverage denominators. A source average does not impose 98% on every individual source.

The Balanced agreement policy's descriptive pooled Wilson 95% lower precision bound is 97.39%. This is not a source-cluster or repeated-model-selection guarantee. The 99%-target Balanced agreement policy measured 98.69% and missed 99% despite its larger review workload.

## Individual Balanced seeds at the 98% target

| Seed | Precision | Near precision | Far precision | Source-average precision | Reviews / 968 | Wrong automatic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3407 | 97.83% | 98.05% | 97.62% | 98.08% | 138 | 18 |
| 1729 | 98.24% | 98.43% | 98.06% | 97.65% | 174 | 14 |
| 20260918 | 97.89% | 97.56% | 98.23% | 97.46% | 163 | 17 |

No best-seed selection was performed. Infeasible outer policies across three seeds: 0.

## Balanced agreement by held source at the 98% target

| Source | Serves | Accepted | Reviews | Wrong automatic | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-group-002 | 46 | 35 | 11 | 2 | 94.29% |
| source-group-003 | 40 | 25 | 15 | 0 | 100.00% |
| source-group-005 | 195 | 140 | 55 | 4 | 97.14% |
| source-group-007 | 145 | 134 | 11 | 0 | 100.00% |
| source-group-009 | 74 | 70 | 4 | 1 | 98.57% |
| source-group-010 | 39 | 28 | 11 | 0 | 100.00% |
| source-group-012 | 56 | 33 | 23 | 1 | 96.97% |
| source-group-013 | 373 | 283 | 90 | 3 | 98.94% |

## Paired agreement outcomes at the 98% target

Each cell counts the same serves under the two encoder-specific agreement policies. Rows are Maximum coverage outcomes; columns are Balanced outcomes.

| Maximum coverage outcome | Balanced review | Balanced correct automatic | Balanced wrong automatic |
| --- | ---: | ---: | ---: |
| Review | 191 | 30 | 2 |
| Correct | 29 | 707 | 0 |
| Wrong | 0 | 0 | 9 |

Of Maximum coverage's wrong automatic answers, 0 become correct automatic Balanced answers, 0 become reviews, and 9 remain wrong. Balanced also makes 0 wrong automatic answers on previously correct answers and 2 on previously reviewed serves. Both policies accept but disagree on 0 serves.

## Verification and artifacts

Nine qualification tests passed before registration. The independent sampler also passed 150 synthetic cases, including exact window matching and invalid-anchor rejection around ignored time. The independent audit rebuilt all Balanced windows from the private cache with that sampler, checked matching image lineage/timestamps/masks/features, verified all 37 logistic teacher partitions and 111 neural fits, and replayed all 72 outer policies and nine final policies. It reproduced every aggregate and agreement metric.

Reloaded 35 selected checkpoints; maximum logit error was 0. All nine final research-model configurations passed CPU/GPU parity; maximum score difference was 2.12e-07. Final policies were selected on source-held development predictions before all-development refits; final in-sample predictions are not evaluation.

Training ran in one fitting process with two CPU threads. Peak sampled resident memory was 1.77 GiB; maximum sampled process swap was 0 KiB. WSL swap counter deltas were 0 pages in and 0 pages out. Minimum WSL available memory was 12.56 GiB.

Frozen registration: `33ad21271ebb2f00fc4349e8fcb3b729d50365704cb1af90e19998df9261ea40`. Durable artifacts use ledger `private-reference-0242`, SSD artifacts use `private-reference-0243`, and Balanced encoder/cache inputs use `private-reference-0244` / `private-reference-0245`. Exact locations, recording identities, row predictions, weights, and logs remain private. Protected labels were not opened. No commit, publication, or production change was performed.

[Complete public aggregates](serving-side-balanced-precision-2026-10-14.json) and [independent audit](serving-side-balanced-precision-audit-2026-10-14.json). See the [registered protocol](serving-side-balanced-precision-2026-10-14.md) and [earlier Maximum coverage results](serving-side-precision-iterations-results-2026-10-14.md).

## Next steps

1. Confirm the encoder-specific candidates on untouched recordings and source groups. These are reused development data; prior pixel exposure, historical encoder selection, and reviewed-candidate recording ranks limit generalization claims.
2. Evaluate the actual production rally and serve candidate pipeline, including false serves, review gates, and production candidate-based ranks, to measure end-to-end wrong automatic answers and review workload.
3. Inspect confident mistakes for label ambiguity, anchor timing, occlusion, and court geometry, prioritizing errors shared by both encoders. Register any new hypothesis instead of tuning the current thresholds on held-source errors.
4. Before promotion, export and verify the appropriate serving head for each production encoder, including centering, valid-frame masking, absolute quality features, thresholds, and agreement. Embeddings from the two encoders are not interchangeable.
