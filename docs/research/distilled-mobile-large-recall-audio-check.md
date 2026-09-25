# Highest-recall distilled Large: audio-only replacement check

**Replacing only the 27 Android audio feature columns with desktop values reduces wholly missed saved human rallies from 2 to 0.** The frozen highest-recall model finds 39 rallies, with 92.29% padded precision, 98.75% retained-core recall and 95.41% `F1_padP_coreR` on the benchmark recording.

This is an offline replay of saved full-video inputs, not a corrected Android run. The selected encoder/TCN weights, normalization, four-head inference and decoder remain fixed. All nonaudio input values remain bitwise identical to the native fused tensor. No training, calibration, threshold selection, new video extraction or device access occurs.

| Input source | Found rallies | Wholly missed / 37 | P_pad | R_core | F1_padP_coreR | Export (s) |
|---|---:|---:|---:|---:|---:|---:|
| Android features | 35 | 2 / 37 | 87.66% | 93.84% | 90.65% | 471.375 |
| Android + desktop audio only | 39 | 0 / 37 | 92.29% | 98.75% | 95.41% | 473.625 |
| Desktop features | 37 | 0 / 37 | 88.12% | 99.26% | 93.36% | 506.625 |

Primary padding is 2s before and after. Model and human padded ranges join only for positive gaps strictly under 3s. Ignored time is subtracted without rejoining. Zero wholly missed rallies does not mean all play time is retained.

| Previously missed human range | Retained after audio replacement | Result |
|---|---:|---|
| 09:22.50-09:28.75 | 4.629s / 6.250s (74.07%) | Partially retained |
| 17:13.61-17:20.36 | 6.749s / 6.749s (100.00%) | Fully retained |

The earlier range still loses its last 1.62s (09:27.125-09:28.746). The later range is fully retained. Other boundary differences remain; this mixed-input result is not identical to desktop inference.

## Required padding sensitivities

| Input source | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Android features | 0s | 80.50% | 85.07% | 82.73% | 339.375 | 321.146 | +18.229 |
| Android features | 1s | 85.16% | 91.38% | 88.16% | 405.375 | 397.146 | +8.229 |
| Android features | 2s | 87.66% | 93.84% | 90.65% | 471.375 | 470.645 | +0.730 |
| Android features | 3s | 88.94% | 95.36% | 92.04% | 546.000 | 544.644 | +1.356 |
| Android + desktop audio only | 0s | 87.51% | 89.79% | 88.64% | 329.500 | 321.146 | +8.354 |
| Android + desktop audio only | 1s | 90.45% | 97.07% | 93.65% | 403.625 | 397.146 | +6.479 |
| Android + desktop audio only | 2s | 92.29% | 98.75% | 95.41% | 473.625 | 470.645 | +2.980 |
| Android + desktop audio only | 3s | 94.11% | 99.38% | 96.67% | 543.625 | 544.644 | -1.019 |
| Desktop features | 0s | 82.30% | 92.93% | 87.29% | 362.625 | 321.146 | +41.479 |
| Desktop features | 1s | 86.15% | 98.24% | 91.80% | 434.625 | 397.146 | +37.479 |
| Desktop features | 2s | 88.12% | 99.26% | 93.36% | 506.625 | 470.645 | +35.980 |
| Desktop features | 3s | 89.46% | 99.57% | 94.24% | 586.250 | 544.644 | +41.606 |

## Verification and scope

- All 4,245 timestamps and all 3,952 input columns use the recall selection's own feature cache and fold normalization. Audio names are derived from the declared 104-column feature contract, then asserted to be exactly columns 77-103 (27 values, including audio availability).
- Independent endpoint-partition interval arithmetic matches all 12 case-by-padding comparisons and both recovered-range coverage values exactly.
- Native and desktop baseline probabilities replay within 6e-7 maximum absolute error. Native decoded boundaries reproduce the original phone run exactly.
- Model/checkpoint/graph hashes and the 37-rally gold snapshot are verified before replay. The benchmark source is `recording-044`; detailed execution evidence and reproduction code resolve through ledger `private-reference-0223`.
- This confirms audio-feature mismatch materially affects both distilled selections on this recording. It does not identify the individual resampling/downmix/DSP defect, establish generalization, or qualify an Android fix.
- Same source, labels and selected models as the [native benchmark](distilled-mobile-large-native-benchmark.md). The saved labels are manually reviewed export-derived ranges, not independently precise serve-contact/dead-ball annotations.

[Structured evidence](distilled-mobile-large-recall-audio-check.json) includes all four padding cases, frozen input identities, rally boundaries and event guardrails.
