# Full-video production-only emulator comparison

Current native preprocessing completed all results in **26m12s**, compared with **34m44s** for repaired Java area resizing and **25m11s** for original point sampling. These are one full-video observation per version, measured sequentially on the same emulator.

Three frozen preprocessing versions process the same full-frame 17m41s recording-044 through production rally detection, serving-side and side-switch results. The normal production suppression analysis is included; rally boundaries remain the ensemble union. No image embeddings or TCN run. All feature caches are bypassed.

| Version | Video AV | Audio | Rallies ready | Score specialists | All ready | Rallies |
|---|---:|---:|---:|---:|---:|---:|
| Original point sampling | 941.953s | 64.324s | 1007.066s | 503.440s | 1510.506s | 61 |
| Repaired Java area | 1470.130s | 68.579s | 1539.477s | 544.395s | 2083.872s | 63 |
| Optimized native area | 952.319s | 72.004s | 1025.241s | 547.182s | 1572.423s | 63 |

| Comparison | All-ready change | Percent change |
|---|---:|---:|
| Original point sampling → Repaired Java area | +573.366s | +37.96% |
| Repaired Java area → Optimized native area | -511.449s | -24.54% |
| Original point sampling → Optimized native area | +61.917s | +4.10% |

## Preprocessing detail

These counters are nested and must not be added to stage totals.

| Version | Resize/color | Timestamp inventory | OpenCV features | Score decode |
|---|---:|---:|---:|---:|
| Original point sampling | 17.815s | 75.757s | 294.237s | 472.200s |
| Repaired Java area | 548.971s | 74.897s | 283.633s | 512.117s |
| Optimized native area | 36.693s | 78.996s | 282.437s | 514.522s |

The native kernel reduces resizing/color time by **93.32%** relative to repaired Java. Complete video AV time ends **1.10% above original point sampling**. Of the remaining 61.917s all-ready difference from original, 43.742s is in score specialists, which receive 63 rather than 61 production intervals. This decomposes the observed difference; a single run does not isolate proposal-count, cache or scheduling effects.

## Saved-human comparison

Declared target: 2 seconds before/after; join positive gaps strictly below 3 seconds. Subtract ignored time without rejoining. R_core measures retained human play duration. Event metrics and missed-rally ranges are in the JSON.

| Version | P_pad | R_core | F1_padP_coreR | Wholly missed / human rallies |
|---|---:|---:|---:|---:|
| Original point sampling | 69.01% | 97.38% | 80.78% | 1 / 37 |
| Repaired Java area | 66.33% | 99.01% | 79.44% | 1 / 37 |
| Optimized native area | 66.33% | 99.01% | 79.44% | 1 / 37 |

## Required padding sensitivity

| Version | Padding each side | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original point sampling | 0s | 66.83% | 90.40% | 76.85% | 434.391 | 321.146 | +113.245 |
| Original point sampling | 1s | 66.97% | 96.21% | 78.97% | 548.516 | 397.146 | +151.371 |
| Original point sampling | 2s | 69.01% | 97.38% | 80.78% | 637.016 | 470.645 | +166.372 |
| Original point sampling | 3s | 73.87% | 97.77% | 84.16% | 701.766 | 544.644 | +157.123 |
| Repaired Java area | 0s | 61.02% | 96.72% | 74.83% | 509.016 | 321.146 | +187.870 |
| Repaired Java area | 1s | 63.60% | 98.78% | 77.38% | 607.016 | 397.146 | +209.871 |
| Repaired Java area | 2s | 66.33% | 99.01% | 79.44% | 694.891 | 470.645 | +224.247 |
| Repaired Java area | 3s | 70.90% | 99.01% | 82.63% | 757.516 | 544.644 | +212.873 |
| Optimized native area | 0s | 61.02% | 96.72% | 74.83% | 509.016 | 321.146 | +187.870 |
| Optimized native area | 1s | 63.60% | 98.78% | 77.38% | 607.016 | 397.146 | +209.871 |
| Optimized native area | 2s | 66.33% | 99.01% | 79.44% | 694.891 | 470.645 | +224.247 |
| Optimized native area | 3s | 70.90% | 99.01% | 82.63% | 757.516 | 544.644 | +212.873 |

## Validation and limits

Java/native exact output checks: rallies **True**, serving-side **True**, side-switch **True**. Packaged model assets are identical across all three APKs; APK hashes match the archived qualified builds. The complete source was hash-verified on the emulator before timing.

- Single full-video observation per version; sequential order, no discarded pipeline warmups; filesystem/JIT and host scheduling effects are not isolated.
- Emulator CPU and codec performance is not physical Pixel performance. API 37 userdebug emulator uses the previously qualified temporary SELinux workaround.
- Original point-sampling control retains corrected audio and the same full-frame input, models and benchmark harness; it is not an exact historical APK.
- Score specialists use each version's own production rallies; proposal changes can change score-stage workload.
- Totals include feature generation and serving/side-switch models; exclude install, transfer, source hashing, result collection and export rendering. Nested counters overlap.
- Same previously investigated recording and manually reviewed export-derived gold; runtime validation, not a new generalization or model-selection result.
- Production AV tensors and per-tick probabilities are not saved; output equality does not establish complete tensor equality.

All 343 recorded wake observations were awake, with zero monitor errors and at least 8,726 MB of host memory available. All 12 padding comparisons passed a second interval implementation with zero numerical difference. SELinux enforcement and non-root ADB were restored, and emulator exit was confirmed after testing.

Raw evidence resolves through private-reference-0223. [Machine-readable report](production-emulator-full-comparison.json). Generated by [report script](../../scripts/report-production-emulator-comparison.py).
