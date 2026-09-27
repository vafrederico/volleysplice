# Browser visual preprocessing validation

The corrected AV preprocessing is implemented as the opt-in `visualPreprocessing: "opencv-area-nearest-grid-v1"` extraction option. **Keep the shipped browser default unchanged.** On the full recording, the correction reduced production F1 and increased extraction time. No model weights, thresholds or score-specialist sampling contracts changed.

Recording-044 was evaluated on the saved 37-rally human snapshot and its ignored intervals. The target product padding is two seconds before/after, with positive gaps joined only when strictly below three seconds. This is a runtime investigation on one recording, not model selection or a dataset-wide result.

## Measured browser pipeline

Desktop Chrome 153.0.8010.48 on Windows read the actual media, generated the shipped 104 AV features, and ran both production stacks and their serve heads plus suppression inference. The timing below is AV extraction; score-specialist video passes, neural image encoding, and export rendering are excluded. Model inference was measured separately in the private receipts. The full recording is 1,061.016489 seconds.

| Input | Previous AV time | Corrected AV time | Change |
| --- | ---: | ---: | ---: |
| Two-minute excerpt (median of two) | 13.194s | 15.834s | +20.0% |
| Full recording (one run each) | 108.888s | 132.222s | +21.4% |

## Full-recording model effects

Precision is padded-export precision; recall is retained human-core time; F1 is `F1_padP_coreR`. Wholly missed counts use zero retained core overlap after product padding.

| Model / AV input | Precision | Recall | F1 | Export seconds | Wholly missed human rallies |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production ensemble, previous | 66.88% | 97.45% | 79.32% | 682.544 | 1 |
| Distilled Large recall: AV-only diagnostic, previous | 87.68% | 99.26% | 93.11% | 510.000 | 0 |
| Production ensemble, corrected | 65.63% | 97.38% | 78.41% | 689.141 | 1 |
| Distilled Large recall: AV-only diagnostic, corrected | 87.96% | 99.26% | 93.27% | 506.375 | 0 |

The neural diagnostic holds the saved native embeddings, image-quality features, frozen highest-recall Distilled MobileNetV3-Large TCN and decoder fixed, replacing only the AV104 block with browser-generated inputs. It uses the same percentile/absolute-feature treatment and fitted scaler. **It is not an end-to-end browser neural benchmark.** The editor lab displays precomputed neural results; it does not run this image encoder on uploaded browser video.

The subsequent [Distilled Large browser experiment](distilled-large-browser-experiment.md)
adds actual browser image encoding for both frozen selections, independently
checks the resulting tensors, and includes both score-specialist pipelines.
It retains current and corrected AV as separate comparisons. Its browser input
contract follows training's uint8 image preparation and FP16 token-cache rounding,
so those results must not be described as merely replacing the AV block of the
native embeddings used in this earlier diagnostic.

The short clip has mixed effects too: production F1 changed from 84.49% to 82.60%, while the controlled neural diagnostic changed from 93.32% to 95.52%. Neither version wholly missed a saved human rally in that excerpt.

## Corrected contract and tests

- Source-resolution, display-oriented cropped RGB reaches OpenCV INTER_AREA. The previous canvas downscale bypassed that operation.
- Nearest actual presentation timestamps replace preceding-frame selection for the AV experiment. Ties choose earlier frames, and nominal 4 Hz rows retain repeated/terminal samples.
- Sparse and sequential corrected extraction produced exactly equal features, timestamps and rally outputs on the excerpt.
- Eight synthetic high-frequency image tests cover all rotations and both integer/fractional crop reductions; worker feature values exactly match direct OpenCV extraction.
- Python-compatible crop rounding and bounds, cache separation, empty input, repeated frames, nearest ties, and cancellation cleanup have unit coverage.
- The default AV sampler and separate score-specialist sampler preserve their previous selection semantics. Existing browser libswresample audio correction remains in place.

The corrected excerpt has 481 rows rather than 480: the old actual-PTS deduplication dropped the valid final nominal tick in the 120.063854-second file. Full-video rows remain 4,245 in both paths. The controlled short neural substitution uses the common first 480 rows (120 seconds) to match the saved embedding fixture.

## Required padding sensitivity

All comparisons below use the same saved labels, ignored spans and strict three-second join rule. Pooled values here represent one recording; no per-video averaging is used.

| Scope / input | Padding | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / production | 0s | 81.04% | 74.05% | 77.39% | 60.714 | 66.441 | -5.727 |
| baseline / production | 1s | 77.12% | 82.90% | 79.91% | 78.712 | 77.441 | +1.272 |
| baseline / production | 2s | 81.75% | 87.42% | 84.49% | 87.712 | 89.940 | -2.228 |
| baseline / production | 3s | 87.29% | 93.42% | 90.25% | 99.320 | 98.940 | +0.380 |
| baseline / neural AV diagnostic | 0s | 97.43% | 70.64% | 81.90% | 48.125 | 66.377 | -18.252 |
| baseline / neural AV diagnostic | 1s | 97.91% | 81.93% | 89.21% | 59.250 | 77.377 | -18.127 |
| baseline / neural AV diagnostic | 2s | 98.24% | 88.88% | 93.32% | 70.250 | 89.876 | -19.626 |
| baseline / neural AV diagnostic | 3s | 98.78% | 94.91% | 96.80% | 81.000 | 98.876 | -17.876 |
| area-nearest / production | 0s | 77.66% | 75.76% | 76.70% | 64.814 | 66.441 | -1.627 |
| area-nearest / production | 1s | 73.85% | 84.76% | 78.93% | 84.064 | 77.441 | +6.623 |
| area-nearest / production | 2s | 76.85% | 89.29% | 82.60% | 96.064 | 89.940 | +6.124 |
| area-nearest / production | 3s | 84.22% | 94.16% | 88.91% | 106.939 | 98.940 | +7.999 |
| area-nearest / neural AV diagnostic | 0s | 98.56% | 75.91% | 85.76% | 51.125 | 66.377 | -15.252 |
| area-nearest / neural AV diagnostic | 1s | 98.81% | 86.45% | 92.22% | 62.250 | 77.377 | -15.127 |
| area-nearest / neural AV diagnostic | 2s | 98.99% | 92.28% | 95.52% | 73.250 | 89.876 | -16.626 |
| area-nearest / neural AV diagnostic | 3s | 99.12% | 96.03% | 97.55% | 84.250 | 98.876 | -14.626 |
| full-baseline / production | 0s | 62.39% | 97.07% | 75.96% | 499.630 | 321.146 | +178.484 |
| full-baseline / production | 1s | 64.71% | 97.14% | 77.68% | 594.877 | 397.146 | +197.731 |
| full-baseline / production | 2s | 66.88% | 97.45% | 79.32% | 682.544 | 470.645 | +211.899 |
| full-baseline / production | 3s | 71.36% | 97.77% | 82.50% | 747.788 | 544.644 | +203.144 |
| full-baseline / neural AV diagnostic | 0s | 82.15% | 92.92% | 87.21% | 363.250 | 321.146 | +42.104 |
| full-baseline / neural AV diagnostic | 1s | 86.02% | 98.29% | 91.75% | 435.250 | 397.146 | +38.104 |
| full-baseline / neural AV diagnostic | 2s | 87.68% | 99.26% | 93.11% | 510.000 | 470.645 | +39.355 |
| full-baseline / neural AV diagnostic | 3s | 89.40% | 99.57% | 94.21% | 586.375 | 544.644 | +41.731 |
| full-area-nearest / production | 0s | 61.08% | 95.46% | 74.50% | 501.891 | 321.146 | +180.745 |
| full-area-nearest / production | 1s | 63.02% | 96.83% | 76.35% | 603.516 | 397.146 | +206.371 |
| full-area-nearest / production | 2s | 65.63% | 97.38% | 78.41% | 689.141 | 470.645 | +218.497 |
| full-area-nearest / production | 3s | 69.74% | 97.69% | 81.38% | 762.766 | 544.644 | +218.123 |
| full-area-nearest / neural AV diagnostic | 0s | 82.08% | 92.61% | 87.03% | 362.375 | 321.146 | +41.229 |
| full-area-nearest / neural AV diagnostic | 1s | 85.97% | 98.24% | 91.70% | 434.375 | 397.146 | +37.229 |
| full-area-nearest / neural AV diagnostic | 2s | 87.96% | 99.26% | 93.27% | 506.375 | 470.645 | +35.730 |
| full-area-nearest / neural AV diagnostic | 3s | 89.33% | 99.57% | 94.17% | 586.000 | 544.644 | +41.356 |

No physical phone was used. Source media, raw feature arrays, browser profiles and benchmark receipts remain under private-reference-0223. The [structured report](web-visual-preprocessing-validation.json) includes event metrics, missed-rally ranges, per-stage timings and all four padding cases. The reusable [loopback harness](../../scripts/benchmark-browser-preprocessing.mjs) takes privately resolved paths and does not serve or deploy the production UI.

## Final verification

After making the correction opt-in, the production default reproduced the old
excerpt timestamps, features and rally outputs exactly. The opt-in worker reproduced
its measured corrected outputs exactly. Its worker-free fallback differed by at most
5.96e-8 in features and produced identical rallies, but took 45.75s for AV extraction
on the excerpt; this slower fallback is not the measured worker default.

Production and lab TypeScript checks passed. The production static build and asset
integrity/hosting-limit checks passed. Of 200 focused and production web tests, 197
passed; the three failures also reproduce in an isolated HEAD snapshot: one existing
score-UI source assertion and two existing suppression interval expectations. All
14 sampler/crop/cache tests passed. The branch privacy audit found no private
references in 792 inspected working files or historical objects. No deployment or
physical-phone validation was performed.
