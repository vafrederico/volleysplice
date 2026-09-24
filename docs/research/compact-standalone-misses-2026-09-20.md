# Compact standalone misses: recording-044

Date: 2026-09-20. This is a descriptive check of the frozen challenge recording, not training or model selection.

## Answer

**The human-kept rally that production catches and compact completely misses is R015, 4:33.366-4:41.114. Both production rally heads detect it; it is not recovered by the serve head.** Removing serve composition leaves its production union boundaries exactly unchanged. The serve at 4:37.990 lies inside the already detected rally.

Of the 52 production candidates outside ignored time, 20 have no compact core overlap. Thirteen have primary rally-head support and seven are pure 2.5-second serve fallback candidates. Only R015 overlaps the reviewed human-kept core. The seven pure serve fallback candidates are R010, R030, R031, R037, R038, R039 and R047; none overlaps retained human core. The two production candidates recovered solely through permissive rally decoding plus serve support (R028 and R035) are both covered by compact.

"No compact overlap" here means zero intersection with unpadded compact core ranges. It is not the event-IoU recall definition and does not imply all 20 candidates are real missed rallies.

## Human-kept omissions and timing

- **R015, 4:33.366-4:41.114:** all 7.749 seconds retained by production, none by compact even at 2-second padding.
- **Corrected human R030, 9:22.496-9:28.746:** both models miss the raw human core. The production candidate with the same ID is earlier, 9:18.996-9:21.746; padding retains only 20.00% of the corrected human interval. Matching IDs would give the wrong conclusion.
- **R029, 9:04.122-9:07.872:** compact core covers 40.01%, but 2-second padding retains all wanted play. R023 and R048 have low overlap with long production spans yet cover 100% of their corrected human cores. R049 covers 82.98% of its corrected human core and all of it after 2-second padding.
- **Manual M01, 11:08.254-11:11.427:** compact retains the entire human range with 2-second padding; production retains none. Compact can recover genuine production misses too.

The reviewed ranges are manual export cuts, not independently precise serve-to-dead-ball annotations. The immutable import and current editable draft have identical rally and ignored-range annotations.

At the target 2-second padding, the human ranges with any compact omitted core time are:

| Human ID | Human interval | Compact retained play | Production ensemble retained play |
| --- | --- | ---: | ---: |
| R011 | 3:35.000-3:44.623 | 79.22% | 100.00% |
| R014 | 4:15.618-4:27.866 | 93.88% | 100.00% |
| R015 | 4:33.366-4:41.114 | 0.00% | 100.00% |
| R016 | 4:52.864-5:13.984 | 99.41% | 100.00% |
| R030 | 9:22.496-9:28.746 | 0.00% | 20.00% |
| R044 | 13:24.865-13:34.114 | 97.30% | 100.00% |
| R050 | 15:22.863-15:34.112 | 82.23% | 100.00% |
| R051 | 15:42.111-15:54.858 | 92.17% | 100.00% |

## Production-only candidate lineage

These are all candidates with zero compact core overlap. "Kept by aggressive" identifies the default suppression-adjusted production result. Candidates without reviewed-core overlap should not be counted as human-confirmed missed rallies.

| Production ID | Interval | Causal source | Kept by aggressive | Human-kept core overlap |
| --- | --- | --- | --- | ---: |
| R010 | 3:27.235-3:29.735 | serve-created-without-primary-rally | No | 0.000s |
| R012 | 3:47.123-3:51.372 | rally-head-backed | No | 0.000s |
| R015 | 4:33.366-4:41.114 | rally-head-backed | Yes | 7.749s |
| R017 | 5:19.000-5:25.124 | rally-head-backed | Yes | 0.000s |
| R019 | 6:06.617-6:16.615 | rally-head-backed | No | 0.000s |
| R021 | 6:36.113-6:43.362 | rally-head-backed | No | 0.000s |
| R024 | 7:31.746-7:44.619 | rally-head-backed | Yes | 0.000s |
| R027 | 8:28.112-8:32.611 | rally-head-backed | No | 0.000s |
| R030 | 9:18.996-9:21.746 | serve-created-without-primary-rally | Yes | 0.000s |
| R031 | 9:33.494-9:36.494 | serve-created-without-primary-rally | Yes | 0.000s |
| R037 | 11:01.747-11:04.247 | serve-created-without-primary-rally | No | 0.000s |
| R038 | 11:15.494-11:17.994 | serve-created-without-primary-rally | Yes | 0.000s |
| R039 | 11:28.492-11:31.742 | serve-created-without-primary-rally | Yes | 0.000s |
| R042 | 12:22.608-12:27.374 | rally-head-backed | No | 0.000s |
| R047 | 14:23.248-14:25.748 | serve-created-without-primary-rally | No | 0.000s |
| R052 | 16:00.874-16:04.873 | rally-head-backed | No | 0.000s |
| R055 | 16:40.368-16:48.617 | rally-head-backed | No | 0.000s |
| R056 | 16:52.116-16:55.865 | rally-head-backed | No | 0.000s |
| R057 | 17:05.989-17:10.114 | rally-head-backed | Yes | 0.000s |
| R059 | 17:35.109-17:41.016 | rally-head-backed | No | 0.000s |

## Export metrics and required padding sensitivity

Target product padding is **2 seconds before and after**. Every case joins positive padded gaps strictly less than 3 seconds, then subtracts ignored time without rejoining across it. Precision compares padded model and padded human unions; recall measures retained human core. The F1 column is `F1_padP_coreR`, not event F1. Export differences are model minus human. Each case uses the same 37 reviewed ranges and ignored opening [0, 151.528966].

| Model | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production ensemble (unsuppressed) | 0s | 62.45% | 97.07% | 76.00% | 499.131 | 321.146 | +177.984 |
| Production ensemble (unsuppressed) | 1s | 64.74% | 97.14% | 77.70% | 594.627 | 397.146 | +197.481 |
| Production ensemble (unsuppressed) | 2s | 66.88% | 97.45% | 79.32% | 682.544 | 470.645 | +211.899 |
| Production ensemble (unsuppressed) | 3s | 71.36% | 97.77% | 82.50% | 747.788 | 544.644 | +203.144 |
| Production (aggressive suppression) | 0s | 72.69% | 97.07% | 83.13% | 428.843 | 321.146 | +107.696 |
| Production (aggressive suppression) | 1s | 76.33% | 97.14% | 85.49% | 504.341 | 397.146 | +107.196 |
| Production (aggressive suppression) | 2s | 78.64% | 97.45% | 87.04% | 580.479 | 470.645 | +109.834 |
| Production (aggressive suppression) | 3s | 83.01% | 97.77% | 89.79% | 640.478 | 544.644 | +95.834 |
| Compact standalone | 0s | 92.36% | 78.37% | 84.79% | 272.478 | 321.146 | -48.668 |
| Compact standalone | 1s | 95.63% | 89.77% | 92.60% | 338.478 | 397.146 | -58.667 |
| Compact standalone | 2s | 96.83% | 93.74% | 95.26% | 404.478 | 470.645 | -66.166 |
| Compact standalone | 3s | 98.13% | 95.02% | 96.55% | 470.478 | 544.644 | -74.165 |

## Reproduction and evidence

Study root: `private-reference-0117`.

- `compact-coverage-comparison-v1.json`: all candidate and human overlaps, four padding cases, immutable input identities.
- `compact-standalone-misses-v1/production-head-lineage.json`: per-component primary decode, serve actions, rescue lineage, dead-end refinement and exact no-serve counterfactual.
- `compact-standalone-misses-v1/production-head-lineage-receipt.json`: parity and source verification.
- `scripts/compare-labeling-compact-coverage.py`, `scripts/trace-labeling-production-lineage.py`, `scripts/trace-labeling-production-lineage.mjs`: reproducible analysis.

The causal replay uses saved production probabilities through the checked-in decoder/composer. Component endpoints/confidences, ensemble union, smoothing and traced composition match exactly. No features or probabilities were fit to this recording. The serving-side near/far classifier is separate from rally/serve composition and does not create rally export intervals.
