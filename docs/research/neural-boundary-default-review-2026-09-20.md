# Automatically apply boundary edits, then review removals

Date: 2026-09-20. Frozen compact short-boost TCN boundary-adviser evaluation; no training, threshold tuning or production promotion.

## Decision

Applying compact boundary changes provisionally and reviewing removed portions can improve export and rally-event accuracy. Reviewing individual removed fragments preserves more useful edits than reverting the entire production parent. The review load is substantial, and removed-time review alone does not establish correct rally separation for score tracking.

Keep the original production boundaries available for undo. Give each removal its own keep/undo decision and allow precise endpoint adjustment. Additional rally starts and split gaps need a separate explicit review decision even when padding or short-gap joining hides their export impact. Do not treat an unreviewed inferred split as a confirmed score increment.

## Evaluation contract

The development cohort contains eight recordings from four source groups, 322 gold rallies and 137.83 minutes of evaluable footage. Predictions come from the existing held-source-group study with seeds 3407, 1729 and 20260918. Metrics pool numerators and denominators across all eight recordings for each seed, then report seed means and ranges. Seeds are repeated predictions on the same recordings, not independent new videos. No protected test or recording-044 challenge labels selected a model or threshold.

The recording-044 recording is reported separately as a descriptive challenge case. Its 37 manually reviewed retained ranges are export labels, not newly annotated exact serve contacts or dead-ball endpoints. Its production baseline is the **aggressive suppression-adjusted ensemble**, with 40 candidates outside the ignored opening; the unsuppressed ensemble has 52.

The fixed target export padding is **2 seconds on each side**. Required sensitivities use 0, 1 and 3 seconds on each side. All export cases merge overlap/touch and join positive gaps **strictly less than 3 seconds** before subtracting ignored intervals. They never rejoin across ignored time. `P_pad` compares model and human padded export unions; `R_core` measures retained human core; `F1_padP_coreR` combines them. Rally event precision/recall/F1 instead use cardinality-first one-to-one matching with IoU at least 0.5, retaining distinct event identities.

These are perfect-human simulations. The review queue is generated without gold labels; gold supplies the simulated review answer. Playback seconds omit human decision, seeking and editing time.

## Tested decisions

1. **Production:** unchanged production event cores and exports.
2. **Apply all boundaries:** use all frozen `head_refined` event boundaries and recompute export from those events. This differs from the existing UI preview, which preserves production export coverage.
3. **Undo whole parent:** queue production core minus new event core. If any human core lies in a removed fragment, undo every edit to that production parent.
4. **Undo individual removal:** use the same queue, but restore only each whole removed fragment that overlaps human core. Prefix/suffix restoration extends the adjacent candidate; internal-gap restoration merges only the candidates linked through that restored gap. No gold endpoint coordinates enter this practical binary policy. Other event identities stay distinct.
5. **Exact restoration upper bounds, recording-044 only:** a human edits the unwanted removal to exact gold boundaries, or restores wanted padded-export portions in an export mask. These assume finer editing than a yes/no undo action. An export-mask correction does not repair event identities.

Every frozen boundary edit is included, including small shifts below the UI flag threshold. On recording-044 this affects 31 parents, while only 27 parents have actionable-threshold flags in the existing panel. A zero removed-core trigger cannot verify a coverage-preserving split. A zero removed-export trigger is even less informative because padding and short-gap joining can conceal removed core.

## Development results at target padding

| Policy | P_pad | R_core | F1_padP_coreR | Event precision | Event recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production | 72.45% | 99.27% | 83.76% | 62.64% | 70.81% | 66.47% |
| Apply all boundaries | 82.88% | 96.04% | 88.97% | 65.26% | 78.36% | 71.22% |
| Review: undo whole parent | 75.51% | 99.23% | 85.76% | 66.12% | 75.57% | 70.53% |
| Review: undo individual removal | 78.71% | 99.23% | 87.79% | 70.47% | 81.26% | 75.48% |

The individual-removal policy improves target export F1 by **4.03 percentage points** and event F1 by **9.01 points**. Its event F1 seed range is 74.64-76.77%; export F1 range is 87.57-87.92%. These ranges describe three fitted seeds, not population confidence intervals. The automatic variant ranks highest on the primary export F1 metric, but its retained-play loss conflicts with the requested priority of avoiding lost rallies.

Individual undo restores all 269.613 seconds of human core newly removed from the original production cores and introduces no additional completely missed rallies. It still loses **0.951 seconds of padded-export-covered human core** relative to production (seed range 0-2.853 seconds): some original padding covered wanted play outside the original parent. Thus retained-play recall is 99.23%, not exactly production's 99.27%. A removal-review implementation should include the export difference as a coverage check in addition to reviewing raw core edits.

| Policy | Total export (s) | Correctly removed (s) | Incorrect export (s) | Wanted human export omitted (s) | Human core omitted (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production | 4943.025 | 3232.930 | 1361.951 | 94.077 | 17.472 |
| Apply all boundaries | 4037.135 | 3903.593 | 691.288 | 329.304 | 94.626 |
| Review: undo whole parent | 4741.042 | 3433.740 | 1161.141 | 95.250 | 18.423 |
| Review: undo individual removal | 4547.964 | 3626.773 | 968.108 | 95.295 | 18.423 |

"Correctly removed" is evaluable time excluded from both the result and the padded human export. "Incorrect export" is retained footage outside the padded human export. "Wanted human export omitted" includes desired padding; "human core omitted" counts actual labeled core only.

## Development review workload

All reviewed policies see the same label-blind queue: approximately **592 removed fragments across 296 production parents**, touching **230 of 322 human rallies**. Exact removed footage totals **16.04 minutes**. Adding 2 seconds of context on either side and joining positive context gaps strictly below 3 seconds gives **385 playback clips / 55.90 minutes** (54.75-56.86 minutes across seeds), touching approximately 288 human rallies. This is about 41% of the evaluable footage, before decision and editing overhead.

Without joining positive context gaps, the same decisions require 502 playback clips / 52.98 minutes. These are playback-window choices, not alternative export-ranking policies.

The granular reviewer reverses approximately 332 of 592 removals, restoring 542.165 seconds, including 272.552 seconds outside gold core. About 260 removals survive. Fourteen candidate groups merge through restored internal gaps. Whole-parent undo instead reverses approximately 231 entire parents and gives up more beneficial boundary changes.

An export-only removal queue misses the actual internal gap of **17.67 out of 22.67 additional-start proposals**, even after adding 2-second viewing context at target export padding. Another edit to the same parent can still enter that queue, so counting reviewed parents would conceal this problem. Raw-core removal review exposes all these internal gaps. Add a separate split/serve-boundary review item; reviewing only removed export footage is inadequate for score tracking. All four padding cases for this diagnostic are saved in `split-gap-review-coverage-v1.json`.

For score tracking, the earlier [complete-parent correction experiment](neural-typed-boundary-results-2026-09-19.md) remains the stronger ideal-human result: 85.53% event F1 at 53.55 minutes of playback. That interaction permits complete event correction within selected parents; it is not binary rollback and keeps production export coverage fixed. The new binary undo policy improves export cleanup, but is not a demonstrated replacement for complete event review at similar playback cost.

## recording-044 results at target padding

| Policy | P_pad | R_core | F1_padP_coreR | Event precision | Event recall | Event F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production | 78.64% | 97.45% | 87.04% | 72.50% | 78.38% | 75.32% |
| Apply all boundaries | 86.91% | 94.85% | 90.71% | 71.43% | 81.08% | 75.95% |
| Review: undo whole parent | 78.64% | 97.45% | 87.04% | 72.50% | 78.38% | 75.32% |
| Review: undo individual removal | 82.81% | 97.45% | 89.54% | 77.50% | 83.78% | 80.52% |
| Exact core editing (upper bound) | 87.98% | 97.45% | 92.47% | 80.49% | 89.19% | 84.62% |
| Exact export mask (upper bound) | 88.26% | 97.45% | 92.63% | 71.43% | 81.08% | 75.95% |

| Policy | Total export (s) | Correctly removed (s) | Incorrect export (s) | Wanted human export omitted (s) | Human core omitted (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production | 580.479 | 314.835 | 124.008 | 14.173 | 8.173 |
| Apply all boundaries | 464.236 | 378.096 | 60.747 | 67.156 | 16.544 |
| Review: undo whole parent | 580.479 | 314.835 | 124.008 | 14.173 | 8.173 |
| Review: undo individual removal | 551.217 | 344.097 | 94.746 | 14.173 | 8.173 |
| Exact core editing (upper bound) | 518.843 | 376.471 | 62.372 | 14.173 | 8.173 |
| Exact export mask (upper bound) | 517.218 | 378.096 | 60.747 | 14.173 | 8.173 |

## recording-044 interpretation

The binary individual-removal reviewer restores 58 of 60 removed fragments: 96.358 seconds, including 56.358 seconds of wanted core and 40.000 seconds of unwanted core. Two accepted removals retain the cleanup improvement. Whole-parent undo reverses all 31 edited parents and returns exactly to production performance.

Reviewing the core removals requires 122.870 seconds of exact fragments across 31 parents, touching 33 of 37 human rallies. With 2-second viewing context and strict gap-less-than-3-second joins, this becomes 40 playback clips totaling **6:24.498**. With only overlap/touch merging for playback, it is 57 clips totaling **5:56.745**. Both playback conventions leave evaluation exports and review decisions unchanged.

The automatic edits make two split proposals. R014 has a false split whose 0.750-second internal gap is concealed in exports by short-gap joining, even at zero padding. R043 contains a useful split, but its 11.748-second proposed gap includes 0.248 seconds of wanted edge play. Binary whole-gap undo restores the entire gap and merges that useful split away. Precise endpoint editing is therefore more capable than binary removal undo, but requires more interaction. Neither policy recovers rallies that production already missed outside its proposed regions.

An export-only review queue at target padding contains 52 windows totaling 116.243 seconds, or **5:32.988** with 2-second context and strict short-gap joining. Perfect exact keep-mask editing raises export F1 to 92.63%, but rally event F1 stays at the automatic model's 75.95%. That export gain must not be presented as a score-tracking improvement.

## Required padding sensitivity: development cohort

Seed means of metrics pooled across recordings. Export durations are seconds; difference is model minus human. The target remains 2 seconds for every policy.

| Policy | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production | 0s | 65.50% | 95.75% | 77.79% | 3489.608 | 2387.151 | 1102.457 |
| Production | 1s | 69.40% | 98.30% | 81.36% | 4223.508 | 3031.151 | 1192.357 |
| Production | 2s | 72.45% | 99.27% | 83.76% | 4943.025 | 3675.151 | 1267.874 |
| Production | 3s | 75.02% | 99.49% | 85.54% | 5637.197 | 4323.263 | 1313.934 |
| Apply all boundaries | 0s | 80.01% | 84.76% | 82.31% | 2529.310 | 2387.151 | 142.159 |
| Apply all boundaries | 1s | 81.62% | 92.95% | 86.92% | 3294.257 | 3031.151 | 263.106 |
| Apply all boundaries | 2s | 82.88% | 96.04% | 88.97% | 4037.135 | 3675.151 | 361.984 |
| Apply all boundaries | 3s | 83.94% | 97.56% | 90.24% | 4769.811 | 4323.263 | 446.548 |
| Review: undo whole parent | 0s | 69.79% | 95.75% | 80.74% | 3274.992 | 2387.151 | 887.841 |
| Review: undo whole parent | 1s | 73.02% | 98.30% | 83.80% | 4013.719 | 3031.151 | 982.568 |
| Review: undo whole parent | 2s | 75.51% | 99.23% | 85.76% | 4741.042 | 3675.151 | 1065.891 |
| Review: undo whole parent | 3s | 77.46% | 99.49% | 87.10% | 5457.069 | 4323.263 | 1133.806 |
| Review: undo individual removal | 0s | 74.47% | 95.75% | 83.78% | 3069.350 | 2387.151 | 682.199 |
| Review: undo individual removal | 1s | 76.97% | 98.30% | 86.34% | 3807.958 | 3031.151 | 776.807 |
| Review: undo individual removal | 2s | 78.71% | 99.23% | 87.79% | 4547.964 | 3675.151 | 872.813 |
| Review: undo individual removal | 3s | 80.20% | 99.49% | 88.81% | 5270.625 | 4323.263 | 947.362 |

## Required padding sensitivity: recording-044

| Policy | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production | 0s | 72.69% | 97.07% | 83.13% | 428.843 | 321.146 | 107.696 |
| Production | 1s | 76.33% | 97.14% | 85.49% | 504.341 | 397.146 | 107.196 |
| Production | 2s | 78.64% | 97.45% | 87.04% | 580.479 | 470.645 | 109.834 |
| Production | 3s | 83.01% | 97.77% | 89.79% | 640.478 | 544.644 | 95.834 |
| Apply all boundaries | 0s | 85.24% | 80.02% | 82.55% | 301.473 | 321.146 | -19.673 |
| Apply all boundaries | 1s | 85.89% | 90.96% | 88.35% | 386.221 | 397.146 | -10.924 |
| Apply all boundaries | 2s | 86.91% | 94.85% | 90.71% | 464.236 | 470.645 | -6.409 |
| Apply all boundaries | 3s | 89.75% | 96.87% | 93.17% | 538.233 | 544.644 | -6.410 |
| Review: undo whole parent | 0s | 72.69% | 97.07% | 83.13% | 428.843 | 321.146 | 107.696 |
| Review: undo whole parent | 1s | 76.33% | 97.14% | 85.49% | 504.341 | 397.146 | 107.196 |
| Review: undo whole parent | 2s | 78.64% | 97.45% | 87.04% | 580.479 | 470.645 | 109.834 |
| Review: undo whole parent | 3s | 83.01% | 97.77% | 89.79% | 640.478 | 544.644 | 95.834 |
| Review: undo individual removal | 0s | 77.48% | 97.07% | 86.17% | 402.330 | 321.146 | 81.184 |
| Review: undo individual removal | 1s | 80.57% | 97.14% | 88.08% | 477.829 | 397.146 | 80.683 |
| Review: undo individual removal | 2s | 82.81% | 97.45% | 89.54% | 551.217 | 470.645 | 80.572 |
| Review: undo individual removal | 3s | 86.70% | 97.77% | 91.90% | 613.216 | 544.644 | 68.572 |
| Exact core editing (upper bound) | 0s | 87.08% | 97.07% | 91.80% | 357.956 | 321.146 | 36.810 |
| Exact core editing (upper bound) | 1s | 87.60% | 97.14% | 92.13% | 439.454 | 397.146 | 42.309 |
| Exact core editing (upper bound) | 2s | 87.98% | 97.45% | 92.47% | 518.843 | 470.645 | 48.199 |
| Exact core editing (upper bound) | 3s | 90.60% | 97.77% | 94.04% | 586.842 | 544.644 | 42.199 |
| Exact export mask (upper bound) | 0s | 87.51% | 97.07% | 92.04% | 356.206 | 321.146 | 35.060 |
| Exact export mask (upper bound) | 1s | 87.60% | 97.14% | 92.13% | 439.454 | 397.146 | 42.309 |
| Exact export mask (upper bound) | 2s | 88.26% | 97.45% | 92.63% | 517.218 | 470.645 | 46.574 |
| Exact export mask (upper bound) | 3s | 90.60% | 97.77% | 94.04% | 586.842 | 544.644 | 42.199 |

## Artifacts and verification

Development artifacts: `private-reference-0119`.

`results-v2.json` SHA-256: `edd689767e7216f9b21ec7b611f99a71096cf04d37803d571138a2986be0cc26`. It preserves `results-v1.json` and adds individual-removal undo plus both playback-context conventions. Forty-four frozen source-file identities matched the prior study; this number is source files, not videos.

recording-044 artifacts: `private-reference-0120`. `results-v2.json` adds practical individual-removal undo while preserving `results.json`. `playback-context-sensitivity-v1.json` records both playback conventions.

recording-044 `results-v2.json` SHA-256: `e1f64694f686c7b56a5ddfaf871f9ba7f9ec585eab0170083ed28dc131608a12`.

An independent audit reproduced all 24 development video/seed granular attachment graphs, 6,048 duration comparisons across four policies and four padding cases, 96 per-recording event matchings and 12 pooled event-metric cases, including pooled accounting and workload. On recording-044, it reproduced 336 duration comparisons, all six event-match counts with an independent matching implementation, exact fragment decisions and geometric event restoration. Practical individual-removal undo uses only binary gold answers, with no gold-derived endpoint coordinates.

Scripts: `scripts/evaluate-boundary-default-review-cohort.py`, `scripts/evaluate-boundary-fragment-veto-cohort.py`, `scripts/audit-boundary-default-review-cohort.py`, `scripts/evaluate-labeling-boundary-default-review.py`.

The existing labeling UI remains a production-preserving preview; this experiment does not switch the product to automatic boundary application or edit the human draft.
