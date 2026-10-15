# Ten serving-side precision experiments: results

**The centered temporal head is the leading candidate.** At the 98% selection target it achieved **98.19% accepted precision with 16.53% of serves reviewed**, averaged across three seeds. The predeclared three-seed agreement policy achieved **98.79% precision with 23.04% reviewed**: **736 correct and nine wrong automatic answers among 745 accepted serves; 223 of 968 went to review**. Production has not changed.

The study reused the same **968 reviewed true non-beach serves, 26 recordings, and eight source groups**. Each outer source is excluded from model fitting and inner policy selection. Correctness is measured against reviewed human serving-side labels. Precision is correct automatic side answers divided by all automatic side answers; review rate is reviews divided by all 968 serves. These results do not include fresh rally detection or the separate serve/no-serve gate.

All ten experiments were registered before their new results were examined. Neural results below are means across seeds 3407, 1729, and 20260918; fractional counts are averages. Logistic experiments are deterministic and evaluated once. A requested target is an inner-validation selection condition, not a guarantee of held-source precision.

## Each experiment at the main 98% target

| # | Experiment | Measured precision | Reviews / 968 | Review rate | Correct automatic / 968 | Wrong automatic / 968 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 | Four absolute quality averages | 96.96% | 507 | 52.38% | 447 | 14 |
| 02 | Quality measured in three phases | 96.99% | 470 | 48.55% | 483 | 15 |
| 03 | Raw court-flow features | 97.72% | 485 | 50.10% | 472 | 11 |
| 04 | Projected embedding changes + specialist | 97.63% | 207 | 21.38% | 743 | 18 |
| 05 | Quality and motion-evidence interactions | 96.97% | 407 | 42.05% | 544 | 17 |
| 06 | Cross-fitted review head | 97.76% | 478 | 49.38% | 479 | 11 |
| 07 | Review head + feature interactions | 98.20% | 467 | 48.24% | 492 | 9 |
| 08 | Temporal neural head + quality | 97.51% | 315 | 32.54% | 636.7 | 16.3 |
| 09 | Centered temporal neural head + quality | 98.19% | 160 | 16.53% | 793.3 | 14.7 |
| 10 | Frame-difference neural head + quality | 97.41% | 325.7 | 33.64% | 625.7 | 16.7 |

## Both-side and source guardrails

Source-average precision includes only groups with accepted predictions. Fully reviewed groups contribute zero automatic coverage and remain included in the review workload.

| # | Near precision | Far precision | Source-average precision | Infeasible outer policies |
| --- | ---: | ---: | ---: | ---: |
| 01 | 97.28% | 96.57% | 95.71% | 1 |
| 02 | 97.76% | 96.09% | 96.15% | 1 |
| 03 | 98.56% | 96.59% | 97.09% | 1 |
| 04 | 98.18% | 97.07% | 97.34% | 0 |
| 05 | 97.95% | 95.90% | 96.40% | 1 |
| 06 | 98.36% | 96.76% | 97.04% | 1 |
| 07 | 98.06% | 98.44% | 96.97% | 1 |
| 08 | 97.60% | 97.41% | 96.30% | 0 |
| 09 | 98.17% | 98.22% | 97.97% | 0 |
| 10 | 97.67% | 97.18% | 96.46% | 0 |

Experiments 07 and 09 clear 98% pooled precision and both-side precision in their main summaries. Their source-average precision is 96.97% and 97.97%, respectively, so **no individual-head summary meets every registered 98% guardrail**. The centered head is close on the source-average guardrail, but rounding it up would be misleading.

**The predeclared centered-head consensus does meet all these empirical guardrails:** pooled 98.79%, near 98.89%, far 98.70%, and source-average 98.49%. Consensus is a separate policy requiring all three seed policies to accept and agree; the main mean-seed result has not been replaced by a favorable seed.

## Matched and historical controls

| Model or policy at the 98% target | Precision | Reviews / 968 | Review rate | Wrong automatic / 968 |
| --- | ---: | ---: | ---: | ---: |
| Previous fixed-L2 specialist | 98.10% | 547 | 56.51% | 8 |
| Previous BCE neural hybrid | 97.66% | 313.7 | 32.40% | 15.3 |
| Previous asymmetric-cost neural hybrid | 97.49% | 275.3 | 28.44% | 17.3 |
| Rank-only specialist with matched L2 search | 97.21% | 502 | 51.86% | 13 |
| Centered head: mean seeds | 98.19% | 160 | 16.53% | 14.7 |
| Centered head: three-seed agreement | 98.79% | 223 | 23.04% | 9 |

The original production uncertainty rule is a different operating point: in the earlier held-source replay it accepted 931 serves, reviewed 37 (3.82%), and made 45 wrong automatic side answers, for 95.17% precision. The new centered consensus has 36 fewer wrong automatic answers and 186 more reviews on the same-sized known-serve population. Exact deployed-weight in-sample measurements are not substituted for held-source results. See the [historical policy analysis](serving-side-review-2026-10-13.md).

The matched L2-search control matters: tuning regularization alone did not improve held-source precision. Feature experiments 01 through 07 should be assessed against that control as well as the historical fixed-L2 specialist.

## Requested 95%, 98%, and 99% targets

Each cell is **measured accepted precision / percentage of all serves reviewed**. These are main deterministic or mean-seed results.

| # | 95% target | 98% target | 99% target |
| --- | ---: | ---: | ---: |
| 01 | 95.53% / 9.81% | 96.96% / 52.38% | 98.68% / 84.30% |
| 02 | 95.18% / 7.75% | 96.99% / 48.55% | 97.62% / 82.64% |
| 03 | 95.58% / 8.88% | 97.72% / 50.10% | 97.96% / 79.75% |
| 04 | 94.44% / 1.55% | 97.63% / 21.38% | 98.68% / 60.85% |
| 05 | 95.05% / 8.26% | 96.97% / 42.05% | 98.26% / 76.24% |
| 06 | 93.57% / 26.14% | 97.76% / 49.38% | 97.49% / 75.31% |
| 07 | 94.65% / 13.12% | 98.20% / 48.24% | 97.95% / 74.79% |
| 08 | 95.70% / 6.20% | 97.51% / 32.54% | 97.90% / 69.97% |
| 09 | 95.25% / 2.03% | 98.19% / 16.53% | 98.13% / 53.58% |
| 10 | 95.14% / 6.44% | 97.41% / 33.64% | 98.26% / 66.01% |

For the recommended **centered-head three-seed agreement policy**:

| Selection target | Measured precision | Accepted | Reviews / 968 | Review rate | Wrong automatic |
| --- | ---: | ---: | ---: | ---: | ---: |
| 95% | 96.19% | 919 | 49 | 5.06% | 35 |
| 98% | 98.79% | 745 | 223 | 23.04% | 9 |
| 99% | 98.52% | 337 | 631 | 65.19% | 5 |

None of the ten experiments reached 99% pooled precision in its main 99%-target summary. The centered 99%-target agreement policy also missed 99% and required far more reviews than its 98%-target policy. Targets select checkpoints and thresholds independently inside the inner folds; different resulting source mixes can make observed outer precision nonmonotonic. Do not relabel a 99%-target result as a separately validated 98% operating point after inspecting outer outcomes.

## Where the leading policy still fails

| Held source | Serves | Accepted | Reviews | Wrong automatic | Accepted precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-group-002 | 46 | 34 | 12 | 2 | 94.12% |
| source-group-003 | 40 | 23 | 17 | 0 | 100.00% |
| source-group-005 | 195 | 142 | 53 | 3 | 97.89% |
| source-group-007 | 145 | 131 | 14 | 0 | 100.00% |
| source-group-009 | 74 | 67 | 7 | 0 | 100.00% |
| source-group-010 | 39 | 29 | 10 | 0 | 100.00% |
| source-group-012 | 56 | 33 | 23 | 1 | 96.97% |
| source-group-013 | 373 | 286 | 87 | 3 | 98.95% |

The nine remaining errors occur in four sources. The weakest accepted-source precision is 94.12%, so the overall result does not establish 98% for every recording. The descriptive pooled Wilson 95% lower bound is 97.72%; this is not a source-cluster or repeated-model-selection guarantee. The individual centered seeds achieved 98.15%, 98.36%, and 98.05% pooled precision, with 159, 175, and 146 reviews. The improvement is present across seeds, while each seed still has a weaker side below 98%.

## Interpretation of the ten experiments

1. **Absolute quality averages:** missed 98% and reviewed over half the serves. Weak motion alone is not a sufficient review signal across sources.
2. **Phase-specific quality:** retained more automatic answers than averages, but precision remained below 97%.
3. **Raw court flow:** improved precision relative to the matched rank-only control, but still missed 98% with about half the serves reviewed.
4. **Projected embedding changes:** a useful lower-complexity direction, reaching 97.63% with only 21.38% reviewed; it still missed the main precision target.
5. **Quality/evidence interactions:** reduced reviews relative to simpler quality features, without reaching the required precision.
6. **Cross-fitted correctness head:** honestly training on specialist mistakes did not by itself reach 98%.
7. **Correctness feature interactions:** reached 98.20% pooled precision, but needed 48.24% review and remained uneven across sources.
8. **Temporal neural head plus quality:** essentially unchanged precision with more reviews than the previous cost-sensitive head.
9. **Centered temporal head:** the strongest improvement in both precision and automatic coverage. Centering subtracts the valid-frame mean embedding within each 13-frame window. This is consistent with suppressing static scene appearance while retaining temporal changes; it does not prove the physical cause of every corrected error.
10. **Frame differences:** lost the benefit of centering; taking consecutive differences alone did not improve the main tradeoff.

Experiments 08 through 10 use identical architecture, labels, quality inputs, seeds, and training settings; the temporal input transform changes. Thus the 08-versus-09 comparison isolates the benefit of the centered representation more directly than a comparison against an older study. The frozen DINO-distilled MobileNetV3-Large encoder and previous/future frame context are retained.

## Next steps

1. Freeze the centered three-seed agreement candidate and its selected policies, then confirm it once on untouched recordings/source groups. These ten arms and prior studies share a development population, so another independent confirmation is needed before promotion.
2. Run the complete production candidate pipeline, including actual rally detections, false-serve cases, the serve/no-serve review gate, and recording ranks computed over the production candidate population. Report end-to-end review workload and wrong automatic answers.
3. Inspect the nine remaining confident mistakes for label ambiguity, anchor timing, occlusion, and court geometry. Use any resulting feature hypothesis in a new registered study; do not tune the current thresholds to those nine cases.
4. If the candidate holds up, export and verify the centered heads and three-seed agreement rule in the target runtime, including valid-frame masking and the 12 raw quality features. Keep 98% as the product target; this run does not support a reliable 99% policy.

## Verification, resources, and artifacts

Nine focused qualification tests passed in the research environment, including source exclusion for correctness teachers, invalid-frame masking, static-offset invariance, and serialization. Tests skip cleanly in environments without the optional research PyTorch dependency.

The [independent audit](serving-side-precision-iterations-audit-2026-10-14.json) passed. It independently rebuilt the frozen features, checked all 93 teacher partitions, 296 CPU fitting partitions, 324 nested neural fits, and nine final neural refits, and verified 2,257 artifact hashes. It replayed all 408 outer policies, 51 final policies, and aggregate/consensus metrics. Reloading 140 selected checkpoints reproduced their logits exactly; all 27 final neural configurations passed CPU/GPU parity with maximum score difference 2.19e-7. Protected labels were not opened.

Training, evaluation, and refitting used one fitting process with two CPU threads. Peak sampled resident memory was 2.09 GiB, and the process had no sampled swap allocation. WSL recorded zero swap-outs and 12 pages of global swap-in during the run; minimum WSL available memory was 12.41 GiB. The subsequent independent audit ran separately and peaked at 1.90 GiB resident memory.

Durable artifacts use ledger `private-reference-0240`; SSD work storage uses `private-reference-0241`. The initial implementation and its stopped run are preserved under `private-reference-0238` / `private-reference-0239`; an empty-batch final-export fix triggered a full restart before any nested outcome comparison. No initial fits were reused.

Frozen registration: `6dddbd097a11b1d37825e2d99929da1563b5d10092be24cd3f08660a7b301796`. All exact data identities, row predictions, weights, locations, and execution logs remain private. The public results contain stable indexes and aggregates. No commit, publication, or production model change was performed.

The ledger-backed privacy scan passed for all 49 modified or untracked serving-side research files, including the aggregate results and this report. This was a working-file check; branch-history publication auditing remains required before any future commit or publication. The private receipt is retained with the study artifacts.

The [complete aggregate JSON](serving-side-precision-iterations-2026-10-14.json) contains all seeds, source groups, sides, consensus results, and final research-model configurations. See the [registered protocol](serving-side-precision-iterations-2026-10-14.md) for objectives, features, folds, and selection rules. Logistic policy field `epoch` is the shared selector's candidate ID: 1/2/3 map to L2 1/0.1/0.01; neural policies use actual epochs 5/15/30/60.
