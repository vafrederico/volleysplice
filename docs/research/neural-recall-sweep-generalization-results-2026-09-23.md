# Neural rally detection: findings overview

**AUDITED DATA: final numerical audit and interactive-report UI checks passed.**

There is no uniform neural winner. At the 95% calibration floor, FP32 and padding of 2s before and 2s after, every original neural family trades higher precision for lower recall and F1 than Production default across all 19 reviewed exports. The 8 production-clean exports reverse the F1 comparison, but still include neural training material. Independent-source results are presented separately.

Published findings overview (ledger `private-reference-0201`) | Interactive recall sweep and complete tables (ledger `private-reference-0202`)

## Population changes the comparison

The study inferred 42 non-beach recordings and scored 34: 9 exact, 6 reviewed-draft and 19 reviewed-export recordings. The other 8 are inference-only because their labels are absent, unvalidated or insufficiently exhaustive; they have no accuracy score. Counts are recordings, not independent matches.

**Scored recording counts by production-exposure filter**

| Filter | Exact | Reviewed draft | Reviewed export |
| --- | ---: | ---: | ---: |
| All labeled | 9 | 6 | 19 |
| No production training | 1 | 0 | 8 |
| No production fit or calibration | 0 | 0 | 8 |

> Zeros in this matrix mean empty populations, not zero accuracy. Empty scopes remain unavailable. Reviewed-draft gold measures reviewed live-interval time coverage, not exact semantic rally boundaries. Keep all three label policies separate.

Across all 19 reviewed-export recordings at floor 95, every original neural family has higher precision but lower recall and F1 than Production default. On the 8 production-clean exports, every original neural F1 exceeds Production default, but only AV-transformer has higher export recall. This reversal shows why population labels must stay visible.

> Export comparison scope: 95% calibration floor; FP32 original-corpus means over all 3 complete draws; padding 2s before and 2s after. Production configurations are fixed. The 8-export no-production-training and strict no-fit-or-calibration scopes have identical membership.

**All labeled exports: 19 recordings / 3 source groups**

| Model | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| AV-TCN | 95.783 | 81.531 | 88.081 |
| DINO-TCN | 96.758 | 79.199 | 87.099 |
| Mobile-TCN | 96.694 | 81.254 | 88.296 |
| Distilled Mobile-TCN | 97.304 | 79.635 | 87.585 |
| AV-transformer | 86.341 | 89.498 | 87.468 |
| DINO-transformer | 96.799 | 77.239 | 85.703 |
| Production default | 85.267 | 91.954 | 88.485 |
| Production union | 74.851 | 94.682 | 83.606 |

**Production-clean exports: 8 recordings / 2 source groups**

| Model | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| AV-TCN | 96.339 | 85.000 | 90.314 |
| DINO-TCN | 97.670 | 83.111 | 89.793 |
| Mobile-TCN | 97.894 | 85.643 | 91.358 |
| Distilled Mobile-TCN | 97.290 | 84.865 | 90.646 |
| AV-transformer | 84.854 | 92.319 | 88.029 |
| DINO-transformer | 97.328 | 81.815 | 88.624 |
| Production default | 82.132 | 91.514 | 86.570 |
| Production union | 71.774 | 94.661 | 81.644 |

> These 8 production-clean exports still include neural fitting sources. They span 2 source groups, and their higher neural F1 is not an independent generalization result. Production filtering removes the 11 Aug16 recordings and changes the population from 19 exports to 8; the change does not estimate a causal penalty from production training exposure. Common-unseen indoor source and September 17 results are shown separately below.

## Common-unseen indoor source and September 17: original families

At the 95% calibration floor, all six original neural families have higher time-based F1 than both production configurations on these two sources, principally through precision. All six retain less of the desired human export than either production configuration. There is no uniform winner.

> Comparison scope: 95% calibration floor; padding 2s before and 2s after; indoor source exact vs September 17 reviewed export; all 3 registered original-corpus draws (neural means). Production configurations are fixed comparators.

**Exact indoor source: P_pad / R_core / F1_padP_coreR**

| Model | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| AV-TCN | 89.582 | 96.929 | 93.087 |
| DINO-TCN | 85.364 | 99.428 | 91.648 |
| Mobile-TCN | 91.820 | 96.917 | 94.290 |
| Distilled Mobile-TCN | 87.950 | 97.676 | 92.546 |
| AV-transformer | 72.719 | 94.948 | 81.972 |
| DINO-transformer | 87.409 | 98.604 | 92.628 |
| Production default | 67.746 | 98.244 | 80.193 |
| Production union | 62.269 | 99.661 | 76.648 |

**Reviewed September 17 export: export-union P / R / F1**

| Model | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| AV-TCN | 93.117 | 89.045 | 91.011 |
| DINO-TCN | 96.068 | 86.582 | 91.070 |
| Mobile-TCN | 95.127 | 88.128 | 91.491 |
| Distilled Mobile-TCN | 95.415 | 87.162 | 91.098 |
| AV-transformer | 82.537 | 90.672 | 85.943 |
| DINO-transformer | 93.830 | 83.736 | 88.269 |
| Production default | 77.194 | 96.504 | 85.776 |
| Production union | 67.269 | 98.140 | 79.824 |

> Neural rows are original-corpus means over all 3 registered draws. Production rows are fixed comparators repeated unchanged across recall floors; they are not selected at floor 95. Production default is shipped aggressive suppression; production union is the shipped unsuppressed ensemble. Table order is not a ranking.

The exact source is one 39-rally indoor source recording: indoor-source-05 (source-group-008). The reviewed-export source is recording-044 (source-group-001). These are two independent source groups, one per label policy; they cannot establish broad real-world generalization.

## Extra export labels: recovery with a precision cost

For Mobile-TCN at floor 95, export-derived training raises desired-export recall from 85.375% to 96.123%, while precision falls from 97.047% to 88.187%. Export F1 rises from 90.787% to 91.960%. Exact rally-core recall falls from 97.429% to 96.115%.

**Mobile-TCN: exact indoor source policy**

| Variant | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| Expanded medium | 81.610 | 97.429 | 88.303 |
| Export labels in training | 83.234 | 96.115 | 88.730 |
| Export labels in selection | 80.601 | 97.739 | 88.137 |

**Mobile-TCN: reviewed-export policy**

| Variant | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| Expanded medium | 97.047 | 85.375 | 90.787 |
| Export labels in training | 88.187 | 96.123 | 91.960 |
| Export labels in selection | 88.314 | 95.027 | 91.488 |

These Mobile contrasts use the same 4 registered draws, all feasible at floor 95. Moving export sources into selection reduces export recall by 1.096 points and export F1 by 0.472 points versus export training; exact recall rises by 1.624 points while exact F1 falls by 0.593 points.

> At floor 95 only Mobile-TCN has all 4 export-selection draws feasible. Each other family has 3/4, so its complete mean is unavailable. Even Mobile export training at 96.123% export recall remains below production default at 96.504% and production union at 98.140%.

Across jointly fully feasible floors, export training improves exact F1 for DINO-TCN, Mobile-TCN, distilled Mobile and DINO-transformer versus expanded medium. AV-TCN loses exact and export F1; AV-transformer improves export F1 while losing exact F1. Export recovery does not establish better rally separation.

## Training diversity, distillation and transformers

- Expanding the medium pool alone lowers floor-95 exact F1 for the three ordinary TCN families. Export F1 changes are small and mixed. More footage does not uniformly resolve recall.
- Increasing expanded-medium to expanded-large training improves distilled Mobile and DINO-transformer F1 over every jointly feasible floor. Distilled Mobile gains 1.342 to 4.357 exact-F1 points and 0.774 to 1.375 export-F1 points; DINO-transformer gains 5.512 to 6.567 exact-F1 points and 2.888 to 4.437 export-F1 points. Ordinary Mobile improves at most floors.
- Wider validation helps some families and harms others. Distilled Mobile does not consistently surpass its teacher or ordinary Mobile. AV-transformer has the weakest high-floor feasibility here. DINO-transformer responds to larger training and export supervision, but the split variation does not establish a general transformer advantage.

> Data amount is confounded with supervision and optimizer budget. AV-TCN used 691 to 936 optimizer steps for expanded medium versus 3561 to 3782 for export training under the same epoch schedule. Export training promotes 18 reviewed recordings to approximate four-head supervision; these are not 674 newly verified exact rally boundaries. Export selection also changes fit/selection membership.

Matched sweep contrasts require identical source IDs and complete registered draw sets on both sides. Floors reuse predictions and candidates; positive-floor counts are not independent replications or significance tests.

## Calibration feasibility and unseen recall

**Original-corpus feasibility**

| Family | Highest floor: all draws feasible | Feasible draws at 99% |
| --- | ---: | ---: |
| AV-TCN | 98% | 2/3 |
| DINO-TCN | 99% | 3/3 |
| Mobile-TCN | 98% | 1/3 |
| Distilled Mobile-TCN | 98% | 2/3 |
| AV-transformer | 95% | 0/3 |
| DINO-transformer | 97% | 0/3 |

The 90 to 100% floors constrain FP32 inner selection. They do not guarantee independent-video recall. Within the tested checkpoint/decoder candidate bank, no family/variant has every registered draw feasible at 100%; a few individual draws qualify. Missing aggregates at 99% or 100% are a material result, not zero scores or compute failures.

> This bank-limited infeasibility does not mean full time recall is impossible. Retaining the entire evaluated video is an outside-bank reference that keeps all evaluated wanted time, with unwanted-footage, precision and export-size costs. It is not a scored or selected model in this study, and full time recall does not supply correct individual rally or serve boundaries.

Expanded-medium Mobile passes the 99% calibration floor in all 4 draws but retains 97.429% of exact core time and 88.551% of desired export time on the independent panels. Original DINO-TCN at floor 99 retains 99.489% of exact core time and 90.548% of desired export time, with event recall of 82.051%.

> Retained-time recall and rally-count recall answer different questions. At floor 95, original DINO-TCN retains 99.428% of core time but loses one entire rally per draw; mean event recall is 82.906% and event F1 is 79.343%. Time scores do not establish correct rally splitting or serve/start boundaries.

Original DINO-TCN floor-95 exact precision spans 74.534% to 93.907% across its 3 draws. High mean core recall can hide large precision variation.

## DINO precision transfer and mobile readiness

This is embedding-extractor transfer only: FP16 and mixed INT8 do not retrain or quantize the temporal heads. FP32 checkpoint/decoder selections stay fixed. Comparisons cover available complete-draw common-unseen aggregates at floors 90 to 100 and target padding.

- FP16: largest absolute exact/export time-F1 differences are 0.067 points for DINO-TCN and 0.127 for DINO-transformer. Event F1 still changes by up to approximately 0.65 points, so small duration differences do not prove identical rally splitting.
- INT8: the largest observed time-F1 losses are -1.860 points for DINO-TCN (export training, exact, floor 98) and -3.924 for DINO-transformer (original, exact, floor 96). DINO-transformer also loses up to 10.811 event-F1 points (original exact, floor 92).
- At original floor 95, INT8 DINO-transformer exact F1 falls from 92.628% to 89.696% while core recall rises from 98.604% to 99.099%. Export F1 changes from 88.269% to 88.346%. Recall alone conceals precision and boundary changes.

> No physical-phone qualification is available. FP16 was CUDA; mixed INT8 was native CPU, and the tested INT8 desktop-browser path failed numerical parity. Detector candidates have no qualified trained rally heads, so there is no fair detector recall sweep here. These results establish no mobile/browser deployment readiness.

## Scope, means and report usage

- Exact P/R/F1 means P_pad / R_core / F1_padP_coreR: precision is the share of model export overlapping padded human export, and recall is the share of human rally-core time retained by model export. Reviewed-export P/R/F1 instead compares the fixed human export union. Keep these policies separate; do not combine their recall claims.
- Target padding is 2 seconds before and 2 seconds after. Join positive padded gaps strictly below 3 seconds. Ignored spans remain outside evaluation. This overview fixes the target case; the interactive report retains all four padding sensitivity cases.
- Within each draw, durations are pooled on the declared recording scope before metrics are calculated. Displayed multi-draw metrics are means of every registered complete draw. Mean F1 is not recomputed from mean P and mean R.
- Original rows use 3 draws; randomized/proxy variants use 4. Original-versus-randomized rows are descriptive, not matched four-draw comparisons. Some splits share physical fits, so the draws are correlated. An infeasible floor stays missing; never average only surviving draws.
- indoor source is clean of known production fitting but has same-group production-suppression calibration exposure. All and no-production-training rows are identical here. The strict no-production-fit-or-calibration exact panel is empty; that is not a zero score.
- recording-044 is known clean of production fitting and calibration within pinned lineage. Historical diagnostic viewing is distinct from training exposure. All-labeled panels include neural fitting/calibration sources; production-clean filtering alone does not make them neural-generalization tests.

In the interactive report, keep source IDs, label policy, precision, padding and registered draw set matched. Inspect the sweep and feasibility first, then individual draws, padding sensitivity and exposure filters. Treat all-labeled panels as descriptive when they include neural fitting or calibration sources.

> Selection was frozen before external evaluation. These evaluated protected videos have now been used for diagnosis and are no longer untouched future tests. This overview reports no statistical significance, release recommendation or selected held-out checkpoint.

## Next research steps

- Add independent non-beach source groups for each label policy, with a new untouched evaluation set reserved before further comparisons.
- Separate data diversity, supervision quality and training budget using matched optimizer-update budgets and predeclared selection rules.
- Review missed rallies, splitting and serve/start boundaries alongside retained-time and export coverage. Validate approximate export cores against exact boundaries where score tracking needs them.
- Qualify ball/person detector features and their rally heads before claiming recall gains; compare with matched qualified baselines.
- Measure runtime, memory and numerical parity on physical phones and the intended browser path, including embedding precision and rally/event behavior.
- Use these findings to define experiments. Do not select a checkpoint from the evaluated held-out outcomes or start training from this overview.

## Reproduction and validation

The [registered protocol](neural-recall-sweep-generalization-protocol-2026-09-23.md) defines the comparison. Selections were frozen before the additional-video outcomes were opened. The experiment records retain fit, calibration, inference and evaluation membership for every task, alongside production training/calibration lineage.

All generated data, checkpoints, caches, browser profiles and report artifacts remain on the NAS:
`private-reference-0141`.
The `final-report-v1` directory contains the final JSON, HTML, audit, source-bound findings extract, operational evidence and publication receipts.

- 162 registered logical experiments; 270 neural precision evaluations plus 2 fixed production comparators.
- Independent numerical audit passed for 319,352 report rows and 7,936 metric scopes. The final report preserves the audited candidate's scientific content.
- Real-report browser QA passed 36 views and checked all 12,454,728 packed table cells against the saved JSON. Its downloadable JSON restores the exact source bytes.
- The overview preserves all 8 source tables. Desktop and 390px browser checks, table parity and manual visual review passed.
- The original failed browser-QA launch is preserved separately; it failed before browser checks because its NAS output directory was missing. Creating that directory allowed the unchanged QA script to complete successfully.
- These are offline accuracy results and report-UI checks. They do not qualify phone inference, browser model parity, latency, memory or thermal behavior. No production model was promoted.

Final report SHA-256: `eee5505fbc7f8aaa737f4756f03c40f804ce9637bda11d29ca39e17862be8bf4`.

Audited scientific-content SHA-256: `d1c61a103b6ba2eec64cdf303406cbb16fcd7b5b1c04a5d35b18ed0b379ca2e1`.

Publication used the shared Publish Review Artifacts skill (ledger `private-reference-0147`); exact uploaded identities and returned artifact URLs are retained in the NAS receipts.
