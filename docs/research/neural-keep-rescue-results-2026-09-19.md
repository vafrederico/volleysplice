**Keep-head rescue results - 2026-09-19**

Adding short proposals from the existing keep head passed the preregistered retention-recovery screen in both compact TCN and DINO+TCN for the reviewed-export cohort. This is a replicated development improvement in retained play. F1 gains were small, and no cohort or architecture passed the separate standard F1 improvement screen. No production model was changed.

The comparison completed after the preceding short-boost study and both independent audits. It used all three prospectively selected cohorts, both architectures and three seeds: eighteen unchanged baseline cells and eighteen decoder candidates, with **zero new fits**. [The prospective protocol](neural-keep-rescue-protocol-2026-09-19.md) fixes the hypothesis, option order, supervision, folds, metric and guardrails. The existing baseline epoch and live/serve/end decoder remained fixed; only the no-op or keep threshold was selected from inner predictions. These results do not combine keep rescue with short-rally loss weighting or the separate shorter-context experiment.

For reviewed-export compact TCN, mean complete rally losses fell from 35.67 to 28.33 (20.56%), and complete losses among rallies no longer than three seconds fell from 23.00 to 18.33 (20.29%). DINO+TCN fell from 21.00 to 15.00 (28.57%) overall and from 17.33 to 12.67 (26.92%) for short rallies. Both reductions occurred in all three seeds. Mean incomplete losses, which include both complete and partial losses, also fell: 72.00 to 65.67 for compact and 54.00 to 49.00 for DINO. Partial-loss counts alone rose by one in both models as some previously missing rallies became partly retained.

The reviewed-export compact gain traded a small amount of padded precision (0.873379 to 0.867922) for recall (0.924917 to 0.934221), adding 75.56 export seconds across the eight evaluation recordings per seed on average. DINO recall rose from 0.958035 to 0.964738 with nearly unchanged precision (0.879177 to 0.879117), adding 46.08 export seconds. Longer-rally core recall rose by 0.005676 and 0.003385 respectively after two-second padding and strictly-less-than-three-second gap joining. Event F1 changed by +0.000043 and +0.023220 respectively.

All 72 inner threshold selections met the 0.95 recall floor; 26 selected the explicit no-op. This inner constraint does not guarantee outer recall: the reviewed-export compact outer recalls were 0.925432, 0.943907 and 0.933324, all below 0.95. The corresponding DINO recalls were 0.967926, 0.954024 and 0.972265. Absolute outer recall was a declared diagnostic, not an added acceptance criterion.

Draft-cohort DINO passed retention recovery, while draft compact missed the 20% overall and short-loss reduction targets. Exact-only rescue produced little benefit: both architectures missed the 20% targets, and compact also lacked reductions in two seeds. The different baseline loss floors and four repeatedly inspected source groups limit broader conclusions. Keep coverage is not invertible; two-second erosion remains a proposal heuristic. Unioning proposals preserves retained core time, while precision and event matching can worsen.

All numbers below use the same exact-label and ignored-range revision. The primary product case is symmetric two-second padding. Model and human exports receive identical padding and strictly less than three-second gap joining before ignored-time subtraction; excluded spans are never rejoined. Each seed pools duration numerators and denominators across all eight recordings before scoring. Tables report means of those three seed scores; padding sensitivity holds predictions fixed. A gap of exactly three seconds remains a cut.

**All cohort and architecture results, loss slices, and required padding sensitivity**

| Cohort | Architecture | Variant | F1_padP_coreR | R_core | P_pad | Event F1 |
|---|---|---|---:|---:|---:|---:|
| exact | tcn | reference | 0.876335 | 0.945632 | 0.816646 | 0.653114 |
| exact | tcn | selected | 0.875957 | 0.946011 | 0.815680 | 0.651473 |
| exact | dino_tcn | reference | 0.908143 | 0.966573 | 0.858208 | 0.736671 |
| exact | dino_tcn | selected | 0.908144 | 0.967269 | 0.857700 | 0.736436 |
| draft | tcn | reference | 0.901112 | 0.912967 | 0.889566 | 0.684021 |
| draft | tcn | selected | 0.901194 | 0.920092 | 0.883212 | 0.681165 |
| draft | dino_tcn | reference | 0.926419 | 0.960708 | 0.894879 | 0.760401 |
| draft | dino_tcn | selected | 0.927514 | 0.965430 | 0.892800 | 0.762254 |
| reviewed_export | tcn | reference | 0.898235 | 0.924917 | 0.873379 | 0.686176 |
| reviewed_export | tcn | selected | 0.899714 | 0.934221 | 0.867922 | 0.686219 |
| reviewed_export | dino_tcn | reference | 0.916319 | 0.958035 | 0.879177 | 0.714516 |
| reviewed_export | dino_tcn | selected | 0.919449 | 0.964738 | 0.879117 | 0.737736 |

| Cohort | Architecture | F1 change | Recall change | F1 screen | Retention screen |
|---|---|---:|---:|---|---|
| exact | tcn | -0.000378 | +0.000378 | False | False |
| exact | dino_tcn | +0.000002 | +0.000696 | False | False |
| draft | tcn | +0.000083 | +0.007125 | False | False |
| draft | dino_tcn | +0.001095 | +0.004722 | False | True |
| reviewed_export | tcn | +0.001479 | +0.009304 | False | True |
| reviewed_export | dino_tcn | +0.003130 | +0.006703 | False | True |

| Cohort | Architecture | Variant | Slice | Complete losses | Partial losses | Core recall |
|---|---|---|---|---:|---:|---:|
| exact | tcn | reference | all | 21.00 | 31.33 | 0.945632 |
| exact | tcn | reference | duration_le_3s | 14.33 | 1.00 | 0.803905 |
| exact | tcn | reference | duration_gt_3s | 6.67 | 30.33 | 0.954102 |
| exact | tcn | reference | ace | 4.00 | 0.00 | 0.850994 |
| exact | tcn | reference | service_fault | 7.33 | 0.67 | 0.788217 |
| exact | tcn | selected | all | 20.67 | 31.33 | 0.946011 |
| exact | tcn | selected | duration_le_3s | 14.00 | 1.00 | 0.807811 |
| exact | tcn | selected | duration_gt_3s | 6.67 | 30.33 | 0.954269 |
| exact | tcn | selected | ace | 3.67 | 0.00 | 0.858737 |
| exact | tcn | selected | service_fault | 7.33 | 0.67 | 0.788217 |
| exact | dino_tcn | reference | all | 14.33 | 26.33 | 0.966573 |
| exact | dino_tcn | reference | duration_le_3s | 10.67 | 0.67 | 0.858322 |
| exact | dino_tcn | reference | duration_gt_3s | 3.67 | 25.67 | 0.973042 |
| exact | dino_tcn | reference | ace | 2.67 | 0.00 | 0.913650 |
| exact | dino_tcn | reference | service_fault | 6.33 | 0.33 | 0.816779 |
| exact | dino_tcn | selected | all | 13.33 | 26.67 | 0.967269 |
| exact | dino_tcn | selected | duration_le_3s | 9.67 | 1.00 | 0.870671 |
| exact | dino_tcn | selected | duration_gt_3s | 3.67 | 25.67 | 0.973042 |
| exact | dino_tcn | selected | ace | 2.33 | 0.00 | 0.927031 |
| exact | dino_tcn | selected | service_fault | 6.00 | 0.33 | 0.825945 |
| draft | tcn | reference | all | 35.33 | 43.00 | 0.912967 |
| draft | tcn | reference | duration_le_3s | 22.33 | 1.67 | 0.670441 |
| draft | tcn | reference | duration_gt_3s | 13.00 | 41.33 | 0.927460 |
| draft | tcn | reference | ace | 5.67 | 2.00 | 0.724233 |
| draft | tcn | reference | service_fault | 13.00 | 0.00 | 0.603760 |
| draft | tcn | selected | all | 30.33 | 44.00 | 0.920092 |
| draft | tcn | selected | duration_le_3s | 19.67 | 2.00 | 0.708383 |
| draft | tcn | selected | duration_gt_3s | 10.67 | 42.00 | 0.932743 |
| draft | tcn | selected | ace | 4.67 | 2.00 | 0.763096 |
| draft | tcn | selected | service_fault | 11.33 | 0.00 | 0.660201 |
| draft | dino_tcn | reference | all | 20.67 | 28.33 | 0.960708 |
| draft | dino_tcn | reference | duration_le_3s | 17.67 | 1.00 | 0.767658 |
| draft | dino_tcn | reference | duration_gt_3s | 3.00 | 27.33 | 0.972244 |
| draft | dino_tcn | reference | ace | 4.00 | 0.33 | 0.861017 |
| draft | dino_tcn | reference | service_fault | 11.00 | 1.00 | 0.687086 |
| draft | dino_tcn | selected | all | 16.00 | 28.33 | 0.965430 |
| draft | dino_tcn | selected | duration_le_3s | 13.33 | 1.33 | 0.823888 |
| draft | dino_tcn | selected | duration_gt_3s | 2.67 | 27.00 | 0.973888 |
| draft | dino_tcn | selected | ace | 3.33 | 0.33 | 0.882142 |
| draft | dino_tcn | selected | service_fault | 8.00 | 0.67 | 0.773479 |
| reviewed_export | tcn | reference | all | 35.67 | 36.33 | 0.924917 |
| reviewed_export | tcn | reference | duration_le_3s | 23.00 | 1.00 | 0.666763 |
| reviewed_export | tcn | reference | duration_gt_3s | 12.67 | 35.33 | 0.940344 |
| reviewed_export | tcn | reference | ace | 7.33 | 0.67 | 0.685292 |
| reviewed_export | tcn | reference | service_fault | 12.67 | 0.33 | 0.605480 |
| reviewed_export | tcn | selected | all | 28.33 | 37.33 | 0.934221 |
| reviewed_export | tcn | selected | duration_le_3s | 18.33 | 1.33 | 0.736768 |
| reviewed_export | tcn | selected | duration_gt_3s | 10.00 | 36.00 | 0.946020 |
| reviewed_export | tcn | selected | ace | 5.67 | 1.00 | 0.749404 |
| reviewed_export | tcn | selected | service_fault | 10.67 | 0.33 | 0.675315 |
| reviewed_export | dino_tcn | reference | all | 21.00 | 33.00 | 0.958035 |
| reviewed_export | dino_tcn | reference | duration_le_3s | 17.33 | 0.33 | 0.773367 |
| reviewed_export | dino_tcn | reference | duration_gt_3s | 3.67 | 32.67 | 0.969070 |
| reviewed_export | dino_tcn | reference | ace | 3.67 | 0.00 | 0.874755 |
| reviewed_export | dino_tcn | reference | service_fault | 11.33 | 0.33 | 0.677620 |
| reviewed_export | dino_tcn | selected | all | 15.00 | 34.00 | 0.964738 |
| reviewed_export | dino_tcn | selected | duration_le_3s | 12.67 | 0.67 | 0.835594 |
| reviewed_export | dino_tcn | selected | duration_gt_3s | 2.33 | 33.33 | 0.972455 |
| reviewed_export | dino_tcn | selected | ace | 3.00 | 0.33 | 0.897652 |
| reviewed_export | dino_tcn | selected | service_fault | 7.33 | 0.67 | 0.798196 |

| Cohort | Architecture | Variant | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact | tcn | reference | 0 | 0.765845 | 0.852482 | 0.806646 | 2658.708 | 2387.151 | +271.557 |
| exact | tcn | reference | 1 | 0.795127 | 0.921658 | 0.853624 | 3353.911 | 3031.151 | +322.760 |
| exact | tcn | reference | 2 | 0.816646 | 0.945632 | 0.876335 | 4021.708 | 3675.151 | +346.557 |
| exact | tcn | reference | 3 | 0.835328 | 0.959630 | 0.893107 | 4676.392 | 4323.263 | +353.129 |
| exact | tcn | selected | 0 | 0.765460 | 0.853067 | 0.806710 | 2661.761 | 2387.151 | +274.610 |
| exact | tcn | selected | 1 | 0.794507 | 0.922127 | 0.853482 | 3358.964 | 3031.151 | +327.813 |
| exact | tcn | selected | 2 | 0.815680 | 0.946011 | 0.875957 | 4029.553 | 3675.151 | +354.402 |
| exact | tcn | selected | 3 | 0.834483 | 0.959900 | 0.892752 | 4684.944 | 4323.263 | +361.681 |
| exact | dino_tcn | reference | 0 | 0.816768 | 0.881879 | 0.845296 | 2597.803 | 2387.151 | +210.652 |
| exact | dino_tcn | reference | 1 | 0.840683 | 0.946227 | 0.888661 | 3266.161 | 3031.151 | +235.010 |
| exact | dino_tcn | reference | 2 | 0.858208 | 0.966573 | 0.908143 | 3919.750 | 3675.151 | +244.599 |
| exact | dino_tcn | reference | 3 | 0.874969 | 0.975343 | 0.921720 | 4558.564 | 4323.263 | +235.301 |
| exact | dino_tcn | selected | 0 | 0.816027 | 0.883125 | 0.845485 | 2603.897 | 2387.151 | +216.746 |
| exact | dino_tcn | selected | 1 | 0.840031 | 0.946836 | 0.888533 | 3273.083 | 3031.151 | +241.932 |
| exact | dino_tcn | selected | 2 | 0.857700 | 0.967269 | 0.908144 | 3928.672 | 3675.151 | +253.521 |
| exact | dino_tcn | selected | 3 | 0.874383 | 0.976179 | 0.921747 | 4570.517 | 4323.263 | +247.254 |
| draft | tcn | reference | 0 | 0.851745 | 0.806490 | 0.828492 | 2260.428 | 2387.151 | -126.723 |
| draft | tcn | reference | 1 | 0.873180 | 0.882489 | 0.877802 | 2880.844 | 3031.151 | -150.307 |
| draft | tcn | reference | 2 | 0.889566 | 0.912967 | 0.901112 | 3480.808 | 3675.151 | -194.343 |
| draft | tcn | reference | 3 | 0.902050 | 0.928750 | 0.915205 | 4081.986 | 4323.263 | -241.277 |
| draft | tcn | selected | 0 | 0.848565 | 0.810119 | 0.828872 | 2279.194 | 2387.151 | -107.957 |
| draft | tcn | selected | 1 | 0.867882 | 0.888429 | 0.877962 | 2921.689 | 3031.151 | -109.462 |
| draft | tcn | selected | 2 | 0.883212 | 0.920092 | 0.901194 | 3540.914 | 3675.151 | -134.237 |
| draft | tcn | selected | 3 | 0.895491 | 0.936681 | 0.915558 | 4157.378 | 4323.263 | -165.885 |
| draft | dino_tcn | reference | 0 | 0.862008 | 0.857032 | 0.859006 | 2376.458 | 2387.151 | -10.693 |
| draft | dino_tcn | reference | 1 | 0.880697 | 0.936378 | 0.907409 | 3019.692 | 3031.151 | -11.459 |
| draft | dino_tcn | reference | 2 | 0.894879 | 0.960708 | 0.926419 | 3651.008 | 3675.151 | -24.143 |
| draft | dino_tcn | reference | 3 | 0.907573 | 0.970727 | 0.937921 | 4272.878 | 4323.263 | -50.385 |
| draft | dino_tcn | selected | 0 | 0.860104 | 0.859973 | 0.859564 | 2389.644 | 2387.151 | +2.493 |
| draft | dino_tcn | selected | 1 | 0.878728 | 0.940632 | 0.908388 | 3045.544 | 3031.151 | +14.393 |
| draft | dino_tcn | selected | 2 | 0.892800 | 0.965430 | 0.927514 | 3689.528 | 3675.151 | +14.377 |
| draft | dino_tcn | selected | 3 | 0.905388 | 0.975548 | 0.939014 | 4323.901 | 4323.263 | +0.638 |
| reviewed_export | tcn | reference | 0 | 0.830479 | 0.821559 | 0.825682 | 2363.353 | 2387.151 | -23.798 |
| reviewed_export | tcn | reference | 1 | 0.854204 | 0.899546 | 0.876082 | 2988.461 | 3031.151 | -42.690 |
| reviewed_export | tcn | reference | 2 | 0.873379 | 0.924917 | 0.898235 | 3587.164 | 3675.151 | -87.987 |
| reviewed_export | tcn | reference | 3 | 0.888981 | 0.937577 | 0.912490 | 4180.139 | 4323.263 | -143.124 |
| reviewed_export | tcn | selected | 0 | 0.826380 | 0.828060 | 0.826967 | 2393.669 | 2387.151 | +6.518 |
| reviewed_export | tcn | selected | 1 | 0.849482 | 0.907991 | 0.877604 | 3041.361 | 3031.151 | +10.210 |
| reviewed_export | tcn | selected | 2 | 0.867922 | 0.934221 | 0.899714 | 3662.725 | 3675.151 | -12.426 |
| reviewed_export | tcn | selected | 3 | 0.883360 | 0.947312 | 0.914109 | 4278.611 | 4323.263 | -44.652 |
| reviewed_export | dino_tcn | reference | 0 | 0.829112 | 0.841643 | 0.833538 | 2436.058 | 2387.151 | +48.907 |
| reviewed_export | dino_tcn | reference | 1 | 0.858556 | 0.931370 | 0.892574 | 3074.997 | 3031.151 | +43.846 |
| reviewed_export | dino_tcn | reference | 2 | 0.879177 | 0.958035 | 0.916319 | 3691.872 | 3675.151 | +16.721 |
| reviewed_export | dino_tcn | reference | 3 | 0.896269 | 0.968848 | 0.930699 | 4303.233 | 4323.263 | -20.030 |
| reviewed_export | dino_tcn | selected | 0 | 0.828454 | 0.849092 | 0.837039 | 2458.731 | 2387.151 | +71.580 |
| reviewed_export | dino_tcn | selected | 1 | 0.858257 | 0.937547 | 0.895369 | 3108.586 | 3031.151 | +77.435 |
| reviewed_export | dino_tcn | selected | 2 | 0.879117 | 0.964738 | 0.919449 | 3737.956 | 3675.151 | +62.805 |
| reviewed_export | dino_tcn | selected | 3 | 0.896123 | 0.975111 | 0.933579 | 4361.150 | 4323.263 | +37.887 |

All threshold candidates, infeasible fallbacks, source-group effects, original-event loss identities and separate replication/interaction screens are retained in JSON. Three seeds and four source groups support descriptive development comparisons. No protected test or production promotion.

**Acceptance checks**

| Cohort | Architecture | Recovery checks passed | Recovery + feasibility | Standard F1 + feasibility | F1-positive seeds | F1-positive groups | Failed recovery conditions |
|---|---|---|---|---|---|---|---|
| exact | tcn | 6/10 | fail | fail | 1 | 0/4 | meanCompleteLossesReducedAtLeast20Percent, meanShortCompleteLossesReducedAtLeast20Percent, atLeastTwoSeedsReduceCompleteLosses, atLeastTwoSeedsReduceShortCompleteLosses |
| exact | dino_tcn | 8/10 | fail | fail | 1 | 1/4 | meanCompleteLossesReducedAtLeast20Percent, meanShortCompleteLossesReducedAtLeast20Percent |
| draft | tcn | 8/10 | fail | fail | 2 | 1/4 | meanCompleteLossesReducedAtLeast20Percent, meanShortCompleteLossesReducedAtLeast20Percent |
| draft | dino_tcn | 10/10 | pass | fail | 3 | 1/4 | none |
| reviewed_export | tcn | 10/10 | pass | fail | 2 | 2/4 | none |
| reviewed_export | dino_tcn | 10/10 | pass | fail | 3 | 3/4 | none |

The standard F1 screen requires a mean gain of at least 0.02; every comparison missed that target. Reviewed-export DINO passed the other standard criteria, while compact also lacked a majority of improving source groups (two of four). The ten recovery checks remain distinct: F1, overall recall, longer-rally recall and event-F1 regression limits; at least 20% fewer overall and short complete losses; no increase in overall or short incomplete losses; and reductions in at least two seeds for both complete-loss counts. All six comparisons passed the F1, overall-recall, long-recall, event-F1 and incomplete-loss regression checks.

| Cohort | Architecture | Long-rally recall delta | Event-F1 delta | Incomplete-loss delta | Short incomplete-loss delta | Overall recovery seeds | Short recovery seeds |
|---|---|---|---|---|---|---|---|
| exact | tcn | +0.000168 | -0.001641 | -0.33 | -0.33 | 1/3 | 1/3 |
| exact | dino_tcn | +0.000000 | -0.000235 | -0.67 | -0.67 | 2/3 | 2/3 |
| draft | tcn | +0.005283 | -0.002856 | -4.00 | -2.33 | 3/3 | 3/3 |
| draft | dino_tcn | +0.001644 | +0.001853 | -4.67 | -4.00 | 3/3 | 3/3 |
| reviewed_export | tcn | +0.005676 | +0.000043 | -6.33 | -4.33 | 3/3 | 3/3 |
| reviewed_export | dino_tcn | +0.003385 | +0.023220 | -5.00 | -4.33 | 3/3 | 3/3 |

**Every seed at the primary padding**

| Cohort | Architecture | Seed | F1 delta | Selected R_core | Complete losses | Short complete losses | Incomplete losses |
|---|---|---|---|---|---|---|---|
| exact | tcn | 3407 | +0.000000 | 0.944893 | 27 -> 27 | 17 -> 17 | 55 -> 55 |
| exact | tcn | 1729 | +0.000016 | 0.954094 | 14 -> 14 | 10 -> 10 | 44 -> 44 |
| exact | tcn | 20260918 | -0.001151 | 0.939046 | 22 -> 21 | 16 -> 15 | 58 -> 57 |
| exact | dino_tcn | 3407 | -0.000109 | 0.952067 | 14 -> 14 | 11 -> 11 | 55 -> 55 |
| exact | dino_tcn | 1729 | -0.000454 | 0.994426 | 6 -> 5 | 5 -> 4 | 13 -> 12 |
| exact | dino_tcn | 20260918 | +0.000568 | 0.955315 | 23 -> 21 | 16 -> 14 | 54 -> 53 |
| draft | tcn | 3407 | -0.000725 | 0.921789 | 29 -> 27 | 19 -> 17 | 75 -> 72 |
| draft | tcn | 1729 | +0.000450 | 0.905663 | 42 -> 40 | 29 -> 27 | 91 -> 88 |
| draft | tcn | 20260918 | +0.000524 | 0.932825 | 35 -> 24 | 19 -> 15 | 69 -> 63 |
| draft | dino_tcn | 3407 | +0.000024 | 0.974680 | 16 -> 14 | 14 -> 12 | 44 -> 42 |
| draft | dino_tcn | 1729 | +0.003078 | 0.966871 | 23 -> 12 | 19 -> 9 | 49 -> 39 |
| draft | dino_tcn | 20260918 | +0.000183 | 0.954739 | 23 -> 22 | 20 -> 19 | 54 -> 52 |
| reviewed_export | tcn | 3407 | +0.000696 | 0.925432 | 37 -> 33 | 23 -> 21 | 74 -> 70 |
| reviewed_export | tcn | 1729 | -0.002198 | 0.943907 | 28 -> 26 | 20 -> 18 | 62 -> 60 |
| reviewed_export | tcn | 20260918 | +0.005939 | 0.933324 | 42 -> 26 | 26 -> 16 | 80 -> 67 |
| reviewed_export | dino_tcn | 3407 | +0.002309 | 0.967926 | 16 -> 12 | 14 -> 11 | 43 -> 40 |
| reviewed_export | dino_tcn | 1729 | +0.006792 | 0.954024 | 33 -> 20 | 26 -> 16 | 66 -> 55 |
| reviewed_export | dino_tcn | 20260918 | +0.000290 | 0.972265 | 14 -> 13 | 12 -> 11 | 53 -> 52 |

**Every source group: mean paired F1 change**

| Cohort | Architecture | source-group-005 | source-group-007 | source-group-009 | source-group-012 |
|---|---|---|---|---|---|
| exact | tcn | +0.000000 | -0.000903 | +0.000000 | +0.000000 |
| exact | dino_tcn | +0.000818 | -0.000223 | -0.000151 | -0.000166 |
| draft | tcn | +0.007089 | -0.003614 | -0.000118 | +0.000000 |
| draft | dino_tcn | +0.006745 | -0.001326 | -0.000236 | -0.000449 |
| reviewed_export | tcn | +0.006471 | +0.004142 | -0.000443 | -0.001745 |
| reviewed_export | dino_tcn | +0.000491 | +0.007186 | +0.003962 | -0.000364 |

These source-group differences explain the cautious interpretation: reviewed-export compact improved F1 in two groups, whereas DINO improved in three. A pooled retention success does not imply improvement for every recording setup.

**Threshold selection and transfer**

| Cohort | Architecture | No-op | .35 | .50 | .65 | .80 |
|---|---|---|---|---|---|---|
| exact | tcn | 8 | 1 | 0 | 1 | 2 |
| exact | dino_tcn | 7 | 4 | 0 | 1 | 0 |
| draft | tcn | 4 | 5 | 2 | 1 | 0 |
| draft | dino_tcn | 4 | 3 | 2 | 2 | 1 |
| reviewed_export | tcn | 2 | 1 | 5 | 3 | 1 |
| reviewed_export | dino_tcn | 1 | 4 | 4 | 3 | 0 |

| Cohort | Retention replication | Standard-F1 replication | F1 difference-in-differences | Overall recovery difference-in-differences | Short recovery difference-in-differences |
|---|---|---|---|---|---|
| exact | fail | fail | +0.000380 | +0.67 | +0.67 |
| draft | fail | fail | +0.001012 | -0.33 | +1.67 |
| reviewed_export | pass | fail | +0.001651 | -1.33 | +0.00 |

Difference-in-differences is DINO's within-architecture effect minus compact's within-architecture effect. It is descriptive and need not be positive. Reviewed-export recovered an average of 4.67 previously completely lost short rallies with each architecture, so its short-recovery interaction is zero while retention replication passes. This tests the keep-rescue decoder change against each architecture's own baseline; it does not establish that the earlier loss-weighting change transferred.

**Verification and reproduction**

The no-op preflight passed 144 exact recording replays from 72 refit files. The independent final audit verified all 18 immutable baseline payloads, 36 metric replays, 72 outer choices, all 360 inner options, 72 selected outer decodes and 1,872 independent padding/scope rows. It revalidated 288 original fit completions and 576 original NPZ artifacts. There are no fresh fit or neural prediction NPZ files. Existing tensor/scaler proofs remain bound to the preceding study; this decoder audit does not rerun every neural forward pass. The fixed helper code, registered sources, labels, inputs and no-op result payloads stayed unchanged. No numerical auditor or recipe corrections were needed.

The first source-archive attempt stopped before creating output because the prepared archive reader omitted the seven PowerShell resource-observer sources included in the verified preceding archive. Its evidence allowlist was extended only for `.ps1` files under the existing resource-monitor/recovery provenance paths, retaining all path and forbidden-directory checks. Thirteen archive tests passed in WSL and Windows, including acceptance of those monitor sources and rejection of unrelated PowerShell or forbidden paths. The original failed attempt, before/after helper sources, tests and repair proof are preserved under `archive-evidence-repair/`. This packaging correction changed no registered analysis source or quality result.

The CPU decoder run took 37.50 seconds and the independent audit 73.90 seconds on this machine while the separate context experiment was running. These timings exclude model training, video processing, DINO feature extraction and phone/browser execution, and are not an inference benchmark. The fourth output head adds 65 parameters relative to the three-head research derivative and requires no new feature extractor; a production implementation would still need its own target-device validation.

Immutable NAS artifacts are under `private-reference-0151` (WSL `private-reference-0152`). The study directory contains `preregistration.json`, `report.json`, `summary.json`, `summary.md`, `loss-identities.json`, registered sources and the source snapshot. The root contains the prospective selection/protocol, source copies, operational runner, all four phase logs and their immutable start/completion receipts. Exact event loss identities, full candidate scores, per-seed/source-group metrics and threshold evidence are retained in the JSON artifacts.

| Evidence | SHA-256 |
|---|---|
| protocol-initial.md | 553fc18e3939809b7a18fc8a85e6bfc1f5d50003c599ec7c08a288d44742b197 |
| prospective-selection.json | c288fdc3f8117b4d01a283064fd9b9654b17637eb32e67a5c4166dd8f4f13199 |
| run-phase.py | 9619570b7d0020fe299ef9b12b4fdff2b849035d7e28473e21a0d07430b67680 |
| preflight/report.json | e849b4e9ab26d29a119c9e74bdd05bdd3274dace6f25fa059555bcbb4b3e1204 |
| study/preregistration.json | a98160859b71384e6a8a3d531285c685a17273b0c602dd3665961db79801930d |
| study/report.json | 0c5cbe3167d09c8c62b94dbb9651184db93cbee710f0d1ec3c9725867b30b01c |
| study/summary.json | cca2a76d2c583e43d95aae86e0299bcc4d6a901bfb6e9f3666b78f5e5c09f6fa |
| study/loss-identities.json | cdf774eee901da8c113b43e4cef24cc4adac06d141fb09fd4c8f48e1548cebd0 |
| preflight-completed.json | 859df80f8ac1913babc60479e381c661afa7b76f0a35eaa59e99c9c1331e71cc |
| registration-completed.json | 614120018e592af2cafd1cfe5dd4acdb660d822e0dc54cb09d8b2edafabfb049 |
| run-completed.json | aaa61ef6a7fa24b12b9258865086e22d9c524c90c746355ae653991067c24a19 |
| summary-completed.json | c6e9138c8b42a06436b60a96f97723f2f230c0e9be442dbe89c395f127a2ead6 |
| archive-evidence-repair/repair-result.json | 50787cee7ee9d4ecebde366b28a527f9dbde0effee4ce8a02e9129f8c6406739 |

Registered contract SHA-256: `876ffc7a8b4fe983daad4527ed6065ee6a59335e862de20e713a642ea6dfaba7`. The preceding study source archive is bound at `40b8ff7948c1b287ce5c18df3c89901d778e1207f39fe534227fd129b45dbfb5`. This follow-up's `study/source-snapshot/research-source.zip`, manifest and verification report provide the exact source/document overlay and independently checked archive hashes. The archive's own checksum is recorded separately to avoid a self-referential document hash.

The protocol, registration, run and final independent audit are complete. The protected test remains unopened, beach remains excluded, and no model or decoder was promoted to production.
