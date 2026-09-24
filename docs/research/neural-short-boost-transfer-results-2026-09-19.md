# Short-rally weighting and DINO transfer: completed results

The bounded short-rally boost passed all ten preregistered retention-recovery checks for **DINO + TCN trained with the reviewed-export cohort**, against both its unweighted baseline and the matched global-positive control. It did **not** pass the same recovery screen for the compact TCN, so the preregistered cross-architecture replication screen failed in all three cohorts. All 18 standard F1 improvement screens also failed. These are development results; no production model was promoted and the protected test was not opened.

For reviewed-export DINO, the boost increased `F1_padP_coreR` from 0.916319 to 0.925127 and retained core play time from 95.8035% to 96.9545%. Mean completely missed rallies fell from 21.00 to 13.67, a 34.92% reduction; completely missed rallies lasting at most three seconds fell from 17.33 to 11.00, a 36.54% reduction. Longer-rally retained play time improved from 96.9070% to 97.6493%, and event F1 improved from 0.714516 to 0.741651. All three seeds reduced complete losses overall and among short rallies. These loss counts are measured after the product export padding and gap joining.

The compact TCN's reviewed-export F1 increased from 0.898235 to 0.903077, but the missed-rally reductions were much smaller: 35.67 to 33.00 overall and 23.00 to 21.67 among short rallies. The number of rallies with any missing play increased from 72.00 to 75.00. Its recovery screen therefore failed. This is evidence of a useful DINO recovery result in this cohort, not evidence that the improvement transfers between architectures.

The highest absolute mean F1 among these 18 configurations was **draft-cohort DINO with the global-positive control**, at 0.930707 (`R_core` 0.969599). That is a different observation from the reviewed-export boost passing its recovery criteria. It did not pass either improvement screen against its own baseline; ranking configurations by absolute score does not establish a successful intervention or authorize deployment.

The experiment, [execution record](./neural-short-boost-transfer-execution-2026-09-19.md), and [timestamp amendment](./neural-transfer-pts-amendment-2026-09-19.md) bind the recipe. Every evaluation uses the same eight exact-label videos, 322 original rallies, and four development source groups. There are three seeds (3407, 1729, 20260918), two architectures, three training cohorts, and three loss arms: 54 result cells.

| Training cohort | Available inputs |
|---|---|
| exact | Eight exact-label videos |
| draft | Those eight plus three videos with approximate rally intervals |
| reviewed_export | Those eleven plus seven raw videos paired with manually reviewed project exports; actual kept ranges supervise the auxiliary keep head |

The largest cohort contains 18 videos across seven source groups, about 5.319 hours and 76,607 feature timestamps. Exact evaluation groups are held out across every training tier in each nested fold. Beach and the protected source group remain excluded. The compact four-head TCN has 29,700 trainable parameters; DINO + TCN has 46,868 trainable downstream parameters, excluding its frozen image encoder. The DINO feature extraction cost remains relevant to desktop inference and was not benchmarked here.

The short boost doubles existing supervised positive live ticks only for original rallies lasting at most three seconds. The global control gives every supervised positive live tick in a recording the multiplier `1 + short_positive_ticks / positive_ticks`, matching the short boost's total positive loss weight for that recording. The live labels, negative and unknown ticks, auxiliary keep targets, optimizer, model architecture, and evaluation decoder search remain unchanged between arms.

The declared product target is symmetric two-second padding. `P_pad` compares the model and human padded export unions; `R_core` measures the human core play retained by the model export; `F1_padP_coreR` is their harmonic mean. Ranges are clipped, padded identically, merged, and joined only when a positive gap is **strictly less than three seconds**; exactly three seconds remains a cut. Ignored spans are subtracted from all evaluation unions after joining, and ranges are not rejoined across those spans. Padding sensitivity at zero, one, two, and three seconds is mandatory, but only the fixed two-second case ranks this study.

Within each seed, intersection numerators and duration denominators are pooled across recordings before calculating ratios and F1. The tables report the arithmetic mean of those three seed-level pooled metrics and export durations. They do not average per-video F1 or pool three copies of the same labels as independent observations.

| Cohort | Architecture | Loss arm | F1 | R_core | P_pad | Event F1 |
|---|---|---|---:|---:|---:|---:|
| exact | tcn | baseline | 0.876335 | 0.945632 | 0.816646 | 0.653114 |
| exact | tcn | short_boost | 0.878432 | 0.943495 | 0.822032 | 0.639530 |
| exact | tcn | global_control | 0.872786 | 0.943620 | 0.812441 | 0.618708 |
| exact | dino_tcn | baseline | 0.908143 | 0.966573 | 0.858208 | 0.736671 |
| exact | dino_tcn | short_boost | 0.913565 | 0.936468 | 0.894164 | 0.705940 |
| exact | dino_tcn | global_control | 0.901551 | 0.922386 | 0.882452 | 0.705120 |
| draft | tcn | baseline | 0.901112 | 0.912967 | 0.889566 | 0.684021 |
| draft | tcn | short_boost | 0.898712 | 0.910462 | 0.887916 | 0.662388 |
| draft | tcn | global_control | 0.898485 | 0.910954 | 0.886437 | 0.684812 |
| draft | dino_tcn | baseline | 0.926419 | 0.960708 | 0.894879 | 0.760401 |
| draft | dino_tcn | short_boost | 0.929435 | 0.964565 | 0.896847 | 0.760022 |
| draft | dino_tcn | global_control | 0.930707 | 0.969599 | 0.895049 | 0.771587 |
| reviewed_export | tcn | baseline | 0.898235 | 0.924917 | 0.873379 | 0.686176 |
| reviewed_export | tcn | short_boost | 0.903077 | 0.924092 | 0.883046 | 0.694493 |
| reviewed_export | tcn | global_control | 0.896269 | 0.917221 | 0.876916 | 0.688156 |
| reviewed_export | dino_tcn | baseline | 0.916319 | 0.958035 | 0.879177 | 0.714516 |
| reviewed_export | dino_tcn | short_boost | 0.925127 | 0.969545 | 0.884748 | 0.741651 |
| reviewed_export | dino_tcn | global_control | 0.920171 | 0.966848 | 0.877905 | 0.736682 |

The standard F1 screen requires at least +0.02 mean F1, positive F1 effects in at least two seeds and a majority of source groups, and no more than 0.005 mean core-recall regression. No contrast achieved the +0.02 requirement. The separate recovery screen requires at least 20% reductions in both overall and short complete losses, no increase in overall or short incomplete losses, complete-loss reductions in at least two seeds for both slices, and bounded regressions in F1, overall recall, longer-rally recall (each at most 0.005), and event F1 (at most 0.01). All 216 selected inner decoder/checkpoint choices met the 0.95 recall eligibility floor; none used the infeasible fallback.

| Within-architecture comparison | F1 delta | R_core delta | F1 screen | Recovery screen |
|---|---:|---:|---|---|
| exact/tcn/short_boost-minus-baseline | +0.002097 | -0.002137 | False | False |
| exact/tcn/short_boost-minus-global_control | +0.005646 | -0.000124 | False | False |
| exact/tcn/global_control-minus-baseline | -0.003550 | -0.002013 | False | False |
| exact/dino_tcn/short_boost-minus-baseline | +0.005423 | -0.030105 | False | False |
| exact/dino_tcn/short_boost-minus-global_control | +0.012015 | +0.014082 | False | False |
| exact/dino_tcn/global_control-minus-baseline | -0.006592 | -0.044187 | False | False |
| draft/tcn/short_boost-minus-baseline | -0.002400 | -0.002505 | False | False |
| draft/tcn/short_boost-minus-global_control | +0.000227 | -0.000492 | False | False |
| draft/tcn/global_control-minus-baseline | -0.002627 | -0.002013 | False | False |
| draft/dino_tcn/short_boost-minus-baseline | +0.003016 | +0.003857 | False | False |
| draft/dino_tcn/short_boost-minus-global_control | -0.001272 | -0.005033 | False | False |
| draft/dino_tcn/global_control-minus-baseline | +0.004288 | +0.008891 | False | False |
| reviewed_export/tcn/short_boost-minus-baseline | +0.004842 | -0.000825 | False | False |
| reviewed_export/tcn/short_boost-minus-global_control | +0.006808 | +0.006871 | False | False |
| reviewed_export/tcn/global_control-minus-baseline | -0.001966 | -0.007696 | False | False |
| reviewed_export/dino_tcn/short_boost-minus-baseline | +0.008808 | +0.011510 | False | True |
| reviewed_export/dino_tcn/short_boost-minus-global_control | +0.004956 | +0.002697 | False | True |
| reviewed_export/dino_tcn/global_control-minus-baseline | +0.003852 | +0.008813 | False | False |

For reviewed-export DINO, the boost also passes against the global control: mean complete losses fall from 20.33 to 13.67 (32.79%), short complete losses from 16.67 to 11.00 (34.00%), and incomplete losses from 45.00 to 43.00. F1 increases by 0.004956 and overall core recall by 0.002697. Longer-rally recall decreases by 0.001547 against this control, within the preregistered 0.005 limit. This qualification matters: the longer-rally result improves against the unweighted baseline, but not against every comparator.

| Boost versus its baseline | Failed recovery conditions |
|---|---|
| exact / compact TCN | Insufficient overall and short complete-loss reduction; more incomplete losses; too few seeds reducing complete losses; event F1 regression |
| exact / DINO + TCN | Overall and longer-rally recall regressions; worse overall and short complete/incomplete losses; too few seeds reducing complete losses; event F1 regression |
| draft / compact TCN | More incomplete losses; longer-rally recall regression; event F1 regression |
| draft / DINO + TCN | Insufficient overall and short complete-loss reduction; more overall and short incomplete losses; too few seeds reducing complete losses |
| reviewed_export / compact TCN | Insufficient overall and short complete-loss reduction; more incomplete losses |
| reviewed_export / DINO + TCN | None; also passes against the matched global control |

The exact-cohort DINO result illustrates why F1 alone is insufficient: the boost raises F1 by 0.005423 while overall retained core play drops by 0.030105 and longer-rally recall by 0.028802. Draft-cohort compact training reduces missed points in all three seeds, but increases incomplete rallies and lowers longer-rally recall by 0.007616. These recall values concern retained play seconds after the two-second export padding and strict short-gap joining, rather than simply whether a rally received any detection.

| Cohort | Architecture | Arm | Complete losses | Short complete losses (<=3s) | Partial losses | Incomplete losses | Longer-rally R_core |
|---|---|---|---:|---:|---:|---:|---:|
| exact | tcn | baseline | 21.00 | 14.33 | 31.33 | 52.33 | 0.954102 |
| exact | tcn | short_boost | 21.33 | 14.33 | 34.00 | 55.33 | 0.951760 |
| exact | tcn | global_control | 23.33 | 16.33 | 32.33 | 55.67 | 0.953550 |
| exact | dino_tcn | baseline | 14.33 | 10.67 | 26.33 | 40.67 | 0.973042 |
| exact | dino_tcn | short_boost | 21.33 | 14.00 | 41.00 | 62.33 | 0.944239 |
| exact | dino_tcn | global_control | 27.00 | 17.33 | 46.67 | 73.67 | 0.932039 |
| draft | tcn | baseline | 35.33 | 22.33 | 43.00 | 78.33 | 0.927460 |
| draft | tcn | short_boost | 28.00 | 17.00 | 53.33 | 81.33 | 0.919844 |
| draft | tcn | global_control | 34.67 | 20.67 | 40.67 | 75.33 | 0.923722 |
| draft | dino_tcn | baseline | 20.67 | 17.67 | 28.33 | 49.00 | 0.972244 |
| draft | dino_tcn | short_boost | 22.00 | 19.33 | 27.67 | 49.67 | 0.978291 |
| draft | dino_tcn | global_control | 17.33 | 14.67 | 22.33 | 39.67 | 0.979279 |
| reviewed_export | tcn | baseline | 35.67 | 23.00 | 36.33 | 72.00 | 0.940344 |
| reviewed_export | tcn | short_boost | 33.00 | 21.67 | 42.00 | 75.00 | 0.938176 |
| reviewed_export | tcn | global_control | 39.00 | 25.00 | 38.67 | 77.67 | 0.933949 |
| reviewed_export | dino_tcn | baseline | 21.00 | 17.33 | 33.00 | 54.00 | 0.969070 |
| reviewed_export | dino_tcn | short_boost | 13.67 | 11.00 | 29.33 | 43.00 | 0.976493 |
| reviewed_export | dino_tcn | global_control | 20.33 | 16.67 | 24.67 | 45.00 | 0.978040 |

Incomplete means completely or partially missing an original rally's evaluable core. The full JSON preserves exact rally identities, boundary metrics, ace/service-fault slices, per-seed results, and raw-core versus padded-export coverage. Fractional loss counts in this document arise from averaging three integer seed counts.

Effects vary by source group. Reviewed-export DINO's boost improves F1 in all three seeds but only two of four source groups, against either control. Its mean F1 change versus baseline is shown below. The recovery screen did not include a majority-of-groups gate; the standard F1 screen did and fails it. Compact reviewed-export F1 improves in all four groups, yet its rally-recovery conditions fail.

| Reviewed-export source group | Compact boost F1 delta vs baseline | DINO boost F1 delta vs baseline | DINO R_core delta vs baseline |
|---|---:|---:|---:|
| source-group-005 | +0.010586 | +0.019238 | +0.013824 |
| source-group-007 | +0.003011 | -0.000627 | -0.012055 |
| source-group-009 | +0.002400 | +0.013544 | +0.036248 |
| source-group-012 | +0.002920 | -0.010877 | +0.000000 |

Replication was defined before training as feasible recovery against both own controls in both architectures within the same cohort. Exact and draft pass none of those four comparisons; reviewed_export passes the two DINO comparisons and fails the two compact comparisons. Consequently, replication fails in all three cohorts. The reviewed-export F1 difference-in-differences is +0.003966 versus baseline and -0.001852 versus global control. These interactions are descriptive: equal benefits in both architectures would give zero interaction and could still replicate.

The following sensitivity table keeps every model's predictions fixed. Human export duration is identical across compared models; duration differences are model minus human, in seconds.

| Cohort | Architecture | Arm | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact | tcn | baseline | 0 | 0.765845 | 0.852482 | 0.806646 | 2658.708 | 2387.151 | +271.557 |
| exact | tcn | baseline | 1 | 0.795127 | 0.921658 | 0.853624 | 3353.911 | 3031.151 | +322.760 |
| exact | tcn | baseline | 2 | 0.816646 | 0.945632 | 0.876335 | 4021.708 | 3675.151 | +346.557 |
| exact | tcn | baseline | 3 | 0.835328 | 0.959630 | 0.893107 | 4676.392 | 4323.263 | +353.129 |
| exact | tcn | short_boost | 0 | 0.770709 | 0.841266 | 0.804098 | 2608.697 | 2387.151 | +221.546 |
| exact | tcn | short_boost | 1 | 0.800153 | 0.916751 | 0.854282 | 3308.389 | 3031.151 | +277.238 |
| exact | tcn | short_boost | 2 | 0.822032 | 0.943495 | 0.878432 | 3972.578 | 3675.151 | +297.427 |
| exact | tcn | short_boost | 3 | 0.839518 | 0.958152 | 0.894761 | 4628.708 | 4323.263 | +305.445 |
| exact | tcn | global_control | 0 | 0.760567 | 0.842592 | 0.798656 | 2652.103 | 2387.151 | +264.952 |
| exact | tcn | global_control | 1 | 0.789537 | 0.916318 | 0.847685 | 3347.331 | 3031.151 | +316.180 |
| exact | tcn | global_control | 2 | 0.812441 | 0.943620 | 0.872786 | 4010.900 | 3675.151 | +335.749 |
| exact | tcn | global_control | 3 | 0.833271 | 0.955842 | 0.890111 | 4648.103 | 4323.263 | +324.840 |
| exact | dino_tcn | baseline | 0 | 0.816768 | 0.881879 | 0.845296 | 2597.803 | 2387.151 | +210.652 |
| exact | dino_tcn | baseline | 1 | 0.840683 | 0.946227 | 0.888661 | 3266.161 | 3031.151 | +235.010 |
| exact | dino_tcn | baseline | 2 | 0.858208 | 0.966573 | 0.908143 | 3919.750 | 3675.151 | +244.599 |
| exact | dino_tcn | baseline | 3 | 0.874969 | 0.975343 | 0.921720 | 4558.564 | 4323.263 | +235.301 |
| exact | dino_tcn | short_boost | 0 | 0.856313 | 0.810961 | 0.830170 | 2275.819 | 2387.151 | -111.332 |
| exact | dino_tcn | short_boost | 1 | 0.878157 | 0.900694 | 0.887359 | 2931.939 | 3031.151 | -99.212 |
| exact | dino_tcn | short_boost | 2 | 0.894164 | 0.936468 | 0.913565 | 3570.969 | 3675.151 | -104.182 |
| exact | dino_tcn | short_boost | 3 | 0.905337 | 0.951414 | 0.926836 | 4206.033 | 4323.263 | -117.230 |
| exact | dino_tcn | global_control | 0 | 0.848974 | 0.812424 | 0.829359 | 2286.056 | 2387.151 | -101.095 |
| exact | dino_tcn | global_control | 1 | 0.868014 | 0.890794 | 0.878626 | 2926.208 | 3031.151 | -104.943 |
| exact | dino_tcn | global_control | 2 | 0.882452 | 0.922386 | 0.901551 | 3554.764 | 3675.151 | -120.387 |
| exact | dino_tcn | global_control | 3 | 0.895929 | 0.939676 | 0.916950 | 4175.753 | 4323.263 | -147.510 |
| draft | tcn | baseline | 0 | 0.851745 | 0.806490 | 0.828492 | 2260.428 | 2387.151 | -126.723 |
| draft | tcn | baseline | 1 | 0.873180 | 0.882489 | 0.877802 | 2880.844 | 3031.151 | -150.307 |
| draft | tcn | baseline | 2 | 0.889566 | 0.912967 | 0.901112 | 3480.808 | 3675.151 | -194.343 |
| draft | tcn | baseline | 3 | 0.902050 | 0.928750 | 0.915205 | 4081.986 | 4323.263 | -241.277 |
| draft | tcn | short_boost | 0 | 0.852783 | 0.794380 | 0.821829 | 2226.769 | 2387.151 | -160.382 |
| draft | tcn | short_boost | 1 | 0.872074 | 0.876050 | 0.873602 | 2870.375 | 3031.151 | -160.776 |
| draft | tcn | short_boost | 2 | 0.887916 | 0.910462 | 0.898712 | 3492.833 | 3675.151 | -182.318 |
| draft | tcn | short_boost | 3 | 0.900798 | 0.930120 | 0.914957 | 4109.864 | 4323.263 | -213.399 |
| draft | tcn | global_control | 0 | 0.849599 | 0.810026 | 0.829256 | 2276.464 | 2387.151 | -110.687 |
| draft | tcn | global_control | 1 | 0.870029 | 0.882017 | 0.875932 | 2896.297 | 3031.151 | -134.854 |
| draft | tcn | global_control | 2 | 0.886437 | 0.910954 | 0.898485 | 3498.422 | 3675.151 | -176.729 |
| draft | tcn | global_control | 3 | 0.898680 | 0.926400 | 0.912302 | 4107.022 | 4323.263 | -216.241 |
| draft | dino_tcn | baseline | 0 | 0.862008 | 0.857032 | 0.859006 | 2376.458 | 2387.151 | -10.693 |
| draft | dino_tcn | baseline | 1 | 0.880697 | 0.936378 | 0.907409 | 3019.692 | 3031.151 | -11.459 |
| draft | dino_tcn | baseline | 2 | 0.894879 | 0.960708 | 0.926419 | 3651.008 | 3675.151 | -24.143 |
| draft | dino_tcn | baseline | 3 | 0.907573 | 0.970727 | 0.937921 | 4272.878 | 4323.263 | -50.385 |
| draft | dino_tcn | short_boost | 0 | 0.863101 | 0.849845 | 0.856336 | 2350.764 | 2387.151 | -36.387 |
| draft | dino_tcn | short_boost | 1 | 0.881975 | 0.939998 | 0.910020 | 2997.664 | 3031.151 | -33.487 |
| draft | dino_tcn | short_boost | 2 | 0.896847 | 0.964565 | 0.929435 | 3622.128 | 3675.151 | -53.023 |
| draft | dino_tcn | short_boost | 3 | 0.908864 | 0.973123 | 0.939862 | 4243.931 | 4323.263 | -79.332 |
| draft | dino_tcn | global_control | 0 | 0.858108 | 0.873237 | 0.865349 | 2431.403 | 2387.151 | +44.252 |
| draft | dino_tcn | global_control | 1 | 0.879591 | 0.949627 | 0.913102 | 3076.711 | 3031.151 | +45.560 |
| draft | dino_tcn | global_control | 2 | 0.895049 | 0.969599 | 0.930707 | 3707.081 | 3675.151 | +31.930 |
| draft | dino_tcn | global_control | 3 | 0.907323 | 0.976350 | 0.940466 | 4338.008 | 4323.263 | +14.745 |
| reviewed_export | tcn | baseline | 0 | 0.830479 | 0.821559 | 0.825682 | 2363.353 | 2387.151 | -23.798 |
| reviewed_export | tcn | baseline | 1 | 0.854204 | 0.899546 | 0.876082 | 2988.461 | 3031.151 | -42.690 |
| reviewed_export | tcn | baseline | 2 | 0.873379 | 0.924917 | 0.898235 | 3587.164 | 3675.151 | -87.987 |
| reviewed_export | tcn | baseline | 3 | 0.888981 | 0.937577 | 0.912490 | 4180.139 | 4323.263 | -143.124 |
| reviewed_export | tcn | short_boost | 0 | 0.842414 | 0.819957 | 0.830962 | 2324.050 | 2387.151 | -63.101 |
| reviewed_export | tcn | short_boost | 1 | 0.864670 | 0.896008 | 0.880023 | 2943.533 | 3031.151 | -87.618 |
| reviewed_export | tcn | short_boost | 2 | 0.883046 | 0.924092 | 0.903077 | 3543.467 | 3675.151 | -131.684 |
| reviewed_export | tcn | short_boost | 3 | 0.897107 | 0.938826 | 0.917483 | 4145.108 | 4323.263 | -178.155 |
| reviewed_export | tcn | global_control | 0 | 0.834812 | 0.819875 | 0.826667 | 2347.983 | 2387.151 | -39.168 |
| reviewed_export | tcn | global_control | 1 | 0.857337 | 0.891352 | 0.873591 | 2956.275 | 3031.151 | -74.876 |
| reviewed_export | tcn | global_control | 2 | 0.876916 | 0.917221 | 0.896269 | 3544.936 | 3675.151 | -130.215 |
| reviewed_export | tcn | global_control | 3 | 0.892350 | 0.930982 | 0.910998 | 4127.583 | 4323.263 | -195.680 |
| reviewed_export | dino_tcn | baseline | 0 | 0.829112 | 0.841643 | 0.833538 | 2436.058 | 2387.151 | +48.907 |
| reviewed_export | dino_tcn | baseline | 1 | 0.858556 | 0.931370 | 0.892574 | 3074.997 | 3031.151 | +43.846 |
| reviewed_export | dino_tcn | baseline | 2 | 0.879177 | 0.958035 | 0.916319 | 3691.872 | 3675.151 | +16.721 |
| reviewed_export | dino_tcn | baseline | 3 | 0.896269 | 0.968848 | 0.930699 | 4303.233 | 4323.263 | -20.030 |
| reviewed_export | dino_tcn | short_boost | 0 | 0.843366 | 0.862223 | 0.852536 | 2441.886 | 2387.151 | +54.735 |
| reviewed_export | dino_tcn | short_boost | 1 | 0.866891 | 0.946347 | 0.904761 | 3101.642 | 3031.151 | +70.491 |
| reviewed_export | dino_tcn | short_boost | 2 | 0.884748 | 0.969545 | 0.925127 | 3739.831 | 3675.151 | +64.680 |
| reviewed_export | dino_tcn | short_boost | 3 | 0.900763 | 0.978352 | 0.937896 | 4364.456 | 4323.263 | +41.193 |
| reviewed_export | dino_tcn | global_control | 0 | 0.831259 | 0.858045 | 0.844268 | 2465.642 | 2387.151 | +78.491 |
| reviewed_export | dino_tcn | global_control | 1 | 0.857055 | 0.945379 | 0.898979 | 3111.214 | 3031.151 | +80.063 |
| reviewed_export | dino_tcn | global_control | 2 | 0.877905 | 0.966848 | 0.920171 | 3727.064 | 3675.151 | +51.913 |
| reviewed_export | dino_tcn | global_control | 3 | 0.894929 | 0.974491 | 0.932959 | 4337.664 | 4323.263 | +14.401 |

This is adaptive development work on four previously inspected source groups, with only three seeds. Seed consistency is not independent evidence about the population of future recordings, and source-group differences limit generalization. Training on more videos did not uniformly improve every architecture or loss arm. The protected test remains reserved for a later, separately justified evaluation.

Before candidate training, a timestamp audit found that the original Pixel variable-frame-rate AV cache used frame-index time and the original DINO path sought different frames. Both representations for all seven raw/project-export videos were regenerated on one media-PTS timeline under the prospective amendment; the eleven exact/draft caches and all labels stayed unchanged. Therefore reviewed-export baselines here were freshly trained. Historical reviewed-export numbers from the expanded and event-balanced experiments are not directly comparable to this corrected cohort. Only six compact exact/draft baseline cells were reused after byte and provenance verification.

The full summary, tensor/scaler audit, and independent interval audit passed after report finalization. They verify 54 result cells, 864 logical fits (768 fresh and 96 reused), 2,808 checkpoints, and 5,616 NPZ artifacts. The summary replayed 54 canonical evaluations and 216 selected inner candidates, verified all 16 registered analysis sources, 18 AV cache hashes, 36 DINO cache/sidecar hashes, and all 18 source/timeline associations. The tensor audit exactly reconstructed four-head supervision, live weighting, fold scalers, held-group membership and paired sampling histories. The independent endpoint-sweep audit checked 2,808 scope/padding rows, 216 selected refit prediction files, and 432 decoded recording traces. It does not rerun every neural forward or fully re-extract source videos.

Two auditor schema bugs were fixed without changing any of the 16 registered model/training/scoring sources, inputs, results, or metric definitions. The tensor auditor now compares path/hash identities while separately verifying optional file-size metadata. The interval auditor compares the exact frozen serializer's start/end/ordered-tag fields while the complete annotation document remains hash-bound; notes and ignored-interval reasons are descriptive metadata omitted by that serializer. Original sources and failed logs are preserved under the recovery folders. Sixteen tensor-auditor tests and fifteen interval-auditor tests passed, including regressions rejecting wrong hashes, sizes, boundaries, tags, event order, counts, and ignored spans. The already successful full summary was not rerun.

All artifacts below are on `private-reference-0178` (the same files are mounted at `private-reference-0084` in WSL). SHA-256 values bind the completed artifacts.

| Artifact | SHA-256 |
|---|---|
| study/report.json (ledger `private-reference-0179`) | `2f6e47590a8c7ee085dd8cebf0f626f833a6dfb7332216e284ade19839f1ec38` |
| study/summary.json (ledger `private-reference-0180`) | `20752c1423fe8d82470c069f3093d0a671f57f92303a01e22cf2f64cc7bcc31d` |
| study/summary.md (ledger `private-reference-0181`) | `3f3f7611ed51f5155d1e77527c4db80f2e3cb1b228e6e42d5b798d333f4b3d60` |
| study/loss-identities.json (ledger `private-reference-0182`) | `d137b291beb24919e3572aee176f6dad83074f9f661d756c5fec936c05b1e581` |
| tensor-scaler-audit-v1.json (ledger `private-reference-0183`) | `137b446f4f40f01dd1aa7db5c1d248ef11ea9e307f9c8f8496def02f28d15a4c` |
| interval-audit-v1.json (ledger `private-reference-0184`) | `efdfc689a6d4d6ceed140141892ca447568e5606f43cf6dad5d1bcba5071405b` |
| runtime-recovery-v4/interval-audit-stage-repair/all-audits-completed.json (ledger `private-reference-0185`) | `7740cb8425baf5a53462bbb116afe210177baa864d89efe5efb31268c2ba33b6` |

The operational recovery records and source snapshots preserve interruptions and the successful continuation separately. The completed audit receipt above binds the successful summary and both independent audits. Further hypotheses require separate prospective protocols; this document reports only the completed short-boost study.
