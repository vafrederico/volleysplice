**Per-rally live-loss experiment — 2026-09-19**

Completed all 144 fits and nine weighted TCN evaluations. Per-rally weighting recovers more complete points, including short points, in every paired seed. It also worsens longer-rally coverage and event matching beyond the predeclared limits. No cohort passes the recovery screen or the standard F1-improvement screen, so this uncapped weighting rule is not selected for production. The model, features, source folds and decoder recipe stayed fixed; production and the protected test set remain unchanged.

**Completed comparison.** Same eight exact-label recordings / 322 rallies, three seeds per cohort. Each seed pools interval numerators and denominators over recordings; tables average those separate runs. The target padding remains two seconds before and after, with positive gaps strictly below three seconds retained. Complete and partial losses refer to original rallies after that fixed export padding.

| Training cohort | Reference F1 | Weighted F1 | Reference R_core | Weighted R_core | Complete losses: reference / weighted | Partial losses: reference / weighted |
|---|---:|---:|---:|---:|---:|---:|
| exact | 0.876335 | 0.877085 | 0.945632 | 0.942389 | 21.00 / 17.00 | 31.33 / 35.00 |
| draft | 0.901112 | 0.886042 | 0.912967 | 0.903469 | 35.33 / 31.67 | 43.00 / 45.33 |
| reviewed_export | 0.895553 | 0.892004 | 0.917649 | 0.919806 | 37.67 / 26.00 | 38.33 / 42.33 |

The reviewed-export cohort shows the clearest point-retention benefit: mean complete losses fall 30.97%, from 37.67 to 26.00, and short-point complete losses fall 34.29%, from 23.33 to 15.33. Its overall core recall improves by 0.216 percentage points at a 0.355-point F1 cost. Nevertheless, longer-rally recall falls by 0.532 points, beyond the allowed 0.5-point regression, and event F1 falls by 3.388 points, beyond the allowed one-point regression. It passes eight of ten recovery conditions; the thresholds remain unchanged.

| Cohort | Complete-loss reduction | <=3s complete-loss reduction | >3s R_core delta, pp | Event-F1 delta, pp | Recovery |
|---|---:|---:|---:|---:|---|
| exact | 19.05% | 20.93% | -0.608 | -3.177 | FAIL |
| draft | 10.38% | 20.90% | -1.514 | -1.143 | FAIL |
| reviewed_export | 30.97% | 34.29% | -0.532 | -3.388 | FAIL |

Exact-only training fails the 20% overall complete-loss reduction requirement (19.05%), the longer-rally recall limit and the event-F1 limit. Draft training also fails the overall F1 and core-recall limits, with only 10.38% fewer complete losses. Every cohort reduces complete losses overall and among <=3-second points in all three seeds, and none increases mean incomplete-rally counts. All 36 inner selections meet their original .95 recall eligibility floor; the failed outer outcomes are not fallback selections.

This controlled result shows that loss allocation can recover points with the existing features, while the current allocation is not an overall quality improvement under the frozen criteria. It does not prove that the features are sufficient or that uncertain draft ticks caused the regressions. A useful next hypothesis is to preserve the original long-rally objective while adding a bounded short-point recovery term, with a matched positive-class-budget control to separate targeted recovery from a global recall shift. That would be a new preregistered experiment; it was not run here.

Inference still uses the same TCN architecture and features. This training-only change adds no inference operations or parameters; no new production model or phone/browser runtime claim is made.

Execution checklist:

- [x] Read repository/shared instructions and review the immutable expanded study.
- [x] Define one weighting rule and declare the target padding and retention guardrails.
- [x] Verify usable event ticks and unchanged masks/chunks/sampling on all 18 recordings.
- [x] Pass independent loss/runner tests and replay uniform-weight GPU controls.
- [x] Register source, data, reference and environment hashes before candidate training.
- [x] Train all 144 candidate fits with three seeds and nested source-group validation.
- [x] Audit artifacts, exposures, weights and all metric replays.
- [x] Record the outcome, failure identities, padding sensitivity and source snapshot.

The Tasks tool is unavailable in this environment; this checklist tracks execution.

Preflight passed 27 new unit/integration checks: 16 weighting checks, five runner checks and six summary checks. Actual-data GPU controls replayed seed 3407 / outer 0 / inner 0 / epoch 5 for all three cohorts with uniform weighting. Every saved model/scaler tensor, probability trace, optimizer-step count and five-epoch sampling history matched the previous study bit-for-bit. The preflight also verified 11 historical source hashes, 66 metadata/cache references, all 18 per-record diagnostics and original chunk/sampling identity across the three seeds. `preflight/preflight-report.json` SHA-256 is `370d41d60f05399bf1baf1d3b606a209d6642618323a75ca495546cddcb59f2d`.

The candidate contract was registered before training with SHA-256 `9f1ce53e50c5bd21a4365e0c30d62bb43dd7e72962d2d7a3341f720489aaebda`. It binds all 13 analysis sources, the reference artifacts, data, environment, weighting formula and comparison rules. Candidate training started at approximately 09:29 UTC on 2026-09-19.

**One experimental change.** For recording r, let n_i be the number of existing supervised live-positive ticks belonging to original rally i, N the number of rallies with n_i > 0, and P = sum(n_i). Give each of those positive ticks multiplier P/(N*n_i). This equalizes positive live-loss mass among usable rallies within that recording while preserving its total positive mass. It does not equalize every rally globally across recordings.

The multiplier is uncapped and applies only to live BCE in the exact and draft tiers. Negative ticks, other output heads, class weights, labels, masks, head/tier weights and sampling are unchanged. Loss still divides by each head's original unweighted valid-mask count, so a minibatch cannot cancel the weighting through its denominator. An original rally remains one event if ignored time splits its visible pieces. An event with no supervised ticks remains unknown and contributes neither a positive nor a negative target. Export coverage has no exact live targets and keeps its existing auxiliary keep loss.

All 322 exact rallies have usable ticks, with at least four each. Of 123 draft ranges, 105 have usable ticks and 18 remain fully censored. Exact positive multipliers range up to 7.835; draft multipliers reach 29.576 because a few approximate intervals have only one usable interior tick. This amplification is a prospective limitation of the uncapped hypothesis. A cap or a different draft policy would be a separate future experiment; neither will be chosen after seeing this run. Float32 storage preserves each recording's total positive mass within about 1.1e-5 ticks in the dataset checks.

**Fixed inputs and controls.** Reuse the immutable manifest at `private-reference-0132`, SHA-256 `48fe4fb5bcb74b2ee998562f61c3a7e8df43e921f8882e1b44fe1c12db5d451f`. The cohorts remain exact8, exact8+draft3, and exact8+draft3+reviewed-export7. All use the same 104 audiovisual features, 64-channel/five-block TCN, 29,700 parameters and live/serve/end/keep outputs. No beach footage or protected test group is opened. Only the eight exact recordings / 322 original rallies supply evaluation targets; seven source groups supply training data, and four exact source groups define outer evaluation.

The matched references are the completed unweighted TCNs for the same three cohorts, seeds and folds. Reference `study/report.json` SHA-256 is `5371829b616bd46fea07446662fc3bb2b53c2a1221e3455e1e2aedd94882b4fc`; reference summary SHA-256 is `e2adfe24cd9cad42c6c918d73dbc0b74d9830fa2a9bb2b4b4a699792ceed4bf3`. Historical source files and artifacts remain unchanged. A uniform-weight path in the new fitter is used only for preflight reproduction of saved GPU checkpoints; it is not a candidate or an additional tuned arm.

At equal epoch prefixes, initial weights, real context chunks, batch membership, exact/auxiliary dropout streams and optimizer steps remain identical to their historical counterparts. AdamW, class weights, clipping, learning rate and all other fitting settings remain fixed. Selected outer refit lengths may differ because selection operates on the new inner predictions. Shared TCN weights mean serve/end/keep predictions can change even though their targets and direct loss definitions stay fixed.

**Selection and evaluation.** Seeds remain 3407/1729/20260918; checkpoints remain 5/15/30/60 epochs. Each outer source group is excluded from every training tier, and each inner group is also excluded from every inner training tier. The decoder grid is unchanged. Among inner candidates with R_core >= .95, select the highest pooled F1_padP_coreR at the predeclared two-second before/after padding. If none qualify, retain the highest F1 and flag infeasibility. New rally-retention guardrails below do not change inner selection.

All evaluation remains unweighted. Pad, clip and merge model/human ranges; retain positive gaps strictly below three seconds, then subtract ignored intervals without rejoining. Pool duration numerators/denominators over recordings within each seed. Report 0/1/2/3-second symmetric padding for the same predictions, including P_pad, R_core, F1_padP_coreR, model/human export seconds and their difference. Seed summaries average separate pooled runs; they do not duplicate recordings into one pooled dataset or estimate uncertainty across new sessions.

**Predeclared recovery screen.** Compare each weighted cohort with its matched unweighted TCN. Every condition must pass:

- Mean F1_padP_coreR change is at least -.005, mean R_core change is at least -.005, and mean event-F1 change is at least -.01.
- Mean R_core for original rallies longer than three seconds changes by at least -.005.
- Mean completely lost original rallies is at most 80% of the reference count, both for all rallies and for those lasting at most three seconds.
- Mean incomplete-rally count (complete plus partial losses) does not increase, for either all rallies or the at-most-three-second subset.
- Complete losses strictly decrease in at least two of three paired seeds, both overall and for the at-most-three-second subset. Ties do not count as reductions.

Use canonical `primaryExportCoverage` after the fixed two-second padding; short/long membership uses original `end-start`, never padded duration. A zero reference count is handled by multiplication rather than a ratio division. Separately report the previous four-part F1-improvement screen (mean gain >= .02, two positive seeds, majority-positive source groups, mean R_core regression no worse than .005), all source-group directions, <=2-second points, aces/faults, original complete/partial loss identities and boundary/event diagnostics. All-seed R_core >= .95 is descriptive, not a newly selected threshold. Neither screen authorizes production promotion.

There are 108 inner fits and 36 selected outer refits: 144 fits, 468 checkpoints and nine pooled candidate evaluations. Disjoint cohort/seed subprocesses share the RTX 3080 with at most three active fits. The parent creates registration before workers start; each worker verifies the same source/data/environment contract and owns separate output paths. Artifacts live under `private-reference-0133`. Runtime sums must distinguish overlapping fit durations from elapsed wall time and per-process CUDA allocation from total GPU memory.

Reproduce after checking the environment into a new output directory:

```bash
private-reference-0110 -u -m analysis.neural_event_balanced_development \
  --manifest private-reference-0134 \
  --reference-study private-reference-0135 \
  --output private-reference-0136 \
  --workers 3
```

**Required padding sensitivity.** These are means of separate seed runs, each pooled over recordings. The same selected predictions appear in all four padding cases. Historical rows are the unchanged matched TCNs.

| Cohort | Run | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact | weighted | 0 | 0.765525 | 0.834723 | 0.798611 | 2603.097 | 2387.151 | +215.946 |
| exact | weighted | 1 | 0.798737 | 0.912261 | 0.851719 | 3306.053 | 3031.151 | +274.902 |
| exact | weighted | 2 | 0.820257 | 0.942389 | 0.877085 | 3989.147 | 3675.151 | +313.996 |
| exact | weighted | 3 | 0.836200 | 0.959692 | 0.893692 | 4671.289 | 4323.263 | +348.026 |
| exact | historical | 0 | 0.765845 | 0.852482 | 0.806646 | 2658.708 | 2387.151 | +271.557 |
| exact | historical | 1 | 0.795127 | 0.921658 | 0.853624 | 3353.911 | 3031.151 | +322.760 |
| exact | historical | 2 | 0.816646 | 0.945632 | 0.876335 | 4021.708 | 3675.151 | +346.557 |
| exact | historical | 3 | 0.835328 | 0.959630 | 0.893107 | 4676.392 | 4323.263 | +353.129 |
| draft | weighted | 0 | 0.833326 | 0.792327 | 0.812178 | 2270.889 | 2387.151 | -116.262 |
| draft | weighted | 1 | 0.854681 | 0.870182 | 0.862263 | 2912.983 | 3031.151 | -118.168 |
| draft | weighted | 2 | 0.869434 | 0.903469 | 0.886042 | 3547.292 | 3675.151 | -127.859 |
| draft | weighted | 3 | 0.882777 | 0.921056 | 0.901446 | 4172.412 | 4323.263 | -150.851 |
| draft | historical | 0 | 0.851745 | 0.806490 | 0.828492 | 2260.428 | 2387.151 | -126.723 |
| draft | historical | 1 | 0.873180 | 0.882489 | 0.877802 | 2880.844 | 3031.151 | -150.307 |
| draft | historical | 2 | 0.889566 | 0.912967 | 0.901112 | 3480.808 | 3675.151 | -194.343 |
| draft | historical | 3 | 0.902050 | 0.928750 | 0.915205 | 4081.986 | 4323.263 | -241.277 |
| reviewed_export | weighted | 0 | 0.828405 | 0.809502 | 0.818146 | 2336.106 | 2387.151 | -51.045 |
| reviewed_export | weighted | 1 | 0.848810 | 0.888657 | 0.867882 | 3000.700 | 3031.151 | -30.451 |
| reviewed_export | weighted | 2 | 0.866476 | 0.919806 | 0.892004 | 3637.558 | 3675.151 | -37.593 |
| reviewed_export | weighted | 3 | 0.881220 | 0.937211 | 0.908055 | 4268.137 | 4323.263 | -55.126 |
| reviewed_export | historical | 0 | 0.834094 | 0.822813 | 0.828235 | 2355.886 | 2387.151 | -31.265 |
| reviewed_export | historical | 1 | 0.856698 | 0.890844 | 0.873291 | 2968.658 | 3031.151 | -62.493 |
| reviewed_export | historical | 2 | 0.874680 | 0.917649 | 0.895553 | 3568.533 | 3675.151 | -106.618 |
| reviewed_export | historical | 3 | 0.890026 | 0.930977 | 0.909992 | 4156.931 | 4323.263 | -166.332 |

**Final verification.** The main runner exited successfully. Independent audits verified 144 new and 144 historical fit memberships; 936 artifact hashes for each study arm set; 468 bit-exact scaler checkpoint comparisons; 144 per-fit weight records; and all 18 independently reconstructed weight vectors. Sampling verification covered 7,350 historical/current epoch prefixes and 4,845 paired cohort prefixes. All 27 historical/current evaluations and 36 selected inner candidates replayed exactly. The separate tensor audit recomputed all 468 new scalers independently with zero error and verified finite, correctly shaped 29,700-parameter models and valid prediction timelines. A further independent checker reconstructed padded interval unions and original-rally coverage directly from raw predictions for all nine weighted/reference pairs and agreed with every screen decision. All 13 registered source hashes remain unchanged.

Recorded fit durations sum to 1.01 process hours; three concurrent workers completed the main run in approximately 22 minutes. These are training measurements, with overlapping fit durations and per-process allocator statistics. They do not measure total GPU memory or phone inference latency.

Artifacts below are relative to `private-reference-0133`. The full summary (ledger `private-reference-0137`) and JSON include every seed, source group, outcome slice, boundary diagnostic and exact new/recovered loss identity.

| Artifact | SHA-256 |
|---|---|
| `study/report.json` | `c78dce8f3498cb1cd79eba368b8663cd09b9be73818b0728ef6684cf6768524c` |
| `study/summary.json` | `beefceaf4f0deb1af09ca9e79d4b8e12060d6883c4d0fa3750925cd692885fc4` |
| `study/summary.md` | `30fd2946e31ecdd37d6513b6ffc1d5b7e29db301b58dd65b641ade4989c6b913` |
| `study/loss-identities.json` | `7abef001046fab8827ce8ebcdedfa9c4d39a7e69008d59d3d5f51497190d9a72` |
| `tensor-scaler-audit-v1.json` | `20f8d7779eeb5225b491efc0d501c120a766b44e317326f48588ea084f825550` |
| `independent-result-check.json` | `9a56547bb942e1759a68fcc977b6079b2e3306dc2c15aa0de0c787188b2f3924` |
| `preflight/preflight-report.json` | `370d41d60f05399bf1baf1d3b606a209d6642618323a75ca495546cddcb59f2d` |

The final bounded source archive is under `source-snapshot/`, with ZIP hash, per-file identities, Git base and verification in its manifest/verification JSON. It preserves this experiment's new research sources and preflight evidence, and binds the earlier source archive separately. Model/data artifacts remain external on the NAS.
