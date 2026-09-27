# Recall-floor sweep and source-diversity experiment

Status: inventory and implementation in progress. This document is the task ledger and design draft; numerical runs require the final hash-bound registration. No new evaluation outcome has been used to set the design.

## Task ledger

- [ ] Inventory every available non-beach labeled recording and reviewed model-feedback/project export, deduplicate raw/proxy/export lineage, and identify label quality and evaluation coverage.
- [ ] Audit every current production ensemble and suppression head's fitting and calibration membership. Mark direct, related-source, validation-only, unknown, and known-unexposed sources.
- [ ] Freeze the 90 through 100 percent recall-floor sweep, source-group split/size variants, common unseen evaluation panel, code, inputs, and model identities.
- [ ] Reuse historical predictions and complete missing checkpoint epochs without changing historical studies; replay the original nested source-held sweep.
- [ ] Fit original-corpus deployment models and randomized composition/size variants. Freeze all checkpoint/decoder selections before opening additional-video outcome metrics.
- [ ] Run label-blind inference on all eligible videos, including additional and historically protected non-beach evaluation sources.
- [ ] Independently audit selection, leakage, artifact lineage, pooled interval metrics, and production-exposure exclusions.
- [ ] Produce and locally verify an interactive HTML artifact and machine-readable results, publish through the shared Artifacts service, and commit/push source and research notes.

## Fixed comparison contract

The requested floors are 90, 91, ..., 100 percent pooled validation retained-core recall. For each floor select maximum `F1_padP_coreR` among eligible epoch/decoder candidates. Preserve candidate order for ties. An infeasible floor remains unavailable, with maximum attained validation recall reported; never lower the floor or substitute production. Floors are selection constraints, not guarantees of unseen-source recall or calibrated score probabilities.

The target product padding is two seconds before and after. Report all four symmetric padding cases 0/1/2/3, holding the chosen setting fixed. Merge touching/overlapping padded intervals and retain positive gaps strictly below three seconds for both model and human. Subtract ignored time afterward and never rejoin across it. Pool duration numerators/denominators across each declared recording population. Rank only on development/validation F1 at the target padding. Additional or protected evaluation labels never choose a configuration, split, threshold, epoch, or seed.

Report `P_pad`, `R_core`, `F1_padP_coreR`, model export, human export, their difference, correctly removed time, incorrect export, wanted human export omitted, missed core time, event P/R/F1, original complete/partial rally losses, and boundary diagnostics where exact rally labels support them. Reviewed export-only annotations receive separately named export-overlap metrics; do not infer rally boundaries or unpadded human core from padded/joined cuts. Unreviewed model-feedback predictions, if present, are agreement references rather than independent human gold.

Include compact audiovisual TCN, DINO TCN, frozen regional MobileNet TCN, DINO-distilled MobileNet TCN, audiovisual transformer, and DINO transformer. DINO FP16 and mixed native-CPU INT8 are precision-transfer variants of the FP32-selected model, with no precision-specific tuning. Retain the failed INT8 browser-parity limitation. Current production is a fixed descriptive comparator, not a newly optimized recall-floor family. Detector configurations that never qualified and have no trained rally head cannot supply a rally sweep.

## Distinct evaluation scopes

1. Historical nested source-held evaluation retains the original eight exact recordings, three training seeds, and original input/masking contract. Save/recover all checkpoint epochs 5/15/30/60 once per fit; decode the existing 48 settings once per epoch and reuse them across floors. No three-seed mean when any registered fold/seed is infeasible or missing.
2. Original-corpus deployment models use only the original eighteen training records. Their checkpoint/decoder selections use original-source out-of-fold predictions, then model weights are fitted on the original corpus. New recording labels and images do not fit, distill, calibrate, or select these models. Their results on original fitting/calibration sources are descriptive; genuinely additional sources are reported separately.
3. Randomized variants change source-group composition and training/validation size, using the same architecture, loss, optimizer, feature recipe, and candidate grid. Their explicit membership and seeds will be frozen from inventory before outcome scoring. A common independent panel is excluded from every variant's fitting, distillation, normalization, and calibration. All-video results additionally carry per-model training/calibration/related/unseen role flags and are never described as all held out.

Raw footage and any derivative proxy or project export remain in one source group. Previously protected non-beach sources are now authorized for final inference/evaluation by the user's explicit all-other-video request, but remain forbidden for any fitting or selection. After this evaluation they must not be called untouched future tests.

New deployment and variant inference receives image/feature inputs without rally, ignored, or export labels. It runs across the full timeline. Human annotations are attached only for evaluation, including ignored-time subtraction; they cannot reset model context or alter decoded boundaries. The historical nested replay retains its old ignored-segment inference behavior for numerical comparability and is identified separately.

Every summary has an all-eligible-video view and a view excluding production-fitting sources and their derivatives. Production validation/calibration exposure and uncertain lineage remain visible; a stricter known-unexposed subset is reported where available. Empty comparison subsets stay empty rather than being presented as zero-error results.

## Inventory-based design freeze

The inventory contains 44 distinct full/raw sources, of which 42 are non-beach. Thirty-four have scorable manually reviewed annotations in distinct quality strata: nine exact rally-label recordings, six continuous reviewed drafts, and nineteen reviewed export-coverage recordings. Eight further non-beach sources are unscored; inference may be inspected, but accuracy must not be invented from absent or unreviewed gold.

There are no new independent exact-labeled training groups beyond the original four groups/eight recordings. The one additional exact-labeled source is the historically protected indoor source match. New training information consists of three continuous drafts within existing source-group-005/source-group-007 groups and eleven reviewed August 16 grass exports from one conservatively grouped session. The September 17 reviewed export remains evaluation-only. This limits how conclusively this study can distinguish insufficient exact-label diversity from inadequate features/model capacity.

The common evaluation panel is the protected indoor source source group (one exact gold recording and one unscored recording) plus the September 17 export-coverage group. No imagery or annotation from either group enters any training, distillation, scaler, or calibration step. Original-corpus models also retain the entire additional-video panel as independent evaluation; expanded variants explicitly change the role of the added training footage, and their generalization comparison uses the common panel.

For the six neural families, the original-corpus deployment experiment fits the original eighteen training records with training seeds 3407, 1729, and 20260918. Global floor selections come exclusively from the original eight recordings' outer out-of-fold checkpoint scores. These are eighteen new full-corpus temporal fits and three full-corpus student encoders, separate from the historical nested estimate.

The randomized experiment uses four balanced split draws, one designated calibration group for each of the original four sorted exact source groups. Independently shuffle the remaining three groups with split seeds 3407, 1729, 20260918, and 20260923 respectively. All randomized variants use the same training seed 3407 to isolate source composition from training-seed variability. Each draw runs four configurations:

- `original-medium`: the first two shuffled groups train; the designated group calibrates. Include only original-eighteen auxiliary recordings whose source does not conflict with calibration or omitted exact groups.
- `expanded-medium`: identical exact train/calibration groups; add the other eligible manually reviewed drafts and the independent August 16 coverage source. This is the matched auxiliary-diversity intervention.
- `expanded-large`: all three other exact groups train; the designated group calibrates; use the expanded auxiliary pool. This changes exact training size while keeping calibration fixed.
- `expanded-wide-validation`: the same two training groups and auxiliary pool as expanded-medium; calibrate on both the designated group and the third omitted exact group. This changes validation coverage while keeping training membership fixed.

Independent auxiliary-only source groups are eligible for training unless part of the common panel. Every derivative of an excluded exact group is excluded too. Preserve original-eighteen supervision for these interventions, and separately identify any authoritative evaluation-label revision. The design comprises 16 variants per family, or 96 temporal fits and 16 student encoders. Their split draws are correlated views of four source groups, not independent new matches. Report each draw; compute an across-draw mean only when all four registered draws have the complete same evaluation population at the requested floor.

All newly fitted temporal models save epochs 5/15/30/60 once. Every floor uses the unchanged 48-decoder grid, for 192 candidates. Architecture/loss/optimizer/sampling rules are held fixed. The historical sweep requires 72 supplemental all-four-epoch outer refits to fill 202 missing epoch slots; ten existing outer students can be reused and two previously skipped outer students must be fitted. Existing checkpoint weights, scores, and histories provide numerical parity checks.

Production exposure is broader than weight fitting. The deployment suppression decoder was calibrated using another recording from the protected indoor source source group. Therefore that exact-labeled recording is production-rally-fitting-clean but calibration-related. The strict no-fitting/no-calibration panel contains only reviewed export coverage; there is no strictly production-unexposed exact rally gold in this inventory. Report whole-product serve-side/side-switch exposure separately from the heads that affect rally selection.

## User-requested export-rally label amendment

The user additionally requested one or two tests treating adjusted, reviewed export labels as approximate rally labels for training and selection. This amendment adds two experiment families without changing the frozen original-corpus or historical runs.

The proxy inventory has 674 saved individual cores across eighteen training-eligible raw sources: 399 August 16 cores and 275 August 29 cores. Each endpoint comes directly from its included corrected range's `coreStart`/`coreEnd`, with final-export cut-ID membership required. Four sub-0.25-second artifacts are excluded under the existing minimum-artifact rule; excluded, absent-final-membership, and fully ignored cores are enumerated. No core boundaries are reconstructed from padded or joined export intervals. Original coordinates and identities survive partial ignored overlap. There are no overlapping retained cores. Outside-game-window time joins the ignored mask, preventing unknown time from becoming a negative when applying the existing exact-tier supervision functions.

Each of the four balanced original-source draws adds:

- `export-rally-training`: the same sources as expanded-medium, but both reviewed export groups provide approximate live/start/end/keep targets rather than keep-only supervision. Calibration remains the designated original exact-labeled group.
- `export-rally-selection`: move one entire export group from fitting into calibration alongside the designated original group; the other export group supplies approximate rally supervision during fitting. Alternate August 29 and August 16 as the export calibration group across the four draws. Calibration is explicitly a mixture of exact labels and approximate export-core proxies.

These are eight additional configurations per model family, or 48 temporal fits and eight student encoders. They keep the original architecture, optimizer, candidate grid, and fixed training seed 3407. The existing tier-ordered student teaching routine is retained; moving records between supervision tiers can also change the deterministic student sample order, so distilled results are not described as changing only temporal target values.

The proxy input manifest is a separate derived artifact. The frozen loader's exact-tier mathematical supervision is reused, but every derived row and selection is explicitly tagged `reviewed-export-rally-proxy`, with approximate endpoints and `independentSemanticGold=false`. This is a training/selection experiment, not promotion of approximate labels to independent evaluation gold. Final scoring still uses authoritative exact, draft, or export-coverage annotations, and the common protected-indoor source/September-17 panel remains excluded from fitting and calibration.

## Resources and output

All bulk outputs, caches, new image inputs, checkpoints, temporary files, and reports go to `private-reference-0061` (`private-reference-0141`). Use the direct WSL NAS mount; the Windows mapping currently denies creating directories in the experiment parent. Disable Python bytecode and direct temporary/cache environment variables to NAS. WSL root and OS-managed swap remain on C:, so monitor RAM and free space.

Use bounded CPU workers and one GPU workload at a time unless an explicit resource amendment is recorded before overlapping work. Do not restart WSL or stop unrelated jobs. No model is promoted to production, and this study does not establish physical-phone performance.

Execution resource amendment: the fixed 32-frame DINO/Mobile wrapper qualification briefly overlapped the historical head worker, after checking 6.94 GiB free GPU memory. Its recorded peak extra PyTorch reservation was 288 MiB; timing is not a benchmark. A single two-thread CPU INT8 extractor may overlap decoding and head fitting on separate CPU affinity. After observing small temporal-head memory use and roughly 34% GPU utilization, one additional non-student temporal-fit worker is allowed on CPU affinity 4–5 alongside the historical worker on 0–3. Stop launching new work below 4 GiB available RAM or 3 GiB free GPU memory. Full image-encoder extraction and student-encoder fitting remain separately scheduled. These process-isolated runs retain their registered seeds, deterministic operators, inputs and losses; overlapped wall times do not support hardware comparisons.

The label-blind timestamp inventory found one missing-camera-frame gap among the 24 new sources. Tick 34.25 s of `recording-026` has a nearest presentation time 144.70 ms away, exceeding the original 125 ms extraction guard. A separately registered amendment retains nearest-frame sampling and permits at most 250 ms error, records every affected tick, and must reproduce unaffected inputs byte-for-byte. It changes no labels, ignored regions, interpolation, decoder, or selection policy. All other scanned sources pass the original guard. The original stager and failed-source directory are preserved; the amendment must pass its independent feature audit before publication.

The final artifact will show precision/recall/F1 versus requested floor, feasibility, source and video membership, training-size/composition comparisons on common unseen data, production-exposure filters, label-quality strata, and downloadable result data. Publication uses the shared Publish Review Artifacts skill (ledger `private-reference-0147`).

Further execution scheduling amendments: the existing two-thread CPU INT8 worker moved from sibling logical CPUs 8/9 to physical-core-separated CPUs 6/8 after a bounded 32-frame check reproduced both raw graph outputs and stored tokens bit-for-bit. The concurrent-load pilot did not show a speedup; no throughput improvement is claimed. The graph, pixels, thread count, batch size and precision recipe are unchanged. Its scheduling protocol and result are preserved under `features-v1/int8-affinity-amendment-v1`.

After the root original non-student fit queue finishes, a single image-encoder worker may replace that worker alongside the historical temporal-head worker. The encoder queue retains two Torch threads on CPUs 10/11, caps its own PyTorch reservation at 1.75 GiB, and requires at least 4 GiB free GPU memory before loading and before each new recording. The existing Linux RAM and C: storage floors remain; the parent also checks Windows memory pressure before the grant. Do not overlap actively executing student-encoder fitting with this encoder queue. Pause or complete image extraction before those student segments. This is a scheduling amendment only; all trained/evaluated inputs and numerical recipes remain frozen, and overlapped timings remain unsuitable for model-speed comparisons.
