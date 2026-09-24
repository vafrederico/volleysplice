# Recognition and temporal attention results — 2026-09-22

Status: all three training arms complete and independently audited. The person-feature arm failed its engineering gate before training. No production model, serve-side model or editor behavior has changed.

## Findings

**Do not promote any new arm from this batch.** Frozen regional MobileNet features improve precision but reduce retained-play recall, leaving F1 effectively unchanged. Both matched transformer replacements regress substantially. The current DINO + TCN remains the strongest neural control in this comparison; production still retains more human play time than every neural standalone.

The regional CNN + TCN reaches **89.90% precision / 90.75% core recall / 90.28% F1**, versus **88.30% / 92.41% / 90.31%** for compact AV + TCN. Precision increases 1.60 percentage points, but core recall declines 1.66 points and long-rally core recall declines 1.90 points. It removes 64.76 seconds of incorrect export while omitting 39.63 additional seconds of human core and 55.41 additional seconds of wanted padded export. Complete misses barely change (33.00 to 32.67), while rallies with any missing core increase from 75.00 to 79.67. This is a cleanup tradeoff, not a successful recall improvement.

The matched AV-only transformer is substantially worse than the compact TCN under this training recipe. At the product's 2-second before/after padding, its seed-mean pooled precision/core recall/F1 are **67.37% / 93.63% / 78.28%**, against **88.30% / 92.41% / 90.31%** for the TCN. Completely missed rallies increase from **33.00 to 38.00**, and short-rally complete misses increase from **21.67 to 25.33**. The modest retained-play recall gain does not satisfy either preregistered improvement screen.

The same architectural change also regresses on DINO features: **82.02% / 93.82% / 87.53%**, against **88.47% / 96.95% / 92.51%** for the matched DINO + TCN. Complete misses increase from **13.67 to 34.67**, short complete misses from **11.00 to 23.00**, and every seed and source group loses F1. source-group-007 core recall falls from 91.23% to 83.15%. Richer frozen inputs improve the transformer's absolute performance relative to AV alone, but do not reverse the regression against its matched TCN. Both prespecified success screens fail.

The person-feature arm was rejected before full extraction or fitting. Both the existing MediaPipe adapter and the pinned OpenCV Zoo NanoDet graph generated gross duplicate boxes or obvious false person locations. This is an engineering failure of the tested detector configurations, not evidence that accurate player movement cannot help rally detection. See the [qualification report](player-motion-engineering-qualification-2026-09-22.md).

The AV failure is not simply a collapsed model or a shifted threshold. All 12 outer fits reduce training loss and produce varying scores, but held-source live/dead tick discrimination is worse than TCN in every group. The frozen 95% inner core-recall rule selects entry threshold 0.20 in 11 of 12 folds. Extra false export occurs in all eight recordings; grass contributes 70.8% of the transformer's incorrect export time. All source/seed F1 cells regress. These observations are consistent with a discrimination/generalization and recall-tradeoff failure under the matched recipe; they do not separate architectural limitations from optimizer, regularization or data effects.

| Source group | AV TCN tick AUC | AV transformer tick AUC |
| --- | ---: | ---: |
| source-group-005 (`source-group-005`) | 0.9296 | 0.8620 |
| source-group-007 (`source-group-007`) | 0.9248 | 0.8129 |
| indoor source (`source-group-009`) | 0.9922 | 0.9559 |
| source-group-012 (`source-group-012`) | 0.9774 | 0.9397 |

These are descriptive diagnostics, not a replacement ranking metric: equal-weight valid cached ticks, half-open human core labels, ignored ticks excluded, pooled within source/seed and then averaged over seeds. The immutable `av-transformer-diagnostic-v1/` bundle records the definitions, all 24 candidate/control source-seed cells and 77 hashed references. No additional fits or decoder settings were evaluated.

## Controlled comparison

The [frozen execution protocol](neural-recognition-execution-2026-09-22.md) is intentionally unchanged: its bytes are part of each experiment's registration. Its checklist records the initial state, not the final completion state. The original [proposal](mobile-recognition-and-transformer-plan-2026-09-22.md) distinguishes subsequent ball, pose, fine-tuning and annotation work from this first batch.

- Training uses the same 18 recordings: 8 exact, 3 reviewed draft and 7 manually reviewed export/coverage recordings. Evaluation uses the same 8 exact recordings, 4 source groups and 322 original rallies. Raw/project/proxy derivatives remain grouped. Beach and the protected test set remain excluded.
- All neural comparisons use reviewed-export supervision, the frozen short-boost loss, seeds 3407/1729/20260918, epochs 5/15/30/60 and the same 48 decoder configurations (192 epoch/decoder candidates per outer fold). Each arm has 30 physical fits: six excluded-source-pair inner fits and four held-source refits per seed. Selection never sees its own outer source group.
- Both temporal architectures have a 125-tick receptive field at 4 Hz. Training preserves historical 128-core/62-halo chunks with up to 252 real ticks. The transformer has two radius-31 attention layers; it does not gain longer context.
- The historical DINO short-boost TCN is the matched control for DINO attention. The previously stronger DINO global-loss result is descriptive context, not a different loss assigned to this comparison.
- Feature extractors receive sanitized video/ROI/timestamp inputs without labels. New features preserve existing supervision and validity masks. Scalers use valid exact training rows only.

Ranking uses pooled `F1_padP_coreR` at symmetric 2-second padding. For each seed, duration numerators and denominators are pooled across recordings before metrics are calculated; table values then average the three seed results. Production is one fixed descriptive reference with historical label exposure. These repeatedly used development recordings do not establish untouched-test generalization.

`P_pad` measures how much exported footage agrees with the padded human export. `R_core` measures retained human play time. Event precision/recall/F1 use one-to-one IoU >= 0.5 matching on uncensored rallies, before padding; they do not establish precise serve/start/end timing and are not the same as retained-play recall. Both export unions join positive gaps strictly below 3 seconds. Identical ignored intervals are subtracted afterward, with no joining across ignored spans. Padding sensitivities 0/1/2/3 are required, and candidates are never ranked by their individually best padding.

The recovery screen requires at least 20% fewer completely missed rallies overall and among short rallies, unchanged incomplete-loss count or better, no overall/long-rally core-recall decline, and at most 0.5 percentage points of F1 loss. The F1 screen requires at least +2 percentage points with no added complete losses or core-recall decline. Magnitude thresholds apply to seed means; at least two seeds must show the corresponding strict improvement direction with their respective guardrails. Incomplete loss means any missing core time, including completely lost rallies. Passing a development screen would still not authorize production replacement.

## Final metrics

Development only: 8 recordings / 4 source groups / 322 original rallies. Target padding is 2 seconds each side; positive gaps strictly below 3 seconds are joined. Neural rows average three seeds after pooling recordings per seed; fractional rally counts are seed means. Production is the default aggressive-suppression reference with historical label exposure. Rows are ordered by the declared F1_padP_coreR metric, not by core recall.

| Model | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Complete misses | Export min | Correctly removed min | Incorrect export min | Wanted export omitted min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO + TCN control | 88.47 | 96.95 | 92.51 | 71.71 | 76.81 | 74.17 | 13.67 | 62.33 | 69.38 | 7.20 | 6.13 |
| Compact AV + TCN control | 88.30 | 92.41 | 90.31 | 69.23 | 69.67 | 69.45 | 33.00 | 59.06 | 69.67 | 6.91 | 9.11 |
| Regional CNN + TCN | 89.90 | 90.75 | 90.28 | 68.69 | 69.36 | 68.93 | 32.67 | 57.05 | 70.75 | 5.83 | 10.03 |
| DINO transformer | 82.02 | 93.82 | 87.53 | 61.10 | 63.77 | 62.37 | 34.67 | 64.80 | 64.94 | 11.65 | 8.10 |
| Production default | 72.45 | 99.27 | 83.76 | 62.64 | 70.81 | 66.47 | 3.00 | 82.38 | 53.88 | 22.70 | 1.57 |
| AV transformer | 67.37 | 93.63 | 78.28 | 46.26 | 52.80 | 49.29 | 38.00 | 80.82 | 50.02 | 26.56 | 6.99 |

All times in the first table are minutes outside ignored intervals. Human padded export totals 61.25 minutes, human core totals 39.79 minutes, and the evaluable video totals 137.83 minutes. **Correctly removed** is non-human-export time left out (true negatives); **incorrect export** is exported time outside the human padded export; **wanted export omitted** is human padded export left out. The latter includes padding and therefore differs from missed core. Total removed time would include both correctly removed and wanted export omitted; it is not the correctly removed column.

### Paired screens

| Candidate | Control | F1 delta pp | Core recall delta pp | Complete misses delta | Recovery screen | F1 screen |
| --- | --- | ---: | ---: | ---: | --- | --- |
| AV transformer | Compact AV + TCN control | -12.03 | +1.22 | +5.00 | Fail | Fail |
| DINO transformer | DINO + TCN control | -4.99 | -3.14 | +21.00 | Fail | Fail |
| Regional CNN + TCN | Compact AV + TCN control | -0.03 | -1.66 | -0.33 | Fail | Fail |

### Production complementarity

Descriptive coverage at 2-second padding, averaged across seeds. Partial retention is not full recovery; these counts do not measure any combined system. Exact rally IDs and retained seconds are in the JSON. These are padded coverage counts, not proof of distinct rally proposals or correct serve boundaries. All partly-retained counts happen to be zero; one or two recoveries do not offset the many new complete losses. Production remains the high-recall base for review.

| Candidate | Production complete misses | Of those: partly retained | Of those: fully retained | New complete misses versus production |
| --- | ---: | ---: | ---: | ---: |
| Compact AV + TCN control | 3 | 0.00 | 0.00 | 30.00 |
| DINO + TCN control | 3 | 0.00 | 1.67 | 12.33 |
| AV transformer | 3 | 0.00 | 1.00 | 36.00 |
| DINO transformer | 3 | 0.00 | 2.00 | 33.67 |
| Regional CNN + TCN | 3 | 0.00 | 1.33 | 31.00 |

### Padding sensitivity

| Model | Padding s | P_pad % | R_core % | F1_padP_coreR % | Model export s | Human export s | Delta s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO + TCN control | 0 | 84.34 | 86.22 | 85.25 | 2441.886 | 2387.151 | +54.735 |
| DINO + TCN control | 1 | 86.69 | 94.63 | 90.48 | 3101.642 | 3031.151 | +70.491 |
| DINO + TCN control | 2 | 88.47 | 96.95 | 92.51 | 3739.831 | 3675.151 | +64.680 |
| DINO + TCN control | 3 | 90.08 | 97.84 | 93.79 | 4364.456 | 4323.263 | +41.193 |
| Compact AV + TCN control | 0 | 84.24 | 82.00 | 83.10 | 2324.050 | 2387.151 | -63.101 |
| Compact AV + TCN control | 1 | 86.47 | 89.60 | 88.00 | 2943.533 | 3031.151 | -87.618 |
| Compact AV + TCN control | 2 | 88.30 | 92.41 | 90.31 | 3543.467 | 3675.151 | -131.684 |
| Compact AV + TCN control | 3 | 89.71 | 93.88 | 91.75 | 4145.108 | 4323.263 | -178.155 |
| Regional CNN + TCN | 0 | 86.43 | 79.45 | 82.67 | 2199.239 | 2387.151 | -187.912 |
| Regional CNN + TCN | 1 | 88.49 | 87.54 | 87.94 | 2815.639 | 3031.151 | -215.512 |
| Regional CNN + TCN | 2 | 89.90 | 90.75 | 90.28 | 3423.300 | 3675.151 | -251.851 |
| Regional CNN + TCN | 3 | 91.01 | 92.46 | 91.70 | 4027.486 | 4323.263 | -295.777 |
| DINO transformer | 0 | 75.51 | 84.95 | 79.95 | 2685.806 | 2387.151 | +298.655 |
| DINO transformer | 1 | 79.20 | 91.56 | 84.93 | 3296.429 | 3031.151 | +265.278 |
| DINO transformer | 2 | 82.02 | 93.82 | 87.53 | 3887.768 | 3675.151 | +212.617 |
| DINO transformer | 3 | 84.41 | 94.90 | 89.35 | 4472.800 | 4323.263 | +149.537 |
| Production default | 0 | 65.50 | 95.75 | 77.79 | 3489.608 | 2387.151 | +1102.457 |
| Production default | 1 | 69.40 | 98.30 | 81.36 | 4223.508 | 3031.151 | +1192.357 |
| Production default | 2 | 72.45 | 99.27 | 83.76 | 4943.025 | 3675.151 | +1267.874 |
| Production default | 3 | 75.02 | 99.49 | 85.54 | 5637.197 | 4323.263 | +1313.934 |
| AV transformer | 0 | 57.79 | 88.30 | 69.69 | 3674.190 | 2387.151 | +1287.039 |
| AV transformer | 1 | 62.62 | 92.15 | 74.45 | 4303.612 | 3031.151 | +1272.461 |
| AV transformer | 2 | 67.37 | 93.63 | 78.28 | 4849.023 | 3675.151 | +1173.872 |
| AV transformer | 3 | 71.91 | 94.44 | 81.60 | 5356.738 | 4323.263 | +1033.475 |

### Rally-retention guardrails at 2-second padding

Complete loss means zero retained evaluable core after export padding; any-core loss includes both partial and complete losses. Short rallies have original duration at most 3 seconds; long rallies are longer than 3 seconds. Counts are per-seed means, not distinct rallies across three runs.

| Model | Complete losses | Short complete losses | Any-core losses | Long R_core % | Missed core s |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINO + TCN control | 13.67 | 11.00 | 43.00 | 97.65 | 72.70 |
| Compact AV + TCN control | 33.00 | 21.67 | 75.00 | 93.82 | 181.20 |
| Regional CNN + TCN | 32.67 | 21.00 | 79.67 | 91.91 | 220.83 |
| DINO transformer | 34.67 | 23.00 | 63.67 | 95.28 | 147.54 |
| Production default | 3.00 | 3.00 | 10.00 | 99.48 | 17.47 |
| AV transformer | 38.00 | 25.33 | 60.33 | 95.44 | 152.16 |

### Regional CNN stability

The CNN improves F1 in two seeds, but only one seed satisfies the directional recall/loss guardrails. Both overall preregistered screens fail. Its core recall spans 88.15–93.53% across seeds, versus 92.21–92.72% for AV + TCN; complete losses span 22–41 versus 31–36. Three seeds describe training variability; four source groups do not justify a strong statistical generalization claim.

| Seed | P_pad % | R_core % | F1_padP_coreR % | Paired core R delta pp | Paired F1 delta pp | Paired complete-loss delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3407 | 92.32 | 90.57 | 91.44 | -1.73 | +0.64 | +3 |
| 1729 | 90.32 | 88.15 | 89.22 | -4.06 | -0.91 | +5 |
| 20260918 | 87.06 | 93.53 | 90.18 | +0.81 | +0.18 | -9 |

| Source group | Control core R % | CNN core R % | Core R delta pp | Control F1 % | CNN F1 % | CNN complete losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source-group-005 | 86.23 | 75.28 | -10.94 | 84.02 | 81.89 | 16.67 |
| source-group-007 | 85.68 | 87.82 | +2.14 | 83.73 | 86.24 | 16.00 |
| source-group-009 | 99.56 | 99.89 | +0.33 | 98.61 | 97.26 | 0.00 |
| source-group-012 | 98.07 | 99.99 | +1.92 | 94.79 | 94.53 | 0.00 |

source-group-005 loses 10.94 points of retained-play recall (86.23% to 75.28%) and 63.80 additional core seconds; its recall declines in all three seeds. source-group-007 and both indoor groups gain mean core recall; only source-group-007 gains mean group F1. source-group-007's mean recall gain comes from one seed (+8.43 points); the other two lose roughly one point each. All CNN complete losses occur in grass footage. This is a source-dependent tradeoff; excellent indoor retention does not resolve the grass failure or demonstrate a broadly successful feature upgrade.

## Recommendation and next experiment

Keep the current production-backed review workflow and the existing TCN controls. These results do not support replacing production with a neural standalone, automatically applying the new CNN's removals, or replacing either TCN with this transformer. No new suppression, union, review-budget or rally-splitting policy was evaluated in this batch; complementarity counts above do not establish such a policy's performance.

The next bounded experiment should target the source-group-005 grass failure and short serves with explicit motion evidence, rather than another undirected architecture swap. First qualify a correctly identified small person detector on manually checked near/far-player frames, plus a separate ball-visibility sample containing toss, flight, occlusion and absent-ball negatives. The tested detector configurations did not pass that gate. Measure localization, duplicate rate, far-player/ball visibility and sustained inference cost on an actual phone before extracting the full dataset. Additional ball/serve-contact or player-track labels would make this a different, explicitly supervised experiment.

If recognition passes those gates, append confidence-aware player velocity/dispersion and ball-flight/contact features to the unchanged AV + TCN baseline. Missing detections must remain missing, never evidence of dead play. Keep the same source-held evaluation and recall guardrails; evaluate any proposal/review use separately against production with human review time and rally-separation accuracy. Frozen DINO-to-mobile distillation or limited visual fine-tuning is another later candidate, but this batch provides no measured benefit for either. The result rules out these particular frozen CNN pools and local-attention recipe as a promotion, not image recognition or transformers in general. No additional training is started here.

## New inputs and model sizes

| Arm | Inputs | Trainable temporal parameters | Additional frozen encoder |
| --- | --- | ---: | --- |
| AV transformer | Existing AV104 | 31,176 | None |
| DINO transformer | Existing AV104 + cached DINO regions | 44,504 | Existing DINOv2, excluded from head count |
| Regional CNN + TCN | AV104 + four 576-value CNN regions + 8 quality/age/availability scalars | 44,692 | MobileNetV3-Small feature encoder: 927,008 parameters |

The CNN is frozen ImageNet-pretrained MobileNetV3-Small with its classifier removed. One aspect-preserving 224-pixel ROI image at 2 Hz produces one spatial map. Four fixed fractional pools summarize the whole ROI, lower half, upper half and middle band; the downstream projection is learned. These are **image-relative regions**, not recognized court/net geometry. Padding is excluded from pooling, although convolutional receptive fields still see padded pixels. The 2 Hz features are held onto the existing 4 Hz AV grid with explicit feature age, availability and selected-frame timestamp offset. Nearest-frame sampling is an offline operation, not a zero-lookahead live-camera claim.

All 18 full caches passed attachment checks: original AV values, target/mask objects, source groups, times and ignored intervals remained unchanged. The attached arrays contain 740,330,048 bytes; the first checked true-length batch was `[1,190,2416]` with four finite output logits per tick. The frozen encoder was not fine-tuned and no DINO distillation was performed.

## Portability qualification

The actual pretrained CNN encoder executes in ONNX Runtime CPU and Windows Chrome WASM with numerical parity. Fixed first-outer-fold trained AV and DINO transformer checkpoints also pass, including their fitted normalization, masked NaN inputs, independent short sequences, all-masked sequences and interior halo equivalence. This validates the tested graphs and fixtures, not mobile accuracy or speed.

The initial static TCN graph only qualified interior 252-tick chunks. A separate dynamic-time graph now also passes true lengths 1, 7, 190 and 252 and recording-edge/interior comparisons against a whole-sequence reference. It requires true-length segments and real halo context; the caller must split at ignored gaps. The fixtures qualify independent segment calls, not the application's gap-segmentation integration. The fixed mobile seed-3407/outer-0 epoch-60 checkpoint and its fitted scaler pass all seven fixtures and four boundary checks in CPU and Chrome. Maximum Chrome raw-logit error is 1.91e-5, within the declared absolute/relative tolerances (1e-5/1e-4); maximum boundary-equivalence error is 3.81e-6.

| Tested graph | ONNX bytes | Gzip bytes | Windows Chrome WASM median |
| --- | ---: | ---: | ---: |
| Pretrained MobileNet feature encoder | 3,718,362 | 3,435,048 | 8.71 ms/image |
| Trained AV transformer + scaler | 2,471,355 | 134,200 | 5.02 ms/252 ticks |
| Trained DINO transformer + scaler | 6,703,468 | 189,064 | 13.86 ms/252 ticks |
| Trained dynamic mobile TCN + scaler | 213,477 | 169,704 | 3.67 ms/252 ticks |

The temporal graphs use fixed first-outer-fold checkpoints, not a fold selected for runtime or accuracy. The browser harness pins ONNX Runtime Web 1.22.0, uses one WASM thread and measures prepared inputs. Its loaded runtime resources add 11,279,437 raw bytes (2,923,978 gzip estimate). Compression explains why the static attention graphs' download sizes are much smaller than their raw ONNX files. Sizes and operator support pass an engineering screen; whole-product memory and time budgets remain open.

The CNN encoder plus trained head/scaler total **3,931,839 raw ONNX bytes** (3,604,752 gzip estimate). Including the harness's shared runtime gives 15,211,276 raw bytes or a 6,528,730-byte gzip estimate, excluding the application and video. Gzip values are estimates, not a claim about the eventual server's transfer configuration.

No Android device was connected through ADB. Physical phone/browser speed, incremental peak memory, sustained thermal behavior, end-to-end decoding/preprocessing, quantization parity and the proposed product budget remain unmeasured. Full pixel-to-rally parity also remains open: research CNN tokens are stored in float16 and then loaded into the FP32 head, so a deployment must preserve or separately validate that rounding, timing alignment and quality-feature path. DINO image encoding is excluded from temporal-head timing. Desktop timings under concurrent workloads must not be presented as phone estimates or speed comparisons between static and dynamic graphs.

## Validation and artifacts

All 76 focused tests passed. A real-data one-epoch control reproduced the historical TCN's weights, probabilities, loss history and sample exposure exactly through the new runner. All three independent study audits passed: 90 physical fits and 108 logical inner views in total, decoder reselection, scaler/exposure/provenance checks and four-padding metrics. Checkpoint prediction replay samples three chunks per recording/checkpoint; it is not a full replay of every tick. The final summarizer verifies audit/report/registration bindings and compares identical gold, source and ignored-range revisions.

Experiment code checkpoint `07c67856` is pushed to Forgejo branch `t3code/plan-neural-network-improvements`. Large artifacts are on the NAS at `private-reference-0174` (WSL: `private-reference-0175`):

- `av-transformer-v1/`, `dino-transformer-v1/`, `mobile-tcn-v1/`: registrations, archived source/protocol bytes, folds, weights, per-seed results, `report.json` and passing `audit.json`.
- `mobile-features-v1/`: all 18 caches, immutable extraction/training indexes, helper archive and `bridge-validation.json`.
- `engineering/`: exact-control check and rejected-person-detector evidence.
- `runtime-*/` and `mobile-preprocessing-audit-v1/`: CPU/Chrome graph fixtures, reports and preprocessing inspection samples.
- `comparison-v1.json` / `.md`: final audited comparison against the frozen controls and production, all padding cases, per-seed/source metrics and exact production-complementarity rally IDs. The earlier `comparison-av-v1` files are interim AV-only results.
- `av-transformer-diagnostic-v1/`: immutable failure diagnostic and its hashed input references; no additional training or selection.

NAS output paths bypass the WSL virtual disk, but OS-managed WSL swap still resides on C:. Its VHDX grew from 3,091,202,048 to 3,695,181,824 bytes during the observed interval; an earlier process snapshot attributed about 1.67 GiB of active swap to the three experiment processes. This identifies a real source of local storage growth without attributing every host free-space change to the experiment. Subsequent jobs explicitly use NAS temporary/cache directories and disable Python bytecode writes. No WSL restart, global swap change, or unrelated-job termination was performed.
